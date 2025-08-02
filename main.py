import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from module_bet import TGNN_out_comp,  TGNN_Closeness, TATKC
from scipy.stats import weightedtau
from nx2graphs import load_real_data, load_real_true, load_train_real_data, load_real_train_true
from utils import loss_cal, compute_kendall_tau, compute_topk_metrics, normalized_mae, normalized_supremum_deviation, compute_topk_metrics_ptd
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

# Argument and global variables
parser = argparse.ArgumentParser('Interface for Experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=1500, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=5, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=3, help='idx for the gpu to use')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['sintime', 'pos_time_aware', 'time', 'hierarchical', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--test', action='store_true', help='Run in test mode')
parser.add_argument('--bet', choices=['sh', 'sfm'], help='Betweenness mode: sh (shortest), sfm (shortest-foremost)')
parser.add_argument('--close', choices=['sh', 'f'], help='Closeness mode: sh (shortest), f (fastest)')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(1)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
testing = args.test
bet_mode = args.bet
close_mode = args.close

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
LR_MODEL_SAVE_PATH = f'./saved_models/{args.agg_method}-{args.attn_mode}-{args.data}_mlp.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

n_feat = np.load('./data/test/Real/processed/seq/ml_{}_node.npy'.format(DATA), allow_pickle=True)
test_real_feat = np.zeros((1400000, 128))

def setSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

setSeeds(89)

if args.bet is not None:
    mode_type = 'bet'
    mode_value = args.bet
elif args.close is not None:
    mode_type = 'close'
    mode_value = args.close
else:
    raise ValueError("You must specify either --bet or --close.")

# Load training data
train_real_src_l, train_real_dst_l, train_real_ts_l, train_real_node_count, \
    train_real_node, train_real_time, train_real_ngh_finder, \
    pass_through_d_list, earl_arrival_list, \
    out_degree_list, out_reach_list = load_train_real_data(
        UNIFORM, mode_type, mode_value
)

# Load training true labels
nodeList_train_real, train_label_l_real = load_real_train_true(
    mode_type, mode_value
)

# Load test data
test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, \
    test_real_node, test_real_time, test_real_ngh_finder, \
    test_pass_through_d, test_earl_arrival, \
    test_out_degree, test_out_reach = load_real_data(
        DATA, mode_type, mode_value
)

if args.bet is not None:
    nodeList_test_real, test_label_l_real = load_real_true('{}'.format(DATA), 'bet', args.bet)
elif args.close is not None:
    nodeList_test_real, test_label_l_real = load_real_true('{}'.format(DATA), 'close', args.close)

train_ts_list, test_ts_list, train_real_ts_list = [], [], []

for idx in range(len(nodeList_train_real)):
    train_real_ts_list.append(np.array([train_real_time[idx]] * len(nodeList_train_real[idx])))

test_real_ts_list = np.array([test_real_time] * len(nodeList_test_real))
TEST_BATCH_SIZE = BATCH_SIZE

num_test_instance = len(nodeList_test_real)
num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    test_src_l_cut = np.array(nodeList_test_real[s_idx:e_idx])
    test_ts_l_cut = np.array(test_real_ts_list[s_idx:e_idx])
    test_real_ngh_finder.preprocess(tuple(test_src_l_cut), tuple(test_ts_l_cut), NUM_LAYER, NUM_NEIGHBORS)

device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
ngh_finder = train_real_ngh_finder[0]

if args.bet is not None:
    bet_mode = args.bet

    tgnn_model = TGNN_out_comp(
        train_real_ngh_finder[0],
        test_real_feat,
        attn_mode=ATTN_MODE,
        use_time=USE_TIME,
        agg_method=AGG_METHOD,
        num_layers=NUM_LAYER,
        n_head=NUM_HEADS,
        drop_out=DROP_OUT
    )
elif args.close is not None:
    close_mode = args.close
    tgnn_model = TGNN_Closeness(
        train_real_ngh_finder[0],
        test_real_feat,
        attn_mode=ATTN_MODE,
        use_time=USE_TIME,
        agg_method=AGG_METHOD,
        num_layers=NUM_LAYER,
        n_head=NUM_HEADS,
        drop_out=DROP_OUT
    )
else:
    raise ValueError("You must specify either --bet (Betweenness mode) or --close (Closeness mode).")


class ResidualSkipGNN_Modulated_Stable(nn.Module):
    def __init__(self, node_dim=128, drop=0.1, delta_scale=1.0):
        super().__init__()
        self.delta_scale = delta_scale

        # Predict Δ correction
        self.correction_mlp = nn.Sequential(
            nn.Linear(node_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 1)
        )

        # Gating over correction (α ∈ [0,1], centered ~0.5)
        self.alpha_gate = nn.Sequential(
            nn.Linear(node_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # PTD scaling
        self.ptd_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_feat, ptd, return_debug=False):
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)

        # Normalize PTD
        norm_ptd = (ptd - ptd.mean()) / (ptd.std() + 1e-6)
        base = self.ptd_scale * norm_ptd.detach().squeeze(1)  # [B]

        # Combine features
        x = torch.cat([src_feat, norm_ptd], dim=1)  # [B, node_dim + 1]

        # Predict modulation α
        raw_alpha = self.alpha_gate(x).squeeze(1)         # unrestricted
        alpha = 0.5 + 0.5 * torch.tanh(raw_alpha)          # ∈ (0, 1)

        # Predict delta correction
        delta = self.correction_mlp(x).squeeze(1)          # [B]
        delta = self.delta_scale * torch.tanh(delta)       # stabilize correction

        final = base + alpha * delta

        if return_debug:
            print("[DEBUG]")
            print("  Mean α:     ", alpha.mean().item())
            print("  Mean Δ:     ", delta.mean().item())
            print("  Std Δ:      ", delta.std().item())
            print("  Min/Max α:  ", alpha.min().item(), "/", alpha.max().item())

            return final, alpha, delta
        return final



class MLPTwoFeaturesAct(nn.Module):
    def __init__(self, node_dim=128, ptd_dim=128, drop=0.1):
        super().__init__()

        # Project scalar PTD to 128-dim
        self.ptd_proj = nn.Sequential(
            nn.Linear(1, ptd_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Main MLP layers
        self.fc_1 = nn.Linear(node_dim + ptd_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        # Learnable output scale for softplus
        self.output_scale = nn.Parameter(torch.tensor(50.0))

        # Weight init
        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd):
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)  # [B, 1]

        ptd_embed = self.ptd_proj(ptd)  # [B, ptd_dim]

        x = torch.cat([src_feat, ptd_embed], dim=1)  # [B, node_dim + ptd_dim]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)

        out = self.fc_3(x).squeeze(1)  # [B]
        return F.softplus(out) * self.output_scale




class MLPTwoFeatures(nn.Module):
    def __init__(self, node_dim=128, ptd_dim=128, drop=0.1):
        super().__init__()

        self.ptd_proj = nn.Sequential(
            nn.Linear(1, 128),
            #nn.LayerNorm(128),  # ← added here
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.fc_1 = nn.Linear(node_dim + ptd_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd):
        # ptd: [B] or [B, 1]
        ptd = (ptd - ptd.mean()) / (ptd.std() + 1e-6)

        #print("src_feat mean:", src_feat.mean().item(), "src_feat std:", src_feat.std().item())
        #print("ptd mean after normalization:", ptd.mean().item(), "ptd std:", ptd.std().item())

        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)  # [B, 1]

        ptd_embed = self.ptd_proj(ptd)  # [B, ptd_dim]

        x = torch.cat([src_feat, ptd_embed], dim=1)  # [B, node_dim + ptd_dim]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)
        return out

class MLPTwoFeaturesSimple(nn.Module):
    def __init__(self, node_dim=128, drop=0.1):
        super().__init__()

        # ptd is scalar → no projection needed
        self.fc_1 = nn.Linear(node_dim + 1, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd):
        # ptd: [B] or [B, 1]
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)  # [B, 1]

        x = torch.cat([src_feat, ptd], dim=1)  # [B, node_dim + 1]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)
        return out

class MLPThreeFeaturesHybrid(nn.Module):
    #layer norm for ptd and arrival, no normalizaton after for arrival_feat
    def __init__(self, node_dim=128, ptd_dim=128, drop=0.1):
        super().__init__()

        self.ptd_proj = nn.Sequential(
            nn.Linear(1, ptd_dim),
            nn.LayerNorm(ptd_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.arrival_proj = nn.Sequential(
            nn.Linear(1, ptd_dim),
            nn.LayerNorm(ptd_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Total input: src_feat + ptd + arrival = node_dim + 2 × ptd_dim
        self.fc_1 = nn.Linear(node_dim + 2 * ptd_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd, arrival_feat):
        # ptd and arrival_feat: [B] or [B, 1]
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)
        if arrival_feat.dim() == 1:
            arrival_feat = arrival_feat.unsqueeze(-1)

        ptd_embed = self.ptd_proj(ptd)                  # [B, ptd_dim]


        arrival_embed = self.arrival_proj(arrival_feat) # [B, ptd_dim]

        x = torch.cat([src_feat, ptd_embed, arrival_embed], dim=1)  # [B, node_dim + 2 * ptd_dim]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)  # [B]
        return out

class MLPTwoFeaturesHybrid(nn.Module):
    # layer norm for ptd only, no normalization after for ptd
    def __init__(self, node_dim=128, ptd_dim=128, drop=0.1):
        super().__init__()

        self.ptd_proj = nn.Sequential(
            nn.Linear(1, ptd_dim),
            nn.LayerNorm(ptd_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Total input: src_feat + ptd = node_dim + ptd_dim
        self.fc_1 = nn.Linear(node_dim + ptd_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd):
        # ptd: [B] or [B, 1]
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)

        ptd_embed = self.ptd_proj(ptd)  # [B, ptd_dim]

        x = torch.cat([src_feat, ptd_embed], dim=1)  # [B, node_dim + ptd_dim]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)  # [B]
        return out


class MLPTwoFeaturesImproved(nn.Module):
    def __init__(self, node_dim=128, ptd_dim=1, drop=0.1):
        super().__init__()

        # Instead of projecting PTD to 128, keep it low-dimensional (or skip projection)
        self.ptd_proj = nn.Identity()  # optionally use nn.Linear(1, 4) if needed

        # New input dim: node_dim + 1
        self.fc_1 = nn.Linear(node_dim + ptd_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd):
        # Normalize PTD
        ptd = (ptd - ptd.mean()) / (ptd.std() + 1e-6)
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)  # [B, 1]

        # Optional low-rank projection (currently identity)
        ptd_embed = self.ptd_proj(ptd)  # [B, 1]

        # Concatenate
        x = torch.cat([src_feat, ptd_embed], dim=1)  # [B, node_dim + 1]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)
        return out


class MLP(torch.nn.Module):
    def __init__(self, dim=128, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 64)
        self.fc_2 = torch.nn.Linear(64, 32)
        self.fc_3 = torch.nn.Linear(32, 1)

        self.act = torch.nn.ReLU()

        torch.nn.init.kaiming_normal_(self.fc_1.weight)
        torch.nn.init.kaiming_normal_(self.fc_2.weight)
        torch.nn.init.kaiming_normal_(self.fc_3.weight)

        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


"""
wkt_ptd, _ = weightedtau(test_pass_through_d, test_label_l_real)
print("PTD Kendall Tau (default) : ", wkt_ptd)
kt_ptd, kt_nonzero_ptd = compute_kendall_tau(test_pass_through_d, test_label_l_real)
print(f"PTD Kendall Tau (all nodes):      {kt_ptd:.4f}")
print(f"PTD Kendall Tau (non-zero only): {kt_nonzero_ptd:.4f}")
k_list = [1, 5, 10, 20]
compute_topk_metrics_ptd(test_pass_through_d, test_label_l_real, k_list=k_list)

"""

#MLP Variants
if mode_type == "bet" and mode_value == "sfm":
    #MLP_model = MLPWThreeFeatures().to(device)
    #MLP_model = MLPWThreeFeaturesSimple().to(device)
    MLP_model = MLPTwoFeaturesHybrid().to(device)

else:
    #MLP_model = MLPTwoFeatures().to(device)
    MLP_model = MLP().to(device)

optimizer = torch.optim.Adam(list(tgnn_model.parameters()) + list(MLP_model.parameters()),lr=LEARNING_RATE)
tgnn_model.to(device)

#Load Model
if testing:
    print("Running in test mode...")
    tgnn_model.load_state_dict(torch.load('./saved_models/model_TGAT_1.pth', weights_only=True))
    MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_1.pth', weights_only=True))

def eval_real_data(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_list = []
    tgan.ngh_finder = sampler
    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        wkt_ptd, _ = weightedtau(test_pass_through_d, label)
        print("PTD Kendall Tau (default) : ", wkt_ptd)
        kt_ptd, kt_nonzero_ptd = compute_kendall_tau(test_pass_through_d, label)
        print(f"PTD Kendall Tau (all nodes):      {kt_ptd:.4f}")
        print(f"PTD Kendall Tau (non-zero only): {kt_nonzero_ptd:.4f}")
        k_list = [1, 5, 10, 20]
        compute_topk_metrics_ptd(test_pass_through_d, label, k_list=k_list)

        #all_alpha = []
        for k in tqdm(range(num_test_batch), desc="Evaluating batches"):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])
            src_embed = tgan.tem_conv(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )

            if mode_type == "bet":
                if test_pass_through_d is None:
                    raise ValueError("test_pass_through_d not loaded for betweenness mode.")

                ptd = test_pass_through_d[test_src_l_cut - 1].float()
                test_pass_through_degree_batch = ptd.unsqueeze(-1)

                if mode_value == "sh":
                    # Shortest Betweenness
                    #test_pred = lr_model(src_embed, test_pass_through_degree_batch)
                    test_pred = lr_model(src_embed)

                    #test_pred, alpha, _ = lr_model(src_embed, test_pass_through_degree_batch, return_debug=True)
                    #all_alpha.append(alpha.cpu())
                elif mode_value == "sfm":
                    # Shortest-Foremost Betweenness
                    if test_earl_arrival is None:
                        raise ValueError("test_earl_arrival not loaded for sfm mode.")

                    earl_arr = test_earl_arrival[test_src_l_cut - 1].float()
                    test_earl_arrival_batch = earl_arr.unsqueeze(-1)

                    #test_pred = lr_model(src_embed, test_pass_through_degree_batch, test_earl_arrival_batch)
                    test_pred = lr_model(src_embed, test_pass_through_degree_batch)

            elif mode_type == "close":
                if mode_value == "f":
                    if test_out_degree is None:
                        raise ValueError("test_out_degree not loaded for closeness-fastest mode.")

                    out_d = test_out_degree[test_src_l_cut - 1].float()
                    test_out_degree_batch = out_d.unsqueeze(-1)

                    test_pred = lr_model(src_embed, test_out_degree_batch)

                elif mode_value == "sh":
                    if test_out_reach is None:
                        raise ValueError("test_out_reach not loaded for closeness-shortest mode.")

                    out_r = test_out_reach[test_src_l_cut - 1].float()
                    test_out_reach_batch = out_r.unsqueeze(-1)

                    test_pred = lr_model(src_embed, test_out_reach_batch)

                else:
                    raise ValueError(f"Unknown closeness mode: {mode_value}")



            test_pred_list.extend(test_pred.cpu().detach().numpy().tolist())

        with open("test_kendaltau/predicted_values_mathoverflow_sparse.txt", "w") as pred_file:
            for value in test_pred_list:
                pred_file.write(f"{value}\n")


        label = np.clip(label, a_min=0.0, a_max=None)
        wkt, _ = weightedtau(test_pred_list, label)


        #Additional Metrics
        sd_value = normalized_supremum_deviation(test_pred_list, label)
        norm_mae = normalized_mae(test_pred_list, label)

        num_nodes = len(label)
        norm_factor = 1 / (num_nodes * (num_nodes - 1))
        norm_label = norm_factor * label

        pred = np.array(test_pred_list)
        true = np.array(label)
        mask = (pred <= 0) & (true > 0)
        count = np.sum(mask)
        print("Number of nodes with pred ≤ 0 and true > 0:", count)
        print("Supremum Deviation Norm:", sd_value)
        print("MAE Norm:", norm_mae)

        print("Kendall Tau (default) : ", wkt)
        kt_all, kt_nonzero = compute_kendall_tau(test_pred_list, label)
        print(f"Kendall Tau (all nodes):      {kt_all:.4f}")
        print(f"Kendall Tau (non-zero only): {kt_nonzero:.4f}")

        if not torch.is_tensor(test_pred_list):
            test_pred_list = torch.tensor(test_pred_list, dtype=torch.float32)
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.float32)

        k_list = [1, 5, 10, 20]
        compute_topk_metrics(test_pred_list, label, k_list=k_list)

        # --- Print α statistics ---
        '''
        if len(all_alpha) > 0:
            all_alpha_tensor = torch.cat(all_alpha)
            print(f"\n[α Modulation Diagnostics]")
            print(f"  Mean α:           {all_alpha_tensor.mean():.4f}")
            print(f"  Median α:         {all_alpha_tensor.median():.4f}")
            print(f"  Std α:            {all_alpha_tensor.std():.4f}")
            print(f"  Min α:            {all_alpha_tensor.min():.4f}")
            print(f"  Max α:            {all_alpha_tensor.max():.4f}")
            print(f"  Fraction α > 0.8: {(all_alpha_tensor > 0.8).float().mean():.2%}")
            print(f"  Fraction α < 0.2: {(all_alpha_tensor < 0.2).float().mean():.2%}")
        '''
    end_time = time.time()
    e_time = (end_time - start_time) / 60.0

    return  e_time

def training_model():
    for epoch in range(NUM_EPOCH):
        epoch_topk_10 = []
        epoch_topk_20 = []
        epoch_topk_1 = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []

        graph_indices = list(range(len(train_real_ts_l)))

        for j in tqdm(graph_indices):
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]

            num_train_instance = len(node_list)
            num_train_batch = math.ceil(num_train_instance / BATCH_SIZE)
            if mode_type == "bet":
                pass_through_degree = pass_through_d_list[j]
                if mode_value == "sfm":
                    earl_arrival = earl_arrival_list[j]
            elif mode_type == "close":
                if mode_value == "f":
                    out_degree = out_degree_list[j]
                elif mode_value == "sh":
                    out_reach = out_reach_list[j]

            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(num_train_instance, s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]

                optimizer.zero_grad()
                scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

                src_embed = tgnn_model.tem_conv(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )
                true_label = torch.tensor(label_l_cut, dtype=torch.float32).to(device)

                if mode_type == "bet":
                    ptd = pass_through_degree[src_l_cut - 1].float()
                    pass_through_degree_batch = ptd.unsqueeze(-1)
                    if mode_value == "sh":
                        #pred_value = MLP_model(src_embed, pass_through_degree_batch)
                        pred_value = MLP_model(src_embed)

                        #pred_value, alpha, _ = MLP_model(src_embed, pass_through_degree_batch, return_debug=True)
                    elif mode_value == "sfm":
                        earl_arr = earl_arrival[src_l_cut - 1].float()
                        earl_arr_batch = earl_arr.unsqueeze(-1)
                        #pred_value = MLP_model(src_embed, pass_through_degree_batch, earl_arr_batch)

                        pred_value = MLP_model(src_embed, pass_through_degree_batch)

                elif mode_type == "close":
                    if mode_value == "f":
                        out_d = out_degree[src_l_cut - 1].float()
                        out_degree_batch = out_d.unsqueeze(-1)
                        pred_value = MLP_model(src_embed, out_degree_batch)
                    elif mode_value == "sh":
                        out_r = out_reach[src_l_cut - 1].float()
                        out_reach_batch = out_r.unsqueeze(-1)
                        pred_value = MLP_model(src_embed, out_reach_batch)

                topk_stats = compute_topk_metrics(pred_value, true_label, k_list=[1 ,5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                loss = loss_cal(pred_value, true_label, len(pred_value), device)


                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                #loss += 0.1 * ((1 - alpha) ** 2).mean()

                loss.backward()

                #for name, param in MLP_model.named_parameters():
                #    if param.grad is not None:
                #        print(name, param.grad.norm().item())

                torch.nn.utils.clip_grad_norm_(list(tgnn_model.parameters()) + list(MLP_model.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                m_loss.append(loss.item())

        avg_topk_1 = np.mean(epoch_topk_1)
        avg_topk_10 = np.mean(epoch_topk_10)
        avg_topk_20 = np.mean(epoch_topk_20)

        print(
            f" Epoch {epoch:02d} Summary : Avg Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | Top@20%: {avg_topk_20:.4f} ")

        epoch_loss = np.mean(m_loss)
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")



if not testing:
    training_model()

#Save Model
if not testing:
    print("Running in training mode...")
    torch.save(MLP_model.state_dict(), './saved_models/model_MLP_3.pth')
    torch.save(tgnn_model.state_dict(), './saved_models/model_TGAT_3.pth')


e_time = eval_real_data('test for real data', tgnn_model, MLP_model, test_real_ngh_finder,
                                              nodeList_test_real, test_real_ts_list, test_label_l_real)


'''
1) mlp_1: TGNN_out_comp  - sh - MLP -  neighbors = 5, epochs = 10, layers = 2
2) mlp_2: TGNN_out_comp - sh - MLP  - epochs = 10, layers = 3 - sparse
3) mlp_3: TGNN_out_comp - sh - MLP  - epochs = 10, layers = 2 - sparse

Bet:
    SFM: MLPTwoFeaturesHybrid > MLPThreeFeaturesHybrid
         TGNN_out_comp - sfm - MLPWThreeFeatures   - neighbors = 5, epochs = 10 - BAD
Closeness:

'''
