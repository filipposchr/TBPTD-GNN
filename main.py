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
import torch.nn.functional as F
from nx2graphs import load_real_data, load_real_true, load_train_real_data, load_real_train_true
from module import ConservativeSimplifiedModel_gemini_CONTR_30MAY
from scipy.stats import kendalltau
from utils import setSeeds, eval_statistics, hits_in_ks, compute_topk_metrics, LabelNormalizer, EnchRankingLoss, AdaptiveReweightedSupConLoss, AdaptiveReweightedSupConLoss_modes
TBetGNN = ConservativeSimplifiedModel_gemini_CONTR_30MAY

parser = argparse.ArgumentParser('Interface for Experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=1500, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=25, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.00007,  help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='sidx for the gpu to use')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['sintime', 'pos_time_aware', 'time', 'hierarchical', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--test', action='store_true', help='Run in test mode')

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.addHandler(ch)
logger.info(args)

n_feat = np.load('./data/test/Real/processed/seq/ml_{}_node.npy'.format(DATA), allow_pickle=True)
test_real_feat = np.zeros((1400000, 128))

setSeeds(89)

# Load training data
train_real_src_l, train_real_dst_l, train_real_ts_l, train_real_node_count, \
    train_real_node, train_real_time, train_real_ngh_finder, \
    pass_through_d_list, ptd_indices = load_train_real_data(UNIFORM)

# Load training true labels
nodeList_train_real, train_label_l_real = load_real_train_true()

# Load test data
test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, \
    test_real_node, test_real_time, test_real_ngh_finder, test_num_nodes, \
    test_pass_through_d, test_pass_through_d_t, test_ptd_index, test_ptd_cache = load_real_data(DATA)

nodeList_test_real, test_label_l_real = load_real_true('{}'.format(DATA))

train_ts_list, test_ts_list, train_real_ts_list = [], [], []

for idx in range(len(nodeList_train_real)):
    train_real_ts_list.append(np.array([train_real_time[idx]] * len(nodeList_train_real[idx])))

test_real_ts_list = np.array([test_real_time] * len(nodeList_test_real))

num_test_instance = len(nodeList_test_real)
num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)

for k in range(num_test_batch):
    s_idx = k * BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + BATCH_SIZE)
    test_src_l_cut = np.array(nodeList_test_real[s_idx:e_idx])
    test_ts_l_cut = np.array(test_real_ts_list[s_idx:e_idx])
    test_real_ngh_finder.preprocess(tuple(test_src_l_cut), tuple(test_ts_l_cut), NUM_LAYER, NUM_NEIGHBORS)

if torch.cuda.is_available():
    GPU = min(GPU, torch.cuda.device_count() - 1)
    device = torch.device(f'cuda:{GPU}')
else:
    device = torch.device('cpu')

ngh_finder = train_real_ngh_finder[0]

tb_model = TBetGNN(
    train_real_ngh_finder[0],
    test_real_feat,
    attn_mode=ATTN_MODE,
    use_time=USE_TIME,
    agg_method=AGG_METHOD,
    num_layers=NUM_LAYER,
    n_head=NUM_HEADS,
    drop_out=DROP_OUT
)

class MLP(nn.Module):
    def __init__(self, input_dim=256, drop: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        

        self.act = nn.LeakyReLU(negative_slope=0.01)        
        self.dropout = nn.Dropout(p=drop)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu") 
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="leaky_relu")
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x)))

        return self.fc4(x).squeeze(-1)
    
MLP_model = MLP().to(device)
tb_model.to(device)
optimizer = torch.optim.AdamW([
    {"params": tb_model.parameters(), "lr": LEARNING_RATE},
    {"params": MLP_model.parameters(),  "lr": LEARNING_RATE},
], weight_decay=1e-4)

#Load Model
if testing:
    print("Running in test mode...")
    tb_model.load_state_dict(torch.load('./saved_models/model_TGAT_2.pth', weights_only=True))
    MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_2.pth', weights_only=True))

normalizer = LabelNormalizer(method='log1p')
all_train_labels = torch.tensor(np.concatenate(train_label_l_real), dtype=torch.float32)
normalizer.fit(all_train_labels)

with torch.no_grad():
    y_log = normalizer.torch_transform(all_train_labels)
mu = float(y_log.mean())
sigma = float(y_log.std() + 1e-8)
MAX_LOG_Y = float(y_log.max())


all_y = torch.tensor(np.concatenate(train_label_l_real),
                     dtype=torch.float32, device=device)
y_log_all = normalizer.torch_transform(all_y)
mu_t = torch.as_tensor(mu, dtype=y_log_all.dtype, device=device)
sd_t = torch.as_tensor(sigma, dtype=y_log_all.dtype, device=device)
z_all = (y_log_all - mu_t) / sd_t

"""
# Quantile Binning
q = torch.quantile(z_all, torch.tensor([0.2, 0.8], device=device))
TBC_CLASS_THRESHOLDS = q.tolist()
edges = torch.tensor(TBC_CLASS_THRESHOLDS, device=device, dtype=torch.float32)
"""

alpha_cl = 0.3

def training():
    print("Learning Rate : ", LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE * 0.1
    )
    ranking_loss_fn = EnchRankingLoss(margin=1.0)
    contrastive_loss_fn = AdaptiveReweightedSupConLoss(temperature=0.07)
    
    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]

    DEBUG_GRAD = False
    DEBUG_GRAD_HEADS = False

    def safe_loss(loss_tensor: torch.Tensor, anchor_for_graph: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(loss_tensor):
            return (anchor_for_graph * 0.0).sum()
        if loss_tensor.requires_grad:
            return loss_tensor
        return (anchor_for_graph * 0.0).sum()

    
    """
    with torch.no_grad():
        all_y = torch.tensor(np.concatenate(train_label_l_real), dtype=torch.float32, device=device)
        y_log_all = normalizer.torch_transform(all_y)
        z_all = (y_log_all - mu) / sigma
        
        z_non_zero = z_all[all_y > 0]
        
        q_probs = torch.tensor([0.75, 0.92], device=device) 
        TBC_CLASS_THRESHOLDS = torch.quantile(z_non_zero, q_probs).tolist()
        print(f"Applying Quantile Binning: Thresholds set to {TBC_CLASS_THRESHOLDS}")

    TBC_CLASS_THRESHOLDS = [mu - sigma, mu + sigma]
    print("Adaptive Thresholding (Variance-based)")
    """

    for epoch in range(NUM_EPOCH):

        epoch_topk_1, epoch_topk_10, epoch_topk_20 = [], [], []
        epoch_cl_loss, epoch_rank_loss = [], []
        epoch_total_loss = []

        g_epoch = dict(
            mlp_gnorm=[],
            mlp_relupd=[],
            mlp_gabsmax=[],
            mlp_zero_frac=[],
            tgnn_gnorm=[],
            tgnn_relupd=[],
            tgnn_gabsmax=[],
            tgnn_zero_frac=[],
        )
        if DEBUG_GRAD_HEADS:
            g_epoch_buckets = {}

        tb_model.train()
        MLP_model.train()

        graph_indices = list(range(len(train_real_ts_l)))
        for j in graph_indices:
            tb_model.ngh_finder = train_real_ngh_finder[j]

            node_list  = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list    = train_real_ts_list[j]

            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)
            for batch_i in range(num_train_batch):
                optimizer.zero_grad(set_to_none=True)

                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)
                if e_idx - s_idx < 1:
                    continue

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut  = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tb_model.set_ptd_vector(ptd_past_1b)

                src_embed = tb_model.tem_conv2(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    ptd_l=None,
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )
                
                true_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_raw)
                mu_t = torch.as_tensor(mu,    dtype=y_log.dtype, device=device)
                sd_t = torch.as_tensor(sigma, dtype=y_log.dtype, device=device)
                true_z = (y_log - mu_t) / sd_t

                """
                # ---- CL classes (in z-space)
                tbc_classes = torch.zeros_like(true_z, dtype=torch.long, device=device)
                med_mask  = (true_z >= TBC_CLASS_THRESHOLDS[0]) & (true_z < TBC_CLASS_THRESHOLDS[1])
                high_mask = (true_z >= TBC_CLASS_THRESHOLDS[1])
                tbc_classes[med_mask]  = 1
                tbc_classes[high_mask] = 2
                """  
                
                edges = torch.tensor(TBC_CLASS_THRESHOLDS, device=device, dtype=true_z.dtype)
                tbc_classes = torch.bucketize(true_z, edges)
                
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tb_model.ptd_vec[idx]
                ptd_enc  = tb_model.ptd_mlp(ptd_pair)
                
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
                pred_log = MLP_model(mlp_in).squeeze(-1) 
                pred_raw = normalizer.torch_inverse(pred_log).clamp_min(0)
                
                z = tb_model.contrast_head(src_embed)
                cl_loss = contrastive_loss_fn(z, tbc_classes, true_z)
                cl_loss = safe_loss(cl_loss, pred_log)

                rank_loss = ranking_loss_fn(pred_log, y_log)
                rank_loss = safe_loss(rank_loss, pred_log) 
                
                loss = alpha_cl * cl_loss + (1.0 - alpha_cl) * rank_loss

                if torch.isnan(loss).any():
                    continue

                loss.backward()
                
                pre_clip = torch.nn.utils.clip_grad_norm_(
                    list(tb_model.parameters()) + list(MLP_model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                with torch.no_grad():
                    topk_stats = compute_topk_metrics(
                        pred_raw, true_raw, k_list=[1, 5, 10, 20, 30], jac=False
                    )
                if not (topk_stats['Top@1%'] == 0.0 or topk_stats['Top@20%'] == 0.0 or topk_stats['Top@30%'] == 0.0):
                    epoch_topk_1.append(topk_stats['Top@1%'])
                    epoch_topk_10.append(topk_stats['Top@10%'])
                    epoch_topk_20.append(topk_stats['Top@20%'])

                epoch_cl_loss.append(float(cl_loss.detach().cpu()))
                epoch_rank_loss.append(float(rank_loss.detach().cpu()))
                epoch_total_loss.append(float(loss.detach().cpu()))

        scheduler.step()

        avg_topk_1  = float(np.mean(epoch_topk_1))  if epoch_topk_1 else 0.0
        avg_topk_10 = float(np.mean(epoch_topk_10)) if epoch_topk_10 else 0.0
        avg_topk_20 = float(np.mean(epoch_topk_20)) if epoch_topk_20 else 0.0
        print(f"Epoch {epoch:02d} | Top@1%={avg_topk_1:.4f} | Top@10%={avg_topk_10:.4f} | Top@20%={avg_topk_20:.4f} | "
              f"α={alpha_cl:.4f} | CL={np.mean(epoch_cl_loss) if epoch_cl_loss else 0:.4f} | "
              f"Rank={np.mean(epoch_rank_loss) if epoch_rank_loss else 0:.4f} |  "
              f"Total Loss={np.mean(epoch_total_loss) if epoch_total_loss else 0:.4f}")

        if DEBUG_GRAD and (g_epoch["mlp_gnorm"] or g_epoch["tgnn_gnorm"]):
            def _m(x): 
                return float(np.median(x)) if len(x) else float("nan")
            print(
                f"[GRAD][Ep {epoch:02d}] "
                f"MLP: gnorm~{_m(g_epoch['mlp_gnorm']):.2e}, rel~{_m(g_epoch['mlp_relupd']):.2e}, "
                f"gmax~{_m(g_epoch['mlp_gabsmax']):.2e}, zero~{_m(g_epoch['mlp_zero_frac']):.1%} | "
                f"TGNN: gnorm~{_m(g_epoch['tgnn_gnorm']):.2e}, rel~{_m(g_epoch['tgnn_relupd']):.2e}, "
                f"gmax~{_m(g_epoch['tgnn_gabsmax']):.2e}, zero~{_m(g_epoch['tgnn_zero_frac']):.1%}"
            )
            if DEBUG_GRAD_HEADS and 'g_epoch_buckets' in locals():
                def _pull(bk, key):
                    arr = [d[key] for d in g_epoch_buckets.get(bk, [])]
                    return _m(arr)
                line = " | ".join([
                    f"{bk}: g~{_pull(bk,'gnorm'):.2e}, rel~{_pull(bk,'rel'):.2e}, "
                    f"z~{_pull(bk,'zf'):.1%}, gmax~{_pull(bk,'gmax'):.1e}"
                    for bk in ["contrast_head","ptd_mlp","temporal_conv","other"]
                ])
                print(f"    [GRAD][Ep {epoch:02d}] TGNN buckets | {line}")

def evaluation(hint, tgan, mlp_model, sampler, src, ts, label):
    start_time = time.time()
    tgan.ngh_finder = sampler

    lr_model = mlp_model.eval()
    tgan = tgan.eval()

    all_pred_raw, all_true_raw = [], []

    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)

    with torch.no_grad():
        for k in range(num_test_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + BATCH_SIZE)
            if e_idx - s_idx < 1:
                continue

            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut  = np.array(ts[s_idx:e_idx])

            t_cut = float(test_ts_l_cut[0])
            ptd_past_1b = test_ptd_cache.get(t_cut)
            if ptd_past_1b is None:
                ptd_past_1b, _, _ = test_ptd_index.snapshot_partition_all(t_cut)d
                test_ptd_cache[t_cut] = ptd_past_1b
            tgan.set_ptd_vector(ptd_past_1b)
            
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            idx0 = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            true_batch = np.asarray(label, dtype=float)[idx0]
            true_raw = torch.as_tensor(true_batch, dtype=torch.float32, device=device)
            y_log_true = normalizer.torch_transform(true_raw)
            
            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )
            

            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]
            ptd_enc  = tgan.ptd_mlp(ptd_pair)
            mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
            
            pred_log = mlp_model(mlp_in).squeeze(-1)
            pred_raw = normalizer.torch_inverse(pred_log).clamp_min(0)

            all_pred_raw.append(pred_raw.detach().cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))

    pred = np.concatenate(all_pred_raw, axis=0)
    true = np.concatenate(all_true_raw, axis=0)

    results = hits_in_ks(true, pred, Ks=[10, 30, 50])

    for K, (hits, pct) in results.items():
        print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")

    for K, (hits, pct) in results.items():
        print(f"Hits RAW PTD@{K}: {hits}/{K} ({pct:.2f}%)")

    eval_statistics(pred, true, test_pass_through_d, hint)

    e_time = (time.time() - start_time)
    print("Evaluation time for {}: {:.2f} seconds".format(hint, e_time))
    return e_time


if not testing:
    training()
    torch.save(MLP_model.state_dict(), './saved_models/model_MLP_2.pth')
    torch.save(tb_model.state_dict(), './saved_models/model_TGAT_2.pth')

e_time = evaluation('test for real data', tb_model, MLP_model, test_real_ngh_finder, nodeList_test_real, test_real_ts_list, test_label_l_real)
