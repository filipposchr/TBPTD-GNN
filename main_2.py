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
from module import AttnModelPTD_IN_logits, TATKC_PTD_20Aug, TATKC_PTD_19sep 
from scipy.stats import weightedtau, spearmanr, kendalltau
from nx2graphs import load_real_data, load_real_true, load_train_real_data, load_real_train_true
from utils import compute_kendall_tau, compute_topk_metrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from utils import compute_ptd_at_t
from utils import  loss_cal_rank_weighted, pass_through_degree_t, ptd_split_at_t

from scipy import stats


from module import AttnModelPTD_IN_logits_Gated
from module import AttnModelPTD_SimpleAdaptive

# Argument and global variables
parser = argparse.ArgumentParser('Interface for Experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=512, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=25, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.003    , help='learning rate')
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

ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(message)s')

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
    incoming_times_list, outcoming_times_list,  ptd_indices, ptd_caches = load_train_real_data(
        UNIFORM, mode_type, mode_value
)

# Load training true labels
nodeList_train_real, train_label_l_real = load_real_train_true(
    mode_type, mode_value
)



# Load test data
test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, \
    test_real_node, test_real_time, test_real_ngh_finder, test_num_nodes, \
    test_pass_through_d, test_pass_through_d_t,  test_earl_arrival, \
    test_incoming_times, test_outcoming_times, test_temporal_edges, \
         test_temporal_edges_1b, test_ptd_index, test_ptd_cache = load_real_data(
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

#device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    GPU = min(GPU, torch.cuda.device_count() - 1)  # clamp GPU index
    device = torch.device(f'cuda:{GPU}')
else:
    device = torch.device('cpu')

ngh_finder = train_real_ngh_finder[0]

#1; TATKC_PTD_20Aug TATKC_PTD_19sep - TBM_baseline
#2: TATKC_PTD_19Sep_Gated_claude (set_ptd_vector_sparse) - TBM_sparse
#3: TATKC_PTD_19Sep_Gated_claude (set_ptd_vector_ranks)   - TBM_ranks
#4: TATKC_PTD_19Sep_Gated_claude_GPT_version - TBM_4
#5: TATKC_PTD_19Sep_Gated_GPT  - TBM_5
#6: Enchanced_TATKC_PTD_19Sep_Gated_GPT - TBM_enchanced

#kt_ptd, _ = weightedtau(test_pass_through_d, test_label_l_real)
#print("W PTD kendal tau :" , wkt_ptd)



from module import TATKC_PTD_19Sep_Gated_claude, TATKC_PTD_19Sep_Gated_claude_GPT_version, TATKC_PTD22Sep_Gated_claude
from module import TATKC_PTD_19Sep_Gated_GPT
from module import Enchanced_TATKC_PTD_19Sep_Gated_GPT #added 223 sept
from module import TATKC_PTD_19Sep_DualPath_GPT
from module import Enchanced_26Sept


from module import ConservativeSimplifiedModel #works ok 
from module import PTDAgnosticModel

from module import ConservativeSimplifiedModel_gemini #works ok 

from module import ConservativeSimplifiedModel_gemini_CONTR #CONTRASTIVE LEARNING FROM GEMINI
tgnn_model = ConservativeSimplifiedModel_gemini_CONTR(
    train_real_ngh_finder[0],
    test_real_feat,
    attn_mode=ATTN_MODE,
    use_time=USE_TIME,
    agg_method=AGG_METHOD,
    num_layers=NUM_LAYER,
    n_head=NUM_HEADS,
    drop_out=DROP_OUT
)


class MLP2(nn.Module):
    def __init__(self, input_dim=256, drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        return self.fc3(x).squeeze(-1)


class MLP4_upd2(nn.Module):
    def __init__(self, input_dim=256, drop: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32) # Added hidden layer
        self.fc4 = nn.Linear(32, 1)  # Output layer
        

        self.act = nn.LeakyReLU(negative_slope=0.01)
        # -----------------------------------------------------
        
        self.dropout = nn.Dropout(p=drop)
        #self.hnorm = nn.LayerNorm(32, elementwise_affine=True)


        # Initialization must match Leaky ReLU
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu") 
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="leaky_relu")
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x))) # Apply activation and dropout to fc3
        #x = self.hnorm(x)                 # <— normalize 32-d features

        return self.fc4(x).squeeze(-1)
    


class OutputCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))  # exp(log_scale) > 0
        self.bias      = nn.Parameter(torch.zeros(()))
        # NEW: discrete sign (+1 or -1), stored as a buffer; not learned by grad
        self.register_buffer("sign", torch.tensor(1.0))

    def forward(self, x):
        return self.sign * torch.exp(self.log_scale) * x + self.bias


class AdaptiveOutputCalibrator(nn.Module):
    def __init__(self, target_stats=None):
        super().__init__()
        # Learn to map from normalized space back to true scale
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        # Optional: target statistics for initialization
        if target_stats:
            self.scale.data = torch.tensor([target_stats['std']])
            self.shift.data = torch.tensor([target_stats['mean']])
    
    def forward(self, x):
        # Apply learned affine transformation
        return x * self.scale + self.shift
    

MLP_model = MLP4_upd2().to(device)
calib = AdaptiveOutputCalibrator().to(device)


#optimizer = torch.optim.Adam(list(tgnn_model.parameters()) + list(MLP_model.parameters()),lr=LEARNING_RATE)

optimizer = torch.optim.AdamW([
    {"params": tgnn_model.parameters(), "lr": LEARNING_RATE},
    {"params": MLP_model.parameters(),  "lr": LEARNING_RATE},
    {"params": calib.parameters(), "lr": LEARNING_RATE * 14.0},
], weight_decay=1e-4)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)


tgnn_model.to(device)


#Load Model
if testing:
    print("Running in test mode...")
    tgnn_model.load_state_dict(torch.load('./saved_models/model_TGAT_2.pth', weights_only=True))
    MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_2.pth', weights_only=True))


#------------------------------------
#HELPERS
from scipy.stats import weightedtau

def unified_evaluation_with_statistics(pred, true, ptd_raw, dataset_name=""):
    """
    UNIFIED evaluation that uses the SAME data for both metrics table and statistics.
    This fixes the data inconsistency bug.
    """
    
    # Convert to numpy arrays and ensure same length
    pred = np.asarray(pred, dtype=float).flatten()
    true = np.asarray(true, dtype=float).flatten() 
    ptd_raw = np.asarray(ptd_raw, dtype=float).flatten()
    
    # CRITICAL: Use the shortest length to ensure alignment
    min_len = min(len(pred), len(true), len(ptd_raw))
    pred = pred[:min_len]
    true = true[:min_len] 
    ptd_raw = ptd_raw[:min_len]
    
    # Apply the same mask to all arrays
    valid_mask = np.isfinite(pred) & np.isfinite(true) & np.isfinite(ptd_raw)
    pred = pred[valid_mask]
    true = true[valid_mask]
    ptd_raw = ptd_raw[valid_mask]
    
    n_points = len(pred)
    print(f"\n=== UNIFIED EVALUATION: {dataset_name} ===")
    print(f"Data points: {n_points}")
    
    if n_points < 3:
        print("ERROR: Insufficient data points")
        return
    
    # ===== COMPUTE ALL METRICS ON SAME DATA =====
    
    # Basic correlations (same as your existing table)
    try:
        kt_model_all, p_kt_model = weightedtau(pred, true)
        kt_ptd_all, p_kt_ptd = weightedtau(ptd_raw, true)
        kt_diff_all = kt_model_all - kt_ptd_all
        
        sp_model_all, p_sp_model = spearmanr(pred, true) 
        sp_ptd_all, p_sp_ptd = spearmanr(ptd_raw, true)
        sp_diff_all = sp_model_all - sp_ptd_all
        
        # Non-zero subset
        nonzero_mask = (true > 0)
        if nonzero_mask.sum() >= 3:
            kt_model_nz = weightedtau(pred[nonzero_mask], true[nonzero_mask])[0]
            kt_ptd_nz = weightedtau(ptd_raw[nonzero_mask], true[nonzero_mask])[0]
            kt_diff_nz = kt_model_nz - kt_ptd_nz
            
            sp_model_nz = spearmanr(pred[nonzero_mask], true[nonzero_mask])[0]
            sp_ptd_nz = spearmanr(ptd_raw[nonzero_mask], true[nonzero_mask])[0]
            sp_diff_nz = sp_model_nz - sp_ptd_nz
        else:
            kt_model_nz = kt_ptd_nz = kt_diff_nz = np.nan
            sp_model_nz = sp_ptd_nz = sp_diff_nz = np.nan
        
        # Log1p versions
        pred_log = np.log1p(np.clip(pred, 0, None))
        ptd_log = np.log1p(np.clip(ptd_raw, 0, None))
        true_log = np.log1p(np.clip(true, 0, None))
        
        sp_model_log = spearmanr(pred_log, true_log)[0]
        sp_ptd_log = spearmanr(ptd_log, true_log)[0]
        sp_diff_log = sp_model_log - sp_ptd_log
        
        # Zero-aware log1p spearman
        def spearman_log1p_zero_aware(a, b):
            x = np.log1p(np.clip(a, 0, None))
            y = np.log1p(np.clip(b, 0, None))
            rx = x.argsort().argsort().astype(float)
            ry = y.argsort().argsort().astype(float)
            rx = (rx - rx.mean()) / (rx.std() + 1e-12)
            ry = (ry - ry.mean()) / (ry.std() + 1e-12)
            return float(np.clip((rx * ry).mean(), -1, 1))
        
        sp_model_log_za = spearman_log1p_zero_aware(pred, true)
        sp_ptd_log_za = spearman_log1p_zero_aware(ptd_raw, true) 
        sp_diff_log_za = sp_model_log_za - sp_ptd_log_za
        
        """
        # ===== DISPLAY UNIFIED RESULTS TABLE =====
        print(f"\nMetrics Table (ALL DATA POINTS: {n_points})")
        print("Metric                   Model       RAW PTD      (Model-PTD)")
        print(f"Kendall Tau (all)        {kt_model_all:8.4f}   {kt_ptd_all:8.4f}   {kt_diff_all:+8.4f}")
        print(f"Kendall Tau (non-zero)   {kt_model_nz:8.4f}   {kt_ptd_nz:8.4f}   {kt_diff_nz:+8.4f}")
        print(f"Spearman (all)           {sp_model_all:8.4f}   {sp_ptd_all:8.4f}   {sp_diff_all:+8.4f}")
        print(f"Spearman (non-zero)      {sp_model_nz:8.4f}   {sp_ptd_nz:8.4f}   {sp_diff_nz:+8.4f}")
        print(f"Spearman (log1p)         {sp_model_log:8.4f}   {sp_ptd_log:8.4f}   {sp_diff_log:+8.4f}")
        print(f"Spearman (log1p-zero aware) {sp_model_log_za:8.4f}   {sp_ptd_log_za:8.4f}   {sp_diff_log_za:+8.4f}")
        
        """

        # ===== STATISTICAL SIGNIFICANCE (SAME DATA) =====
        print(f"\nStatistical Significance (SAME {n_points} data points)")
        print(f"Weight Kendall Tau:")
        print(f"  Model: τ={kt_model_all:.4f}, p={p_kt_model:.4f} {'*' if p_kt_model < 0.05 else ''}")
        print(f"  PTD:   τ={kt_ptd_all:.4f}, p={p_kt_ptd:.4f} {'*' if p_kt_ptd < 0.05 else ''}")
        
        if abs(kt_diff_all) < 0.01:
            kt_interp = "equivalent performance"
        elif kt_diff_all > 0:
            kt_interp = f"model BETTER by {kt_diff_all:.4f}"
        else:
            kt_interp = f"model WORSE by {abs(kt_diff_all):.4f}"
        print(f"  Difference: {kt_diff_all:+.4f} ({kt_interp})")
        
        print(f"\nSpearman Correlation:")
        print(f"  Model: ρ={sp_model_all:.4f}, p={p_sp_model:.2e} {'*' if p_sp_model < 0.05 else ''}")
        print(f"  PTD:   ρ={sp_ptd_all:.4f}, p={p_sp_ptd:.2e} {'*' if p_sp_ptd < 0.05 else ''}")
        
        if abs(sp_diff_all) < 0.01:
            sp_interp = "equivalent performance"
        elif sp_diff_all > 0:
            sp_interp = f"model BETTER by {sp_diff_all:.4f}"
        else:
            sp_interp = f"model WORSE by {abs(sp_diff_all):.4f}"
        print(f"  Difference: {sp_diff_all:+.4f} ({sp_interp})")
        
        # ===== STEIGER'S TEST FOR SIGNIFICANCE OF DIFFERENCE =====
        try:
            r_model_ptd = np.corrcoef(pred, ptd_raw)[0, 1]
            n = len(pred)
            
            # Steiger's test for dependent correlations
            r_yx = sp_model_all  # model vs true
            r_zx = sp_ptd_all    # ptd vs true
            r_yz = r_model_ptd   # model vs ptd
            
            if abs(r_yz) >= 0.999:
                z_steiger = 0.0
                p_steiger = 1.0
            else:
                r_mean = (r_yx + r_zx) / 2
                h = (1 - r_yz) / (2 * (1 - r_mean**2))
                
                if h <= 0:
                    z_steiger = 0.0
                    p_steiger = 1.0
                else:
                    z_steiger = (r_yx - r_zx) * np.sqrt((n - 3) * h)
                    p_steiger = 2 * (1 - stats.norm.cdf(abs(z_steiger)))
            
            print(f"\nSteiger's Test (is the difference significant?):")
            print(f"  Z = {z_steiger:.3f}, p = {p_steiger:.2e }")
            print(f"  Model-PTD correlation: r = {r_model_ptd:.3f}")
            
            if p_steiger < 0.05:
                print(f"  → Difference IS statistically significant")
            else:
                print(f"  → Difference is NOT statistically significant")
                
        except Exception as e:
            print(f"  Steiger's test failed: {e}")
        
        # ===== INTERPRETATION SUMMARY =====
        print(f"\n=== SUMMARY INTERPRETATION ===")
        
        # Overall performance assessment
        main_metrics_better = (kt_diff_all > 0.02 and sp_diff_all > 0.02)
        main_metrics_worse = (kt_diff_all < -0.02 and sp_diff_all < -0.02) 
        
        if main_metrics_better:
            print("✅ MODEL OUTPERFORMS PTD on primary metrics")
        elif main_metrics_worse:
            print("❌ MODEL UNDERPERFORMS PTD on primary metrics")
        else:
            print("⚖️  MODEL and PTD show SIMILAR performance")
        
        # Dataset difficulty assessment
        if kt_ptd_all > 0.8 and sp_ptd_all > 0.8:
            print("🎯 HIGH-QUALITY dataset: PTD already excellent, limited room for improvement")
        elif kt_ptd_all < 0.3 and sp_ptd_all < 0.3:
            print("⚠️  CHALLENGING dataset: PTD poorly predictive")
        else:
            print("📊 MODERATE dataset: Good opportunity for model improvements")
            
        return {
            'kendall_model': kt_model_all,
            'kendall_ptd': kt_ptd_all, 
            'kendall_diff': kt_diff_all,
            'spearman_model': sp_model_all,
            'spearman_ptd': sp_ptd_all,
            'spearman_diff': sp_diff_all,
            'n_points': n_points
        }
        
    except Exception as e:
        print(f"ERROR in unified evaluation: {e}")
        return None



class LabelNormalizer:
    """
    method = 'log1p'  -> y_n = log1p(alpha * y)
    method = 'zscore' -> y_n = (y - mu) / (sigma + eps)
    """
    def __init__(self, method: str = 'log1p', eps: float = 1e-8):
        assert method in ('log1p', 'zscore')
        self.method = method
        self.eps = eps
        self.alpha = None   # for log1p
        self.mu = None      # for zscore
        self.sigma = None   # for zscore

    def fit(self, y_all: torch.Tensor):
        y_all = y_all.detach().float().reshape(-1)
        if self.method == 'log1p':
            pos = y_all[y_all > 0]
            if pos.numel() == 0:
                self.alpha = 1.0
            else:
                med = torch.median(pos)
                # map median to ~1.0: log1p(alpha*med) ≈ 1 -> alpha ≈ (exp(1)-1)/med
                self.alpha = float((torch.exp(torch.tensor(1.0)) - 1.0) / (med + self.eps))
        else:
            self.mu = float(torch.mean(y_all))
            self.sigma = float(torch.std(y_all) + self.eps)

    def torch_transform(self, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        if self.method == 'log1p':
            alpha = torch.tensor(self.alpha, dtype=y.dtype, device=y.device)
            return torch.log1p(alpha * torch.clamp(y, min=0.0))
        else:
            mu = torch.tensor(self.mu, dtype=y.dtype, device=y.device)
            sigma = torch.tensor(self.sigma, dtype=y.dtype, device=y.device)
            return (y - mu) / sigma

    def torch_inverse(self, y_n: torch.Tensor) -> torch.Tensor:
        y_n = y_n.float()
        if self.method == 'log1p':
            alpha = torch.tensor(self.alpha, dtype=y_n.dtype, device=y_n.device)
            return torch.clamp((torch.expm1(y_n) / (alpha + self.eps)), min=0.0)
        else:
            mu = torch.tensor(self.mu, dtype=y_n.dtype, device=y_n.device)
            sigma = torch.tensor(self.sigma, dtype=y_n.dtype, device=y_n.device)
            return y_n * sigma + mu

    # numpy helpers if you need them elsewhere
    def np_inverse(self, y_n_np):
        import numpy as np
        y_n = np.asarray(y_n_np, np.float32)
        if self.method == 'log1p':
            return np.maximum(np.expm1(y_n) / (self.alpha + self.eps), 0.0)
        else:
            return y_n * self.sigma + self.mu



class PTDEpochTracker:

    """
    Track PTD statistics across batches and report epoch summary.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset for new epoch."""
        self.raw_stats = {
            'min': [], 'max': [], 'mean': [], 'std': [], 
            'non_zero_ratio': []
        }
        self.encoded_stats = {
            'zlog_min': [], 'zlog_max': [], 'zlog_mean': [], 'zlog_std': [],
            'pct_min': [], 'pct_max': [], 'pct_mean': [], 'pct_std': [],
            'unique_ratio': []  # How many unique values vs total
        }
        self.quality_stats = {
            'quality': [], 'alpha': [], 'robust_cv': [], 'non_zero': []
        }
        self.batch_count = 0
    
    def record_raw_ptd(self, ptd_array):
        """Record raw PTD statistics before encoding."""
        self.raw_stats['min'].append(float(ptd_array.min()))
        self.raw_stats['max'].append(float(ptd_array.max()))
        self.raw_stats['mean'].append(float(ptd_array.mean()))
        self.raw_stats['std'].append(float(ptd_array.std()))
        self.raw_stats['non_zero_ratio'].append(float((ptd_array > 0.01).mean()))
    
    def record_encoded_ptd(self, ptd_encoded):
        """Record encoded PTD statistics after set_ptd_vector."""
        if isinstance(ptd_encoded, torch.Tensor):
            ptd_encoded = ptd_encoded.cpu().numpy()
        
        # Column 0: zlog values
        self.encoded_stats['zlog_min'].append(float(ptd_encoded[:, 0].min()))
        self.encoded_stats['zlog_max'].append(float(ptd_encoded[:, 0].max()))
        self.encoded_stats['zlog_mean'].append(float(ptd_encoded[:, 0].mean()))
        self.encoded_stats['zlog_std'].append(float(ptd_encoded[:, 0].std()))
        
        # Column 1: percentile values
        self.encoded_stats['pct_min'].append(float(ptd_encoded[:, 1].min()))
        self.encoded_stats['pct_max'].append(float(ptd_encoded[:, 1].max()))
        self.encoded_stats['pct_mean'].append(float(ptd_encoded[:, 1].mean()))
        self.encoded_stats['pct_std'].append(float(ptd_encoded[:, 1].std()))
        
        # Unique ratio (important for detecting collapse)
        unique_vals = len(np.unique(np.round(ptd_encoded[:, 0], 3)))  # Round to 3 decimals
        total_vals = len(ptd_encoded)
        self.encoded_stats['unique_ratio'].append(unique_vals / total_vals)
        
        self.batch_count += 1
    
    def record_attention_stats(self, attn_module):
        """Record statistics from attention module."""
        if hasattr(attn_module, 'ptd_quality_history') and attn_module.ptd_quality_history:
            self.quality_stats['quality'].extend(attn_module.ptd_quality_history)
            attn_module.ptd_quality_history = []
        
        if hasattr(attn_module, 'alpha_history') and attn_module.alpha_history:
            self.quality_stats['alpha'].extend(attn_module.alpha_history)
            attn_module.alpha_history = []
        elif hasattr(attn_module, 'alpha_attn'):
            self.quality_stats['alpha'].append(attn_module.alpha_attn)
    
    def get_epoch_summary(self):
        """Get comprehensive epoch summary."""
        summary = {
            'raw_ptd': {},
            'encoded_ptd': {},
            'quality': {},
            'warnings': []
        }
        
        # Raw PTD summary
        if self.raw_stats['mean']:
            summary['raw_ptd'] = {
                'mean': np.mean(self.raw_stats['mean']),
                'std': np.mean(self.raw_stats['std']),
                'max_avg': np.mean(self.raw_stats['max']),
                'max_peak': np.max(self.raw_stats['max']),
                'non_zero_ratio': np.mean(self.raw_stats['non_zero_ratio'])
            }
        
        # Encoded PTD summary
        if self.encoded_stats['zlog_std']:
            summary['encoded_ptd'] = {
                'zlog_std': np.mean(self.encoded_stats['zlog_std']),
                'zlog_range': np.mean([m - n for m, n in zip(
                    self.encoded_stats['zlog_max'], 
                    self.encoded_stats['zlog_min']
                )]),
                'pct_std': np.mean(self.encoded_stats['pct_std']),
                'unique_ratio': np.mean(self.encoded_stats['unique_ratio'])
            }
            
            # Check for encoding problems
            if summary['encoded_ptd']['zlog_std'] < 0.1:
                summary['warnings'].append("PTD encoding variance collapsed (std < 0.1)")
            if summary['encoded_ptd']['unique_ratio'] < 0.3:
                summary['warnings'].append("PTD encoding has low diversity (< 30% unique)")
        
        # Quality and alpha summary
        if self.quality_stats['alpha']:
            summary['quality'] = {
                'alpha_mean': np.mean(self.quality_stats['alpha']),
                'alpha_std': np.std(self.quality_stats['alpha']),
                'alpha_min': np.min(self.quality_stats['alpha']),
                'alpha_max': np.max(self.quality_stats['alpha'])
            }
            
            if self.quality_stats['quality']:
                summary['quality']['quality_mean'] = np.mean(self.quality_stats['quality'])
        
        summary['batch_count'] = self.batch_count
        
        return summary
    
    def print_epoch_summary(self, epoch):
        """Print formatted epoch summary."""
        summary = self.get_epoch_summary()
        
        print(f"\n{'='*60}")
        print(f"PTD Epoch {epoch:02d} Summary ({summary['batch_count']} batches)")
        print(f"{'='*60}")
        
        if summary['raw_ptd']:
            print(f"Raw PTD:")
            print(f"  Mean: {summary['raw_ptd']['mean']:.3f} ± {summary['raw_ptd']['std']:.3f}")
            print(f"  Max (avg): {summary['raw_ptd']['max_avg']:.1f}, Peak: {summary['raw_ptd']['max_peak']:.1f}")
            print(f"  Non-zero: {summary['raw_ptd']['non_zero_ratio']:.1%}")
        
        if summary['encoded_ptd']:
            print(f"Encoded PTD:")
            print(f"  Z-log std: {summary['encoded_ptd']['zlog_std']:.3f}")
            print(f"  Z-log range: {summary['encoded_ptd']['zlog_range']:.3f}")
            print(f"  Unique values: {summary['encoded_ptd']['unique_ratio']:.1%}")
        
        if summary['quality']:
            print(f"Adaptive Alpha:")
            print(f"  Range: [{summary['quality']['alpha_min']:.3f}, {summary['quality']['alpha_max']:.3f}]")
            print(f"  Mean: {summary['quality']['alpha_mean']:.3f} ± {summary['quality']['alpha_std']:.3f}")
            if 'quality_mean' in summary['quality']:
                print(f"  Quality: {summary['quality']['quality_mean']:.3f}")
        
        if summary['warnings']:
            print(f"⚠️  Warnings:")
            for warning in summary['warnings']:
                print(f"  - {warning}")
        
        print(f"{'='*60}\n")


def _average_ranks_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    sorter = np.argsort(x, kind="mergesort")
    x_sorted = x[sorter]
    ranks = np.empty_like(x_sorted, dtype=float)
    vals, starts, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    for s, c in zip(starts, counts):
        ranks[s:s+c] = s + (c - 1)/2.0 + 1.0  # 1-based average rank
    out = np.empty_like(ranks)
    out[sorter] = ranks
    return out

def spearman_log1p_zero_aware(a, b):
    x = np.log1p(np.asarray(a)); y = np.log1p(np.asarray(b))
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx = (rx - rx.mean()) / (rx.std()+1e-12); ry = (ry - ry.mean()) / (ry.std()+1e-12)
    return float(np.clip((rx*ry).mean(), -1, 1))


def spearman_no_pandas(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    ra = _average_ranks_np(a); rb = _average_ranks_np(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum())
    return 0.0 if denom == 0 else float((ra * rb).sum() / denom)


def _spearman_log1p(a, b):
    a = np.log1p(np.clip(np.asarray(a, dtype=float).ravel(), 0, None))
    b = np.log1p(np.clip(np.asarray(b, dtype=float).ravel(), 0, None))
    return float(spearman_no_pandas(a, b))


def compute_all_metrics(pred, true):
    """Return a dict with all requested metrics for 1D arrays."""
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()

    # Your helper that returns (kt_all, kt_nz, sp_all, sp_nz)
    kt_all, kt_nz, sp_all, sp_nz = compute_kendall_tau(pred, true)
    sp_log_zero_aware = spearman_log1p_zero_aware(pred,true)
    sp_log = _spearman_log1p(pred, true)

    return {
        "kt_all":  kt_all,
        "kt_nz":   kt_nz,
        "sp_all":  sp_all,
        "sp_log":  sp_log,
        "sp_nz":   sp_nz,
        "sp_log_zero_aware": sp_log_zero_aware,

    }


def ptd_snapshot_report(ptd_vec_np, label_np, k_list=(50,100)):
    """
    ptd_vec_np: raw PTD per node BEFORE log1p*scale or AFTER – either is fine, this is comparative.
    label_np:   ground-truth scalar per node for the same snapshot
    """
    raw = np.asarray(ptd_vec_np, dtype=np.float32)
    y   = np.asarray(label_np, dtype=np.float32)
    ok  = np.isfinite(raw) & np.isfinite(y)
    if ok.sum() < 3:
        return {"spearman": np.nan, "topk_jaccard": {}}

    # percentile ranking of PTD
    ranks = raw.argsort().argsort()
    pct   = (ranks + 0.5) / max(1, len(raw))

    sp = spearmanr(raw[ok], y[ok]).correlation
    out = {"spearman": float(sp), "raw_min": float(raw.min()), "raw_max": float(raw.max()),
           "pct_mean": float(pct.mean()), "pct_std": float(pct.std())}

    # top-k overlap (PTD vs label)
    order_ptd = np.argsort(-raw)
    order_y   = np.argsort(-y)
    for k in k_list:
        A = set(order_ptd[:k].tolist())
        B = set(order_y[:k].tolist())
        j = len(A & B) / max(1, len(A | B))
        out[f"jacc_top{k}"] = float(j)
    return out


def print_model_vs_ptd_table(pred, true, ptd_raw, title=""):
    m = compute_all_metrics(pred, true)
    b = compute_all_metrics(ptd_raw, true)

    def row(name, k):
        mv = m[k]; bv = b[k]; dv = mv - bv
        return f"{name:<24} {mv:>8.4f}   {bv:>8.4f}   {dv:>+8.4f}"

    print(title)
    print("Metric                   Model       RAW PTD      Î”(Model-PTD)")
    print(row("Kendall Tau (all)",      "kt_all"))
    print(row("Kendall Tau (non-zero)", "kt_nz"))
    print(row("Spearman (all)",         "sp_all"))
    print(row("Spearman (non-zero)",    "sp_nz"))
    print(row("Spearman (log1p)",       "sp_log"))
    print(row("Spearman (log1p-zero aware)",       "sp_log_zero_aware"))


def analyze_ptd_distribution(ptd_all_nodes, label_l_cut):
    """
    Analyze PTD distribution and its relationship with labels.
    Add this to your training loop for better understanding.
    """
    ptd_array = np.array(ptd_all_nodes[:len(label_l_cut)])
    label_array = np.array(label_l_cut)
    
    # Remove outliers using IQR method
    q1 = np.percentile(ptd_array, 25)
    q3 = np.percentile(ptd_array, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Count outliers
    outliers = (ptd_array < lower_bound) | (ptd_array > upper_bound)
    outlier_ratio = outliers.mean()
    
    # Compute correlation without outliers
    if (~outliers).sum() > 10:
        ptd_clean = ptd_array[~outliers]
        label_clean = label_array[~outliers]
        if ptd_clean.std() > 0 and label_clean.std() > 0:
            corr_clean = np.corrcoef(ptd_clean, label_clean)[0, 1]
        else:
            corr_clean = 0
    else:
        corr_clean = 0
    
    # Original correlation (with outliers)
    if ptd_array.std() > 0 and label_array.std() > 0:
        corr_orig = np.corrcoef(ptd_array, label_array)[0, 1]
    else:
        corr_orig = 0
    
    return {
        'outlier_ratio': outlier_ratio,
        'corr_clean': corr_clean,
        'corr_orig': corr_orig,
        'q1': q1,
        'q3': q3,
        'median': np.median(ptd_array),
        'max': ptd_array.max(),
        'non_zero_ratio': (ptd_array > 0.01).mean()
    }
    

from scipy.stats import weightedtau, spearmanr
from scipy import stats
import numpy as np

def unified_evaluation_with_statistics2(pred, true, ptd_raw, dataset_name=""):
    """
    UNIFIED evaluation that uses the SAME data for both metrics table and statistics.
    Now also prints STD and MAE.
    """
    # Convert to numpy arrays and ensure same length
    pred = np.asarray(pred, dtype=float).flatten()
    true = np.asarray(true, dtype=float).flatten()
    ptd_raw = np.asarray(ptd_raw, dtype=float).flatten()

    # Use shortest length to ensure alignment
    min_len = min(len(pred), len(true), len(ptd_raw))
    pred = pred[:min_len]
    true = true[:min_len]
    ptd_raw = ptd_raw[:min_len]

    # Apply the same mask to all arrays
    valid_mask = np.isfinite(pred) & np.isfinite(true) & np.isfinite(ptd_raw)
    pred = pred[valid_mask]
    true = true[valid_mask]
    ptd_raw = ptd_raw[valid_mask]

    n_points = len(pred)
    print(f"\n=== UNIFIED EVALUATION: {dataset_name} ===")
    print(f"Data points: {n_points}")

    if n_points < 3:
        print("ERROR: Insufficient data points")
        return

    # ---- NEW: dispersion + error summary ----
    abs_err = np.abs(pred - true)
    
    mae = float(abs_err.mean())                     # standard MAE (mean absolute error)
    min_ae = float(abs_err.min())                   # minimum absolute error (if you need it)
    max_ae = float(abs_err.max())                   # (handy for context)
    std_pred = float(pred.std(ddof=1))              # sample STD
    std_true = float(true.std(ddof=1))              # sample STD

    range_true = float(true.max() - true.min())
    iqr_true = float(np.percentile(true, 75) - np.percentile(true, 25))
    mean_abs_true = float(np.mean(np.abs(true)))
    eps = 1e-12  # for zero-safety where appropriate

    # Normalized MAEs
    nmae_range = float(mae / range_true) if range_true > 0 else np.nan
    nmae_std   = float(mae / std_true)   if std_true   > 0 else np.nan
    nmae_iqr   = float(mae / iqr_true)   if iqr_true   > 0 else np.nan
    nmae_meanabs = float(mae / (mean_abs_true + eps))

    print("\nDispersion & Error:")
    print(f"  STD(pred) = {std_pred:.6g}")
    print(f"  STD(true) = {std_true:.6g}")
    print(f"  MAE (mean abs error) = {mae:.6g}")
    print(f"  NMAE (by range)        = {nmae_range:.6g}    # MAE / (max-min)")
    print(f"  MinAE (minimum abs error) = {min_ae:.6g}")
    print(f"  MaxAE (maximum abs error) = {max_ae:.6g}")

    # ===== COMPUTE ALL METRICS ON SAME DATA =====
    try:
        kt_model_all, p_kt_model = kendalltau(pred, true)
        kt_ptd_all, p_kt_ptd = kendalltau(ptd_raw, true)
        kt_diff_all = kt_model_all - kt_ptd_all

        sp_model_all, p_sp_model = spearmanr(pred, true)
        sp_ptd_all, p_sp_ptd = spearmanr(ptd_raw, true)
        sp_diff_all = sp_model_all - sp_ptd_all

        # Non-zero subset
        nonzero_mask = (true > 0)
        if nonzero_mask.sum() >= 3:
            kt_model_nz = kendalltau(pred[nonzero_mask], true[nonzero_mask])[0]
            kt_ptd_nz = kendalltau(ptd_raw[nonzero_mask], true[nonzero_mask])[0]
            kt_diff_nz = kt_model_nz - kt_ptd_nz

            sp_model_nz = spearmanr(pred[nonzero_mask], true[nonzero_mask])[0]
            sp_ptd_nz = spearmanr(ptd_raw[nonzero_mask], true[nonzero_mask])[0]
            sp_diff_nz = sp_model_nz - sp_ptd_nz
        else:
            kt_model_nz = kt_ptd_nz = kt_diff_nz = np.nan
            sp_model_nz = sp_ptd_nz = sp_diff_nz = np.nan

        # Log1p versions
        pred_log = np.log1p(np.clip(pred, 0, None))
        ptd_log = np.log1p(np.clip(ptd_raw, 0, None))
        true_log = np.log1p(np.clip(true, 0, None))

        sp_model_log = spearmanr(pred_log, true_log)[0]
        sp_ptd_log = spearmanr(ptd_log, true_log)[0]
        sp_diff_log = sp_model_log - sp_ptd_log

        # Zero-aware log1p spearman
        def spearman_log1p_zero_aware(a, b):
            x = np.log1p(np.clip(a, 0, None))
            y = np.log1p(np.clip(b, 0, None))
            rx = x.argsort().argsort().astype(float)
            ry = y.argsort().argsort().astype(float)
            rx = (rx - rx.mean()) / (rx.std() + 1e-12)
            ry = (ry - ry.mean()) / (ry.std() + 1e-12)
            return float(np.clip((rx * ry).mean(), -1, 1))

        sp_model_log_za = spearman_log1p_zero_aware(pred, true)
        sp_ptd_log_za = spearman_log1p_zero_aware(ptd_raw, true)
        sp_diff_log_za = sp_model_log_za - sp_ptd_log_za

        # ===== Statistical Significance =====
        print(f"\nStatistical Significance (SAME {n_points} data points)")
        print(f"Weighted Kendall Tau:")
        print(f"  Model: τ={kt_model_all:.4f}, p={p_kt_model:.4f} {'*' if p_kt_model < 0.05 else ''}")
        print(f"  PTD:   τ={kt_ptd_all:.4f}, p={p_kt_ptd:.4f} {'*' if p_kt_ptd < 0.05 else ''}")

        if abs(kt_diff_all) < 0.01:
            kt_interp = "equivalent performance"
        elif kt_diff_all > 0:
            kt_interp = f"model BETTER by {kt_diff_all:.4f}"
        else:
            kt_interp = f"model WORSE by {abs(kt_diff_all):.4f}"
        print(f"  Difference: {kt_diff_all:+.4f} ({kt_interp})")

        print(f"\nSpearman Correlation:")
        print(f"  Model: ρ={sp_model_all:.4f}, p={p_sp_model:.2e} {'*' if p_sp_model < 0.05 else ''}")
        print(f"  PTD:   ρ={sp_ptd_all:.4f}, p={p_sp_ptd:.2e} {'*' if p_sp_ptd < 0.05 else ''}")

        if abs(sp_diff_all) < 0.01:
            sp_interp = "equivalent performance"
        elif sp_diff_all > 0:
            sp_interp = f"model BETTER by {sp_diff_all:.4f}"
        else:
            sp_interp = f"model WORSE by {abs(sp_diff_all):.4f}"
        print(f"  Difference: {sp_diff_all:+.4f} ({sp_interp})")

        # ===== Steiger's Test =====
        try:
            r_model_ptd = float(np.corrcoef(pred, ptd_raw)[0, 1])
            n = len(pred)

            r_yx = sp_model_all  # model vs true
            r_zx = sp_ptd_all    # ptd vs true
            r_yz = r_model_ptd   # model vs ptd

            if abs(r_yz) >= 0.999:
                z_steiger = 0.0
                p_steiger = 1.0
            else:
                r_mean = (r_yx + r_zx) / 2
                h = (1 - r_yz) / (2 * (1 - r_mean**2))
                if h <= 0:
                    z_steiger = 0.0
                    p_steiger = 1.0
                else:
                    z_steiger = (r_yx - r_zx) * np.sqrt((n - 3) * h)
                    p_steiger = 2 * (1 - stats.norm.cdf(abs(z_steiger)))

            print(f"\nSteiger's Test (is the difference significant?):")
            print(f"  Z = {z_steiger:.3f}, p = {p_steiger:.2e}")
            print(f"  Model-PTD correlation: r = {r_model_ptd:.3f}")
            if p_steiger < 0.05:
                print("  → Difference IS statistically significant")
            else:
                print("  → Difference is NOT statistically significant")

        except Exception as e:
            print(f"  Steiger's test failed: {e}")

        # ===== Interpretation Summary =====
        print(f"\n=== SUMMARY INTERPRETATION ===")
        main_metrics_better = (kt_diff_all > 0.02 and sp_diff_all > 0.02)
        main_metrics_worse  = (kt_diff_all < -0.02 and sp_diff_all < -0.02)

        if main_metrics_better:
            print("✅ MODEL OUTPERFORMS PTD on primary metrics")
        elif main_metrics_worse:
            print("❌ MODEL UNDERPERFORMS PTD on primary metrics")
        else:
            print("⚖️  MODEL and PTD show SIMILAR performance")

        if kt_ptd_all > 0.8 and sp_ptd_all > 0.8:
            print("🎯 HIGH-QUALITY dataset: PTD already excellent, limited room for improvement")
        elif kt_ptd_all < 0.3 and sp_ptd_all < 0.3:
            print("⚠️  CHALLENGING dataset: PTD poorly predictive")
        else:
            print("📊 MODERATE dataset: Good opportunity for model improvements")

        return {
            'kendall_model': kt_model_all,
            'kendall_ptd': kt_ptd_all,
            'kendall_diff': kt_diff_all,
            'spearman_model': sp_model_all,
            'spearman_ptd': sp_ptd_all,
            'spearman_diff': sp_diff_all,
            'n_points': n_points,
            # new outputs
            'std_pred': std_pred,
            'std_true': std_true,
            'mae': mae,
            'min_abs_error': min_ae,
            'max_abs_error': max_ae,
        }

    except Exception as e:
        print(f"ERROR in unified evaluation: {e}")
        return None


# change creation to two-stage
normalizer = LabelNormalizer(method='log1p')  # keep existing
all_train_labels = torch.tensor(np.concatenate(train_label_l_real), dtype=torch.float32)
normalizer.fit(all_train_labels)

# NEW: compute z-params on top of log1p
with torch.no_grad():
    y_log = normalizer.torch_transform(all_train_labels)
mu = float(y_log.mean())
sigma = float(y_log.std() + 1e-8)
print(f"[LabelNormalizer-2stage] mu={mu:.4f}, sigma={sigma:.4f}")


#------------------------------------

#Losses:
def loss_cal_rank_weighted(y_pred, y_true, sample_num: int = None,
                           p_pos: float = 0.7, min_gap: float = 0.05, margin: float = 0.5, device=None):
    """
    y_pred,y_true: normalized (z-scored log1p) tensors, shape [N].
    p_pos: prob that at least one of the pair has y_true > 0 (raw positivity proxy in log space).
    min_gap: min |y_true[i]-y_true[j]| (in normalized space) to keep pair.
    margin: MarginRankingLoss margin (smaller after z-scoring).
    """
    if device is None: device = y_pred.device
    N = y_true.shape[0]
    if sample_num is None:
        sample_num = max(100 * N, 10000)

    # indices sorted by descending true
    order = torch.argsort(-y_true)
    # sample candidate pairs uniformly on positions (stable)
    i = torch.randint(0, N, (sample_num,), device=device)
    j = torch.randint(0, N, (sample_num,), device=device)

    a = y_pred[order[i]]
    b = y_pred[order[j]]

    # ranking target: earlier (higher y_true) should be larger
    tgt = torch.sign(-(i - j)).float()

    # filter by min_gap on true labels
    gap_ok = (y_true[order[i]] - y_true[order[j]]).abs() >= min_gap

    # bias toward pairs where at least one is "positive-ish" in log space
    pos_mask = ((y_true[order[i]] > 0) | (y_true[order[j]] > 0))
    keep = torch.where(pos_mask, torch.rand_like(pos_mask.float()) < p_pos, torch.rand_like(pos_mask.float()) < (1 - p_pos))
    mask = gap_ok & keep

    if mask.sum() == 0:
        # fallback: original loss
        return torch.nn.MarginRankingLoss(margin=margin)(a, b, tgt)

    a = a[mask]; b = b[mask]; tgt = tgt[mask]
    return torch.nn.MarginRankingLoss(margin=margin)(a, b, tgt)
def corr_loss(pred, target, eps=1e-8):
    pred_c = pred - pred.mean()
    targ_c = target - target.mean()
    denom = (pred_c.norm() * targ_c.norm()).clamp_min(eps)
    # 1 - Pearson corr (on normalized target space) -> minimize
    return 1.0 - (pred_c * targ_c).sum() / denom
def enhanced_rank_loss_with_ptd_prior(y_pred, y_true, ptd_features, margin=0.5, ptd_weight=0.1):
    """
    Enhanced ranking loss that incorporates PTD information
    """
    # Original ranking loss
    base_loss = loss_cal_rank_weighted(y_pred, y_true, margin=margin)
    
    # PTD-informed ranking consistency loss
    # Encourage predictions to be consistent with PTD ranking order
    ptd_ranks = ptd_features[:, 0]  # Use first feature (percentile)
    
    # Sample pairs for efficiency
    N = len(y_pred)
    num_pairs = min(N * 10, 5000)
    
    indices = torch.combinations(torch.arange(N, device=y_pred.device), 2)
    if len(indices) > num_pairs:
        perm = torch.randperm(len(indices))[:num_pairs]
        indices = indices[perm]
    
    i, j = indices[:, 0], indices[:, 1]
    
    # PTD ordering constraint: if PTD[i] > PTD[j], then pred[i] should > pred[j]
    ptd_order = torch.sign(ptd_ranks[i] - ptd_ranks[j])
    pred_order = torch.sign(y_pred[i] - y_pred[j])
    
    # Penalize when prediction order disagrees with PTD order
    consistency_loss = torch.mean((ptd_order - pred_order) ** 2)
    
    return base_loss + ptd_weight * consistency_loss



#------------------------------------


def improved_ranking_loss(y_pred, y_true, neighbor_counts, sample_num=None,
                         p_pos=0.7, min_gap=0.05, margin=0.5):
    """Your existing ranking loss but weighted by neighbor quality."""
    device = y_pred.device
    N = y_true.shape[0]
    if sample_num is None:
        sample_num = max(100 * N, 10000)
    
    # Your existing ranking loss logic
    order = torch.argsort(-y_true)
    i = torch.randint(0, N, (sample_num,), device=device)
    j = torch.randint(0, N, (sample_num,), device=device)
    a = y_pred[order[i]]
    b = y_pred[order[j]]
    tgt = torch.sign(-(i - j)).float()
    
    gap_ok = (y_true[order[i]] - y_true[order[j]]).abs() >= min_gap
    pos_mask = ((y_true[order[i]] > 0) | (y_true[order[j]] > 0))
    keep = torch.where(pos_mask, torch.rand_like(pos_mask.float()) < p_pos, 
                      torch.rand_like(pos_mask.float()) < (1 - p_pos))
    mask = gap_ok & keep
    
    if mask.sum() == 0:
        base_loss = torch.nn.MarginRankingLoss(margin=margin)(a, b, tgt)
    else:
        a = a[mask]; b = b[mask]; tgt = tgt[mask]
        base_loss = torch.nn.MarginRankingLoss(margin=margin)(a, b, tgt)
    
    # Weight by neighbor quality
    weights = torch.clamp(neighbor_counts.float() / 5.0, min=0.2, max=1.0)
    return (base_loss * weights.mean())




import numpy as np
from typing import Iterable, Tuple, Dict, List, Optional

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _topk_indices(x: np.ndarray, K: int, tie_policy: str = "stable") -> np.ndarray:
    """
    Get top-K indices for 'x' (higher is better).
    tie_policy:
      - "stable": stable sort (deterministic), takes first K if ties span boundary
      - "expand": include ALL indices tied at the Kth score (can return >K)
      - "drop":   drop ties crossing the boundary (strict top-K by value, but if ties
                  exceed K, arbitrarily drop some; uses argpartition + value filter)
    """
    N = x.shape[0]
    K = int(max(1, min(K, N)))
    if tie_policy not in {"stable", "expand", "drop"}:
        raise ValueError("tie_policy must be one of {'stable','expand','drop'}")

    if tie_policy == "stable":
        # stable full sort (O(N log N)), deterministic
        order = np.argsort(-x, kind="mergesort")  # mergesort is stable
        return order[:K]

    # argpartition for O(N) then boundary handling
    part_idx = np.argpartition(-x, K-1)[:K]
    kth_val = x[part_idx].min()  # value at the boundary (since these are top-K unsorted)
    if tie_policy == "drop":
        # Keep exactly K, even if more are tied above kth_val
        return part_idx
    else:  # "expand"
        # Include all elements with value >= kth_val
        mask = x >= kth_val
        idx = np.flatnonzero(mask)
        # For determinism, sort these by value desc, then index asc
        idx = idx[np.lexsort((idx, -x[idx]))]
        return idx

def hits_in_k(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    K: int,
    tie_policy: str = "stable",
    return_indices: bool = False,
) -> Tuple[int, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (hits, %hits[, true_idx, pred_idx]) where:
      hits = |TopK_pred ∩ TopK_true|
      %hits = 100 * hits / K_effective

    Notes:
      - If tie_policy='expand', K_effective can be > K for each side.
        We use denominator = min(len(pred_top), K) by default would be odd,
        so here we normalize by K (the requested K) for comparability.
        If you prefer using len(pred_top) or len(true_top), adjust below.
    """
    yt = _to_numpy(y_true).astype(float).reshape(-1)
    yp = _to_numpy(y_pred).astype(float).reshape(-1)

    # common mask: finite on both
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    N = yt.shape[0]
    if N == 0:
        return 0, 0.0, None, None

    K = int(max(1, min(K, N)))

    true_top = _topk_indices(yt, K, tie_policy=tie_policy)
    pred_top = _topk_indices(yp, K, tie_policy=tie_policy)

    # Intersection size
    hits = int(len(set(true_top.tolist()).intersection(set(pred_top.tolist()))))

    # Normalization choice:
    #  - normalize by requested K (most common & comparable across runs)
    pct = 100.0 * hits / float(K)

    if return_indices:
        return hits, pct, true_top, pred_top
    else:
        return hits, pct, None, None

def hits_in_ks(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    Ks: Iterable[int],
    tie_policy: str = "stable",
    verbose: bool = True,
) -> Dict[int, Tuple[int, float]]:
    """
    Evaluate Hits@K for multiple K values.
    Returns {K: (hits, pct)}.
    """
    results = {}
    for K in Ks:
        hits, pct, _, _ = hits_in_k(y_true, y_pred, int(K), tie_policy=tie_policy, return_indices=False)
        results[int(K)] = (hits, pct)
        if verbose:
            print(f"Hits@{int(K)}: {hits}/{int(K)}  ({pct:.2f}%)")
    return results


def training_model():
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    for epoch in range(NUM_EPOCH):
        epoch_topk_10 = []
        epoch_topk_20 = []
        epoch_topk_1 = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            num_nodes = len(node_list)
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]

            incoming_times = incoming_times_list[j]
            outcoming_times = outcoming_times_list[j]

            num_train_instance = len(node_list)
            num_train_batch = math.ceil(num_train_instance / BATCH_SIZE)
            

            #sept 26
            pass_through_degree = pass_through_d_list[j]
            ptd_all_nodes = pass_through_degree


            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(num_train_instance, s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]

                t_cut = ts_l_cut[0]  # or np.unique(ts_l_cut)[0] if you want to be safe

                #ptd_all_nodes = pass_through_degree_t(incoming_times, outcoming_times, num_nodes,
                #                                 t_cut)  # shape: [num_nodes]

            
                tgnn_model.set_ptd_vector(ptd_all_nodes)  # does: ptd_vec = log1p(scale * raw)

                

                optimizer.zero_grad()

                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    ptd_l=None,  # ignored inside tem_conv2
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )
                
                # Retrieve the valid_counts
                if hasattr(tgnn_model, 'last_valid_counts'):
                    neighbor_valid_counts = tgnn_model.last_valid_counts
                else:
                    # Fallback if not available
                    neighbor_valid_counts = torch.ones(len(src_l_cut), device=device) * 10  # Assume default


                # ----- label prep -----
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)  # log1p(alpha * y)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)  # ensure on device
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t  # z-scored log1p target

                # ----- features / MLP input -----
                idx = torch.as_tensor(src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]  # [B, 2]
                ptd_enc = tgnn_model.ptd_encoder(ptd_pair)  # [B, 128]

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                pred_norm = MLP_model(mlp_in).squeeze(-1)                       # predict z-scored log1p


                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                #rank_loss = loss_cal_rank_weighted(pred_norm, true_label, margin=0.5)
                rank_loss = improved_ranking_loss(pred_norm, true_label, neighbor_valid_counts)
                reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
                c_loss = corr_loss(pred_norm, true_label)


                loss = rank_loss + 0.5 * reg_loss + 0.1 * c_loss # Used for all the testings
                """
                C, A, B = get_adaptive_weights(epoch, NUM_EPOCH)
                loss = C * rank_loss + A * reg_loss + B * c_loss

                with torch.no_grad():
                    loss_components = {
                        'rank': rank_loss.item(),
                        'reg': reg_loss.item(),
                        'corr': c_loss.item(),
                        'weighted_rank': (C * rank_loss).item(),
                        'weighted_reg': (A * reg_loss).item(),
                        'weighted_corr': (B * c_loss).item()
                    }
                    
                    # Log every 100 batches
                    if batch_i % 2000 == 0:
                        print(f"Loss breakdown: Rank={loss_components['rank']:.3f} ({100*loss_components['weighted_rank']/loss:.1f}%), "
                            f"Reg={loss_components['reg']:.3f} ({100*loss_components['weighted_reg']/loss:.1f}%), "
                            f"Corr={loss_components['corr']:.3f} ({100*loss_components['weighted_corr']/loss:.1f}%)")

                """

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(tgnn_model.parameters()) + list(MLP_model.parameters()),
                                               max_norm=1.0)
                optimizer.step()

                m_loss.append(loss.item())




        scheduler.step()
        avg_topk_1 = np.mean(epoch_topk_1)
        avg_topk_10 = np.mean(epoch_topk_10)
        avg_topk_20 = np.mean(epoch_topk_20)
        
        #print(C, A, B)
        print(
            f" Epoch {epoch:02d} Summary : Avg Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | Top@20%: {avg_topk_20:.4f} ")

        epoch_loss = np.mean(m_loss)
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")

import numpy as np
from typing import Iterable, Tuple, Union


def safe_spearman_correlation(x, y, return_pvalue=False):
    """
    Safely compute Spearman correlation, handling constant arrays gracefully
    """
    import numpy as np
    from scipy.stats import spearmanr
    import warnings
    
    # Convert to numpy and ensure float type
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Check for equal lengths
    if len(x) != len(y):
        print(f"WARNING: Array length mismatch: {len(x)} vs {len(y)}")
        return (0.0, 1.0) if return_pvalue else 0.0
    
    # Check for sufficient data points
    if len(x) < 2:
        return (0.0, 1.0) if return_pvalue else 0.0
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if valid_mask.sum() < 2:
        return (0.0, 1.0) if return_pvalue else 0.0
    
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    # Check for constant arrays (std = 0)
    x_std = np.std(x_clean)
    y_std = np.std(y_clean)
    
    if x_std == 0.0 or y_std == 0.0:
        # One or both arrays are constant
        if x_std == 0.0 and y_std == 0.0:
            # Both constant - correlation is undefined, return 1.0 if values match, 0.0 if not
            corr = 1.0 if np.allclose(x_clean, y_clean) else 0.0
        else:
            # One is constant, other varies - correlation is 0
            corr = 0.0
        return (corr, 1.0) if return_pvalue else corr
    
    # Check for near-constant arrays (very small std)
    if x_std < 1e-10 or y_std < 1e-10:
        return (0.0, 1.0) if return_pvalue else 0.0
    
    # Compute correlation with warning suppression
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='An input array is constant')
            
            if return_pvalue:
                corr, p_value = spearmanr(x_clean, y_clean)
                # Handle NaN results
                corr = 0.0 if np.isnan(corr) else float(corr)
                p_value = 1.0 if np.isnan(p_value) else float(p_value)
                return corr, p_value
            else:
                corr, _ = spearmanr(x_clean, y_clean)
                return 0.0 if np.isnan(corr) else float(corr)
                
    except Exception as e:
        print(f"Correlation computation failed: {e}")
        return (0.0, 1.0) if return_pvalue else 0.0
        
class DetailedModelDiagnostics:
    """Enhanced diagnostics to understand model behavior on different dataset types"""
    
    def __init__(self):
        self.training_history = []
        self.evaluation_history = []
        
    def analyze_training_batch(self, epoch, batch_idx, pred_norm, true_label, ptd_raw_batch, 
                              src_embed, ptd_enc, loss_components, ptd_quality):
        """Detailed analysis of training batch"""
        
        batch_analysis = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'ptd_quality': ptd_quality,
            'batch_size': len(pred_norm),
        }
        
        # Prediction quality metrics
        with torch.no_grad():
            pred_np = pred_norm.detach().cpu().numpy()
            true_np = true_label.detach().cpu().numpy()
            ptd_np = ptd_raw_batch.detach().cpu().numpy() if torch.is_tensor(ptd_raw_batch) else ptd_raw_batch
            
            # Correlation analysis
            from scipy.stats import spearmanr
            try:
                model_corr, _ = spearmanr(pred_np, true_np)
                ptd_corr, _ = spearmanr(ptd_np, true_np)
                model_ptd_corr, _ = spearmanr(pred_np, ptd_np)
                
                batch_analysis.update({
                    'model_true_corr': float(model_corr) if not np.isnan(model_corr) else 0.0,
                    'ptd_true_corr': float(ptd_corr) if not np.isnan(ptd_corr) else 0.0,
                    'model_ptd_corr': float(model_ptd_corr) if not np.isnan(model_ptd_corr) else 0.0,
                })
            except:
                batch_analysis.update({
                    'model_true_corr': 0.0, 'ptd_true_corr': 0.0, 'model_ptd_corr': 0.0
                })
            
            # Feature analysis
            batch_analysis.update({
                'src_embed_mean': float(src_embed.mean()),
                'src_embed_std': float(src_embed.std()),
                'ptd_enc_mean': float(ptd_enc.mean()),
                'ptd_enc_std': float(ptd_enc.std()),
                'pred_mean': float(pred_norm.mean()),
                'pred_std': float(pred_norm.std()),
                'true_mean': float(true_label.mean()),
                'true_std': float(true_label.std()),
            })
            
            # Loss component analysis
            batch_analysis.update({
                'total_loss': float(loss_components.get('total_loss', 0)),
                'rank_loss': float(loss_components.get('rank_loss', 0)),
                'reg_loss': float(loss_components.get('reg_loss', 0)),
                'corr_loss': float(loss_components.get('corr_loss', 0)),
            })
        
        self.training_history.append(batch_analysis)
        
        # Print detailed analysis for challenging batches
        if ptd_quality < 0.5 and batch_idx % 200 == 0:
            print(f"\n=== CHALLENGING BATCH ANALYSIS ===")
            print(f"Epoch {epoch}, Batch {batch_idx}, PTD Quality: {ptd_quality:.3f}")
            print(f"Model-True Correlation: {batch_analysis['model_true_corr']:.3f}")
            print(f"PTD-True Correlation: {batch_analysis['ptd_true_corr']:.3f}")
            print(f"Model-PTD Correlation: {batch_analysis['model_ptd_corr']:.3f}")
            print(f"Loss Components - Total: {batch_analysis['total_loss']:.3f}, "
                  f"Rank: {batch_analysis['rank_loss']:.3f}, "
                  f"Reg: {batch_analysis['reg_loss']:.3f}")
            
            # Identify specific issues
            if batch_analysis['model_ptd_corr'] < 0.2:
                print("⚠️  Issue: Model not using PTD information effectively")
            if batch_analysis['model_true_corr'] < batch_analysis['ptd_true_corr'] - 0.1:
                print("⚠️  Issue: Model significantly worse than PTD baseline")
            if batch_analysis['src_embed_std'] < 0.1:
                print("⚠️  Issue: Source embeddings lack diversity")
            if batch_analysis['ptd_enc_std'] < 0.1:
                print("⚠️  Issue: PTD encodings lack diversity")
                
    def analyze_epoch_patterns(self, epoch):
        """Analyze patterns across the epoch"""
        epoch_batches = [b for b in self.training_history if b['epoch'] == epoch]
        if not epoch_batches:
            return
            
        # Group by PTD quality
        high_quality = [b for b in epoch_batches if b['ptd_quality'] > 0.7]
        medium_quality = [b for b in epoch_batches if 0.3 <= b['ptd_quality'] <= 0.7]
        low_quality = [b for b in epoch_batches if b['ptd_quality'] < 0.3]
        
        print(f"\n=== EPOCH {epoch} PATTERN ANALYSIS ===")
        
        for group_name, group_data in [('High', high_quality), ('Medium', medium_quality), ('Low', low_quality)]:
            if not group_data:
                continue
                
            avg_model_corr = np.mean([b['model_true_corr'] for b in group_data])
            avg_ptd_corr = np.mean([b['ptd_true_corr'] for b in group_data])
            avg_model_ptd_corr = np.mean([b['model_ptd_corr'] for b in group_data])
            
            print(f"{group_name} Quality Batches ({len(group_data)} batches):")
            print(f"  Model-True Corr: {avg_model_corr:.3f}")
            print(f"  PTD-True Corr: {avg_ptd_corr:.3f}")
            print(f"  Model-PTD Corr: {avg_model_ptd_corr:.3f}")
            print(f"  Performance Gap: {avg_model_corr - avg_ptd_corr:+.3f}")


class AdaptiveLossForYourModel:
    """Simplified adaptive loss that works with your current setup"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.ptd_quality_cache = {}
        
    def compute_ptd_quality(self, ptd_raw, true_labels):
        """Quick PTD quality assessment"""
        try:
            from scipy.stats import spearmanr
            ptd_np = ptd_raw.detach().cpu().numpy() if torch.is_tensor(ptd_raw) else np.array(ptd_raw)
            true_np = true_labels.detach().cpu().numpy() if torch.is_tensor(true_labels) else np.array(true_labels)
            
            valid_mask = np.isfinite(ptd_np) & np.isfinite(true_np) & (true_np >= 0)
            if valid_mask.sum() < 10:
                return 0.3  # Default medium quality
                
            corr, _ = spearmanr(ptd_np[valid_mask], true_np[valid_mask])
            return max(0.0, min(1.0, float(corr))) if not np.isnan(corr) else 0.3
        except:
            return 0.3
        
    def adaptive_loss(self, pred_norm, true_label, ptd_raw_batch, neighbor_counts=None):
        """
        Adaptive loss that changes strategy based on PTD quality
        """
        # Get PTD quality for this batch
        ptd_quality = self.compute_ptd_quality(ptd_raw_batch, true_label)
        
        # FIX: Handle neighbor_counts properly
        if neighbor_counts is None:
            neighbor_counts = torch.ones_like(pred_norm) * 10  # Default fallback
        
        # Your existing loss components
        rank_loss = improved_ranking_loss(pred_norm, true_label, neighbor_counts)  # Fixed line
        reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
        c_loss = corr_loss(pred_norm, true_label)
        
        # ADAPTIVE WEIGHTS based on PTD quality
        if ptd_quality > 0.8:
            # High-quality PTD: Focus on fine-tuning with ranking
            weights = [1.0, 0.1, 0.05]  # [rank, reg, corr]
        elif ptd_quality > 0.5:
            # Medium-quality PTD: Balanced approach
            weights = [0.7, 0.4, 0.15]
        else:
            # Low-quality PTD: Focus heavily on regression and correlation
            weights = [0.4, 0.8, 0.3]
            
        # For datasets where PTD is weak, add PTD guidance loss
        if ptd_quality < 0.6:
            # Encourage model to at least match PTD ranking
            ptd_tensor = ptd_raw_batch if torch.is_tensor(ptd_raw_batch) else torch.tensor(ptd_raw_batch, device=pred_norm.device, dtype=torch.float32)
            
            # Make sure ptd_tensor has the right shape
            if len(ptd_tensor.shape) == 0:
                ptd_tensor = ptd_tensor.unsqueeze(0)
            if ptd_tensor.shape[0] != pred_norm.shape[0]:
                print(f"WARNING: PTD tensor shape {ptd_tensor.shape} doesn't match pred shape {pred_norm.shape}")
                ptd_guidance_loss = torch.tensor(0.0, device=pred_norm.device)
            else:
                try:
                    pred_ranks = torch.argsort(torch.argsort(pred_norm)).float()
                    ptd_ranks = torch.argsort(torch.argsort(ptd_tensor)).float()
                    ptd_guidance_loss = torch.nn.functional.mse_loss(pred_ranks, ptd_ranks) * 0.2
                except Exception as e:
                    print(f"WARNING: PTD guidance loss failed: {e}")
                    ptd_guidance_loss = torch.tensor(0.0, device=pred_norm.device)
        else:
            ptd_guidance_loss = torch.tensor(0.0, device=pred_norm.device)
        
        total_loss = (weights[0] * rank_loss + 
                     weights[1] * reg_loss + 
                     weights[2] * c_loss + 
                     ptd_guidance_loss)
        
        return total_loss, ptd_quality


class EnhancedAdaptiveLoss:
    """Enhanced version with detailed component tracking"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.diagnostics = DetailedModelDiagnostics()
        
    def compute_ptd_quality(self, ptd_raw, true_labels):
        """Same as before but with more robust error handling"""
        try:
            from scipy.stats import spearmanr
            ptd_np = ptd_raw.detach().cpu().numpy() if torch.is_tensor(ptd_raw) else np.array(ptd_raw)
            true_np = true_labels.detach().cpu().numpy() if torch.is_tensor(true_labels) else np.array(true_labels)
            
            valid_mask = np.isfinite(ptd_np) & np.isfinite(true_np) & (true_np >= 0)
            if valid_mask.sum() < 10:
                return 0.3
                
            corr, _ = spearmanr(ptd_np[valid_mask], true_np[valid_mask])
            return max(0.0, min(1.0, float(corr))) if not np.isnan(corr) else 0.3
        except:
            return 0.3
    
    def adaptive_loss_with_diagnostics(self, pred_norm, true_label, ptd_raw_batch, neighbor_counts, 
                                     src_embed, ptd_enc, epoch, batch_idx):
        """Enhanced adaptive loss with detailed diagnostics"""
        
        ptd_quality = self.compute_ptd_quality(ptd_raw_batch, true_label)
        
        if neighbor_counts is None:
            neighbor_counts = torch.ones_like(pred_norm) * 10
        
        # Compute loss components individually for analysis
        rank_loss = improved_ranking_loss(pred_norm, true_label, neighbor_counts)
        reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
        c_loss = corr_loss(pred_norm, true_label)
        
        # Enhanced adaptive weighting based on analysis
        if ptd_quality > 0.8:
            weights = [1.0, 0.1, 0.05]
        elif ptd_quality > 0.5:
            weights = [0.7, 0.4, 0.15]
        else:
            # For very challenging datasets, focus more on learning from PTD
            weights = [0.3, 0.6, 0.4]  # Less ranking, more regression/correlation
        
        # Enhanced PTD guidance for challenging cases
        ptd_guidance_loss = torch.tensor(0.0, device=pred_norm.device)
        if ptd_quality < 0.6:
            ptd_tensor = ptd_raw_batch if torch.is_tensor(ptd_raw_batch) else torch.tensor(ptd_raw_batch, device=pred_norm.device, dtype=torch.float32)
            
            if ptd_tensor.shape[0] == pred_norm.shape[0]:
                try:
                    # Stronger PTD guidance for very challenging datasets
                    guidance_weight = 0.5 if ptd_quality < 0.4 else 0.2
                    
                    # Multiple PTD guidance strategies
                    pred_ranks = torch.argsort(torch.argsort(pred_norm)).float()
                    ptd_ranks = torch.argsort(torch.argsort(ptd_tensor)).float()
                    ranking_guidance = torch.nn.functional.mse_loss(pred_ranks, ptd_ranks)
                    
                    # Direct value guidance (encourage model to match PTD magnitude)
                    value_guidance = torch.nn.functional.l1_loss(
                        F.normalize(pred_norm, dim=0), 
                        F.normalize(ptd_tensor, dim=0)
                    )
                    
                    ptd_guidance_loss = guidance_weight * (ranking_guidance + 0.3 * value_guidance)
                    
                except Exception as e:
                    ptd_guidance_loss = torch.tensor(0.0, device=pred_norm.device)
        
        total_loss = (weights[0] * rank_loss + 
                     weights[1] * reg_loss + 
                     weights[2] * c_loss + 
                     ptd_guidance_loss)
        
        # Store detailed diagnostics
        loss_components = {
            'total_loss': total_loss,
            'rank_loss': rank_loss,
            'reg_loss': reg_loss,
            'corr_loss': c_loss,
            'ptd_guidance_loss': ptd_guidance_loss,
            'weights': weights,
            'ptd_quality': ptd_quality
        }
        
        # Detailed analysis for debugging
        self.diagnostics.analyze_training_batch(
            epoch, batch_idx, pred_norm, true_label, ptd_raw_batch,
            src_embed, ptd_enc, loss_components, ptd_quality
        )
        
        return total_loss, ptd_quality, loss_components


def compute_batch_importance_weight(ptd_quality, true_labels):
    """Compute importance weight based on node centrality and PTD quality"""
    
    # Nodes with higher true centrality are more important
    centrality_weight = torch.log1p(true_labels.mean() + 1e-6)
    
    # Higher PTD quality indicates more learnable examples  
    quality_weight = max(0.1, ptd_quality)  # Minimum weight of 0.1
    
    # Combined importance score
    importance = centrality_weight * quality_weight
    
    return float(importance.clamp(0.1, 2.0))  # Weight between 0.1x and 2.0x



def training_model_26sept_claude_agnostic():
    # Initialize adaptive loss
    adaptive_loss_fn = EnhancedAdaptiveLoss(device=device)
    
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    for epoch in range(NUM_EPOCH):
        epoch_stats = {'high_quality': 0, 'medium_quality': 0, 'low_quality': 0}
        
        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []
        epoch_ptd_qualities = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)
            
            ptd_idx = ptd_indices[j]
            ptd_cache = ptd_caches[j]

            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)
                
                optimizer.zero_grad()

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )
                


                # Get neighbor counts
                neighbor_valid_counts = getattr(tgnn_model, 'last_valid_counts', None)
                if neighbor_valid_counts is None:
                    neighbor_valid_counts = torch.ones(len(src_l_cut), device=device) * 10

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_encoder(ptd_pair)
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
                pred_norm = MLP_model(mlp_in).squeeze(-1)


                # FIX: Get raw PTD values for this batch correctly
                try:
                    # Get raw PTD values for the batch nodes
                    src_l_cut_clamped = np.clip(src_l_cut, 1, len(ptd_past_1b) - 1)  # Ensure valid indices
                    ptd_raw_batch = ptd_past_1b[src_l_cut_clamped]  # Get raw PTD for these nodes
                    ptd_raw_batch = np.array(ptd_raw_batch, dtype=np.float32)  # Ensure it's numpy array
                    
                    # Convert to tensor for consistency
                    ptd_raw_tensor = torch.tensor(ptd_raw_batch, device=device, dtype=torch.float32)
                    
                except Exception as e:
                    print(f"Error getting PTD raw batch: {e}")
                    # Fallback to zeros
                    ptd_raw_tensor = torch.zeros_like(pred_norm)

                # Enhanced loss computation with diagnostics
                loss, ptd_quality, loss_components = \
                    adaptive_loss_fn.adaptive_loss_with_diagnostics(
                    pred_norm, true_label, ptd_raw_tensor, neighbor_valid_counts,
                    src_embed, ptd_enc, epoch, batch_i
                )


                # ADAPTIVE LOSS - This is the key change
                #loss, ptd_quality = adaptive_loss_fn.adaptive_loss(
                #    pred_norm, true_label, ptd_raw_tensor, neighbor_valid_counts
                #)
                
                if hasattr(tgnn_model, 'last_ptd_qualities') and tgnn_model.last_ptd_qualities is not None:
                    avg_predicted_quality = tgnn_model.last_ptd_qualities.mean().item()
                else:
                    avg_predicted_quality = 0.5  # fallback
                
                # Categorize batch
                if avg_predicted_quality > 0.7:
                    epoch_stats['high_quality'] += 1
                elif avg_predicted_quality > 0.4:
                    epoch_stats['medium_quality'] += 1
                else:
                    epoch_stats['low_quality'] += 1

                epoch_ptd_qualities.append(ptd_quality)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())

                # Periodic logging with PTD quality
                #if batch_i % 100 == 0:
                #    print(f"Epoch {epoch}, Graph {j}, Batch {batch_i}: Loss={loss.item():.4f}, PTD Quality={ptd_quality:.3f}")




        adaptive_loss_fn.diagnostics.analyze_epoch_patterns(epoch)
        scheduler.step()
        
        total_batches = sum(epoch_stats.values())
        print(f"Epoch {epoch} PTD Quality Distribution:")
        print(f"  High Quality: {epoch_stats['high_quality']}/{total_batches} "
              f"({epoch_stats['high_quality']/total_batches*100:.1f}%)")
        print(f"  Medium Quality: {epoch_stats['medium_quality']}/{total_batches} "
              f"({epoch_stats['medium_quality']/total_batches*100:.1f}%)")
        print(f"  Low Quality: {epoch_stats['low_quality']}/{total_batches} "
              f"({epoch_stats['low_quality']/total_batches*100:.1f}%)")


        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0
        avg_ptd_quality = np.mean(epoch_ptd_qualities) if epoch_ptd_qualities else 0
        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f} | Avg PTD Quality: {avg_ptd_quality:.3f}")
        
        epoch_loss = np.mean(m_loss) if m_loss else float('inf')
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")


def training_model_26sept_claude():
    # Initialize adaptive loss
    adaptive_loss_fn = EnhancedAdaptiveLoss(device=device)
    
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    for epoch in range(NUM_EPOCH):
        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []
        epoch_ptd_qualities = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)
            
            ptd_idx = ptd_indices[j]
            ptd_cache = ptd_caches[j]

            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)
                
                optimizer.zero_grad()

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )
                


                # Get neighbor counts
                neighbor_valid_counts = getattr(tgnn_model, 'last_valid_counts', None)
                if neighbor_valid_counts is None:
                    neighbor_valid_counts = torch.ones(len(src_l_cut), device=device) * 10

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_encoder(ptd_pair)
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
                pred_norm = MLP_model(mlp_in).squeeze(-1)


                # FIX: Get raw PTD values for this batch correctly
                try:
                    # Get raw PTD values for the batch nodes
                    src_l_cut_clamped = np.clip(src_l_cut, 1, len(ptd_past_1b) - 1)  # Ensure valid indices
                    ptd_raw_batch = ptd_past_1b[src_l_cut_clamped]  # Get raw PTD for these nodes
                    ptd_raw_batch = np.array(ptd_raw_batch, dtype=np.float32)  # Ensure it's numpy array
                    
                    # Convert to tensor for consistency
                    ptd_raw_tensor = torch.tensor(ptd_raw_batch, device=device, dtype=torch.float32)
                    
                except Exception as e:
                    print(f"Error getting PTD raw batch: {e}")
                    # Fallback to zeros
                    ptd_raw_tensor = torch.zeros_like(pred_norm)

                # Enhanced loss computation with diagnostics
                loss, ptd_quality, loss_components = \
                    adaptive_loss_fn.adaptive_loss_with_diagnostics(
                    pred_norm, true_label, ptd_raw_tensor, neighbor_valid_counts,
                    src_embed, ptd_enc, epoch, batch_i
                )


                epoch_ptd_qualities.append(ptd_quality)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())

                # Periodic logging with PTD quality
                #if batch_i % 100 == 0:
                #    print(f"Epoch {epoch}, Graph {j}, Batch {batch_i}: Loss={loss.item():.4f}, PTD Quality={ptd_quality:.3f}")


        adaptive_loss_fn.diagnostics.analyze_epoch_patterns(epoch)
        scheduler.step()
        
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0
        avg_ptd_quality = np.mean(epoch_ptd_qualities) if epoch_ptd_qualities else 0
        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f} | Avg PTD Quality: {avg_ptd_quality:.3f}")
        
        epoch_loss = np.mean(m_loss) if m_loss else float('inf')
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")


def eval_real_data_sept26_improved(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    tgan.ngh_finder = sampler
    all_pred_raw = []
    all_true_raw = []
    all_ptd_raw = []
    
    # Track evaluation metrics
    batch_ptd_qualities = []
    
    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        
        print(f"Evaluating {hint}: {num_test_instance} instances in {num_test_batch} batches")
        
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])
            
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            test_ts = np.array(ts[s_idx:e_idx], dtype=float)
            t_cut = float(test_ts[0])
            
            # FIX: Syntax error in your original code
            ptd_past_1b = test_ptd_cache.get(t_cut)
            if ptd_past_1b is None:
                ptd_past_1b, _, _ = test_ptd_index.snapshot_partition_all(t_cut)  # Fixed: use _, _ instead of *, *
                test_ptd_cache[t_cut] = ptd_past_1b
            
            # Set PTD vector (simplified 2D version)
            tgan.set_ptd_vector(ptd_past_1b)
            
            # Safe 0-based indices
            idx0 = np.clip(test_src, 1, test_num_nodes) - 1
            
            # Labels aligned to idx0
            true_batch = np.asarray(label, dtype=float)[idx0]
            
            # Forward pass
            try:
                src_embed = tgan.tem_conv2(
                    src_idx_l=test_src_l_cut,
                    cut_time_l=test_ts_l_cut,
                    ptd_l=None,
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )
                
                # Check for NaN in embeddings
                if torch.isnan(src_embed).any():
                    print(f"WARNING: NaN in src_embed for batch {k}")
                    # Skip this batch or use fallback
                    continue
                
                # PTD features for MLP (2D version)
                idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
                ptd_pair = tgan.ptd_vec[idx]  # [B, 2]
                ptd_enc = tgan.ptd_encoder(ptd_pair)  # [B, 64]
                
                # Check PTD encoding
                if torch.isnan(ptd_enc).any():
                    print(f"WARNING: NaN in PTD encoding for batch {k}")
                    ptd_enc = torch.zeros_like(ptd_enc)
                
                # MLP prediction
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
                pred_norm = lr_model(mlp_in).squeeze(-1)
                
                # Check final prediction
                if torch.isnan(pred_norm).any():
                    print(f"WARNING: NaN in predictions for batch {k}")
                    continue
                
            except Exception as e:
                print(f"ERROR in forward pass for batch {k}: {e}")
                continue
            
            # Optional: Compute PTD quality for this batch (diagnostic)
            try:
                from scipy.stats import spearmanr
                ptd_raw_batch = ptd_past_1b[test_src_l_cut.clip(1, len(ptd_past_1b)-1)]
                if len(ptd_raw_batch) > 3 and len(true_batch) > 3:
                    corr, _ = spearmanr(ptd_raw_batch, true_batch)
                    if not np.isnan(corr):
                        batch_ptd_qualities.append(float(corr))
            except:
                pass  # Skip if computation fails
            
            # Accumulate results
            all_pred_raw.append(pred_norm.detach().clamp_min(0).cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))
            
            # Progress reporting
            if k % 10 == 0:
                print(f"Processed batch {k}/{num_test_batch}")
    
    # Check if we have any results
    if not all_pred_raw:
        print("ERROR: No valid predictions generated!")
        return float('inf')
    
    # Combine all results
    pred = np.concatenate(all_pred_raw, axis=0)
    true = np.concatenate(all_true_raw, axis=0)
    
    # Print diagnostic info
    print(f"Evaluation Results for {hint}:")
    print(f"  Final prediction shape: {pred.shape}")
    print(f"  Final true shape: {true.shape}")
    print(f"  Prediction stats: mean={pred.mean():.4f}, std={pred.std():.4f}, min={pred.min():.4f}, max={pred.max():.4f}")
    print(f"  True stats: mean={true.mean():.4f}, std={true.std():.4f}, min={true.min():.4f}, max={true.max():.4f}")
    
    if batch_ptd_qualities:
        avg_ptd_quality = np.mean(batch_ptd_qualities)
        print(f"  Average batch PTD quality: {avg_ptd_quality:.3f}")
        
        # Categorize dataset difficulty
        if avg_ptd_quality > 0.8:
            difficulty = "HIGH-QUALITY (limited improvement expected)"
        elif avg_ptd_quality > 0.5:
            difficulty = "MODERATE-QUALITY (good improvement opportunity)"
        else:
            difficulty = "CHALLENGING (major improvement opportunity)"
        print(f"  Dataset difficulty: {difficulty}")
    
    # Compute hits@K metrics
    results = hits_in_ks(true, pred, Ks=[5, 10, 20])
    results_ptd = hits_in_ks(true, test_pass_through_d, Ks=[5, 10, 20])
    
    print("\nModel Performance:")
    for K, (hits, pct) in results.items():
        print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")
    
    print("\nPTD Baseline Performance:")
    for K, (hits, pct) in results_ptd.items():
        print(f"Hits PTD@{K}: {hits}/{K} ({pct:.2f}%)")
    
    # Enhanced evaluation with comparison
    print(f"\n=== DETAILED COMPARISON: {hint} ===")
    unified_evaluation_with_statistics2(pred, true, test_pass_through_d, hint)
    
    end_time = time.time()
    e_time = (end_time - start_time) / 60.0
    
    print(f"Evaluation completed in {e_time:.2f} minutes")
    
    return e_time


def eval_real_data_sept26_claude(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_list = []
    tgan.ngh_finder = sampler

    all_pred_raw = []
    all_true_raw = []
    all_ptd_raw  = []

    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        ptd_t_all_nodes = test_pass_through_d
        
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            test_t_cut = test_ts_l_cut[0]

            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            test_ts  = np.array(ts[s_idx:e_idx],  dtype=float)

            t_cut = float(test_ts[0])

            # Get PTD data (same as before)
            ptd_past_1b = test_ptd_cache.get(t_cut)
            if ptd_past_1b is None:
                ptd_past_1b, _, _ = test_ptd_index.snapshot_partition_all(t_cut)
                test_ptd_cache[t_cut] = ptd_past_1b

            # CHANGE: Use simplified PTD vector setting
            tgan.set_ptd_vector(ptd_past_1b)  # This now creates [N, 2] instead of [N, 4]

            # Safe 0-based indices (same)
            idx0  = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            # Labels aligned to idx0 (same)
            true_batch = np.asarray(label, dtype=float)[idx0]

            # Forward pass (EXACTLY the same)
            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )

            # CHANGE: PTD features for MLP (now 2D instead of 4D)
            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]  # [B, 2] instead of [B, 4]
            ptd_enc = tgan.ptd_encoder(ptd_pair)  # [B, 64] instead of [B, 128]

            mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
            pred_norm = lr_model(mlp_in).squeeze(-1)

            # Accumulate results (same)
            all_pred_raw.append(pred_norm.detach().clamp_min(0).cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))

        # Final evaluation (same as before)
        pred = np.concatenate(all_pred_raw, axis=0)
        true = np.concatenate(all_true_raw, axis=0)

        results = hits_in_ks(true, pred, Ks=[5, 10, 20])
        results_ptd = hits_in_ks(true, ptd_t_all_nodes, Ks=[ 5, 10, 20])

        for K, (hits, pct) in results.items():
            print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")
        for K, (hits, pct) in results_ptd.items():
            print(f"Hits PTD@{K}: {hits}/{K} ({pct:.2f}%)")

        unified_evaluation_with_statistics2(pred, true, ptd_t_all_nodes, hint)
        
        end_time = time.time()
        e_time = (end_time - start_time) / 60.0

        return e_time


def training_model_26sept():
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    for epoch in range(NUM_EPOCH):
        epoch_topk_10 = []
        epoch_topk_20 = []
        epoch_topk_1 = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            num_nodes = len(node_list)
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]

            num_train_instance = len(node_list)
            num_train_batch = math.ceil(num_train_instance / BATCH_SIZE)
            

            ptd_idx   = ptd_indices[j]      # PTDIndex1B
            ptd_cache = ptd_caches[j]       # dict[t_cut] -> ndarray [N+1, feat_dim]
            num_nodes = int(max(node_list)) # or the stored num_nodes_j


            #sept 26
            #pass_through_degree = pass_through_d_list[j]
            #ptd_all_nodes = pass_through_degree

            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(num_train_instance, s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]

                t_cut = ts_l_cut[0]  # or np.unique(ts_l_cut)[0] if you want to be safe


                feats_all_1b = ptd_cache.get(t_cut)
                if feats_all_1b is None:
                    ptd_past_1b, _, ptd_total_1b = ptd_idx.snapshot_partition_all(t_cut)
                    feats_all_1b = np.stack([ptd_past_1b, ptd_total_1b], axis=1).astype(np.float32)  # [N+1, 2]
                    ptd_cache[t_cut] = feats_all_1b

                # MODEL IS 0-BASED: drop the dummy row 0
                ptd_past_1b, _, ptd_total_1b = ptd_indices[j].snapshot_partition_all(t_cut)  # each is [max_id+1]


                feats_all_0b = feats_all_1b[1:]   # shape [N, 2]

                #tgnn_model.set_ptd_vector(feats_all_0b)  # does: ptd_vec = log1p(scale * raw)
                tgnn_model.set_ptd_vector(ptd_past_1b)  # 1-based vector; method drops row 0 and builds [N,4]

                
                optimizer.zero_grad()

                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    ptd_l=None,  # ignored inside tem_conv2
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )
                
                # Retrieve the valid_counts
                if hasattr(tgnn_model, 'last_valid_counts'):
                    neighbor_valid_counts = tgnn_model.last_valid_counts
                else:
                    # Fallback if not available
                    neighbor_valid_counts = torch.ones(len(src_l_cut), device=device) * 10  # Assume default


                # ----- label prep -----
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)  # log1p(alpha * y)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)  # ensure on device
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t  # z-scored log1p target

                # ----- features / MLP input -----
                idx = torch.as_tensor(src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]  # [B, 4]
                ptd_enc = tgnn_model.ptd_encoder(ptd_pair)  # [B, 128]

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                pred_norm = MLP_model(mlp_in).squeeze(-1)                       # predict z-scored log1p


                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                #rank_loss = loss_cal_rank_weighted(pred_norm, true_label, margin=0.5)
                rank_loss = improved_ranking_loss(pred_norm, true_label, neighbor_valid_counts)
                reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
                c_loss = corr_loss(pred_norm, true_label)

                loss = rank_loss + 0.5 * reg_loss + 0.1 * c_loss # Used for all the testings


                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(tgnn_model.parameters()) + list(MLP_model.parameters()),
                                               max_norm=1.0)
                optimizer.step()

                m_loss.append(loss.item())




        scheduler.step()
        avg_topk_1 = np.mean(epoch_topk_1)
        avg_topk_10 = np.mean(epoch_topk_10)
        avg_topk_20 = np.mean(epoch_topk_20)
        
        #print(C, A, B)
        print(
            f" Epoch {epoch:02d} Summary : Avg Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | Top@20%: {avg_topk_20:.4f} ")

        epoch_loss = np.mean(m_loss)
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")

def eval_real_data(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_list = []
    tgan.ngh_finder = sampler

    all_pred_raw = []
    all_true_raw = []
    all_ptd_raw  = []


    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)



        ptd_idx = PTDIndex(test_temporal_edges, test_num_nodes)
        ptd_all_t = np.zeros(num_test_instance, dtype=np.int64)


        ptd_defualt_all_nodes = test_pass_through_d
        ptd_all_nodes = ptd_defualt_all_nodes
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            test_t_cut = test_ts_l_cut[0]
            #ptd_all_nodes = compute_ptd_at_t(test_incoming_times, test_outcoming_times, test_num_nodes,
            #                                 test_t_cut)  # shape: [num_nodes]
            
            
            #ptd_all_nodes = pass_through_degree_t(test_temporal_edges, test_num_nodes, test_t_cut)
            batch_nodes = np.array(nodeList_test_real[s_idx:e_idx], dtype=np.int64)
            batch_t = np.array(test_real_ts_list[s_idx:e_idx], dtype=np.float64)
            #ptd_past, ptd_future, ptd_all_nodes = ptd_idx.query_partition(batch_nodes, batch_t)

            #ptd_all_t[s_idx:e_idx] = ptd_past                     
            #print("ptd_past: ", ptd_past)
            #print("ptd_futures: ", ptd_future)
            #print("ptd_total: ", ptd_all_nodes)

            

            #stats = ptd_snapshot_report(ptd_all_nodes, label)  # label must be the same snapshot labels
            #spear = float(stats['spearman'])
            #if hasattr(tgan, 'set_ptd_gates_soft'):
            #    gate = tgan.set_ptd_gates_soft(spear, low=0.20, mid=0.45, pct_floor=0.30)
            #    print(f"[PTD gate] {gate}")
            #LOW, HIGH = 0.20, 0.60  # tune if you like
            #alpha = 0.0 if spear <= LOW else (1.0 if spear >= HIGH else (spear - LOW) / (HIGH - LOW))

            # 1-based node ids and cut times
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            test_ts  = np.array(ts[s_idx:e_idx],  dtype=float)

            # ----- PTD snapshot at the first time in the batch -----
            t_cut = float(test_ts[0])

            #26 sept
            #ptd_all_nodes = compute_ptd_at_t(test_incoming_times, test_outcoming_times, test_num_nodes, t_cut)
            #ptd_all_nodes = pass_through_degree_t(test_incoming_times, test_outcoming_times, test_num_nodes, t_cut)
            


            # Write 2-D PTD feature table [log1p(scale*raw), percentile]
            tgan.set_ptd_vector(ptd_all_nodes)

            # Safe 0-based indices
            idx0  = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            # Labels aligned to idx0
            true_batch = np.asarray(label, dtype=float)[idx0]

            #-----

            #tgan.set_ptd_vector(ptd_all_nodes)

            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,  # [B, 1]
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )
            # Retrieve the valid_counts
            if hasattr(tgnn_model, 'last_valid_counts'):
                neighbor_valid_counts = tgnn_model.last_valid_counts
            else:
                # Fallback if not available
                neighbor_valid_counts = torch.ones(len(test_src_l_cut), device=device) * 10  # Assume default


            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]  # [B, 2]

            ptd_raw_batch = torch.as_tensor(
                ptd_all_nodes[idx0], device=device, dtype=torch.float32
            ).clamp_min(0)
        

            ptd_enc = tgan.ptd_encoder(ptd_pair)
            ptd_enc_scaled = tgan.ptd_encoder(ptd_pair)
            #ptd_pair_scaled = ptd_pair * alpha
            #ptd_enc_scaled = ptd_enc * alpha


            mlp_in = torch.cat([src_embed, ptd_enc_scaled], dim=1)

            pred_norm = lr_model(mlp_in).squeeze(-1)  # normalized prediction
            pred_log1p = pred_norm * sigma + mu
            pred_raw = normalizer.torch_inverse(pred_log1p)
            test_pred = pred_norm

            # Accumulate (flatten to 1-D)
            all_pred_raw.append(pred_raw.detach().clamp_min(0).cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))
            all_ptd_raw.append(ptd_raw_batch.detach().cpu().numpy().reshape(-1))

        #print(ptd__defualt_all_nodes.shape())
        #wkt_ptd, _ = weightedtau(ptd__defualt_all_nodes, ptd_all_t)

        #print("W PTD kendal tau :" , wkt_ptd)

        # ===== whole-graph arrays =====
        pred = np.concatenate(all_pred_raw, axis=0)
        true = np.concatenate(all_true_raw, axis=0)
        ptd  = np.concatenate(all_ptd_raw,  axis=0)

        results = hits_in_ks(true, pred, Ks=[5, 10, 20])

        results_ptd = hits_in_ks(true,ptd, Ks=[ 5, 10, 20])

        # Access programmatically
        for K, (hits, pct) in results.items():
            print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")
        for K, (hits, pct) in results_ptd.items():
            print(f"Hits PTD@{K}: {hits}/{K} ({pct:.2f}%)")



        unified_evaluation_with_statistics2(pred, true, ptd, hint)
        
        end_time = time.time()
        e_time = (end_time - start_time) / 60.0

        return e_time


    end_time = time.time()
    e_time = (end_time - start_time) / 60.0

    return e_time



def eval_real_data_sept26(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_list = []
    tgan.ngh_finder = sampler

    all_pred_raw = []
    all_true_raw = []
    all_ptd_raw  = []


    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)



        ptd_defualt_all_nodes = test_pass_through_d
        ptd_all_nodes = ptd_defualt_all_nodes
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            test_t_cut = test_ts_l_cut[0]

            
            #ptd_all_nodes = pass_through_degree_t(test_temporal_edges, test_num_nodes, test_t_cut)
            batch_nodes = np.array(nodeList_test_real[s_idx:e_idx], dtype=np.int64)
            batch_t = np.array(test_real_ts_list[s_idx:e_idx], dtype=np.float64)

            #ptd_all_t[s_idx:e_idx] = ptd_past                     
            #print("ptd_past: ", ptd_past)
            #print("ptd_futures: ", ptd_future)
            #print("ptd_total: ", ptd_all_nodes)

            

            # 1-based node ids and cut times
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            test_ts  = np.array(ts[s_idx:e_idx],  dtype=float)

            # ----- PTD snapshot at the first time in the batch -----
            t_cut = float(test_ts[0])


            ptd_past_1b = test_ptd_cache.get(t_cut)
            if ptd_past_1b is None:
                ptd_past_1b, _, _ = test_ptd_index.snapshot_partition_all(t_cut)  # [max_id+1], 1-based
                test_ptd_cache[t_cut] = ptd_past_1b


            # Write 2-D PTD feature table [log1p(scale*raw), percentile]
            tgan.set_ptd_vector(ptd_past_1b)

            # Safe 0-based indices
            idx0  = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            # Labels aligned to idx0
            true_batch = np.asarray(label, dtype=float)[idx0]

            #-----

            #tgan.set_ptd_vector(ptd_all_nodes)

            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,  # [B, 1]
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )
            # Retrieve the valid_counts
            if hasattr(tgnn_model, 'last_valid_counts'):
                neighbor_valid_counts = tgnn_model.last_valid_counts
            else:
                # Fallback if not available
                neighbor_valid_counts = torch.ones(len(test_src_l_cut), device=device) * 10  # Assume default


            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]  # [B, 2]

            #ptd_raw_batch = torch.as_tensor(
            #    ptd_all_nodes[idx0], device=device, dtype=torch.float32
            #).clamp_min(0)
        

            ptd_enc = tgan.ptd_encoder(ptd_pair)
            ptd_enc_scaled = tgan.ptd_encoder(ptd_pair)
            #ptd_pair_scaled = ptd_pair * alpha
            #ptd_enc_scaled = ptd_enc * alpha


            mlp_in = torch.cat([src_embed, ptd_enc_scaled], dim=1)

            pred_norm = lr_model(mlp_in).squeeze(-1)  # normalized prediction
            #pred_log1p = pred_norm * sigma + mu
            #pred_raw = normalizer.torch_inverse(pred_log1p)
            #test_pred = pred_norm

            # Accumulate (flatten to 1-D)
            all_pred_raw.append(pred_norm.detach().clamp_min(0).cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))
            #all_ptd_raw.append(ptd_raw_batch.detach().cpu().numpy().reshape(-1))

        #print(ptd__defualt_all_nodes.shape())
        #wkt_ptd, _ = weightedtau(ptd__defualt_all_nodes, ptd_all_t)

        #print("W PTD kendal tau :" , wkt_ptd)

        # ===== whole-graph arrays =====
        pred = np.concatenate(all_pred_raw, axis=0)
        true = np.concatenate(all_true_raw, axis=0)
        #ptd  = np.concatenate(test_pass_through_d,  axis=0)

        results = hits_in_ks(true, pred, Ks=[5, 10, 20])

        results_ptd = hits_in_ks(true,test_pass_through_d, Ks=[ 5, 10, 20])

        # Access programmatically
        for K, (hits, pct) in results.items():
            print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")
        for K, (hits, pct) in results_ptd.items():
            print(f"Hits PTD@{K}: {hits}/{K} ({pct:.2f}%)")



        unified_evaluation_with_statistics2(pred, true, test_pass_through_d, hint)
        
        end_time = time.time()
        e_time = (end_time - start_time) / 60.0

        return e_time


    end_time = time.time()
    e_time = (end_time - start_time) / 60.0

    return e_time


def eval_real_data(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_list = []
    tgan.ngh_finder = sampler

    all_pred_raw = []
    all_true_raw = []
    all_ptd_raw  = []


    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)



        ptd_idx = PTDIndex(test_temporal_edges, test_num_nodes)
        ptd_all_t = np.zeros(num_test_instance, dtype=np.int64)


        ptd_defualt_all_nodes = test_pass_through_d
        ptd_all_nodes = ptd_defualt_all_nodes
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            test_t_cut = test_ts_l_cut[0]
            #ptd_all_nodes = compute_ptd_at_t(test_incoming_times, test_outcoming_times, test_num_nodes,
            #                                 test_t_cut)  # shape: [num_nodes]
            
            
            #ptd_all_nodes = pass_through_degree_t(test_temporal_edges, test_num_nodes, test_t_cut)
            batch_nodes = np.array(nodeList_test_real[s_idx:e_idx], dtype=np.int64)
            batch_t = np.array(test_real_ts_list[s_idx:e_idx], dtype=np.float64)
            #ptd_past, ptd_future, ptd_all_nodes = ptd_idx.query_partition(batch_nodes, batch_t)

            #ptd_all_t[s_idx:e_idx] = ptd_past                     
            #print("ptd_past: ", ptd_past)
            #print("ptd_futures: ", ptd_future)
            #print("ptd_total: ", ptd_all_nodes)

            

            #stats = ptd_snapshot_report(ptd_all_nodes, label)  # label must be the same snapshot labels
            #spear = float(stats['spearman'])
            #if hasattr(tgan, 'set_ptd_gates_soft'):
            #    gate = tgan.set_ptd_gates_soft(spear, low=0.20, mid=0.45, pct_floor=0.30)
            #    print(f"[PTD gate] {gate}")
            #LOW, HIGH = 0.20, 0.60  # tune if you like
            #alpha = 0.0 if spear <= LOW else (1.0 if spear >= HIGH else (spear - LOW) / (HIGH - LOW))

            # 1-based node ids and cut times
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            test_ts  = np.array(ts[s_idx:e_idx],  dtype=float)

            # ----- PTD snapshot at the first time in the batch -----
            t_cut = float(test_ts[0])

            #26 sept
            #ptd_all_nodes = compute_ptd_at_t(test_incoming_times, test_outcoming_times, test_num_nodes, t_cut)
            #ptd_all_nodes = pass_through_degree_t(test_incoming_times, test_outcoming_times, test_num_nodes, t_cut)
            


            # Write 2-D PTD feature table [log1p(scale*raw), percentile]
            tgan.set_ptd_vector(ptd_all_nodes)

            # Safe 0-based indices
            idx0  = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            # Labels aligned to idx0
            true_batch = np.asarray(label, dtype=float)[idx0]

            #-----

            #tgan.set_ptd_vector(ptd_all_nodes)

            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,  # [B, 1]
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )
            # Retrieve the valid_counts
            if hasattr(tgnn_model, 'last_valid_counts'):
                neighbor_valid_counts = tgnn_model.last_valid_counts
            else:
                # Fallback if not available
                neighbor_valid_counts = torch.ones(len(test_src_l_cut), device=device) * 10  # Assume default


            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]  # [B, 2]

            ptd_raw_batch = torch.as_tensor(
                ptd_all_nodes[idx0], device=device, dtype=torch.float32
            ).clamp_min(0)
        

            ptd_enc = tgan.ptd_encoder(ptd_pair)
            ptd_enc_scaled = tgan.ptd_encoder(ptd_pair)
            #ptd_pair_scaled = ptd_pair * alpha
            #ptd_enc_scaled = ptd_enc * alpha


            mlp_in = torch.cat([src_embed, ptd_enc_scaled], dim=1)

            pred_norm = lr_model(mlp_in).squeeze(-1)  # normalized prediction
            pred_log1p = pred_norm * sigma + mu
            pred_raw = normalizer.torch_inverse(pred_log1p)
            test_pred = pred_norm

            # Accumulate (flatten to 1-D)
            all_pred_raw.append(pred_raw.detach().clamp_min(0).cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))
            all_ptd_raw.append(ptd_raw_batch.detach().cpu().numpy().reshape(-1))

        #print(ptd__defualt_all_nodes.shape())
        #wkt_ptd, _ = weightedtau(ptd__defualt_all_nodes, ptd_all_t)

        #print("W PTD kendal tau :" , wkt_ptd)

        # ===== whole-graph arrays =====
        pred = np.concatenate(all_pred_raw, axis=0)
        true = np.concatenate(all_true_raw, axis=0)
        ptd  = np.concatenate(all_ptd_raw,  axis=0)

        results = hits_in_ks(true, pred, Ks=[5, 10, 20])

        results_ptd = hits_in_ks(true,ptd, Ks=[ 5, 10, 20])

        # Access programmatically
        for K, (hits, pct) in results.items():
            print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")
        for K, (hits, pct) in results_ptd.items():
            print(f"Hits PTD@{K}: {hits}/{K} ({pct:.2f}%)")



        unified_evaluation_with_statistics2(pred, true, ptd, hint)
        
        end_time = time.time()
        e_time = (end_time - start_time) / 60.0

        return e_time


    end_time = time.time()
    e_time = (end_time - start_time) / 60.0

    return e_time


def training_model_edit_gpt():
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    epoch_ptd_analysis = []

    for epoch in range(NUM_EPOCH):
        epoch_topk_10 = []
        epoch_topk_20 = []
        epoch_topk_1 = []

        epoch_stats = {
                'outlier_ratios': [],
                'clean_correlations': [],
                'qualities': [],
                'alphas': []
            }

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        ptd_agg = {"count": 0, "sum": 0, "sumsq":0.0, "min": float("inf"), "max": float("-inf")}
        graph_indices = list(range(len(train_real_ts_l)))

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            num_nodes = len(node_list)
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]

            incoming_times = incoming_times_list[j]
            outcoming_times = outcoming_times_list[j]

            num_train_instance = len(node_list)
            num_train_batch = math.ceil(num_train_instance / BATCH_SIZE)
            
            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(num_train_instance, s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]

                t_cut = ts_l_cut[0]  # or np.unique(ts_l_cut)[0] if you want to be safe

                ptd_all_nodes = compute_ptd_at_t(incoming_times, outcoming_times, num_nodes,
                                                 t_cut)  # shape: [num_nodes]
                ptd_array = np.array(ptd_all_nodes)

                ptd_analysis = analyze_ptd_distribution(ptd_all_nodes, label_l_cut)
                epoch_stats['outlier_ratios'].append(ptd_analysis['outlier_ratio'])
                epoch_stats['clean_correlations'].append(abs(ptd_analysis['corr_clean']))

                # Clip at 95th percentile to remove extreme outliers
                if (ptd_array > 0).sum() > 10:
                    percentile_95 = np.percentile(ptd_array[ptd_array > 0], 95)
                    ptd_clipped = np.minimum(ptd_array, percentile_95)
                else:
                    ptd_clipped = ptd_array


                tgnn_model.set_ptd_vector(ptd_clipped)  # does: ptd_vec = log1p(scale * raw)
                


                # DEBUG: Check PTD after encoding
                #with torch.no_grad():
                #    if tgnn_model.ptd_vec is not None:
                #        ptd_encoded = tgnn_model.ptd_vec.cpu().numpy()
                #        print(f"[PTD-Encoded] shape={ptd_encoded.shape} "
                ##              f"col0: min={ptd_encoded[:,0].min():.2f} max={ptd_encoded[:,0].max():.2f} "
                 #             f"std={ptd_encoded[:,0].std():.3f}")
                

                optimizer.zero_grad()

                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    ptd_l=None,  # ignored inside tem_conv2
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )

                # ----- label prep -----
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)  # log1p(alpha * y)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)  # ensure on device
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t  # z-scored log1p target

                # ----- features / MLP input -----
                idx = torch.as_tensor(src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]  # [B, 2]
                ptd_enc = tgnn_model.ptd_encoder(ptd_pair)  # [B, 128]

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                pred_norm = MLP_model(mlp_in).squeeze(-1)                       # predict z-scored log1p


                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                rank_loss = loss_cal_rank_weighted(pred_norm, true_label, margin=0.5)
                reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
                c_loss = corr_loss(pred_norm, true_label)

                loss = rank_loss + 0.5 * reg_loss + 0.1 * c_loss

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                for attn in tgnn_model.attn_model_list:
                    if hasattr(attn, 'alpha_attn'):
                        epoch_stats['alphas'].append(attn.alpha_attn)
                

                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(tgnn_model.parameters()) + list(MLP_model.parameters()),
                                               max_norm=1.0)
                optimizer.step()

                m_loss.append(loss.item())

        # Epoch summary with PTD analysis
        avg_outlier_ratio = np.mean(epoch_stats['outlier_ratios']) if epoch_stats['outlier_ratios'] else 0
        avg_clean_corr = np.mean(epoch_stats['clean_correlations']) if epoch_stats['clean_correlations'] else 0
        avg_quality = np.mean(epoch_stats['qualities']) if epoch_stats['qualities'] else 0

        print(f" Epoch {epoch:02d} PTD Analysis:")
        print(f"   Outlier ratio: {avg_outlier_ratio:.3f}")
        print(f"   Clean correlation: {avg_clean_corr:.3f}")
        #print(f"   Alpha range: [{min(epoch_stats['alphas']):.3f}, {max(epoch_stats['alphas']):.3f}]")
        
        # Only print alpha range if we have values
        if epoch_stats['alphas']:
            print(f"   Alpha range: [{min(epoch_stats['alphas']):.3f}, {max(epoch_stats['alphas']):.3f}]")
            print(f"   Alpha mean: {np.mean(epoch_stats['alphas']):.3f}")
        else:
            print("   No alpha values collected (check attention module)")


            # After each epoch's training
        qualities = []
        alphas = []
        for attn in tgnn_model.attn_model_list:
            if hasattr(attn, 'ptd_quality_history'):
                qualities.extend(attn.ptd_quality_history)
                alphas.extend(attn.alpha_history)
                attn.ptd_quality_history = []
                attn.alpha_history = []

        #print(f"[PTD-EpochStats] epoch {epoch:02d} | "
        #    f"quality={np.mean(qualities):.3f} "
            #f"alpha={np.mean(alphas):.3f} "
            #f"(min={min(alphas):.2f}, max={max(alphas):.2f})")


        scheduler.step()
        avg_topk_1 = np.mean(epoch_topk_1)
        avg_topk_10 = np.mean(epoch_topk_10)
        avg_topk_20 = np.mean(epoch_topk_20)

        print(
            f" Epoch {epoch:02d} Summary : Avg Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | Top@20%: {avg_topk_20:.4f} ")

        epoch_loss = np.mean(m_loss)
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")





def training_model_edit_claude():
    def debug_ptd_encoding(tgnn_model):
        """
        Check what's happening in PTD encoding.
        Call this after set_ptd_vector to debug.
        """
        if tgnn_model.ptd_vec is None:
            print("[DEBUG] PTD vec is None!")
            return
        
        ptd = tgnn_model.ptd_vec.cpu().numpy()
        
        print(f"\n[PTD Debug]")
        print(f"  Shape: {ptd.shape}")
        print(f"  Column 0 (zlog):")
        print(f"    Min: {ptd[:,0].min():.3f}, Max: {ptd[:,0].max():.3f}")
        print(f"    Mean: {ptd[:,0].mean():.3f}, Std: {ptd[:,0].std():.3f}")
        print(f"    Zeros: {(ptd[:,0] == 0).mean():.1%}")
        print(f"  Column 1 (percentile):")
        print(f"    Min: {ptd[:,1].min():.3f}, Max: {ptd[:,1].max():.3f}")
        print(f"    Mean: {ptd[:,1].mean():.3f}, Std: {ptd[:,1].std():.3f}")
        
        # Check if encoding is collapsing
        unique_vals = len(np.unique(ptd[:,0]))
        print(f"  Unique values in col 0: {unique_vals} / {len(ptd)}")
        
        if ptd[:,0].std() < 0.01:
            print("  WARNING: PTD encoding has collapsed (no variance)!")



    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    epoch_ptd_analysis = []
    ptd_tracker = PTDEpochTracker()

    for epoch in range(NUM_EPOCH):
        ptd_tracker.reset()
        epoch_topk_10 = []
        epoch_topk_20 = []
        epoch_topk_1 = []

        epoch_stats = {
                'outlier_ratios': [],
                'clean_correlations': [],
                'qualities': [],
                'alphas': []
            }

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        ptd_agg = {"count": 0, "sum": 0, "sumsq":0.0, "min": float("inf"), "max": float("-inf")}
        graph_indices = list(range(len(train_real_ts_l)))

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            num_nodes = len(node_list)
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]

            incoming_times = incoming_times_list[j]
            outcoming_times = outcoming_times_list[j]

            num_train_instance = len(node_list)
            num_train_batch = math.ceil(num_train_instance / BATCH_SIZE)
            
            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(num_train_instance, s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]

                t_cut = ts_l_cut[0]  # or np.unique(ts_l_cut)[0] if you want to be safe

                ptd_all_nodes = compute_ptd_at_t(incoming_times, outcoming_times, num_nodes,
                                                 t_cut)  # shape: [num_nodes]
                ptd_array = np.array(ptd_all_nodes)

                # Record raw PTD stats (no printing)
                ptd_tracker.record_raw_ptd(ptd_array)

                ptd_analysis = analyze_ptd_distribution(ptd_all_nodes, label_l_cut)
                epoch_stats['outlier_ratios'].append(ptd_analysis['outlier_ratio'])
                epoch_stats['clean_correlations'].append(abs(ptd_analysis['corr_clean']))

                # Clip at 95th percentile to remove extreme outliers
                if (ptd_array > 0).sum() > 10:
                    percentile_95 = np.percentile(ptd_array[ptd_array > 0], 95)
                    ptd_clipped = np.minimum(ptd_array, percentile_95)
                else:
                    ptd_clipped = ptd_array


                tgnn_model.set_ptd_vector(ptd_clipped)  # does: ptd_vec = log1p(scale * raw)
            
                # Record encoded PTD stats (no printing)
                if tgnn_model.ptd_vec is not None:
                    ptd_tracker.record_encoded_ptd(tgnn_model.ptd_vec)
                

                # DEBUG: Check PTD after encoding
                #with torch.no_grad():
                #    if tgnn_model.ptd_vec is not None:
                #        ptd_encoded = tgnn_model.ptd_vec.cpu().numpy()
                #        print(f"[PTD-Encoded] shape={ptd_encoded.shape} "
                ##              f"col0: min={ptd_encoded[:,0].min():.2f} max={ptd_encoded[:,0].max():.2f} "
                 #             f"std={ptd_encoded[:,0].std():.3f}")
                

                optimizer.zero_grad()

                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    ptd_l=None,  # ignored inside tem_conv2
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )

                # ----- label prep -----
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)  # log1p(alpha * y)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)  # ensure on device
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t  # z-scored log1p target

                # ----- features / MLP input -----
                idx = torch.as_tensor(src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]  # [B, 2]
                ptd_enc = tgnn_model.ptd_encoder(ptd_pair)  # [B, 128]

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                pred_norm = MLP_model(mlp_in).squeeze(-1)                       # predict z-scored log1p


                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                rank_loss = loss_cal_rank_weighted(pred_norm, true_label, margin=0.5)
                reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
                c_loss = corr_loss(pred_norm, true_label)

                loss = rank_loss + 0.5 * reg_loss + 0.1 * c_loss

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                for attn in tgnn_model.attn_model_list:
                    if hasattr(attn, 'alpha_attn'):
                        epoch_stats['alphas'].append(attn.alpha_attn)
                

                # Collect attention stats (no printing)
                for attn in tgnn_model.attn_model_list:
                    ptd_tracker.record_attention_stats(attn)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(tgnn_model.parameters()) + list(MLP_model.parameters()),
                                               max_norm=1.0)
                optimizer.step()

                m_loss.append(loss.item())

        # Epoch summary with PTD analysis
        avg_outlier_ratio = np.mean(epoch_stats['outlier_ratios']) if epoch_stats['outlier_ratios'] else 0
        avg_clean_corr = np.mean(epoch_stats['clean_correlations']) if epoch_stats['clean_correlations'] else 0
        avg_quality = np.mean(epoch_stats['qualities']) if epoch_stats['qualities'] else 0


        ptd_tracker.print_epoch_summary(epoch)

        print(f" Epoch {epoch:02d} PTD Analysis:")
        print(f"   Outlier ratio: {avg_outlier_ratio:.3f}")
        print(f"   Clean correlation: {avg_clean_corr:.3f}")
        #print(f"   Alpha range: [{min(epoch_stats['alphas']):.3f}, {max(epoch_stats['alphas']):.3f}]")
        
        # Only print alpha range if we have values
        if epoch_stats['alphas']:
                print(f"   Alpha range: [{min(epoch_stats['alphas']):.3f}, "
                    f"{max(epoch_stats['alphas']):.3f}]")
        else:
                print("   No alpha values collected")


            # After each epoch's training
        qualities = []
        alphas = []
        for attn in tgnn_model.attn_model_list:
            if hasattr(attn, 'ptd_quality_history'):
                qualities.extend(attn.ptd_quality_history)
                alphas.extend(attn.alpha_history)
                attn.ptd_quality_history = []
                attn.alpha_history = []

        #print(f"[PTD-EpochStats] epoch {epoch:02d} | "
        #    f"quality={np.mean(qualities):.3f} "
            #f"alpha={np.mean(alphas):.3f} "
            #f"(min={min(alphas):.2f}, max={max(alphas):.2f})")


        scheduler.step()
        avg_topk_1 = np.mean(epoch_topk_1)
        avg_topk_10 = np.mean(epoch_topk_10)
        avg_topk_20 = np.mean(epoch_topk_20)

        print(
            f" Epoch {epoch:02d} Summary : Avg Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | Top@20%: {avg_topk_20:.4f} ")
        # After each epoch
        for li, attn in enumerate(tgnn_model.attn_model_list, 1):
            if hasattr(attn, "dump_epoch_diag_and_reset"):
                attn.dump_epoch_diag_and_reset(epoch, prefix=f"[Layer{li}] ")

        epoch_loss = np.mean(m_loss)
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")




def _pair_targets_by_true(y_true, i, j):
    # +1 if y_true[i] > y_true[j], -1 if <, 0 if tie (we drop ties later)
    return torch.sign(y_true[i] - y_true[j]).float().clamp(min=-1.0, max=1.0)

def improved_ranking_loss2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    neighbor_counts: torch.Tensor,
    sample_num: int = None,
    min_gap: float = 0.05,
    p_pos: float = 0.7,
    pair_loss: str = "logistic",   # "logistic" (BPR) or "hinge" (margin ranking)
    margin: float = 0.25,
    hard_frac: float = 0.5,        # sample split: hard vs uniform
    eps: float = 1e-6,
):
    """
    Pairwise ranking with *per-pair* weighting from neighbor quality.
    y_pred, y_true are normalized (z-scored log1p) 1-D tensors length N.
    neighbor_counts is per-item [N]. We weight a pair by mean of the two items' counts.
    """
    device = y_pred.device
    N = y_true.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ---- sample pairs --------------------------------------------------------
    if sample_num is None:
        sample_num = max(100 * N, 10000)

    # Precompute ordering by true (desc) for "hard" sampling
    order = torch.argsort(-y_true)
    # uniform pairs
    i_u = torch.randint(0, N, (sample_num,), device=device)
    j_u = torch.randint(0, N, (sample_num,), device=device)

    # "hard" pairs: close ranks but with different labels
    k = int(sample_num * hard_frac)
    if k > 0:
        pos = torch.randint(0, N-1, (k,), device=device)
        # neighbors in the sorted list (hard near-ties)
        i_h = order[pos]
        j_h = order[pos + 1]
        i = torch.cat([i_u, i_h], dim=0)
        j = torch.cat([j_u, j_h], dim=0)
    else:
        i, j = i_u, j_u

    # avoid identical indices
    same = (i == j)
    if same.any():
        j = torch.where(same, (j + 1) % N, j)

    # targets from true labels (+1 if yi>yj, -1 if yi<yj)
    tgt = _pair_targets_by_true(y_true, i, j)          # [-1, 0, +1]
    # drop ties and tiny gaps
    gap = (y_true[i] - y_true[j]).abs()
    mask_gap = gap >= min_gap
    mask_tgt = tgt != 0
    mask = mask_gap & mask_tgt

    if mask.sum() == 0:
        # fallback: single margin loss on random pair set
        a, b = y_pred[i], y_pred[j]
        tgt_mr = torch.where((y_true[i] > y_true[j]), torch.ones_like(a), -torch.ones_like(a))
        return F.margin_ranking_loss(a, b, tgt_mr, margin=margin)

    i = i[mask]; j = j[mask]; tgt = tgt[mask]

    # per-pair neighbor weights (mean of the two endpoints, normalized a bit)
    w_i = neighbor_counts[i].float()
    w_j = neighbor_counts[j].float()
    w = 0.5 * (w_i + w_j)
    # squash to reasonable range to avoid exploding grads
    w = torch.clamp(w / 5.0, min=0.2, max=1.0)         # [pairs]

    a = y_pred[i]; b = y_pred[j]

    # ---- pair losses ---------------------------------------------------------
    if pair_loss == "logistic":
        # BPR / logistic pairwise loss: -log σ( tgt * (a-b) )
        s = tgt * (a - b)
        # log-sigmoid is stable; weight and mean
        loss_vec = -F.logsigmoid(s)
    elif pair_loss == "hinge":
        # margin ranking hinge: max(0, margin - tgt*(a-b))
        loss_vec = F.relu(margin - tgt * (a - b))
    else:
        raise ValueError("pair_loss must be 'logistic' or 'hinge'")

    # weighted mean (avoid zero denominator)
    num = (w * loss_vec).sum()
    den = w.sum().clamp_min(eps)
    return num / den


def nvc_pair_weights(neighbor_counts: torch.Tensor, sp: float,
                     floor: float = 0.25, ceil: float = 1.0):
    """
    Build per-item weights from NVC and batch Spearman sp.
    sp < 0  : concave upweight high-NVC
    sp > 0  : flatten or slightly downweight high-NVC
    Returns per-item weights in [floor, ceil].
    """
    nvc = neighbor_counts.float()
    # normalize within batch (robust)
    nvcn = (nvc - nvc.mean()) / (nvc.std(unbiased=False) + 1e-6)

    if sp <= -0.10:
        # helpful neighbors: concave boost
        w = torch.log1p(nvc.clamp_min(0)) / torch.log1p(nvc.max().clamp_min(1.0))
    elif sp >= 0.10:
        # harmful neighbors: flatten / mild inverse
        w = 1.0 - 0.25 * torch.tanh(nvcn)  # around 0.75 ± ~0.25
    else:
        # neutral: near-linear mild scaling
        w = 0.5 + 0.5 * torch.tanh(0.5 * nvcn)

    return torch.clamp(w, floor, ceil)

def improved_ranking_loss3(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    neighbor_counts: torch.Tensor,
    sample_num: int = None,
    min_gap: float = 0.05,
    pair_loss: str = "logistic",   # "logistic" or "hinge"
    margin: float = 0.25,
    hard_frac: float = 0.5,
    batch_spearman: float = 0.0,   # <-- NEW: pass your per-batch NVC↔|err| Spearman here
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pairwise ranking with *adaptive per-pair weighting* based on neighbor counts and batch Spearman.
    - y_pred, y_true: [N]
    - neighbor_counts: [N]
    """
    device = y_pred.device
    N = y_true.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- sample pairs ---
    if sample_num is None:
        sample_num = max(100 * N, 10000)

    order = torch.argsort(-y_true)  # by true desc for "hard" neighbors

    # uniform
    i_u = torch.randint(0, N, (sample_num,), device=device)
    j_u = torch.randint(0, N, (sample_num,), device=device)

    # hard: adjacent in the sorted-by-true ranking
    k = int(sample_num * hard_frac)
    if k > 0 and N > 1:
        pos = torch.randint(0, N - 1, (k,), device=device)
        i_h = order[pos]
        j_h = order[pos + 1]
        i = torch.cat([i_u, i_h], dim=0)
        j = torch.cat([j_u, j_h], dim=0)
    else:
        i, j = i_u, j_u

    # avoid identical pairs
    same = (i == j)
    if same.any():
        j = torch.where(same, (j + 1) % N, j)

    # targets from true: +1 if yi > yj, -1 if yi < yj; drop ties
    diff_true = y_true[i] - y_true[j]
    tgt = torch.sign(diff_true).float()  # [-1, 0, +1]
    # filter small gaps & ties
    mask = (tgt != 0) & (diff_true.abs() >= min_gap)
    if mask.sum() == 0:
        # fallback: compute a single margin loss on a reduced set
        a, b = y_pred[i], y_pred[j]
        tgt_mr = torch.where(y_true[i] > y_true[j], torch.ones_like(a), -torch.ones_like(a))
        return F.margin_ranking_loss(a, b, tgt_mr, margin=margin)

    i = i[mask]; j = j[mask]; tgt = tgt[mask]

    # per-pair weights from NVC and batch Spearman
    w_i = nvc_pair_weights(neighbor_counts[i], sp=float(batch_spearman))
    w_j = nvc_pair_weights(neighbor_counts[j], sp=float(batch_spearman))
    w = 0.5 * (w_i + w_j)

    # pairwise scores
    a = y_pred[i]; b = y_pred[j]
    if pair_loss == "logistic":
        # BPR/logistic: -log σ( tgt * (a-b) )
        s = tgt * (a - b)
        loss_vec = -F.logsigmoid(s)
    elif pair_loss == "hinge":
        # hinge: max(0, margin - tgt*(a-b))
        loss_vec = F.relu(margin - tgt * (a - b))
    else:
        raise ValueError("pair_loss must be 'logistic' or 'hinge'")

    num = (w * loss_vec).sum()
    den = w.sum().clamp_min(eps)
    return num / den

def training_model_27_gemini():

    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

    ptd_prior = torch.nn.Linear(tgnn_model.ptd_embed_dim, 1).to(device)


    for epoch in range(NUM_EPOCH):
        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []
        epoch_ptd_qualities = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        epoch_nvc_sum = 0.0
        epoch_nvc_sqsum = 0.0
        epoch_nvc_count = 0
        epoch_nvc_zerocnt = 0
        epoch_nvc_min = float('inf')
        epoch_nvc_max = float('-inf')

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)
            
            ptd_idx = ptd_indices[j]
            ptd_cache = ptd_caches[j]

            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)
                
                optimizer.zero_grad()

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )
                
                # Get neighbor counts (float, device-safe)
                neighbor_valid_counts = getattr(tgnn_model, 'last_valid_counts', None)
                if neighbor_valid_counts is None or neighbor_valid_counts.numel() != len(src_l_cut):
                    neighbor_valid_counts = torch.ones(len(src_l_cut), device=device) * 10.0
                else:
                    neighbor_valid_counts = neighbor_valid_counts.to(device).float()

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_mlp(ptd_pair)
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                pred_norm = MLP_model(mlp_in).squeeze(-1)

                
                LOG_EVERY = 1000  # adjust as you like
                with torch.no_grad():
                    # accumulate epoch stats
                    nvc = neighbor_valid_counts
                    epoch_nvc_sum    += float(nvc.sum())
                    epoch_nvc_sqsum  += float((nvc**2).sum())
                    epoch_nvc_count  += nvc.numel()
                    epoch_nvc_zerocnt += int((nvc == 0).sum())
                    epoch_nvc_min = min(epoch_nvc_min, float(nvc.min()))
                    epoch_nvc_max = max(epoch_nvc_max, float(nvc.max()))

                    if (batch_i % LOG_EVERY)==0 :
                        # quick batch summary
                        b_mean = float(nvc.mean())
                        b_std  = float(nvc.std(unbiased=False))
                        b_min  = int(nvc.min())
                        b_max  = int(nvc.max())
                        zero_rate = float((nvc == 0).float().mean())

                        # error–NVC Spearman (negative is good)
                        abs_err = (pred_norm - true_label).abs()
                        rx = abs_err.argsort().argsort().float()
                        ry = nvc.argsort().argsort().float()
                        rx = (rx - rx.mean()) / (rx.std(unbiased=False) + 1e-6)
                        ry = (ry - ry.mean()) / (ry.std(unbiased=False) + 1e-6)
                        sp = float((rx * ry).mean().clamp(-1, 1))
                        if (sp > 0.10 or zero_rate > 0.40):
                            #print(f"[Batch {batch_i:04d}] NVC mean={b_mean:.2f} std={b_std:.2f} "
                            #    f"min={b_min} max={b_max} zero%={100*zero_rate:.1f} | "
                            #    f"NVC↔|err| Spearman={sp:+.3f}")

                            # optional coarse buckets (compact)
                            # bins: 0, 1-2, 3-5, 6-10, 11+
                            edges = torch.tensor([0,1,3,6,11], device=device)
                            nvc_cpu = nvc.detach().cpu()
                            ae_cpu = abs_err.detach().cpu().numpy()
                            # simple boolean masks
                            masks = [
                                (nvc_cpu == 0),
                                (nvc_cpu >= 1) & (nvc_cpu < 3),
                                (nvc_cpu >= 3) & (nvc_cpu < 6),
                                (nvc_cpu >= 6) & (nvc_cpu < 11),
                                (nvc_cpu >= 11),
                            ]
                            labels = ["0","1-2","3-5","6-10","11+"]
                            for name, m in zip(labels, masks):
                                if m.any():
                                    m_idx = m.nonzero(as_tuple=False).squeeze(1).numpy()
                                    mae = ae_cpu[m_idx].mean()
                                    #print(f"  [Bucket {name:>4}] n={m.sum().item():4d} MAE={mae:.4f}")
                #print(f"[NVC] mean={b_mean:.2f} std={b_std:.2f} min={b_min} max={b_max} zero%={100*zero_rate:.1f}")

                

                rank_loss = improved_ranking_loss(pred_norm, true_label, neighbor_valid_counts)
                reg_loss = torch.nn.functional.l1_loss(pred_norm, true_label)
                c_loss = corr_loss(pred_norm, true_label)

                loss = rank_loss + 0.5 * reg_loss + 0.1 * c_loss # Used for all the testings

                mask0 = (neighbor_valid_counts == 0)
                if mask0.any():
                    prior_pred = ptd_prior(ptd_enc[mask0]).squeeze(-1)
                    prior_loss = 0.05 * F.smooth_l1_loss(prior_pred, true_label[mask0])
                    loss = loss + prior_loss


                #loss = F.smooth_l1_loss(pred_norm, true_label, beta=0.5)

                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())

                # Periodic logging with PTD quality
                #if batch_i % 100 == 0:
                #    print(f"Epoch {epoch}, Graph {j}, Batch {batch_i}: Loss={loss.item():.4f}, PTD Quality={ptd_quality:.3f}")


        #adaptive_loss_fn.diagnostics.analyze_epoch_patterns(epoch)
        scheduler.step()
        
        if epoch_nvc_count > 0:
            mean = epoch_nvc_sum / epoch_nvc_count
            var  = epoch_nvc_sqsum / epoch_nvc_count - mean**2
            std  = max(var, 0.0) ** 0.5
            zero_pct = 100.0 * (epoch_nvc_zerocnt / epoch_nvc_count)
            print(f"[Epoch {epoch:02d} NVC] mean={mean:.2f} std={std:.2f} "
                f"min={epoch_nvc_min:.0f} max={epoch_nvc_max:.0f} zero%={zero_pct:.1f}")
    
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0
        avg_ptd_quality = np.mean(epoch_ptd_qualities) if epoch_ptd_qualities else 0
        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f} | Avg PTD Quality: {avg_ptd_quality:.3f}")
        
        epoch_loss = np.mean(m_loss) if m_loss else float('inf')
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")


def eval_real_data_27_gemini(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_list = []
    tgan.ngh_finder = sampler

    all_pred_raw = []
    all_true_raw = []
    all_ptd_raw  = []


    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        ptd_defualt_all_nodes = test_pass_through_d
        ptd_all_nodes = ptd_defualt_all_nodes
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            # 1-based node ids and cut times
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            test_ts  = np.array(ts[s_idx:e_idx],  dtype=float)

            # ----- PTD snapshot at the first time in the batch -----
            t_cut = float(test_ts[0])

            ptd_past_1b = test_ptd_cache.get(t_cut)
            if ptd_past_1b is None:
                ptd_past_1b, _, _ = test_ptd_index.snapshot_partition_all(t_cut)  # [max_id+1], 1-based
                test_ptd_cache[t_cut] = ptd_past_1b


            # Write 2-D PTD feature table [log1p(scale*raw), percentile]
            tgan.set_ptd_vector(ptd_past_1b)

            # Safe 0-based indices
            idx0  = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            # Labels aligned to idx0
            true_batch = np.asarray(label, dtype=float)[idx0]

            #-----

            #tgan.set_ptd_vector(ptd_all_nodes)

            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,  # [B, 1]
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )

            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]  # [B, 2]


            ptd_enc_scaled = tgan.ptd_mlp(ptd_pair)
            mlp_in = torch.cat([src_embed, ptd_enc_scaled], dim=1)
            #pred_norm = lr_model(mlp_in).squeeze(-1)  # normalized prediction
            pred_norm = calib(lr_model(mlp_in)).squeeze(-1)  # normalized prediction

            mu_t = torch.tensor(mu, dtype=pred_norm.dtype, device=pred_norm.device)
            sigma_t = torch.tensor(sigma, dtype=pred_norm.dtype, device=pred_norm.device)
            pred_log = pred_norm * sigma_t + mu_t

            #pred_raw = normalizer.torch_inverse(pred_log)

            # NUMERICAL GUARDRAILS:
            # clamp the log-domain to avoid exp overflow on outliers
            max_z = 8.0                      # ~> exp(μ + 8σ) is already astronomical
            pred_log = torch.clamp(pred_log, max=mu_t + max_z * sigma_t)
            pred_raw = normalizer.torch_inverse(pred_log)  # (expm1 & /α inside)
            pred_raw = pred_raw.clamp_min(0)               # centralities are ≥ 0
            
            # Accumulate (flatten to 1-D)
            all_pred_raw.append(pred_raw.detach().clamp_min(0).cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))
            #all_ptd_raw.append(ptd_raw_batch.detach().cpu().numpy().reshape(-1))



        # ===== whole-graph arrays =====
        pred = np.concatenate(all_pred_raw, axis=0)
        true = np.concatenate(all_true_raw, axis=0)
        #ptd  = np.concatenate(test_pass_through_d,  axis=0)

        results = hits_in_ks(true, pred, Ks=[10, 30, 50])

        # Access programmatically
        for K, (hits, pct) in results.items():
            print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")


        unified_evaluation_with_statistics2(pred, true, test_pass_through_d, hint)
        
        end_time = time.time()
        e_time = (end_time - start_time) / 60.0

        return e_time


    end_time = time.time()
    e_time = (end_time - start_time) / 60.0

    return e_time


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, base_loss=nn.CrossEntropyLoss(reduction="mean")):
        super().__init__()
        self.temperature = temperature
        self.base_loss = base_loss

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        """
        Calculates Supervised Contrastive Loss (NT-Xent adapted for classification).
        z: Contrastive embeddings
        labels: Centrality classes (long type indices 0, 1, 2,...)
        """
        device = z.device
        B = z.size(0)
        
        # 1. Normalize embeddings
        z = F.normalize(z, dim=1)

        # 2. Compute similarity matrix (dot product)
        # Sim = Z @ Z.T
        sim_matrix = torch.matmul(z, z.T) / self.temperature #
        
        # 3. Create identity mask (set self-similarity to -inf)
        logits_mask = torch.scatter(
            torch.ones_like(sim_matrix, dtype=torch.bool),
            1,
            torch.arange(B, device=device).view(-1, 1),
            0
        )
        sim_matrix = sim_matrix[logits_mask].view(B, -1) #

        # 4. Create Positive and Negative masks
        # Labels must be expanded to match the similarity matrix dimensions
        labels = labels.contiguous().view(-1, 1)
        
        # Mask where labels are equal (Positives)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-loops from the positive mask
        positive_mask = positive_mask[logits_mask].view(B, -1) #
        
        # 5. Compute Contrastive Loss (Numerator is log-softmax on positive pairs)
        # Logits are the similarities of the B*B-1 pairs
        logits = sim_matrix
        
        # Logits of positive pairs only
        log_prob = F.log_softmax(logits, dim=-1) #
        
        # Sum over positive log probabilities (only non-zero entries in positive_mask contribute)
        # Summing log(exp(sim_pos) / sum(exp(sim_all)))
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1e-6)
        
        # Final Contrastive Loss (negative mean of log probabilities)
        loss = -mean_log_prob_pos.mean()
        
        return loss




class ImprovedRankingLoss(nn.Module):
    """
    Explicitly penalizes rank inversions (i.e., when pred_i < pred_j but true_i > true_j).
    The penalty is weighted by the magnitude of the true difference (true_diff.abs()),
    ensuring that ordering errors involving high-TBC nodes contribute the most to the loss.
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, pred: torch.Tensor, true_label: torch.Tensor) -> torch.Tensor:
        B = pred.size(0)
        
        # 1. Create all possible unique pairs of indices (i, j)
        indices = torch.arange(B, device=pred.device)
        i_idx, j_idx = torch.meshgrid(indices, indices, indexing='ij')
        
        # Mask out self-comparisons (i!= j)
        mask = (i_idx!= j_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        # 2. Gather data for all pairs
        pred_i, pred_j = pred[i_idx], pred[j_idx]
        true_i, true_j = true_label[i_idx], true_label[j_idx]

        # 3. Identify truly ordered pairs (where true_i > true_j)
        order_mask = (true_i > true_j)
        
        # Apply mask to focus only on pairs where a specific order is required
        true_diff = (true_i - true_j)[order_mask]
        pred_diff = (pred_i - pred_j)[order_mask]
        
        # 4. Calculate the Rank Reversal Penalty
        
        # Ratio normalization: Penalizes when the predicted difference (pred_diff) is 
        # small relative to the true required difference (true_diff).
        # Normalization stabilizes the gradient.
        ratio_norm = pred_diff / (true_diff.abs() + 1e-6) 

        # Penalty = ReLU(margin - ratio_norm)
        # Loss is incurred when pred_i is not sufficiently larger than pred_j
        penalty = self.relu(self.margin - ratio_norm)
        
        if penalty.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        return penalty.mean()



def corr_loss(pred, target, eps=1e-8):
    """Calculates 1 - Pearson Correlation Coefficient."""
    pred_c = pred - pred.mean()
    targ_c = target - target.mean()
    denom = (pred_c.norm() * targ_c.norm()).clamp_min(eps)
    return 1.0 - (pred_c * targ_c).sum() / denom

def training_model_27_gemini_CONTR():
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE*0.1)


    # === NEW: Initialize Contrastive Loss ===
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.05)
    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]
    LAMBDA_MAE = 0.1
    LAMBDA_RANK = 0.9
    ranking_loss_fn = ImprovedRankingLoss(margin=1.0)
    PTD_INVERSE_BOOST_C = 2.0 # Constant C to control amplification power
    PTD_STABILITY_TERM = 0.1 # Epsilon to prevent division by zero near log(1)=0


    for epoch in range(NUM_EPOCH):

        # --- NEW: Loss and Prediction Statistic Trackers ---
        epoch_cl_loss = []
        epoch_weighted_mae_loss = []
        epoch_rank_loss = []
        epoch_pred_std = [] 
        epoch_pred_mean =[]
        epoch_attn_std = []
        epoch_fc3_grad_norm = []
        epoch_fc1_grad_norm = []

        epoch_total_loss = []
        epoch_cl_contrib = []
        epoch_reg_contrib = []
        epoch_mae_contrib = []
        epoch_rank_contrib = []
        epoch_reg_loss_combined = []

        epoch_avg_inv_w = []
        # --- START: Initialize Expanded Loss Accumulators ---
        epoch_total_loss, epoch_cl_contrib, epoch_reg_contrib = [], [], []
        epoch_huber_contrib, epoch_corr_contrib = [], []
        
        # Raw (unweighted) values for deeper analysis
        epoch_raw_cl, epoch_raw_huber, epoch_raw_corr = [], [], []
    # --- END: Initialize ---

        # --- END NEW TRACKERS ---

        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []
        epoch_ptd_qualities = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        epoch_nvc_sum = 0.0
        epoch_nvc_sqsum = 0.0
        epoch_nvc_count = 0
        epoch_nvc_zerocnt = 0
        epoch_nvc_min = float('inf')
        epoch_nvc_max = float('-inf')

        epoch_z_norm = []

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)

            for batch_i in range(num_train_batch):


                optimizer.zero_grad()

                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)


                attn_std_metric = getattr(tgnn_model, 'last_attn_std', 0.0) 
                epoch_attn_std.append(attn_std_metric) # NEW

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t


                # --- STEP 1: Centrality Class Assignment (Supervision for CL) ---
                
                tbc_classes = torch.zeros_like(true_label, dtype=torch.long, device=device)
                
                medium_mask = (true_label >= TBC_CLASS_THRESHOLDS[0]) & (true_label < TBC_CLASS_THRESHOLDS[1])

                # Mask for High Centrality (Class 2 / Minority)
                high_mask = (true_label >= TBC_CLASS_THRESHOLDS[1])


                # Assign Classes
                tbc_classes[medium_mask] = 1
                tbc_classes[high_mask] = 2

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_mlp(ptd_pair)
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
                missing = [n for n,p in MLP_model.named_parameters() if id(p) not in opt_param_ids]
                assert not missing, f"MLP params missing from optimizer: {missing}"

                # 2) Ensure no accidental no_grad around forward/loss
                assert torch.is_grad_enabled(), "Grad disabled somewhere (no_grad?)"

                # 3) Ensure inputs require grads
                assert mlp_in.requires_grad, "mlp_in has no grad (detached upstream?)"


                log_ptd_value = ptd_pair[..., 0].abs() # Use the log magnitude
                inverse_weight_rank = PTD_INVERSE_BOOST_C / (log_ptd_value + PTD_STABILITY_TERM)
                avg_inverse_weight = inverse_weight_rank.mean()


                # --- STEP 2: Continue to Contrastive Loss Calculation ---
                z = tgnn_model.contrast_head(src_embed)
                epoch_z_norm.append(z.norm(dim=1).mean().item())
                cl_loss = contrastive_loss_fn(z, tbc_classes)

                

                mlp_output_raw = MLP_model(mlp_in) # Shape is [B, 1] before squeeze
                pred_norm = calib(mlp_output_raw).squeeze(-1)

                huber = nn.SmoothL1Loss(beta=1.0, reduction='none')
                weighted_mae_loss = (huber(pred_norm, true_label) * true_label.abs()).mean()

                rank_loss = ranking_loss_fn(pred_norm, true_label)
                rank_loss = rank_loss * avg_inverse_weight

                reg_loss_combined = LAMBDA_MAE * weighted_mae_loss + LAMBDA_RANK * rank_loss
            


                loss = 0.6 * cl_loss + (1.0 - 0.6) * reg_loss_combined

                epoch_cl_loss.append(cl_loss.item())
                epoch_weighted_mae_loss.append(weighted_mae_loss.item())
                epoch_rank_loss.append(rank_loss.item())
                epoch_reg_loss_combined.append(reg_loss_combined.item())
                epoch_total_loss.append(loss.item())


                epoch_pred_std.append(pred_norm.std().item())
                epoch_pred_mean.append(pred_norm.mean().item())
        
 
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()



                with torch.no_grad():
                    g_fc1 = (MLP_model.fc1.weight.grad.norm().item()
                            if MLP_model.fc1.weight.grad is not None else float('nan'))
                    g_fc3 = (MLP_model.fc3.weight.grad.norm().item()
                            if MLP_model.fc3.weight.grad is not None else float('nan'))
                epoch_fc1_grad_norm.append(g_fc1)
                epoch_fc3_grad_norm.append(g_fc3)


                # --- NEW: ADD TARGETED GRADIENT CLIPPING FOR THE MLP ---
                torch.nn.utils.clip_grad_norm_(MLP_model.parameters(), max_norm=0.5)
                # --------------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())


        scheduler.step()
        
        if epoch_nvc_count > 0:
            mean = epoch_nvc_sum / epoch_nvc_count
            var  = epoch_nvc_sqsum / epoch_nvc_count - mean**2
            std  = max(var, 0.0) ** 0.5
            zero_pct = 100.0 * (epoch_nvc_zerocnt / epoch_nvc_count)
            print(f"[Epoch {epoch:02d} NVC] mean={mean:.2f} std={std:.2f} "
                f"min={epoch_nvc_min:.0f} max={epoch_nvc_max:.0f} zero%={zero_pct:.1f}")
    
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0
        avg_fc3_grad_norm = np.mean(epoch_fc3_grad_norm) if epoch_fc3_grad_norm else 0.0
        avg_fc1_grad_norm = np.mean(epoch_fc1_grad_norm) if epoch_fc1_grad_norm else 0.0

        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f}")
        
        epoch_loss = np.mean(m_loss) if m_loss else float('inf')

         # --- DIAGNOSTIC AVERAGES ---
        avg_cl = np.mean(epoch_cl_loss) if epoch_cl_loss else 0.0
        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0

        true_std_ref = 1.0 

        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
            f"Top@20%: {avg_topk_20:.4f} ")
        
        avg_z_norm = np.mean(epoch_z_norm) if epoch_z_norm else 0.0

        mlp_output_wt_norm = MLP_model.fc3.weight.norm().item()
        avg_attn_std = np.mean(epoch_attn_std) if epoch_attn_std else 0.0
        avg_pred_mean = np.mean(epoch_pred_mean) if epoch_pred_mean else 0.0

        # --- DIAGNOSTIC PRINT ---
        print("\n--- (Avg per Epoch) ---")

        avg_total_loss = np.mean(epoch_total_loss) if epoch_total_loss else 0


        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0

        avg_rank_loss = np.mean(epoch_rank_loss) if epoch_rank_loss else 0
        avg_weighted_mae_loss = np.mean(epoch_weighted_mae_loss) if epoch_weighted_mae_loss else 0
        avg_reg_loss_combined = np.mean(epoch_reg_loss_combined) if epoch_reg_loss_combined else 0

        # --- END ACCUMULATION ---


        print("\n" + "="*50)
        print(f"EPOCH {epoch:02d} LOSS DIAGNOSTICS")
        print("="*50)
        print(f"  Avg Total Loss:         {avg_total_loss:.6f}")
        print("-" * 50)
        print(f"  WEIGHTED CONTRIBUTIONS:")
        print(f"  ├─ CL (contrastive_loss_fn) Contrib:  {avg_cl:.6f} ")
        print(f"  ├─ rank_loss Contrib:  {avg_rank_loss:.6f} ")
        print(f"  ├─ weighted_mae_loss Contrib:  {avg_weighted_mae_loss:.6f} ")
        print(f"  ├─ reg_loss_combined Contrib:  {avg_reg_loss_combined:.6f} ")
        print("="*50 + "\n")

        print(f"-------------------------------------------------")
        print(f"Pred STD (Target {true_std_ref:.4f}): {avg_pred_std:.4f}")
        print(f"Pred Mean: {avg_pred_mean:.4f}")
        print(f"MLP Output Wt Norm: {mlp_output_wt_norm:.4f}") # CHECK: Should stabilize, not explode
        print(f"Avg Z Norm (Goal 1.0000): {avg_z_norm:.4f}") # Diagnostic Check
        print(f"Avg Attn STD (Goal High): {avg_attn_std:.4f}") # NEW
        print(f"Avg MLP FC1 Output Grad Norm: {avg_fc1_grad_norm:.6f}") # NEW
        print(f"Avg MLP FC4 Output Grad  Norm: {avg_fc3_grad_norm:.6f}") # NEW


        print("-------------------------------------------------")

        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")


def training_model_27_gemini_CONTR():
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE*0.1)


    # === NEW: Initialize Contrastive Loss ===
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.05)
    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]
    LAMBDA_MAE = 0.1
    LAMBDA_RANK = 0.9
    ranking_loss_fn = ImprovedRankingLoss(margin=1.0)
    PTD_INVERSE_BOOST_C = 2.0 # Constant C to control amplification power
    PTD_STABILITY_TERM = 0.1 # Epsilon to prevent division by zero near log(1)=0


    for epoch in range(NUM_EPOCH):

        # --- NEW: Loss and Prediction Statistic Trackers ---
        epoch_cl_loss = []
        epoch_weighted_mae_loss = []
        epoch_rank_loss = []
        epoch_pred_std = [] 
        epoch_pred_mean =[]
        epoch_attn_std = []
        epoch_fc3_grad_norm = []
        epoch_fc1_grad_norm = []

        epoch_total_loss = []
        epoch_cl_contrib = []
        epoch_reg_contrib = []
        epoch_mae_contrib = []
        epoch_rank_contrib = []
        epoch_reg_loss_combined = []

        epoch_avg_inv_w = []
        # --- START: Initialize Expanded Loss Accumulators ---
        epoch_total_loss, epoch_cl_contrib, epoch_reg_contrib = [], [], []
        epoch_huber_contrib, epoch_corr_contrib = [], []
        
        # Raw (unweighted) values for deeper analysis
        epoch_raw_cl, epoch_raw_huber, epoch_raw_corr = [], [], []
    # --- END: Initialize ---

        # --- END NEW TRACKERS ---

        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []
        epoch_ptd_qualities = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        epoch_nvc_sum = 0.0
        epoch_nvc_sqsum = 0.0
        epoch_nvc_count = 0
        epoch_nvc_zerocnt = 0
        epoch_nvc_min = float('inf')
        epoch_nvc_max = float('-inf')

        epoch_z_norm = []

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)

            for batch_i in range(num_train_batch):


                optimizer.zero_grad()

                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)


                attn_std_metric = getattr(tgnn_model, 'last_attn_std', 0.0) 
                epoch_attn_std.append(attn_std_metric) # NEW

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t


                # --- STEP 1: Centrality Class Assignment (Supervision for CL) ---
                
                tbc_classes = torch.zeros_like(true_label, dtype=torch.long, device=device)
                
                medium_mask = (true_label >= TBC_CLASS_THRESHOLDS[0]) & (true_label < TBC_CLASS_THRESHOLDS[1])

                # Mask for High Centrality (Class 2 / Minority)
                high_mask = (true_label >= TBC_CLASS_THRESHOLDS[1])


                # Assign Classes
                tbc_classes[medium_mask] = 1
                tbc_classes[high_mask] = 2

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_mlp(ptd_pair)
                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
                missing = [n for n,p in MLP_model.named_parameters() if id(p) not in opt_param_ids]
                assert not missing, f"MLP params missing from optimizer: {missing}"

                # 2) Ensure no accidental no_grad around forward/loss
                assert torch.is_grad_enabled(), "Grad disabled somewhere (no_grad?)"

                # 3) Ensure inputs require grads
                assert mlp_in.requires_grad, "mlp_in has no grad (detached upstream?)"


                log_ptd_value = ptd_pair[..., 0].abs() # Use the log magnitude
                inverse_weight_rank = PTD_INVERSE_BOOST_C / (log_ptd_value + PTD_STABILITY_TERM)
                avg_inverse_weight = inverse_weight_rank.mean()


                # --- STEP 2: Continue to Contrastive Loss Calculation ---
                z = tgnn_model.contrast_head(src_embed)
                epoch_z_norm.append(z.norm(dim=1).mean().item())
                cl_loss = contrastive_loss_fn(z, tbc_classes)

                

                mlp_output_raw = MLP_model(mlp_in) # Shape is [B, 1] before squeeze
                pred_norm = calib(mlp_output_raw).squeeze(-1)

                huber = nn.SmoothL1Loss(beta=1.0, reduction='none')
                weighted_mae_loss = (huber(pred_norm, true_label) * true_label.abs()).mean()

                rank_loss = ranking_loss_fn(pred_norm, true_label)
                rank_loss = rank_loss * avg_inverse_weight

                reg_loss_combined = LAMBDA_MAE * weighted_mae_loss + LAMBDA_RANK * rank_loss
            


                loss = 0.6 * cl_loss + (1.0 - 0.6) * reg_loss_combined

                epoch_cl_loss.append(cl_loss.item())
                epoch_weighted_mae_loss.append(weighted_mae_loss.item())
                epoch_rank_loss.append(rank_loss.item())
                epoch_reg_loss_combined.append(reg_loss_combined.item())
                epoch_total_loss.append(loss.item())


                epoch_pred_std.append(pred_norm.std().item())
                epoch_pred_mean.append(pred_norm.mean().item())
        
 
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()



                with torch.no_grad():
                    g_fc1 = (MLP_model.fc1.weight.grad.norm().item()
                            if MLP_model.fc1.weight.grad is not None else float('nan'))
                    g_fc3 = (MLP_model.fc3.weight.grad.norm().item()
                            if MLP_model.fc3.weight.grad is not None else float('nan'))
                epoch_fc1_grad_norm.append(g_fc1)
                epoch_fc3_grad_norm.append(g_fc3)


                # --- NEW: ADD TARGETED GRADIENT CLIPPING FOR THE MLP ---
                torch.nn.utils.clip_grad_norm_(MLP_model.parameters(), max_norm=0.5)
                # --------------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())


        scheduler.step()
        
        if epoch_nvc_count > 0:
            mean = epoch_nvc_sum / epoch_nvc_count
            var  = epoch_nvc_sqsum / epoch_nvc_count - mean**2
            std  = max(var, 0.0) ** 0.5
            zero_pct = 100.0 * (epoch_nvc_zerocnt / epoch_nvc_count)
            print(f"[Epoch {epoch:02d} NVC] mean={mean:.2f} std={std:.2f} "
                f"min={epoch_nvc_min:.0f} max={epoch_nvc_max:.0f} zero%={zero_pct:.1f}")
    
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0
        avg_fc3_grad_norm = np.mean(epoch_fc3_grad_norm) if epoch_fc3_grad_norm else 0.0
        avg_fc1_grad_norm = np.mean(epoch_fc1_grad_norm) if epoch_fc1_grad_norm else 0.0

        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f}")
        
        epoch_loss = np.mean(m_loss) if m_loss else float('inf')

         # --- DIAGNOSTIC AVERAGES ---
        avg_cl = np.mean(epoch_cl_loss) if epoch_cl_loss else 0.0
        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0

        true_std_ref = 1.0 

        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
            f"Top@20%: {avg_topk_20:.4f} ")
        
        avg_z_norm = np.mean(epoch_z_norm) if epoch_z_norm else 0.0

        mlp_output_wt_norm = MLP_model.fc3.weight.norm().item()
        avg_attn_std = np.mean(epoch_attn_std) if epoch_attn_std else 0.0
        avg_pred_mean = np.mean(epoch_pred_mean) if epoch_pred_mean else 0.0

        # --- DIAGNOSTIC PRINT ---
        print("\n--- (Avg per Epoch) ---")

        avg_total_loss = np.mean(epoch_total_loss) if epoch_total_loss else 0


        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0

        avg_rank_loss = np.mean(epoch_rank_loss) if epoch_rank_loss else 0
        avg_weighted_mae_loss = np.mean(epoch_weighted_mae_loss) if epoch_weighted_mae_loss else 0
        avg_reg_loss_combined = np.mean(epoch_reg_loss_combined) if epoch_reg_loss_combined else 0

        # --- END ACCUMULATION ---


        print("\n" + "="*50)
        print(f"EPOCH {epoch:02d} LOSS DIAGNOSTICS")
        print("="*50)
        print(f"  Avg Total Loss:         {avg_total_loss:.6f}")
        print("-" * 50)
        print(f"  WEIGHTED CONTRIBUTIONS:")
        print(f"  ├─ CL (contrastive_loss_fn) Contrib:  {avg_cl:.6f} ")
        print(f"  ├─ rank_loss Contrib:  {avg_rank_loss:.6f} ")
        print(f"  ├─ weighted_mae_loss Contrib:  {avg_weighted_mae_loss:.6f} ")
        print(f"  ├─ reg_loss_combined Contrib:  {avg_reg_loss_combined:.6f} ")
        print("="*50 + "\n")

        print(f"-------------------------------------------------")
        print(f"Pred STD (Target {true_std_ref:.4f}): {avg_pred_std:.4f}")
        print(f"Pred Mean: {avg_pred_mean:.4f}")
        print(f"MLP Output Wt Norm: {mlp_output_wt_norm:.4f}") # CHECK: Should stabilize, not explode
        print(f"Avg Z Norm (Goal 1.0000): {avg_z_norm:.4f}") # Diagnostic Check
        print(f"Avg Attn STD (Goal High): {avg_attn_std:.4f}") # NEW
        print(f"Avg MLP FC1 Output Grad Norm: {avg_fc1_grad_norm:.6f}") # NEW
        print(f"Avg MLP FC4 Output Grad  Norm: {avg_fc3_grad_norm:.6f}") # NEW


        print("-------------------------------------------------")

        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")

# In module.py, add this new class


# In module.py, add this new, more advanced class

class AdaptiveReweightedSupConLoss(nn.Module):
    """
    An adaptive, re-weighted SupCon loss.
    Weights are a product of static importance (from TBC difference)
    and dynamic difficulty (from embedding distance).
    """
    def __init__(self, temperature=0.1, gamma_pos=0.5, gamma_neg=0.5):
        super().__init__()
        self.temperature = temperature
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, z: torch.Tensor, tbc_classes: torch.Tensor, tbc_values: torch.Tensor):
        device = z.device
        B = z.size(0)
        z = F.normalize(z, dim=1)
        
        # --- Masks (same as before) ---
        class_mask = tbc_classes.unsqueeze(1) == tbc_classes.unsqueeze(0)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        class_mask = class_mask & self_mask
        
        # --- Ground Truth Differences (same as before) ---
        tbc_diff = torch.abs(tbc_values.unsqueeze(1) - tbc_values.unsqueeze(0))
        tbc_median = torch.median(tbc_values[tbc_values > 0]) if (tbc_values > 0).any() else 1.0

        pos_mask = class_mask & (tbc_diff > 0) & (tbc_diff <= self.gamma_pos * tbc_median)
        neg_mask = class_mask & (tbc_diff >= self.gamma_neg * tbc_median)

        # --- INNOVATION: Calculate Dynamic Difficulty ---
        # Pairwise squared Euclidean distance in the embedding space
        embedding_dist_sq = torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2)

        # --- Calculate Adaptive β weights ---
        # Static importance (from CLGNN)
        static_beta_pos = (tbc_median * self.gamma_pos) / (tbc_diff + 1e-8)
        static_beta_neg = tbc_diff / (tbc_median * self.gamma_neg + 1e-8)

        # Dynamic difficulty (our innovation)
        # For positives, weight is higher if they are far apart
        # For negatives, weight is higher if they are close together
        dynamic_difficulty_pos = embedding_dist_sq
        dynamic_difficulty_neg = 1.0 / (embedding_dist_sq + 1e-8)

        # Final adaptive weights
        beta_pos = static_beta_pos * dynamic_difficulty_pos
        beta_neg = static_beta_neg * dynamic_difficulty_neg

        # --- Weighted Log-Sum-Exp (same logic, new weights) ---
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        exp_sim_pos = torch.exp(sim_matrix) * beta_pos
        exp_sim_neg = torch.exp(sim_matrix) * beta_neg

        numerator = torch.log(torch.where(pos_mask, exp_sim_pos, 0).sum(dim=1) + 1e-8)
        denominator = torch.log(
            torch.where(pos_mask, exp_sim_pos, 0).sum(dim=1) + 
            torch.where(neg_mask, exp_sim_neg, 0).sum(dim=1) + 1e-8
        )
        
        losses = denominator - numerator
        valid_anchors = pos_mask.sum(dim=1) > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device)
            
        return losses[valid_anchors].mean()

class ReweightedSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss, re-weighted by TBC differences, inspired by the CLGNN paper.
    """
    def __init__(self, temperature=0.1, gamma_pos=0.5, gamma_neg=0.5):
        super().__init__()
        self.temperature = temperature
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, z: torch.Tensor, tbc_classes: torch.Tensor, tbc_values: torch.Tensor):
        """
        z: Contrastive embeddings, shape [B, D]
        tbc_classes: Class assignments (e.g., 0, 1, 2), shape [B]
        tbc_values: Raw normalized TBC values, shape [B]
        """
        device = z.device
        B = z.size(0)
        z = F.normalize(z, dim=1)
        
        # Create masks
        class_mask = tbc_classes.unsqueeze(1) == tbc_classes.unsqueeze(0)
        # Mask out self-comparisons
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        class_mask = class_mask & self_mask
        
        # Calculate TBC differences for all pairs
        tbc_diff = torch.abs(tbc_values.unsqueeze(1) - tbc_values.unsqueeze(0))
        tbc_median = torch.median(tbc_values[tbc_values > 0]) if (tbc_values > 0).any() else 1.0

        # Define positive and negative sample masks based on TBC values
        pos_mask = class_mask & (tbc_diff > 0) & (tbc_diff <= self.gamma_pos * tbc_median)
        neg_mask = class_mask & (tbc_diff >= self.gamma_neg * tbc_median)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # --- Calculate β weights ---
        # β_pos = (median * γ_pos) / |TBC_u - TBC_v|
        beta_pos = (tbc_median * self.gamma_pos) / (tbc_diff + 1e-8)
        
        # β_neg = |TBC_u - TBC_w| / (median * γ_neg)
        beta_neg = tbc_diff / (tbc_median * self.gamma_neg + 1e-8)

        # --- Weighted Log-Sum-Exp ---
        # Numerator: Sum over positive pairs
        exp_sim_pos = torch.exp(sim_matrix) * beta_pos
        numerator = torch.log(torch.where(pos_mask, exp_sim_pos, 0).sum(dim=1) + 1e-8)

        # Denominator: Sum over all valid positive and negative pairs in the same class
        exp_sim_neg = torch.exp(sim_matrix) * beta_neg
        denominator = torch.log(
            torch.where(pos_mask, exp_sim_pos, 0).sum(dim=1) + 
            torch.where(neg_mask, exp_sim_neg, 0).sum(dim=1) + 1e-8
        )
        
        # The loss for each anchor is log(numerator) - log(denominator)
        # We take the negative because we want to maximize the objective
        losses = denominator - numerator
        
        # Average loss only over anchors that have at least one positive pair
        valid_anchors = pos_mask.sum(dim=1) > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device)
            
        return losses[valid_anchors].mean()
    




def training_model_27_gemini_CONTR_2():
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE*0.1)


    # === NEW: Initialize Contrastive Loss ===
    #contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.05)
    #contrastive_loss_fn = ReweightedSupConLoss(temperature=0.07, gamma_pos=0.5, gamma_neg=0.5)
    contrastive_loss_fn = AdaptiveReweightedSupConLoss(temperature=0.07)

    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]
    LAMBDA_MAE = 0.1
    LAMBDA_RANK = 0.9
    ranking_loss_fn = ImprovedRankingLoss(margin=1.0)
    PTD_INVERSE_BOOST_C = 2.0 # Constant C to control amplification power
    PTD_STABILITY_TERM = 0.1 # Epsilon to prevent division by zero near log(1)=0


    for epoch in range(NUM_EPOCH):

        # --- NEW: Loss and Prediction Statistic Trackers ---
        epoch_cl_loss = []
        epoch_weighted_mae_loss = []
        epoch_rank_loss = []
        epoch_pred_std = [] 
        epoch_pred_mean =[]
        epoch_attn_std = []
        epoch_fc3_grad_norm = []
        epoch_fc1_grad_norm = []

        epoch_total_loss = []
        epoch_cl_contrib = []
        epoch_reg_contrib = []
        epoch_mae_contrib = []
        epoch_rank_contrib = []
        epoch_reg_loss_combined = []

        epoch_avg_inv_w = []
        # --- START: Initialize Expanded Loss Accumulators ---
        epoch_total_loss, epoch_cl_contrib, epoch_reg_contrib = [], [], []
        epoch_huber_contrib, epoch_corr_contrib = [], []
        
        # Raw (unweighted) values for deeper analysis
        epoch_raw_cl, epoch_raw_huber, epoch_raw_corr = [], [], []
    # --- END: Initialize ---

        # --- END NEW TRACKERS ---

        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []
        epoch_ptd_qualities = []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))

        epoch_nvc_sum = 0.0
        epoch_nvc_sqsum = 0.0
        epoch_nvc_count = 0
        epoch_nvc_zerocnt = 0
        epoch_nvc_min = float('inf')
        epoch_nvc_max = float('-inf')

        epoch_z_norm = []

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)

            for batch_i in range(num_train_batch):


                optimizer.zero_grad()

                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)


                attn_std_metric = getattr(tgnn_model, 'last_attn_std', 0.0) 
                epoch_attn_std.append(attn_std_metric) # NEW

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t


                # --- STEP 1: Centrality Class Assignment (Supervision for CL) ---
                
                tbc_classes = torch.zeros_like(true_label, dtype=torch.long, device=device)
                
                medium_mask = (true_label >= TBC_CLASS_THRESHOLDS[0]) & (true_label < TBC_CLASS_THRESHOLDS[1])

                # Mask for High Centrality (Class 2 / Minority)
                high_mask = (true_label >= TBC_CLASS_THRESHOLDS[1])


                # Assign Classes
                tbc_classes[medium_mask] = 1
                tbc_classes[high_mask] = 2

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_mlp(ptd_pair)



                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)

                opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
                missing = [n for n,p in MLP_model.named_parameters() if id(p) not in opt_param_ids]
                assert not missing, f"MLP params missing from optimizer: {missing}"

                # 2) Ensure no accidental no_grad around forward/loss
                assert torch.is_grad_enabled(), "Grad disabled somewhere (no_grad?)"

                # 3) Ensure inputs require grads
                assert mlp_in.requires_grad, "mlp_in has no grad (detached upstream?)"


                log_ptd_value = ptd_pair[..., 0].abs() # Use the log magnitude
                inverse_weight_rank = PTD_INVERSE_BOOST_C / (log_ptd_value + PTD_STABILITY_TERM)
                avg_inverse_weight = inverse_weight_rank.mean()


                # --- STEP 2: Continue to Contrastive Loss Calculation ---
                z = tgnn_model.contrast_head(src_embed)
                epoch_z_norm.append(z.norm(dim=1).mean().item())
                #cl_loss = contrastive_loss_fn(z, tbc_classes)
                cl_loss = contrastive_loss_fn(z, tbc_classes, true_label)
                

                # --- START: FINAL HYBRID DECOUPLED LOSS ---

                # 1. Get PRE- and POST-calibration outputs
                mlp_output_raw = MLP_model(mlp_in) # Output from the unbounded MLP
                pred_norm = calib(mlp_output_raw).squeeze(-1)

                # 2. SAFE Magnitude Loss (applied PRE-calibration)
                # This provides a stable anchor for the MLP's raw output.
                huber_loss = F.smooth_l1_loss(mlp_output_raw.squeeze(-1), true_label)

                # 3. POWERFUL Rank Loss (applied POST-calibration)
                # Keep your successful, PTD-weighted rank loss.
                rank_loss = ranking_loss_fn(pred_norm, true_label)
                rank_loss = rank_loss * avg_inverse_weight

                # 4. Combine with a Rank-Dominant Weighting
                LAMBDA_HUBER = 0.1   # Keep the magnitude anchor small
                LAMBDA_RANK = 0.9    # Make the rank signal dominant

                reg_loss_combined = LAMBDA_HUBER * huber_loss + LAMBDA_RANK * rank_loss

                loss = 0.9 * cl_loss + (1.0 - 0.9) * reg_loss_combined
                # --- END: FINAL HYBRID DECOUPLED LOSS ---

                epoch_cl_loss.append(cl_loss.item())
                epoch_weighted_mae_loss.append(huber_loss.item())
                epoch_rank_loss.append(rank_loss.item())
                epoch_reg_loss_combined.append(reg_loss_combined.item())
                epoch_total_loss.append(loss.item())


                epoch_pred_std.append(pred_norm.std().item())
                epoch_pred_mean.append(pred_norm.mean().item())
        
 
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()



                with torch.no_grad():
                    g_fc1 = (MLP_model.fc1.weight.grad.norm().item()
                            if MLP_model.fc1.weight.grad is not None else float('nan'))
                    g_fc3 = (MLP_model.fc3.weight.grad.norm().item()
                            if MLP_model.fc3.weight.grad is not None else float('nan'))
                epoch_fc1_grad_norm.append(g_fc1)
                epoch_fc3_grad_norm.append(g_fc3)


                # --- NEW: ADD TARGETED GRADIENT CLIPPING FOR THE MLP ---
                torch.nn.utils.clip_grad_norm_(MLP_model.parameters(), max_norm=0.5)
                # --------------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())


        scheduler.step()
        
        if epoch_nvc_count > 0:
            mean = epoch_nvc_sum / epoch_nvc_count
            var  = epoch_nvc_sqsum / epoch_nvc_count - mean**2
            std  = max(var, 0.0) ** 0.5
            zero_pct = 100.0 * (epoch_nvc_zerocnt / epoch_nvc_count)
            print(f"[Epoch {epoch:02d} NVC] mean={mean:.2f} std={std:.2f} "
                f"min={epoch_nvc_min:.0f} max={epoch_nvc_max:.0f} zero%={zero_pct:.1f}")
    
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0
        avg_fc3_grad_norm = np.mean(epoch_fc3_grad_norm) if epoch_fc3_grad_norm else 0.0
        avg_fc1_grad_norm = np.mean(epoch_fc1_grad_norm) if epoch_fc1_grad_norm else 0.0

        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f}")
        
        epoch_loss = np.mean(m_loss) if m_loss else float('inf')

         # --- DIAGNOSTIC AVERAGES ---
        avg_cl = np.mean(epoch_cl_loss) if epoch_cl_loss else 0.0
        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0

        true_std_ref = 1.0 

        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
            f"Top@20%: {avg_topk_20:.4f} ")
        
        avg_z_norm = np.mean(epoch_z_norm) if epoch_z_norm else 0.0

        mlp_output_wt_norm = MLP_model.fc3.weight.norm().item()
        avg_attn_std = np.mean(epoch_attn_std) if epoch_attn_std else 0.0
        avg_pred_mean = np.mean(epoch_pred_mean) if epoch_pred_mean else 0.0

        # --- DIAGNOSTIC PRINT ---
        print("\n--- (Avg per Epoch) ---")

        avg_total_loss = np.mean(epoch_total_loss) if epoch_total_loss else 0


        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0

        avg_rank_loss = np.mean(epoch_rank_loss) if epoch_rank_loss else 0
        avg_weighted_mae_loss = np.mean(epoch_weighted_mae_loss) if epoch_weighted_mae_loss else 0
        avg_reg_loss_combined = np.mean(epoch_reg_loss_combined) if epoch_reg_loss_combined else 0

        # --- END ACCUMULATION ---


        print("\n" + "="*50)
        print(f"EPOCH {epoch:02d} LOSS DIAGNOSTICS")
        print("="*50)
        print(f"  Avg Total Loss:         {avg_total_loss:.6f}")
        print("-" * 50)
        print(f"  WEIGHTED CONTRIBUTIONS:")
        print(f"  ├─ CL (contrastive_loss_fn) Contrib:  {avg_cl:.6f} ")
        print(f"  ├─ rank_loss Contrib:  {avg_rank_loss:.6f} ")
        print(f"  ├─ Huber Loss Contrib:  {avg_weighted_mae_loss:.6f} ")
        print(f"  ├─ reg_loss_combined Contrib:  {avg_reg_loss_combined:.6f} ")
        print("="*50 + "\n")

        print(f"-------------------------------------------------")
        print(f"Pred STD (Target {true_std_ref:.4f}): {avg_pred_std:.4f}")
        print(f"Pred Mean: {avg_pred_mean:.4f}")
        print(f"MLP Output Wt Norm: {mlp_output_wt_norm:.4f}") # CHECK: Should stabilize, not explode
        print(f"Avg Z Norm (Goal 1.0000): {avg_z_norm:.4f}") # Diagnostic Check
        print(f"Avg Attn STD (Goal High): {avg_attn_std:.4f}") # NEW
        print(f"Avg MLP FC1 Output Grad Norm: {avg_fc1_grad_norm:.6f}") # NEW
        print(f"Avg MLP FC4 Output Grad  Norm: {avg_fc3_grad_norm:.6f}") # NEW


        print("-------------------------------------------------")

        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")

class ScaleMatchingLoss(nn.Module):
    def __init__(self, alpha_mse=0.3, alpha_rank=0.5, alpha_scale=0.2):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_rank = alpha_rank  
        self.alpha_scale = alpha_scale
        
    def forward(self, pred, true_label):
        # MSE/Huber for absolute accuracy
        mse_loss = F.smooth_l1_loss(pred, true_label)
        
        # Ranking loss for order preservation
        rank_loss = self.pairwise_ranking_loss(pred, true_label)
        
        # Distribution matching
        pred_std = pred.std()
        true_std = true_label.std()
        scale_loss = ((pred_std - true_std) / true_std) ** 2
        
        # Match percentiles
        if pred.numel() > 20:
            for q in [0.5, 0.9, 0.95]:
                pred_q = torch.quantile(pred, q)
                true_q = torch.quantile(true_label, q)
                scale_loss += 0.1 * F.mse_loss(pred_q, true_q)
        
        return (self.alpha_mse * mse_loss + 
                self.alpha_rank * rank_loss + 
                self.alpha_scale * scale_loss)
    
    def pairwise_ranking_loss(self, pred, true_label, margin=0.5):
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Sample pairs
        idx1 = torch.randint(0, n, (min(n*2, 100),), device=pred.device)
        idx2 = torch.randint(0, n, (min(n*2, 100),), device=pred.device)
        
        # Only consider pairs where true values differ
        true_diff = true_label[idx1] - true_label[idx2]
        pred_diff = pred[idx1] - pred[idx2]
        
        # Hinge loss for ranking
        mask = true_diff.abs() > 1e-6
        if mask.any():
            target_sign = torch.sign(true_diff[mask])
            loss = F.relu(margin - target_sign * pred_diff[mask])
            return loss.mean()
        return torch.tensor(0.0, device=pred.device)
    
def training_model_27_gemini_CONTR_3():
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE*0.1)


    # === NEW: Initialize Contrastive Loss ===
    #contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.05)
    #contrastive_loss_fn = ReweightedSupConLoss(temperature=0.07, gamma_pos=0.5, gamma_neg=0.5)
    contrastive_loss_fn = AdaptiveReweightedSupConLoss(temperature=0.07)

    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]
    LAMBDA_MAE = 0.1
    LAMBDA_RANK = 0.9
    ranking_loss_fn = ImprovedRankingLoss(margin=1.0)
    PTD_INVERSE_BOOST_C = 2.0 # Constant C to control amplification power
    PTD_STABILITY_TERM = 0.1 # Epsilon to prevent division by zero near log(1)=0


    ALPHA_CL_START = 0.15
    ALPHA_CL_END = 0.05

    for epoch in range(NUM_EPOCH):

        # --- NEW: Loss and Prediction Statistic Trackers ---
        epoch_cl_loss = []
        epoch_weighted_mae_loss = []
        epoch_rank_loss = []
        epoch_pred_std = [] 
        epoch_pred_mean =[]
        epoch_attn_std = []
        epoch_fc3_grad_norm = []
        epoch_fc1_grad_norm = []

        epoch_total_loss = []
        epoch_reg_loss_combined = []

        epoch_avg_inv_w = []
        # --- START: Initialize Expanded Loss Accumulators ---
        epoch_total_loss, epoch_cl_contrib, epoch_reg_contrib = [], [], []
        epoch_huber_contrib, epoch_corr_contrib = [], []
        
        # Raw (unweighted) values for deeper analysis
        epoch_raw_cl, epoch_raw_huber, epoch_raw_corr = [], [], []
    # --- END: Initialize ---

        # --- END NEW TRACKERS ---

        # --- NEW: Add accumulators for residual components ---
        epoch_baseline_loss = []
        epoch_residual_reg = []
        epoch_ptd_baseline_mean = []
        epoch_ptd_baseline_std = []
        epoch_model_residual_mean = []
        epoch_model_residual_std = []
        # --- END NEW ---

        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))


        epoch_z_norm = []

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)

            for batch_i in range(num_train_batch):
                optimizer.zero_grad()

                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)


                attn_std_metric = getattr(tgnn_model, 'last_attn_std', 0.0) 
                epoch_attn_std.append(attn_std_metric) # NEW

                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t


                # --- STEP 1: Centrality Class Assignment (Supervision for CL) ---
                
                tbc_classes = torch.zeros_like(true_label, dtype=torch.long, device=device)
                
                medium_mask = (true_label >= TBC_CLASS_THRESHOLDS[0]) & (true_label < TBC_CLASS_THRESHOLDS[1])

                # Mask for High Centrality (Class 2 / Minority)
                high_mask = (true_label >= TBC_CLASS_THRESHOLDS[1])


                # Assign Classes
                tbc_classes[medium_mask] = 1
                tbc_classes[high_mask] = 2

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_mlp(ptd_pair)

                #added 1 oct
                ptd_baseline = tgnn_model.ptd_baseline_predictor(ptd_pair)

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)                

                # 1. Get PRE- and POST-calibration outputs
                model_residual = MLP_model(mlp_in).squeeze(-1)

                pred_before_calib = ptd_baseline + model_residual
                pred_norm = calib(pred_before_calib).squeeze(-1)

                log_ptd_value = ptd_pair[..., 0].abs() # Use the log magnitude
                inverse_weight_rank = PTD_INVERSE_BOOST_C / (log_ptd_value + PTD_STABILITY_TERM)
                avg_inverse_weight = inverse_weight_rank.mean()


                z = tgnn_model.contrast_head(src_embed)
                epoch_z_norm.append(z.norm(dim=1).mean().item())
                cl_loss = contrastive_loss_fn(z, tbc_classes, true_label)

                huber_loss = F.smooth_l1_loss(pred_norm, true_label)
                rank_loss = ranking_loss_fn(pred_norm, true_label) * avg_inverse_weight

                baseline_loss = F.smooth_l1_loss(ptd_baseline, true_label) * 0.1

                residual_reg = model_residual.abs().mean() * 0.01

                LAMBDA_HUBER = 0.15
                LAMBDA_RANK = 0.85

                reg_loss_combined = LAMBDA_HUBER * huber_loss + LAMBDA_RANK * rank_loss

                if NUM_EPOCH > 1:
                    # Define the period over which alpha will decay (first half of training)
                    decay_epochs = NUM_EPOCH // 2
                    
                    if epoch < decay_epochs:
                        # Calculate the linear decay
                        decay_rate = (ALPHA_CL_START - ALPHA_CL_END) / decay_epochs
                        alpha_cl = ALPHA_CL_START - epoch * decay_rate
                    else:
                        # After the decay period, keep alpha at its minimum value
                        alpha_cl = ALPHA_CL_END
                else:
                    # If only one epoch, just use the starting alpha
                    alpha_cl = ALPHA_CL_START


                #loss = alpha_cl * cl_loss + (1.0 - alpha_cl) * reg_loss_combined \
                #    + (baseline_loss * 0.3) + (residual_reg * 0.2)
                
                #best
                #loss = alpha_cl * cl_loss + (1.0 - alpha_cl) * reg_loss_combined \
                #    + (baseline_loss * 0.15) + (residual_reg * 0.02)
                
                loss = alpha_cl * cl_loss + (1.0 - alpha_cl) * reg_loss_combined \
                    + (baseline_loss * 0.05) + (residual_reg * 0.01)
                

                epoch_cl_loss.append(cl_loss.item())
                epoch_weighted_mae_loss.append(huber_loss.item())
                epoch_rank_loss.append(rank_loss.item())
                epoch_reg_loss_combined.append(reg_loss_combined.item())
                epoch_total_loss.append(loss.item())

                epoch_pred_std.append(pred_norm.std().item())
                epoch_pred_mean.append(pred_norm.mean().item())
    
                epoch_model_residual_mean.append(model_residual.mean().item())
                epoch_model_residual_std.append(model_residual.std().item())

                epoch_residual_reg.append(residual_reg.item())

                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()

                with torch.no_grad():
                    g_fc1 = (MLP_model.fc1.weight.grad.norm().item()
                            if MLP_model.fc1.weight.grad is not None else float('nan'))
                    g_fc3 = (MLP_model.fc3.weight.grad.norm().item()
                            if MLP_model.fc3.weight.grad is not None else float('nan'))
                epoch_fc1_grad_norm.append(g_fc1)
                epoch_fc3_grad_norm.append(g_fc3)


                # --- NEW: ADD TARGETED GRADIENT CLIPPING FOR THE MLP ---
                torch.nn.utils.clip_grad_norm_(MLP_model.parameters(), max_norm=0.5)
                # --------------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())


        scheduler.step()
        
    
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0

        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f}")
        

         # --- DIAGNOSTIC AVERAGES ---
        avg_cl = np.mean(epoch_cl_loss) if epoch_cl_loss else 0.0
        avg_rank_loss = np.mean(epoch_rank_loss) if epoch_rank_loss else 0
        avg_weighted_mae_loss = np.mean(epoch_weighted_mae_loss) if epoch_weighted_mae_loss else 0
        avg_reg_loss_combined = np.mean(epoch_reg_loss_combined) if epoch_reg_loss_combined else 0

        avg_total_loss = np.mean(epoch_total_loss) if epoch_total_loss else 0

        #avg_baseline_loss = np.mean(epoch_baseline_loss) if epoch_baseline_loss else 0
        avg_residual_reg = np.mean(epoch_residual_reg) if epoch_residual_reg else 0

        #avg_ptd_baseline_mean = np.mean(epoch_ptd_baseline_mean) if epoch_ptd_baseline_mean else 0
        avg_ptd_baseline_std = np.mean(epoch_ptd_baseline_std) if epoch_ptd_baseline_std else 0
        avg_model_residual_mean = np.mean(epoch_model_residual_mean) if epoch_model_residual_mean else 0
        avg_model_residual_std = np.mean(epoch_model_residual_std) if epoch_model_residual_std else 0


        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0
        avg_z_norm = np.mean(epoch_z_norm) if epoch_z_norm else 0.0
        mlp_output_wt_norm = MLP_model.fc3.weight.norm().item()
        avg_attn_std = np.mean(epoch_attn_std) if epoch_attn_std else 0.0
        avg_pred_mean = np.mean(epoch_pred_mean) if epoch_pred_mean else 0.0

        true_std_ref = 1.0 

        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
            f"Top@20%: {avg_topk_20:.4f} ")
        

        print("\n" + "="*50)
        print(f"EPOCH {epoch:02d} LOSS DIAGNOSTICS")
        print("="*50)
        print(f"  Avg Total Loss:         {avg_total_loss:.6f}")
        print("-" * 50)
        print(f"  LOSS WEIGHTED CONTRIBUTIONS:")
        print(f"  ├─ Contrastive:  {avg_cl:.6f} ")
        print(f"  └─ Reg_loss_combined Contrib:  {avg_reg_loss_combined:.6f} ")
        print(f"      ├─ Rank Loss (on reg_loss_combined):  {avg_rank_loss:.6f} ")
        print(f"      ├─ Huber Loss (on reg_loss-combined):  {avg_weighted_mae_loss:.6f} ")
        print(f"  Helper Losses:")
        #print(f"  ├─ Baseline Loss:        {avg_baseline_loss * 0.15:.6f}") # Show weighted contribution
        print(f"  └─ Residual Regularizer: {avg_residual_reg * 0.02:.6f}") # Show weighted contribution
        print("-" * 50)
        print(f"  RESIDUAL SYSTEM DIAGNOSTICS:")
        print(f"  └─ GNN Residual -> Mean: {avg_model_residual_mean:+.4f}, Std: {avg_model_residual_std:.4f}")

        print("="*50 + "\n")
        print(f"-------------------------------------------------")
        print(f"Pred STD (Target {true_std_ref:.4f}): {avg_pred_std:.4f}")
        print(f"Pred Mean: {avg_pred_mean:.4f}")
        #print(f"MLP Output Wt Norm: {mlp_output_wt_norm:.4f}") # CHECK: Should stabilize, not explode
        print(f"Avg Z Norm (Goal 1.0000): {avg_z_norm:.4f}") # Diagnostic Check
        print(f"Avg Attn STD (Goal High): {avg_attn_std:.4f}") # NEW
        #print(f"Avg MLP FC1 Output Grad Norm: {avg_fc1_grad_norm:.6f}") # NEW
        #print(f"Avg MLP FC4 Output Grad  Norm: {avg_fc3_grad_norm:.6f}") # NEW

        print("-------------------------------------------------")

        #logger.info(f"Epoch {epoch}: Avg Loss {avg_total_loss:.5f}")





def training_model_27_gemini_CONTR_3_debug():
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE*0.1)

    contrastive_loss_fn = AdaptiveReweightedSupConLoss(temperature=0.07)

    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]

    ranking_loss_fn = ImprovedRankingLoss(margin=1.0)
    PTD_INVERSE_BOOST_C = 2.0 # Constant C to control amplification power
    PTD_STABILITY_TERM = 0.1 # Epsilon to prevent division by zero near log(1)=0



    LAMBDA_HUBER = 0.1 # Reduced from 0.15
    LAMBDA_RANK = 0.90  # Increased to 0.90

    ALPHA_INITIAL = 0.05  # Start with a very low CL weight
    ALPHA_PEAK = 0.25      # The optimal value found in the paper
    ALPHA_FINAL = 0.05    # The final CL weight for fine-tuning
    RAMPUP_EPOCHS = 10    # Number of epochs to ramp up to the peak


    for epoch in range(NUM_EPOCH):

        # --- NEW: Loss and Prediction Statistic Trackers ---
        epoch_cl_loss = []
        epoch_weighted_mae_loss = []
        epoch_rank_loss = []
        epoch_pred_std = [] 
        epoch_pred_mean =[]
        epoch_attn_std = []
        epoch_fc3_grad_norm = []
        epoch_fc1_grad_norm = []

        epoch_total_loss = []

        epoch_avg_inv_w = []
        # --- START: Initialize Expanded Loss Accumulators ---
        epoch_total_loss = []
        

        # --- END NEW TRACKERS --
        epoch_ptd_baseline_std = []
        epoch_model_residual_mean = []
        epoch_model_residual_std = []
        # --- END NEW ---

        epoch_topk_10, epoch_topk_20, epoch_topk_1 = [], [], []

        tgnn_model.train()
        MLP_model.train()
        m_loss = []
        
        graph_indices = list(range(len(train_real_ts_l)))
        epoch_z_norm = []

        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]
            num_train_batch = math.ceil(len(node_list) / BATCH_SIZE)

            for batch_i in range(num_train_batch):
                optimizer.zero_grad()

                s_idx = batch_i * BATCH_SIZE
                e_idx = min(len(node_list), s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]
                t_cut = ts_l_cut[0]

                # Get PTD data
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)



                # Forward pass
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut, cut_time_l=ts_l_cut, ptd_l=None,
                    curr_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS
                )

                # Label processing
                true_label_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)
                y_log = normalizer.torch_transform(true_label_raw)
                mu_t = torch.tensor(mu, dtype=y_log.dtype, device=device)
                sigma_t = torch.tensor(sigma, dtype=y_log.dtype, device=device)
                true_label = (y_log - mu_t) / sigma_t


                # --- STEP 1: Centrality Class Assignment (Supervision for CL) ---
                
                tbc_classes = torch.zeros_like(true_label, dtype=torch.long, device=device)
                
                medium_mask = (true_label >= TBC_CLASS_THRESHOLDS[0]) & (true_label < TBC_CLASS_THRESHOLDS[1])

                # Mask for High Centrality (Class 2 / Minority)
                high_mask = (true_label >= TBC_CLASS_THRESHOLDS[1])


                # Assign Classes
                tbc_classes[medium_mask] = 1
                tbc_classes[high_mask] = 2

                # MLP prediction
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc = tgnn_model.ptd_mlp(ptd_pair)

                #added 1 oct
                #ptd_baseline = tgnn_model.ptd_baseline_predictor(ptd_pair)

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)                

                # 1. Get PRE- and POST-calibration outputs
                model_residual = MLP_model(mlp_in).squeeze(-1)

                #pred_before_calib = ptd_baseline + model_residual
                pred_norm = calib(model_residual).squeeze(-1)

                log_ptd_value = ptd_pair[..., 0].abs() # Use the log magnitude
                inverse_weight_rank = PTD_INVERSE_BOOST_C / (log_ptd_value + PTD_STABILITY_TERM)
                avg_inverse_weight = inverse_weight_rank.mean()

                z = tgnn_model.contrast_head(src_embed)
                epoch_z_norm.append(z.norm(dim=1).mean().item())
                cl_loss = contrastive_loss_fn(z, tbc_classes, true_label)

                rank_loss = ranking_loss_fn(pred_norm, true_label) * avg_inverse_weight


                reg_loss_combined = rank_loss

                if epoch < RAMPUP_EPOCHS:
                    # Phase 1: Linearly ramp up from INITIAL to PEAK
                    alpha_cl = ALPHA_INITIAL + (ALPHA_PEAK - ALPHA_INITIAL) * (epoch / RAMPUP_EPOCHS)
                else:
                    # Phase 2: Linearly decay from PEAK to FINAL over the remaining epochs
                    remaining_epochs = NUM_EPOCH - RAMPUP_EPOCHS
                    epochs_into_decay = epoch - RAMPUP_EPOCHS
                    decay_rate = (ALPHA_PEAK - ALPHA_FINAL) / remaining_epochs
                    alpha_cl = ALPHA_PEAK - epochs_into_decay * decay_rate

                # Ensure alpha does not go below the final value
                alpha_cl = max(ALPHA_FINAL, alpha_cl)

                
                loss = alpha_cl * cl_loss + (1.0 - alpha_cl) * reg_loss_combined
                
                epoch_cl_loss.append(cl_loss.item())
                epoch_rank_loss.append(rank_loss.item())
                epoch_total_loss.append(loss.item())

                epoch_pred_std.append(pred_norm.std().item())
                epoch_pred_mean.append(pred_norm.mean().item())
    
                epoch_model_residual_mean.append(model_residual.mean().item())
                epoch_model_residual_std.append(model_residual.std().item())


                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN loss in epoch {epoch}, batch {batch_i}, skipping")
                    continue
                
                # Compute metrics
                topk_stats = compute_topk_metrics(pred_norm, true_label, k_list=[1, 5, 10, 20, 30], jac=False)
                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])


                loss.backward()



                # --- NEW: ADD TARGETED GRADIENT CLIPPING FOR THE MLP ---
                torch.nn.utils.clip_grad_norm_(MLP_model.parameters(), max_norm=0.5)
                # --------------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                m_loss.append(loss.item())


        scheduler.step()
        
    
        # Epoch summary with PTD quality
        avg_topk_1 = np.mean(epoch_topk_1) if epoch_topk_1 else 0
        avg_topk_10 = np.mean(epoch_topk_10) if epoch_topk_10 else 0
        avg_topk_20 = np.mean(epoch_topk_20) if epoch_topk_20 else 0

        
        print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
              f"Top@20%: {avg_topk_20:.4f}")
        

         # --- DIAGNOSTIC AVERAGES ---
        avg_cl = np.mean(epoch_cl_loss) if epoch_cl_loss else 0.0
        avg_rank_loss = np.mean(epoch_rank_loss) if epoch_rank_loss else 0
        avg_total_loss = np.mean(epoch_total_loss) if epoch_total_loss else 0


        avg_model_residual_mean = np.mean(epoch_model_residual_mean) if epoch_model_residual_mean else 0
        avg_model_residual_std = np.mean(epoch_model_residual_std) if epoch_model_residual_std else 0


        avg_pred_std = np.mean(epoch_pred_std) if epoch_pred_std else 0.0
        avg_z_norm = np.mean(epoch_z_norm) if epoch_z_norm else 0.0
        avg_attn_std = np.mean(epoch_attn_std) if epoch_attn_std else 0.0
        avg_pred_mean = np.mean(epoch_pred_mean) if epoch_pred_mean else 0.0

        true_std_ref = 1.0 

        #print(f"Epoch {epoch:02d} Summary: Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | "
        #    f"Top@20%: {avg_topk_20:.4f} ")
        

        # Component dominance
        #if epoch_model_residual_std:
        #    res_dominance = np.mean(epoch_model_residual_std) / (np.mean(epoch_ptd_baseline_std) + 1e-6)
        #    print(f"\nComponent Analysis:")
        #    print(f"  Residual dominance ratio: {res_dominance:.3f}")
        #    print(f"  {'PTD dominant' if res_dominance < 0.3 else 'Balanced' if res_dominance < 3 else 'Residual dominant'}")
        
        
        # Scale matching
        #print(f"\nScale Matching:")
        #print(f"  Prediction STD: {avg_pred_std:.3f} (target: 1.0)")
        #print(f"  STD ratio: {avg_pred_std:.3f}")
        #print(f"  Mean shift: {avg_pred_mean:.3f} (target: ~0)")

        #print("\n" + "="*50)
        #print(f"EPOCH {epoch:02d} LOSS DIAGNOSTICS")
        #print("="*50)
        #print(f"  Avg Total Loss:         {avg_total_loss:.6f}")
        #print("-" * 50)
        #print(f"  LOSS WEIGHTED CONTRIBUTIONS:")
        #print(f"  ├─ Contrastive:  {avg_cl:.6f}, alpha={alpha_cl:.6f} ")
        #print(f"  ├─ Rank Loss (on reg_loss_combined):  {avg_rank_loss:.6f} ")
        #print("-" * 50)
        #print(f"  RESIDUAL SYSTEM DIAGNOSTICS:")
        #print(f"  └─ GNN Residual -> Mean: {avg_model_residual_mean:+.4f}, Std: {avg_model_residual_std:.4f}")

        ##print("="*50 + "\n")
        #print(f"-------------------------------------------------")
        #print(f"Pred STD (Target {true_std_ref:.4f}): {avg_pred_std:.4f}")
        #print(f"Pred Mean: {avg_pred_mean:.4f}")
        #print(f"Avg Z Norm (Goal 1.0000): {avg_z_norm:.4f}") # Diagnostic Check
        #print(f"Avg Attn STD (Goal High): {avg_attn_std:.4f}") # NEW

        print("-------------------------------------------------")


def eval_raw_ptd(src, ts, real_values, ptd):
    """
    Evaluates prediction metrics on raw values.
    
    Args:
        src: Source variable (not used in calculations).
        ts: Time series variable (not used in calculations).
        real_values (list or torch.Tensor): The actual values.
        ptd (list or torch.Tensor): The predicted values.
        
    Returns:
        dict: A dictionary containing MAE metrics and Spearman's correlation.
    """
    # Convert inputs to NumPy arrays for consistent operations
    real_values = np.array(real_values)
    ptd = np.array(ptd)
    
    if len(real_values) != len(ptd):
        raise ValueError("Input arrays must have the same length.")

    # Calculate MAE metrics
    abs_err = np.abs(real_values - ptd)
    mae = np.mean(abs_err)
    min_ae = np.min(abs_err)
    max_ae = np.max(abs_err)

    # Calculate Spearman's correlation
    try:
        spearman_corr, _ = spearmanr(real_values, ptd)
    except ValueError:
        # Handles cases where there is no variance in the data, e.g., all same values
        spearman_corr = np.nan

    # Store all metrics in a dictionary
    metrics = {
        "MAE": mae,
        "Spearman_Correlation": spearman_corr
    }
    print(metrics)

if not testing:
    training_model_27_gemini_CONTR_3_debug()
#Save Model
if not testing:
    print("Running in training mode...")
    torch.save(MLP_model.state_dict(), './saved_models/model_MLP_2.pth')
    torch.save(tgnn_model.state_dict(), './saved_models/model_TGAT_2.pth')


#eval_raw_ptd(nodeList_test_real, test_real_ts_list, test_label_l_real, test_pass_through_d)

e_time = eval_real_data_27_gemini('test for real data', tgnn_model, MLP_model, test_real_ngh_finder, nodeList_test_real, test_real_ts_list, test_label_l_real)
