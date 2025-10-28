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
from utils import compute_kendall_tau, compute_topk_metrics, compute_ptd_at_t, pass_through_degree_t, ptd_split_at_t
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from typing import Iterable, Tuple, Dict, List, Optional, Union 
from scipy import stats
from module import ConservativeSimplifiedModel_gemini_CONTR #CONTRASTIVE LEARNING FROM GEMINI

# Argument and global variables
parser = argparse.ArgumentParser('Interface for Experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=1500, help='batch_size')
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


class MLP(nn.Module):
    def __init__(self, input_dim=256, drop: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32) # Added hidden layer
        self.fc4 = nn.Linear(32, 1)  # Output layer
        

        self.act = nn.LeakyReLU(negative_slope=0.01)        
        self.dropout = nn.Dropout(p=drop)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu") 
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="leaky_relu")
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x))) # Apply activation and dropout to fc3

        return self.fc4(x).squeeze(-1)
    

MLP_model = MLP().to(device)
tgnn_model.to(device)


optimizer = torch.optim.AdamW([
    {"params": tgnn_model.parameters(), "lr": LEARNING_RATE},
    {"params": MLP_model.parameters(),  "lr": LEARNING_RATE},
], weight_decay=1e-4)


#Load Model
if testing:
    print("Running in test mode...")
    tgnn_model.load_state_dict(torch.load('./saved_models/model_TGAT_2.pth', weights_only=True))
    MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_2.pth', weights_only=True))


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
        y_n = np.asarray(y_n_np, np.float32)
        if self.method == 'log1p':
            return np.maximum(np.expm1(y_n) / (self.alpha + self.eps), 0.0)
        else:
            return y_n * self.sigma + self.mu


# change creation to two-stage
normalizer = LabelNormalizer(method='log1p')  # keep existing
all_train_labels = torch.tensor(np.concatenate(train_label_l_real), dtype=torch.float32)
normalizer.fit(all_train_labels)
with torch.no_grad():
    y_log = normalizer.torch_transform(all_train_labels)
mu = float(y_log.mean())
sigma = float(y_log.std() + 1e-8)


def unified_evaluation_with_statistics(pred, true, ptd_raw, dataset_name=""):

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

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
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

def safe_spearman_correlation(x, y, return_pvalue=False):
    """
    Safely compute Spearman correlation, handling constant arrays gracefully
    """
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
        

#---------------LOSSES----------------------------------------------------------
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
#--------------------------------------------------------------------------------


def training_model_28Oct_simpler():

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epoch, eta_min=LEARNING_RATE * 0.1
    )
    contrastive_loss_fn = AdaptiveReweightedSupConLoss(temperature=0.07)
    ranking_loss_fn = ImprovedRankingLoss(margin=1.0)

    # CL supervision thresholds in z-space
    TBC_CLASS_THRESHOLDS = [-0.5, 2.0]

    # CL ramp
    ALPHA_INITIAL = 0.001
    ALPHA_PEAK    = 0.010
    ALPHA_FINAL   = 0.020
    RAMPUP_EPOCHS = 15

    # Guard against overflow when mapping z -> raw
    MAX_Z = 8.0

    def safe_loss(loss_tensor: torch.Tensor, anchor_for_graph: torch.Tensor) -> torch.Tensor:
        # Ensure we always return a tensor on the right device, keeping autograd graph alive
        if not torch.is_tensor(loss_tensor):
            return (anchor_for_graph * 0.0).sum()
        if loss_tensor.requires_grad:
            return loss_tensor
        # degenerate: make a zero-like differentiable scalar
        return (anchor_for_graph * 0.0).sum()

    for epoch in range(NUM_EPOCH):
        # Alpha schedule
        if epoch < RAMPUP_EPOCHS:
            alpha_cl = ALPHA_INITIAL + (ALPHA_PEAK - ALPHA_INITIAL) * (epoch / RAMPUP_EPOCHS)
        else:
            remaining = max(1, NUM_EPOCH - RAMPUP_EPOCHS)
            decay_rate = (ALPHA_PEAK - ALPHA_FINAL) / remaining
            alpha_cl = max(ALPHA_FINAL, ALPHA_PEAK - (epoch - RAMPUP_EPOCHS) * decay_rate)

        # Trackers
        epoch_topk_1, epoch_topk_10, epoch_topk_20 = [], [], []
        epoch_cl_loss, epoch_rank_loss, epoch_raw_huber = [], [], []
        epoch_total_loss = []
        epoch_z_norm = []

        tgnn_model.train()
        MLP_model.train()

        graph_indices = list(range(len(train_real_ts_l)))
        for j in graph_indices:
            tgnn_model.ngh_finder = train_real_ngh_finder[j]

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

                # PTD feed
                ptd_past_1b, _, _ = ptd_indices[j].snapshot_partition_all(t_cut)
                tgnn_model.set_ptd_vector(ptd_past_1b)

                # ---- Forward TGNN
                src_embed = tgnn_model.tem_conv2(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    ptd_l=None,
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )

                # ---- Targets (z-space)
                true_raw = torch.tensor(label_l_cut, dtype=torch.float32, device=device)  # ≥ 0
                y_log = normalizer.torch_transform(true_raw)  # log1p/α inside
                mu_t = torch.as_tensor(mu,    dtype=y_log.dtype, device=device)
                sd_t = torch.as_tensor(sigma, dtype=y_log.dtype, device=device)
                true_z = (y_log - mu_t) / sd_t

                # ---- CL classes (in z-space)
                tbc_classes = torch.zeros_like(true_z, dtype=torch.long, device=device)
                med_mask  = (true_z >= TBC_CLASS_THRESHOLDS[0]) & (true_z < TBC_CLASS_THRESHOLDS[1])
                high_mask = (true_z >= TBC_CLASS_THRESHOLDS[1])
                tbc_classes[med_mask]  = 1
                tbc_classes[high_mask] = 2

                # ---- PTD encodings + MLP prediction (z-space)
                idx = torch.as_tensor(src_l_cut, device=device).long().clamp(min=1) - 1
                ptd_pair = tgnn_model.ptd_vec[idx]
                ptd_enc  = tgnn_model.ptd_mlp(ptd_pair)

                mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
                pred_z = MLP_model(mlp_in).squeeze(-1)

                # ---- Losses
                # Contrastive on z
                z = tgnn_model.contrast_head(src_embed)
                epoch_z_norm.append(z.norm(dim=1).mean().item())
                cl_loss = contrastive_loss_fn(z, tbc_classes, true_z)
                cl_loss = safe_loss(cl_loss, pred_z)

                # Ranking in z
                rank_loss = ranking_loss_fn(pred_z, true_z)
                rank_loss = safe_loss(rank_loss, pred_z)

                # Tiny raw-space Huber for MAE control (keep gradients!)
                pred_log = pred_z * sd_t + mu_t
                pred_log = torch.clamp(pred_log, max=mu_t + MAX_Z * sd_t)
                pred_raw = normalizer.torch_inverse(pred_log).clamp_min(0)
               
                reg_loss = rank_loss
                loss = alpha_cl * cl_loss + (1.0 - alpha_cl) * reg_loss

                # ---- Backward
                if torch.isnan(loss).any():
                    # skip pathological batches rather than crashing
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(tgnn_model.parameters()) + list(MLP_model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                # ---- Metrics
                with torch.no_grad():
                    # compute raw again (already computed) for top-k
                    topk_stats = compute_topk_metrics(
                        pred_raw, true_raw, k_list=[1, 5, 10, 20, 30], jac=False
                    )
                if not (topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0):
                    epoch_topk_1.append(topk_stats['Top@1%'])
                    epoch_topk_10.append(topk_stats['Top@10%'])
                    epoch_topk_20.append(topk_stats['Top@20%'])

                epoch_cl_loss.append(float(cl_loss.detach().cpu()))
                epoch_rank_loss.append(float(rank_loss.detach().cpu()))
                #epoch_raw_huber.append(float(raw_huber.detach().cpu()))
                epoch_total_loss.append(float(loss.detach().cpu()))

        scheduler.step()

        # ---- epoch summaries
        avg_topk_1  = float(np.mean(epoch_topk_1))  if epoch_topk_1 else 0.0
        avg_topk_10 = float(np.mean(epoch_topk_10)) if epoch_topk_10 else 0.0
        avg_topk_20 = float(np.mean(epoch_topk_20)) if epoch_topk_20 else 0.0
        print(f"Epoch {epoch:02d} | Top@1%={avg_topk_1:.4f} | Top@10%={avg_topk_10:.4f} | Top@20%={avg_topk_20:.4f} | "
              f"α={alpha_cl:.4f} | CL={np.mean(epoch_cl_loss) if epoch_cl_loss else 0:.4f} | "
              f"Rank={np.mean(epoch_rank_loss) if epoch_rank_loss else 0:.4f} | ")

def eval_real_28oct_simpler(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    tgan.ngh_finder = sampler

    lr_model = lr_model.eval()
    tgan = tgan.eval()

    MAX_Z = 8.0

    all_pred_raw = []
    all_true_raw = []

    TEST_BATCH_SIZE = BATCH_SIZE
    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    with torch.no_grad():
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            if e_idx - s_idx < 1:
                continue

            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut  = np.array(ts[s_idx:e_idx])

            # ----- PTD snapshot at the first time in the batch -----
            t_cut = float(test_ts_l_cut[0])
            ptd_past_1b = test_ptd_cache.get(t_cut)
            if ptd_past_1b is None:
                ptd_past_1b, _, _ = test_ptd_index.snapshot_partition_all(t_cut)  # [max_id+1], 1-based
                test_ptd_cache[t_cut] = ptd_past_1b
            tgan.set_ptd_vector(ptd_past_1b)

            # Safe 0-based indices
            test_src = np.array(src[s_idx:e_idx], dtype=np.int64)
            idx0 = np.clip(test_src, 1, test_num_nodes) - 1
            idx_t = torch.as_tensor(idx0, device=device).long()

            # True labels (raw space)
            true_batch = np.asarray(label, dtype=float)[idx0]
            true_raw = torch.as_tensor(true_batch, dtype=torch.float32, device=device)

            # ----- Forward TGNN + MLP (predict z)
            src_embed = tgan.tem_conv2(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                ptd_l=None,
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )

            idx = torch.as_tensor(test_src_l_cut, device=src_embed.device).long().clamp(min=1) - 1
            ptd_pair = tgan.ptd_vec[idx]              # [B, 2]
            ptd_enc  = tgan.ptd_mlp(ptd_pair)         # encoded PTD

            mlp_in = torch.cat([src_embed, ptd_enc], dim=1)
            pred_z = lr_model(mlp_in).squeeze(-1)     # normalized prediction (z-space)

            # ----- z -> raw (log-domain clamp for safety)
            mu_t = torch.as_tensor(mu,    dtype=pred_z.dtype, device=pred_z.device)
            sd_t = torch.as_tensor(sigma, dtype=pred_z.dtype, device=pred_z.device)
            pred_log = pred_z * sd_t + mu_t
            pred_log = torch.clamp(pred_log, max=mu_t + MAX_Z * sd_t)
            pred_raw = normalizer.torch_inverse(pred_log).clamp_min(0)

            # Collect
            all_pred_raw.append(pred_raw.detach().cpu().numpy().reshape(-1))
            all_true_raw.append(np.clip(true_batch, 0, None).reshape(-1))

    # ===== whole-graph arrays =====
    pred = np.concatenate(all_pred_raw, axis=0)
    true = np.concatenate(all_true_raw, axis=0)

    results = hits_in_ks(true, pred, Ks=[10, 30, 50])
    for K, (hits, pct) in results.items():
        print(f"Hits@{K}: {hits}/{K} ({pct:.2f}%)")

    unified_evaluation_with_statistics2(pred, true, test_pass_through_d, hint)

    e_time = (time.time() - start_time) / 60.0
    return e_time



if not testing:
    training_model_28Oct_simpler()
if not testing:
    torch.save(MLP_model.state_dict(), './saved_models/model_MLP_2.pth')
    torch.save(tgnn_model.state_dict(), './saved_models/model_TGAT_2.pth')

e_time = eval_real_28oct_simpler('test for real data', tgnn_model, MLP_model, test_real_ngh_finder, nodeList_test_real, test_real_ts_list, test_label_l_real)
