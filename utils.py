import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import random
from scipy.stats import  spearmanr, kendalltau
from typing import Iterable, Tuple, List, Dict, Optional
from collections import defaultdict
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import ndcg_score

def setSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_topk_metrics(preds, labels, k_list=[1, 10, 20, 30], jac=True):
    """
    Computes:
    - Top@k% precision = |pred ∩ true| / k
    - Recall@k% for positives = |pred ∩ true positives| / (# true positives)
    - Jaccard index between predicted and true positive sets
    """

    stats = {}
    N = len(preds)
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    true_positive_indices = set(torch.where(labels > 0)[0].tolist())
    num_true_positives = len(true_positive_indices)

    if num_true_positives == 0:
        for k in k_list:
            stats[f"Top@{k}%"] = 0.0
            stats[f"Recall@{k}%"] = 0.0
        if jac:
            stats["Jaccard"] = 0.0
        return stats

    result_line = []

    for k in k_list:
        topk = max(1, math.ceil(N * (k / 100)))
        pred_topk_indices = set(torch.topk(preds, topk).indices.tolist())
        true_topk_indices = set(torch.topk(labels, topk).indices.tolist())

        intersection = pred_topk_indices & true_topk_indices
        precision_at_k = len(intersection) / topk
        stats[f"Top@{k}%"] = precision_at_k

        intersection_with_true_pos = pred_topk_indices & true_positive_indices
        recall_at_k = len(intersection_with_true_pos) / num_true_positives
        stats[f"Recall@{k}%"] = recall_at_k

        result_line.append(f"Top@{k}%: {precision_at_k:.4f} | Recall@{k}%: {recall_at_k:.4f}")

    if jac:
        pred_full_set = set(torch.topk(preds, num_true_positives).indices.tolist())
        union = pred_full_set | true_positive_indices
        inter = pred_full_set & true_positive_indices
        jaccard = len(inter) / len(union)
        stats["Jaccard"] = jaccard
        result_line.append(f"Jaccard: {jaccard:.4f}")

    return stats

def build_temporal_adjacency(src_list, dst_list, ts_list):
    adj = defaultdict(list)
    for u, v, t in zip(src_list, dst_list, ts_list):
        adj[u].append((v, t))
    return adj

def edge_time_range(temporal_edges):
    temporal_edge_min = {}
    temporal_edge_max = {}

    for src, dst, ts in temporal_edges:
        key = (src, dst)
        if key not in temporal_edge_min or ts < temporal_edge_min[key]:
            temporal_edge_min[key] = ts
        if key not in temporal_edge_max or ts > temporal_edge_max[key]:
            temporal_edge_max[key] = ts

    sorted_min = sorted(temporal_edge_min.items(), key=lambda x: x[1])
    sorted_max = sorted(temporal_edge_max.items(), key=lambda x: x[1])

    return sorted_min, sorted_max

def count_less_than(arr: List[int], t: int) -> int:
    left, right = 0, len(arr) - 1
    pos = 0
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < t:
            pos = mid + 1
            left = mid + 1
        else:
            right = mid - 1
    return pos

def pass_through_degree(temporal_edges, num_nodes):
    sorted_min, sorted_max = edge_time_range(temporal_edges)

    min_arrival_times = [[] for _ in range(num_nodes)]
    for (src, dst), t in sorted_min:
        if dst < num_nodes:
            min_arrival_times[dst].append(t)

    ptd = np.zeros(num_nodes, dtype=int)
    for (src, dst), t in sorted_max:
        if src < num_nodes:
            ptd[src-1] += count_less_than(min_arrival_times[src], t)

    return ptd


class LabelNormalizer:
    """
    method = 'log1p'  -> y_n = log1p(alpha * y)
    method = 'zscore' -> y_n = (y - mu) / (sigma + eps)
    """
    def __init__(self, method: str = 'log1p', eps: float = 1e-8):
        assert method in ('log1p', 'zscore')
        self.method = method
        self.eps = eps
        self.alpha = None
        self.mu = None
        self.sigma = None

    def fit(self, y_all: torch.Tensor):
        y_all = y_all.detach().float().reshape(-1)
        if self.method == 'log1p':
            pos = y_all[y_all > 0]
            if pos.numel() == 0:
                self.alpha = 1.0
            else:
                med = torch.median(pos)
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

    def np_inverse(self, y_n_np):
        y_n = np.asarray(y_n_np, np.float32)
        if self.method == 'log1p':
            return np.maximum(np.expm1(y_n) / (self.alpha + self.eps), 0.0)
        else:
            return y_n * self.sigma + self.mu


def eval_statistics(pred, true, ptd_raw, dataset_name=""):
    pred = np.asarray(pred, dtype=float).flatten()
    true = np.asarray(true, dtype=float).flatten()
    ptd_raw = np.asarray(ptd_raw, dtype=float).flatten()

    min_len = min(len(pred), len(true), len(ptd_raw))
    pred = pred[:min_len]
    true = true[:min_len]
    ptd_raw = ptd_raw[:min_len]

    valid_mask = np.isfinite(pred) & np.isfinite(true) & np.isfinite(ptd_raw)
    pred = pred[valid_mask]
    true = true[valid_mask]
    ptd_raw = ptd_raw[valid_mask]

    n_points = len(pred)

    abs_err = np.abs(pred - true)    
    mae = float(abs_err.mean())
    min_ae = float(abs_err.min())
    max_ae = float(abs_err.max())
    std_pred = float(pred.std(ddof=1))
    std_true = float(true.std(ddof=1))
    #ptd raw
    abs_err_ptd = np.abs(ptd_raw - true)
    mae_ptd = float(abs_err_ptd.mean())
    min_ae_ptd = float(abs_err_ptd.min())
    max_ae_ptd = float(abs_err_ptd.max())
    std_ptd = float(ptd_raw.std(ddof=1))

    range_true = float(true.max() - true.min())
    iqr_true = float(np.percentile(true, 75) - np.percentile(true, 25))
    mean_abs_true = float(np.mean(np.abs(true)))
    eps = 1e-12

    # Normalized MAEs
    nmae_range = float(mae / range_true) if range_true > 0 else np.nan
    nmae_std   = float(mae / std_true)   if std_true   > 0 else np.nan
    nmae_iqr   = float(mae / iqr_true)   if iqr_true   > 0 else np.nan
    nmae_meanabs = float(mae / (mean_abs_true + eps))

    print(f"  STD(pred) = {std_pred:.6g}")
    print(f"  STD(true) = {std_true:.6g}")
    print(f"  MAE (mean abs error) = {mae:.6g}")
    print(f"  MAE (ptd)         = {mae_ptd:.6g} ")
    
    try:
        true_shifted = true - true.min() if true.min() < 0 else true

        ndcg_model_full = ndcg_score([true_shifted], [pred])
        ndcg_ptd_full   = ndcg_score([true_shifted], [ptd_raw])
        ndcg_diff_full  = ndcg_model_full - ndcg_ptd_full

        ndcg_results = {}
        for k in [10, 30, 50]:
            if k <= n_points:
                ndcg_model_k = ndcg_score([true_shifted], [pred],    k=k)
                ndcg_ptd_k   = ndcg_score([true_shifted], [ptd_raw], k=k)
                ndcg_diff_k  = ndcg_model_k - ndcg_ptd_k
                ndcg_results[k] = {
                    'model': ndcg_model_k,
                    'ptd':   ndcg_ptd_k,
                    'diff':  ndcg_diff_k,
                }

        print(f"  Full ranking — Model: {ndcg_model_full:.4f}, PTD: {ndcg_ptd_full:.4f}  "
              f"(diff: {ndcg_diff_full:+.4f})")
        for k, res in ndcg_results.items():
            marker = "Yes" if res['diff'] > 0 else ("No" if res['diff'] < -0.01 else "≈")
            print(f"  NDCG@{k:<3} — Model: {res['model']:.4f}, PTD: {res['ptd']:.4f}  "
                  f"(diff: {res['diff']:+.4f}) {marker}")

    except Exception as e:
        ndcg_model_full = ndcg_ptd_full = ndcg_diff_full = np.nan
        ndcg_results = {}

    try:
        kt_model_all, p_kt_model = kendalltau(pred, true)
        kt_ptd_all, p_kt_ptd = kendalltau(ptd_raw, true)
        kt_diff_all = kt_model_all - kt_ptd_all

        sp_model_all, p_sp_model = spearmanr(pred, true)
        sp_ptd_all, p_sp_ptd = spearmanr(ptd_raw, true)
        sp_diff_all = sp_model_all - sp_ptd_all

        stat, p_mae = wilcoxon(abs_err, abs_err_ptd, alternative='less')
        print(f"Wilcoxon Test for MAE: p = {p_mae:.2e}")
        if p_mae < 0.05:
            print("  Model MAE is STATISTICALLY BETTER than PTD")

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

        pred_log = np.log1p(np.clip(pred, 0, None))
        ptd_log = np.log1p(np.clip(ptd_raw, 0, None))
        true_log = np.log1p(np.clip(true, 0, None))

        sp_model_log = spearmanr(pred_log, true_log)[0]
        sp_ptd_log = spearmanr(ptd_log, true_log)[0]
        sp_diff_log = sp_model_log - sp_ptd_log

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

        if abs(kt_diff_all) < 0.01:
            kt_interp = "equivalent performance"
        elif kt_diff_all > 0:
            kt_interp = f"model BETTER by {kt_diff_all:.4f}"
        else:
            kt_interp = f"model WORSE by {abs(kt_diff_all):.4f}"
        print(f"WKT Difference: {kt_diff_all:+.4f} ({kt_interp})")

        print(f"\n Spearman Correlation:")
        print(f"  Model: ρ={sp_model_all:.4f}, p={p_sp_model:.2e} {'*' if p_sp_model < 0.05 else ''}")
        print(f"  PTD:   ρ={sp_ptd_all:.4f}, p={p_sp_ptd:.2e} {'*' if p_sp_ptd < 0.05 else ''}")

        if abs(sp_diff_all) < 0.01:
            sp_interp = "equivalent performance"
        elif sp_diff_all > 0:
            sp_interp = f"model BETTER by {sp_diff_all:.4f}"
        else:
            sp_interp = f"model WORSE by {abs(sp_diff_all):.4f}"
        print(f"      Spearman  Difference: {sp_diff_all:+.4f} ({sp_interp})")

        # Steiger's Test
        try:
            r_model_ptd = float(np.corrcoef(pred, ptd_raw)[0, 1])
            n = len(pred)

            r_yx = sp_model_all
            r_zx = sp_ptd_all
            r_yz = r_model_ptd

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

            print(f"\nSteiger's Test:")
            print(f"  Z = {z_steiger:.3f}, p = {p_steiger:.2e}")
            print(f"  Model-PTD correlation: r = {r_model_ptd:.3f}")
            if p_steiger < 0.05:
                print("  → Difference IS statistically significant")
            else:
                print("  → Difference is NOT statistically significant")

        except Exception as e:
            print(f"  Steiger's test failed: {e}")


        try:
            from scipy.stats import norm
            
            n = len(pred)
            pred_ranked = np.argsort(np.argsort(pred)).astype(float)
            ptd_ranked = np.argsort(np.argsort(ptd_raw)).astype(float)
            r_yz = float(np.corrcoef(pred_ranked, ptd_ranked)[0, 1])

            r_yx = sp_model_all
            r_zx = sp_ptd_all

            if abs(r_yz) >= 0.999:
                z_test = 0.0
                p_test = 1.0
            else:
                r_mean = (r_yx + r_zx) / 2
                f = (1 - r_yz) / (2 * (1 - r_mean**2))
                
                h = (1 - r_yz) / (2 * (1 - r_mean**2) * (1 - 0.5 * r_mean**2))
                
                if h <= 0:
                    z_test = 0.0
                    p_test = 1.0
                else:
                    z_test = (r_yx - r_zx) * np.sqrt((n - 3) / (2 * (1 - r_yz) * h))
                    p_test = 2 * (1 - norm.cdf(abs(z_test)))

            print(f"\nDependent Spearman Test (Hittner Modification):")
            print(f"  Z = {z_test:.3f}, p = {p_test:.2e}")
            print(f"  Rank Cross-correlation (r_model_ptd): r = {r_yz:.3f}")
            if p_test < 0.05:
                print("  Rank sorting difference IS statistically significant")
            else:
                print("  Rank sorting difference is NOT statistically significant")

        except Exception as e:
            print(f"  Rank significance test failed: {e}")

        return {
            'kendall_model': kt_model_all,
            'kendall_ptd': kt_ptd_all,
            'kendall_diff': kt_diff_all,
            'spearman_model': sp_model_all,
            'spearman_ptd': sp_ptd_all,
            'spearman_diff': sp_diff_all,
            'n_points': n_points,
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

    N = x.shape[0]
    K = int(max(1, min(K, N)))
    if tie_policy not in {"stable", "expand", "drop"}:
        raise ValueError("tie_policy must be one of {'stable','expand','drop'}")

    if tie_policy == "stable":
        order = np.argsort(-x, kind="mergesort")
        return order[:K]
    part_idx = np.argpartition(-x, K-1)[:K]
    kth_val = x[part_idx].min()
    if tie_policy == "drop":
        return part_idx
    else:
        mask = x >= kth_val
        idx = np.flatnonzero(mask)
        idx = idx[np.lexsort((idx, -x[idx]))]
        return idx

def hits_in_k(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    K: int,
    tie_policy: str = "stable",
    return_indices: bool = False,
) -> Tuple[int, float, Optional[np.ndarray], Optional[np.ndarray]]:
    
    yt = _to_numpy(y_true).astype(float).reshape(-1)
    yp = _to_numpy(y_pred).astype(float).reshape(-1)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    N = yt.shape[0]
    if N == 0:
        return 0, 0.0, None, None

    K = int(max(1, min(K, N)))

    true_top = _topk_indices(yt, K, tie_policy=tie_policy)
    pred_top = _topk_indices(yp, K, tie_policy=tie_policy)

    hits = int(len(set(true_top.tolist()).intersection(set(pred_top.tolist()))))
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

#--------------------Losses--------------------

class EnchRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, pred: torch.Tensor, true_label: torch.Tensor) -> torch.Tensor:
        B = pred.size(0)
        
        indices = torch.arange(B, device=pred.device)
        i_idx, j_idx = torch.meshgrid(indices, indices, indexing='ij')
        
        mask = (i_idx!= j_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        pred_i, pred_j = pred[i_idx], pred[j_idx]
        true_i, true_j = true_label[i_idx], true_label[j_idx]

        order_mask = (true_i > true_j)
        
        true_diff = (true_i - true_j)[order_mask]
        pred_diff = (pred_i - pred_j)[order_mask]
        
        ratio_norm = pred_diff / (true_diff.abs() + 1e-6) 
        penalty = self.relu(self.margin - ratio_norm)
        
        if penalty.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        return penalty.mean()


class AdaptiveReweightedSupConLoss(nn.Module):
    def __init__(self, temperature=0.1, gamma_pos=0.5, gamma_neg=0.5):
        super().__init__()
        self.temperature = temperature
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, z: torch.Tensor,
                tbc_classes: torch.Tensor,
                tbc_values: torch.Tensor):
        device = z.device
        B = z.size(0)
        if B <= 1:
            return z.new_tensor(0.0)

        z = F.normalize(z, dim=1)

        eye = torch.eye(B, dtype=torch.bool, device=device)

        same = (tbc_classes.unsqueeze(1) == tbc_classes.unsqueeze(0)) & ~eye
        diff = (tbc_classes.unsqueeze(1) != tbc_classes.unsqueeze(0)) & ~eye

        tbc_diff = (tbc_values.unsqueeze(1) - tbc_values.unsqueeze(0)).abs()
        if (tbc_values > 0).any():
            tbc_median = torch.median(tbc_values[tbc_values > 0])
        else:
            tbc_median = tbc_values.new_tensor(1.0)

        pos_mask = same & (tbc_diff > 0) & (tbc_diff <= self.gamma_pos * tbc_median)
        neg_mask = diff | (same & (tbc_diff >= self.gamma_neg * tbc_median))

        if not pos_mask.any():
            return z.new_tensor(0.0)

        sim = torch.matmul(z, z.t()) / self.temperature

        emb_dist_sq = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(dim=2)

        static_beta_pos = (tbc_median * self.gamma_pos) / (tbc_diff + 1e-8)
        static_beta_neg = tbc_diff / (tbc_median * self.gamma_neg + 1e-8)

        dyn_pos = emb_dist_sq
        dyn_neg = 1.0 / (emb_dist_sq + 1e-8)

        beta_pos = static_beta_pos * dyn_pos
        beta_neg = static_beta_neg * dyn_neg

        exp_sim = sim.exp()

        pos_term = (exp_sim * beta_pos * pos_mask).sum(dim=1)
        neg_term = (exp_sim * beta_neg * neg_mask).sum(dim=1)

        pos_term = pos_term + 1e-8
        neg_term = neg_term + 1e-8

        losses = -torch.log(pos_term / (pos_term + neg_term))

        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            return z.new_tensor(0.0)

        return losses[valid].mean()


#For Experiments (different loss modes)
class AdaptiveReweightedSupConLoss_modes(nn.Module):
    def __init__(self, temperature=0.1, gamma_pos=0.5, gamma_neg=0.5, ablation_mode='no_closeness'):
        super().__init__()
        self.temperature = temperature
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.ablation_mode = ablation_mode 
        # Modes: 
        # 'full'       -> Original adaptive dynamic + static
        # '1a', '1b', '1c' -> As previously defined
        # 'random_neg' -> Uses all other nodes in batch as negatives (standard InfoNCE style)
        # 'no_closeness' -> Positives are anyone in the same class, regardless of TBC value
        print("Mode:", self.ablation_mode)

    def forward(self, z, tbc_classes, tbc_values):
        device = z.device
        B = z.size(0)
        if B <= 1: return z.new_tensor(0.0)

        z = F.normalize(z, dim=1)
        eye = torch.eye(B, dtype=torch.bool, device=device)

        same = (tbc_classes.unsqueeze(1) == tbc_classes.unsqueeze(0)) & ~eye
        diff = (tbc_classes.unsqueeze(1) != tbc_classes.unsqueeze(0)) & ~eye
        tbc_diff = (tbc_values.unsqueeze(1) - tbc_values.unsqueeze(0)).abs()
        
        if (tbc_values > 0).any():
            tbc_median = torch.median(tbc_values[tbc_values > 0])
        else:
            tbc_median = tbc_values.new_tensor(1.0)

        if self.ablation_mode == '1c' or self.ablation_mode == 'no_closeness':
            pos_mask = same
            neg_mask = diff
        elif self.ablation_mode == 'random_neg':
            pos_mask = same & (tbc_diff > 0) & (tbc_diff <= self.gamma_pos * tbc_median)
            neg_mask = ~eye & ~pos_mask 
        else:
            pos_mask = same & (tbc_diff > 0) & (tbc_diff <= self.gamma_pos * tbc_median)
            neg_mask = diff | (same & (tbc_diff >= self.gamma_neg * tbc_median))

        if not pos_mask.any(): return z.new_tensor(0.0)

        sim = torch.matmul(z, z.t()) / self.temperature
        exp_sim = sim.exp()

        if self.ablation_mode == '1a':
            beta_pos = torch.ones_like(sim)
            beta_neg = torch.ones_like(sim)
        else:
            static_beta_pos = (tbc_median * self.gamma_pos) / (tbc_diff + 1e-8)
            static_beta_neg = tbc_diff / (tbc_median * self.gamma_neg + 1e-8)

            if self.ablation_mode == '1b' or self.ablation_mode == 'no_closeness' or self.ablation_mode == 'random_neg':
                beta_pos = static_beta_pos
                beta_neg = static_beta_neg
            else:
                emb_dist_sq = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(dim=2)
                dyn_pos = emb_dist_sq
                dyn_neg = 1.0 / (emb_dist_sq + 1e-8)
                beta_pos = static_beta_pos * dyn_pos
                beta_neg = static_beta_neg * dyn_neg

        pos_term = (exp_sim * beta_pos * pos_mask).sum(dim=1) + 1e-8
        neg_term = (exp_sim * beta_neg * neg_mask).sum(dim=1) + 1e-8

        losses = -torch.log(pos_term / (pos_term + neg_term))
        valid = pos_mask.sum(dim=1) > 0
        return losses[valid].mean() if valid.any() else z.new_tensor(0.0)
