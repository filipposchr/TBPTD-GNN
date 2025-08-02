import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from scipy.stats import weightedtau, rankdata
import math

def rank_with_id_tiebreak(values):
    values = np.array(values)
    return rankdata(values, method='ordinal')  # unique ranks even with ties

def compute_kendall_tau(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    # Standard weighted KT over all nodes
    label_ranked_full = rank_with_id_tiebreak(labels)
    pred_ranked_full = rank_with_id_tiebreak(preds)
    kt_full, _ = weightedtau(pred_ranked_full, label_ranked_full)

    # Filtered weighted KT (only label > 0)
    nonzero_mask = labels > 0
    if np.sum(nonzero_mask) > 1:
        label_filtered = labels[nonzero_mask]
        pred_filtered = preds[nonzero_mask]

        label_ranked_filt = rank_with_id_tiebreak(label_filtered)
        pred_ranked_filt = rank_with_id_tiebreak(pred_filtered)
        kt_filt, _ = weightedtau(pred_ranked_filt, label_ranked_filt)
    else:
        kt_filt = 0.0  # can't compute with ≤1 item

    return kt_full, kt_filt


def temporal_adjacency_list(src_list, dst_list, ts_list, num_nodes):
    tal = [[] for _ in range(num_nodes + 1)]  # +1 for 1-based indexing
    # Populate the temporal adjacency list
    for src, dst, ts in zip(src_list, dst_list, ts_list):
        for u, v, t in zip(src, dst, ts):
            tal[u].append((v, t))

    return tal


def loss_cal(y_out, true_val, num_nodes, device):
    _, order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes * 80
    ind_1 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)

    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()

    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)

    return loss_rank

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

        # Top@k precision as originally computed
        intersection = pred_topk_indices & true_topk_indices
        precision_at_k = len(intersection) / topk
        stats[f"Top@{k}%"] = precision_at_k

        # NEW: Recall@k for positives
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

    print(" | ".join(result_line))
    return stats


def compute_topk_metrics_ptd(preds, labels, k_list=[1, 10, 20, 30], jac=True):
    """
    Computes:
    - Top@k% precision = |pred ∩ true| / k
    - Recall@k% for positives = |pred ∩ true positives| / (# true positives)
    - Jaccard index between predicted and true positive sets
    """

    stats = {}
    N = len(preds)
    preds = preds.detach().cpu()

    labels = torch.tensor(labels)  # convert list to tensor
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

        # Top@k precision as originally computed
        intersection = pred_topk_indices & true_topk_indices
        precision_at_k = len(intersection) / topk
        stats[f"Top@{k}%"] = precision_at_k

        # NEW: Recall@k for positives
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

    print(" | ".join(result_line))
    return stats
import heapq
from collections import defaultdict

def build_temporal_adjacency(src_list, dst_list, ts_list):
    adj = defaultdict(list)
    for u, v, t in zip(src_list, dst_list, ts_list):
        adj[u].append((v, t))
    return adj

def compute_earliest_arrival(num_nodes, src_list, dst_list, ts_list):
    adj = build_temporal_adjacency(src_list, dst_list, ts_list)

    in_deg = [0] * (num_nodes + 1)
    for v in dst_list:
        in_deg[v] += 1

    arrival_time = [float('inf')] * (num_nodes + 1)
    heap = []

    for u in range(1, num_nodes + 1):
        heapq.heappush(heap, (0, u))

    while heap:
        curr_time, u = heapq.heappop(heap)
        for v, t in adj.get(u, []):
            if t >= curr_time and t < arrival_time[v]:
                arrival_time[v] = t
                heapq.heappush(heap, (t, v))

    max_ts = max(ts_list)
    arrival_time = [min(t, max_ts) for t in arrival_time[1:]]

    return np.array(arrival_time, dtype=np.float32)


from collections import deque
from collections import defaultdict, deque
def normalized_supremum_deviation(pred, label):
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    label = (label - np.min(label)) / (np.max(label) - np.min(label))
    return np.abs(np.mean(pred) - np.mean(label))


def normalized_mae(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    range_label = np.max(label) - np.min(label) + 1e-8
    return np.mean(np.abs(pred - label)) / range_label


def edge_time_range(temporal_edges):
    """
    Args:
        temporal_edges: List of tuples (src, dst, timestamp)

    Returns:
        sorted_min: List of ((src, dst), min_ts) sorted by min_ts
        sorted_max: List of ((src, dst), max_ts) sorted by max_ts
    """
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

def compute_temporal_degrees(temporal_edges, num_nodes):
    out_deg = np.zeros(num_nodes, dtype=int)
    in_deg = np.zeros(num_nodes, dtype=int)

    for src, dst, t in temporal_edges:
        out_deg[src - 1] += 1
        in_deg[dst - 1] += 1

    return out_deg, in_deg


def compute_temporal_out_reach(num_nodes, src_list, dst_list, ts_list, max_hops=2):
    from collections import defaultdict, deque

    adj = defaultdict(list)
    for u, v, t in zip(src_list, dst_list, ts_list):
        adj[u].append((v, t))

    out_reach = np.zeros(num_nodes)

    for s in range(1, num_nodes + 1):
        visited = set()
        queue = deque()
        queue.append((s, -float('inf'), 0))

        while queue:
            u, t_curr, hops = queue.popleft()
            if u != s:
                visited.add(u)
            if hops >= max_hops:
                continue
            for v, t_next in adj.get(u, []):
                if t_next > t_curr:
                    queue.append((v, t_next, hops + 1))

        out_reach[s - 1] = len(visited)

    return out_reach
