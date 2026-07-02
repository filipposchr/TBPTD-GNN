import pandas as pd
import numpy as np
import pickle
import os
import torch
from graph import NeighborFinder
from utils import pass_through_degree
from typing import Iterable, Tuple, List

def load_real_data(dataName):
    g_df = pd.read_csv('./data/test/Real/processed/seq/ml_{}.csv'.format(dataName))
    print(f"Testing dataset: {dataName}")

    src, dst, ts = g_df['u'].values, g_df['i'].values, g_df['ts'].values

    num_nodes = len(set(np.unique(np.hstack([src, dst]))))

    src_list = g_df.u.values
    dst_list = g_df.i.values
    ts_list = g_df.ts.values

    max_idx = max(g_df.u.values.max(), g_df.i.values.max())
    node_count = len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values]))))
    node_list = np.unique(np.hstack([src_list, dst_list]))
    maxTime_list = max(g_df.ts.values)

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_list, dst_list, g_df.idx.values, ts_list):
        adj_list[dst].append((src, eidx, ts))

    adj_list_out = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_list, dst_list, g_df.idx.values, ts_list):
        adj_list_out[src].append((dst, eidx, ts))

    for ngh_list in adj_list_out:
        ngh_list.sort(key=lambda x: x[2])

    ngh_finder = NeighborFinder(adj_list, uniform=False)

    temporal_edges = list(zip(src_list, dst_list, ts_list))

    pass_through_d = None

    temporal_edges_1b = list(zip(src_list.tolist(), dst_list.tolist(), ts_list.tolist()))
    ptd_index  = PTDIndex(temporal_edges_1b, max_idx)
    ptd_cache  = {}

    ptd_total_1b = ptd_index.total
    pass_through_d_t = torch.tensor(ptd_total_1b[1:], dtype=torch.float32)

    pass_through_d = pass_through_degree(temporal_edges, num_nodes)
    pass_through_d = torch.tensor(pass_through_d, dtype=torch.float32)

    return src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder, num_nodes, pass_through_d, pass_through_d_t, ptd_index, ptd_cache

class PTDIndex:
    def __init__(self, temporal_edges_1b: Iterable[Tuple[int,int,float]], num_nodes_1b: int):
        self.N = int(num_nodes_1b)
        T_in:  List[List[float]] = [[] for _ in range(self.N + 1)]
        T_out: List[List[float]] = [[] for _ in range(self.N + 1)]
        for s, d, ts in temporal_edges_1b:
            if 1 <= d <= self.N: T_in[d].append(ts)
            if 1 <= s <= self.N: T_out[s].append(ts)

        self.T_out  = [np.asarray(sorted(T_out[u]), dtype=np.float64)  for u in range(self.N + 1)]
        self.prefix = [np.empty(0, dtype=np.int64) for _ in range(self.N + 1)]
        self.total  = np.zeros(self.N + 1, dtype=np.int64)

        for u in range(1, self.N + 1):
            tin  = np.asarray(sorted(T_in[u]), dtype=np.float64)
            tout = self.T_out[u]
            if tout.size == 0:
                continue
            counts = np.searchsorted(tin, tout, side="left").astype(np.int64)
            pref = counts.cumsum()
            self.prefix[u] = pref
            self.total[u]  = int(pref[-1])

    def snapshot_partition_all(self, t: float):
        past  = np.zeros(self.N + 1, dtype=np.int64)
        fut   = np.zeros(self.N + 1, dtype=np.int64)
        total = self.total.copy()
        for u in range(1, self.N + 1):
            tout = self.T_out[u]
            if tout.size == 0:
                continue
            split = np.searchsorted(tout, t, side="left")
            pref  = self.prefix[u]
            p     = int(pref[split-1]) if split > 0 else 0
            past[u] = p
            fut[u]  = int(total[u] - p)
        return past, fut, total


def load_train_real_data(UNIFORM, save_dir="graph_features"):
    src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder = [], [], [], [], [], [], []

    pass_through_d_list = []
    pass_through_d = None

    train_real_datasets = ['edit-mrwiktionary', 'edit-siwiktionary', 'edit-stwiktionary', 'edit-wowiktionary',
                           'edit-tkwiktionary', 'edit-aywiktionary', 'edit-anwiktionary', 'edit-pawiktionary',
                           'edit-iawiktionary', 'edit-sowiktionary', 'edit-tiwiktionary', 'edit-sswiktionary',
                           'edit-gnwiktionary', 'edit-iewiktionary', 'edit-pnbwiktionary', 'edit-gdwiktionary',
                           'edit-srwikiquote', 'edit-nowikiquote', 'edit-etwikiquote',
                           'edit-jawikiquote', 'edit-mtwiktionary', 'edit-dvwiktionary', 'edit-iuwiktionary',
                           'edit-kuwikiquote', 'edit-suwiktionary', 'edit-nawiktionary', 'edit-miwiktionary',
                           'edit-roa_rupwiktionary', 'edit-tpiwiktionary', 'edit-gdwiktionary',
                           'edit-lnwiktionary', 'edit-omwiktionary', 'edit-sgwiktionary', 'edit-quwiktionary',
                           'edit-rwwiktionary', 'edit-stwikipedia', 'edit-olowikipedia', 'edit-tnwikipedia',
                           'edit-ffwikipedia', 'edit-dzwikipedia', 'edit-tyvwikipedia',
                           'edit-xhwikipedia',  'edit-tswikipedia', 'edit-bgwikiquote',
                            'edit-idwikiquote', 'edit-aswikiquote', 'edit-yiwikiquote', 'edit-sawikiquote']
   
    save_all_graph_features(train_real_datasets, save_dir="graph_features")

    ptd_indices = []

    for dataset_name in train_real_datasets:
        g_df = pd.read_csv(f'./data/train/Real/processed/seq/ml_{dataset_name}.csv')
        src_list.append(g_df.u.values)
        dst_list.append(g_df.i.values)
        ts_list.append(g_df.ts.values)
        max_idx = max(g_df.u.values.max(), g_df.i.values.max())

        file_path = os.path.join(save_dir, f"{dataset_name}_features.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                graph_features = pickle.load(f)


        num_nodes_j = int(max(g_df.u.max(), g_df.i.max()))
        temporal_edges_1b = list(zip(g_df.u.values.astype(int),
                                    g_df.i.values.astype(int),
                                    g_df.ts.values.astype(float)))
        ptd_indices.append(PTDIndex(temporal_edges_1b, num_nodes_j))

        pass_through_d = graph_features.get("pass_through_d")
        pass_through_d_list.append(pass_through_d)

        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_list[-1], dst_list[-1], g_df.idx.values, ts_list[-1]):
            adj_list[dst].append((src, eidx, ts))

        adj_list_out = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_list[-1], dst_list[-1], g_df.idx.values, ts_list[-1]):
            adj_list_out[src].append((dst, eidx, ts))

        node_count.append(len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))))

        node_list.append(np.unique(np.hstack([src_list[-1], dst_list[-1]])))
        maxTime_list.append(max(g_df.ts.values))
        #ngh_finder.append(NeighborFinder(adj_list, adj_list_out, uniform=UNIFORM))
        ngh_finder.append(NeighborFinder(adj_list, uniform=UNIFORM))

    return (src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder, pass_through_d_list, ptd_indices)


def load_real_true(dataName):
    path = f'./data/test/Real/scores/graph_{dataName}_bet.txt'
    g_df = pd.read_csv(path, names=['node_id', 'score'], sep=' ')
    test_nodeList = g_df['node_id'].astype(int).tolist()
    test_List = g_df['score'].tolist()

    return test_nodeList, test_List


def load_real_train_true():
    train_nodeList, train_true = [], []

    train_real_datasets = ['edit-mrwiktionary', 'edit-siwiktionary', 'edit-stwiktionary', 'edit-wowiktionary',
                           'edit-tkwiktionary', 'edit-aywiktionary', 'edit-anwiktionary', 'edit-pawiktionary',
                           'edit-iawiktionary', 'edit-sowiktionary', 'edit-tiwiktionary', 'edit-sswiktionary',
                           'edit-gnwiktionary', 'edit-iewiktionary', 'edit-pnbwiktionary', 'edit-gdwiktionary',
                           'edit-srwikiquote', 'edit-nowikiquote', 'edit-etwikiquote',
                           'edit-jawikiquote', 'edit-mtwiktionary', 'edit-dvwiktionary', 'edit-iuwiktionary',
                           'edit-kuwikiquote', 'edit-suwiktionary', 'edit-nawiktionary', 'edit-miwiktionary',
                           'edit-roa_rupwiktionary', 'edit-tpiwiktionary', 'edit-gdwiktionary',
                           'edit-lnwiktionary', 'edit-omwiktionary', 'edit-sgwiktionary', 'edit-quwiktionary',
                           'edit-rwwiktionary', 'edit-stwikipedia', 'edit-olowikipedia', 'edit-tnwikipedia',
                           'edit-ffwikipedia', 'edit-dzwikipedia', 'edit-tyvwikipedia',
                           'edit-xhwikipedia',  'edit-tswikipedia', 'edit-bgwikiquote',
                            'edit-idwikiquote', 'edit-aswikiquote', 'edit-yiwikiquote', 'edit-sawikiquote']

    for index in range(len(train_real_datasets)):
        dataset_name = train_real_datasets[index]
        

        path = f'./data/train/Real/scores/bc_scores/{dataset_name}_bc.txt'
        g_df = pd.read_csv(path, names=['node_id', 'score'], sep=' ')

        nodeList = g_df['node_id'].astype(int).tolist()
        scoreList = g_df['score'].tolist()

        train_nodeList.append(nodeList)
        train_true.append(scoreList)

    return train_nodeList, train_true

def preprocess_data(csv_file):
    df = pd.read_csv(csv_file, skiprows=1, header=None, usecols=[1, 2, 3])

    source_nodes = df.iloc[:, 0].tolist()
    destination_nodes = df.iloc[:, 1].tolist()
    timestamps = df.iloc[:, 2].tolist()

    edge_index = torch.tensor([source_nodes, destination_nodes], dtype=torch.long)
    edge_time = torch.tensor(timestamps, dtype=torch.float)

    return edge_index, edge_time

def save_all_graph_features(train_real_datasets, save_dir="graph_features"):
    os.makedirs(save_dir, exist_ok=True)
    graph_features = {
        "pass_through_d": None,
    }

    for dataset_name in train_real_datasets:
        file_path = os.path.join(save_dir, f"{dataset_name}_features.pkl")
        g_df = pd.read_csv(f'./data/train/Real/processed/seq/ml_{dataset_name}.csv')

        src, dst, ts = g_df['u'].values, g_df['i'].values, g_df['ts'].values
        num_nodes = len(set(np.unique(np.hstack([src, dst]))))

        temporal_edges = list(zip(src, dst, ts))

        pass_through_d = pass_through_degree(temporal_edges, num_nodes)
        pass_through_d = torch.tensor(pass_through_d, dtype=torch.float32)
        graph_features["pass_through_d"] = pass_through_d

        with open(file_path, "wb") as f:
            pickle.dump(graph_features, f)
