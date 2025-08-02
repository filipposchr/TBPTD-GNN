import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import torch
from graph import NeighborFinder
from utils import temporal_adjacency_list, pass_through_degree, compute_earliest_arrival, compute_temporal_out_reach, compute_temporal_degrees

def load_real_data(dataName, mode_type, mode_value):
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

    ngh_finder = NeighborFinder(adj_list, adj_list_out, uniform=False)

    temporal_edges = list(zip(src_list, dst_list, ts_list))

    pass_through_d = None
    earl_arrival   = None
    out_degree     = None
    out_reach      = None

    if mode_type == 'bet':
        pass_through_d = pass_through_degree(temporal_edges, num_nodes)
        pass_through_d = torch.tensor(pass_through_d, dtype=torch.float32)

        if mode_value == 'sfm':
            earl_arrival = compute_earliest_arrival(num_nodes, src_list, dst_list, ts_list)
            earl_arrival = torch.tensor(earl_arrival, dtype=torch.float32)

    elif mode_type == 'close':
        if mode_value == 'f':
            out_degree, _ = compute_temporal_degrees(temporal_edges, num_nodes)
            out_degree = torch.tensor(out_degree, dtype=torch.float32)
        elif mode_value == 'sh':
            out_reach = compute_temporal_out_reach(num_nodes, src_list, dst_list, ts_list)
            out_reach = torch.tensor(out_reach, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown closeness mode: {mode_value}")


    return src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder, pass_through_d, earl_arrival, out_degree, out_reach

def load_train_real_data(UNIFORM,  mode_type, mode_value, save_dir="graph_features"):
    src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder = [], [], [], [], [], [], []

    pass_through_d_list = []
    earl_arrival_list = []
    out_degree_list = []
    out_reach_list = []

    pass_through_d = None
    earl_arrival = None
    out_degree = None
    out_reach = None


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

    train_real_datasets = [f"temporal_ER_graph_{i}_seq" for i in range(44)]

    print("Total training graphs : ", len(train_real_datasets))


    save_all_graph_features(train_real_datasets,mode_type, mode_value, save_dir="graph_features")

    for dataset_name in train_real_datasets:
        # Load the dataset
        #g_df = pd.read_csv(f'./data/train/Real/processed/seq/ml_{dataset_name}.csv')
        g_df = pd.read_csv(f'./data/train/Real/processed/ER_15k/ml_edit-{dataset_name}.csv')

        src_list.append(g_df.u.values)
        dst_list.append(g_df.i.values)
        ts_list.append(g_df.ts.values)
        max_idx = max(g_df.u.values.max(), g_df.i.values.max())

        # Load precomputed features from pickle
        file_path = os.path.join(save_dir, f"{dataset_name}_features.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                graph_features = pickle.load(f)

        if mode_type == "bet":
            pass_through_d = graph_features.get("pass_through_d")
            pass_through_d_list.append(pass_through_d)
            if mode_value == "sfm":
                earl_arrival = graph_features.get("earl_arrival")
                earl_arrival_list.append(earl_arrival)
        elif mode_type == "close":
            if mode_value == "f":
                out_degree = graph_features.get("out_degree")
                out_degree_list.append(out_degree)
            elif mode_value == "sh":
                out_reach = graph_features.get("out_reach")
                out_reach_list.append(out_reach)
            else:
                raise ValueError(f"Unknown closeness mode: {mode_value}")
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")


        # Populate adjacency list for NeighborFinder
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_list[-1], dst_list[-1], g_df.idx.values, ts_list[-1]):
            adj_list[dst].append((src, eidx, ts))

        adj_list_out = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_list[-1], dst_list[-1], g_df.idx.values, ts_list[-1]):
            adj_list_out[src].append((dst, eidx, ts))

        # Add graph-specific details
        node_count.append(len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))))

        node_list.append(np.unique(np.hstack([src_list[-1], dst_list[-1]])))
        maxTime_list.append(max(g_df.ts.values))
        ngh_finder.append(NeighborFinder(adj_list, adj_list_out, uniform=UNIFORM))

    return (src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder, pass_through_d_list, earl_arrival_list, out_degree_list, out_reach_list)


def load_real_true(dataName, mode_type, mode_value):
    """
    mode_type: either 'bet' or 'close'
    mode_value: the mode string, e.g. 'sh', 'sfm', 'f'
    """
    if mode_type == 'bet':
        if mode_value == 'sh':
            print("  Temporal Shortest Betweenness...")
            path = f'./data/test/Real/scores/graph_{dataName}_bet.txt'
        elif mode_value == 'sfm':
            print("  Temporal Shortest-Foremost Betweenness...")
            path = f'./data/test/Real/shf-bc_scores/graph_{dataName}_shf_bet.txt'
        else:
            raise ValueError(f"Unknown betweenness mode: {mode_value}")
    elif mode_type == 'close':
        if mode_value == 'f':
            print("  Temporal Fastest Closeness...")
            path = f'./data/test/Real/f-cl_scores/graph_{dataName}_cf.txt'
        elif mode_value == 'sh':
            print("  Temporal Shortest Closeness...")
            path = f'./data/test/Real/sh-cl_scores/graph_{dataName}_shc.txt'
        else:
            raise ValueError(f"Unknown closeness mode: {mode_value}")
    else:
        raise ValueError(f"Unknown mode type: {mode_type}")


    g_df = pd.read_csv(path, names=['node_id', 'score'], sep=' ')
    test_nodeList = g_df['node_id'].astype(int).tolist()
    test_List = g_df['score'].tolist()

    return test_nodeList, test_List


def load_real_train_true(mode_type, mode_value):
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

    train_real_datasets = [f"sparse_temporal_ER_graph_{i}_seq" for i in range(44)]

    for index in range(len(train_real_datasets)):
        dataset_name = train_real_datasets[index]

        if mode_type == 'bet':
            if mode_value == 'sh':
                #path = f'./data/train/Real/scores/bc_scores/{dataset_name}_bc.txt'
                path = f'./data/train/Real/scores/ER_sh_bet_scores/{dataset_name}_bc.txt'

            elif mode_value == 'sfm':
                path = f'./data/train/Real/scores/shf_scores/{dataset_name}_bc.txt'
            else:
                raise ValueError(f"Unknown betweenness mode: {mode_value}")
        elif mode_type == 'close':
            if mode_value == 'f':
                path = f'./data/train/Real/scores/close_fast_scores/{dataset_name}_cl.txt'
            elif mode_value == 'sh':
                path = f'./data/train/Real/scores/close_sh_scores/{dataset_name}_sh_cl.txt'
            else:
                raise ValueError(f"Unknown closeness mode: {mode_value}")

        else:
            raise ValueError(f"Unknown mode type: {mode_type}")

        g_df = pd.read_csv(path, names=['node_id', 'score'], sep=' ')

        nodeList = g_df['node_id'].astype(int).tolist()
        scoreList = g_df['score'].tolist()

        train_nodeList.append(nodeList)
        train_true.append(scoreList)

    return train_nodeList, train_true


def preprocess_data(csv_file):
    """
    Reads a temporal graph from a CSV file and returns edge_index and edge_time tensors.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        edge_index (torch.Tensor): Shape [2, num_edges], source and destination nodes.
        edge_time (torch.Tensor): Shape [num_edges], timestamps for each edge.
    """
    # Read the CSV file, skip the first row (header), and use only necessary columns
    df = pd.read_csv(csv_file, skiprows=1, header=None, usecols=[1, 2, 3])

    # Extract the source, destination, and time columns
    source_nodes = df.iloc[:, 0].tolist()
    destination_nodes = df.iloc[:, 1].tolist()
    timestamps = df.iloc[:, 2].tolist()

    # Convert to PyTorch tensors
    edge_index = torch.tensor([source_nodes, destination_nodes], dtype=torch.long)
    edge_time = torch.tensor(timestamps, dtype=torch.float)

    return edge_index, edge_time

def save_all_graph_features(train_real_datasets, mode_type, mode_value, save_dir="graph_features"):
    os.makedirs(save_dir, exist_ok=True)
    graph_features = {
        "pass_through_d": None,
        "earl_arrival": None,
        "out_degree": None,
        "out_reach": None
    }

    for dataset_name in train_real_datasets:
        file_path = os.path.join(save_dir, f"{dataset_name}_features.pkl")

        #g_df = pd.read_csv(f'./data/train/Real/processed/seq/ml_{dataset_name}.csv')
        g_df = pd.read_csv(f'./data/train/Real/processed/ER_15k/ml_edit-{dataset_name}.csv')

        src, dst, ts = g_df['u'].values, g_df['i'].values, g_df['ts'].values
        num_nodes = len(set(np.unique(np.hstack([src, dst]))))

        temporal_edges = list(zip(src, dst, ts))

        if mode_type == 'bet':
            pass_through_d = pass_through_degree(temporal_edges, num_nodes)
            pass_through_d = torch.tensor(pass_through_d, dtype=torch.float32)
            graph_features["pass_through_d"] = pass_through_d

            if mode_value == 'sfm':
                earl_arrival = compute_earliest_arrival(num_nodes, src, dst, ts)
                earl_arrival = torch.tensor(earl_arrival, dtype=torch.float32)
                graph_features["earl_arrival"] = earl_arrival

        elif mode_type == 'close':
            if mode_value == 'f':
                out_degree, _ = compute_temporal_degrees(temporal_edges, num_nodes)
                out_degree = torch.tensor(out_degree, dtype=torch.float32)
                graph_features["out_degree"] = out_degree

            elif mode_value == 'sh':
                out_reach = compute_temporal_out_reach(num_nodes, src, dst, ts)
                out_reach = torch.tensor(out_reach, dtype=torch.float32)
                graph_features["out_reach"] = out_reach

            else:
                raise ValueError(f"Unknown closeness mode: {mode_value}")
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")

        with open(file_path, "wb") as f:
            pickle.dump(graph_features, f)
