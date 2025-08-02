import functools
import numpy as np
import torch
import multiprocessing
import numba
from numba import njit, jit

class NeighborFinder:
    def __init__(self, adj_list, adj_list_out=None, uniform=False):
        """
        Params
        ------
        adj_list: List[List[Tuple[int, int, float]]]
            The adjacency list for in-neighbors.

        adj_list_out: Optional[List[List[Tuple[int, int, float]]]]
            The adjacency list for out-neighbors. Optional.

        uniform: bool
            Whether to sample uniformly when selecting neighbors.
        """

        # Initialize IN neighbors
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l

        self.uniform = uniform

        # Initialize OUT neighbors if provided
        if adj_list_out is not None:
            (
                node_idx_l_out,
                node_ts_l_out,
                edge_idx_l_out,
                off_set_l_out
            ) = self.init_off_set(adj_list_out)

            self.node_idx_l_out = node_idx_l_out
            self.node_ts_l_out = node_ts_l_out
            self.edge_idx_l_out = edge_idx_l_out
            self.off_set_l_out = off_set_l_out
        else:
            # If no out-adjacency list provided, set attributes to None
            self.node_idx_l_out = None
            self.node_ts_l_out = None
            self.edge_idx_l_out = None
            self.off_set_l_out = None

    def init_off_set(self, adj_list):
        """
        Converts adjacency list into flat arrays for fast lookup.

        Params
        ------
        adj_list: List[List[Tuple[int, int, float]]]

        Returns
        -------
        node_idx_l, node_ts_l, edge_idx_l, off_set_l
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            off_set_l.append(len(n_idx_l))

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def preprocess(self, src_idx_l, cut_time_l, layer=2, num_neighbors=20):
        if layer == 0:
            return 1
        src_ngh_node_batch, src_ngh_t_batch = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
        self.preprocess(tuple(src_ngh_node_batch_flat), tuple(src_ngh_t_batch_flat), layer - 1, num_neighbors)

    #@functools.lru_cache(maxsize=None, typed=True)
    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        off_set_l = self.off_set_l
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts

        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_ts[:left]

    #@functools.lru_cache(maxsize=None, typed=True)
    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """

        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                # 1.uniform sampling:
                # sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                #
                # out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                # out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                #
                # # resort based on time
                # pos = out_ngh_t_batch[i, :].argsort()
                # out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                # out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]

                # 2.Recent interaction sampling:
                # ngh_idx = ngh_idx[-num_neighbors:]
                # ngh_ts = ngh_ts[-num_neighbors:]

                # 3.Farthest interaction sampling:
                # ngh_idx = ngh_idx[:num_neighbors]
                # ngh_ts = ngh_ts[:num_neighbors]

                # Expanded neighbor sampling:
                # if len(ngh_idx) > num_neighbors:
                #     ngh_idx, ngh_ts = self.evenly_sample_increasing_sequence(ngh_idx, ngh_ts, num_neighbors)
                # else:
                #     ngh_idx = ngh_idx[:num_neighbors]
                #     ngh_ts = ngh_ts[:num_neighbors]

                # Degree-based sampling:
                if len(ngh_idx) > num_neighbors:
                    # degree_batch = np.diff(self.off_set_l[ngh_idx])
                    degree_batch = self.off_set_l[ngh_idx + 1] - self.off_set_l[ngh_idx]
                    # for m, k in enumerate(ngh_idx):
                    #     degree_batch[m] = self.off_set_l[k + 1] - self.off_set_l[k]
                    neighbors = np.argsort(degree_batch)[-num_neighbors:]
                    ngh_idx = ngh_idx[neighbors]
                    ngh_ts = ngh_ts[neighbors]

                assert (len(ngh_idx) <= num_neighbors)
                assert (len(ngh_ts) <= num_neighbors)

                out_ngh_node_batch[i, -len(ngh_idx):] = ngh_idx
                out_ngh_t_batch[i, -len(ngh_ts):] = ngh_ts

                indice = num_neighbors - len(ngh_idx)
                out_ngh_node_batch[i, indice:] = ngh_idx
                out_ngh_t_batch[i, indice:] = ngh_ts

        return out_ngh_node_batch, out_ngh_t_batch

    def find_before_out(self, src_idx, cut_time):
        """
        Find out-neighbors of src_idx occurring at or after cut_time.
        """
        if self.node_idx_l_out is None:
            raise RuntimeError("Out adjacency list not loaded. Pass adj_list_out in the constructor.")

        node_idx_l = self.node_idx_l_out
        node_ts_l = self.node_ts_l_out
        off_set_l = self.off_set_l_out

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_ts) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        left = 0
        right = len(neighbors_ts)

        while left < right:
            mid = (left + right) // 2
            if neighbors_ts[mid] < cut_time:
                left = mid + 1
            else:
                right = mid

        # neighbors_ts[left:] should all be ≥ cut_time
        neighbors_idx = neighbors_idx[left:]
        neighbors_ts = neighbors_ts[left:]

        # filter just in case
        mask = neighbors_ts >= cut_time
        neighbors_idx = neighbors_idx[mask]
        neighbors_ts = neighbors_ts[mask]

        return neighbors_idx, neighbors_ts

    #@functools.lru_cache(maxsize=None, typed=True)
    def get_temporal_out_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        if self.node_idx_l_out is None:
            raise RuntimeError("Out adjacency list not loaded. Pass adj_list_out in the constructor.")

        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.full((len(src_idx_l), num_neighbors), 0, dtype=np.int32)

        cut_time_l_arr = np.array(cut_time_l, dtype=np.float32)
        out_ngh_t_batch = np.full(
            (len(src_idx_l), num_neighbors),
            cut_time_l_arr[:, None] + 1,
            dtype=np.float64
        )

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_ts = self.find_before_out(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if len(ngh_idx) > num_neighbors:
                    degree_batch = self.off_set_l_out[ngh_idx + 1] - self.off_set_l_out[ngh_idx]
                    neighbors = np.argsort(degree_batch)[-num_neighbors:]
                    ngh_idx = ngh_idx[neighbors]
                    ngh_ts = ngh_ts[neighbors]

                pad_len = num_neighbors - len(ngh_idx)
                if pad_len > 0:
                    pad_idx = np.zeros(pad_len, dtype=np.int32)
                    pad_ts = np.full(pad_len, cut_time + 1, dtype=np.float32)
                    ngh_idx = np.concatenate([pad_idx, ngh_idx])
                    ngh_ts = np.concatenate([pad_ts, ngh_ts])

                out_ngh_node_batch[i, :] = ngh_idx
                out_ngh_t_batch[i, :] = ngh_ts

            else:
                out_ngh_node_batch[i, :] = 0
                out_ngh_t_batch[i, :] = cut_time + 1


        return out_ngh_node_batch, out_ngh_t_batch

    def evenly_sample_increasing_sequence(self, idx_list, ts_list, num_samples):
        length = len(idx_list)
        interval = length / num_samples
        start_index = 0

        ngh_idx, ngh_ts = [], []
        for i in range(num_samples):
            index = int(start_index + i * interval)
            ngh_idx.append(idx_list[index])
            ngh_ts.append(ts_list[index])
        return ngh_idx, ngh_ts
