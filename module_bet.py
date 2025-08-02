import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2)  # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])  # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1)  # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])  # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x lq x lk

        # Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3)  # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3)  # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


class TimeEncode(nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)
        return harmonic


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim

        # self.att_dim = feat_dim + edge_dim + time_dim
        self.att_dim = feat_dim + time_dim
        # self.att_dim = feat_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :]  # hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_t], dim=2)  # [B, N, Dt + D]
        hn = seq_x.mean(dim=1)  # [B, Dt + D]
        output = self.merger(hn, src_x)
        return output, None


class MultiHeadAttention2(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)
        # nn.init.kaiming_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        mask = mask.bool()

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)
        # nn.init.kaiming_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn


class AttnModel2(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        """
        args:
          feat_dim: dim for the node features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel2, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = (feat_dim + time_dim)
        # self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        # self.act = torch.nn.ReLU()

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention2(n_head,
                                                         d_model=256,
                                                         d_k=256 // n_head,
                                                         d_v=256 // n_head,
                                                         dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

        self.q_proj = nn.Linear(512, 256)

    def forward(self, src, src_t, seq, seq_t, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        # baseline query (shape [B, 1, 256]):
        # q = torch.cat([src_ext, src_t], dim=2)

        # new query (shape [B, 1, 512]):
        src_ext = src.unsqueeze(1)  # [B, 1, d]

        # repeat src to match in/out pairs
        src_ext_doubled = torch.cat([src_ext, src_ext], dim=2)  # [B, 1, 2d]
        src_t_doubled = torch.cat([src_t, src_t], dim=2)  # [B, 1, 2d_T]

        q = torch.cat([src_ext_doubled, src_t_doubled], dim=2)  # [B, 1, 512]

        k = torch.cat([seq, seq_t], dim=2)  # [B, 1, D + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        q = self.q_proj(q)
        k = self.q_proj(k)

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]

        # Only squeeze dimension 1
        if output.dim() == 3 and output.shape[1] == 1:
            output = output.squeeze(1)

        attn = attn.squeeze()

        # Check shape safety
        if output.shape[0] == 0 or src.shape[0] == 0:
            merged = torch.zeros((src.shape[0], self.merger.fc2.out_features), device=src.device)
        elif output.dim() < 2 or src.dim() < 2:
            merged = torch.zeros((src.shape[0], self.merger.fc2.out_features), device=src.device)
        else:
            merged = self.merger(output, src)

        return merged, attn


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        """
        args:
          feat_dim: dim for the node features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = (feat_dim + time_dim)
        # self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        # self.act = torch.nn.ReLU()

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_t], dim=2)  # [B, 1, D + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()  # output: [B, D]
        attn = attn.squeeze()  # weight: [B, N]

        output = self.merger(output, src)
        return output, attn


class TGNN_out_comp(nn.Module):
    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time',
                 agg_method='lstm', num_layers=3, n_head=4, null_idx=0, num_heads=2, drop_out=0.3, seq_len=None):
        super(TGNN_out_comp, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.feat_dim
        self.model_dim = 128

        self.path_proj = torch.nn.Linear(2 * self.model_dim, self.model_dim)
        self.path_time_proj = torch.nn.Linear(2 * self.model_dim, self.model_dim)

        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        self.attn_model_list = torch.nn.ModuleList([AttnModel2(self.feat_dim,
                                                               self.feat_dim,
                                                               attn_mode=attn_mode,
                                                               n_head=n_head,
                                                               drop_out=drop_out) for _ in range(num_layers)])

        # Time Encoding
        if use_time == 'pos':
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)
        elif use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('Invalid time encoding method!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=7):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=7, max_paths=50):
        """
        Temporal graph convolution using IN and OUT neighbors
        and valid temporal paths, with time separate from features.
        """
        assert curr_layers >= 0

        device = self.n_feat_th.device
        batch_size = len(src_idx_l)

        # Fetch input nodes
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)

        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat

        # Recurse for current nodes
        src_node_conv_feat = self.tem_conv(
            src_idx_l,
            cut_time_l,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)

        # --------------------------
        # 1. Retrieve neighbors
        # --------------------------

        # IN neighbors
        src_ngh_node_batch_in, src_ngh_t_batch_in = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )

        # OUT neighbors
        src_ngh_node_batch_out, src_ngh_t_batch_out = self.ngh_finder.get_temporal_out_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )

        # Convert neighbors to tensors
        src_ngh_node_batch_in_th = torch.from_numpy(src_ngh_node_batch_in).long().to(device)
        src_ngh_node_batch_out_th = torch.from_numpy(src_ngh_node_batch_out).long().to(device)

        # Time deltas (clip to avoid tiny negatives)
        src_ngh_t_batch_in_delta = np.maximum(
            cut_time_l[:, np.newaxis] - src_ngh_t_batch_in, 0.0
        )
        src_ngh_t_batch_out_delta = np.maximum(
            src_ngh_t_batch_out - cut_time_l[:, np.newaxis], 0.0
        )

        # Optional sanity checks:
        # assert (src_ngh_t_batch_in_delta >= 0).all(), "Negative IN deltas!"
        # assert (src_ngh_t_batch_out_delta >= 0).all(), "Negative OUT deltas!"

        src_ngh_t_batch_in_th = torch.from_numpy(src_ngh_t_batch_in_delta).float().to(device)
        src_ngh_t_batch_out_th = torch.from_numpy(src_ngh_t_batch_out_delta).float().to(device)

        # Flatten neighbors for recursion
        src_ngh_node_batch_in_flat = src_ngh_node_batch_in.flatten()
        src_ngh_t_batch_in_flat = src_ngh_t_batch_in.flatten()

        src_ngh_node_batch_out_flat = src_ngh_node_batch_out.flatten()
        src_ngh_t_batch_out_flat = src_ngh_t_batch_out.flatten()

        # Recurse for IN neighbors
        src_ngh_node_conv_feat_in = self.tem_conv(
            src_ngh_node_batch_in_flat,
            src_ngh_t_batch_in_flat,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_ngh_feat_in = src_ngh_node_conv_feat_in[..., :self.feat_dim].reshape(batch_size, num_neighbors, -1)
        src_ngh_feat_in = F.normalize(src_ngh_feat_in, p=2, dim=1)

        # Recurse for OUT neighbors
        src_ngh_node_conv_feat_out = self.tem_conv(
            src_ngh_node_batch_out_flat,
            src_ngh_t_batch_out_flat,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_ngh_feat_out = src_ngh_node_conv_feat_out[..., :self.feat_dim].reshape(batch_size, num_neighbors, -1)
        src_ngh_feat_out = F.normalize(src_ngh_feat_out, p=2, dim=1)

        # Encode times
        # src_ngh_t_embed_in = self.time_encoder(src_ngh_t_batch_in_th)
        # src_ngh_t_embed_out = self.time_encoder(src_ngh_t_batch_out_th)

        all_times = torch.cat([
            src_ngh_t_batch_in_th, src_ngh_t_batch_out_th
        ], dim=1)  # [B, 2N]

        all_times_encoded = self.time_encoder(all_times.unsqueeze(-1))  # [B, 2N, Dt]
        src_ngh_t_embed_in, src_ngh_t_embed_out = torch.split(
            all_times_encoded, num_neighbors, dim=1
        )

        # --------------------------
        # 2. Build all path pairs
        # --------------------------

        valid_paths_feats = []
        valid_paths_times = []
        path_masks = []

        for b in range(batch_size):

            in_feats = src_ngh_feat_in[b]  # [N, D]
            out_feats = src_ngh_feat_out[b]  # [N, D]
            in_times = src_ngh_t_embed_in[b]  # [N, Dt]
            out_times = src_ngh_t_embed_out[b]  # [N, Dt]

            t_in = src_ngh_t_batch_in_th[b]  # [N]
            t_out = src_ngh_t_batch_out_th[b]  # [N]

            # Compute valid temporal paths
            valid_mask = (t_in.unsqueeze(1) < t_out.unsqueeze(0))  # [N, N]
            valid_indices = valid_mask.nonzero(as_tuple=False)

            num_paths = valid_indices.shape[0]

            if num_paths > 0:
                i_idx = valid_indices[:, 0]
                j_idx = valid_indices[:, 1]

                feats_in_valid = in_feats[i_idx]  # [num_paths, D]
                feats_out_valid = out_feats[j_idx]  # [num_paths, D]
                times_in_valid = in_times[i_idx]  # [num_paths, Dt]
                times_out_valid = out_times[j_idx]  # [num_paths, Dt]

                paths_feats_tensor = torch.cat(
                    [feats_in_valid, feats_out_valid], dim=1
                )  # [num_paths, 2D]

                paths_times_tensor = torch.cat(
                    [times_in_valid, times_out_valid], dim=1
                )  # [num_paths, 2Dt]

                # truncate if too many
                if num_paths > max_paths:
                    paths_feats_tensor = paths_feats_tensor[:max_paths, :]
                    paths_times_tensor = paths_times_tensor[:max_paths, :]
                    path_mask = torch.ones(max_paths, dtype=torch.float, device=device)
                else:
                    # pad up to max_paths
                    pad_size = max_paths - num_paths
                    pad_feats = torch.zeros(
                        (pad_size, paths_feats_tensor.shape[1]), device=device
                    )
                    pad_times = torch.zeros(
                        (pad_size, paths_times_tensor.shape[1]), device=device
                    )
                    paths_feats_tensor = torch.cat(
                        [paths_feats_tensor, pad_feats], dim=0
                    )
                    paths_times_tensor = torch.cat(
                        [paths_times_tensor, pad_times], dim=0
                    )
                    path_mask = torch.cat(
                        [torch.ones(num_paths, device=device),
                         torch.zeros(pad_size, device=device)]
                    )
            else:
                paths_feats_tensor = torch.zeros((max_paths, 2 * self.feat_dim), device=device)
                paths_times_tensor = torch.zeros((max_paths, 2 * self.feat_dim), device=device)
                path_mask = torch.zeros(max_paths, device=device)

            valid_paths_feats.append(paths_feats_tensor)
            valid_paths_times.append(paths_times_tensor)
            path_masks.append(path_mask)

        path_feats_tensor = torch.stack(valid_paths_feats, dim=0)  # [B, max_paths, 2D]
        path_times_tensor = torch.stack(valid_paths_times, dim=0)  # [B, max_paths, 2Dt]
        path_masks_tensor = torch.stack(path_masks, dim=0)  # [B, max_paths]

        num_valid_paths_per_node = path_masks_tensor.sum(dim=1).long()
        #for i, num_paths in enumerate(num_valid_paths_per_node.tolist()):
        #    print(f"Node {i}: {num_paths} valid temporal paths")

        nonzero_mask = num_valid_paths_per_node > 0
        nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]  # shape [B_nonzero]


        if nonzero_indices.numel() == 0:
            output = torch.zeros_like(src_node_conv_feat)
            return output

        else:
            src_node_conv_feat_valid = src_node_conv_feat[nonzero_indices]
            src_node_t_embed_valid = src_node_t_embed[nonzero_indices]
            path_feats_tensor_valid = path_feats_tensor[nonzero_indices]
            path_times_tensor_valid = path_times_tensor[nonzero_indices]
            path_masks_tensor_valid = path_masks_tensor[nonzero_indices]

            # Run attention only for these nodes
            attn_m = self.attn_model_list[curr_layers - 1]

            neighbor_feats_valid, weight = attn_m(
                src_node_conv_feat_valid,
                src_node_t_embed_valid,
                path_feats_tensor_valid,
                path_times_tensor_valid,
                path_masks_tensor_valid
            )

            neighbor_feats_valid = F.normalize(neighbor_feats_valid, p=2, dim=1)

            # ⭐ Put the results back into a tensor of shape [B, D]
            neighbor_feats = torch.zeros_like(src_node_conv_feat)
            neighbor_feats[nonzero_indices] = neighbor_feats_valid

            return neighbor_feats


class TGNN_out(nn.Module):
    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time',
                 agg_method='lstm', num_layers=2, n_head=3, null_idx=0, num_heads=2, drop_out=0.3, seq_len=None):
        super(TGNN_out, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                              self.feat_dim,
                                                              attn_mode=attn_mode,
                                                              n_head=n_head,
                                                              drop_out=drop_out) for _ in range(num_layers)])

        # Time Encoding
        if use_time == 'pos':
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)
        elif use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('Invalid time encoding method!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=15):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
            src_ngh_node_batch_in, src_ngh_t_batch_in = self.ngh_finder.get_temporal_neighbor(tuple(src_idx_l),
                                                                                              tuple(cut_time_l),
                                                                                              num_neighbors=20)

            src_ngh_node_batch_out, src_ngh_t_batch_out = self.ngh_finder.get_temporal_out_neighbor(
                tuple(src_idx_l), tuple(cut_time_l), num_neighbors=20)

            src_ngh_node_batch_in_th = torch.from_numpy(src_ngh_node_batch_in).long().to(device)
            src_ngh_node_batch_out_th = torch.from_numpy(src_ngh_node_batch_out).long().to(device)

            src_ngh_t_batch_in_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch_in
            src_ngh_t_batch_in_th = torch.from_numpy(src_ngh_t_batch_in_delta).float().to(device)

            src_ngh_t_batch_out_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch_out
            src_ngh_t_batch_out_th = torch.from_numpy(src_ngh_t_batch_out_delta).float().to(device)



            # get previous layer's node features
            src_ngh_node_batch_in_flat = src_ngh_node_batch_in.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_in_flat = src_ngh_t_batch_in.flatten()  # reshape(batch_size, -1)

            src_ngh_node_batch_out_flat = src_ngh_node_batch_out.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_out_flat = src_ngh_t_batch_out.flatten()  # reshape(batch_size, -1)

            src_ngh_node_conv_feat_in = self.tem_conv(src_ngh_node_batch_in_flat,
                                                      src_ngh_t_batch_in_flat,
                                                      curr_layers=curr_layers - 1,
                                                      num_neighbors=num_neighbors)

            src_ngh_node_conv_feat_out = self.tem_conv(src_ngh_node_batch_out_flat,
                                                       src_ngh_t_batch_out_flat,
                                                       curr_layers=curr_layers - 1,
                                                       num_neighbors=num_neighbors)

            src_ngh_feat_in = src_ngh_node_conv_feat_in.view(batch_size, num_neighbors, -1)
            src_ngh_feat_in = F.normalize(src_ngh_feat_in, p=2, dim=1)

            src_ngh_feat_out = src_ngh_node_conv_feat_out.view(batch_size, num_neighbors, -1)
            src_ngh_feat_out = F.normalize(src_ngh_feat_out, p=2, dim=1)

            # get node time features
            src_ngh_t_embed_in = self.time_encoder(src_ngh_t_batch_in_th)
            src_ngh_t_embed_out = self.time_encoder(src_ngh_t_batch_out_th)

            # attention aggregation
            mask = src_ngh_node_batch_in_th == 0
            mask = src_ngh_node_batch_out_th == 0

            attn_m = self.attn_model_list[curr_layers - 1]

            local_in, weight_in = attn_m(src_node_conv_feat,
                                         src_node_t_embed,
                                         src_ngh_feat_in,
                                         src_ngh_t_embed_in,
                                         mask)

            local_out, weight_out = attn_m(src_node_conv_feat,
                                           src_node_t_embed,
                                           src_ngh_feat_out,
                                           src_ngh_t_embed_out,
                                           mask)

            local = torch.mul(local_in, local_out)
            local = F.normalize(local, p=2, dim=1)

            return local


class TGNN_Closeness(nn.Module):
    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time',
                 agg_method='lstm', num_layers=3, n_head=4, null_idx=0, num_heads=2, drop_out=0.3, seq_len=None):
        super(TGNN_Closeness, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                              self.feat_dim,
                                                              attn_mode=attn_mode,
                                                              n_head=n_head,
                                                              drop_out=drop_out) for _ in range(num_layers)])

        # Time Encoding
        if use_time == 'pos':
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)
        elif use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('Invalid time encoding method!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=15):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
            src_ngh_node_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_out_neighbor(tuple(src_idx_l),
                                                                                            tuple(cut_time_l),
                                                                                            num_neighbors=20)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            src_ngh_feat = F.normalize(src_ngh_feat, p=2, dim=1)

            # get node time features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   mask)
            local = F.normalize(local, p=2, dim=1)

            return local


class TATKC(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        # self.n_feat_th = n_feat
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]

        self.n_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.feat_dim,
                                                                  attn_mode=attn_mode,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
            raise ValueError('invalid agg_method value, use attn or lstm')

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        elif use_time == 'pos':
            assert (seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=12):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
            src_ngh_node_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(tuple(src_idx_l),
                                                                                        tuple(cut_time_l),
                                                                                        num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            src_ngh_feat = F.normalize(src_ngh_feat, p=2, dim=1)

            # get node time features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   mask)
            local = F.normalize(local, p=2, dim=1)

            return local
