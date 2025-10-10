import logging
import numpy as np
import torch
import torch.nn as nn
import numpy as np, torch.nn.functional as F


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
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, attn_bias=None, return_logits: bool = False):
        # scores: [n*B, Lq, Lk]
        scores = torch.bmm(q, k.transpose(1, 2)) / self.temperature

        # optional bias (same shape as scores)
        if attn_bias is not None:
            scores = scores + attn_bias

        if mask is not None:
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        if return_logits:
            return output, attn, scores
        return output, attn



class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)

    def forward(self, q, k, v, mask=None, attn_bias=None, return_logits: bool = False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        B, Lq, _ = q.size(); Lk = k.size(1)
        residual = q

        # projections: [B,L,dim] -> [n*B,L,dk/dv]
        q = self.w_qs(q).view(B, Lq, n_head, d_k).permute(2,0,1,3).contiguous().view(-1, Lq, d_k)
        k = self.w_ks(k).view(B, Lk, n_head, d_k).permute(2,0,1,3).contiguous().view(-1, Lk, d_k)
        v = self.w_vs(v).view(B, Lk, n_head, d_v).permute(2,0,1,3).contiguous().view(-1, Lk, d_v)

        # mask -> [n*B, Lq, Lk]
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=q.device)
            else:
                mask = mask.to(dtype=torch.bool, device=q.device)
            if mask.dim() == 3 and mask.size(1) == 1 and Lq > 1:
                mask = mask.expand(B, Lq, Lk)
            mask = mask.repeat(n_head, 1, 1)

        # attn_bias -> float, broadcast like mask to [n*B, Lq, Lk]
        bias_rep = None
        if attn_bias is not None:
            if not isinstance(attn_bias, torch.Tensor):
                attn_bias = torch.as_tensor(attn_bias, dtype=q.dtype, device=q.device)
            else:
                attn_bias = attn_bias.to(dtype=q.dtype, device=q.device)
            if attn_bias.dim() == 3 and attn_bias.size(1) == 1 and Lq > 1:
                attn_bias = attn_bias.expand(B, Lq, Lk)   # from [B,1,Lk]
            bias_rep = attn_bias.repeat(n_head, 1, 1)

        if return_logits:
            output, attn, scores = self.attention(q, k, v, mask=mask, attn_bias=bias_rep, return_logits=True)
        else:
            output, attn = self.attention(q, k, v, mask=mask, attn_bias=bias_rep)

        # restore shapes
        output = output.view(n_head, B, Lq, d_v).permute(1,2,0,3).contiguous().view(B, Lq, n_head * d_v)
        output = self.layer_norm(self.dropout(self.fc(output)) + residual)

        attn_mean = attn.view(n_head, B, Lq, Lk).permute(1,2,0,3).contiguous().mean(dim=2)

        if return_logits:
            logits = scores.view(n_head, B, Lq, Lk).permute(1,0,2,3).contiguous()  # [B,n_head,Lq,Lk]
            return output, attn_mean, logits
        return output, attn_mean

class AttnModelPTD_IN_logits(nn.Module):
    """
    Uses PTD only as an attention prior (logit bias).
    Content (V) is feature+time only.
    """
    def __init__(self, feat_dim, time_dim, ptd_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim  = ptd_dim

        # d_model excludes PTD (we do NOT concatenate PTD into Q/K/V)
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0

        # map PTD -> scalar bias per neighbor
        self.ptd2bias = nn.Linear(ptd_dim, 1)
        nn.init.xavier_uniform_(self.ptd2bias.weight); nn.init.zeros_(self.ptd2bias.bias)
        self.alpha_attn = 1.0  # scale of PTD prior; set from trust if you like

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        if attn_mode != 'prod':
            raise ValueError('prior path shown for prod attention; add similar threading for map if needed.')
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )

        # diagnostics
        self.diag_enabled = True
        self.diag_every   = 200
        self.diag_max_rows = 64
        self._diag_step = 0

    def set_alpha(self, alpha: float):
        self.alpha_attn = float(alpha)

    def set_diag(self, enabled=True, every=200, max_rows=64):
        self.diag_enabled = bool(enabled)
        self.diag_every   = int(max(1, every))
        self.diag_max_rows = int(max(1, max_rows))

    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask, return_logits: bool = False):
        """
        src     : [B, D]
        src_t   : [B, 1, Dt]
        src_ptd : [B, Dp]   (unused in content; only used if you want a query bias as well)
        seq     : [B, N, D]
        seq_t   : [B, N, Dt]
        seq_ptd : [B, N, Dp] -> produces attention logit bias
        mask    : [B, N] or [B, 1, N] (True = padded)
        """
        eps = 1e-12
        device = seq.device
        B, N, _ = seq.size()
        mask_ = mask.unsqueeze(1) if mask.dim() == 2 else mask  # [B,1,N]

        # Q/K/V (NO PTD in content)
        q = torch.cat( [src.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)  # [B,1,D+DtDp]
        k = torch.cat([seq, seq_t, seq_ptd], dim=2)    # [B,N,D+Dt]
        v = k                                            # same content

        # PTD prior -> bias over neighbors, broadcast over heads and query-length
        bias = self.ptd2bias(seq_ptd).squeeze(-1)        # [B,N]
        # normalize per-row to stabilize scale, then apply alpha
        bias = (bias - bias.mean(dim=-1, keepdim=True)) / (bias.std(dim=-1, keepdim=True) + 1e-6)
        attn_bias = self.alpha_attn * bias.unsqueeze(1)  # [B,1,N] -> treated like scores’ bias

        # MAIN pass (no logits to avoid OOM)
        out_on, attn_on = self.multi_head_target(q=q, k=k, v=v, mask=mask_, attn_bias=attn_bias, return_logits=False)
        attn_on = attn_on.squeeze(1)                     # [B,N]
        output  = self.merger(out_on.squeeze(1), src)    # [B,D]

        # DIAGNOSTICS on a SMALL slice (compare PTD-ON vs PTD-OFF)
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            with torch.no_grad():
                bs  = min(B, self.diag_max_rows)
                idx = torch.arange(bs, device=device)
                m_s = mask_[idx]
                nbv = (~m_s.squeeze(1)).float()          # [bs,N]

                # OFF = no bias
                _, attn_off, logits_off = self.multi_head_target(
                    q=q[idx], k=k[idx], v=v[idx], mask=m_s, attn_bias=None, return_logits=True
                )
                # ON  = with bias
                _, attn_on_s, logits_on = self.multi_head_target(
                    q=q[idx], k=k[idx], v=v[idx], mask=m_s, attn_bias=attn_bias[idx], return_logits=True
                )
                attn_off = attn_off.squeeze(1); attn_on_s = attn_on_s.squeeze(1)  # [bs,N]

                # renormalize over valid
                p = (attn_on_s * nbv); p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
                q_ = (attn_off * nbv); q_ = q_ / q_.sum(dim=-1, keepdim=True).clamp_min(eps)

                rows_ok = (nbv.sum(dim=-1) >= 2)
                if rows_ok.any():
                    rv = rows_ok
                    delta_attn = (p[rv] - q_[rv]).abs().mean().item()
                    # KL(p||q) and entropy
                    p_r = p[rv].clamp_min(eps); q_r = q_[rv].clamp_min(eps)
                    kl  = (p_r * (p_r.log() - q_r.log())).sum(dim=-1).mean().item()
                    ent = (-(p_r) * p_r.log()).sum(dim=-1).mean().item()
                    # Δlogits (mean over heads & valid)
                    s_on  = logits_on.mean(dim=1).squeeze(2)   # [bs,N]
                    s_off = logits_off.mean(dim=1).squeeze(2)  # [bs,N]
                    delta_logits = float(((s_on - s_off).abs() * nbv[rv]).sum() / nbv[rv].sum().clamp_min(1.0))

                    # Spearman(attn, PTD)
                    ptd_score = seq_ptd[idx].norm(dim=-1)      # [bs,N] (or pick a specific PTD channel)
                    xm = torch.where(nbv.bool(), p, p.min(dim=-1, keepdim=True).values - 1)
                    ym = torch.where(nbv.bool(), ptd_score, ptd_score.min(dim=-1, keepdim=True).values - 1)
                    rx = xm.argsort(dim=-1).argsort(dim=-1).float()
                    ry = ym.argsort(dim=-1).argsort(dim=-1).float()
                    denom = nbv.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    mux = (rx * nbv).sum(dim=-1, keepdim=True) / denom
                    muy = (ry * nbv).sum(dim=-1, keepdim=True) / denom
                    sx = torch.sqrt(((rx - mux).pow(2) * nbv).sum(dim=-1, keepdim=True) / denom).clamp_min(1e-6)
                    sy = torch.sqrt(((ry - muy).pow(2) * nbv).sum(dim=-1, keepdim=True) / denom).clamp_min(1e-6)
                    zx = (rx - mux) / sx; zy = (ry - muy) / sy
                    spearman_row = ((zx[rv] * zy[rv]).mean(dim=-1)).mean().clamp(-1.0, 1.0).item()
                else:
                    delta_attn = kl = ent = delta_logits = spearman_row = 0.0

                print(f"[PTD-DIAG attn] Δattn={delta_attn:.4f} Δlogits={delta_logits:.4f} "
                      f"KL={kl:.4f} H={ent:.4f} Spearman(attn,PTD)={spearman_row:.4f}")

        self._diag_step += 1

        # We keep the API: logits for full batch aren’t returned (avoid OOM)
        if return_logits:
            return output, attn_on, None
        return output, attn_on



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


class AttnModelPTD_IN_logits_19Sep(nn.Module):

    """
    PTD as a *soft* attention prior (logit bias):
      - learnable alpha_attn (starts at 0)
      - masked z-score across valid neighbors
      - clipped z-score to bound influence
    """
    def __init__(self, feat_dim, time_dim, ptd_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2, bias_clip: float = 2.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim  = ptd_dim
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0

        self.ptd2bias = nn.Linear(ptd_dim, 1)
        nn.init.xavier_uniform_(self.ptd2bias.weight); nn.init.zeros_(self.ptd2bias.bias)

        # 🔑 start neutral; we’ll ramp this during training
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))
        self.bias_clip  = float(bias_clip)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        if attn_mode != 'prod':
            raise ValueError('wired for prod attention')
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head, d_model=self.model_dim,
            d_k=self.model_dim // n_head, d_v=self.model_dim // n_head, dropout=drop_out
        )

        self.diag_enabled, self.diag_every, self.diag_max_rows = True, 200, 64
        self._diag_step = 0

    def _masked_zscore(self, x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x:[B,N], valid_mask:[B,N] (True if VALID)
        v = valid_mask.float()
        cnt  = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        var  = ((x - mean)**2 * v).sum(dim=1, keepdim=True) / cnt
        std  = var.sqrt().clamp_min(eps)
        z    = (x - mean) / std
        return torch.where(valid_mask, z, torch.zeros_like(z))

    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask, return_logits: bool = False):
        eps = 1e-12
        B, N, _ = seq.size()
        mask_3d  = mask.unsqueeze(1) if mask.dim() == 2 else mask    # [B,1,N] True=PAD
        valid_bn = ~(mask if mask.dim()==2 else mask.squeeze(1))     # [B,N]   True=VALID

        # Q/K/V content includes PTD enc, as in your current design
        q = torch.cat([src.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)  # [B,1,D+Dt+Dp]
        k = torch.cat([seq,             seq_t, seq_ptd],              dim=2)   # [B,N,D+Dt+Dp]
        v = k

        # ----- PTD prior (masked z-score + clip + learnable α) -----
        raw_bias = self.ptd2bias(seq_ptd).squeeze(-1)                        # [B,N]
        bias_z   = self._masked_zscore(raw_bias, valid_bn).clamp_(-self.bias_clip, self.bias_clip)
        attn_bias = self.alpha_attn * bias_z.unsqueeze(1)                    # [B,1,N]

        out, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask_3d, attn_bias=attn_bias, return_logits=False)
        out_feat  = self.merger(out.squeeze(1), src)                         # [B,D]
        attn_out  = attn.squeeze(1)                                          # [B,N]

        # (diagnostics kept as in your file if you like)
        self._diag_step += 1

        if return_logits:
            return out_feat, attn_out, None
        return out_feat, attn_out


class TATKC_PTD_19sep(torch.nn.Module):
    
    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC_PTD_19sep, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([AttnModelPTD_IN_logits(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False

 # ----- Adaptive neighbor fanout (per-hop) -----
        # static/base degrees per layer (fallback)
        self.base_degrees   = [seq_len or  self.feat_dim*0 + 20] * self.num_layers  # harmless default
        # active degrees used during tem_conv2 (mutated by set_trust_alpha_for_sampling)
        self.active_degrees = list(self.base_degrees)

        # bounds for hop-1 fanout (budget-aware): fewer when trust is HIGH (alpha→1)
        self.n1_min = max(5, int(0.5 * (self.base_degrees[0] if len(self.base_degrees) else 20)))
        self.n1_max = int(2.0 * (self.base_degrees[0] if len(self.base_degrees) else 20))
        self.compensate_hop2 = True  # if True, adjust hop-2 to keep cost roughly stable



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_trust_alpha_for_sampling(self, alpha: float):
        """
        alpha in [0,1]: 1.0 = high trust in PTD -> use FEWER hop-1 neighbors.
        Call this once per snapshot/batch before tem_conv2().
        """
        if not isinstance(alpha, (int, float)):
            alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))

        # interpolate hop-1
        n1 = int(round(self.n1_min + (1.0 - alpha) * (self.n1_max - self.n1_min)))
        n1 = max(1, n1)
        if not self.active_degrees:
            self.active_degrees = [n1] + [20] * max(0, self.num_layers - 1)
        else:
            self.active_degrees[0] = n1

        # optional compensation on hop-2 to stabilize compute
        if self.compensate_hop2 and len(self.active_degrees) > 1:
            n1_base = max(1, self.base_degrees[0])
            n2_base = max(1, self.base_degrees[1])
            n2 = int(round(n2_base * (n1_base / float(n1))))
            self.active_degrees[1] = max(1, n2)

        # keep deeper hops at base
        for l in range(2, len(self.active_degrees)):
            self.active_degrees[l] = self.base_degrees[l]

    def set_ptd_vector(self, ptd_vec_np):
        """
        Build PTD tables for the current snapshot time t_cut.

        Inputs
        ------
        ptd_vec_np : np.ndarray [N]
            RAW PTD counts per node (1-based IDs in the graph).
            Deprecated/ignored. Kept for API compatibility.

        What it writes
        --------------
        self.ptd_vec     : [N, 2] float32  -> [ zlog1p(raw), percentile ]
        self.ptd_log_col : [N]    float32  ->   log1p(raw)  (for diagnostics)

        Notes
        -----
        - Uses robust z-scoring in log-space: (log1p(raw) - median) / MAD.
        - If you set `self.ptd_stats = {'log_median': ..., 'log_mad': ...}` before training
        (computed on TRAIN snapshots), those are used. Otherwise falls back to per-snapshot stats.
        """
        import numpy as np
        import torch

        # ----- raw & log1p(raw) -----
        raw  = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        logv = np.log1p(raw)  # scale-free, unitless

        # ----- robust stats (prefer train stats if provided) -----
        use_train_stats = hasattr(self, "ptd_stats") and isinstance(self.ptd_stats, dict) \
                        and ("log_median" in self.ptd_stats) and ("log_mad" in self.ptd_stats)

        if use_train_stats:
            log_med = float(self.ptd_stats["log_median"])
            log_mad = float(self.ptd_stats["log_mad"])
            if not np.isfinite(log_mad) or log_mad <= 0:
                # safety fallback if train stats are degenerate
                nz = logv[raw > 0]
                if nz.size > 5:
                    log_med = float(np.median(nz))
                    mad = float(np.median(np.abs(nz - log_med)))
                    log_mad = 1.4826 * mad + 1e-6
                else:
                    log_med = float(np.median(logv))
                    log_mad = float(np.std(logv) + 1e-6)
        else:
            # per-snapshot fallback (works but less stable than train-wide stats)
            nz = logv[raw > 0]
            if nz.size > 5:
                log_med = float(np.median(nz))
                mad = float(np.median(np.abs(nz - log_med)))
                log_mad = 1.4826 * mad + 1e-6
            else:
                log_med = float(np.median(logv))
                log_mad = float(np.std(logv) + 1e-6)

        # z-scored log feature (bounded)
        zlog = (logv - log_med) / log_mad
        zlog = np.clip(zlog, -5.0, 5.0).astype(np.float32)

        # percentile on RAW counts (stable, unitless)
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = (ranks + 0.5) / max(1.0, float(len(raw)))  # (0,1], float32

        pair = np.stack([zlog, pct.astype(np.float32)], axis=-1)

        # ----- move to the correct device -----
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))

        self.ptd_log_col = torch.from_numpy(logv.astype(np.float32)).to(dev)
        self.ptd_log_col.requires_grad_(False)

        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)

        # (optional) stash last-used stats for debugging
        self._ptd_stats_last = {"log_median": log_med, "log_mad": log_mad}

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Temporal convolution using IN-neighbors + PTD fetched from self.ptd_vec.
        Expects you already called set_ptd_vector(...) for this snapshot.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")

        B = len(src_idx_l)

# choose fanout for THIS layer (curr_layers is 1..L here)
        fanout_this = num_neighbors
        if hasattr(self, "active_degrees") and isinstance(self.active_degrees, list):
            idx = max(0, min(len(self.active_degrees)-1, curr_layers-1))
            fanout_this = int(self.active_degrees[idx]) if self.active_degrees[idx] else num_neighbors

        # ---- source ids & times ----
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)  # [B] (1-based, 0=pad)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)  # [B,1]

        # raw node emb + time emb
        src_node_feat = self.node_raw_embed(src_ids_th)  # [B,D]
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))  # [B,Dt]

        # PTD for sources (from table)
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1  # [B] -> 0-based
        ptd_src_vals = self.ptd_vec[src_idx0]  # [B, C]  (C=1 if only raw PTD; C=2 if raw+pct)
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)  # [B, Dptd]

        if curr_layers == 0:
            return src_node_feat

        # ---- recurse on sources (one layer down) ----
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,  # ignored inside
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)






        # ---- neighbors (ids & times) ----
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        N = ngh_nodes_np.shape[1]                      # <- use the actual fanout
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)  # [B,N]
        mask = (ngh_ids_th == 0)                       # [B,N]

        # time deltas and enc
        dt_np  = cut_time_l[:, np.newaxis] - ngh_times_np    # [B,N]
        dt_th  = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)                 # [B,N,Dt]

        # recurse neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)                  # [B*N]
        t_flat   = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )                                                   # [B*N, D]
        ngh_conv = ngh_conv_flat.view(B, N, -1)             # [B,N,D]  <-- use N here
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)

        # PTD for neighbors

        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1       # [B,N]
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]               # [B,N,C]
        ptd_ngh_enc  = self.ptd_encoder(ptd_ngh_vals)       # [B,N,Dptd]
        ptd_ngh_enc  = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)





        # ---- attention (with PTD injected) ----
        attn_m = self.attn_model_list[curr_layers - 1]
        local, _ = attn_m(
            src=src_node_conv_feat,
            src_t=src_node_t_embed,
            src_ptd=ptd_embed_src,
            seq=ngh_conv,
            seq_t=ngh_t_enc,
            seq_ptd=ptd_ngh_enc,
            mask=mask
        )
        return F.normalize(local, p=2, dim=1)
    

class TATKC_PTD_20Aug(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC_PTD_20Aug, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([AttnModelPTD_IN_logits(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vector(self, ptd_vec_np):
        """
        Build PTD tables for the current snapshot time t_cut.

        Inputs
        ------
        ptd_vec_np : np.ndarray [N]
            RAW PTD counts per node (1-based IDs in the graph).
            Deprecated/ignored. Kept for API compatibility.

        What it writes
        --------------
        self.ptd_vec     : [N, 2] float32  -> [ zlog1p(raw), percentile ]
        self.ptd_log_col : [N]    float32  ->   log1p(raw)  (for diagnostics)

        Notes
        -----
        - Uses robust z-scoring in log-space: (log1p(raw) - median) / MAD.
        - If you set `self.ptd_stats = {'log_median': ..., 'log_mad': ...}` before training
        (computed on TRAIN snapshots), those are used. Otherwise falls back to per-snapshot stats.
        """
        import numpy as np
        import torch

        # ----- raw & log1p(raw) -----
        raw  = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        logv = np.log1p(raw)  # scale-free, unitless

        # ----- robust stats (prefer train stats if provided) -----
        use_train_stats = hasattr(self, "ptd_stats") and isinstance(self.ptd_stats, dict) \
                        and ("log_median" in self.ptd_stats) and ("log_mad" in self.ptd_stats)

        if use_train_stats:
            log_med = float(self.ptd_stats["log_median"])
            log_mad = float(self.ptd_stats["log_mad"])
            if not np.isfinite(log_mad) or log_mad <= 0:
                # safety fallback if train stats are degenerate
                nz = logv[raw > 0]
                if nz.size > 5:
                    log_med = float(np.median(nz))
                    mad = float(np.median(np.abs(nz - log_med)))
                    log_mad = 1.4826 * mad + 1e-6
                else:
                    log_med = float(np.median(logv))
                    log_mad = float(np.std(logv) + 1e-6)
        else:
            # per-snapshot fallback (works but less stable than train-wide stats)
            nz = logv[raw > 0]
            if nz.size > 5:
                log_med = float(np.median(nz))
                mad = float(np.median(np.abs(nz - log_med)))
                log_mad = 1.4826 * mad + 1e-6
            else:
                log_med = float(np.median(logv))
                log_mad = float(np.std(logv) + 1e-6)

        # z-scored log feature (bounded)
        zlog = (logv - log_med) / log_mad
        zlog = np.clip(zlog, -5.0, 5.0).astype(np.float32)

        # percentile on RAW counts (stable, unitless)
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = (ranks + 0.5) / max(1.0, float(len(raw)))  # (0,1], float32

        pair = np.stack([zlog, pct.astype(np.float32)], axis=-1)

        # ----- move to the correct device -----
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))

        self.ptd_log_col = torch.from_numpy(logv.astype(np.float32)).to(dev)
        self.ptd_log_col.requires_grad_(False)

        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)

        # (optional) stash last-used stats for debugging
        self._ptd_stats_last = {"log_median": log_med, "log_mad": log_mad}

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Temporal convolution using IN-neighbors + PTD fetched from self.ptd_vec.
        Expects you already called set_ptd_vector(...) for this snapshot.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")

        B = len(src_idx_l)

        # ---- source ids & times ----
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)  # [B] (1-based, 0=pad)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)  # [B,1]

        # raw node emb + time emb
        src_node_feat = self.node_raw_embed(src_ids_th)  # [B,D]
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))  # [B,Dt]

        # PTD for sources (from table)
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1  # [B] -> 0-based
        ptd_src_vals = self.ptd_vec[src_idx0]  # [B, C]  (C=1 if only raw PTD; C=2 if raw+pct)
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)  # [B, Dptd]

        if curr_layers == 0:
            return src_node_feat

        # ---- recurse on sources (one layer down) ----
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,  # ignored inside
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)

        # ---- neighbors (ids & times) ----
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)  # [B,N]
        mask = (ngh_ids_th == 0)  # [B,N] padding mask

        # time deltas and enc
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np  # [B,N]
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)  # [B,N,Dt]

        # ---- recurse on neighbors (flatten -> recurse -> reshape) ----
        ngh_flat = ngh_nodes_np.reshape(-1)  # [B*N]
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,  # ignored
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )  # [B*N, D]
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)  # [B,N,D]
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)

        # ---- PTD for neighbors (from table; NOT repeating source PTD) ----
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1  # [B,N]
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]  # [B,N,C]
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)  # [B,N,Dptd]
        ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)

        # ---- attention (with PTD injected) ----
        attn_m = self.attn_model_list[curr_layers - 1]
        local, _ = attn_m(
            src=src_node_conv_feat,
            src_t=src_node_t_embed,
            src_ptd=ptd_embed_src,
            seq=ngh_conv,
            seq_t=ngh_t_enc,
            seq_ptd=ptd_ngh_enc,
            mask=mask
        )
        return F.normalize(local, p=2, dim=1)
    


    import torch
import torch.nn as nn
import torch.nn.functional as F

# Modify your existing AttnModelPTD_IN_logits class
class AttnModelPTD_IN_logits_Gated(nn.Module):
    """Your original class with minimal gating modification"""
    def __init__(self, feat_dim, time_dim, ptd_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        super().__init__()
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim = ptd_dim
        use_gating=True
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0
        
        # Original components
        self.ptd2bias = nn.Linear(ptd_dim, 1)
        nn.init.xavier_uniform_(self.ptd2bias.weight)
        nn.init.zeros_(self.ptd2bias.bias)
        self.alpha_attn = 1.0
        
        # NEW: PTD strength estimator
        self.ptd_strength_net = nn.Sequential(
            nn.Linear(2, 32),  # 2 for [zlog1p, percentile]
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
                # ADD: Simple PTD reliability estimator
        self.ptd_reliability = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # CRITICAL: Initialize to output positive values
        nn.init.constant_(self.ptd_reliability[-1].bias, 1.0)
        # NEW: Gate networks for source and neighbors
        if use_gating:
            # Gate for source features
            self.src_gate_net = nn.Sequential(
                nn.Linear(2 + feat_dim, 64),  # PTD + features
                nn.ReLU(),
                nn.Dropout(drop_out),
                nn.Linear(64, feat_dim),
                nn.Sigmoid()
            )
            
            # Gate for neighbor features
            self.ngh_gate_net = nn.Sequential(
                nn.Linear(2 + feat_dim, 64),
                nn.ReLU(),
                nn.Dropout(drop_out),
                nn.Linear(64, feat_dim),
                nn.Sigmoid()
            )
        
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        
        if attn_mode != 'prod':
            raise ValueError('Currently supports prod attention')
        
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )
        
        self.temperature = gate_temperature
        self.diag_enabled = True
        self.diag_every = 200
        self.diag_max_rows = 64
        self._diag_step = 0
    
    def compute_adaptive_gates(self, features, ptd_raw):
        """
        Compute PTD-conditioned gates for features.
        
        Args:
            features: [B, D] or [B, N, D] node features
            ptd_raw: [B, 2] or [B, N, 2] raw PTD values
        
        Returns:
            gated_features: Features with adaptive gating applied
            ptd_strength: Estimated PTD signal strength
        """
        # Estimate PTD signal strength
        ptd_strength = self.ptd_strength_net(ptd_raw)  # [B, 1] or [B, N, 1]
        
        if not self.use_gating:
            return features, ptd_strength.squeeze(-1)
        
        # Compute gates based on PTD and features
        if features.dim() == 2:
            # Source features [B, D]
            gate_input = torch.cat([ptd_raw, features], dim=-1)
            gates = self.src_gate_net(gate_input)
        else:
            # Neighbor features [B, N, D]
            gate_input = torch.cat([ptd_raw, features], dim=-1)
            gates = self.ngh_gate_net(gate_input)
        
        # Apply temperature-scaled gating with residual
        gates = torch.sigmoid((gates - 0.5) / self.temperature)
        gated_features = gates * features + (1 - gates) * features.detach()
        
        return gated_features, ptd_strength.squeeze(-1)
    
    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask, 
                ptd_vec_table=None, src_indices=None, seq_indices=None,
                return_logits=False):
        """
        Enhanced forward pass with PTD-adaptive gating.
        
        Additional args:
            ptd_vec_table: The full PTD table (self.ptd_vec from main model)
            src_indices: Source node indices for PTD lookup
            seq_indices: Sequence node indices for PTD lookup
        """
        eps = 1e-12
        device = seq.device
        B, N, _ = seq.size()
        mask_ = mask.unsqueeze(1) if mask.dim() == 2 else mask
        
        # Get raw PTD values if available
        if ptd_vec_table is not None and src_indices is not None:
            # Lookup raw PTD values [zlog1p, percentile]
            src_ptd_raw = ptd_vec_table[src_indices]  # [B, 2]
            seq_ptd_raw = ptd_vec_table[seq_indices] if seq_indices is not None else None
        else:
            # Fallback: extract from encoded PTD (less ideal)
            src_ptd_raw = src_ptd[:, :2] if src_ptd.size(1) >= 2 else torch.zeros(B, 2, device=device)
            seq_ptd_raw = seq_ptd[:, :, :2] if seq_ptd.size(-1) >= 2 else torch.zeros(B, N, 2, device=device)
        
        # Apply adaptive gating to features
        src_gated, src_strength = self.compute_adaptive_gates(src, src_ptd_raw)
        if seq_ptd_raw is not None:
            seq_gated, seq_strength = self.compute_adaptive_gates(seq, seq_ptd_raw)
        else:
            seq_gated, seq_strength = seq, torch.ones(B, N, device=device)
        
        # Adapt alpha based on average PTD strength
        avg_strength = src_strength.mean()
        self.alpha_attn = 0.1 + 1.9 * avg_strength  # Scale from 0.1 to 2.0
        
        # Build Q/K/V with gated features
        q = torch.cat([src_gated.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)
        k = torch.cat([seq_gated, seq_t, seq_ptd], dim=2)
        v = k
        
        # PTD bias (as before but with adaptive alpha)
        bias = self.ptd2bias(seq_ptd).squeeze(-1)
        bias = (bias - bias.mean(dim=-1, keepdim=True)) / (bias.std(dim=-1, keepdim=True) + 1e-6)
        attn_bias = self.alpha_attn * bias.unsqueeze(1)
        
        # Multi-head attention
        out_on, attn_on = self.multi_head_target(
            q=q, k=k, v=v, mask=mask_, attn_bias=attn_bias, return_logits=False
        )
        attn_on = attn_on.squeeze(1)
        output = self.merger(out_on.squeeze(1), src)
        
        # Enhanced diagnostics
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            with torch.no_grad():
                print(f"[PTD-Gate] avg_strength={avg_strength:.3f} "
                      f"alpha={self.alpha_attn:.3f} "
                      f"src_gate_mean={src_gated.mean():.3f} "
                      f"seq_gate_mean={seq_gated.mean():.3f}")
        
        self._diag_step += 1
        
        if return_logits:
            return output, attn_on, None
        return output, attn_on



class AttnModelPTD_SimpleAdaptive(nn.Module):
    """
    Fixed version with better alpha calculation and quality tracking.
    """
    def __init__(self, feat_dim, time_dim, ptd_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim = ptd_dim
        
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0
        
        # Original components
        self.ptd2bias = nn.Linear(ptd_dim, 1)
        nn.init.xavier_uniform_(self.ptd2bias.weight)
        nn.init.zeros_(self.ptd2bias.bias)
        
        from module import MergeLayer, MultiHeadAttention
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )
        
        self.diag_enabled = True
        self.diag_every = 200
        self._diag_step = 0
        
        # Adaptive alpha parameters
        self.alpha_attn = 1.0
        self.adaptive_alpha = True
        
        # FIXED: Better alpha range and calculation
        self.alpha_min = 0.5  # Increased from 0.3
        self.alpha_max = 1.8  # Keep max at 1.8
        
        # Track PTD quality for analysis
        self.ptd_quality_history = []
        self.alpha_history = []
    
    def compute_ptd_quality(self, seq_ptd):
        """
        Better PTD quality estimation based on distribution characteristics.
        """
        with torch.no_grad():
            # Flatten PTD across batch and sequence
            ptd_flat = seq_ptd.reshape(-1, seq_ptd.size(-1))
            
            # Compute multiple quality indicators
            ptd_std = ptd_flat.std(dim=0).mean().item()
            ptd_mean = ptd_flat.abs().mean().item()
            
            # Coefficient of variation (std/mean) - good indicator of signal quality
            if ptd_mean > 1e-6:
                cv = ptd_std / ptd_mean
            else:
                cv = 0.0
            
            # Also check range (max - min)
            ptd_range = (ptd_flat.max(dim=0).values - ptd_flat.min(dim=0).values).mean().item()
            
            # Combine indicators (normalized to [0, 1])
            # CV typically ranges from 0 to 2+
            quality_cv = min(cv / 1.0, 1.0)  # Normalize by expected CV ~1.0
            
            # Range normalized by expected scale (PTD is typically in [-5, 5] after encoding)
            quality_range = min(ptd_range / 10.0, 1.0)
            
            # Combined quality score
            quality = 0.5 * quality_cv + 0.5 * quality_range
            
            return quality, {'cv': cv, 'std': ptd_std, 'mean': ptd_mean, 'range': ptd_range}
    
    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask, return_logits=False):
        """
        Forward with better adaptive alpha calculation.
        """
        device = seq.device
        B, N, _ = seq.size()
        mask_ = mask.unsqueeze(1) if mask.dim() == 2 else mask
        
        if self.adaptive_alpha:
            # Compute PTD quality
            quality, quality_stats = self.compute_ptd_quality(seq_ptd)
            
            # Store for tracking
            self.ptd_quality_history.append(quality)
            
            # Map quality to alpha with better scaling
            # quality=0 -> alpha=alpha_min
            # quality=1 -> alpha=alpha_max
            self.alpha_attn = self.alpha_min + (self.alpha_max - self.alpha_min) * quality
            
            # Store alpha for tracking
            self.alpha_history.append(self.alpha_attn)
        
        # Standard forward pass
        q = torch.cat([src.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)
        k = torch.cat([seq, seq_t, seq_ptd], dim=2)
        v = k
        
        # PTD bias with adaptive alpha
        bias = self.ptd2bias(seq_ptd).squeeze(-1)
        bias = (bias - bias.mean(dim=-1, keepdim=True)) / (bias.std(dim=-1, keepdim=True) + 1e-6)
        attn_bias = self.alpha_attn * bias.unsqueeze(1)
        
        # Attention
        out, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask_, attn_bias=attn_bias)
        attn = attn.squeeze(1)
        output = self.merger(out.squeeze(1), src)
        
        # Enhanced diagnostics
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            print(f"[Simple-Adaptive] alpha={self.alpha_attn:.3f} quality={quality:.3f} "
                  f"cv={quality_stats['cv']:.3f} range={quality_stats['range']:.3f}")
        
        self._diag_step += 1
        
        if return_logits:
            return output, attn, None
        return output, attn




class AttnModelPTD_RobustAdaptive(nn.Module):
    """
    Robust version that handles PTD outliers and uses percentile-based quality estimation.
    """
    def __init__(self, feat_dim, time_dim, ptd_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim = ptd_dim
        
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0
        
        # Original components
        self.ptd2bias = nn.Linear(ptd_dim, 1)
        nn.init.xavier_uniform_(self.ptd2bias.weight)
        nn.init.zeros_(self.ptd2bias.bias)
        
        from module import MergeLayer, MultiHeadAttention
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )
        
        self.diag_enabled = True
        self.diag_every = 200
        self._diag_step = 0
        
        # Adaptive parameters
        self.alpha_attn = 1.0
        self.adaptive_alpha = True
        
        # Better alpha range for your data
        self.alpha_min = 0.8   # Higher minimum since PTD seems informative
        self.alpha_max = 1.5   # Lower maximum to avoid over-reliance
        
        # Track statistics
        self.ptd_quality_history = []
        self.alpha_history = []
        
        # Running statistics for normalization
        self.ptd_stats_buffer = []
        self.stats_window = 100  # Keep last 100 batches for statistics
    
    def compute_robust_ptd_quality(self, seq_ptd):
        """
        Robust PTD quality estimation using percentiles to handle outliers.
        """
        with torch.no_grad():
            ptd_flat = seq_ptd.reshape(-1, seq_ptd.size(-1))
            
            # Check both dimensions
            zlog_vals = ptd_flat[:, 0]
            pct_vals = ptd_flat[:, 1]
            
            # Quality based on actual variance in the data
            # For sparse data, check variance among non-zero values
            non_zero_mask = (zlog_vals != -1.0)  # Our encoding uses -1 for zeros
            
            if non_zero_mask.sum() > 10:
                # Variance among non-zero values
                nz_std = zlog_vals[non_zero_mask].std().item()
                nz_range = (zlog_vals[non_zero_mask].max() - zlog_vals[non_zero_mask].min()).item()
                
                # Quality indicators
                quality_std = min(nz_std / 0.5, 1.0)  # Expect std ~0.5 for good signal
                quality_range = min(nz_range / 2.0, 1.0)  # Expect range ~2.0
                quality_ratio = non_zero_mask.float().mean().item()  # Non-zero ratio
                
                # Combined quality
                quality = (
                    0.3 * quality_std +      # Variance component
                    0.3 * quality_range +    # Range component  
                    0.4 * quality_ratio      # Density component
                )
            else:
                # Too few non-zero values
                quality = 0.1
            
            stats = {
                'non_zero_ratio': non_zero_mask.float().mean().item(),
                'nz_std': nz_std if 'nz_std' in locals() else 0,
                'quality': quality
            }
            
            return quality, stats
        
    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask, return_logits=False):
        """
        Forward with robust adaptive alpha.
        """
        device = seq.device
        B, N, _ = seq.size()
        mask_ = mask.unsqueeze(1) if mask.dim() == 2 else mask
        
        if self.adaptive_alpha:
            # Compute robust PTD quality
            quality, quality_stats = self.compute_robust_ptd_quality(seq_ptd)
            
            # Store for tracking
            self.ptd_quality_history.append(quality)
            
            # Map quality to alpha
            self.alpha_attn = self.alpha_min + (self.alpha_max - self.alpha_min) * quality
            
            # Store alpha
            self.alpha_history.append(self.alpha_attn)
            
            # Update running statistics buffer
            self.ptd_stats_buffer.append(quality_stats)
            if len(self.ptd_stats_buffer) > self.stats_window:
                self.ptd_stats_buffer.pop(0)
        
        # Standard forward pass
        q = torch.cat([src.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)
        k = torch.cat([seq, seq_t, seq_ptd], dim=2)
        v = k
        
        # PTD bias with adaptive alpha
        bias = self.ptd2bias(seq_ptd).squeeze(-1)
        bias = (bias - bias.mean(dim=-1, keepdim=True)) / (bias.std(dim=-1, keepdim=True) + 1e-6)
        attn_bias = self.alpha_attn * bias.unsqueeze(1)
        
        # Attention
        out, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask_, attn_bias=attn_bias)
        attn = attn.squeeze(1)
        output = self.merger(out.squeeze(1), src)
        
        # Diagnostics
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            #if quality_stats:
                print(f"[Robust-Adaptive] alpha={self.alpha_attn:.3f} quality={quality:.3f} "
                      f"non_zero={quality_stats['non_zero_ratio']:.3f}")
        
        self._diag_step += 1
        
        if return_logits:
            return output, attn, None
        return output, attn


class AttnModelPTD_FinalAdaptive(nn.Module):
    """
    Final version optimized for your sparse PTD data.
    """
    def __init__(self, feat_dim, time_dim, ptd_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim = ptd_dim
        
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0
        
        # PTD to bias
        self.ptd2bias = nn.Linear(ptd_dim, 1)
        nn.init.xavier_uniform_(self.ptd2bias.weight)
        nn.init.zeros_(self.ptd2bias.bias)
        
        # Better alpha range for your data
        #self.alpha_min = 0.8   # Lower bound
        #self.alpha_max = 1.0   # Upper bound (not too high given noisy PTD)
        
        self.alpha_attn = 1.0  # Default
        
        from module import MergeLayer, MultiHeadAttention
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )
        
        # Track statistics
        self.batch_count = 0
        
    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask, return_logits=False):
        device = seq.device
        B, N, _ = seq.size()
        mask_ = mask.unsqueeze(1) if mask.dim() == 2 else mask
        
        # Simple adaptive alpha based on PTD variance
        with torch.no_grad():
            # Check PTD discriminative power
            ptd_std = seq_ptd.std()
            ptd_values = seq_ptd[:, :, 0].flatten()  # Get all PTD values
            zero_ratio = (ptd_values < -0.4).float().mean().item()  # Values near -0.5 are zeros
    
            if zero_ratio > 0.95:
                self.alpha_min = 0.9
                self.alpha_max = 1.1
            elif zero_ratio > 0.85:
                self.alpha_min = 0.8
                self.alpha_max = 1.0
            else:
                self.alpha_min = 0.85
                self.alpha_max = 1.05

            # Map std to alpha
            # Low std -> low alpha (PTD not useful)
            # High std -> higher alpha (PTD is useful)
            if ptd_std > 0.1:  # Has some signal
                quality = min(ptd_std.item() / 1.0, 1.0)  # Normalize by expected std
                self.alpha_attn = self.alpha_min + (self.alpha_max - self.alpha_min) * quality
            else:
                self.alpha_attn = self.alpha_min
            
            # Periodic logging
            if self.batch_count % 200 == 0:
                print(f"[Adaptive] std={ptd_std:.3f} alpha={self.alpha_attn:.3f}")
            self.batch_count += 1
        
        # Standard forward
        q = torch.cat([src.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)
        k = torch.cat([seq, seq_t, seq_ptd], dim=2)
        v = k
        
        # PTD bias with adaptive alpha
        bias = self.ptd2bias(seq_ptd).squeeze(-1)
        bias = (bias - bias.mean(dim=-1, keepdim=True)) / (bias.std(dim=-1, keepdim=True) + 1e-6)
        attn_bias = self.alpha_attn * bias.unsqueeze(1)
        
        # Attention
        out, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask_, attn_bias=attn_bias)
        attn = attn.squeeze(1)
        output = self.merger(out.squeeze(1), src)
        
        if return_logits:
            return output, attn, None
        return output, attn
    

class TATKC_PTD_19Sep_Gated_claude(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC_PTD_19Sep_Gated_claude, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([AttnModelPTD_FinalAdaptive(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vecto_sparse(self, ptd_vec_np):
        """
        Final optimized PTD encoding for extremely sparse data (85% zeros).
        This version maximizes diversity in encoded features.
        """
        import numpy as np
        import torch
        
        # Get raw PTD
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Print input stats (remove in production)
        print(f"[PTD-Input] zeros={100*(raw==0).mean():.1f}% "
            f"max={raw.max():.1f} p95={(np.percentile(raw[raw>0], 95) if (raw>0).any() else 0):.1f}")
        
        # Clip outliers more aggressively for stability
        if (raw > 0).sum() > 10:
            # Use 90th percentile instead of 95th
            cap = np.percentile(raw[raw > 0], 90)
            raw = np.minimum(raw, cap)
        
        # FEATURE 1: Rank-based encoding (guaranteed diversity)
        # This preserves relative ordering perfectly
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = ranks / max(1.0, float(len(raw) - 1))  # Normalize to [0, 1]
        
        # FEATURE 2: Power transform instead of log for better spread
        # Power < 1 expands small values, which is what we need
        power = 0.25  # Fourth root - expands small values more than log
        
        # Create the magnitude feature
        magnitude = np.zeros_like(raw)
        non_zero_mask = raw > 0
        
        if non_zero_mask.sum() > 0:
            # Apply power transform to non-zero values
            nz_vals = raw[non_zero_mask]
            transformed = np.power(nz_vals, power)
            
            # Normalize to roughly [-1, 2] range
            # Use robust statistics
            p50 = np.percentile(transformed, 50)
            p90 = np.percentile(transformed, 90)
            
            if p90 > p50:
                # Scale based on upper quartile range
                scaled = (transformed - p50) / (p90 - p50)
                magnitude[non_zero_mask] = scaled
            else:
                # Fallback: simple normalization
                magnitude[non_zero_mask] = transformed / (transformed.max() + 1e-6)
        
        # Mark zeros distinctly (important for model to distinguish)
        magnitude[~non_zero_mask] = -0.5
        
        # Create 3-feature encoding for better expressiveness
        # Feature 3: Binary indicator (helps model identify patterns)
        is_nonzero = non_zero_mask.astype(np.float32)
        
        # Stack features - but for compatibility, keep 2D
        # Combine binary indicator with magnitude
        enhanced_magnitude = magnitude + 0.1 * is_nonzero  # Slight boost for non-zeros
        
        pair = np.stack([enhanced_magnitude, pct], axis=-1)
        
        # Print encoding stats
        unique_mag = len(np.unique(np.round(enhanced_magnitude, 3)))
        print(f"[PTD-Encoded] magnitude: std={enhanced_magnitude[non_zero_mask].std() if non_zero_mask.any() else 0:.3f} "
            f"unique={100*unique_mag/len(enhanced_magnitude):.1f}% "
            f"range=[{enhanced_magnitude.min():.2f},{enhanced_magnitude.max():.2f}]")
        
        # Move to device
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)
        
        # Store additional info
        self.ptd_log_col = torch.from_numpy(np.log1p(raw)).to(dev)
        self.ptd_nonzero_mask = torch.from_numpy(is_nonzero).to(dev)
        

    def set_ptd_vector2(self, ptd_vec_np):
        """
        Alternative: Use binned encoding for better diversity.
        This creates discrete bins which guarantees diversity.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Create binned encoding
        # Bin 0: zeros
        # Bins 1-9: non-zero values split by percentiles
        binned = np.zeros_like(raw)
        non_zero_mask = raw > 0
        
        if non_zero_mask.sum() > 10:
            nz_vals = raw[non_zero_mask]
            
            # Create 9 bins for non-zero values using percentiles
            #percentiles = np.percentile(nz_vals, [11, 22, 33, 44, 55, 66, 77, 88, 100])
            percentiles = np.percentile(nz_vals, [20, 40, 60, 80, 100])
            
            # Assign bins
            for i, p in enumerate(percentiles):
                mask = (raw > (percentiles[i-1] if i > 0 else 0)) & (raw <= p)
                binned[mask] = (i + 1) / 10.0  # Normalize to [0.1, 0.9]
        
        # Zeros stay at 0
        binned[~non_zero_mask] = 0.0
        
        # Feature 2: Continuous percentile rank
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = ranks / max(1.0, float(len(raw) - 1))
        
        # Stack
        pair = np.stack([binned, pct], axis=-1)
        
        print(f"[PTD-Binned] bins used: {len(np.unique(binned))}, "
            f"non-zero: {100*non_zero_mask.mean():.1f}%")
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)    
        # Modification for your main model's tem_conv2
    
    def set_ptd_vector3(self, ptd_vec_np):
        """
        set_ptd_vector_robust
        Robust encoding that generalizes across datasets.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Soft clip outliers using log compression for extreme values
        if (raw > 0).sum() > 10:
            p90 = np.percentile(raw[raw > 0], 90)
            # Soft clipping: linear up to p90, log-compressed above
            raw = np.where(raw > p90, p90 + np.log1p(raw - p90), raw)
        
        # Feature 1: Robust rank transform (always works regardless of distribution)
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_normalized = 2.0 * (ranks / max(1.0, len(ranks) - 1)) - 1.0  # Scale to [-1, 1]
        
        # Feature 2: Sign-preserving square root (expands small values)
        sqrt_transform = np.sign(raw) * np.sqrt(np.abs(raw))
        
        # Robust normalization using median and MAD
        if (raw > 0).sum() > 5:
            non_zero_sqrt = sqrt_transform[raw > 0]
            median = np.median(non_zero_sqrt)
            mad = np.median(np.abs(non_zero_sqrt - median))
            scale = max(mad * 1.4826, 0.1)  # Robust std estimate
            
            # Normalize non-zero values
            sqrt_normalized = np.zeros_like(sqrt_transform)
            sqrt_normalized[raw > 0] = (sqrt_transform[raw > 0] - median) / scale
            sqrt_normalized[raw == 0] = -1.0  # Clear signal for zeros
        else:
            sqrt_normalized = np.zeros_like(raw) - 1.0
        
        # Clip to reasonable range
        sqrt_normalized = np.clip(sqrt_normalized, -2.0, 2.0)
        
        # Stack features
        pair = np.stack([sqrt_normalized, rank_normalized], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)
    
    def set_ptd_vector(self, ptd_vec_np):
        """
        Simplest robust encoding using only ranks.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Normalized ranks
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_norm = ranks / max(1.0, len(ranks) - 1)  # [0, 1]
        
        # Feature 2: Binary indicator for non-zero
        is_nonzero = (raw > 0).astype(np.float32)
        
        # Combine: use rank for non-zero, -0.5 for zeros
        combined = np.where(raw > 0, rank_norm, -0.5)
        
        # Stack with pure rank (for percentile info)
        pair = np.stack([combined, rank_norm], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Enhanced tem_conv2 that passes PTD table info to attention modules.
        Replace the tem_conv2 method in your TATKC_PTD_20Aug class.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")
        
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Get features and embeddings as before
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)
        
        if curr_layers == 0:
            return src_node_feat
        
        # Recursive call
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
        ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Enhanced attention with PTD table info
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _ = attn_m(
            src=src_node_conv_feat,
            src_t=src_node_t_embed,
            src_ptd=ptd_embed_src,
            seq=ngh_conv,
            seq_t=ngh_t_enc,
            seq_ptd=ptd_ngh_enc,
            mask=mask,

        )

        
        return F.normalize(local, p=2, dim=1)
    

class TATKC_PTD_19Sep_Gated_GPT(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC_PTD_19Sep_Gated_GPT, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([AttnModelPTD_AdaptivePrior_GPT(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_enc_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vector(self, ptd_vec_np, cap_quantile: float = 0.995):
        """
        Build PTD tables for the current snapshot time t_cut.

        - Safe heavy-tail squash: log1p only applied on (raw > cap) entries.
        - Robust z-score on log1p(raw).
        - Percentile from RAW counts.
        """
        import numpy as np
        import torch

        # ----- raw -----
        raw = np.asarray(ptd_vec_np, dtype=np.float32)
        # sanitize
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.clip(raw, 0.0, None)

        # ----- heavy-tail squash (SAFE: mask before log1p) -----
        if (raw > 0).any():
            cap = float(np.quantile(raw[raw > 0], cap_quantile))
            if cap > 0.0:
                adj = raw.copy()
                above = adj > cap
                if np.any(above):
                    delta = adj[above] - cap          # >= 0
                    adj[above] = cap + np.log1p(delta)  # safe
                raw_squashed = adj
            else:
                raw_squashed = raw
        else:
            raw_squashed = raw

        # ----- log1p(raw) for robust stats (no NaNs) -----
        logv = np.log1p(raw_squashed).astype(np.float32)

        # ----- robust stats (use train stats if available) -----
        use_train_stats = hasattr(self, "ptd_stats") and isinstance(self.ptd_stats, dict) and \
                        ("log_median" in self.ptd_stats) and ("log_mad" in self.ptd_stats)

        if use_train_stats:
            log_med = float(self.ptd_stats["log_median"])
            log_mad = float(self.ptd_stats["log_mad"])
            if not np.isfinite(log_mad) or log_mad <= 0:
                nz = logv[raw > 0]
                if nz.size > 5:
                    log_med = float(np.median(nz))
                    log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
                else:
                    log_med = float(np.median(logv))
                    log_mad = float(np.std(logv) + 1e-6)
        else:
            nz = logv[raw > 0]
            if nz.size > 5:
                log_med = float(np.median(nz))
                log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
            else:
                log_med = float(np.median(logv))
                log_mad = float(np.std(logv) + 1e-6)

        zlog = (logv - log_med) / log_mad
        zlog = np.clip(zlog, -5.0, 5.0).astype(np.float32)

        # percentile on RAW (stable, unitless)
        N = float(len(raw))
        if N > 0:
            ranks = raw.argsort().argsort().astype(np.float32)
            pct = (ranks + 0.5) / max(1.0, N)
        else:
            pct = np.zeros_like(raw, dtype=np.float32)

        pair = np.stack([zlog, pct], axis=-1).astype(np.float32)

        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_log_col = torch.from_numpy(np.log1p(raw).astype(np.float32)).to(dev).requires_grad_(False)
        self.ptd_vec     = torch.from_numpy(pair).to(dev).requires_grad_(False)

        # (optional) keep last-used stats

        self._ptd_stats_last = {"log_median": log_med, "log_mad": log_mad}


    # Modification for your main model's tem_conv2
    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Enhanced tem_conv2 that passes PTD table info to attention modules.
        Replace the tem_conv2 method in your TATKC_PTD_20Aug class.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")
        
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Get features and embeddings as before
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)
        
        if curr_layers == 0:
            return src_node_feat
        
        # Recursive call
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
        ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Enhanced attention with PTD table info
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _ = attn_m(
            src=src_node_conv_feat, src_t=src_node_t_embed, src_ptd_enc=ptd_embed_src,
            seq=ngh_conv,           seq_t=ngh_t_enc,        seq_ptd_enc=ptd_ngh_enc,
            mask=mask,
            src_ptd_raw=ptd_src_vals, seq_ptd_raw=ptd_ngh_vals
        )

        
        return F.normalize(local, p=2, dim=1)
 


class TATKC_PTD_19Sep_Gated_claude_GPT_version(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC_PTD_19Sep_Gated_claude_GPT_version, self).__init__()

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

        #AttnModelPTD_FinalAdaptive_GPT_version
        #AttnModelPTD_FinalAdaptiveV3_GPT
        self.attn_model_list = torch.nn.ModuleList([AttnModelPTD_FinalAdaptiveV3_GPT(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_enc_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vector1(self, ptd_vec_np):
        """
        Final optimized PTD encoding for extremely sparse data (85% zeros).
        This version maximizes diversity in encoded features.
        """
        import numpy as np
        import torch
        
        # Get raw PTD
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Print input stats (remove in production)
        print(f"[PTD-Input] zeros={100*(raw==0).mean():.1f}% "
            f"max={raw.max():.1f} p95={(np.percentile(raw[raw>0], 95) if (raw>0).any() else 0):.1f}")
        
        # Clip outliers more aggressively for stability
        if (raw > 0).sum() > 10:
            # Use 90th percentile instead of 95th
            cap = np.percentile(raw[raw > 0], 90)
            raw = np.minimum(raw, cap)
        
        # FEATURE 1: Rank-based encoding (guaranteed diversity)
        # This preserves relative ordering perfectly
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = ranks / max(1.0, float(len(raw) - 1))  # Normalize to [0, 1]
        
        # FEATURE 2: Power transform instead of log for better spread
        # Power < 1 expands small values, which is what we need
        power = 0.25  # Fourth root - expands small values more than log
        
        # Create the magnitude feature
        magnitude = np.zeros_like(raw)
        non_zero_mask = raw > 0
        
        if non_zero_mask.sum() > 0:
            # Apply power transform to non-zero values
            nz_vals = raw[non_zero_mask]
            transformed = np.power(nz_vals, power)
            
            # Normalize to roughly [-1, 2] range
            # Use robust statistics
            p50 = np.percentile(transformed, 50)
            p90 = np.percentile(transformed, 90)
            
            if p90 > p50:
                # Scale based on upper quartile range
                scaled = (transformed - p50) / (p90 - p50)
                magnitude[non_zero_mask] = scaled
            else:
                # Fallback: simple normalization
                magnitude[non_zero_mask] = transformed / (transformed.max() + 1e-6)
        
        # Mark zeros distinctly (important for model to distinguish)
        magnitude[~non_zero_mask] = -0.5
        
        # Create 3-feature encoding for better expressiveness
        # Feature 3: Binary indicator (helps model identify patterns)
        is_nonzero = non_zero_mask.astype(np.float32)
        
        # Stack features - but for compatibility, keep 2D
        # Combine binary indicator with magnitude
        enhanced_magnitude = magnitude + 0.1 * is_nonzero  # Slight boost for non-zeros
        
        pair = np.stack([enhanced_magnitude, pct], axis=-1)
        
        # Print encoding stats
        unique_mag = len(np.unique(np.round(enhanced_magnitude, 3)))
        print(f"[PTD-Encoded] magnitude: std={enhanced_magnitude[non_zero_mask].std() if non_zero_mask.any() else 0:.3f} "
            f"unique={100*unique_mag/len(enhanced_magnitude):.1f}% "
            f"range=[{enhanced_magnitude.min():.2f},{enhanced_magnitude.max():.2f}]")
        
        # Move to device
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)
        
        # Store additional info
        self.ptd_log_col = torch.from_numpy(np.log1p(raw)).to(dev)
        self.ptd_nonzero_mask = torch.from_numpy(is_nonzero).to(dev)
        

    def set_ptd_vector2(self, ptd_vec_np):
        """
        Alternative: Use binned encoding for better diversity.
        This creates discrete bins which guarantees diversity.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Create binned encoding
        # Bin 0: zeros
        # Bins 1-9: non-zero values split by percentiles
        binned = np.zeros_like(raw)
        non_zero_mask = raw > 0
        
        if non_zero_mask.sum() > 10:
            nz_vals = raw[non_zero_mask]
            
            # Create 9 bins for non-zero values using percentiles
            #percentiles = np.percentile(nz_vals, [11, 22, 33, 44, 55, 66, 77, 88, 100])
            percentiles = np.percentile(nz_vals, [20, 40, 60, 80, 100])
            
            # Assign bins
            for i, p in enumerate(percentiles):
                mask = (raw > (percentiles[i-1] if i > 0 else 0)) & (raw <= p)
                binned[mask] = (i + 1) / 10.0  # Normalize to [0.1, 0.9]
        
        # Zeros stay at 0
        binned[~non_zero_mask] = 0.0
        
        # Feature 2: Continuous percentile rank
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = ranks / max(1.0, float(len(raw) - 1))
        
        # Stack
        pair = np.stack([binned, pct], axis=-1)
        
        print(f"[PTD-Binned] bins used: {len(np.unique(binned))}, "
            f"non-zero: {100*non_zero_mask.mean():.1f}%")
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)    
        # Modification for your main model's tem_conv2
    
    def set_ptd_vector3(self, ptd_vec_np):
        """
        set_ptd_vector_robust
        Robust encoding that generalizes across datasets.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Soft clip outliers using log compression for extreme values
        if (raw > 0).sum() > 10:
            p90 = np.percentile(raw[raw > 0], 90)
            # Soft clipping: linear up to p90, log-compressed above
            raw = np.where(raw > p90, p90 + np.log1p(raw - p90), raw)
        
        # Feature 1: Robust rank transform (always works regardless of distribution)
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_normalized = 2.0 * (ranks / max(1.0, len(ranks) - 1)) - 1.0  # Scale to [-1, 1]
        
        # Feature 2: Sign-preserving square root (expands small values)
        sqrt_transform = np.sign(raw) * np.sqrt(np.abs(raw))
        
        # Robust normalization using median and MAD
        if (raw > 0).sum() > 5:
            non_zero_sqrt = sqrt_transform[raw > 0]
            median = np.median(non_zero_sqrt)
            mad = np.median(np.abs(non_zero_sqrt - median))
            scale = max(mad * 1.4826, 0.1)  # Robust std estimate
            
            # Normalize non-zero values
            sqrt_normalized = np.zeros_like(sqrt_transform)
            sqrt_normalized[raw > 0] = (sqrt_transform[raw > 0] - median) / scale
            sqrt_normalized[raw == 0] = -1.0  # Clear signal for zeros
        else:
            sqrt_normalized = np.zeros_like(raw) - 1.0
        
        # Clip to reasonable range
        sqrt_normalized = np.clip(sqrt_normalized, -2.0, 2.0)
        
        # Stack features
        pair = np.stack([sqrt_normalized, rank_normalized], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)
    
    def set_ptd_vector(self, ptd_vec_np):
        """
        Simplest robust encoding using only ranks.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Normalized ranks
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_norm = ranks / max(1.0, len(ranks) - 1)  # [0, 1]
        
        # Feature 2: Binary indicator for non-zero
        is_nonzero = (raw > 0).astype(np.float32)
        
        # Combine: use rank for non-zero, -0.5 for zeros
        combined = np.where(raw > 0, rank_norm, -0.5)
        
        # Stack with pure rank (for percentile info)
        pair = np.stack([combined, rank_norm], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Enhanced tem_conv2 that passes PTD table info to attention modules.
        Replace the tem_conv2 method in your TATKC_PTD_20Aug class.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")
        
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Get features and embeddings as before
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)
        
        if curr_layers == 0:
            return src_node_feat
        
        # Recursive call
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
        #ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Enhanced attention with PTD table info
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _ = attn_m(
            src=src_node_conv_feat,
            src_t=src_node_t_embed,
            src_ptd=ptd_embed_src,
            seq=ngh_conv,
            seq_t=ngh_t_enc,
            seq_ptd=ptd_ngh_enc,
            mask=mask,
            src_ptd_raw =ptd_src_vals,
            seq_ptd_raw= ptd_ngh_vals

        )
        return F.normalize(local, p=2, dim=1)
  

class AttnModelPTD_FinalAdaptive_GPT_version(nn.Module):
    """
    Monotone, robust prior:
      - Prior = softmax( alpha * zscore( s * raw_col0 ) ) over VALID neighbors
      - s is learned positive scalar (softplus)
      - If valid_count <= 1 for a row => prior=uniform (attn_bias=0)
      - Alpha adapts from nz_frac and rowwise std of raw_col0 (after masking)
    """
    def __init__(self, feat_dim, time_dim, ptd_enc_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2,
                 alpha_min=0.10, alpha_max=0.40, bias_clip=3.0, diag_every=200):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        self.model_dim = feat_dim + time_dim + ptd_enc_dim
        assert self.model_dim % n_head == 0

        # positive scalar to scale RAW col0 (monotone mapping)
        self._raw_scale = nn.Parameter(torch.tensor(0.0))  # softplus(_) >= 0
        self.bias_clip = float(bias_clip)

        from module import MergeLayer, MultiHeadAttention
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        if attn_mode != 'prod':
            raise ValueError('Only "prod" attention is supported.')
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )

        # alpha schedule
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # diagnostics
        self.diag_every = int(diag_every)
        self._diag_step = 0
        self.alpha_attn = 0.0
        self.alpha_history, self.ptd_quality_history = [], []
        self._epoch = {k: [] for k in
            ("alpha","bias_z_std","bias_z_mean","attn_entropy","kl_p_prior",
             "rho_attn_ptd","valid_frac","nz_frac","valid_cnt")}

    # ---------- utils ----------
    @staticmethod
    def _masked_mean_std(x, valid, eps=1e-6):
        v = valid.float()
        cnt = v.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(-1, keepdim=True) / cnt
        var  = ((x - mean)**2 * v).sum(-1, keepdim=True) / cnt
        std  = var.clamp_min(eps).sqrt()
        return mean, std, cnt

    @staticmethod
    def _masked_entropy(p, valid, eps=1e-9):
        v = valid.float()
        p = p * v
        p = p / p.sum(-1, keepdim=True).clamp_min(eps)
        H = (-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(-1))
        return H

    @staticmethod
    def _masked_spearman(x, y, valid, eps=1e-9):
        v = valid.float()
        # sentinel for pads then rank
        xm = torch.where(valid, x, x.min(dim=-1, keepdim=True).values - 1.0)
        ym = torch.where(valid, y, y.min(dim=-1, keepdim=True).values - 1.0)
        rx = xm.argsort(-1).argsort(-1).float()
        ry = ym.argsort(-1).argsort(-1).float()
        cnt = v.sum(-1, keepdim=True).clamp_min(1.0)
        mx = (rx*v).sum(-1, keepdim=True)/cnt; my = (ry*v).sum(-1, keepdim=True)/cnt
        sx = (((rx-mx)**2*v).sum(-1, keepdim=True)/cnt).clamp_min(eps).sqrt()
        sy = (((ry-my)**2*v).sum(-1, keepdim=True)/cnt).clamp_min(eps).sqrt()
        zxr = (rx-mx)/sx; zyr=(ry-my)/sy
        rho = ((zxr*zyr*v).sum(-1)/cnt.squeeze(-1)).clamp(-1,1)
        return rho

    def _alpha_from_quality(self, raw_col0, valid):
        # robust “quality”: mix of nz fraction + rowwise std
        mean, std, cnt = self._masked_mean_std(raw_col0, valid)  # [B,1]s
        nz = ((raw_col0 > -0.4) & valid).float().sum(-1, keepdim=True)
        denom = cnt.clamp_min(1.0)
        nz_frac = (nz / denom).clamp(0.0, 1.0)
        # normalize std to [0,1] by a soft cap ~ 0.25
        std_q = (std / 0.25).clamp(0.0, 1.0)
        q = (0.6 * nz_frac + 0.4 * std_q).clamp(0.0, 1.0)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * q
        return alpha, std, nz_frac, denom

    # ---------- forward ----------
    def forward(self,
                src, src_t, src_ptd,
                seq, seq_t, seq_ptd,
                mask,
                src_ptd_raw=None,     # [B,2]  -> REQUIRED for best results
                seq_ptd_raw=None,     # [B,N,2]
                return_logits=False):

        src_ptd_enc = src_ptd
        seq_ptd_enc = seq_ptd
        B, N, _ = seq.shape
        mask3 = mask.unsqueeze(1) if mask.dim()==2 else mask
        valid = ~(mask if mask.dim()==2 else mask.squeeze(1))  # [B,N]

        # RAW fallbacks (if you forgot to pass them, we'll proxy from enc)
        if src_ptd_raw is None:
            src_ptd_raw = (src_ptd_enc[..., :2].detach()
                           if src_ptd_enc.size(-1) >= 2 else torch.zeros(B,2, device=seq.device))
        if seq_ptd_raw is None:
            seq_ptd_raw = (seq_ptd_enc[..., :2].detach()
                           if seq_ptd_enc.size(-1) >= 2 else torch.zeros(B,N,2, device=seq.device))

        # ----- content (unchanged) -----
        q = torch.cat([src.unsqueeze(1),  src_t,  src_ptd_enc.unsqueeze(1)], dim=2)
        k = torch.cat([seq,               seq_t,  seq_ptd_enc],             dim=2)
        v = k

        # ----- PRIOR from RAW col0 (monotone) -----
        col0 = seq_ptd_raw[..., 0]                          # [B,N]
        s = F.softplus(self._raw_scale) + 1e-6              # positive
        scaled = s * col0

        # per-row zscore on VALID; if <=1 valid → zero prior
        mean, std, cnt = self._masked_mean_std(scaled, valid)
        few_valid = (cnt.squeeze(-1) <= 1.5)                # [B]
        z = torch.where(valid, (scaled - mean)/std, torch.zeros_like(scaled))
        z = z.clamp(-self.bias_clip, self.bias_clip)
        z[few_valid] = 0.0                                  # neutral when too few neighbors

        alpha_row, std_row, nz_frac, denom = self._alpha_from_quality(col0, valid)  # [B,1]
        attn_bias = alpha_row.unsqueeze(1) * z.unsqueeze(1)  # [B,1,N]

        # ----- attention -----
        out, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask3, attn_bias=attn_bias, return_logits=False)
        out = out.squeeze(1); attn = attn.squeeze(1)
        output = self.merger(out, src)

        # ----- diagnostics -----
        with torch.no_grad():
            vfloat = valid.float()
            # renormalize attn on valid
            p = (attn * vfloat); p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)

            # prior distribution implied by z (no model logits)
            prior_logits = alpha_row * z + (~valid) * (-1e9)
            prior = torch.softmax(prior_logits, dim=-1)
            prior = (prior * vfloat); prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-6)

            H = self._masked_entropy(p, valid).mean().item()
            kl = (p.clamp_min(1e-8)*(p.clamp_min(1e-8).log()-prior.clamp_min(1e-8).log())).sum(-1).mean().item()
            rho = self._masked_spearman(col0, p, valid).mean().item()

            # bias stats across valid per row
            b_mean = (z * vfloat).sum(-1) / vfloat.sum(-1).clamp_min(1.0)
            b_std  = (((z - b_mean.unsqueeze(-1))**2 * vfloat).sum(-1) /
                      vfloat.sum(-1).clamp_min(1.0)).clamp_min(1e-6).sqrt().mean().item()

            self.alpha_attn = float(alpha_row.mean().cpu())
            self.alpha_history.append(self.alpha_attn)
            self.ptd_quality_history.append(float(std_row.mean().cpu()))

            self._epoch["alpha"].append(self.alpha_attn)
            self._epoch["bias_z_std"].append(b_std)
            self._epoch["bias_z_mean"].append(float(b_mean.mean().cpu()))
            self._epoch["attn_entropy"].append(H)
            self._epoch["kl_p_prior"].append(kl)
            self._epoch["rho_attn_ptd"].append(rho)
            self._epoch["valid_frac"].append(float((vfloat.mean(-1)).mean().cpu()))
            self._epoch["nz_frac"].append(float(nz_frac.mean().cpu()))
            self._epoch["valid_cnt"].append(float(cnt.mean().cpu()))

            if (self._diag_step % self.diag_every) == 0:
                print(f"[Adaptive] std={float(std_row.mean()):.3f} alpha={self.alpha_attn:.3f} "
                      f"H={H:.3f} KL={kl:.3f} rho={rho:.3f} bias_std={b_std:.3f} "
                      f"valid_cnt={float(cnt.mean()):.2f} nz={float(nz_frac.mean()):.3f}")
            self._diag_step += 1

        if return_logits:
            return output, attn, None
        return output, attn

    def dump_epoch_diag_and_reset(self, epoch: int, prefix: str = ""):
        if not self._epoch["alpha"]:
            print(f"{prefix}[Diag] epoch {epoch:02d}: no stats.")
            return
        import numpy as np
        agg = {k: float(np.mean(v)) for k,v in self._epoch.items() if v}
        print(
            f"{prefix}[Diag] epoch {epoch:02d} | α={agg['alpha']:.3f}  "
            f"H={agg['attn_entropy']:.3f}  KL={agg['kl_p_prior']:.3f}  "
            f"ρ(attn,PTD)={agg['rho_attn_ptd']:.3f}  biasσ={agg['bias_z_std']:.3f}  "
            f"valid_cnt={agg['valid_cnt']:.2f}  valid_frac={agg['valid_frac']:.3f}  nz={agg['nz_frac']:.3f}"
        )
        for k in self._epoch: self._epoch[k].clear()

 

class AttnModelPTD_AdaptivePrior_GPT(nn.Module):



    """
    PTD used ONLY as a soft attention prior (logit bias).
    Content = [feat || time || ptd_enc]; no value gating.
    Alpha (prior strength) is learned-adaptive from a robust, per-batch PTD quality in [0,1].
    """
    def __init__(self, feat_dim, time_dim, ptd_enc_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2,
                 alpha_min=0.05, alpha_max=0.30,  # <<< much lower than 0.8+
                 bias_clip=2.0):
        super().__init__()
        assert attn_mode == 'prod'
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        self.model_dim = feat_dim + time_dim + ptd_enc_dim
        assert self.model_dim % n_head == 0

        # prior from percentile only (seq_ptd_raw[...,1] ∈ (0,1])
        self.bias_clip = float(bias_clip)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self._alpha_last = float((alpha_min + alpha_max) / 2)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )

        self.diag_enabled, self.diag_every, self.diag_max_rows = True, 200, 64
        self._diag_step = 0

    @staticmethod
    def _clean(x: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        return x.clamp(-clip, clip)

    @staticmethod
    def _masked_std(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x:[B,N], valid_mask:[B,N] True=VALID
        v    = valid_mask.float()
        cnt  = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        var  = (((x - mean) * v).pow(2)).sum(dim=1, keepdim=True) / cnt
        return torch.sqrt(var + eps).squeeze(1)  # [B]

    @staticmethod
    def _masked_zscore(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x:[B,N], valid_mask:[B,N] True=VALID
        v    = valid_mask.float()
        cnt  = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        xc   = (x - mean) * v
        var  = (xc * xc).sum(dim=1, keepdim=True) / cnt
        std  = torch.sqrt(var + eps)
        z    = torch.where(valid_mask, (x - mean) / std, torch.zeros_like(x))
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _quality_from_percentile(self, pct: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        pct:   [B,N] in (0,1]; valid: [B,N] (True=VALID)
        returns scalar quality in [0,1] (detached)
        """
        # Row-wise stats on valid neighbors
        std_row = self._masked_std(pct, valid)                 # [B]
        # how many neighbors have non-trivial PTD (pct > 0.01)
        non_zero_frac = ((pct > 0.01) & valid).float().mean(dim=1)  # [B]

        # normalize std into ~[0,1]; thresholds tuned to be forgiving
        # 0.03 ~ tiny dispersion; 0.20 ~ good dispersion
        std_norm = torch.clamp((std_row - 0.03) / 0.17, 0.0, 1.0)   # [B]
        q = 0.6 * std_norm + 0.4 * non_zero_frac                    # convex combo
        q = q.mean().detach()                                       # batch scalar
        return q

    def forward(self,
                src, src_t, src_ptd_enc,
                seq, seq_t, seq_ptd_enc,
                mask,
                src_ptd_raw=None,   # [B,2] -> we only use percentile channel in bias
                seq_ptd_raw=None,   # [B,N,2]
                return_logits: bool = False):

        B, N, _ = seq.size()
        mask_3d  = mask.unsqueeze(1) if mask.dim() == 2 else mask     # [B,1,N] True=PAD
        pad_bn   = mask if mask.dim() == 2 else mask.squeeze(1)       # [B,N]
        valid_bn = ~pad_bn                                            # [B,N]
        has_valid = valid_bn.any(dim=1)

        # sanitize content
        src         = self._clean(src);         src_t       = self._clean(src_t)
        src_ptd_enc = self._clean(src_ptd_enc)
        seq         = self._clean(seq);         seq_t       = self._clean(seq_t)
        seq_ptd_enc = self._clean(seq_ptd_enc)

        # raw PTD fallbacks
        if seq_ptd_raw is None:
            # if enc has >=2, use enc's first 2 dims as weak proxy
            if seq_ptd_enc.size(-1) >= 2:
                seq_ptd_raw = seq_ptd_enc[..., :2].detach()
            else:
                seq_ptd_raw = torch.zeros(B, N, 2, device=seq.device)
        if src_ptd_raw is None:
            if src_ptd_enc.size(-1) >= 2:
                src_ptd_raw = src_ptd_enc[..., :2].detach()
            else:
                src_ptd_raw = torch.zeros(B, 2, device=src.device)

        # percentile channel (bounded; safer for prior)
        pct = torch.nan_to_num(seq_ptd_raw[..., 1], nan=0.0).clamp(0.0, 1.0)  # [B,N]
        pct = pct * valid_bn.float()  # zero out pads

        # --- adaptive alpha from per-batch quality ---
        quality = self._quality_from_percentile(pct, valid_bn)     # scalar in [0,1]
        alpha   = self.alpha_min + (self.alpha_max - self.alpha_min) * quality
        self._alpha_last = float(alpha)

        # content
        vmask = valid_bn.unsqueeze(-1).float()
        k_content = self._clean(torch.cat([seq * vmask, self._clean(seq_t * vmask), self._clean(seq_ptd_enc * vmask)], dim=2))
        v_content = k_content
        q_content = self._clean(torch.cat([src.unsqueeze(1), src_t, src_ptd_enc.unsqueeze(1)], dim=2))

        # if a row has no content after masking, skip attention
        content_signal = (k_content.abs().sum(dim=(1, 2)) > 0)   # [B]
        use_attn_row   = has_valid & content_signal
        out_feat = src.new_zeros(B, self.feat_dim)
        attn_out = src.new_zeros(B, N)

        # prior bias = z-scored percentile (over valid neighbors), clipped & scaled by α
        bias_z = self._masked_zscore(pct, valid_bn).clamp(-self.bias_clip, self.bias_clip)  # [B,N]
        attn_bias_full = (alpha * bias_z).unsqueeze(1)                                       # [B,1,N]

        idx_valid = torch.nonzero(use_attn_row, as_tuple=True)[0]
        idx_empty = torch.nonzero(~use_attn_row, as_tuple=True)[0]

        if idx_valid.numel() > 0:
            qv = q_content.index_select(0, idx_valid)
            kv = k_content.index_select(0, idx_valid)
            vv = v_content.index_select(0, idx_valid)
            mv = mask_3d.index_select(0, idx_valid)
            bv = attn_bias_full.index_select(0, idx_valid)
            out_v, attn_v = self.multi_head_target(q=qv, k=kv, v=vv, mask=mv, attn_bias=bv, return_logits=False)
            merged_v = self.merger(out_v.squeeze(1), src.index_select(0, idx_valid))
            out_feat.index_copy_(0, idx_valid, self._clean(merged_v))
            attn_out.index_copy_(0, idx_valid, self._clean(attn_v.squeeze(1)))

        if idx_empty.numel() > 0:
            zeros_model = src.new_zeros(idx_empty.numel(), self.model_dim)
            merged_e = self.merger(zeros_model, src.index_select(0, idx_empty))
            out_feat.index_copy_(0, idx_empty, self._clean(merged_e))
            attn_out.index_copy_(0, idx_empty, src.new_zeros(idx_empty.numel(), N))

        # light diag
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            with torch.no_grad():
                nz_frac = (pct > 0.01).float().mean().item()
                std_mean = self._masked_std(pct, valid_bn).mean().item()
                print(f"[AdaptivePrior] alpha={self._alpha_last:.3f} quality={float(quality):.3f} "
                      f"std_mean={std_mean:.3f} nz_frac={nz_frac:.3f}")

        self._diag_step += 1

        if return_logits:
            return out_feat, attn_out, None
        return out_feat, attn_out
    
class AttnModelPTD_FinalAdaptiveV3_GPT(nn.Module):
    """
    Robust prior with alignment-aware alpha:
      - Prior from RAW col0 (monotone via positive scale).
      - α is per-row and shrinks when:
          * very few valid neighbors,
          * PTD non-zeros are rare or RAW variance is tiny,
          * RAW PTD disagrees with a cosine-similarity content proxy.
      - Falls back to neutral prior for rows with ≤2 valid neighbors.
    """
    def __init__(self, feat_dim, time_dim, ptd_enc_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2,
                 alpha_min=0.10, alpha_max=0.40, bias_clip=3.0, diag_every=200):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        self.model_dim = feat_dim + time_dim + ptd_enc_dim
        assert self.model_dim % n_head == 0

        # strictly-positive scale for RAW PTD → prior (keeps monotonicity)
        self._raw_scale = nn.Parameter(torch.tensor(0.0))  # softplus(_) >= 0
        self.bias_clip = float(bias_clip)

        from module import MergeLayer, MultiHeadAttention
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        if attn_mode != 'prod':
            raise ValueError('Only "prod" attention is supported.')
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # diagnostics
        self.diag_every = int(diag_every)
        self._diag_step = 0
        self.alpha_attn = 0.0
        self._epoch = {k: [] for k in
            ("alpha","bias_z_std","attn_entropy","kl_p_prior",
             "rho_attn_ptd","valid_cnt","valid_frac","nz_frac")}

    # ---------- helpers ----------
    @staticmethod
    def _masked_mean_std(x, valid, eps=1e-6):
        v = valid.float()
        cnt = v.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(-1, keepdim=True) / cnt
        var  = ((x - mean)**2 * v).sum(-1, keepdim=True) / cnt
        std  = var.clamp_min(eps).sqrt()
        return mean, std, cnt

    @staticmethod
    def _masked_entropy(p, valid, eps=1e-9):
        v = valid.float()
        p = p * v
        p = p / p.sum(-1, keepdim=True).clamp_min(eps)
        return (-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(-1))

    @staticmethod
    def _masked_pearson(x, y, valid, eps=1e-9):
        """Pearson corr over valid positions, per-row, then mean."""
        v = valid.float()
        cnt = v.sum(-1, keepdim=True).clamp_min(1.0)
        xm = (x*v).sum(-1, keepdim=True) / cnt
        ym = (y*v).sum(-1, keepdim=True) / cnt
        xs = (((x-xm)**2)*v).sum(-1, keepdim=True).clamp_min(eps).sqrt()
        ys = (((y-ym)**2)*v).sum(-1, keepdim=True).clamp_min(eps).sqrt()
        zx = (x-xm)/xs; zy = (y-ym)/ys
        r  = ((zx*zy*v).sum(-1, keepdim=True) / cnt).squeeze(-1)
        return r  # [B], clipped by caller if needed

    def _alpha_base(self, raw_col0, valid):
        # base quality from nz fraction + raw std (both per-row)
        _, std, cnt = self._masked_mean_std(raw_col0, valid)             # [B,1]
        nz = ((raw_col0 > -0.4) & valid).float().sum(-1, keepdim=True)   # [B,1]
        denom = cnt.clamp_min(1.0)
        nz_frac = (nz / denom).clamp(0,1)                                # [B,1]
        std_q = (std / 0.25).clamp(0,1)                                  # [B,1] soft cap
        q = (0.6 * nz_frac + 0.4 * std_q).clamp(0,1)                     # [B,1]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * q   # [B,1]
        return alpha, std, nz_frac, cnt

    # ---------- forward ----------
    def forward(self,
                src, src_t, src_ptd,
                seq, seq_t, seq_ptd,
                mask,
                src_ptd_raw=None,     # [B,2]
                seq_ptd_raw=None,     # [B,N,2]
                return_logits=False):

        src_ptd_enc = src_ptd
        seq_ptd_enc = seq_ptd

        B, N, _ = seq.shape
        mask3 = mask.unsqueeze(1) if mask.dim()==2 else mask
        valid = ~(mask if mask.dim()==2 else mask.squeeze(1))  # [B,N]
        vfloat = valid.float()

        # RAW fallbacks (proxy from enc if not provided)
        if src_ptd_raw is None:
            src_ptd_raw = (src_ptd_enc[..., :2].detach()
                           if src_ptd_enc.size(-1) >= 2 else torch.zeros(B,2, device=seq.device))
        if seq_ptd_raw is None:
            seq_ptd_raw = (seq_ptd_enc[..., :2].detach()
                           if seq_ptd_enc.size(-1) >= 2 else torch.zeros(B,N,2, device=seq.device))

        # ---- content path (as before) ----
        q = torch.cat([src.unsqueeze(1),  src_t,  src_ptd_enc.unsqueeze(1)], dim=2)
        k = torch.cat([seq,               seq_t,  seq_ptd_enc],             dim=2)
        v = k

        # ---- prior from RAW col0 (monotone) ----
        col0 = seq_ptd_raw[..., 0]                         # [B,N]
        s = F.softplus(self._raw_scale) + 1e-6
        scaled = s * col0

        mean, std, cnt = self._masked_mean_std(scaled, valid)
        few_valid = (cnt.squeeze(-1) <= 2.0)               # stricter: need ≥3 to trust prior
        z = torch.where(valid, (scaled - mean)/std, torch.zeros_like(scaled))
        z = z.clamp(-self.bias_clip, self.bias_clip)
        z[few_valid] = 0.0

        # ---- alpha with robustness & alignment gates (per-row) ----
        alpha_row, std_row, nz_frac, cnt_row = self._alpha_base(col0, valid)  # [B,1]

        # (1) valid-count gate: saturates toward 1 as (cnt-1) grows
        m_valid = ((cnt_row - 1.0).clamp_min(0.0) / ((cnt_row - 1.0) + 2.0))

        # (2) nonzero gate (mild): sqrt to soften
        m_nz = nz_frac.sqrt().clamp(0,1)

        # (3) bias-shape gate: if z is near-constant, shrink alpha
        #     use per-row std of z across valid positions
        z_mean, z_std, _ = self._masked_mean_std(z, valid)
        m_bias = (z_std / 0.5).clamp(0, 1.0)

        # (4) alignment gate: cosine-similarity proxy vs RAW PTD
        sim = F.cosine_similarity(src.unsqueeze(1), seq, dim=-1)  # [B,N]
        corr = self._masked_pearson(sim, col0, valid).unsqueeze(-1)  # [B,1]
        m_align = corr.clamp_min(0.0)                               # negative → 0
        # soften alignment: map [0,1] → [0.4,1] to avoid killing alpha too hard
        m_align = 0.4 + 0.6 * m_align

        alpha_row = alpha_row * m_valid * m_nz * m_bias * m_align
        alpha_row = alpha_row.clamp(min=0.0, max=self.alpha_max)    # allow 0 floor

        attn_bias = alpha_row.unsqueeze(1) * z.unsqueeze(1)         # [B,1,N]

        # ---- attention ----
        out, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask3, attn_bias=attn_bias, return_logits=False)
        out = out.squeeze(1); attn = attn.squeeze(1)
        output = self.merger(out, src)

        # ---- diagnostics ----
        with torch.no_grad():
            # renormalize attn on valid
            p = (attn * vfloat); p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)

            # implied prior distribution
            prior_logits = alpha_row * z + (~valid) * (-1e9)
            prior = torch.softmax(prior_logits, dim=-1)
            prior = (prior * vfloat); prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-6)

            H = self._masked_entropy(p, valid).mean().item()
            kl = (p.clamp_min(1e-8)*(p.clamp_min(1e-8).log()-prior.clamp_min(1e-8).log())).sum(-1).mean().item()
            rho = self._masked_pearson(col0, p, valid).mean().item()

            # z spread across valid
            b_mean = (z * vfloat).sum(-1) / vfloat.sum(-1).clamp_min(1.0)
            b_std  = (((z - b_mean.unsqueeze(-1))**2 * vfloat).sum(-1) /
                      vfloat.sum(-1).clamp_min(1.0)).clamp_min(1e-6).sqrt().mean().item()

            self.alpha_attn = float(alpha_row.mean().cpu())
            self._epoch["alpha"].append(self.alpha_attn)
            self._epoch["bias_z_std"].append(b_std)
            self._epoch["attn_entropy"].append(H)
            self._epoch["kl_p_prior"].append(kl)
            self._epoch["rho_attn_ptd"].append(rho)
            self._epoch["valid_cnt"].append(float(cnt_row.mean().cpu()))
            self._epoch["valid_frac"].append(float(vfloat.mean().cpu()))
            self._epoch["nz_frac"].append(float(nz_frac.mean().cpu()))

            if (self._diag_step % self.diag_every) == 0:
                print(f"[Adaptive] std={float(std_row.mean()):.3f} "
                      f"alpha={self.alpha_attn:.3f} H={H:.3f} KL={kl:.3f} rho={rho:.3f} "
                      f"bias_std={b_std:.3f} valid_cnt={float(cnt_row.mean()):.2f} nz={float(nz_frac.mean()):.3f}")
            self._diag_step += 1

        if return_logits:
            return output, attn, None
        return output, attn

    def dump_epoch_diag_and_reset(self, epoch: int, prefix: str = ""):
        import numpy as np
        if not self._epoch["alpha"]:
            print(f"{prefix}[Diag] epoch {epoch:02d}: no stats.")
            return

        agg = {k: float(np.mean(v)) for k,v in self._epoch.items() if v}
        print(
            f"{prefix}[Diag] epoch {epoch:02d} | α={agg['alpha']:.3f}  "
            f"H={agg['attn_entropy']:.3f}  KL={agg['kl_p_prior']:.3f}  "
            f"ρ(attn,PTD)={agg['rho_attn_ptd']:.3f}  biasσ={agg['bias_z_std']:.3f}  "
            f"valid_cnt={agg['valid_cnt']:.2f}  valid_frac={agg['valid_frac']:.3f}  nz={agg['nz_frac']:.3f}"
        )
        for k in self._epoch: self._epoch[k].clear()   


from typing import Optional, Dict, Tuple


class AttnModelPTD_Direct(nn.Module):
    """
    Direct PTD attention model that uses raw 2-D PTD for bias computation.
    Addresses sparse neighbor issues and provides extensive debugging.
    """
    def __init__(self, feat_dim: int, time_dim: int, ptd_dim: int = 128,
                 attn_mode: str = 'prod', n_head: int = 2, drop_out: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim = ptd_dim
        
        self.model_dim = feat_dim + time_dim + ptd_dim
        assert self.model_dim % n_head == 0
        
        # Direct 2-D PTD to bias mapping (bypasses 128-D encoding)
        self.ptd2bias = nn.Linear(2, 1)
        nn.init.zeros_(self.ptd2bias.weight)
        nn.init.constant_(self.ptd2bias.weight[0, 1], 1.0)  # Percentile is more reliable
        nn.init.constant_(self.ptd2bias.weight[0, 0], 0.3)  # Magnitude less weight
        
        # Quality-aware alpha computation
        self.quality_net = nn.Sequential(
            nn.Linear(4, 16),  # [valid_ratio, ptd_std, ptd_range, non_zero_ratio]
            nn.ReLU(),
            nn.Dropout(drop_out * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Alpha range parameters
        self.alpha_min = 0.7
        self.alpha_max = 1.2
        self.alpha_default = 0.85
        
        # Components from original
        from module import MergeLayer, MultiHeadAttention
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )
        
        # Debug tracking
        self.debug_mode = True
        self.debug_every = 100
        self.debug_counter = 0
        self.debug_stats = {
            'neighbor_counts': [],
            'bias_stats': [],
            'quality_scores': [],
            'alpha_values': [],
            'attention_entropy': []
        }
    
    def compute_neighbor_quality(self, seq_ptd_raw: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute quality metrics per row based on neighbor statistics.
        
        Args:
            seq_ptd_raw: [B, N, 2] raw PTD values [magnitude, percentile]
            mask: [B, N] True for invalid neighbors
        
        Returns:
            quality: [B] quality score per sample
            stats: dict with detailed statistics
        """
        B, N, _ = seq_ptd_raw.size()
        valid_mask = ~mask  # [B, N]
        
        # 1. Valid neighbor ratio
        valid_count = valid_mask.sum(dim=1).float()  # [B]
        valid_ratio = valid_count / N  # [B]
        
        # 2. PTD diversity among valid neighbors (using percentile column)
        ptd_pct = seq_ptd_raw[:, :, 1]  # [B, N] percentile values
        ptd_pct_valid = ptd_pct.masked_fill(mask, 0)
        
        # Compute std only for valid neighbors
        valid_mean = (ptd_pct_valid * valid_mask.float()).sum(dim=1) / valid_count.clamp_min(1)
        valid_var = ((ptd_pct_valid - valid_mean.unsqueeze(1)).pow(2) * valid_mask.float()).sum(dim=1) / valid_count.clamp_min(1)
        ptd_std = torch.sqrt(valid_var + 1e-6)  # [B]
        
        # 3. PTD range (max - min among valid)
        ptd_pct_masked = ptd_pct.masked_fill(mask, -1e6)
        ptd_max = ptd_pct_masked.max(dim=1).values
        ptd_pct_masked = ptd_pct.masked_fill(mask, 1e6)
        ptd_min = ptd_pct_masked.min(dim=1).values
        ptd_range = (ptd_max - ptd_min).clamp(0, 1)  # [B]
        
        # 4. Non-zero ratio (magnitude > threshold)
        non_zero = (seq_ptd_raw[:, :, 0] > -0.4) & valid_mask  # Not the -0.5 zero marker
        non_zero_ratio = non_zero.float().sum(dim=1) / valid_count.clamp_min(1)  # [B]
        
        # Combine features for quality network
        quality_features = torch.stack([
            valid_ratio,
            ptd_std / 0.3,  # Normalize by expected std
            ptd_range,
            non_zero_ratio
        ], dim=1)  # [B, 4]
        
        # Compute quality score
        quality = self.quality_net(quality_features).squeeze(-1)  # [B]
        
        stats = {
            'valid_count': valid_count.mean().item(),
            'valid_ratio': valid_ratio.mean().item(),
            'ptd_std': ptd_std.mean().item(),
            'ptd_range': ptd_range.mean().item(),
            'non_zero_ratio': non_zero_ratio.mean().item(),
            'quality': quality.mean().item()
        }
        
        return quality, stats
    
    def compute_adaptive_bias(self, seq_ptd_raw: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute PTD bias with row-wise normalization and adaptive scaling.
        
        Args:
            seq_ptd_raw: [B, N, 2] raw PTD values
            mask: [B, N] True for invalid
        
        Returns:
            bias: [B, N] normalized bias
            alpha: float, adaptive scaling factor
        """
        B, N = mask.size()
        valid_mask = ~mask
        
        # Direct bias from 2-D raw PTD
        bias = self.ptd2bias(seq_ptd_raw).squeeze(-1)  # [B, N]
        
        # Mask invalid neighbors
        bias = bias.masked_fill(mask, 0)
        
        # Row-wise z-score normalization (critical for sparse neighbors)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        
        # Mean and std only over valid neighbors
        bias_sum = (bias * valid_mask.float()).sum(dim=1, keepdim=True)
        bias_mean = bias_sum / valid_count
        
        bias_centered = bias - bias_mean
        bias_sq = (bias_centered.pow(2) * valid_mask.float()).sum(dim=1, keepdim=True)
        bias_std = torch.sqrt(bias_sq / valid_count + 1e-6)
        
        # Z-score normalization
        bias = bias_centered / (bias_std + 1e-6)
        bias = bias.masked_fill(mask, -1e6)  # Large negative for softmax masking
        
        # Compute quality-based alpha
        quality, quality_stats = self.compute_neighbor_quality(seq_ptd_raw, mask)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * quality.mean()
        
        # Store for debugging
        if self.debug_mode:
            self.debug_stats['quality_scores'].append(quality.mean().item())
            self.debug_stats['alpha_values'].append(alpha.item())
        
        return bias, alpha.item()
    
    def compute_attention_entropy(self, attn_weights: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution (higher = more uniform).
        """
        # attn_weights: [B, N]
        valid_mask = ~mask
        attn_valid = attn_weights * valid_mask.float()
        
        # Renormalize over valid
        attn_sum = attn_valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attn_norm = attn_valid / attn_sum
        
        # Compute entropy
        entropy = -(attn_norm * torch.log(attn_norm + 1e-6)).sum(dim=1)
        return entropy.mean().item()
    
    def forward(self, src, src_t, src_ptd, seq, seq_t, seq_ptd, mask,
                seq_ptd_raw: Optional[torch.Tensor] = None,
                return_logits: bool = False,
                return_debug: bool = False):
        """
        Enhanced forward with raw PTD bias and debugging.
        
        Args:
            src, src_t, src_ptd: Source features, time, and encoded PTD
            seq, seq_t, seq_ptd: Sequence features, time, and encoded PTD
            mask: [B, N] invalid neighbor mask
            seq_ptd_raw: [B, N, 2] raw PTD values (magnitude, percentile)
            return_debug: Return debug statistics
        
        Returns:
            output: [B, D] final features
            attn: [B, N] attention weights
            debug_info: (optional) debugging statistics
        """
        device = seq.device
        B, N, _ = seq.size()
        mask_ = mask.unsqueeze(1) if mask.dim() == 2 else mask
        
        # Check sparse neighbor problem
        valid_count = (~mask).sum(dim=1)
        min_valid = valid_count.min().item()
        avg_valid = valid_count.float().mean().item()
        
        if self.debug_mode:
            self.debug_stats['neighbor_counts'].append({
                'min': min_valid,
                'avg': avg_valid,
                'max': valid_count.max().item()
            })
        
        # Use raw PTD if available, otherwise extract from encoded
        if seq_ptd_raw is None:
            # Fallback: try to extract from encoded PTD
            if seq_ptd.size(-1) >= 2:
                seq_ptd_raw = seq_ptd[:, :, :2]
            else:
                # Create dummy raw PTD
                seq_ptd_raw = torch.zeros(B, N, 2, device=device)
                print(f"[WARNING] No raw PTD provided, using zeros")
        
        # Compute adaptive bias from raw PTD
        bias, alpha = self.compute_adaptive_bias(seq_ptd_raw, mask)
        
        # Store bias statistics
        if self.debug_mode:
            valid_bias = bias.masked_fill(mask, 0)
            self.debug_stats['bias_stats'].append({
                'mean': valid_bias.mean().item(),
                'std': valid_bias.std().item(),
                'min': valid_bias[~mask].min().item() if (~mask).any() else 0,
                'max': valid_bias[~mask].max().item() if (~mask).any() else 0
            })
        
        # Build Q/K/V (using encoded PTD for main features)
        q = torch.cat([src.unsqueeze(1), src_t, src_ptd.unsqueeze(1)], dim=2)
        k = torch.cat([seq, seq_t, seq_ptd], dim=2)
        v = k
        
        # Apply bias with adaptive alpha
        attn_bias = alpha * bias.unsqueeze(1)  # [B, 1, N]
        
        # Multi-head attention
        out, attn = self.multi_head_target(
            q=q, k=k, v=v,
            mask=mask_,
            attn_bias=attn_bias,
            return_logits=False
        )
        
        attn = attn.squeeze(1)  # [B, N]
        output = self.merger(out.squeeze(1), src)
        
        # Compute attention entropy for debugging
        if self.debug_mode:
            entropy = self.compute_attention_entropy(attn, mask)
            self.debug_stats['attention_entropy'].append(entropy)
        
        # Periodic debug output
        self.debug_counter += 1
        if self.debug_mode and self.debug_counter % self.debug_every == 0:
            self.print_debug_summary()
        
        if return_debug:
            debug_info = {
                'alpha': alpha,
                'valid_neighbors': avg_valid,
                'bias_std': bias.std().item(),
                'attention_entropy': self.debug_stats['attention_entropy'][-1] if self.debug_stats['attention_entropy'] else 0
            }
            return output, attn, debug_info
        
        if return_logits:
            return output, attn, None
        
        return output, attn
    
    def print_debug_summary(self):
        """Print comprehensive debug statistics."""
        print(f"\n{'='*60}")
        print(f"AttnModelPTD_Direct Debug Summary (last {self.debug_every} batches)")
        print(f"{'='*60}")
        
        if self.debug_stats['neighbor_counts']:
            nc = self.debug_stats['neighbor_counts'][-self.debug_every:]
            avg_min = np.mean([x['min'] for x in nc])
            avg_avg = np.mean([x['avg'] for x in nc])
            print(f"Neighbors: min={avg_min:.1f}, avg={avg_avg:.1f}")
        
        if self.debug_stats['bias_stats']:
            bs = self.debug_stats['bias_stats'][-self.debug_every:]
            avg_std = np.mean([x['std'] for x in bs])
            print(f"Bias: std={avg_std:.3f}")
        
        if self.debug_stats['quality_scores']:
            qs = self.debug_stats['quality_scores'][-self.debug_every:]
            print(f"Quality: {np.mean(qs):.3f} ± {np.std(qs):.3f}")
        
        if self.debug_stats['alpha_values']:
            av = self.debug_stats['alpha_values'][-self.debug_every:]
            print(f"Alpha: {np.mean(av):.3f} ± {np.std(av):.3f}")
        
        if self.debug_stats['attention_entropy']:
            ae = self.debug_stats['attention_entropy'][-self.debug_every:]
            print(f"Attention Entropy: {np.mean(ae):.3f}")
            if np.mean(ae) < 0.5:
                print("  ⚠️ WARNING: Low entropy - attention may be collapsing!")
        
        print(f"{'='*60}\n")
    
    def reset_debug_stats(self):
        """Reset debug statistics."""
        self.debug_stats = {
            'neighbor_counts': [],
            'bias_stats': [],
            'quality_scores': [],
            'alpha_values': [],
            'attention_entropy': []
        }
        self.debug_counter = 0

class TATKC_PTD22Sep_Gated_claude(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(TATKC_PTD22Sep_Gated_claude, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([AttnModelPTD_Direct(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vector1(self, ptd_vec_np):
        """
        Final optimized PTD encoding for extremely sparse data (85% zeros).
        This version maximizes diversity in encoded features.
        """
        import numpy as np
        import torch
        
        # Get raw PTD
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Print input stats (remove in production)
        print(f"[PTD-Input] zeros={100*(raw==0).mean():.1f}% "
            f"max={raw.max():.1f} p95={(np.percentile(raw[raw>0], 95) if (raw>0).any() else 0):.1f}")
        
        # Clip outliers more aggressively for stability
        if (raw > 0).sum() > 10:
            # Use 90th percentile instead of 95th
            cap = np.percentile(raw[raw > 0], 90)
            raw = np.minimum(raw, cap)
        
        # FEATURE 1: Rank-based encoding (guaranteed diversity)
        # This preserves relative ordering perfectly
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = ranks / max(1.0, float(len(raw) - 1))  # Normalize to [0, 1]
        
        # FEATURE 2: Power transform instead of log for better spread
        # Power < 1 expands small values, which is what we need
        power = 0.25  # Fourth root - expands small values more than log
        
        # Create the magnitude feature
        magnitude = np.zeros_like(raw)
        non_zero_mask = raw > 0
        
        if non_zero_mask.sum() > 0:
            # Apply power transform to non-zero values
            nz_vals = raw[non_zero_mask]
            transformed = np.power(nz_vals, power)
            
            # Normalize to roughly [-1, 2] range
            # Use robust statistics
            p50 = np.percentile(transformed, 50)
            p90 = np.percentile(transformed, 90)
            
            if p90 > p50:
                # Scale based on upper quartile range
                scaled = (transformed - p50) / (p90 - p50)
                magnitude[non_zero_mask] = scaled
            else:
                # Fallback: simple normalization
                magnitude[non_zero_mask] = transformed / (transformed.max() + 1e-6)
        
        # Mark zeros distinctly (important for model to distinguish)
        magnitude[~non_zero_mask] = -0.5
        
        # Create 3-feature encoding for better expressiveness
        # Feature 3: Binary indicator (helps model identify patterns)
        is_nonzero = non_zero_mask.astype(np.float32)
        
        # Stack features - but for compatibility, keep 2D
        # Combine binary indicator with magnitude
        enhanced_magnitude = magnitude + 0.1 * is_nonzero  # Slight boost for non-zeros
        
        pair = np.stack([enhanced_magnitude, pct], axis=-1)
        
        # Print encoding stats
        unique_mag = len(np.unique(np.round(enhanced_magnitude, 3)))
        print(f"[PTD-Encoded] magnitude: std={enhanced_magnitude[non_zero_mask].std() if non_zero_mask.any() else 0:.3f} "
            f"unique={100*unique_mag/len(enhanced_magnitude):.1f}% "
            f"range=[{enhanced_magnitude.min():.2f},{enhanced_magnitude.max():.2f}]")
        
        # Move to device
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)
        
        # Store additional info
        self.ptd_log_col = torch.from_numpy(np.log1p(raw)).to(dev)
        self.ptd_nonzero_mask = torch.from_numpy(is_nonzero).to(dev)
        

    
    def set_ptd_vector_robust(self, ptd_vec_np):
        """
        set_ptd_vector_robust
        Robust encoding that generalizes across datasets.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Soft clip outliers using log compression for extreme values
        if (raw > 0).sum() > 10:
            p90 = np.percentile(raw[raw > 0], 90)
            # Soft clipping: linear up to p90, log-compressed above
            raw = np.where(raw > p90, p90 + np.log1p(raw - p90), raw)
        
        # Feature 1: Robust rank transform (always works regardless of distribution)
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_normalized = 2.0 * (ranks / max(1.0, len(ranks) - 1)) - 1.0  # Scale to [-1, 1]
        
        # Feature 2: Sign-preserving square root (expands small values)
        sqrt_transform = np.sign(raw) * np.sqrt(np.abs(raw))
        
        # Robust normalization using median and MAD
        if (raw > 0).sum() > 5:
            non_zero_sqrt = sqrt_transform[raw > 0]
            median = np.median(non_zero_sqrt)
            mad = np.median(np.abs(non_zero_sqrt - median))
            scale = max(mad * 1.4826, 0.1)  # Robust std estimate
            
            # Normalize non-zero values
            sqrt_normalized = np.zeros_like(sqrt_transform)
            sqrt_normalized[raw > 0] = (sqrt_transform[raw > 0] - median) / scale
            sqrt_normalized[raw == 0] = -1.0  # Clear signal for zeros
        else:
            sqrt_normalized = np.zeros_like(raw) - 1.0
        
        # Clip to reasonable range
        sqrt_normalized = np.clip(sqrt_normalized, -2.0, 2.0)
        
        # Stack features
        pair = np.stack([sqrt_normalized, rank_normalized], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)
    
    def set_ptd_vector_ranks_old(self, ptd_vec_np):
        """
        Simplest robust encoding using only ranks.
        """
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Normalized ranks
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_norm = ranks / max(1.0, len(ranks) - 1)  # [0, 1]
        
        # Feature 2: Binary indicator for non-zero
        is_nonzero = (raw > 0).astype(np.float32)
        
        # Combine: use rank for non-zero, -0.5 for zeros
        combined = np.where(raw > 0, rank_norm, -0.5)
        
        # Stack with pure rank (for percentile info)
        pair = np.stack([combined, rank_norm], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)

    def set_ptd_vector(self, ptd_vec_np):
        """Use this version - it gave you the best results."""
        import numpy as np
        import torch
        
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Normalized ranks
        ranks = raw.argsort().argsort().astype(np.float32)
        rank_norm = ranks / max(1.0, len(ranks) - 1)  # [0, 1]
        
        # Feature 2: Combined feature (rank for non-zero, -0.5 for zeros)
        combined = np.where(raw > 0, rank_norm, -0.5)
        
        # Stack as 2-D feature [combined, rank_norm]
        pair = np.stack([combined, rank_norm], axis=-1)
        
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_vec = torch.from_numpy(pair).to(dev)
        self.ptd_vec.requires_grad_(False)

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Modified tem_conv2 that passes raw PTD to attention.
        Add this as a method to your TATKC_PTD class.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")
        
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Get features
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]  # RAW 2-D PTD
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)  # Encoded 128-D
        
        if curr_layers == 0:
            return src_node_feat
        
        # Recursive call
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Check for sparse neighbors - try to get more if needed
        #valid_counts = (~mask).sum(dim=1)
        #min_valid = valid_counts.min().item()
        
        #if min_valid < 5 and num_neighbors < 40:
        #    # Try fetching more neighbors
        #    print(f"[Sparse Alert] Min neighbors: {min_valid}, fetching more...")
        #    ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
        #        tuple(src_idx_l), tuple(cut_time_l), 
        #        num_neighbors=min(num_neighbors * 2, 40)
        #    )
        #    ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        #    mask = (ngh_ids_th == 0)
        
        # Time encoding
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, -1, self.feat_dim)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # Get both raw and encoded PTD for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_raw = self.ptd_vec[idx_ngh0]  # RAW 2-D PTD
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_raw)  # Encoded 128-D
        ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Call attention with both raw and encoded PTD
        attn_m = self.attn_model_list[curr_layers - 1]
        
        if isinstance(attn_m, AttnModelPTD_Direct):
            # Pass raw PTD for direct bias computation
            local, _, debug_info = attn_m(
                src=src_node_conv_feat,
                src_t=src_node_t_embed,
                src_ptd=ptd_embed_src,
                seq=ngh_conv,
                seq_t=ngh_t_enc,
                seq_ptd=ptd_ngh_enc,
                mask=mask,
                seq_ptd_raw=ptd_ngh_raw,  # Pass raw 2-D PTD
                return_debug=True
            )
            
        if debug_info and hasattr(self, 'debug_mode') and self.debug_mode:
            self.last_debug_info = debug_info
        else:
            # Fallback for other attention types
            local, _ = attn_m(
                src=src_node_conv_feat,
                src_t=src_node_t_embed,
                src_ptd=ptd_embed_src,
                seq=ngh_conv,
                seq_t=ngh_t_enc,
                seq_ptd=ptd_ngh_enc,
                mask=mask
            )
        
        return F.normalize(local, p=2, dim=1)


#23 SEPT - giving claude as input the model "TATKC_PTD_19Sep_Gated_GPT" and suggesting the following:
 # RevertedAttnModelPTD_AdaptivePrior_GPT, 
 # EnhancedPTDProcessor,
 # Enchanced_TATKC_PTD_19Sep_Gated_GPT

class RevertedAttnModelPTD_AdaptivePrior_GPT(nn.Module):
    """
    Reverted to your original working approach with minimal improvements
    """
    def __init__(self, feat_dim, time_dim, ptd_enc_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2,
                 alpha_min=0.05, alpha_max=0.30,
                 bias_clip=2.0):
        super().__init__()
        assert attn_mode == 'prod'
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        self.model_dim = feat_dim + time_dim + ptd_enc_dim
        assert self.model_dim % n_head == 0

        self.bias_clip = float(bias_clip)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self._alpha_last = float((alpha_min + alpha_max) / 2)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        self.multi_head_target = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )

        self.diag_enabled, self.diag_every, self.diag_max_rows = True, 200, 64
        self._diag_step = 0

    @staticmethod
    def _clean(x: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        return x.clamp(-clip, clip)

    @staticmethod
    def _masked_std(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        v = valid_mask.float()
        cnt = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        var = (((x - mean) * v).pow(2)).sum(dim=1, keepdim=True) / cnt
        return torch.sqrt(var + eps).squeeze(1)

    @staticmethod
    def _masked_zscore(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        v = valid_mask.float()
        cnt = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        xc = (x - mean) * v
        var = (xc * xc).sum(dim=1, keepdim=True) / cnt
        std = torch.sqrt(var + eps)
        z = torch.where(valid_mask, (x - mean) / std, torch.zeros_like(x))
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _quality_from_percentile(self, pct: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        ORIGINAL working quality assessment - keep this unchanged
        """
        std_row = self._masked_std(pct, valid)
        non_zero_frac = ((pct > 0.01) & valid).float().mean(dim=1)
        
        # Your original thresholds that were working well
        std_norm = torch.clamp((std_row - 0.03) / 0.17, 0.0, 1.0)
        q = 0.6 * std_norm + 0.4 * non_zero_frac
        q = q.mean().detach()  # FIX: Add .detach() here
        return q

    def forward(self,
                src, src_t, src_ptd_enc,
                seq, seq_t, seq_ptd_enc,
                mask,
                src_ptd_raw=None,
                seq_ptd_raw=None,
                return_logits: bool = False):

        B, N, _ = seq.size()
        mask_3d = mask.unsqueeze(1) if mask.dim() == 2 else mask
        pad_bn = mask if mask.dim() == 2 else mask.squeeze(1)
        valid_bn = ~pad_bn
        has_valid = valid_bn.any(dim=1)

        # Sanitize content
        src = self._clean(src); src_t = self._clean(src_t)
        src_ptd_enc = self._clean(src_ptd_enc)
        seq = self._clean(seq); seq_t = self._clean(seq_t)
        seq_ptd_enc = self._clean(seq_ptd_enc)

        # PTD fallbacks
        if seq_ptd_raw is None:
            if seq_ptd_enc.size(-1) >= 2:
                seq_ptd_raw = seq_ptd_enc[..., :2].detach()
            else:
                seq_ptd_raw = torch.zeros(B, N, 2, device=seq.device)
        if src_ptd_raw is None:
            if src_ptd_enc.size(-1) >= 2:
                src_ptd_raw = src_ptd_enc[..., :2].detach()
            else:
                src_ptd_raw = torch.zeros(B, 2, device=src.device)

        # Percentile channel
        pct = torch.nan_to_num(seq_ptd_raw[..., 1], nan=0.0).clamp(0.0, 1.0)
        pct = pct * valid_bn.float()

        # ORIGINAL quality assessment (that was working!)
        quality = self._quality_from_percentile(pct, valid_bn)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * quality
        self._alpha_last = float(alpha.detach())  # FIX: Add .detach()

        #neigbor counts
        valid_counts = (~mask).sum(dim=1)

        # Content preparation
        vmask = valid_bn.unsqueeze(-1).float()
        k_content = self._clean(torch.cat([
            seq * vmask, 
            self._clean(seq_t * vmask), 
            self._clean(seq_ptd_enc * vmask)
        ], dim=2))
        v_content = k_content
        q_content = self._clean(torch.cat([
            src.unsqueeze(1), 
            src_t, 
            src_ptd_enc.unsqueeze(1)
        ], dim=2))

        content_signal = (k_content.abs().sum(dim=(1, 2)) > 0)
        use_attn_row = has_valid & content_signal
        out_feat = src.new_zeros(B, self.feat_dim)
        attn_out = src.new_zeros(B, N)

        # ORIGINAL bias computation (no temporal decay complications)
        bias_z = self._masked_zscore(pct, valid_bn).clamp(-self.bias_clip, self.bias_clip)
        attn_bias_full = (alpha * bias_z).unsqueeze(1)

        # Process valid and empty rows
        idx_valid = torch.nonzero(use_attn_row, as_tuple=True)[0]
        idx_empty = torch.nonzero(~use_attn_row, as_tuple=True)[0]

        if idx_valid.numel() > 0:
            qv = q_content.index_select(0, idx_valid)
            kv = k_content.index_select(0, idx_valid)
            vv = v_content.index_select(0, idx_valid)
            mv = mask_3d.index_select(0, idx_valid)
            bv = attn_bias_full.index_select(0, idx_valid)
            
            out_v, attn_v = self.multi_head_target(q=qv, k=kv, v=vv, mask=mv, attn_bias=bv, return_logits=False)
            merged_v = self.merger(out_v.squeeze(1), src.index_select(0, idx_valid))
            out_feat.index_copy_(0, idx_valid, self._clean(merged_v))
            attn_out.index_copy_(0, idx_valid, self._clean(attn_v.squeeze(1)))

        if idx_empty.numel() > 0:
            zeros_model = src.new_zeros(idx_empty.numel(), self.model_dim)
            merged_e = self.merger(zeros_model, src.index_select(0, idx_empty))
            out_feat.index_copy_(0, idx_empty, self._clean(merged_e))
            attn_out.index_copy_(0, idx_empty, src.new_zeros(idx_empty.numel(), N))

        # ORIGINAL diagnostics (that were working)
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            with torch.no_grad():
                nz_frac = (pct > 0.01).float().mean().item()
                std_mean = self._masked_std(pct, valid_bn).mean().item()
                #print(f"[AdaptivePrior] alpha={self._alpha_last:.3f} quality={float(quality):.3f} "
                #      f"std_mean={std_mean:.3f} nz_frac={nz_frac:.3f}")

        self._diag_step += 1

        if return_logits:
            return out_feat, attn_out, None
        
        return out_feat, attn_out, valid_counts


class EnhancedPTDProcessor:
    """Enhanced PTD processing with multiple encoding strategies"""
    
    @staticmethod
    def multi_scale_ptd_encoding(ptd_vec_np, num_scales=3):
        """
        Create multi-scale PTD features combining different statistical views
        """
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        features = []
        
        # Scale 1: Your existing robust z-score + percentile
        logv = np.log1p(raw)
        nz = logv[raw > 0]
        if nz.size > 5:
            log_med = float(np.median(nz))
            log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
        else:
            log_med = float(np.median(logv))
            log_mad = float(np.std(logv) + 1e-6)
        
        zlog = np.clip((logv - log_med) / log_mad, -5.0, 5.0)
        ranks = raw.argsort().argsort().astype(np.float32)
        pct = (ranks + 0.5) / max(1.0, len(raw))
        
        features.extend([zlog, pct])
        
        # Scale 2: Local neighborhood statistics
        window_size = max(10, len(raw) // 50)
        sorted_idx = np.argsort(raw)
        local_means = np.zeros_like(raw)
        
        for i in range(len(raw)):
            pos = np.where(sorted_idx == i)[0][0]
            start = max(0, pos - window_size // 2)
            end = min(len(raw), pos + window_size // 2)
            local_means[i] = raw[sorted_idx[start:end]].mean()
        
        local_deviation = (raw - local_means) / (local_means + 1e-6)
        features.append(local_deviation)
        
        # Scale 3: Quantile-based features
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        qtl_vals = np.percentile(raw, [q*100 for q in quantiles])
        qtl_features = np.searchsorted(qtl_vals, raw) / len(quantiles)
        features.append(qtl_features)
        
        return np.stack(features, axis=-1).astype(np.float32)
    

class Enchanced_TATKC_PTD_19Sep_Gated_GPT(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(Enchanced_TATKC_PTD_19Sep_Gated_GPT, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([RevertedAttnModelPTD_AdaptivePrior_GPT(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_enc_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # <-- was 1, now 2
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 128),  # Changed from 2 to 4
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vector(self, ptd_vec_np, cap_quantile: float = 0.995):
        """
        Build PTD tables for the current snapshot time t_cut.

        - Safe heavy-tail squash: log1p only applied on (raw > cap) entries.
        - Robust z-score on log1p(raw).
        - Percentile from RAW counts.
        """
        import numpy as np
        import torch

        # ----- raw -----
        raw = np.asarray(ptd_vec_np, dtype=np.float32)
        # sanitize
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.clip(raw, 0.0, None)

        # ----- heavy-tail squash (SAFE: mask before log1p) -----
        if (raw > 0).any():
            cap = float(np.quantile(raw[raw > 0], cap_quantile))
            if cap > 0.0:
                adj = raw.copy()
                above = adj > cap
                if np.any(above):
                    delta = adj[above] - cap          # >= 0
                    adj[above] = cap + np.log1p(delta)  # safe
                raw_squashed = adj
            else:
                raw_squashed = raw
        else:
            raw_squashed = raw

        # ----- log1p(raw) for robust stats (no NaNs) -----
        logv = np.log1p(raw_squashed).astype(np.float32)

        # ----- robust stats (use train stats if available) -----
        use_train_stats = hasattr(self, "ptd_stats") and isinstance(self.ptd_stats, dict) and \
                        ("log_median" in self.ptd_stats) and ("log_mad" in self.ptd_stats)

        if use_train_stats:
            log_med = float(self.ptd_stats["log_median"])
            log_mad = float(self.ptd_stats["log_mad"])
            if not np.isfinite(log_mad) or log_mad <= 0:
                nz = logv[raw > 0]
                if nz.size > 5:
                    log_med = float(np.median(nz))
                    log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
                else:
                    log_med = float(np.median(logv))
                    log_mad = float(np.std(logv) + 1e-6)
        else:
            nz = logv[raw > 0]
            if nz.size > 5:
                log_med = float(np.median(nz))
                log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
            else:
                log_med = float(np.median(logv))
                log_mad = float(np.std(logv) + 1e-6)

        zlog = (logv - log_med) / log_mad
        zlog = np.clip(zlog, -5.0, 5.0).astype(np.float32)

        # percentile on RAW (stable, unitless)
        N = float(len(raw))
        if N > 0:
            ranks = raw.argsort().argsort().astype(np.float32)
            pct = (ranks + 0.5) / max(1.0, N)
        else:
            pct = np.zeros_like(raw, dtype=np.float32)

        #pair = np.stack([zlog, pct], axis=-1).astype(np.float32)
        pair = EnhancedPTDProcessor.multi_scale_ptd_encoding(ptd_vec_np)
        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_log_col = torch.from_numpy(np.log1p(raw).astype(np.float32)).to(dev).requires_grad_(False)
        self.ptd_vec     = torch.from_numpy(pair).to(dev).requires_grad_(False)

        # (optional) keep last-used stats

        self._ptd_stats_last = {"log_median": log_med, "log_mad": log_mad}


            

    # Modification for your main model's tem_conv2
    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Enhanced tem_conv2 that passes PTD table info to attention modules.
        Replace the tem_conv2 method in your TATKC_PTD_20Aug class.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")
        
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Get features and embeddings as before
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)
        
        if curr_layers == 0:
            return src_node_feat
        
        # Recursive call
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
        ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Enhanced attention with PTD table info
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _, valid_counts = attn_m(
            src=src_node_conv_feat, src_t=src_node_t_embed, src_ptd_enc=ptd_embed_src,
            seq=ngh_conv,           seq_t=ngh_t_enc,        seq_ptd_enc=ptd_ngh_enc,
            mask=mask,
            src_ptd_raw=ptd_src_vals, seq_ptd_raw=ptd_ngh_vals
        )

        self.last_valid_counts = valid_counts
        
        return F.normalize(local, p=2, dim=1)

class PTDEncodingStrategies:
    """
    Multiple PTD encoding strategies optimized for ranking tasks
    """
    
    @staticmethod
    def ranking_aware_encoding(ptd_vec_np, num_bins=10):
        """
        Encoding specifically designed for ranking tasks.
        Creates features that emphasize relative ordering.
        """
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Rank-based percentiles (most important for ranking)
        ranks = raw.argsort().argsort().astype(np.float32)
        percentiles = ranks / max(1.0, len(raw) - 1)  # [0,1]
        
        # Feature 2: Binned ranks (discrete ranking signal)
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(percentiles, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        binned_ranks = bin_indices.astype(np.float32) / max(1.0, num_bins - 1)
        
        # Feature 3: Local ranking context (how does this node rank within its neighborhood?)
        window_size = max(10, len(raw) // 20)
        sorted_indices = raw.argsort()
        local_ranks = np.zeros_like(raw)
        
        for i in range(len(raw)):
            pos = np.where(sorted_indices == i)[0][0]
            start = max(0, pos - window_size // 2)
            end = min(len(raw), pos + window_size // 2)
            local_window = raw[sorted_indices[start:end]]
            local_rank = np.searchsorted(np.sort(local_window), raw[i])
            local_ranks[i] = local_rank / max(1.0, len(local_window) - 1)
        
        # Feature 4: Ranking confidence (how separated is this node from its neighbors?)
        confidence = np.zeros_like(raw)
        for i in range(len(raw)):
            pos = np.where(sorted_indices == i)[0][0]
            left_val = raw[sorted_indices[max(0, pos-1)]] if pos > 0 else raw[i]
            right_val = raw[sorted_indices[min(len(raw)-1, pos+1)]] if pos < len(raw)-1 else raw[i]
            
            if raw[i] > 0:
                left_gap = (raw[i] - left_val) / (raw[i] + 1e-6)
                right_gap = (right_val - raw[i]) / (raw[i] + 1e-6)
                confidence[i] = min(left_gap, right_gap)
            else:
                confidence[i] = 0.0
        
        confidence = np.clip(confidence, 0, 1)
        
        return np.stack([percentiles, binned_ranks, local_ranks, confidence], axis=-1).astype(np.float32)
    
    @staticmethod
    def contrastive_encoding(ptd_vec_np, temperature=0.1):
        """
        Encoding that emphasizes differences between nodes (good for contrastive ranking).
        """
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Standard percentile
        ranks = raw.argsort().argsort().astype(np.float32)
        percentiles = ranks / max(1.0, len(raw) - 1)
        
        # Feature 2: Softmax-based relative importance
        if raw.max() > 0:
            softmax_weights = np.exp(raw / (temperature * raw.max()))
            softmax_weights = softmax_weights / softmax_weights.sum()
        else:
            softmax_weights = np.ones_like(raw) / len(raw)
        
        # Feature 3: Pairwise contrast strength
        contrast_strength = np.zeros_like(raw)
        for i in range(len(raw)):
            # How much does this node stand out from the average?
            mean_others = (raw.sum() - raw[i]) / max(1, len(raw) - 1)
            if mean_others > 0:
                contrast_strength[i] = raw[i] / (mean_others + 1e-6)
            else:
                contrast_strength[i] = 1.0 if raw[i] > 0 else 0.0
        
        contrast_strength = np.log1p(contrast_strength)
        contrast_strength = contrast_strength / (contrast_strength.max() + 1e-6)
        
        # Feature 4: Relative deviation from median
        median_val = np.median(raw[raw > 0]) if (raw > 0).any() else 0
        relative_dev = np.where(raw > 0, 
                               (raw - median_val) / (median_val + 1e-6),
                               -1.0)
        relative_dev = np.clip(relative_dev, -2, 2) / 2.0  # Normalize to [-1, 1]
        
        return np.stack([percentiles, softmax_weights, contrast_strength, relative_dev], axis=-1).astype(np.float32)
    
    @staticmethod
    def adaptive_resolution_encoding(ptd_vec_np, resolution_levels=3):
        """
        Multi-resolution encoding that captures patterns at different scales.
        """
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        features = []
        
        # Base feature: percentile
        ranks = raw.argsort().argsort().astype(np.float32)
        percentiles = ranks / max(1.0, len(raw) - 1)
        features.append(percentiles)
        
        # Multi-resolution features
        for level in range(1, resolution_levels + 1):
            # Create coarser resolution by grouping nodes
            group_size = max(1, len(raw) // (2 ** level))
            if group_size < 2:
                # If groups become too small, repeat the finest resolution
                features.append(features[-1])
                continue
                
            # Group nodes and compute group statistics
            grouped_features = np.zeros_like(raw)
            for i in range(0, len(raw), group_size):
                end_idx = min(i + group_size, len(raw))
                group_raw = raw[i:end_idx]
                
                if len(group_raw) > 0:
                    # Rank within group
                    if len(group_raw) > 1:
                        group_ranks = group_raw.argsort().argsort().astype(np.float32)
                        group_percentiles = group_ranks / (len(group_raw) - 1)
                    else:
                        group_percentiles = np.array([0.5])
                    
                    grouped_features[i:end_idx] = group_percentiles
            
            features.append(grouped_features)
        
        # Ensure we have exactly 4 features
        while len(features) < 4:
            features.append(features[-1])  # Duplicate last feature
        
        return np.stack(features[:4], axis=-1).astype(np.float32)
    
    @staticmethod
    def robust_quantile_encoding(ptd_vec_np, quantiles=[0.1, 0.25, 0.75, 0.9]):
        """
        Encoding based on robust quantiles, less sensitive to outliers.
        """
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Compute quantile thresholds
        if (raw > 0).sum() > len(quantiles):
            thresholds = np.percentile(raw[raw > 0], [q * 100 for q in quantiles])
        else:
            # Fallback for sparse data
            max_val = raw.max()
            thresholds = [q * max_val for q in quantiles]
        
        features = []
        for threshold in thresholds:
            # Feature: What fraction of nodes have lower PTD than this node?
            feature = (raw[:, np.newaxis] >= raw).mean(axis=1)  # Broadcast comparison
            features.append(feature)
        
        return np.stack(features, axis=-1).astype(np.float32)
    
    @staticmethod
    def entropy_aware_encoding(ptd_vec_np):
        """
        Encoding that considers information content and entropy.
        """
        raw = np.clip(np.asarray(ptd_vec_np, dtype=np.float32), 0, None)
        
        # Feature 1: Standard percentile
        ranks = raw.argsort().argsort().astype(np.float32)
        percentiles = ranks / max(1.0, len(raw) - 1)
        
        # Feature 2: Information content (negative log probability)
        total = raw.sum()
        if total > 0:
            probs = (raw + 1e-8) / (total + 1e-8 * len(raw))  # Add smoothing
            info_content = -np.log(probs)
            info_content = info_content / info_content.max()  # Normalize
        else:
            info_content = np.ones_like(raw) / len(raw)
        
        # Feature 3: Local entropy contribution
        window_size = max(5, len(raw) // 10)
        local_entropy = np.zeros_like(raw)
        
        for i in range(len(raw)):
            # Get local window around this node (by rank)
            rank_pos = int(percentiles[i] * (len(raw) - 1))
            start = max(0, rank_pos - window_size // 2)
            end = min(len(raw), rank_pos + window_size // 2)
            
            local_raw = raw[ranks[start:end]]
            if local_raw.sum() > 0:
                local_probs = local_raw / local_raw.sum()
                local_probs = local_probs[local_probs > 0]  # Remove zeros for entropy calc
                entropy = -np.sum(local_probs * np.log(local_probs))
                local_entropy[i] = entropy / np.log(len(local_probs))  # Normalize
            else:
                local_entropy[i] = 0.0
        
        # Feature 4: Concentration measure
        sorted_raw = np.sort(raw)[::-1]  # Descending
        cumsum = np.cumsum(sorted_raw)
        total_sum = cumsum[-1] if len(cumsum) > 0 else 1
        
        concentration = np.zeros_like(raw)
        for i in range(len(raw)):
            # What fraction of total "mass" is concentrated in nodes ranked higher than i?
            rank_pos = int(percentiles[i] * (len(raw) - 1))
            if total_sum > 0:
                concentration[i] = cumsum[rank_pos] / total_sum
            else:
                concentration[i] = percentiles[i]
        
        return np.stack([percentiles, info_content, local_entropy, concentration], axis=-1).astype(np.float32)

class AttnModelPTD_DualPathHybrid(nn.Module):
    """
    Dual-path attention:
      - Path A: content-only attention (no PTD bias)
      - Path B: prior-biased attention (logit bias from PTD percentile)
    Row-wise gate g mixes them: out = g * out_prior + (1-g) * out_content
    """
    def __init__(self, feat_dim, time_dim, ptd_enc_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2,
                 alpha_min=0.02, alpha_max=0.25,  # gentle prior; tune 0.02~0.30
                 bias_clip=1.5, diag_every=200):
        super().__init__()
        assert attn_mode == 'prod'
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        self.model_dim = feat_dim + time_dim + ptd_enc_dim
        assert self.model_dim % n_head == 0

        from module import MergeLayer, MultiHeadAttention  # your impl
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        self.mha     = MultiHeadAttention(
            n_head=n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )

        # prior strength bounds
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.bias_clip = float(bias_clip)
        self._alpha_last = float((alpha_min + alpha_max) / 2)

        # tiny gating MLP: uses per-row stats → g in (0,1)
        # features: [alpha_row, nz_frac_row, std_row_norm]
        self.gate = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

        # diagnostics
        self.diag_every = int(diag_every)
        self._diag_step = 0
        self.diag_enabled = True

    @staticmethod
    def _clean(x: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        return x.clamp(-clip, clip)

    @staticmethod
    def _masked_std(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        v = valid_mask.float()
        cnt = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        var  = (((x - mean) * v).pow(2)).sum(dim=1, keepdim=True) / cnt
        return torch.sqrt(var + eps).squeeze(1)  # [B]

    @staticmethod
    def _masked_zscore(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        v = valid_mask.float()
        cnt = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * v).sum(dim=1, keepdim=True) / cnt
        std  = torch.sqrt((((x - mean) * v).pow(2)).sum(dim=1, keepdim=True) / cnt + eps)
        z = torch.where(valid_mask, (x - mean) / std, torch.zeros_like(x))
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _quality_and_rowstats(self, pct: torch.Tensor, valid: torch.Tensor):
        """Return (alpha_row, nz_frac_row, std_row_norm, alpha_scalar_for_log)."""
        std_row = self._masked_std(pct, valid)                      # [B]
        nz_frac_row = ((pct > 0.01) & valid).float().mean(dim=1)    # [B]
        # normalize std into ~[0,1]
        std_norm_row = torch.clamp((std_row - 0.03) / 0.17, 0.0, 1.0)
        # same quality recipe you liked (row-wise, then mean for logging)
        q_row = 0.6 * std_norm_row + 0.4 * nz_frac_row              # [B]
        alpha_row = self.alpha_min + (self.alpha_max - self.alpha_min) * q_row  # [B]
        alpha_scalar = float(q_row.mean().detach().cpu())  # for the debug line
        return alpha_row, nz_frac_row, std_norm_row, alpha_scalar

    def _entropy(self, attn: torch.Tensor, mask_bn: torch.Tensor, eps=1e-8):
        # attn:[B,N]; mask_bn True=PAD
        p = attn.masked_fill(mask_bn, 0.0)
        p = p / (p.sum(dim=1, keepdim=True) + eps)
        h = -(p * (p + eps).log()).sum(dim=1)
        return h.mean().item()

    def forward(self,
                src, src_t, src_ptd_enc,
                seq, seq_t, seq_ptd_enc,
                mask,
                src_ptd_raw=None,   # expect [...,2] with [:,1] = percentile in [0,1]
                seq_ptd_raw=None,
                return_logits: bool = False):

        B, N, _ = seq.size()
        mask_3d  = mask.unsqueeze(1) if mask.dim() == 2 else mask   # [B,1,N]
        pad_bn   = mask if mask.dim() == 2 else mask.squeeze(1)     # [B,N]
        valid_bn = ~pad_bn                                           # [B,N]
        has_valid = valid_bn.any(dim=1)

        # sanitize content
        src = self._clean(src); src_t = self._clean(src_t); src_ptd_enc = self._clean(src_ptd_enc)
        seq = self._clean(seq); seq_t = self._clean(seq_t); seq_ptd_enc = self._clean(seq_ptd_enc)

        # raw PTD fallbacks -> use enc proxy if needed
        if seq_ptd_raw is None:
            if seq_ptd_enc.size(-1) >= 2:  seq_ptd_raw = seq_ptd_enc[..., :2].detach()
            else:                          seq_ptd_raw = torch.zeros(B, N, 2, device=seq.device)
        if src_ptd_raw is None:
            if src_ptd_enc.size(-1) >= 2:  src_ptd_raw = src_ptd_enc[..., :2].detach()
            else:                          src_ptd_raw = torch.zeros(B, 2, device=src.device)

        # percentile channel
        pct = torch.nan_to_num(seq_ptd_raw[..., 1], nan=0.0).clamp(0.0, 1.0) * valid_bn.float()  # [B,N]

        # row-wise alpha and stats
        alpha_row, nz_frac_row, std_norm_row, alpha_scalar = self._quality_and_rowstats(pct, valid_bn)
        self._alpha_last = float((alpha_row.mean()).item())

        # content tensors
        vmask = valid_bn.unsqueeze(-1).float()
        k_content = torch.cat([seq * vmask, self._clean(seq_t * vmask), self._clean(seq_ptd_enc * vmask)], dim=2)
        v_content = k_content
        q_content = torch.cat([src.unsqueeze(1), src_t, src_ptd_enc.unsqueeze(1)], dim=2)

        # bias from z-scored percentile
        bias_z = self._masked_zscore(pct, valid_bn).clamp(-self.bias_clip, self.bias_clip)  # [B,N]
        attn_bias_prior = (alpha_row.view(B,1,1) * bias_z.unsqueeze(1))  # [B,1,N]

        # rows with no usable content fallback to identity merge
        content_signal = (k_content.abs().sum(dim=(1,2)) > 0)
        use_attn_row = has_valid & content_signal
        idx_valid = torch.nonzero(use_attn_row, as_tuple=True)[0]
        idx_empty = torch.nonzero(~use_attn_row, as_tuple=True)[0]

        out_feat = src.new_zeros(B, self.feat_dim)
        attn_mix = src.new_zeros(B, N)

        if idx_valid.numel() > 0:
            qv = q_content.index_select(0, idx_valid)
            kv = k_content.index_select(0, idx_valid)
            vv = v_content.index_select(0, idx_valid)
            mv = mask_3d.index_select(0, idx_valid)

            # Path A: content-only (no bias)
            out_A, attn_A = self.mha(q=qv, k=kv, v=vv, mask=mv, attn_bias=None, return_logits=False)

            # Path B: prior-biased
            bv = attn_bias_prior.index_select(0, idx_valid)
            out_B, attn_B = self.mha(q=qv, k=kv, v=vv, mask=mv, attn_bias=bv, return_logits=False)

            # Row-wise gate g = σ(W·[α, nz_frac, std_norm] + b)
            f = torch.stack([
                alpha_row.index_select(0, idx_valid),
                nz_frac_row.index_select(0, idx_valid),
                std_norm_row.index_select(0, idx_valid)
            ], dim=1)  # [b,3]
            g = self.gate(f).view(-1,1,1)  # [b,1,1]

            out_v = g * out_B + (1 - g) * out_A
            attn_v = g.squeeze(1) * attn_B.squeeze(1) + (1 - g.squeeze(1)) * attn_A.squeeze(1)

            merged_v = self.merger(out_v.squeeze(1), src.index_select(0, idx_valid))
            out_feat.index_copy_(0, idx_valid, self._clean(merged_v))
            attn_mix.index_copy_(0, idx_valid, self._clean(attn_v))

        if idx_empty.numel() > 0:
            zeros_model = src.new_zeros(idx_empty.numel(), self.model_dim)
            merged_e = self.merger(zeros_model, src.index_select(0, idx_empty))
            out_feat.index_copy_(0, idx_empty, self._clean(merged_e))
            attn_mix.index_copy_(0, idx_empty, src.new_zeros(idx_empty.numel(), N))

        # light diagnostics
        if self.diag_enabled and (self._diag_step % self.diag_every == 0):
            with torch.no_grad():
                # compute quick stats on the mixed attention
                H = self._entropy(attn_mix, pad_bn)
                # KL vs uniform ~ log(N) - H
                KL = (torch.log(torch.tensor(max(N,1.), device=attn_mix.device)) - H)
                # correlation attn vs pct where valid
                vmask = valid_bn.float()
                attn_row = (attn_mix * vmask)
                if vmask.sum() > 0:
                    # safe Pearson corr per-batch (rough)
                    a = attn_row.view(B, -1); p = (pct * vmask).view(B, -1)
                    a = a - a.mean(dim=1, keepdim=True)
                    p = p - p.mean(dim=1, keepdim=True)
                    rho = (a*p).sum(dim=1) / (a.norm(dim=1)+1e-6) / (p.norm(dim=1)+1e-6)
                    rho = float(rho.mean().item())
                else:
                    rho = 0.0
                print(f"[DualPath] α={self._alpha_last:.3f}  H={H:.3f}  KL={KL:.3f}  ρ(attn,PTD)={rho:+.3f}  "
                      f"ḡ={float(self.gate[0].weight.detach().abs().mean().cpu() if isinstance(self.gate[0], nn.Linear) else 0):.3f}")
        self._diag_step += 1

        if return_logits:
            return out_feat, attn_mix, None
        return out_feat, attn_mix


class Enchanced_26Sept(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None):
        super(Enchanced_26Sept, self).__init__()

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

        self.attn_model_list = torch.nn.ModuleList([RevertedAttnModelPTD_AdaptivePrior_GPT(self.feat_dim,
                                                                self.feat_dim,
                                                                ptd_enc_dim=128,
                                                                attn_mode=attn_mode,
                                                                n_head=n_head,
                                                                drop_out=drop_out) for _ in range(num_layers)])


        self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim,
                                         1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)



        self.ptd_scale = nn.Parameter(torch.tensor(1.0))
        self.ptd_shift = nn.Parameter(torch.tensor(0.0))


        self.debug_ptd = False  # turn on when you want diagnostics
        self.debug_every = 1  # run the diagnostic every N batches
        self._debug_step = 0  # internal counter

        self.pre_head_norm = nn.LayerNorm(self.feat_dim + 2 + 128)

        self.ptd_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 128),  # Changed from 2 to 4
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )

        # runtime PTD tables (set by set_ptd_vector each snapshot)
        self.ptd_vec = None        # [N, 2] = [log1p(scale*raw), percentile]
        self.ptd_log_col = None    # [N]    = log1p(scale*raw)

        # debug flags
        self.debug_temconv = False



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
    
    def set_ptd_vector(self, ptd_raw_1b: np.ndarray, cap_quantile: float = 0.995):
        """
        Accepts a 1-based raw PTD vector of shape [max_id+1], index 0 unused.
        Produces 4-dim features per node (0-based) and stores them in self.ptd_vec with shape [N, 4].
        Keeps a log1p(raw) column in self.ptd_log_col for convenience.
        """
        import numpy as np
        import torch

        # ---- sanitize input ----
        raw_1b = np.asarray(ptd_raw_1b, dtype=np.float32).copy()
        raw_1b = np.nan_to_num(raw_1b, nan=0.0, posinf=0.0, neginf=0.0)
        raw_1b = np.clip(raw_1b, 0.0, None)

        # Drop the dummy 0th row (we keep the model 0-based internally)
        if raw_1b.ndim != 1:
            raise ValueError("set_ptd_vector expects a 1-D raw vector (1-based).")
        if raw_1b.size <= 1:
            # no nodes
            dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
            self.ptd_log_col = torch.zeros(0, dtype=torch.float32, device=dev)
            self.ptd_vec     = torch.zeros(0, 4, dtype=torch.float32, device=dev)
            return

        raw = raw_1b[1:]  # shape [N], 0-based for the model

        # ---- heavy-tail squash (optional, safe) ----
        if (raw > 0).any():
            cap = float(np.quantile(raw[raw > 0], cap_quantile))
            if cap > 0.0:
                adj = raw.copy()
                above = adj > cap
                if np.any(above):
                    delta = adj[above] - cap
                    adj[above] = cap + np.log1p(delta)  # smooth beyond cap
                raw_squashed = adj
            else:
                raw_squashed = raw
        else:
            raw_squashed = raw

        logv = np.log1p(raw_squashed).astype(np.float32)

        # ---- robust stats for z-score on log1p ----
        nz = logv[raw > 0]
        if nz.size > 5:
            log_med = float(np.median(nz))
            log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
        else:
            log_med = float(np.median(logv))
            log_mad = float(np.std(logv) + 1e-6)

        zlog = np.clip((logv - log_med) / log_mad, -5.0, 5.0)

        # ---- global percentile (stable, handles ties) ----
        N = raw.shape[0]
        order = np.argsort(raw, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(N, dtype=np.float32)
        pct_all = (ranks + 0.5) / max(1.0, float(N))

        # ---- nonzero-only percentile (zeros get 0) ----
        mask_nz = raw > 0
        pct_nz = np.zeros(N, dtype=np.float32)
        if np.any(mask_nz):
            raw_nz = raw[mask_nz]
            order_nz = np.argsort(raw_nz, kind="mergesort")
            ranks_nz = np.empty_like(order_nz, dtype=np.float32)
            ranks_nz[order_nz] = np.arange(order_nz.size, dtype=np.float32)
            pct_nz[mask_nz] = (ranks_nz + 0.5) / max(1.0, float(order_nz.size))

        # ---- extra scale (raw log1p) to make 4 dims ----
        log1p_raw = np.log1p(raw).astype(np.float32)

        # ---- pack features: [zlog, pct_all, log1p_raw, pct_nz] -> 4 dims ----
        pair = np.stack([zlog, pct_all, log1p_raw, pct_nz], axis=1).astype(np.float32)

        dev = getattr(self.n_feat_th, "device", torch.device("cpu"))
        self.ptd_log_col = torch.from_numpy(log1p_raw).to(dev).requires_grad_(False)  # [N]
        self.ptd_vec     = torch.from_numpy(pair).to(dev).requires_grad_(False)       # [N, 4]

        # keep last-used stats if you like
        self._ptd_stats_last = {"log_median": log_med, "log_mad": log_mad}


            

    # Modification for your main model's tem_conv2
    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        Enhanced tem_conv2 that passes PTD table info to attention modules.
        Replace the tem_conv2 method in your TATKC_PTD_20Aug class.
        """
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")
        
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Get features and embeddings as before
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)
        
        if curr_layers == 0:
            return src_node_feat
        
        # Recursive call
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l,
            cut_time_l=cut_time_l,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat,
            cut_time_l=t_flat,
            ptd_l=None,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]
        ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
        ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Enhanced attention with PTD table info
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _, valid_counts = attn_m(
            src=src_node_conv_feat, src_t=src_node_t_embed, src_ptd_enc=ptd_embed_src,
            seq=ngh_conv,           seq_t=ngh_t_enc,        seq_ptd_enc=ptd_ngh_enc,
            mask=mask,
            src_ptd_raw=ptd_src_vals, seq_ptd_raw=ptd_ngh_vals
        )

        self.last_valid_counts = valid_counts
        
        return F.normalize(local, p=2, dim=1)

class TATKC_PTD_19Sep_DualPath_GPT(torch.nn.Module):



    def __init__(self, ngh_finder, n_feat,
                 attn_mode='prod', use_time='time', agg_method='attn',
                 num_layers=2, n_head=2, null_idx=0, drop_out=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx

        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]

        from module import MergeLayer, TimeEncode
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.time_encoder = TimeEncode(expand_dim=self.feat_dim)

        # dual-path attention per layer
        self.attn_model_list = nn.ModuleList([
            AttnModelPTD_DualPathHybrid(self.feat_dim, self.feat_dim, ptd_enc_dim=128,
                                        attn_mode=attn_mode, n_head=n_head, drop_out=drop_out,
                                        alpha_min=0.02, alpha_max=0.25, bias_clip=1.5)
            for _ in range(num_layers)
        ])

        # PTD encoder (2 → 128) just like your best model
        self.ptd_encoder = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128))

        # final scorer
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

        # runtime PTD tables
        self.ptd_vec = None        # [N,2] (zlog, percentile)
        self.ptd_log_col = None

    @torch.no_grad()
    def set_ptd_vector(self, ptd_vec_np, cap_quantile: float = 0.995):
        import numpy as np, torch
        raw = np.asarray(ptd_vec_np, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)

        if (raw > 0).any():
            cap = float(np.quantile(raw[raw > 0], cap_quantile))
            if cap > 0.0:
                adj = raw.copy(); above = adj > cap
                if np.any(above):
                    delta = adj[above] - cap
                    adj[above] = cap + np.log1p(delta)
                raw_squashed = adj
            else:
                raw_squashed = raw
        else:
            raw_squashed = raw

        logv = np.log1p(raw_squashed).astype(np.float32)
        nz = logv[raw > 0]
        if nz.size > 5:
            log_med = float(np.median(nz))
            log_mad = 1.4826 * float(np.median(np.abs(nz - log_med))) + 1e-6
        else:
            log_med = float(np.median(logv))
            log_mad = float(np.std(logv) + 1e-6)

        zlog = np.clip((logv - log_med) / log_mad, -5.0, 5.0).astype(np.float32)

        N = float(len(raw))
        ranks = raw.argsort().argsort().astype(np.float32) if N > 0 else np.zeros_like(raw, np.float32)
        pct = (ranks + 0.5) / max(1.0, N)

        pair = np.stack([zlog, pct], axis=-1).astype(np.float32)

        dev = self.n_feat_th.device
        self.ptd_log_col = torch.from_numpy(np.log1p(raw).astype(np.float32)).to(dev).requires_grad_(False)
        self.ptd_vec     = torch.from_numpy(pair).to(dev).requires_grad_(False)

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=20):
        assert curr_layers >= 0
        device = self.n_feat_th.device
        if self.ptd_vec is None:
            raise RuntimeError("Call set_ptd_vector(...) before tem_conv2.")

        import numpy as np
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)

        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))

        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        ptd_src_vals = self.ptd_vec[src_idx0]
        ptd_embed_src = self.ptd_encoder(ptd_src_vals)

        if curr_layers == 0:
            return src_node_feat

        # recurse lower layer
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l, cut_time_l=cut_time_l, ptd_l=None,
            curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)

        # neighbors
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)

        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)

        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat   = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat, cut_time_l=t_flat, ptd_l=None,
            curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)

        # PTD enc/raw for neighbors
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        ptd_ngh_vals = self.ptd_vec[idx_ngh0]
        ptd_ngh_enc  = self.ptd_encoder(ptd_ngh_vals)
        ptd_ngh_enc  = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)

        attn_m = self.attn_model_list[curr_layers - 1]
        local, _ = attn_m(
            src=src_node_conv_feat, src_t=src_node_t_embed, src_ptd_enc=ptd_embed_src,
            seq=ngh_conv,           seq_t=ngh_t_enc,        seq_ptd_enc=ptd_ngh_enc,
            mask=mask,
            src_ptd_raw=ptd_src_vals, seq_ptd_raw=ptd_ngh_vals
        )
        return F.normalize(local, p=2, dim=1)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed    = self.tem_conv2(src_idx_l,    cut_time_l, None, self.num_layers, num_neighbors)
        target_embed = self.tem_conv2(target_idx_l, cut_time_l, None, self.num_layers, num_neighbors)

        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score
    





# ADD THESE CLASSES TO YOUR module.py FILE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimplifiedPTDEncoder(nn.Module):
    """Simplified 2D PTD encoding instead of complex 4D"""
    def __init__(self, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, output_dim),  # Only [log1p_raw, percentile]
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, ptd_features):
        """
        ptd_features: [batch, 2] = [log1p(raw), percentile]
        """
        return self.encoder(ptd_features)





class SaferSimplifiedPTDModel(nn.Module):
    """Better PTD integration while keeping stability"""
    def __init__(self, feat_dim, time_dim, ptd_enc_dim=64, n_head=2, drop_out=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        
        # Your existing merger (proven to work)
        self.merger = MergeLayer(feat_dim + time_dim + ptd_enc_dim, feat_dim, feat_dim, feat_dim)
        
        # Enhanced PTD integration
        self.ptd_attention = nn.Sequential(
            nn.Linear(ptd_enc_dim, feat_dim),
            nn.Tanh(),
            nn.Linear(feat_dim, feat_dim),
            nn.Softmax(dim=-1)
        )
        
        # PTD-aware neighbor weighting
        self.neighbor_ptd_weight = nn.Sequential(
            nn.Linear(ptd_enc_dim * 2, 32),  # src + neighbor PTD
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(feat_dim)
        
    def forward(self, src, src_t, src_ptd_enc, seq, seq_t, seq_ptd_enc, mask):
        batch_size, seq_len, _ = seq.size()
        
        # Safe masking
        valid_mask = ~mask
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        
        # Enhanced PTD-aware aggregation
        src_ptd_expanded = src_ptd_enc.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute PTD-based neighbor weights
        ptd_pairs = torch.cat([src_ptd_expanded, seq_ptd_enc], dim=-1)  # [B, N, 2*ptd_dim]
        neighbor_weights = self.neighbor_ptd_weight(ptd_pairs).squeeze(-1)  # [B, N]
        neighbor_weights = neighbor_weights * valid_mask.float()
        neighbor_weights = F.softmax(neighbor_weights.masked_fill(mask, -1e9), dim=1)
        
        # Weighted aggregation using PTD weights
        seq_agg = torch.sum(seq * neighbor_weights.unsqueeze(-1), dim=1)
        seq_t_agg = torch.sum(seq_t * neighbor_weights.unsqueeze(-1), dim=1)  
        seq_ptd_agg = torch.sum(seq_ptd_enc * neighbor_weights.unsqueeze(-1), dim=1)
        
        # Combine features
        combined_features = torch.cat([seq_agg, seq_t_agg, seq_ptd_agg], dim=-1)
        
        # Apply merger
        merged_output = self.merger(combined_features, src)
        
        # PTD attention mechanism
        ptd_attention_weights = self.ptd_attention(src_ptd_enc)
        enhanced_output = merged_output * ptd_attention_weights + merged_output
        
        # Final normalization
        final_output = self.layer_norm(enhanced_output)
        
        # Valid counts for loss
        neighbor_valid_counts = valid_mask.sum(dim=1).float()
        
        return final_output, neighbor_weights, neighbor_valid_counts


def safer_set_simplified_ptd_vector_rank(model, ptd_raw_1b: np.ndarray):
    # Input validation
    if ptd_raw_1b is None or len(ptd_raw_1b) == 0:
        print("WARNING: Empty PTD vector provided")
        device = getattr(model.n_feat_th, "device", torch.device("cpu"))
        model.ptd_vec = torch.zeros(1, 2, dtype=torch.float32, device=device)
        return
    
    raw_1b = np.asarray(ptd_raw_1b, dtype=np.float32).copy()
    
    # Check for NaN/Inf values
    nan_count = np.isnan(raw_1b).sum()
    inf_count = np.isinf(raw_1b).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: Found {nan_count} NaN and {inf_count} Inf values in PTD, cleaning...")
    
    # Clean the data
    raw_1b = np.nan_to_num(raw_1b, nan=0.0, posinf=0.0, neginf=0.0)
    raw_1b = np.clip(raw_1b, 0.0, 1e10)  # Reasonable upper bound
    
    if raw_1b.size <= 1:
        print("WARNING: PTD vector too small")
        device = getattr(model.n_feat_th, "device", torch.device("cpu"))
        model.ptd_vec = torch.zeros(1, 2, dtype=torch.float32, device=device)
        return
    
    raw = raw_1b[1:]  # Drop dummy 0th row
    raw_normalized = raw / np.max(raw).clip(min=1.0) 
    # Safe percentile ranking
    N = raw.shape[0]    

    if N <= 1:
        # Fallback remains the same
        ptd_features = np.array([[0.0, 0.5]], dtype=np.float32)
    else:
        # --- CRITICAL CHANGE: Stack Normalized Raw Value and Percentile Rank ---
        order = np.argsort(raw, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(N, dtype=np.float32)
        percentile = (ranks + 0.5) / max(1.0, float(N))
    
        ptd_features = np.stack([raw_normalized, percentile], axis=1).astype(np.float32)
        # --------------------------------------------------------------------
    
    if np.any(np.isnan(ptd_features)) or np.any(np.isinf(ptd_features)):
        print("ERROR: NaN/Inf in final PTD features, using zeros")
        ptd_features = np.zeros((N, 2), dtype=np.float32)

    # Convert to tensor
    device = getattr(model.n_feat_th, "device", torch.device("cpu"))
    model.ptd_vec = torch.from_numpy(ptd_features).to(device).requires_grad_(False)


# SAFER PTD VECTOR SETTING WITH NaN CHECKS
def safer_set_simplified_ptd_vector(model, ptd_raw_1b: np.ndarray):
    """Safer PTD vector creation with extensive error checking"""
    
    # Input validation
    if ptd_raw_1b is None or len(ptd_raw_1b) == 0:
        print("WARNING: Empty PTD vector provided")
        device = getattr(model.n_feat_th, "device", torch.device("cpu"))
        model.ptd_vec = torch.zeros(1, 2, dtype=torch.float32, device=device)
        return
    
    raw_1b = np.asarray(ptd_raw_1b, dtype=np.float32).copy()
    
    # Check for NaN/Inf values
    nan_count = np.isnan(raw_1b).sum()
    inf_count = np.isinf(raw_1b).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: Found {nan_count} NaN and {inf_count} Inf values in PTD, cleaning...")
    
    # Clean the data
    raw_1b = np.nan_to_num(raw_1b, nan=0.0, posinf=0.0, neginf=0.0)
    raw_1b = np.clip(raw_1b, 0.0, 1e10)  # Reasonable upper bound
    
    if raw_1b.size <= 1:
        print("WARNING: PTD vector too small")
        device = getattr(model.n_feat_th, "device", torch.device("cpu"))
        model.ptd_vec = torch.zeros(1, 2, dtype=torch.float32, device=device)
        return
    
    raw = raw_1b[1:]  # Drop dummy 0th row
    
    # Safe log1p transformation
    log1p_raw = np.log1p(np.clip(raw, 0.0, None)).astype(np.float32)
    
    # Safe percentile ranking
    N = raw.shape[0]
    if N <= 1:
        print("WARNING: Only one node, using dummy values")
        ptd_features = np.array([[0.0, 0.5]], dtype=np.float32)
    else:
        try:
            # Stable sorting for percentiles
            order = np.argsort(raw, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(N, dtype=np.float32)
            percentile = (ranks + 0.5) / max(1.0, float(N))
            
            # Stack features
            ptd_features = np.stack([log1p_raw, percentile], axis=1).astype(np.float32)
            
        except Exception as e:
            print(f"ERROR in percentile computation: {e}")
            # Fallback to simple features
            ptd_features = np.column_stack([
                log1p_raw,
                np.linspace(0.0, 1.0, N, dtype=np.float32)
            ])
    
    # Final validation
    if np.any(np.isnan(ptd_features)) or np.any(np.isinf(ptd_features)):
        print("ERROR: NaN/Inf in final PTD features, using zeros")
        ptd_features = np.zeros((N, 2), dtype=np.float32)
    
    # Convert to tensor
    device = getattr(model.n_feat_th, "device", torch.device("cpu"))
    model.ptd_vec = torch.from_numpy(ptd_features).to(device).requires_grad_(False)
    
    # Debug info
    #print(f"PTD vector set: shape={model.ptd_vec.shape}, "
    #      f"mean={model.ptd_vec.mean():.4f}, "
    #      f"std={model.ptd_vec.std():.4f}, "
    #      f"min={model.ptd_vec.min():.4f}, "
    #      f"max={model.ptd_vec.max():.4f}")


# CONSERVATIVE SIMPLIFIED MODEL - Start with minimal changes
class ConservativeSimplifiedModel(torch.nn.Module):
    """Most conservative approach - minimal changes to your original"""
    
    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time', 
                 agg_method='attn', num_layers=3, n_head=4, null_idx=0, 
                 num_heads=2, drop_out=0.3, seq_len=None):
        super().__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        
        # Keep your EXACT original components
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        
        # ONLY change the PTD encoding - keep attention models the same for now
        self.attn_model_list = torch.nn.ModuleList([
            SaferSimplifiedPTDModel(  # Use safer version
                feat_dim=self.feat_dim,
                time_dim=self.feat_dim,
                ptd_enc_dim=64,  # Reduced from 128
                n_head=n_head,
                drop_out=drop_out
            ) for _ in range(num_layers)
        ])
        
        # Simplified PTD encoder
        self.ptd_encoder = SimplifiedPTDEncoder(output_dim=64)
        
        # PTD vector storage
        self.ptd_vec = None
        self.last_valid_counts = None

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv2(src_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        target_embed = self.tem_conv2(target_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score
    
    def set_ptd_vector(self, ptd_raw_1b):
        """Use safer PTD vector setting"""
        safer_set_simplified_ptd_vector(self, ptd_raw_1b)

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """Keep your exact logic, just change PTD dimensions"""
        
        if curr_layers == 0:
            device = self.n_feat_th.device
            src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
            return self.node_raw_embed(src_ids_th)
        
        device = self.n_feat_th.device
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Source features (exactly as before)
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        # PTD features with safety checks
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        if self.ptd_vec is not None and src_idx0.max() < self.ptd_vec.shape[0]:
            ptd_src_vals = self.ptd_vec[src_idx0]  # [B, 2]
            ptd_embed_src = self.ptd_encoder(ptd_src_vals)
            
            # Check for NaN in PTD embedding
            if torch.isnan(ptd_embed_src).any():
                print("WARNING: NaN in PTD embedding, using zeros")
                ptd_embed_src = torch.zeros_like(ptd_embed_src)
        else:
            print("WARNING: PTD vector not set or index out of bounds")
            ptd_embed_src = torch.zeros(B, 64, device=device)
        
        # Recursive call (exactly as before)
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l, cut_time_l=cut_time_l, ptd_l=None,
            curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors (exactly as before)
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding (exactly as before)
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors (exactly as before)
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat, cut_time_l=t_flat, ptd_l=None,
            curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors with safety
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        if self.ptd_vec is not None and idx_ngh0.max() < self.ptd_vec.shape[0]:
            ptd_ngh_vals = self.ptd_vec[idx_ngh0]
            ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
            ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
            
            # Check for NaN
            if torch.isnan(ptd_ngh_enc).any():
                print("WARNING: NaN in neighbor PTD encoding")
                ptd_ngh_enc = torch.zeros_like(ptd_ngh_enc)
        else:
            ptd_ngh_enc = torch.zeros(B, num_neighbors, 64, device=device)
        
        # Use safer attention model
        attn_m = self.attn_model_list[curr_layers - 1]
        local, attn_weights, valid_counts = attn_m(
            src=src_node_conv_feat, src_t=src_node_t_embed, src_ptd_enc=ptd_embed_src,
            seq=ngh_conv, seq_t=ngh_t_enc, seq_ptd_enc=ptd_ngh_enc,
            mask=mask
        )
        
        # Check for NaN in output
        if torch.isnan(local).any():
            print("ERROR: NaN in attention output!")
            local = torch.zeros_like(local)
        
        self.last_valid_counts = valid_counts
        
        return F.normalize(local, p=2, dim=1)




class SimplePTDAgnosticLayer(nn.Module):
    """Simpler version that's less likely to have dimension issues"""
    
    def __init__(self, feat_dim, time_dim, ptd_enc_dim=64, n_head=2, drop_out=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_enc_dim = ptd_enc_dim
        
        # PTD Quality Predictor
        self.ptd_quality_predictor = nn.Sequential(
            nn.Linear(ptd_enc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Keep your existing merger but make it PTD-quality aware
        self.structure_merger = MergeLayer(feat_dim + time_dim, feat_dim, feat_dim, feat_dim)
        self.ptd_merger = MergeLayer(feat_dim + time_dim + ptd_enc_dim, feat_dim, feat_dim, feat_dim)
        
        # Simple weighted combination
        self.final_combination = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        self.layer_norm = nn.LayerNorm(feat_dim)
        
    def forward(self, src, src_t, src_ptd_enc, seq, seq_t, seq_ptd_enc, mask):
        batch_size, seq_len, _ = seq.size()
        
        # Predict PTD quality
        predicted_quality = self.ptd_quality_predictor(src_ptd_enc).squeeze(-1)
        
        # Safe masking and aggregation
        valid_mask = ~mask
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        
        # Simple mean aggregation for structure pathway
        seq_agg = torch.sum(seq * valid_mask_expanded, dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        seq_t_agg = torch.sum(seq_t * valid_mask_expanded, dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        seq_ptd_agg = torch.sum(seq_ptd_enc * valid_mask_expanded, dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        
        # Structure pathway (without PTD)
        structure_features = torch.cat([seq_agg, seq_t_agg], dim=-1)
        structure_output = self.structure_merger(structure_features, src)
        
        # PTD pathway (with PTD)
        ptd_features = torch.cat([seq_agg, seq_t_agg, seq_ptd_agg], dim=-1)
        ptd_output = self.ptd_merger(ptd_features, src)
        
        # Adaptive combination based on PTD quality
        quality_weight = predicted_quality.unsqueeze(-1)
        
        # Combine the two pathways
        combined_features = torch.cat([
            structure_output * (1 - quality_weight),
            ptd_output * quality_weight
        ], dim=-1)
        
        final_output = self.final_combination(combined_features)
        final_output = self.layer_norm(final_output)
        
        # Create dummy attention weights for compatibility
        attention_weights = torch.ones(batch_size, seq_len, device=src.device) / seq_len
        attention_weights = attention_weights.masked_fill(mask, 0.0)
        
        neighbor_valid_counts = valid_mask.sum(dim=1).float()
        
        return final_output, attention_weights, neighbor_valid_counts
class PTDAgnosticModel(torch.nn.Module):
    """Modified version of your ConservativeSimplifiedModel with PTD-agnostic attention"""
    
    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time', 
                 agg_method='attn', num_layers=3, n_head=4, null_idx=0, 
                 num_heads=2, drop_out=0.3, seq_len=None):
        super().__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        
        # Keep your EXACT original components
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        
        # REPLACE: Use PTD-agnostic attention layers
        self.attn_model_list = torch.nn.ModuleList([
            SimplePTDAgnosticLayer(  # <-- This is the key change
                feat_dim=self.feat_dim,
                time_dim=self.feat_dim,
                ptd_enc_dim=64,
                n_head=n_head,
                drop_out=drop_out
            ) for _ in range(num_layers)
        ])
        
        # Keep your existing PTD encoder
        self.ptd_encoder = SimplifiedPTDEncoder(output_dim=64)
        
        # PTD vector storage
        self.ptd_vec = None
        self.last_valid_counts = None
        
        # Track PTD quality predictions for analysis
        self.last_ptd_qualities = None

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv2(src_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        target_embed = self.tem_conv2(target_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score
    
    def set_ptd_vector(self, ptd_raw_1b):
        """Keep your existing PTD vector setting"""
        safer_set_simplified_ptd_vector(self, ptd_raw_1b)

    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """Same as your existing method, just using PTD-agnostic attention"""
        
        if curr_layers == 0:
            device = self.n_feat_th.device
            src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
            return self.node_raw_embed(src_ids_th)
        
        device = self.n_feat_th.device
        B = len(src_idx_l)
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)
        
        # Source features (same as before)
        src_node_feat = self.node_raw_embed(src_ids_th)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_th))
        
        # PTD features (same as before)
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        if self.ptd_vec is not None and src_idx0.max() < self.ptd_vec.shape[0]:
            ptd_src_vals = self.ptd_vec[src_idx0]
            ptd_embed_src = self.ptd_encoder(ptd_src_vals)
            if torch.isnan(ptd_embed_src).any():
                ptd_embed_src = torch.zeros_like(ptd_embed_src)
        else:
            ptd_embed_src = torch.zeros(B, 64, device=device)
        
        # Recursive call (same as before)
        src_node_conv_feat = self.tem_conv2(
            src_idx_l=src_idx_l, cut_time_l=cut_time_l, ptd_l=None,
            curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )
        src_node_conv_feat = F.normalize(src_node_conv_feat, p=2, dim=1)
        
        # Get neighbors (same as before)
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)
        mask = (ngh_ids_th == 0)
        
        # Time encoding (same as before)
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)
        ngh_t_enc = self.time_encoder(dt_th)
        
        # Recursive neighbors (same as before)
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        ngh_conv_flat = self.tem_conv2(
            src_idx_l=ngh_flat, cut_time_l=t_flat, ptd_l=None,
            curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )
        ngh_conv = ngh_conv_flat.view(B, num_neighbors, -1)
        ngh_conv = F.normalize(ngh_conv, p=2, dim=2)
        
        # PTD for neighbors (same as before)
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1
        if self.ptd_vec is not None and idx_ngh0.max() < self.ptd_vec.shape[0]:
            ptd_ngh_vals = self.ptd_vec[idx_ngh0]
            ptd_ngh_enc = self.ptd_encoder(ptd_ngh_vals)
            ptd_ngh_enc = ptd_ngh_enc.masked_fill(mask.unsqueeze(-1), 0.0)
            if torch.isnan(ptd_ngh_enc).any():
                ptd_ngh_enc = torch.zeros_like(ptd_ngh_enc)
        else:
            ptd_ngh_enc = torch.zeros(B, num_neighbors, 64, device=device)
        
        # USE PTD-AGNOSTIC ATTENTION (this is the key difference)
        attn_m = self.attn_model_list[curr_layers - 1]
        local, attn_weights, valid_counts = attn_m(
            src=src_node_conv_feat, src_t=src_node_t_embed, src_ptd_enc=ptd_embed_src,
            seq=ngh_conv, seq_t=ngh_t_enc, seq_ptd_enc=ptd_ngh_enc,
            mask=mask
        )
        
        # Check for NaN in output
        if torch.isnan(local).any():
            print("ERROR: NaN in attention output!")
            local = torch.zeros_like(local)
        
        self.last_valid_counts = valid_counts
        
        return F.normalize(local, p=2, dim=1)
    



#27 Sept
class SimplePTDEncoder(nn.Module):
    """Simple, stable PTD encoder"""
    def __init__(self, input_dim=2, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),  # Add normalization for stability
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Better initialization
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, ptd_features):
        return self.encoder(ptd_features)


class SimplifiedPTDAttention(nn.Module):
    """Simplified attention model with direct PTD integration"""
    def __init__(self, feat_dim, time_dim, ptd_dim=64, n_head=2, dropout=0.2):
        super().__init__()
        
        # Calculate total input dimension
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.ptd_dim = ptd_dim
        self.total_dim = feat_dim + time_dim + ptd_dim
        
        # Projection layers for combined features
        self.src_projector = nn.Linear(self.total_dim, feat_dim)
        self.neighbor_projector = nn.Linear(self.total_dim, feat_dim)
        
        # Multi-head attention for neighbor aggregation
        self.attention = MultiHeadAttention(
            n_head=n_head, 
            d_model=feat_dim, 
            d_k=feat_dim // n_head, 
            d_v=feat_dim // n_head,
            dropout=dropout
        )
        
        # PTD-guided gating (simpler version)
        self.ptd_gate = nn.Sequential(
            nn.Linear(ptd_dim, feat_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feat_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.src_projector.weight)
        nn.init.xavier_uniform_(self.neighbor_projector.weight)
    
    def forward(self, src_feat, src_time, src_ptd, neighbor_feats, neighbor_times, neighbor_ptds, mask):
        """
        src_feat: [B, feat_dim] - source node features
        src_time: [B, time_dim] - source time encoding
        src_ptd: [B, ptd_dim] - source PTD encoding
        neighbor_feats: [B, N, feat_dim] - neighbor features
        neighbor_times: [B, N, time_dim] - neighbor time encodings
        neighbor_ptds: [B, N, ptd_dim] - neighbor PTD encodings
        mask: [B, N] - neighbor mask (True = invalid)
        """
        B, N, _ = neighbor_feats.shape
        
        # Ensure src tensors are 2D
        if src_feat.dim() == 3:
            src_feat = src_feat.squeeze(1)
        if src_time.dim() == 3:
            src_time = src_time.squeeze(1)
        if src_ptd.dim() == 3:
            src_ptd = src_ptd.squeeze(1)
        
        # Combine neighbor features
        neighbor_combined = torch.cat([neighbor_feats, neighbor_times, neighbor_ptds], dim=-1)  # [B, N, total_dim]
        
        # Project neighbor features to attention dimension
        neighbor_projected = self.neighbor_projector(neighbor_combined)  # [B, N, feat_dim]
        neighbor_projected = self.layer_norm(neighbor_projected)
        
        # Prepare query from source
        src_combined = torch.cat([src_feat, src_time, src_ptd], dim=-1)  # [B, total_dim]
        src_query = self.src_projector(src_combined).unsqueeze(1)  # [B, 1, feat_dim]
        
        # Apply attention (using neighbors as both key and value)
        mask_expanded = mask.unsqueeze(1) if mask is not None else None
        attended, attn_weights = self.attention(src_query, neighbor_projected, neighbor_projected, mask=mask_expanded)
        attended = attended.squeeze(1)  # [B, feat_dim]
        
        # Apply PTD gating
        gate = self.ptd_gate(src_ptd)  # [B, feat_dim]
        output = attended * gate + attended * (1 - gate) * 0.5
        
        # Residual connection with source features
        output = self.layer_norm(output + src_feat * 0.5)
        
        # Calculate valid neighbor counts for loss
        valid_counts = (~mask).sum(dim=1).float() if mask is not None else torch.ones(B, device=src_feat.device) * N
        
        return output, attn_weights.squeeze(1), valid_counts


# === NEW: PTD encoder (replaces SimplifiedPTDEncoder usage) ===================
class PTD_MLP(nn.Module):
    """
    Encodes raw PTD features to a dense embedding.
    Input dim=2 by default: [log1p(raw), percentile] from safer_set_simplified_ptd_vector.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2] or [B, N, 2]
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.view(B * N, D)
            x = self.act(self.fc1(x))
            x = self.fc2(x)
            return x.view(B, N, -1)
        else:
            x = self.act(self.fc1(x))
            return self.fc2(x)


# === NEW: PTD-enhanced multi-head attention ==================================
class PTDMultiHeadAttention(nn.Module):
    def __init__(self, node_dim: int, time_dim: int, ptd_dim: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert node_dim % n_head == 0, "node_dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = node_dim // n_head

        q_in = node_dim + ptd_dim
        kv_in = node_dim + time_dim + ptd_dim

        self.W_Q = nn.Linear(q_in, n_head * self.head_dim, bias=False)
        self.W_K = nn.Linear(kv_in, n_head * self.head_dim, bias=False)
        self.W_V = nn.Linear(kv_in, n_head * self.head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_head * self.head_dim, node_dim, bias=False)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, q_base: torch.Tensor, kv_base: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        q_base: [B, node_dim+ptd_dim]
        kv_base: [B, N, node_dim+time_dim+ptd_dim]
        mask: [B, N] (True where neighbor is invalid/pad). Can be None.
        """
        B, N, _ = kv_base.shape
        device = q_base.device
        node_dim = self.n_head * self.head_dim

        # Early exit: handle batches with zero valid neighbors
        if mask is not None:
            empty = mask.all(dim=1)               # [B]
        else:
            # no mask provided -> assume at least one neighbor
            empty = torch.zeros(B, dtype=torch.bool, device=device)

        out = torch.zeros(B, node_dim, device=device)
        attn_std_metric = 0.0  # <--- FIX: Initialize here to prevent UnboundLocalError


        if (~empty).any():
            sel = (~empty).nonzero(as_tuple=False).squeeze(1)  # indices with some valid neighbors
            q_sel = q_base.index_select(0, sel)                # [B_sel, qdim]
            kv_sel = kv_base.index_select(0, sel)              # [B_sel, N, dim]
            m_sel = None if mask is None else mask.index_select(0, sel)

            Q = self.W_Q(q_sel)                      # [B_sel, H*d]
            K = self.W_K(kv_sel)                     # [B_sel, N, H*d]
            V = self.W_V(kv_sel)                     # [B_sel, N, H*d]

            B_sel = q_sel.size(0)
            Q = Q.view(B_sel, self.n_head, self.head_dim).unsqueeze(2)     # [B_sel,H,1,d]
            K = K.view(B_sel, N, self.n_head, self.head_dim).permute(0,2,1,3)  # [B_sel,H,N,d]
            V = V.view(B_sel, N, self.n_head, self.head_dim).permute(0,2,1,3)  # [B_sel,H,N,d]

            scores = torch.matmul(Q, K.transpose(2,3)) / (self.head_dim ** 0.5)  # [B_sel,H,1,N]
            
            # Added 1 october
            # learnable temperature
            #self.temp = nn.Parameter(torch.ones(1) * 0.5)  # Start at 0.5
            #scores = scores / self.temp

            if m_sel is not None:
                m_exp = m_sel.view(B_sel, 1, 1, N).expand(-1, self.n_head, -1, -1)
                scores = scores.masked_fill(m_exp, float('-inf'))

                # If an entire row is -inf (shouldn’t happen after empty filtering), fix to zeros:
                bad = torch.isinf(scores).all(dim=-1, keepdim=True)  # [B_sel,H,1,1]
                if bad.any():
                    scores = torch.where(bad, torch.zeros_like(scores), scores)



            alpha = torch.softmax(scores, dim=-1)      # [B_sel,H,1,N]
            alpha = self.dropout(alpha)

            # Added 1 october
            #alpha = F.dropout(alpha, p=0.3, training=self.training)  # Higher dropout

            alpha_squeezed = alpha.squeeze(2)

            attn_std_metric = alpha_squeezed.std(dim=-1).mean(dim=1).mean().item()

            agg = torch.matmul(alpha, V).squeeze(2)     # [B_sel,H,d]
            agg = agg.reshape(B_sel, node_dim)          # [B_sel,H*d]
            out_sel = self.out(agg)                     # [B_sel,node_dim]
            out.index_copy_(0, sel, out_sel)

        # rows with empty neighbors get zeros → residual in conv layer keeps h_v_prev
        #added 28 sept
        return out, attn_std_metric # Note: out needs final projection applied if not done above.

        # return out
    

class PTDGate(nn.Module):
    """
    Multiplicative gate based on PTD embedding.
    Returns a vector in (0,1) with shape [B, node_dim].
    """
    def __init__(self, ptd_dim: int, node_dim: int):
        super().__init__()
        self.lin = nn.Linear(ptd_dim, node_dim)     # keep a handle to the Linear
        self.act = nn.Sigmoid()

        # Proper init
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, ptd_enc_src: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin(ptd_enc_src))
    
    
# === NEW: One PTD-aware temporal conv layer (attention + mean + residual) ====
class TemporalConvPTD(nn.Module):
    def __init__(self,
                 node_dim: int,
                 time_dim: int,
                 ptd_dim: int,
                 n_head: int,
                 drop_out: float = 0.2,
                 lambda_mix: float = 0.5):
        super().__init__()
        self.lambda_mix = float(lambda_mix)

        # PTD-aware attention aggregator
        self.attn = PTDMultiHeadAttention(node_dim=node_dim, time_dim=time_dim,
                                          ptd_dim=ptd_dim, n_head=n_head, dropout=drop_out)

        # Mean aggregation: input = [h_v | ptd_v | mean(h_u) | mean(time) | mean(ptd_u)]
        mean_in = 2 * (node_dim + ptd_dim) + time_dim
        self.mean_mlp = nn.Sequential(
            nn.Linear(mean_in, node_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(node_dim, node_dim)
        )

        # Fuse each path with previous h_v
        self.fuse_attn = nn.Linear(2 * node_dim, node_dim)
        self.fuse_mean = nn.Linear(2 * node_dim, node_dim)

        self.ptd_gate = PTDGate(ptd_dim=ptd_dim, node_dim=node_dim)
        
        self.ln = nn.LayerNorm(node_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self,
                h_v_prev: torch.Tensor,           # [B,node_dim]
                h_u_prev: torch.Tensor,           # [B,N,node_dim]
                time_enc_ngh: torch.Tensor,       # [B,N,time_dim]
                ptd_enc_src: torch.Tensor,        # [B,ptd_dim]
                ptd_enc_ngh: torch.Tensor,        # [B,N,ptd_dim]
                mask: torch.Tensor                # [B,N] True for invalid
                ) -> torch.Tensor:

        # 1) Attention aggregation (PTD in Q/K/V)
        q_base = torch.cat([h_v_prev, ptd_enc_src], dim=-1)                 # [B,node+ptd]
        kv_base = torch.cat([h_u_prev, time_enc_ngh, ptd_enc_ngh], dim=-1)  # [B,N,node+time+ptd]

        #new sept 28
        attn_agg_output_raw, self.last_attn_std = self.attn(q_base, kv_base, mask=mask) # Catch the STD
        attn_out = attn_agg_output_raw 

        #attn_out = self.attn(q_base, kv_base, mask=mask)                    # [B,node]

        # 2) Mean aggregation (mask out padded)
        valid = (~mask).float().clamp_min(0.0)          # [B,N]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]

        h_u_prev = h_u_prev * valid.unsqueeze(-1)
        time_enc_ngh = time_enc_ngh * valid.unsqueeze(-1)
        ptd_enc_ngh = ptd_enc_ngh * valid.unsqueeze(-1)

        mean_h = h_u_prev.sum(dim=1) / denom            # [B,node_dim]
        mean_t = time_enc_ngh.sum(dim=1) / denom        # [B,time_dim]
        mean_p = ptd_enc_ngh.sum(dim=1) / denom         # [B,ptd_dim]

        mean_in = torch.cat([h_v_prev, ptd_enc_src, mean_h, mean_t, mean_p], dim=-1)
        mean_out = self.mean_mlp(mean_in)               # [B,node]

        # 3) Fuse each with previous state
        attn_upd = self.fuse_attn(torch.cat([attn_out, h_v_prev], dim=-1))  # Now uses the output of PTDMultiHeadAttention
        mean_upd = self.fuse_mean(torch.cat([mean_out, h_v_prev], dim=-1))  # [B,node]

        # --- CRITICAL STEP: Apply PTD Gate to the Attention Update ---
        # Gate = Sigmoid(PTD_enc_src)
        gate_scalar = self.ptd_gate(ptd_enc_src)
        
        # Scale the attention update vector by the gate
        attn_upd_gated = attn_upd * gate_scalar
        mixed = self.lambda_mix * mean_upd + (1.0 - self.lambda_mix) * attn_upd_gated 
        
        out = self.ln(mixed + h_v_prev)
        return out


class PTDResidualPredictor(nn.Module):
    """
    Creates a direct prediction from PTD that serves as baseline.
    This is separate from PTD_MLP which creates embeddings for the main model.
    """
    def __init__(self, input_dim=2):
        super().__init__()
        # Direct path: PTD features -> scalar prediction
        self.direct = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )
        
        # Initialize conservatively - start near identity
        for m in self.direct:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, ptd_pair):
        # ptd_pair: [B, 2] containing [log1p(raw), percentile]
        return self.direct(ptd_pair).squeeze(-1)
class ConservativeSimplifiedModel_gemini(torch.nn.Module):
    """Most conservative approach - minimal changes to your original"""

    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time',
                 agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None, lambda_mix: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx

        # Keep your original components
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

        # === NEW: PTD encoder & PTD-aware conv stack (replaces attn_model_list) ===
        self.ptd_mlp = PTD_MLP(input_dim=2, hidden_dim=128, output_dim=128)   # matches safer_set_simplified_ptd_vector
        self.ptd_embed_dim = 128

        self.conv_layers = nn.ModuleList([
            TemporalConvPTD(node_dim=self.feat_dim,
                            time_dim=self.feat_dim,
                            ptd_dim=self.ptd_embed_dim,
                            n_head=n_head,
                            drop_out=drop_out,
                            lambda_mix=lambda_mix)
            for _ in range(num_layers)
        ])

        # PTD vector storage (set via set_ptd_vector)
        self.ptd_vec = None
        self.last_valid_counts = None



    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv2(src_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        target_embed = self.tem_conv2(target_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

    def set_ptd_vector(self, ptd_raw_1b):
        """Use safer PTD vector setting (already in your file)."""
        safer_set_simplified_ptd_vector(self, ptd_raw_1b)

    # === CHANGED: ConservativeSimplifiedModel.tem_conv2 =======================
    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        PTD-aware temporal conv:
          - Base case returns raw node embeddings.
          - Otherwise, recursively gets neighbor representations (l-1),
            builds PTD/time features, and applies TemporalConvPTD.
        """
        if curr_layers == 0:
            device = self.n_feat_th.device
            src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
            return self.node_raw_embed(src_ids_th)

        device = self.n_feat_th.device
        B = len(src_idx_l)
        L = curr_layers

        # Indices and times
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)              # [B]
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)  # [B,1]

        # Previous state h^{l-1}_v(t)
        h_v_prev = self.tem_conv2(src_idx_l=src_idx_l, cut_time_l=cut_time_l,
                                  ptd_l=None, curr_layers=L-1, num_neighbors=num_neighbors)  # [B,feat]
        h_v_prev = F.normalize(h_v_prev, p=2, dim=1)

        # Temporal neighbors (1 hop back in time)
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)           # [B,N]
        mask = (ngh_ids_th == 0)                                                # [B,N]

        # Time encodings for neighbors: Δt = t_cut - t_ngh
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)                      # [B,N]
        time_enc_ngh = self.time_encoder(dt_th)                                 # [B,N,feat_dim]

        # Neighbor representations h^{l-1}_u(t_u): recurse on flattened neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        h_u_prev_flat = self.tem_conv2(src_idx_l=ngh_flat, cut_time_l=t_flat,
                                       ptd_l=None, curr_layers=L-1, num_neighbors=num_neighbors)  # [B*N,feat]
        h_u_prev = h_u_prev_flat.view(B, num_neighbors, -1)
        h_u_prev = F.normalize(h_u_prev, p=2, dim=2)

        # === PTD encodings (using self.ptd_vec prepared by set_ptd_vector) ===
        # Source PTD: gather indices (0-based)
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        if self.ptd_vec is not None and src_idx0.max() < self.ptd_vec.shape[0]:
            ptd_src_vals = self.ptd_vec[src_idx0]                        # [B,2]
            ptd_enc_src = self.ptd_mlp(ptd_src_vals)                     # [B,128]
        else:
            ptd_enc_src = torch.zeros(B, self.ptd_embed_dim, device=device)

        # Neighbor PTD: gather and mask
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1                    # [B,N]
        if self.ptd_vec is not None and idx_ngh0.max() < self.ptd_vec.shape[0]:
            ptd_ngh_vals = self.ptd_vec[idx_ngh0]                        # [B,N,2]
            ptd_enc_ngh = self.ptd_mlp(ptd_ngh_vals)                     # [B,N,128]
            ptd_enc_ngh = ptd_enc_ngh.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            ptd_enc_ngh = torch.zeros(B, num_neighbors, self.ptd_embed_dim, device=device)

        # === Apply one PTD-aware temporal conv layer at depth L ===
        convL = self.conv_layers[L-1]
        out = convL(h_v_prev=h_v_prev,
                    h_u_prev=h_u_prev,
                    time_enc_ngh=time_enc_ngh,
                    ptd_enc_src=ptd_enc_src,
                    ptd_enc_ngh=ptd_enc_ngh,
                    mask=mask)

        # Track valid neighbor counts (used by your loss)
        self.last_valid_counts = (~mask).sum(dim=1).float()

        return F.normalize(out, p=2, dim=1)


class ContrastiveProjectionHead(nn.Module):
    """
    Non-linear projection head g(.) for contrastive learning.
    Maps the final GNN output (src_embed) to a stable, unit-norm contrastive space.
    """
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.hidden_dim = input_dim // 2
        
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)
        self.act = nn.ReLU()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Hidden layers (where explosion occurs)
        x = self.fc1(x)
        x = self.act(x)
        
        # 2. Final Projection
        z_raw = self.fc2(x)
        

        z_normalized = F.normalize(z_raw, dim=1) 
        return z_normalized



# In module.py, replace the old GlobalAttentionReadout

class StrongerAttentionReadout(nn.Module):
    """
    A more powerful attention readout using a set of learnable global memory tokens.
    This implements a two-stage attention process: graph-to-token, then token-to-node.
    """
    def __init__(self, embed_dim: int, n_head: int, num_tokens: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Learnable memory tokens that will act as a global summary
        self.memory_tokens = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # Attention layer for tokens to attend to nodes (graph-to-token)
        self.g2t_mha = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
        self.g2t_ln = nn.LayerNorm(embed_dim)

        # Attention layer for nodes to attend to tokens (token-to-node)
        self.t2n_mha = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
        self.t2n_ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape [Batch, Dim]
        batch_size = x.size(0)
        
        # Reshape for MHA: [Batch, Sequence, Dim]
        x_seq = x.unsqueeze(1)
        
        # Expand memory tokens to match the batch size
        tokens = self.memory_tokens.expand(batch_size, -1, -1)

        # --- Stage 1: Graph-to-Token Attention ---
        # The memory tokens (query) attend to the node embeddings (key, value)
        # to create a summary of the batch.
        # Note: We pass the whole batch x as key/value for each token.
        updated_tokens, _ = self.g2t_mha(tokens, x_seq, x_seq)
        updated_tokens = self.g2t_ln(tokens + updated_tokens)

        # --- Stage 2: Token-to-Node Attention ---
        # Each node embedding (query) attends to the set of updated summary tokens.
        global_context, _ = self.t2n_mha(x_seq, updated_tokens, updated_tokens)
        
        # Squeeze the sequence dimension back out and apply residual connection
        global_context = self.t2n_ln(x + global_context.squeeze(1))
        
        return global_context


class ConservativeSimplifiedModel_gemini_CONTR(torch.nn.Module):
    """Most conservative approach - minimal changes to your original"""

    def __init__(self, ngh_finder, n_feat, attn_mode='prod', use_time='time',
                 agg_method='attn', num_layers=3, n_head=4, null_idx=0,
                 num_heads=2, drop_out=0.3, seq_len=None, lambda_mix: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx

        # Keep your original components
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

        # === NEW: PTD encoder & PTD-aware conv stack (replaces attn_model_list) ===
        self.ptd_mlp = PTD_MLP(input_dim=2, hidden_dim=128, output_dim=128)   # matches safer_set_simplified_ptd_vector
        self.ptd_embed_dim = 128

        self.conv_layers = nn.ModuleList([
            TemporalConvPTD(node_dim=self.feat_dim,
                            time_dim=self.feat_dim,
                            ptd_dim=self.ptd_embed_dim,
                            n_head=n_head,
                            drop_out=drop_out,
                            lambda_mix=lambda_mix)
            for _ in range(num_layers)
        ])

        self.attention_readout = StrongerAttentionReadout(embed_dim=self.feat_dim, n_head=n_head)

        # PTD vector storage (set via set_ptd_vector)
        self.ptd_vec = None
        self.last_valid_counts = None


        #added 1 oct
        self.ptd_baseline_predictor = PTDResidualPredictor(input_dim=2)

        self.contrast_head = ContrastiveProjectionHead(
                    input_dim=self.feat_dim, 
                    output_dim=64 # Final dimension of the contrastive embedding 'z'
                )
    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv2(src_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        target_embed = self.tem_conv2(target_idx_l, cut_time_l, None, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

    def set_ptd_vector(self, ptd_raw_1b):
        """Use safer PTD vector setting (already in your file)."""
        safer_set_simplified_ptd_vector(self, ptd_raw_1b)

    # === CHANGED: ConservativeSimplifiedModel.tem_conv2 =======================
    def tem_conv2(self, src_idx_l, cut_time_l, ptd_l, curr_layers, num_neighbors=12):
        """
        PTD-aware temporal conv:
          - Base case returns raw node embeddings.
          - Otherwise, recursively gets neighbor representations (l-1),
            builds PTD/time features, and applies TemporalConvPTD.
        """
        if curr_layers == 0:
            device = self.n_feat_th.device
            src_ids_th = torch.from_numpy(src_idx_l).long().to(device)
            return self.node_raw_embed(src_ids_th)

        device = self.n_feat_th.device
        B = len(src_idx_l)
        L = curr_layers

        # Indices and times
        src_ids_th = torch.from_numpy(src_idx_l).long().to(device)              # [B]
        cut_time_th = torch.from_numpy(cut_time_l).float().to(device).unsqueeze(1)  # [B,1]

        # Previous state h^{l-1}_v(t)
        h_v_prev = self.tem_conv2(src_idx_l=src_idx_l, cut_time_l=cut_time_l,
                                  ptd_l=None, curr_layers=L-1, num_neighbors=num_neighbors)  # [B,feat]
        h_v_prev = F.normalize(h_v_prev, p=2, dim=1)

        # Temporal neighbors (1 hop back in time)
        ngh_nodes_np, ngh_times_np = self.ngh_finder.get_temporal_neighbor(
            tuple(src_idx_l), tuple(cut_time_l), num_neighbors=num_neighbors
        )
        ngh_ids_th = torch.from_numpy(ngh_nodes_np).long().to(device)           # [B,N]
        mask = (ngh_ids_th == 0)                                                # [B,N]

        # Time encodings for neighbors: Δt = t_cut - t_ngh
        dt_np = cut_time_l[:, np.newaxis] - ngh_times_np
        dt_th = torch.from_numpy(dt_np).float().to(device)                      # [B,N]
        time_enc_ngh = self.time_encoder(dt_th)                                 # [B,N,feat_dim]

        # Neighbor representations h^{l-1}_u(t_u): recurse on flattened neighbors
        ngh_flat = ngh_nodes_np.reshape(-1)
        t_flat = ngh_times_np.reshape(-1)
        h_u_prev_flat = self.tem_conv2(src_idx_l=ngh_flat, cut_time_l=t_flat,
                                       ptd_l=None, curr_layers=L-1, num_neighbors=num_neighbors)  # [B*N,feat]
        h_u_prev = h_u_prev_flat.view(B, num_neighbors, -1)
        h_u_prev = F.normalize(h_u_prev, p=2, dim=2)

        # === PTD encodings (using self.ptd_vec prepared by set_ptd_vector) ===
        # Source PTD: gather indices (0-based)
        src_idx0 = torch.clamp(src_ids_th, min=1) - 1
        if self.ptd_vec is not None and src_idx0.max() < self.ptd_vec.shape[0]:
            ptd_src_vals = self.ptd_vec[src_idx0]                        # [B,2]
            ptd_enc_src = self.ptd_mlp(ptd_src_vals)                     # [B,128]
        else:
            ptd_enc_src = torch.zeros(B, self.ptd_embed_dim, device=device)

        # Neighbor PTD: gather and mask
        idx_ngh0 = torch.clamp(ngh_ids_th, min=1) - 1                    # [B,N]
        if self.ptd_vec is not None and idx_ngh0.max() < self.ptd_vec.shape[0]:
            ptd_ngh_vals = self.ptd_vec[idx_ngh0]                        # [B,N,2]
            ptd_enc_ngh = self.ptd_mlp(ptd_ngh_vals)                     # [B,N,128]
            ptd_enc_ngh = ptd_enc_ngh.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            ptd_enc_ngh = torch.zeros(B, num_neighbors, self.ptd_embed_dim, device=device)

        # === Apply one PTD-aware temporal conv layer at depth L ===
        convL = self.conv_layers[L-1]
        out = convL(h_v_prev=h_v_prev,
                    h_u_prev=h_u_prev,
                    time_enc_ngh=time_enc_ngh,
                    ptd_enc_src=ptd_enc_src,
                    ptd_enc_ngh=ptd_enc_ngh,
                    mask=mask)

        # Track valid neighbor counts (used by your loss)
        self.last_valid_counts = (~mask).sum(dim=1).float()
        self.last_attn_std = convL.last_attn_std 

        return F.normalize(out, p=2, dim=1)

