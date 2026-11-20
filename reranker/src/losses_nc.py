"""
Contrastive loss over pointwise yes/no scores, plus optional KD.

The group (1 positive + K negatives) is encoded as K+1 SEPARATE sequences, and
InfoNCE softmaxes over their K+1 scalars. Consequence worth remembering: unlike
the jina listwise model -- where every negative lives inside the positive's own
sequence and the loss decomposes per example -- here the loss COUPLES the K+1
sequences, so they must all be in the same backward pass and gradient
accumulation across a group is not possible.

No dispersive or dual loss: both are artifacts of extracting embeddings from a
shared context window and have no meaning for independently-scored pairs.
"""
import torch
import torch.nn.functional as F


def infonce(scores, pos_idx, tau: float):
    """scores: [B, K+1] fp32 logit gaps. Random guessing = ln(K+1)."""
    return F.cross_entropy(scores.float() / tau, pos_idx)


def kd_loss(scores, teacher, pos_idx, tau_s: float, tau_t: float, row_w=None):
    """KL over the NEGATIVES only, positive excluded and the rest renormalised,
    so rank loss owns positive-vs-rest and KD owns the ordering among negatives.
    Gathers rather than masking with -inf (that gives 0 * inf = NaN)."""
    B, K = scores.shape
    keep = torch.ones_like(scores, dtype=torch.bool)
    keep.scatter_(1, pos_idx.unsqueeze(1), False)
    s_neg = scores.float().masked_select(keep).view(B, K - 1)
    t_neg = teacher.float().masked_select(keep).view(B, K - 1)
    log_p_s = F.log_softmax(s_neg / tau_s, dim=-1)
    p_t = F.softmax(t_neg / tau_t, dim=-1)
    per_row = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(-1)
    if row_w is not None:
        w = row_w.to(per_row.dtype)
        return (per_row * w).sum() / w.sum().clamp(min=1e-6) * (tau_s ** 2)
    return per_row.mean() * (tau_s ** 2)


def rank_distill_loss(scores, trank, tau, row_w=None):
    """Rank distillation from a LISTWISE teacher's order (not used by any paper row): RankNet over every pair the
    teacher orders within the group, nDCG-weighted so pairs near the top count most.
      scores [B, G] student scores; trank [B, G] teacher ranks (1 = best, among ALL the query's
      candidates -- only their order inside the group is used).
      pair (i above j): w_ij * softplus(-(s_i - s_j) / tau),  w_ij = |1/log2(1+l_i) - 1/log2(1+l_j)|
      with l = the teacher's rank WITHIN the group (1..G).
    Uses the order only: no teacher scores, calibration or teacher temperature.
    Returns (loss, agreement): agreement = weighted share of teacher pairs the student already orders
    the same way (for logging)."""
    t = trank.float()
    loc = t.argsort(1).argsort(1).float() + 1.0                     # in-group teacher rank 1..G
    gain = 1.0 / torch.log2(1.0 + loc)
    above = t[:, :, None] < t[:, None, :]                            # teacher puts i above j
    w = (gain[:, :, None] - gain[:, None, :]).abs() * above
    d = (scores[:, :, None] - scores[:, None, :]).float()
    wsum = w.sum((1, 2)).clamp(min=1e-9)
    per_row = (torch.nn.functional.softplus(-d / tau) * w).sum((1, 2)) / wsum
    agree = ((d > 0).float() * w).sum((1, 2)) / wsum
    if row_w is not None:
        rw = row_w.to(per_row.dtype)
        return (per_row * rw).sum() / rw.sum().clamp(min=1e-6), (agree * rw).sum() / rw.sum().clamp(min=1e-6)
    return per_row.mean(), agree.mean()


def total_loss(scores, pos_idx, tau, teacher=None, w_kd=0.0, tau_t=1.0,
               kd_row_w=None, trank=None, w_rkd=0.0, rkd_row_w=None, tau_s_kd=None):
    """tau_s_kd: STUDENT temperature for the KD term only (LCE always uses `tau`).

    It defaults to `tau` -- what run J used, with tau_t = 2. That is not Hinton's KD: the KL is
    minimised at softmax(s/tau_s) = softmax(t/tau_t), i.e. s = (tau_s/tau_t) t + c, so tau_s != tau_t
    asks the student to reproduce the teacher's scores at a DIFFERENT SCALE. J's scores ended at half
    I's (pos 11.61 vs 24.38, neg 4.57 vs 15.19) for exactly that reason. Setting tau_s_kd = tau_t
    restores the standard fixed point s = t + c, softening both sides equally.
    """
    parts = {"rank": infonce(scores, pos_idx, tau)}
    if teacher is not None and w_kd:
        parts["kd"] = kd_loss(scores, teacher, pos_idx, tau if tau_s_kd is None else tau_s_kd,
                              tau_t, row_w=kd_row_w)
    if trank is not None:
        if w_rkd:
            parts["rkd"], parts["rkd_agree"] = rank_distill_loss(scores, trank, tau, row_w=rkd_row_w)
        else:   # KD off but teacher ranks present: log agreement with the teacher, no gradient
            with torch.no_grad():
                parts["rkd"], parts["rkd_agree"] = rank_distill_loss(scores.detach(), trank, tau, row_w=rkd_row_w)
    w = {"rank": 1.0, "kd": w_kd, "rkd": w_rkd, "rkd_agree": 0.0}      # rkd_agree: logged only
    # zero-weight parts are logged only and never enter the loss (x + 0.0 == x, so KD runs are unchanged)
    return sum(w[k] * v for k, v in parts.items() if w[k]), parts
