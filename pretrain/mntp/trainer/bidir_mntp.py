"""LLM2Vec-style masked next-token prediction (MNTP) for a decoder with bidirectional attention.

Input positions (never BOS, never position 0) are replaced by the mask token with probability p.
The output at position t predicts the ORIGINAL token at t+1, and only when t+1 was masked. Every selected
position is replaced (no BERT 80/10/10 keep/random), so a target token is never visible to its predictor.
Masks are a deterministic function of (seed, rank, key_a, key_b).
"""
import torch


def make_mntp(mask_id, bos_id, prob, seed, rank):
    checked = [False]

    def mntp(x, y, key_a, key_b):
        if not checked[0]:
            assert torch.equal(y[:, :-1], x[:, 1:]), "loader targets must be inputs shifted by one"
            checked[0] = True
        g = torch.Generator(device=x.device)
        g.manual_seed((((seed * 1000003 + rank) * 1000003 + key_a) * 1000003 + key_b) % (2**63 - 1))
        m = (torch.rand(x.shape, generator=g, device=x.device) < prob) & (x != bos_id)
        m[:, 0] = False
        x_in = torch.where(m, torch.full_like(x, mask_id), x)
        tgt = torch.full_like(y, -1)
        tgt[:, :-1] = torch.where(m[:, 1:], y[:, :-1], torch.full_like(y[:, :-1], -1))
        return x_in, tgt

    return mntp
