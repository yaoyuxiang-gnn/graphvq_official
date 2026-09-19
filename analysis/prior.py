"""Autoregressive GRU / Transformer priors and their training/sampling helpers.

Both priors satisfy the same contract used by ``train_prior`` / ``sample_tokens``:
``forward(seq, cond)`` returns logits of shape ``(B, L, vocab)`` where position ``i``
predicts ``seq[i+1]`` given ``seq[<=i]`` (and ``cond`` when provided).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUPrior(nn.Module):
    """Autoregressive GRU prior with an optional conditioning input.

    When ``cond_dim == 0`` (and ``cond`` is None) this reduces to an unconditional
    token prior; when ``cond_dim > 0`` the initial GRU state is derived from the
    conditioning vector (used by the edge prior of the two-stage generator).
    """

    def __init__(self, vocab: int, hidden: int, cond_dim: int = 0, num_layers: int = 2) -> None:
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab, hidden)
        self.cond = nn.Linear(cond_dim, hidden * num_layers) if cond_dim > 0 else None
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, seq, cond=None):
        h = self.embed(seq)
        h0 = None
        if self.cond is not None and cond is not None:
            h0 = self.cond(cond).view(self.num_layers, seq.size(0), self.hidden)
        out, _ = self.gru(h, h0)
        return self.head(out)


class TransformerPrior(nn.Module):
    """Causal transformer prior (reviewer Q1 ablation: GRU vs. transformer).

    When ``cond_dim > 0`` the conditioning vector is projected to ``hidden`` and
    prepended as the first (unmasked) token, so every subsequent position can
    attend to it; its own output position is dropped to preserve the
    ``logits[i] predicts seq[i+1]`` contract shared with :class:`GRUPrior`.
    """

    def __init__(self, vocab: int, hidden: int, cond_dim: int = 0,
                 num_layers: int = 2, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden % nhead == 0, "hidden must be divisible by nhead"
        self.vocab = vocab
        self.hidden = hidden
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden) if cond_dim > 0 else None
        layer = nn.TransformerEncoderLayer(
            hidden, nhead, dim_feedforward=4 * hidden, dropout=dropout,
            batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, seq, cond=None):
        h = self.embed(seq)  # (B, L, H)
        use_cond = self.cond_proj is not None and cond is not None
        if use_cond:
            c = self.cond_proj(cond).unsqueeze(1)  # (B, 1, H)
            h = torch.cat([c, h], dim=1)           # (B, 1+L, H)
        L = h.size(1)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1)
        out = self.enc(h, mask=mask)
        if use_cond:
            # out[j] predicts input[j+1]; logits[i] must predict seq[i+1] = input[i+2].
            out = out[:, 1:]
        return self.head(out)


def make_prior(vocab: int, hidden: int, cond_dim: int = 0,
               prior_type: str = "gru", num_layers: int = 2) -> nn.Module:
    """Factory for autoregressive priors (``gru`` | ``transformer``)."""
    if prior_type == "gru":
        return GRUPrior(vocab, hidden, cond_dim=cond_dim, num_layers=num_layers)
    if prior_type == "transformer":
        nhead = 4 if hidden % 4 == 0 else 2
        return TransformerPrior(vocab, hidden, cond_dim=cond_dim,
                                num_layers=num_layers, nhead=nhead)
    raise ValueError(f"unknown prior_type: {prior_type}")


def train_prior(prior, seqs, epochs, lr, device, sos, eos, bs=64, conds=None):
    opt = torch.optim.Adam(prior.parameters(), lr=lr)
    seqs = [torch.tensor([sos] + t + [eos], dtype=torch.long) for t in seqs]
    for _ in range(epochs):
        perm = torch.randperm(len(seqs), device=device)
        for s in range(0, len(seqs), bs):
            batch = seqs[s:s + bs]
            L = max(len(x) for x in batch)
            padded = torch.zeros(len(batch), L, dtype=torch.long, device=device)
            for k, x in enumerate(batch):
                padded[k, :len(x)] = x
            cond = torch.stack([conds[i] for i in perm[s:s + bs]]).to(device) if conds is not None else None
            opt.zero_grad()
            logits = prior(padded, cond)[:, :-1].reshape(-1, prior.vocab)
            target = padded[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            opt.step()
    return prior


def sample_tokens(prior, n_steps, cond, device, sos, eos, temperature=1.0, mask_eos=False, stop_on_eos=True):
    """Autoregressively sample a token sequence (excluding the initial ``sos``)."""
    seq = [sos]
    for _ in range(n_steps):
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        logits = prior(inp, cond.unsqueeze(0) if cond is not None else None)[0, -1] / temperature
        logits[sos] = -1e9
        if mask_eos:
            logits[eos] = -1e9
        probs = F.softmax(logits, dim=-1)
        t = torch.multinomial(probs, 1).item()
        if stop_on_eos and t == eos:
            break
        seq.append(t)
    return seq[1:]
