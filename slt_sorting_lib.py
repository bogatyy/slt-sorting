"""
slt_sorting_lib.py -- "Same loss, different algorithm": digit-sorting transformers.

Two small decoder-only transformers are trained on exactly the same sorting task to
(essentially) the same training loss, but are steered -- via auxiliary supervision of an
intermediate "state" -- into two different algorithms:

  * MIN   ("track the current minimum"): at each output step keep the last emitted value v
          and emit the smallest *still-available* input digit >= v.  All quantities are
          comparisons ("is digit d still available?"), so nothing depends on the absolute
          length of the sequence.
  * HIST  ("counting sort"): compute a histogram of absolute counts of the 10 digits once,
          then emit digit d for output indices  C(d-1) <= t < C(d)  (C = cumulative counts).
          This needs absolute counts and the absolute output index t.

Everything (data, model, training, evaluation, mechanistic tests, LLC estimation) is in this
one file so that the notebook can inline it and stay self-contained.
"""
import math, time, copy, json, os, random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------
# Tokens.  Input digits and output digits use *different* token ids so that the model can
# tell them apart by content alone (no positional information needed for that).
# ----------------------------------------------------------------------------------------
N_DIGITS = 10
SEP = 10                # separator token
OUT0 = 11               # output digit d  ->  token id OUT0 + d
VOCAB = OUT0 + N_DIGITS # 21

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def sample_inputs(B: int, n: int, alphabet: int = N_DIGITS, device=DEVICE, generator=None):
    """B random digit strings of length n, digits i.i.d. uniform on {0..alphabet-1}."""
    return torch.randint(0, alphabet, (B, n), device=device, generator=generator)


def build_tokens(x: torch.Tensor):
    """[x_1..x_n, SEP, y_1..y_n] with y = sorted(x).  Shape (B, 2n+1)."""
    B, n = x.shape
    y = x.sort(dim=1).values
    sep = torch.full((B, 1), SEP, dtype=torch.long, device=x.device)
    return torch.cat([x, sep, y + OUT0], dim=1)


def histogram(x: torch.Tensor):
    """Absolute counts of each digit, (B, 10) float."""
    return F.one_hot(x, N_DIGITS).sum(1).float()


def availability(x: torch.Tensor, y: torch.Tensor):
    """availability[b, t, d] = (count of d in x_b) > (count of d among y_b[:t]).
    This is the state of the MIN algorithm at output step t (0-indexed): which digits are
    still left to emit.  y_t == min{d : availability[t, d]} always holds."""
    h = F.one_hot(x, N_DIGITS).sum(1)                       # (B, 10)
    yo = F.one_hot(y, N_DIGITS)                             # (B, n, 10)
    out_before = torch.cumsum(yo, dim=1) - yo               # counts of y[:, :t] (exclusive)
    return (h[:, None, :] - out_before) > 0                 # (B, n, 10) bool


def counting_sort_state(x: torch.Tensor):
    """State of the HIST algorithm: the histogram (same at every output step) and the output
    index t.  Returns (hist (B,10), t (n,))."""
    B, n = x.shape
    return histogram(x), torch.arange(n, device=x.device, dtype=torch.float)


def state_targets(x: torch.Tensor, y: torch.Tensor, mode: str):
    """The 11-dim supervised state at each of the n decision points, (B, n, 11), or None.
      MIN : [availability bits (10) | v]  with v = previously emitted digit (-1 at the separator)
      HIST: [cumulative counts C(d) (10) | t] with C(d) = #inputs <= d, t = 0-indexed output index
    Both states carry 10 per-digit numbers plus one 'where am I' number: in *value* space for the
    min-tracker (the current minimum v), in *index* space for counting sort (the output index t)."""
    B, n = x.shape
    if mode == "min":
        a = availability(x, y).float()
        v = torch.cat([torch.full((B, 1), -1.0, device=x.device), y[:, :-1].float()], 1)
        return torch.cat([a, v[:, :, None]], -1)
    if mode == "hist":
        h, t = counting_sort_state(x)
        C = torch.cumsum(h, 1)                                   # C[d] = #inputs <= d  (C[9] = n)
        return torch.cat([C[:, None, :].expand(B, n, N_DIGITS), t[None, :, None].expand(B, n, 1)], -1)
    return None


# ----------------------------------------------------------------------------------------
# Model: a tiny GPT-style decoder with (optional) learned absolute positional embeddings,
# a 10-way sorting head, and three *auxiliary linear probe heads* that read the residual
# stream after `probe_layer` blocks.  The probe heads are only used during training.
# ----------------------------------------------------------------------------------------
@dataclass
class Config:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 512
    max_len: int = 96        # supports input length n <= 47
    pos: str = "learned"     # 'learned' (zero-init absolute) | 'none' (NoPE) | 'sinusoidal'
    probe_layer: int = 1     # residual stream after this many blocks feeds the (non-bottleneck) probes
    bottleneck: bool = True  # route the sorting decision through an explicit K-dim "state" vector
    state_dim: int = 11      # 10 per-digit slots + 1 "where am I" slot (v for MIN, t for HIST)
    readout_hidden: int = 64


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, return_attn=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))          # (B, h, T, dh)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)         # (B, h, T, T)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        a = s.softmax(-1)
        y = (a @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y), (a if return_attn else None)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(),
                                 nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x, return_attn=False):
        a, attn = self.attn(self.ln1(x), return_attn)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, attn


def sinusoidal_table(max_len, d):
    pos = torch.arange(max_len, dtype=torch.float)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = ang.sin(); pe[:, 1::2] = ang.cos()
    return pe


class SortingTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB, cfg.d_model)
        if cfg.pos == "learned":
            # zero-initialised: positions that are never seen in training stay *exactly* zero,
            # i.e. an unseen position contributes no signal at all (rather than random noise).
            self.pos_emb = nn.Parameter(torch.zeros(cfg.max_len, cfg.d_model))
        elif cfg.pos == "sinusoidal":
            self.register_buffer("pos_table", sinusoidal_table(cfg.max_len, cfg.d_model))
            self.pos_emb = None
        else:
            self.pos_emb = None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if cfg.bottleneck:
            # residual -> K-dim state -> small MLP readout -> 10 logits.  The *only* path from the
            # transformer body to the sorting decision goes through the K numbers in `state`.
            self.state = nn.Linear(cfg.d_model, cfg.state_dim)
            self.readout = nn.Sequential(nn.Linear(cfg.state_dim, cfg.readout_hidden), nn.GELU(),
                                         nn.Linear(cfg.readout_hidden, cfg.readout_hidden), nn.GELU(),
                                         nn.Linear(cfg.readout_hidden, N_DIGITS))
        else:
            self.head = nn.Linear(cfg.d_model, N_DIGITS)
        # auxiliary probe heads (training-time scaffolding for the non-bottleneck variant only)
        self.probe_hist = nn.Linear(cfg.d_model, N_DIGITS)   # absolute counts / 10
        self.probe_idx = nn.Linear(cfg.d_model, 1)           # output index t / 10
        self.probe_avail = nn.Linear(cfg.d_model, N_DIGITS)  # availability bits (logits)

    def main_params(self):
        """Parameters of the sorting model proper (probe heads excluded)."""
        return [p for name, p in self.named_parameters() if not name.startswith("probe_")]

    def forward(self, tokens, return_attn=False):
        B, T = tokens.shape
        x = self.tok_emb(tokens)
        if self.cfg.pos == "learned":
            x = x + self.pos_emb[:T][None]
        elif self.cfg.pos == "sinusoidal":
            x = x + self.pos_table[:T][None]
        attns, resid_probe = [], None
        for i, blk in enumerate(self.blocks):
            x, a = blk(x, return_attn)
            attns.append(a)
            if i + 1 == self.cfg.probe_layer:
                resid_probe = x
        if resid_probe is None:
            resid_probe = x
        xf = self.ln_f(x)
        if self.cfg.bottleneck:
            state = self.state(xf)
            logits = self.readout(state)
        else:
            state = None
            logits = self.head(xf)
        return {"logits": logits, "resid": resid_probe, "final": x, "state": state, "attn": attns}


def n_params(model):
    return sum(p.numel() for p in model.main_params())


# ----------------------------------------------------------------------------------------
# Losses.  Predictions for y_0..y_{n-1} are read at positions n..2n-1 (SEP, y_0, .., y_{n-2}).
# ----------------------------------------------------------------------------------------
def forward_sort(model, x, return_attn=False):
    B, n = x.shape
    toks = build_tokens(x)
    out = model(toks[:, :2 * n], return_attn=return_attn)
    y = toks[:, n + 1:] - OUT0
    return out, y


def compute_losses(model, x, mode: str, aux_weight: float = 1.0):
    """mode in {'min', 'hist', 'none'}.  Returns (total, dict of scalars)."""
    B, n = x.shape
    out, y = forward_sort(model, x)
    logits = out["logits"][:, n:, :]                       # (B, n, 10)
    ce = F.cross_entropy(logits.reshape(-1, N_DIGITS), y.reshape(-1))
    acc = (logits.argmax(-1) == y).float().mean()
    resid = out["resid"][:, n:, :]                         # (B, n, d) at the n decision points
    stats = {"ce": ce.item(), "acc": acc.item()}
    aux = torch.zeros((), device=x.device)
    if model.cfg.bottleneck:
        tgt = state_targets(x, y, mode)
        if tgt is not None:
            state = out["state"][:, n:, :]
            aux = F.mse_loss(state, tgt)
            stats["aux_state_mse"] = aux.item()
            if mode == "min":
                stats["avail_bit_acc"] = ((state[..., :10] > 0.5) == (tgt[..., :10] > 0.5)).float().mean().item()
            else:
                stats["count_acc"] = (state[..., :10].round() == tgt[..., :10]).float().mean().item()
    elif mode == "hist":
        h, t = counting_sort_state(x)
        ph = model.probe_hist(resid)                       # (B, n, 10)
        pt = model.probe_idx(resid).squeeze(-1)            # (B, n)
        l_h = F.mse_loss(ph, (h / 10.0)[:, None, :].expand_as(ph))
        l_t = F.mse_loss(pt, (t / 10.0)[None, :].expand_as(pt))
        aux = l_h + l_t
        stats.update({"aux_hist": l_h.item(), "aux_idx": l_t.item()})
    elif mode == "min":
        a = availability(x, y).float()
        pa = model.probe_avail(resid)
        l_a = F.binary_cross_entropy_with_logits(pa, a)
        aux = l_a
        stats.update({"aux_avail": l_a.item(), "avail_bit_acc": ((pa > 0) == (a > 0.5)).float().mean().item()})
    total = ce + aux_weight * aux
    stats["aux"] = aux.item(); stats["total"] = total.item()
    return total, stats


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
def train_model(mode, cfg: Config, steps=8000, state_frac=0.7, batch=256, lr=1e-3, lr_joint=3e-4,
                wd=0.01, n_range=(4, 10), state_noise=0.1, readout_lr_mult=3.0, warmup=200, seed=0,
                log_every=250, device=DEVICE, verbose=True):
    """Supervised-state training in two phases.

    Phase 1 ("state phase", the first `state_frac` of the steps; skipped for mode 'none'):
      * the transformer body + state head are trained ONLY to output the algorithm's state
        (MSE to `state_targets`), they never see the sorting loss;
      * the read-out MLP is trained ONLY on the sorting loss, and reads the *true* state
        (plus a little Gaussian noise so that it learns margins) -- teacher forcing of the state.
      The state is therefore a hard, supervised interface: nothing else can flow through it.
    Phase 2 ("joint phase"): everything is fine-tuned end-to-end on the pure sorting loss with a
      smaller learning rate, so that the final weights are a minimum of the objective we compare.
    The log records, at every step, the sorting loss of the *composed* model (body -> state -> read-out),
    which is what is evaluated afterwards."""
    set_seed(seed)
    model = SortingTransformer(cfg).to(device)
    assert cfg.bottleneck, "supervised-state training needs the state bottleneck"
    S1 = int(steps * state_frac) if mode != "none" else 0
    S2 = steps - S1
    log, t0 = [], time.time()

    def cosine(s, S, base):
        if s < warmup:
            return base * (s + 1) / warmup
        p = (s - warmup) / max(1, S - warmup)
        return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    # ---------------- phase 1: state phase ----------------
    ro_params = list(model.readout.parameters()); ro_ids = {id(p) for p in ro_params}
    body_params = [p for p in model.parameters() if id(p) not in ro_ids]
    opt = torch.optim.AdamW([{"params": body_params, "mult": 1.0}, {"params": ro_params, "mult": readout_lr_mult}],
                            lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    for s in range(S1):
        for g in opt.param_groups:
            g["lr"] = cosine(s, S1, lr) * g["mult"]
        n = random.randint(*n_range)
        x = sample_inputs(batch, n, device=device)
        out, y = forward_sort(model, x)
        tgt = state_targets(x, y, mode)                              # (B, n, K)
        state = out["state"][:, n:, :]
        l_state = F.mse_loss(state, tgt)                             # body + state head
        logits_tf = model.readout(tgt + state_noise * torch.randn_like(tgt))
        l_read = F.cross_entropy(logits_tf.reshape(-1, N_DIGITS), y.reshape(-1))   # read-out only
        loss = l_state + l_read
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == S1 - 1:
            with torch.no_grad():
                logits = out["logits"][:, n:, :]
                ce = F.cross_entropy(logits.reshape(-1, N_DIGITS), y.reshape(-1)).item()
                acc = (logits.argmax(-1) == y).float().mean().item()
                read_acc = (logits_tf.argmax(-1) == y).float().mean().item()
            rec = {"phase": 1, "step": s, "n": n, "ce": ce, "acc": acc, "state_mse": l_state.item(),
                   "readout_ce": l_read.item(), "readout_acc": read_acc, "lr": cosine(s, S1, lr), "time": time.time() - t0}
            log.append(rec)
            if verbose:
                print(f"[{mode:4s} P1] step {s:5d} n={n:2d} composed ce={ce:.4f} acc={acc:.4f} | state mse={l_state.item():.5f} | read-out(true state) ce={l_read.item():.4f} acc={read_acc:.4f} ({rec['time']:.0f}s)")

    # ---------------- phase 2: joint fine-tuning on the pure sorting loss ----------------
    base = lr if mode == "none" else lr_joint
    opt = torch.optim.AdamW(model.parameters(), lr=base, weight_decay=wd, betas=(0.9, 0.98))
    for s in range(S2):
        for g in opt.param_groups:
            g["lr"] = cosine(s, S2, base)
        n = random.randint(*n_range)
        x = sample_inputs(batch, n, device=device)
        out, y = forward_sort(model, x)
        logits = out["logits"][:, n:, :]
        loss = F.cross_entropy(logits.reshape(-1, N_DIGITS), y.reshape(-1))
        acc = (logits.argmax(-1) == y).float().mean().item()
        rec = {"phase": 2, "step": S1 + s, "n": n, "ce": loss.item(), "acc": acc, "lr": cosine(s, S2, base), "time": time.time() - t0}
        if mode != "none":
            with torch.no_grad():
                rec["state_mse"] = F.mse_loss(out["state"][:, n:, :], state_targets(x, y, mode)).item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == S2 - 1:
            log.append(rec)
            if verbose:
                sm = f" | state mse={rec['state_mse']:.5f}" if "state_mse" in rec else ""
                print(f"[{mode:4s} P2] step {S1 + s:5d} n={n:2d} ce={loss.item():.4f} acc={acc:.4f}{sm} ({rec['time']:.0f}s)")
    model.eval()
    return model, log


def train_model_joint(mode, cfg: Config, steps=6000, batch=256, lr=1e-3, wd=0.01, n_range=(4, 10),
                      aux_weight=1.0, aux_off_frac=0.8, warmup=200, seed=0, log_every=250, device=DEVICE,
                      verbose=True):
    """The naive version of supervised-state training, used only in the ablation: no bottleneck,
    the sorting loss and an auxiliary *probe* loss on the residual stream are minimised jointly
    (the probe loss is switched off for the last (1 - aux_off_frac) of the steps)."""
    set_seed(seed)
    model = SortingTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    log, t0 = [], time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        n = random.randint(*n_range)
        x = sample_inputs(batch, n, device=device)
        w = aux_weight if s < aux_off_frac * steps else 0.0
        loss, stats = compute_losses(model, x, mode, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            stats.update({"step": s, "n": n, "aux_weight": w, "time": time.time() - t0})
            log.append(stats)
            if verbose:
                print(f"[{mode:4s} joint] step {s:5d} n={n:2d} ce={stats['ce']:.4f} acc={stats['acc']:.4f} aux={stats['aux']:.4f} aux_w={w:.1f} ({stats['time']:.0f}s)")
    model.eval()
    return model, log


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, x):
    """Greedy autoregressive sorting: returns predicted sorted digits (B, n)."""
    B, n = x.shape
    seq = torch.cat([x, torch.full((B, 1), SEP, dtype=torch.long, device=x.device)], 1)
    for t in range(n):
        logits = model(seq)["logits"][:, -1]
        seq = torch.cat([seq, (logits.argmax(-1) + OUT0)[:, None]], 1)
    return seq[:, n + 1:] - OUT0


@torch.no_grad()
def evaluate(model, n, B=1024, alphabet=N_DIGITS, seed=1234, device=DEVICE):
    """Free-running exact-match accuracy, free-running token accuracy, teacher-forced token
    accuracy and teacher-forced mean CE for inputs of length n."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = sample_inputs(B, n, alphabet=alphabet, device=device, generator=g)
    y = x.sort(1).values
    yhat = generate(model, x)
    exact = (yhat == y).all(1).float().mean().item()
    tok = (yhat == y).float().mean().item()
    out, _ = forward_sort(model, x)
    logits = out["logits"][:, n:, :]
    tf_acc = (logits.argmax(-1) == y).float().mean().item()
    ce = F.cross_entropy(logits.reshape(-1, N_DIGITS), y.reshape(-1)).item()
    return {"n": n, "exact": exact, "token": tok, "tf_token": tf_acc, "ce": ce}


def evaluate_lengths(model, ns, **kw):
    return [evaluate(model, n, **kw) for n in ns]


@torch.no_grad()
def train_loss_estimate(model, n_range=(4, 10), B=2048, seed=999, device=DEVICE):
    """Mean per-token CE and per-sequence NLL on the training distribution."""
    g = torch.Generator(device=device).manual_seed(seed)
    ces, nlls, accs = [], [], []
    for n in range(n_range[0], n_range[1] + 1):
        x = sample_inputs(B, n, device=device, generator=g)
        out, y = forward_sort(model, x)
        logits = out["logits"][:, n:, :]
        ce_tok = F.cross_entropy(logits.reshape(-1, N_DIGITS), y.reshape(-1), reduction="none").view(B, n)
        ces.append(ce_tok.mean().item()); nlls.append(ce_tok.sum(1).mean().item())
        accs.append((logits.argmax(-1) == y).all(1).float().mean().item())
    return {"ce_per_token": float(np.mean(ces)), "nll_per_sequence": float(np.mean(nlls)),
            "seq_acc_teacher_forced": float(np.mean(accs))}


@torch.no_grad()
def state_fidelity(model, n_range=(4, 10), B=1024, seed=321, device=DEVICE):
    """How well does the bottleneck state of a trained model match *each* algorithm's state?
    Returns per-slot errors for the MIN and HIST targets (lower = closer)."""
    g = torch.Generator(device=device).manual_seed(seed)
    res = {}
    for mode in ["min", "hist"]:
        errs, bit_acc, cnt_acc = [], [], []
        for n in range(n_range[0], n_range[1] + 1):
            x = sample_inputs(B, n, device=device, generator=g)
            out, y = forward_sort(model, x)
            s = out["state"][:, n:, :]
            tgt = state_targets(x, y, mode)
            errs.append(F.mse_loss(s, tgt).item())
            if mode == "min":
                bit_acc.append(((s[..., :10] > 0.5) == (tgt[..., :10] > 0.5)).float().mean().item())
            else:
                cnt_acc.append((s[..., :10].round() == tgt[..., :10]).float().mean().item())
        res[f"mse_vs_{mode.upper()}_state"] = float(np.mean(errs))
        if mode == "min":
            res["avail_bit_acc"] = float(np.mean(bit_acc))
        else:
            res["count_acc"] = float(np.mean(cnt_acc))
    return res


# ----------------------------------------------------------------------------------------
# Mechanistic test 1: does the model condition on the *previous outputs* (MIN) or only on the
# input histogram and the output index (HIST)?  We teacher-force a corrupted prefix: the last
# emitted token y_{t-1} is replaced by a different digit v'.  A min-tracking model follows the
# corrupted value (it emits the smallest available digit >= v'); a counting-sort model ignores
# the prefix and still emits the true y_t.
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def prefix_sensitivity(model, n=10, B=2048, seed=7, device=DEVICE):
    """Teacher-force a prefix whose last emitted digit y_{t-1} is replaced by v' and read the next
    prediction.  Reports the fraction of cases in which the model (a) emits what the MIN algorithm
    would emit given the corrupted prefix, (b) emits the true y_t (what counting sort emits, since
    it never looks at the prefix), (c) anything else.  v' is chosen so that the two answers differ
    and so that some digit >= v' is still available (no fall-back ambiguity); v' is always a digit that
    occurs in the input, so the corrupted prefix is a plausible one."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = sample_inputs(B, n, device=device, generator=g)
    y = x.sort(1).values
    t = torch.randint(1, n, (B,), device=device, generator=g)           # corrupt y_{t-1}, predict y_t
    true_next = y.gather(1, t[:, None]).squeeze(1)
    h = F.one_hot(x, N_DIGITS).sum(1)
    digits = torch.arange(N_DIGITS, device=device)
    # emitted counts of the uncorrupted prefix y_{<t-1}
    yo = F.one_hot(y, N_DIGITS)
    cum = torch.cumsum(yo, 1)                                            # cum[:, i] = counts of y[:, :i+1]
    before = torch.where((t - 2)[:, None] >= 0, cum[torch.arange(B), (t - 2).clamp(min=0)], torch.zeros_like(h))
    best_v, best_pred, found = torch.full((B,), -1, device=device), torch.full((B,), -1, device=device), torch.zeros(B, dtype=torch.bool, device=device)
    order = torch.randperm(N_DIGITS, generator=torch.Generator().manual_seed(seed)).to(device)
    for vp in order.tolist():                                            # random order over candidate v'
        vprime = torch.full((B,), vp, device=device)
        emitted = before + F.one_hot(vprime, N_DIGITS)
        avail = (h - emitted) > 0
        cand = avail & (digits[None] >= vp)
        ok = cand.any(1) & (vprime != y.gather(1, (t - 1)[:, None]).squeeze(1)) & (h[:, vp] > 0)
        pred = torch.where(cand, digits[None], torch.full_like(cand, 99, dtype=torch.long)).min(1).values
        ok = ok & (pred != true_next) & ~found
        best_v = torch.where(ok, vprime, best_v); best_pred = torch.where(ok, pred, best_pred); found |= ok
    keep = found
    x, y, t, true_next, best_v, best_pred = x[keep], y[keep], t[keep], true_next[keep], best_v[keep], best_pred[keep]
    ycor = y.clone(); ycor.scatter_(1, (t - 1)[:, None], best_v[:, None])
    toks = torch.cat([x, torch.full((x.shape[0], 1), SEP, dtype=torch.long, device=device), ycor + OUT0], 1)[:, :2 * n]
    pred = model(toks)["logits"][torch.arange(x.shape[0]), n + t].argmax(-1)
    m = x.shape[0]
    return {"follows_prefix(MIN-like)": (pred == best_pred).float().mean().item(),
            "ignores_prefix(HIST-like)": (pred == true_next).float().mean().item(),
            "other": ((pred != best_pred) & (pred != true_next)).float().mean().item(), "n_cases": m}


# ----------------------------------------------------------------------------------------
# Mechanistic test 2: post-hoc linear probes.  After training, freeze the model and fit
# *fresh* linear probes for both algorithm states on held-out data.  A state that the model
# actually computes should be linearly decodable from its residual stream.
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def collect_features(model, n_range=(4, 10), B=512, seed=11, which="resid", device=DEVICE):
    g = torch.Generator(device=device).manual_seed(seed)
    feats, H, T, A = [], [], [], []
    for n in range(n_range[0], n_range[1] + 1):
        x = sample_inputs(B, n, device=device, generator=g)
        out, y = forward_sort(model, x)
        r = out[which][:, n:, :]                                           # (B, n, d) or (B, n, K)
        h, t = counting_sort_state(x)
        a = availability(x, y).float()
        feats.append(r.reshape(-1, r.shape[-1]))
        H.append((h[:, None, :].expand(B, n, N_DIGITS)).reshape(-1, N_DIGITS))
        T.append(t[None, :].expand(B, n).reshape(-1))
        A.append(a.reshape(-1, N_DIGITS))
    return torch.cat(feats), torch.cat(H), torch.cat(T), torch.cat(A)


def ridge_fit(X, Y, lam=1e-2):
    """Closed-form ridge regression with bias; returns predictor function."""
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    d = Xb.shape[1]
    W = torch.linalg.solve(Xb.T @ Xb + lam * torch.eye(d, device=X.device), Xb.T @ Y)
    return lambda Z: torch.cat([Z, torch.ones(Z.shape[0], 1, device=Z.device)], 1) @ W


def r2(pred, Y):
    ss_res = ((pred - Y) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def logistic_fit(X, Y, steps=300, lr=0.05, wd=1e-4):
    """Multi-label linear logistic probe trained with Adam (full batch)."""
    lin = nn.Linear(X.shape[1], Y.shape[1]).to(X.device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(lin(X), Y).backward()
        opt.step()
    return lambda Z: lin(Z)


def probe_report(model, which="resid", device=DEVICE):
    """Fit probes on one seed's data, evaluate on another.  Returns R^2 for the histogram
    counts, R^2 for the index t, bit-accuracy for availability, and the accuracy of the
    *derived* sort prediction argmin{d: avail_d} from the availability probe."""
    Xtr, Htr, Ttr, Atr = collect_features(model, seed=11, which=which, device=device)
    Xte, Hte, Tte, Ate = collect_features(model, seed=12, which=which, device=device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    fh = ridge_fit(Xtr, Htr); ft = ridge_fit(Xtr, Ttr[:, None])
    with torch.enable_grad():
        fa = logistic_fit(Xtr, Atr)
    with torch.no_grad():
        res = {"hist_R2": r2(fh(Xte), Hte),
               "hist_count_acc": (fh(Xte).round() == Hte).float().mean().item(),
               "idx_R2": r2(ft(Xte), Tte[:, None]),
               "avail_bit_acc": ((fa(Xte) > 0) == (Ate > 0.5)).float().mean().item()}
    return res


# ----------------------------------------------------------------------------------------
# Local Learning Coefficient (Lau, Furman, Wang, Murfet, Wei 2023) via SGLD.
#   lambda_hat(w*) = n*beta * ( E_{w ~ tempered local posterior}[L_n(w)] - L_n(w*) )
#   SGLD:  dw = eps/2 * ( -n*beta * grad L_batch(w) + gamma * (w* - w) ) + N(0, eps)
# L_n is the mean *per-sequence* negative log-likelihood (sum over the n output tokens).
# ----------------------------------------------------------------------------------------
class FixedDataset:
    """A fixed i.i.d. sample of N sequences from the training distribution, grouped by n."""
    def __init__(self, N=16384, n_range=(4, 10), seed=2024, device=DEVICE):
        g = torch.Generator(device=device).manual_seed(seed)
        ns = list(range(n_range[0], n_range[1] + 1))
        per = N // len(ns)
        self.groups = {n: sample_inputs(per, n, device=device, generator=g) for n in ns}
        self.N = per * len(ns)
        self.device = device

    def sample(self, batch, generator=None):
        n = random.choice(list(self.groups))
        x = self.groups[n]
        idx = torch.randint(0, x.shape[0], (batch,), device=self.device, generator=generator)
        return x[idx]

    def full_loss(self, model, loss_fn, batch=2048):
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for n, x in self.groups.items():
                for i in range(0, x.shape[0], batch):
                    xb = x[i:i + batch]
                    tot += loss_fn(model, xb).item() * xb.shape[0]; cnt += xb.shape[0]
        return tot / cnt


def sequence_nll(model, x):
    """Mean over the batch of the per-sequence NLL (summed over output tokens)."""
    B, n = x.shape
    out, y = forward_sort(model, x)
    logits = out["logits"][:, n:, :]
    ce = F.cross_entropy(logits.reshape(-1, N_DIGITS), y.reshape(-1), reduction="none").view(B, n)
    return ce.sum(1).mean()


def estimate_llc(model, data: FixedDataset, eps=1e-4, gamma=100.0, nbeta=None, n_chains=4,
                 n_steps=1000, burnin=400, batch=256, seed=0, loss_fn=sequence_nll, verbose=False,
                 param_filter=None):
    """SGLD-based LLC estimate.  Returns dict with per-chain estimates, the loss traces and the
    reference loss L_n(w*).  All chains start at w* and only the sorting model's parameters
    (not the probe heads) are sampled.  `param_filter(name) -> bool` restricts the sampled
    parameters to a subset (the *weight-refined* LLC of Wang et al. 2024); all other
    parameters stay frozen at w*."""
    if nbeta is None:
        nbeta = data.N / math.log(data.N)            # beta* = 1/log n  (Lau et al. 2023)
    device = data.device
    L0 = data.full_loss(model, loss_fn)
    names = [n for n, _ in model.named_parameters() if not n.startswith("probe_")]
    if param_filter is not None:
        names = [n for n in names if param_filter(n)]
    w_star = [p.detach().clone() for n, p in model.named_parameters() if n in set(names)]
    traces = np.zeros((n_chains, n_steps))
    chains_llc = []
    for c in range(n_chains):
        g = torch.Generator(device=device).manual_seed(seed * 1000 + c)
        torch.manual_seed(seed * 1000 + c)
        m = copy.deepcopy(model)
        name_set = set(names)
        params = [p for n, p in m.named_parameters() if n in name_set]
        for p, w0 in zip(params, w_star):
            p.data.copy_(w0)
        for s in range(n_steps):
            x = data.sample(batch, generator=g)
            loss = loss_fn(m, x)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, gr, w0 in zip(params, grads, w_star):
                    drift = -nbeta * gr + gamma * (w0 - p)
                    p.add_(0.5 * eps * drift + math.sqrt(eps) * torch.randn_like(p))
            traces[c, s] = loss.item()
        chain_est = nbeta * (traces[c, burnin:].mean() - L0)
        chains_llc.append(chain_est)
        if verbose:
            print(f"  chain {c}: L0={L0:.4f} mean(L)={traces[c, burnin:].mean():.4f} llc={chain_est:.2f}")
    chains_llc = np.array(chains_llc)
    return {"llc": float(chains_llc.mean()), "llc_std": float(chains_llc.std()),
            "per_chain": chains_llc.tolist(), "traces": traces, "L0": L0, "nbeta": nbeta,
            "eps": eps, "gamma": gamma, "burnin": burnin, "n_params": int(sum(w.numel() for w in w_star))}


# ----------------------------------------------------------------------------------------
# Attention pattern helper (for visualisation)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def attention_patterns(model, x):
    """Returns list over layers of (B, heads, T, T) attention weights for tokens [x, SEP, y[:-1]]."""
    out, y = forward_sort(model, x, return_attn=True)
    return out["attn"], y


def token_labels(x_row):
    x_row = [int(v) for v in x_row]
    y_row = sorted(x_row)
    return [str(v) for v in x_row] + ["|"] + [f"{v}'" for v in y_row[:-1]]
