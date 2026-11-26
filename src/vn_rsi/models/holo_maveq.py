"""Holo-MAVE-Q (MAVE-Q v3): Holographic Neuro-Symbolic Alpha Evolution for Emerging Markets.

Author: Tran Duy Khang — AI Software Engineer / Freelance Vibe Researcher

Five Architectural Breakthroughs over MAVE-Q v1/v2 & Literature Baselines:
1. Exact Dimensional-Algebraic Type System (`TypedASTGuard`):
   Propagates physical exponent pairs (Price^p, Volume^v) through AST operators (where
   `turnover_vnd = P^1 V^1`), enforcing root scale-invariance `(0, 0)` and eliminating
   algebraic tautologies (`sub(x,x)`, `safe_div(x,x)`, double squashing) at zero backtest cost.
2. Topological Tree-Positional Masked AST Diffusion (`TopologicalTreeDiffusion`):
   Replaces flat 1D token diffusion with exact prefix-tree structural coordinates
   `(parent_op, arg_slot, depth)` + Confidence-Ordered Entropy Remasking + Attribution-Guided
   Weak-Subtree Infilling + Edge-Localized Shapley `diffu-GRPO` (dual PyTorch / vectorized NumPy).
3. Holographic Compositional Tree-Recursive Reasoner (`HoloHRM`):
   Bottom-up non-commutative syntax-tree encoder coupled with multi-timescale `(z_H, z_L)`
   recursion and a 3-head epistemic bootstrap surrogate predicting both robust IC and turnover
   for Upper-Confidence-Bound (`UCB`) latent Best-of-K pruning.
4. Multi-Scale Causal State-Space Filter Bank (`MSSF`):
   Strictly inside a hermetically isolated `search`-zone panel (`get_agent_visible_bars("search")`),
   analytically matches each candidate's causal temporal hysteresis horizon `ts_mean(., w*)` to
   Vietnam's T+2.5 / 3-day Top-6 boundary-turnover friction before spending a single evaluator call.
5. Regime-Invariant Spectral Risk-Parity & Shrinkage Subspace Synthesizer (`SRSS`):
   Closed-form minimax cross-fold IC-covariance + squared-turnover shrinkage solver over a 5x4
   Behavioral MAP-Elites archive, paired with Counterfactual Leave-One-Out Subspace Shapley Credit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.dsl.ast_spec import (
    ASTNode,
    ASTVerifier,
    FactorSpec,
    compute_feature_panel,
    evaluate_factor_spec,
)
from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator
from vn_rsi.eval.metrics import compute_rank_ic_series
from vn_rsi.models.alpha_mining_baselines import GrammarTreeSampler, validation_selection_score
from vn_rsi.models.ast_vocab import MECHANISM_FAMILIES, ASTVocab
from vn_rsi.models.cloud_presets import (
    amp_autocast_and_scaler,
    get_preset_spec,
    resolve_torch_device,
    xla_mark_step_if_needed,
)
from vn_rsi.sandbox.seal import SealedTestReuseError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# 1. Exact Physical-Exponent Dimensional Homogeneity & Algebraic Guard
# ---------------------------------------------------------------------------
# Physical exponents (price_exp, volume_exp); (0, 0) = dimensionless & time-stationary
_TERMINAL_EXPONENTS: dict[str, tuple[int, int]] = {
    "close": (1, 0),
    "open": (1, 0),
    "high": (1, 0),
    "low": (1, 0),
    "volume": (0, 1),
    "turnover_vnd": (1, 1),
}
_SCALE_INVARIANT_TS_OPS = frozenset({"ts_return", "ts_zscore", "ts_rank"})
_HOMOGENEOUS_UNARY_OPS = frozenset(
    {"ts_mean", "ts_std", "ts_delta", "ts_min", "ts_max", "lag", "neg", "abs_val"}
)
_STATIONARY_ONLY_UNARY_OPS = frozenset({"cs_rank", "cs_zscore", "tanh_squash"})
_UNIT_STANDARDIZED_OPS = frozenset({"cs_rank", "cs_zscore", "ts_rank", "ts_zscore", "tanh_squash"})
_TERMINAL_SCALE_CLASS: dict[str, str] = {
    "return_1d": "R",
    "intraday_range": "R",
    "close_to_open": "R",
    "dist_to_ceiling": "R",
    "dist_to_floor": "R",
    "foreign_net_ratio": "R",
    "pit_roe": "F",
    "pit_revenue_yoy": "F",
    "pit_value_quality": "F",
    "fol_room_pct": "F",
    "pit_pb_ratio": "M",
    "turnover_shock_20d": "M",
}


def _dimensionless_scale_class(node: ASTNode) -> str:
    """Classify magnitude scale of dimensionless expressions to prevent adding 1% returns to 3.0x P/B."""
    if node.node_type == "terminal":
        return _TERMINAL_SCALE_CLASS.get(node.name, "L")
    if node.name in _UNIT_STANDARDIZED_OPS:
        return "Z"
    if node.name == "ts_return":
        return "R"
    if len(node.args) == 1:
        return _dimensionless_scale_class(node.args[0])
    if node.name in ("add", "sub"):
        return _dimensionless_scale_class(node.args[0])
    return "U"

ECONOMIC_NICHES: tuple[str, ...] = ("fundamental", "flow", "momentum", "micro", "liquidity")
_NICHE_TERMINALS: dict[str, tuple[str, ...]] = {
    "fundamental": ("pit_roe", "pit_pb_ratio", "pit_revenue_yoy", "pit_value_quality"),
    "flow": ("foreign_net_ratio", "fol_room_pct"),
    "momentum": ("close", "open", "high", "low", "return_1d"),
    "micro": ("intraday_range", "close_to_open", "dist_to_ceiling", "dist_to_floor"),
    "liquidity": ("volume", "turnover_vnd", "turnover_shock_20d"),
}
_TERM_TO_NICHE = {t: k for k, vs in _NICHE_TERMINALS.items() for t in vs}
TURNOVER_BUCKETS: tuple[float, ...] = (0.03, 0.07, 0.13)


def exact_dimensional_exponents(node: ASTNode) -> tuple[int, int] | None:
    """Return physical (price_exp, vol_exp) of `node`, or None if dimensionally/algebraically invalid."""
    if node.node_type == "terminal":
        return _TERMINAL_EXPONENTS.get(node.name, (0, 0))
    child_dims = [exact_dimensional_exponents(ch) for ch in node.args]
    if any(d is None for d in child_dims):
        return None
    name = node.name
    if len(child_dims) == 1:
        d0 = child_dims[0]
        assert d0 is not None
        # Reject redundant double idempotent/canceling wrappers & short-window TS diff/std on quarterly PIT step-series
        ch0 = node.args[0]
        if name == "neg" and ch0.name == "neg":
            return None
        if name in ("cs_rank", "cs_zscore") and ch0.name in ("cs_rank", "cs_zscore"):
            return None
        if name == "tanh_squash" and ch0.name == "tanh_squash":
            return None
        if (
            ch0.node_type == "terminal"
            and ch0.name in _NICHE_TERMINALS["fundamental"]
            and name in ("ts_std", "ts_delta", "ts_return", "ts_zscore", "ts_rank")
        ):
            return None
        if name in _SCALE_INVARIANT_TS_OPS:
            return (0, 0)
        if name in _STATIONARY_ONLY_UNARY_OPS:
            return (0, 0) if d0 == (0, 0) else None
        if name in _HOMOGENEOUS_UNARY_OPS:
            return d0
        return None

    da, db = child_dims[0], child_dims[1]
    assert da is not None and db is not None
    # Reject degenerate x - x, x + x, x / x
    if name in ("add", "sub", "safe_div") and node.args[0].canonical_expr() == node.args[1].canonical_expr():
        return None
    if name in ("add", "sub"):
        if da != db:
            return None
        if da == (0, 0):
            sa = _dimensionless_scale_class(node.args[0])
            sb = _dimensionless_scale_class(node.args[1])
            if sa != sb and "U" not in (sa, sb):
                return None
        return da
    if name == "mul":
        return (da[0] + db[0], da[1] + db[1])
    if name == "safe_div":
        return (da[0] - db[0], da[1] - db[1])
    return None


def is_typed_stationary_ast(node: ASTNode) -> bool:
    """True iff `node` is algebraically non-degenerate and dimensionally scale-free `(0, 0)`."""
    return exact_dimensional_exponents(node) == (0, 0)


def dominant_economic_niche(node: ASTNode) -> str:
    counts = {k: 0 for k in ECONOMIC_NICHES}

    def _visit(n: ASTNode) -> None:
        if n.node_type == "terminal":
            counts[_TERM_TO_NICHE.get(n.name, "momentum")] += 1
        for ch in n.args:
            _visit(ch)

    _visit(node)
    # Prioritize non-price structural information when tied (e.g., flow * return_1d -> flow)
    priority = ("fundamental", "flow", "micro", "liquidity", "momentum")
    return max(priority, key=lambda g: (counts[g], -priority.index(g)))


def horizon_bucket(turnover: float) -> int:
    for idx, edge in enumerate(TURNOVER_BUCKETS):
        if turnover < edge:
            return idx
    return len(TURNOVER_BUCKETS)


# ---------------------------------------------------------------------------
# 2. Prefix-Tree Structural Coordinates & Topological Tree Diffusion
# ---------------------------------------------------------------------------
def compute_prefix_tree_coords(
    vocab: ASTVocab, token_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (parent_token_id, arg_slot[0=root,1=arg0,2=arg1], depth[1..6]) for every slot in seq."""
    seq_len = len(token_ids)
    parents = np.full(seq_len, vocab.bos_id, dtype=np.int64)
    slots = np.zeros(seq_len, dtype=np.int64)
    depths = np.ones(seq_len, dtype=np.int64)

    # Stack items: (parent_tok_id, next_slot_idx, total_arity, child_depth)
    stack: list[tuple[int, int, int, int]] = [(vocab.bos_id, 0, 1, 1)]
    for pos in range(1, seq_len):
        if not stack:
            parents[pos] = vocab.eos_id
            slots[pos] = 0
            depths[pos] = 1
            continue
        p_tok, s_idx, arity, d = stack.pop()
        parents[pos] = p_tok
        slots[pos] = 0 if p_tok == vocab.bos_id else (1 if s_idx == 0 else 2)
        depths[pos] = min(6, d)
        if s_idx + 1 < arity:
            stack.append((p_tok, s_idx + 1, arity, d))
        tid = token_ids[pos]
        if tid not in (vocab.bos_id, vocab.eos_id, vocab.pad_id, vocab.mask_id):
            spec = vocab.id_to_spec[tid]
            if spec.arity > 0:
                stack.append((tid, 0, spec.arity, d + 1))
    return parents, slots, depths


_NICHE_PRIORS_TERMS: dict[str, tuple[str, ...]] = {
    "momentum_liquidity": ("close", "return_1d", "turnover_shock_20d", "high"),
    "reversal_microstructure": ("close", "return_1d", "turnover_shock_20d", "dist_to_ceiling", "close_to_open", "intraday_range"),
    "value_quality_pit": ("pit_value_quality", "pit_roe", "pit_pb_ratio", "pit_revenue_yoy"),
    "foreign_flow_regime": ("foreign_net_ratio", "fol_room_pct", "pit_value_quality", "turnover_shock_20d"),
}

_NICHE_PRIORS_OPS: dict[str, tuple[str, ...]] = {
    "momentum_liquidity": ("OP:safe_div", "OP:ts_return:15", "OP:ts_return:20", "OP:ts_std:20", "OP:ts_mean:5"),
    "reversal_microstructure": ("OP:neg", "OP:mul", "OP:sub", "OP:ts_return:3", "OP:ts_return:5", "OP:ts_zscore:10"),
    "value_quality_pit": ("OP:add", "OP:sub", "OP:cs_zscore", "OP:cs_rank", "OP:ts_mean:10"),
    "foreign_flow_regime": ("OP:ts_mean:5", "OP:ts_mean:10", "OP:add", "OP:mul", "OP:cs_zscore"),
}


if HAS_TORCH:

    class _TopologicalTreeNet(nn.Module):
        """Tree-coordinate conditioned Masked AST Transformer."""

        def __init__(
            self,
            vocab_size: int,
            n_fams: int,
            d_model: int,
            n_layers: int,
            n_heads: int,
            d_ff: int | None = None,
        ) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.parent_emb = nn.Embedding(vocab_size, d_model)
            self.slot_emb = nn.Embedding(3, d_model)
            self.depth_emb = nn.Embedding(8, d_model)
            self.pos_emb = nn.Embedding(16, d_model)
            self.fam_emb = nn.Embedding(n_fams, d_model)
            self.mr_proj = nn.Linear(1, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff or (d_model * 4),
                dropout=0.05,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
            # DiT / ControlNet-style zero-init readout: starts cleanly at grammar+edge prior at round 1
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(
            self,
            toks: torch.Tensor,
            parents: torch.Tensor,
            slots: torch.Tensor,
            depths: torch.Tensor,
            fams: torch.Tensor,
            mr: torch.Tensor,
        ) -> torch.Tensor:
            bsz, seq_len = toks.shape
            pos = torch.arange(seq_len, device=toks.device).unsqueeze(0).expand(bsz, seq_len)
            h = (
                self.tok_emb(toks)
                + self.parent_emb(parents)
                + self.slot_emb(slots)
                + self.depth_emb(depths.clamp(0, 7))
                + self.pos_emb(pos)
                + self.fam_emb(fams).unsqueeze(1)
                + self.mr_proj(mr.unsqueeze(-1)).unsqueeze(1)
            )
            return self.head(self.ln(self.enc(h)))


class TopologicalTreeDiffusion:
    """Topology-Aware Masked AST Diffusion (`Tree-MDLM`) with Subtree-Edge Shapley `diffu-GRPO`."""

    def __init__(self, preset: str = "cpu_fast", seed: int = 2026, device: str | None = None) -> None:
        self.preset_spec = get_preset_spec(preset)
        self.preset = self.preset_spec.name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vocab = ASTVocab(max_seq_len=16)
        v = self.vocab.vocab_size
        n_f = len(MECHANISM_FAMILIES)
        self.denoise_steps = self.preset_spec.diff_denoise_steps
        self.kl_beta = 0.04
        self.use_amp = self.preset_spec.use_amp
        self.is_xla_tpu = False
        self.prior_bias = self._init_domain_priors()

        # Structural parent->(slot, child) bilinear grammar tensor (active in BOTH PyTorch & NumPy!)
        # Captures asymmetric idioms such as safe_div(ts_return, ts_std) vs safe_div(ts_std, ts_return)
        self.edge_logits = self._init_structural_edge_priors()

        d_model = self.preset_spec.diff_d_model
        n_layers = self.preset_spec.diff_n_layers
        n_heads = self.preset_spec.diff_n_heads
        d_ff = self.preset_spec.diff_d_ff

        if HAS_TORCH:
            self.device, self.is_xla_tpu = resolve_torch_device(device)
            torch.manual_seed(seed)
            self.net: Any = _TopologicalTreeNet(v, n_f, d_model, n_layers, n_heads, d_ff=d_ff).to(self.device)
            self.opt: Any = torch.optim.AdamW(self.net.parameters(), lr=self.preset_spec.diff_lr, weight_decay=1e-2)
            self.backend_runtime = (
                f"pytorch_xla_tree_mdlm:{self.device}" if self.is_xla_tpu else f"pytorch_tree_mdlm:{self.device}"
            )
            self.np_params: dict[str, np.ndarray] = {}
        else:
            self.device = "cpu"
            self.net = None
            self.opt = None
            # Genuine 2-layer Residual Bilinear Tree-Context Neural Core in vectorized NumPy
            np_d = min(d_model, 128)
            scale = 0.06
            self.np_params = {
                "tok_emb": self.rng.normal(0.0, scale, size=(v, np_d)),
                "parent_emb": self.rng.normal(0.0, scale, size=(v, np_d)),
                "slot_depth_emb": self.rng.normal(0.0, scale, size=(3 * 8, np_d)),
                "fam_pos_emb": self.rng.normal(0.0, scale, size=(n_f * 16, np_d)),
                "w1": self.rng.normal(0.0, scale, size=(np_d, np_d * 2)),
                "b1": np.zeros(np_d * 2, dtype=float),
                # DiT / LoRA-style zero-initialized readout: zero noise before GRPO updates!
                "w2": np.zeros((np_d * 2, v), dtype=float),
                "b2": np.zeros(v, dtype=float),
            }
            self.backend_runtime = "numpy_bilinear_tree_mdlm:cpu"

    def _init_domain_priors(self) -> np.ndarray:
        bias = np.zeros((len(MECHANISM_FAMILIES), self.vocab.vocab_size), dtype=float)
        for fi, fam in enumerate(MECHANISM_FAMILIES):
            for t in _NICHE_PRIORS_TERMS.get(fam, ()):
                tid = self.vocab.token_to_id.get(f"TERM:{t}")
                if tid is not None:
                    bias[fi, tid] += 1.4
            for op in _NICHE_PRIORS_OPS.get(fam, ()):
                oid = self.vocab.token_to_id.get(op)
                if oid is not None:
                    bias[fi, oid] += 1.3
        return bias

    def _init_structural_edge_priors(self) -> np.ndarray:
        """Weakly initialize syntactically stationary parent->(slot, child) couplings."""
        v = self.vocab.vocab_size
        edges = np.zeros((len(MECHANISM_FAMILIES), 3, v, v), dtype=float)
        t2i = self.vocab.token_to_id
        sdiv = t2i.get("OP:safe_div")
        mul_id = t2i.get("OP:mul")
        neg_id = t2i.get("OP:neg")
        if sdiv is not None:
            # Encourage dimensionless return/delta numerators (slot=1) and volatility denominators (slot=2)
            for w in (10, 15, 20):
                num_id = t2i.get(f"OP:ts_return:{w}")
                if num_id is not None:
                    edges[:, 1, sdiv, num_id] += 1.1
            for w in (10, 15, 20):
                den_id = t2i.get(f"OP:ts_std:{w}")
                if den_id is not None:
                    edges[:, 2, sdiv, den_id] += 1.2
        if mul_id is not None and neg_id is not None:
            edges[1, 1, neg_id, mul_id] += 1.0
        return edges

    def parameter_count(self) -> int:
        edge_n = int(self.edge_logits.size)
        if HAS_TORCH and self.net is not None:
            return edge_n + int(sum(p.numel() for p in self.net.parameters()))
        return edge_n + sum(int(arr.size) for arr in self.np_params.values())

    def estimate_training_vram_mb(self) -> float:
        bytes_per_param = 10.0 if self.use_amp else 16.0
        return round((self.parameter_count() * bytes_per_param) / (1024.0**2) + 48.0, 2)

    def save_checkpoint(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if HAS_TORCH and self.net is not None:
            torch.save(
                {
                    "preset": self.preset,
                    "state_dict": self.net.state_dict(),
                    "edge_logits": self.edge_logits,
                },
                out,
            )
            return out
        npz_path = out.with_suffix(".npz")
        payload: dict[str, Any] = {"edge_logits": self.edge_logits, **self.np_params}
        np.savez(npz_path, **payload)
        return npz_path

    def load_checkpoint(self, path: str | Path) -> None:
        src = Path(path)
        if HAS_TORCH and self.net is not None and src.exists():
            ckpt = torch.load(src, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ckpt["state_dict"])
            if "edge_logits" in ckpt:
                self.edge_logits = np.asarray(ckpt["edge_logits"], dtype=float)
        elif src.with_suffix(".npz").exists():
            data = np.load(src.with_suffix(".npz"))
            self.edge_logits = np.asarray(data["edge_logits"], dtype=float)
            for k in list(self.np_params.keys()):
                if k in data:
                    self.np_params[k] = np.asarray(data[k], dtype=float)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * (x**3))))

    def _np_forward_seq(
        self, seq: list[int], parents: np.ndarray, slots: np.ndarray, depths: np.ndarray, fam_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.np_params
        seq_arr = np.asarray(seq, dtype=np.int64)
        sd_idx = slots * 8 + np.clip(depths, 0, 7)
        fp_idx = fam_id * 16 + np.arange(len(seq), dtype=np.int64)
        ctx_mean = p["tok_emb"][seq_arr].mean(axis=0, keepdims=True)
        h0 = (
            p["tok_emb"][seq_arr]
            + p["parent_emb"][parents]
            + p["slot_depth_emb"][sd_idx]
            + p["fam_pos_emb"][fp_idx]
            + 0.35 * ctx_mean
        )
        mu = h0.mean(axis=-1, keepdims=True)
        sd = h0.std(axis=-1, keepdims=True) + 1e-6
        h_norm = (h0 - mu) / sd
        hid = self._gelu(h_norm @ p["w1"] + p["b1"])
        logits = hid @ p["w2"] + p["b2"]
        return logits, h_norm, hid

    def _predict_logits(self, seq: list[int], fam_id: int, mask_ratio: float) -> np.ndarray:
        parents, slots, depths = compute_prefix_tree_coords(self.vocab, seq)
        if HAS_TORCH and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                x = torch.tensor([seq], dtype=torch.long, device=self.device)
                pt = torch.tensor([parents], dtype=torch.long, device=self.device)
                st = torch.tensor([slots], dtype=torch.long, device=self.device)
                dt = torch.tensor([depths], dtype=torch.long, device=self.device)
                ft = torch.tensor([fam_id], dtype=torch.long, device=self.device)
                mr = torch.tensor([mask_ratio], dtype=torch.float32, device=self.device)
                base = self.net(x, pt, st, dt, ft, mr)[0].detach().cpu().numpy()
        else:
            base, _, _ = self._np_forward_seq(seq, parents, slots, depths, fam_id)

        edge_term = self.edge_logits[fam_id, slots, parents]
        return base + edge_term + self.prior_bias[fam_id][None, :]

    def _stationary_aware_mask(
        self, prefix: list[int], base_valid: np.ndarray, parent_tok: int
    ) -> np.ndarray:
        """Prevent emitting raw price/volume levels directly under cs_zscore / cs_rank / tanh_squash."""
        mask = base_valid.copy()
        p_spec = self.vocab.id_to_spec.get(parent_tok)
        if p_spec is not None and p_spec.op_or_term in _STATIONARY_ONLY_UNARY_OPS:
            for raw_t in _TERMINAL_EXPONENTS:
                tid = self.vocab.token_to_id.get(f"TERM:{raw_t}")
                if tid is not None:
                    mask[tid] = False
        if not mask.any():
            return base_valid
        return mask

    def _decode_confidence_remasking(
        self,
        init_seq: list[int],
        fam_id: int,
        max_depth: int,
        temperature: float,
        rng: np.random.Generator,
        use_tree_coords: bool = True,
    ) -> list[int]:
        """Confidence-guided iterative unmasking over valid prefix frontier."""
        seq = list(init_seq)
        max_body = min(self.vocab.max_seq_len - 2, 10)
        steps = max(2, self.denoise_steps)

        for step in range(steps):
            masked_pos = [i for i in range(1, self.vocab.max_seq_len - 1) if seq[i] == self.vocab.mask_id]
            if not masked_pos:
                break
            mask_ratio = len(masked_pos) / max(1, self.vocab.max_seq_len - 2)
            logits = (
                self._predict_logits(seq, fam_id, mask_ratio)
                if use_tree_coords
                else (self.prior_bias[fam_id][None, :] * np.ones((self.vocab.max_seq_len, 1)))
            )
            chunk = max(1, int(math.ceil(len(masked_pos) / max(1, steps - step))))
            # Unmask contiguous prefix frontier while adapting step temperature by posterior confidence
            for pos in masked_pos[:chunk]:
                parents, _, _ = compute_prefix_tree_coords(self.vocab, seq)
                vm = self.vocab.valid_next_token_mask(seq[:pos], max_depth=max_depth, max_body_tokens=max_body)
                if use_tree_coords:
                    vm = self._stationary_aware_mask(seq[:pos], vm, int(parents[pos]))
                raw_l = np.where(vm, logits[pos], -1e9)
                # Confidence-adaptive temperature: high confidence sharpens, uncertain slots explore
                p0 = np.exp(raw_l - float(np.max(raw_l)))
                p0 /= float(np.sum(p0)) + 1e-12
                conf = float(np.max(p0))
                eff_temp = max(0.18, float(temperature) * (1.15 - 0.45 * conf))
                scaled = np.where(vm, logits[pos] / eff_temp, -1e9)
                probs = np.exp(scaled - float(np.max(scaled)))
                probs /= float(np.sum(probs)) + 1e-12
                seq[pos] = int(rng.choice(self.vocab.vocab_size, p=probs))

        for pos in range(1, self.vocab.max_seq_len):
            if seq[pos] == self.vocab.mask_id:
                vm = self.vocab.valid_next_token_mask(seq[:pos], max_depth=max_depth, max_body_tokens=max_body)
                ids = np.where(vm)[0]
                seq[pos] = int(ids[0]) if len(ids) > 0 else self.vocab.pad_id
        return seq

    def sample_factor(
        self,
        family: str,
        factor_id: str,
        hypothesis_id: str,
        max_depth: int = 4,
        temperature: float = 0.65,
        seed: int | None = None,
        use_tree_coords: bool = True,
    ) -> FactorSpec:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        fam_id = self.vocab.family_to_id(family)
        init_seq = [self.vocab.bos_id] + [self.vocab.mask_id] * (self.vocab.max_seq_len - 2) + [self.vocab.eos_id]
        seq = self._decode_confidence_remasking(
            init_seq, fam_id, max_depth=max_depth, temperature=temperature, rng=rng, use_tree_coords=use_tree_coords
        )
        return self.vocab.to_factor_spec(seq, factor_id=factor_id, hypothesis_id=hypothesis_id, family=family)

    def attribution_guided_infill(
        self,
        parent_spec: FactorSpec,
        weak_start_pos: int | None,
        family: str,
        factor_id: str,
        hypothesis_id: str,
        max_depth: int = 4,
        temperature: float = 0.55,
        seed: int | None = None,
        use_tree_coords: bool = True,
    ) -> FactorSpec:
        """Mask the Holo-HRM-identified weakest subtree span and conditionally denoise in place."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        fam_id = self.vocab.family_to_id(family)
        seq = self.vocab.encode_ast(parent_spec.root)
        active = [
            i for i in range(1, len(seq)) if seq[i] not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
        ]
        if len(active) <= 1:
            return self.sample_factor(
                family, factor_id, hypothesis_id, max_depth=max_depth, temperature=temperature, seed=seed,
                use_tree_coords=use_tree_coords,
            )
        if weak_start_pos is not None and weak_start_pos in active and weak_start_pos > 1:
            start = weak_start_pos
        else:
            choices = active[1:] if len(active) > 1 else active
            start = int(rng.choice(choices))
        s0, s1 = self.vocab.subtree_span(seq, start)
        masked = list(seq)
        for pos in range(s0, min(self.vocab.max_seq_len - 1, max(s1, s0 + 2))):
            masked[pos] = self.vocab.mask_id
        out_seq = self._decode_confidence_remasking(
            masked, fam_id, max_depth=max_depth, temperature=temperature, rng=rng, use_tree_coords=use_tree_coords
        )
        return self.vocab.to_factor_spec(out_seq, factor_id=factor_id, hypothesis_id=hypothesis_id, family=family)

    def bridge_fuse_elites(
        self,
        left_root: ASTNode,
        right_root: ASTNode,
        family: str,
        factor_id: str,
        hypothesis_id: str,
        temperature: float = 0.50,
        seed: int | None = None,
        use_tree_coords: bool = True,
    ) -> FactorSpec:
        """Masked Operator Bridge Diffusion (`Bridge-MDLM`): denoise optimal binary operator fusing 2 elites."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        fam_id = self.vocab.family_to_id(family)
        # Trim deep elites to compact depth<=3 subtrees & standardize if scale classes differ
        l_sub = left_root.args[0] if left_root.depth() >= 4 and left_root.args else left_root
        r_sub = right_root.args[0] if right_root.depth() >= 4 and right_root.args else right_root
        if (
            _dimensionless_scale_class(l_sub) != _dimensionless_scale_class(r_sub)
            and l_sub.depth() <= 2
            and r_sub.depth() <= 2
        ):
            l_sub = ASTNode(node_type="op", name="cs_zscore", args=[l_sub])
            r_sub = ASTNode(node_type="op", name="cs_zscore", args=[r_sub])
        l_toks = [
            t for t in self.vocab.encode_ast(l_sub)
            if t not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
        ]
        r_toks = [
            t for t in self.vocab.encode_ast(r_sub)
            if t not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
        ]
        if len(l_toks) + len(r_toks) + 3 > self.vocab.max_seq_len:
            return self.sample_factor(
                family, factor_id, hypothesis_id, max_depth=4, temperature=temperature, seed=seed,
                use_tree_coords=use_tree_coords,
            )
        seq = [self.vocab.bos_id, self.vocab.mask_id] + l_toks + r_toks + [self.vocab.eos_id]
        while len(seq) < self.vocab.max_seq_len:
            seq.append(self.vocab.pad_id)
        logits = (
            self._predict_logits(seq, fam_id, mask_ratio=0.15)[1]
            if use_tree_coords
            else self.prior_bias[fam_id]
        )
        bin_mask = np.zeros(self.vocab.vocab_size, dtype=bool)
        bin_mask[self.vocab.binary_op_ids] = True
        scaled = np.where(bin_mask, logits / max(0.20, temperature), -1e9)
        probs = np.exp(scaled - float(np.max(scaled)))
        probs /= float(np.sum(probs)) + 1e-12
        seq[1] = int(rng.choice(self.vocab.vocab_size, p=probs))
        return self.vocab.to_factor_spec(seq, factor_id=factor_id, hypothesis_id=hypothesis_id, family=family)

    def fit_shapley_diffu_grpo(
        self,
        rollouts: list[dict[str, Any]],
        epochs: int = 10,
    ) -> dict[str, float]:
        """Edge-Localized Subtree Shapley `diffu-GRPO`: updates both Transformer/MLP & structural edge tensor."""
        if len(rollouts) < 2:
            return {"loss": 0.0, "mean_reward": 0.0}

        items: list[tuple[list[int], int, float, np.ndarray, np.ndarray, np.ndarray]] = []
        for r in rollouts:
            try:
                sp = FactorSpec.model_validate(r["factor_spec"])
                seq = self.vocab.encode_ast(sp.root)
                fid = self.vocab.family_to_id(str(r.get("family", "value_quality_pit")))
                rew = float(r["shapley_reward"])
                parents, slots, depths = compute_prefix_tree_coords(self.vocab, seq)
                items.append((seq, fid, rew, parents, slots, depths))
            except Exception:
                continue

        if len(items) < 2:
            return {"loss": 0.0, "mean_reward": 0.0}

        rews = np.asarray([x[2] for x in items], dtype=float)
        advs = np.clip((rews - float(rews.mean())) / (float(rews.std()) + 1e-6), -2.5, 2.5)

        # Compute empirical edge value table across batch for localized credit attribution!
        edge_sum: dict[tuple[int, int, int], float] = {}
        edge_cnt: dict[tuple[int, int, int], int] = {}
        for (seq, _, _, parents, slots, _), adv in zip(items, advs, strict=True):
            for pos in range(1, len(seq) - 1):
                tid = seq[pos]
                if tid in (self.vocab.pad_id, self.vocab.eos_id):
                    break
                ek = (int(slots[pos]), int(parents[pos]), int(tid))
                edge_sum[ek] = edge_sum.get(ek, 0.0) + float(adv)
                edge_cnt[ek] = edge_cnt.get(ek, 0) + 1

        # Update structural edge tensor with localized 60% tree + 40% edge-counterfactual advantage
        lr_edge = 0.16
        for _ in range(max(2, epochs // 2)):
            self.edge_logits *= 1.0 - 0.015 * self.kl_beta
            for (seq, fid, _, parents, slots, _), adv in zip(items, advs, strict=True):
                for pos in range(1, len(seq) - 1):
                    tid = seq[pos]
                    if tid in (self.vocab.pad_id, self.vocab.eos_id):
                        break
                    sl, pr = int(slots[pos]), int(parents[pos])
                    ek = (sl, pr, int(tid))
                    local_adv = 0.60 * float(adv) + 0.40 * (edge_sum[ek] / max(1, edge_cnt[ek]))
                    row = self.edge_logits[fid, sl, pr] + self.prior_bias[fid]
                    probs = np.exp(row - float(np.max(row)))
                    probs /= float(np.sum(probs)) + 1e-12
                    grad = -probs
                    grad[tid] += 1.0
                    self.edge_logits[fid, sl, pr] += lr_edge * local_adv * grad / len(items)

        if HAS_TORCH and self.net is not None and self.opt is not None:
            self.net.train()
            x_clean = torch.tensor([x[0] for x in items], dtype=torch.long, device=self.device)
            p_t = torch.tensor(np.stack([x[3] for x in items]), dtype=torch.long, device=self.device)
            s_t = torch.tensor(np.stack([x[4] for x in items]), dtype=torch.long, device=self.device)
            d_t = torch.tensor(np.stack([x[5] for x in items]), dtype=torch.long, device=self.device)
            f_t = torch.tensor([x[1] for x in items], dtype=torch.long, device=self.device)
            adv_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                mr0 = torch.full((len(items),), 0.35, dtype=torch.float32, device=self.device)
                ref_lp = F.log_softmax(self.net(x_clean, p_t, s_t, d_t, f_t, mr0), dim=-1)
            amp_ctx, scaler = amp_autocast_and_scaler(self.device, self.use_amp, self.is_xla_tpu)
            last_loss = 0.0
            for _ in range(epochs):
                ratio = float(self.rng.uniform(0.20, 0.65))
                rmask = (torch.rand(x_clean.shape, device=self.device) < ratio) & (x_clean != self.vocab.pad_id)
                rmask[:, 0] = False
                if not rmask.any():
                    rmask[:, 1] = True
                x_noisy = x_clean.clone()
                x_noisy[rmask] = self.vocab.mask_id
                mr = torch.full((len(items),), ratio, dtype=torch.float32, device=self.device)
                with amp_ctx:
                    lp = F.log_softmax(self.net(x_noisy, p_t, s_t, d_t, f_t, mr).float(), dim=-1)
                    nll = F.nll_loss(
                        lp.view(-1, self.vocab.vocab_size), x_clean.view(-1), reduction="none"
                    ).view(x_clean.shape)
                    seq_nll = (nll * rmask.float()).sum(dim=1) / rmask.float().sum(dim=1).clamp(min=1.0)
                    loss = (adv_t * seq_nll).mean() + self.kl_beta * F.kl_div(
                        lp, ref_lp.float().exp(), reduction="batchmean"
                    )
                self.opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.opt.step()
                xla_mark_step_if_needed(self.is_xla_tpu)
                last_loss = float(loss.item())
            return {"loss": round(last_loss, 5), "mean_reward": round(float(rews.mean()), 5)}

        # Vectorized NumPy output-layer + bilinear readout diffu-GRPO update
        p = self.np_params
        lr_np = 0.06
        for _ in range(max(3, epochs // 2)):
            dw2 = np.zeros_like(p["w2"])
            db2 = np.zeros_like(p["b2"])
            for (seq, fid, _, parents, slots, depths), adv in zip(items, advs, strict=True):
                masked = list(seq)
                for pos in range(1, len(seq) - 1):
                    if seq[pos] != self.vocab.pad_id and self.rng.random() < 0.40:
                        masked[pos] = self.vocab.mask_id
                logits, _, hid = self._np_forward_seq(masked, parents, slots, depths, fid)
                for pos in range(1, len(seq) - 1):
                    tid = seq[pos]
                    if tid in (self.vocab.pad_id, self.vocab.eos_id):
                        break
                    row = logits[pos] + self.edge_logits[fid, slots[pos], parents[pos]] + self.prior_bias[fid]
                    probs = np.exp(row - float(np.max(row)))
                    probs /= float(np.sum(probs)) + 1e-12
                    g_out = probs.copy()
                    g_out[tid] -= 1.0  # d(NLL)/d(logits) = probs - one_hot
                    # Minimize adv * NLL -> step in direction -adv * g_out
                    dw2 += (-float(adv)) * np.outer(hid[pos], g_out) / len(items)
                    db2 += (-float(adv)) * g_out / len(items)
            p["w2"] = (1.0 - 0.004) * p["w2"] + lr_np * np.clip(dw2, -1.0, 1.0)
            p["b2"] = (1.0 - 0.004) * p["b2"] + lr_np * np.clip(db2, -1.0, 1.0)

        return {"loss": round(float(-np.mean(advs * rews)), 5), "mean_reward": round(float(rews.mean()), 5)}


# ---------------------------------------------------------------------------
# 3. Holographic Compositional Tree-Recursive Reasoner (`HoloHRM`)
# ---------------------------------------------------------------------------
if HAS_TORCH:

    class _HoloResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(F.gelu(self.fc1(self.ln(x))))

    class _PyTorchHoloTreeNet(nn.Module):
        """GPU/TPU-accelerated Compositional Syntax-Tree Encoder + Multi-Head Epistemic Surrogate."""

        def __init__(self, vocab_size: int, d_hidden: int, n_layers: int, n_heads_ens: int = 3) -> None:
            super().__init__()
            self.d_hidden = d_hidden
            self.tok_emb = nn.Embedding(vocab_size, d_hidden)
            self.parent_emb = nn.Embedding(vocab_size, d_hidden)
            self.slot_emb = nn.Embedding(3, d_hidden)
            self.depth_emb = nn.Embedding(8, d_hidden)
            self.regime_enc = nn.Sequential(
                nn.Linear(14, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_hidden),
            )
            self.low_cell = nn.GRUCell(d_hidden * 2, d_hidden)
            self.low_blocks = nn.ModuleList([_HoloResidualBlock(d_hidden) for _ in range(n_layers)])
            self.high_cell = nn.GRUCell(d_hidden * 2, d_hidden)
            self.high_blocks = nn.ModuleList([_HoloResidualBlock(d_hidden) for _ in range(n_layers)])
            self.ens_heads = nn.ModuleList([nn.Linear(d_hidden, 2) for _ in range(n_heads_ens)])
            for h in self.ens_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

        def forward(
            self,
            toks: torch.Tensor,
            parents: torch.Tensor,
            slots: torch.Tensor,
            depths: torch.Tensor,
            regime_vec: torch.Tensor,
        ) -> torch.Tensor:
            # Non-commutative slot-signed pooling preserves safe_div(A, B) != safe_div(B, A)
            slot_sign = torch.where(slots == 2, -1.0, 1.0).unsqueeze(-1)
            node_h = (
                self.tok_emb(toks)
                + self.parent_emb(parents)
                + self.slot_emb(slots)
                + self.depth_emb(depths.clamp(0, 7))
            )
            x_tree = (node_h * slot_sign).mean(dim=1)
            x_reg = self.regime_enc(regime_vec)
            z_h = x_reg
            z_l = x_tree
            dev_type = toks.device.type if toks.device.type in ("cuda", "xla", "cpu") else "cpu"
            cell_dtype = self.low_cell.weight_ih.dtype
            for _ in range(2):
                for _ in range(2):
                    with torch.amp.autocast(device_type=dev_type, enabled=False):
                        z_l = self.low_cell(
                            torch.cat([z_h.to(cell_dtype), x_tree.to(cell_dtype)], dim=-1),
                            z_l.to(cell_dtype),
                        )
                    for blk in self.low_blocks:
                        z_l = blk(z_l)
                with torch.amp.autocast(device_type=dev_type, enabled=False):
                    z_h = self.high_cell(
                        torch.cat([z_l.to(cell_dtype), x_reg.to(cell_dtype)], dim=-1),
                        z_h.to(cell_dtype),
                    )
                for blk in self.high_blocks:
                    z_h = blk(z_h)
            # Stack (B, H, 2) predictions for (utility, turnover)
            return torch.stack([head(z_l) for head in self.ens_heads], dim=1)


class HoloHRMReasoner:
    """Bottom-Up Non-Commutative Syntax-Tree Encoder + Multi-Head Epistemic UCB Surrogate (`z_H, z_L`)."""

    def __init__(
        self,
        preset: str = "cpu_fast",
        seed: int = 2057,
        n_heads_ens: int = 3,
        device: str | None = None,
    ) -> None:
        self.preset_spec = get_preset_spec(preset)
        self.preset = self.preset_spec.name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vocab = ASTVocab(max_seq_len=16)
        self.dim = self.preset_spec.holo_hrm_np_dim
        self.n_heads_ens = n_heads_ens
        self.use_amp = self.preset_spec.use_amp
        self.is_xla_tpu = False
        v = self.vocab.vocab_size
        s = 0.08
        # Deterministic compositional tree projection matrices (non-commutative left vs right!)
        self.emb = self.rng.normal(0.0, s, size=(v, self.dim))
        self.w_unary = self.rng.normal(0.0, s, size=(self.dim * 2, self.dim))
        self.w_binary = self.rng.normal(0.0, s, size=(self.dim * 4, self.dim))
        self.w_regime = self.rng.normal(0.0, s, size=(14, self.dim))
        # Fast z_L & Slow z_H recursive cells
        self.w_zl = self.rng.normal(0.0, s, size=(self.dim * 2, self.dim))
        self.w_zh = self.rng.normal(0.0, s, size=(self.dim * 2, self.dim))
        # Compact low-variance feature representation + empirical Bayesian token credit memory
        feat_dim = self.dim + len(ECONOMIC_NICHES) + 5
        self.head_w = np.zeros((n_heads_ens, feat_dim, 2), dtype=float)
        self.head_b = np.zeros((n_heads_ens, 2), dtype=float)
        self.tok_val_ema = np.zeros(v, dtype=float)
        self.tok_turn_ema = np.full(v, 0.08, dtype=float)
        self.tok_seen_cnt = np.zeros(v, dtype=float)
        self.global_util_prior = 0.0
        self.global_turn_prior = 0.08
        self.policy_w = np.zeros((self.dim, len(MECHANISM_FAMILIES)), dtype=float)

        if HAS_TORCH:
            self.device: Any
            self.device, self.is_xla_tpu = resolve_torch_device(device)
            torch.manual_seed(seed)
            self.torch_net: Any = _PyTorchHoloTreeNet(
                vocab_size=v,
                d_hidden=self.preset_spec.hrm_d_hidden,
                n_layers=self.preset_spec.hrm_n_layers,
                n_heads_ens=n_heads_ens,
            ).to(self.device)
            self.torch_opt: Any = torch.optim.AdamW(
                self.torch_net.parameters(), lr=self.preset_spec.hrm_lr, weight_decay=1e-2
            )
            self.backend_runtime = (
                f"pytorch_xla_holo_hrm:{self.device}" if self.is_xla_tpu else f"pytorch_holo_hrm:{self.device}"
            )
        else:
            self.device = "cpu"
            self.torch_net = None
            self.torch_opt = None
            self.backend_runtime = "holo_tree_hrm_ensemble"

    def parameter_count(self) -> int:
        np_n = int(
            self.emb.size
            + self.w_unary.size
            + self.w_binary.size
            + self.w_regime.size
            + self.w_zl.size
            + self.w_zh.size
            + self.head_w.size
            + self.head_b.size
            + self.tok_val_ema.size
            + self.tok_turn_ema.size
            + self.policy_w.size
        )
        if HAS_TORCH and self.torch_net is not None:
            return np_n + int(sum(p.numel() for p in self.torch_net.parameters()))
        return np_n

    def estimate_training_vram_mb(self) -> float:
        bytes_per_param = 10.0 if self.use_amp else 16.0
        return round((self.parameter_count() * bytes_per_param) / (1024.0**2) + 36.0, 2)

    def save_checkpoint(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np_state = {
            "emb": self.emb,
            "w_unary": self.w_unary,
            "w_binary": self.w_binary,
            "w_regime": self.w_regime,
            "w_zl": self.w_zl,
            "w_zh": self.w_zh,
            "head_w": self.head_w,
            "head_b": self.head_b,
            "tok_val_ema": self.tok_val_ema,
            "tok_turn_ema": self.tok_turn_ema,
            "tok_seen_cnt": self.tok_seen_cnt,
            "policy_w": self.policy_w,
        }
        if HAS_TORCH and self.torch_net is not None:
            torch.save(
                {
                    "preset": self.preset,
                    "state_dict": self.torch_net.state_dict(),
                    "np_state": np_state,
                    "priors": (self.global_util_prior, self.global_turn_prior),
                },
                out,
            )
            return out
        npz_path = out.with_suffix(".npz")
        npz_payload: dict[str, Any] = {
            "priors": np.array([self.global_util_prior, self.global_turn_prior], dtype=float),
            **np_state,
        }
        np.savez(npz_path, **npz_payload)
        return npz_path

    def load_checkpoint(self, path: str | Path) -> None:
        src = Path(path)
        if HAS_TORCH and self.torch_net is not None and src.exists():
            ckpt = torch.load(src, map_location=self.device, weights_only=False)
            self.torch_net.load_state_dict(ckpt["state_dict"])
            for k, arr in ckpt.get("np_state", {}).items():
                setattr(self, k, np.asarray(arr, dtype=float))
            if "priors" in ckpt:
                self.global_util_prior, self.global_turn_prior = map(float, ckpt["priors"])
        elif src.with_suffix(".npz").exists():
            data = np.load(src.with_suffix(".npz"))
            for k in (
                "emb",
                "w_unary",
                "w_binary",
                "w_regime",
                "w_zl",
                "w_zh",
                "head_w",
                "head_b",
                "tok_val_ema",
                "tok_turn_ema",
                "tok_seen_cnt",
                "policy_w",
            ):
                if k in data:
                    setattr(self, k, np.asarray(data[k], dtype=float))
            if "priors" in data:
                self.global_util_prior = float(data["priors"][0])
                self.global_turn_prior = float(data["priors"][1])

    def _bayesian_token_stats(self, tid: int, m_prior: float = 2.0) -> tuple[float, float]:
        cnt = float(self.tok_seen_cnt[tid])
        u = (cnt * float(self.tok_val_ema[tid]) + m_prior * self.global_util_prior) / (cnt + m_prior)
        t = (cnt * float(self.tok_turn_ema[tid]) + m_prior * self.global_turn_prior) / (cnt + m_prior)
        return u, t

    @staticmethod
    def _ln_tanh(x: np.ndarray) -> np.ndarray:
        z = (x - float(np.mean(x))) / (float(np.std(x)) + 1e-6)
        return np.tanh(z)

    def encode_tree_recursive(self, node: ASTNode) -> np.ndarray:
        """Bottom-up non-commutative syntax-tree encoding preserving argument order (`sub`, `safe_div`)."""
        if node.node_type == "terminal":
            tid = self.vocab.token_to_id.get(f"TERM:{node.name}", 0)
            return self.emb[tid]
        if node.window is not None:
            w = self.vocab._nearest_window(node.window)
            op_id = self.vocab.token_to_id.get(f"OP:{node.name}:{w}", 0)
        else:
            op_id = self.vocab.token_to_id.get(f"OP:{node.name}", 0)
        e_op = self.emb[op_id]
        if len(node.args) == 1:
            c0 = self.encode_tree_recursive(node.args[0])
            return self._ln_tanh(c0 + np.concatenate([e_op, c0]) @ self.w_unary)
        c_l = self.encode_tree_recursive(node.args[0])
        c_r = self.encode_tree_recursive(node.args[1])
        cat = np.concatenate([e_op, c_l, c_r, c_l - c_r])
        return self._ln_tanh(0.5 * (c_l + c_r) + cat @ self.w_binary)

    def _feature_vec(self, node: ASTNode, regime_vec: np.ndarray) -> np.ndarray:
        h_tree = self.encode_tree_recursive(node)
        h_reg = np.tanh(regime_vec @ self.w_regime)
        z_h = h_reg.copy()
        z_l = h_tree.copy()
        for _ in range(2):
            for _ in range(2):
                z_l = self._ln_tanh(z_l + np.concatenate([z_h, h_tree]) @ self.w_zl)
            z_h = self._ln_tanh(z_h + np.concatenate([z_l, h_reg]) @ self.w_zh)

        toks = [
            t for t in self.vocab.encode_ast(node)
            if t not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
        ]
        emp_u = float(np.mean([self.tok_val_ema[t] for t in toks])) if toks else 0.0
        emp_t = float(np.mean([self.tok_turn_ema[t] for t in toks])) if toks else 0.10
        niche_oh = np.zeros(len(ECONOMIC_NICHES), dtype=float)
        niche_oh[ECONOMIC_NICHES.index(dominant_economic_niche(node))] = 1.0
        meta = np.array(
            [
                emp_u * 5.0,
                emp_t * 2.0,
                float(node.depth()) / 5.0,
                1.0 if is_typed_stationary_ast(node) else 0.0,
                float(regime_vec[0]),
            ],
            dtype=float,
        )
        return np.concatenate([z_l, niche_oh, meta])

    def score_ucb(self, node: ASTNode, regime_vec: np.ndarray, ucb_beta: float = 0.18) -> tuple[float, float, float]:
        """Return (ucb_acquisition, mean_utility_pred, epistemic_std)."""
        feat = self._feature_vec(node, regime_vec)
        preds = np.einsum("hfd,f->hd", self.head_w, feat) + self.head_b  # (H, 2)
        if HAS_TORCH and self.torch_net is not None:
            self.torch_net.eval()
            seq = self.vocab.encode_ast(node)
            parents, slots, depths = compute_prefix_tree_coords(self.vocab, seq)
            with torch.no_grad():
                t_pred = (
                    self.torch_net(
                        torch.tensor([seq], dtype=torch.long, device=self.device),
                        torch.tensor([parents], dtype=torch.long, device=self.device),
                        torch.tensor([slots], dtype=torch.long, device=self.device),
                        torch.tensor([depths], dtype=torch.long, device=self.device),
                        torch.tensor([regime_vec], dtype=torch.float32, device=self.device),
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            preds = 0.55 * preds + 0.45 * t_pred
        util_preds = preds[:, 0] - 0.08 * np.clip(preds[:, 1], 0.0, 0.5)
        mu = float(np.mean(util_preds))
        sd = float(np.std(util_preds))
        return mu + ucb_beta * sd, mu, sd

    def identify_weakest_subtree_pos(self, spec: FactorSpec, regime_vec: np.ndarray) -> int | None:
        """Counterfactual token-credit attribution: locate lowest-value non-root AST subtree."""
        seq = self.vocab.encode_ast(spec.root)
        active = [
            i for i in range(2, len(seq)) if seq[i] not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
        ]
        if not active:
            return None
        _ = regime_vec
        worst_score = 1e9
        worst_pos: int | None = None
        for pos in active[:6]:
            s0, s1 = self.vocab.subtree_span(seq, pos)
            sub_toks = [
                seq[j] for j in range(s0, min(len(seq), s1))
                if seq[j] not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
            ]
            if not sub_toks:
                continue
            sc = float(np.mean([self.tok_val_ema[t] for t in sub_toks]))
            if sc < worst_score:
                worst_score = sc
                worst_pos = pos
        return worst_pos

    def fit_replay(self, replay: list[dict[str, Any]]) -> float:
        """Update empirical token value statistics, regularized E=3 bootstrap heads, and optional PyTorch tree net."""
        if len(replay) < 6:
            return 0.0
        tok_u_sum = np.zeros(self.vocab.vocab_size, dtype=float)
        tok_t_sum = np.zeros(self.vocab.vocab_size, dtype=float)
        tok_cnt = np.zeros(self.vocab.vocab_size, dtype=float)
        u_vals = [float(r["utility"]) for r in replay]
        t_vals = [float(r["turnover"]) for r in replay]
        self.global_util_prior = float(np.median(u_vals))
        self.global_turn_prior = float(np.median(t_vals))

        for r in replay:
            toks = {
                t for t in self.vocab.encode_ast(r["node"])
                if t not in (self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id)
            }
            for t in toks:
                tok_u_sum[t] += float(r["utility"])
                tok_t_sum[t] += float(r["turnover"])
                tok_cnt[t] += 1.0
        seen = tok_cnt > 0
        self.tok_val_ema[seen] = 0.30 * self.tok_val_ema[seen] + 0.70 * (tok_u_sum[seen] / tok_cnt[seen])
        self.tok_turn_ema[seen] = 0.30 * self.tok_turn_ema[seen] + 0.70 * (tok_t_sum[seen] / tok_cnt[seen])
        self.tok_seen_cnt = tok_cnt

        feats = np.vstack([self._feature_vec(r["node"], r["regime_vec"]) for r in replay])
        targets = np.column_stack([np.asarray(u_vals, dtype=float), np.asarray(t_vals, dtype=float)])
        n = len(replay)
        d = feats.shape[1]
        losses = []
        for h in range(self.n_heads_ens):
            boot_idx = self.rng.integers(0, n, size=n)
            xb = feats[boot_idx]
            yb = targets[boot_idx]
            x_aug = np.column_stack([xb, np.ones(n, dtype=float)])
            reg = 10.0 * np.eye(d + 1)
            reg[-1, -1] = 0.0
            sol = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ yb)
            self.head_w[h] = 0.35 * self.head_w[h] + 0.65 * sol[:-1]
            self.head_b[h] = 0.35 * self.head_b[h] + 0.65 * sol[-1]
            pred = feats @ self.head_w[h] + self.head_b[h]
            losses.append(float(np.mean((pred - targets) ** 2)))

        if HAS_TORCH and self.torch_net is not None and self.torch_opt is not None:
            self.torch_net.train()
            seqs, ps, ss, ds = [], [], [], []
            for r in replay:
                sq = self.vocab.encode_ast(r["node"])
                p_arr, s_arr, d_arr = compute_prefix_tree_coords(self.vocab, sq)
                seqs.append(sq)
                ps.append(p_arr)
                ss.append(s_arr)
                ds.append(d_arr)
            x_t = torch.tensor(seqs, dtype=torch.long, device=self.device)
            p_t = torch.tensor(np.stack(ps), dtype=torch.long, device=self.device)
            s_t = torch.tensor(np.stack(ss), dtype=torch.long, device=self.device)
            d_t = torch.tensor(np.stack(ds), dtype=torch.long, device=self.device)
            r_t = torch.tensor(np.stack([r["regime_vec"] for r in replay]), dtype=torch.float32, device=self.device)
            y_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
            amp_ctx, scaler = amp_autocast_and_scaler(self.device, self.use_amp, self.is_xla_tpu)
            for _ in range(4):
                with amp_ctx:
                    out = self.torch_net(x_t, p_t, s_t, d_t, r_t)  # (B, H, 2)
                    loss_t = F.smooth_l1_loss(out, y_t.unsqueeze(1).expand_as(out))
                self.torch_opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss_t).backward()
                    scaler.unscale_(self.torch_opt)
                    torch.nn.utils.clip_grad_norm_(self.torch_net.parameters(), 1.0)
                    scaler.step(self.torch_opt)
                    scaler.update()
                else:
                    loss_t.backward()
                    torch.nn.utils.clip_grad_norm_(self.torch_net.parameters(), 1.0)
                    self.torch_opt.step()
                xla_mark_step_if_needed(self.is_xla_tpu)
            losses.append(float(loss_t.item()))

        return round(float(np.mean(losses)), 6)


# ---------------------------------------------------------------------------
# 4. Behavioral 5x4 MAP-Elites Archive + Screened Elite Representation
# ---------------------------------------------------------------------------
@dataclass
class HoloElite:
    spec: FactorSpec                 # causal FactorSpec (already smoothed w* & sign-aligned if depth<=4)
    sign: float                      # +1.0 or -1.0 (handles depth-5 smoothed negated factors cleanly!)
    smooth_window: int
    family: str
    niche: str
    cell: tuple[str, int]
    utility: float
    rank_ic: float
    min_fold_ic: float
    icir: float
    net_sharpe: float
    turnover: float
    ic_series: pd.Series = field(repr=False)
    signal_vec: np.ndarray = field(repr=False)
    signal_df: pd.DataFrame = field(repr=False)

    @property
    def fingerprint(self) -> str:
        return f"{self.spec.root.structural_fingerprint()}:{int(self.sign)}"


class HoloQDArchive:
    """5x4 Behavioral Quality-Diversity Archive over (economic_niche x horizon_turnover_bucket)."""

    def __init__(self) -> None:
        self.cells: dict[tuple[str, int], HoloElite] = {}

    def insert(self, elite: HoloElite) -> bool:
        cur = self.cells.get(elite.cell)
        if cur is None or elite.utility > cur.utility:
            self.cells[elite.cell] = elite
            return True
        return False

    def all_elites(self) -> list[HoloElite]:
        return sorted(self.cells.values(), key=lambda e: e.utility, reverse=True)

    @property
    def coverage(self) -> int:
        return len(self.cells)


# ---------------------------------------------------------------------------
# 5. Holo-MAVE-Q Engine (MSSF + SRSS + Shapley Subtree Credit)
# ---------------------------------------------------------------------------
@dataclass
class HoloRoundLog:
    round_idx: int
    latent_pool: int
    screened: int
    archive_coverage: int
    archive_inserts: int
    ensemble_size: int
    search_objective: float
    mean_shapley_reward: float
    diffu_loss: float
    hrm_loss: float
    val_rank_ic: float
    val_net_sharpe: float
    val_selection_score: float
    evaluator_trials: int


@dataclass
class HoloMAVEQResult:
    preset: str
    backend_runtime: str
    specialist_total_params: int
    round_logs: list[HoloRoundLog]
    champion_round: int
    champion_expressions: list[str]
    champion_weights: list[float]
    champion_smoothing: list[int]
    champion_niches: list[str]
    champion_search_metrics: dict[str, float]
    champion_validation_metrics: dict[str, float]
    champion_validation_fold_std: dict[str, float]
    champion_sealed_test_metrics: dict[str, Any]
    search_evaluations: int
    total_evaluator_trials: int
    archive_coverage: int
    typed_guard_rejections: int
    validation_daily_returns: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float), repr=False
    )


def _curated_domain_seeds() -> list[tuple[str, ASTNode]]:
    """Compact stationary economic seeds across all 5 VN niches (strictly causal DSL expressions)."""
    t = lambda n: ASTNode(node_type="terminal", name=n)  # noqa: E731
    op = lambda n, *a, w=None: ASTNode(node_type="op", name=n, window=w, args=list(a))  # noqa: E731
    return [
        ("value_quality_pit", op("cs_zscore", t("pit_value_quality"))),
        ("value_quality_pit", op("sub", op("cs_rank", t("pit_roe")), op("cs_rank", t("pit_pb_ratio")))),
        ("foreign_flow_regime", op("ts_mean", t("foreign_net_ratio"), w=5)),
        ("foreign_flow_regime", op("sub", op("cs_rank", op("ts_mean", t("foreign_net_ratio"), w=10)), op("cs_rank", t("fol_room_pct")))),
        ("momentum_liquidity", op("safe_div", op("ts_return", t("close"), w=15), op("ts_std", t("return_1d"), w=20))),
        ("reversal_microstructure", op("neg", op("mul", op("ts_return", t("close"), w=3), t("turnover_shock_20d")))),
        ("reversal_microstructure", op("sub", t("dist_to_ceiling"), t("intraday_range"))),
        ("momentum_liquidity", op("neg", op("ts_zscore", t("turnover_shock_20d"), w=10))),
    ]


class HoloMAVEQEngine:
    """Holographic Neuro-Symbolic Alpha Evolution (strictly search-zone learning, sealed test touched once)."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        cfg: dict[str, Any],
        preset: str = "cpu_fast",
        device: str | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        self.preset = preset
        self.seed = int(cfg.get("seed", 2026))
        self.rng = np.random.default_rng(self.seed)

        self.diffusion = TopologicalTreeDiffusion(preset=preset, seed=self.seed, device=device)
        self.hrm = HoloHRMReasoner(preset=preset, seed=self.seed + 31, device=device)
        self.verifier = ASTVerifier(max_depth=5)
        self.sampler = GrammarTreeSampler(max_depth=4)
        self.archive = HoloQDArchive()

        # Ablation switches
        ab = cfg.get("ablations", {})
        self.use_typed_guard = bool(ab.get("typed_guard", True))
        self.use_tree_diffusion = bool(ab.get("tree_diffusion", True))
        self.use_holo_hrm = bool(ab.get("holo_hrm", True))
        self.use_mssf = bool(ab.get("mssf_smoothing", True))
        self.use_srss = bool(ab.get("srss_spectral_allocation", True))

        # Hermetically isolated SEARCH-ZONE-ONLY feature panel & forward returns!
        # Validation & sealed_test rows physically do NOT exist in `self.search_panel`!
        search_bars = evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "search")
        search_end = pd.Timestamp(evaluator.sandbox_cfg["zones"]["search"]["end"])
        pit_search = evaluator.snapshot.fundamentals_pit[
            evaluator.snapshot.fundamentals_pit["announcement_date"] <= search_end
        ].copy()
        self.search_panel = compute_feature_panel(search_bars, pit_search)
        s_close = self.search_panel["close"]
        s_dates = pd.DatetimeIndex(s_close.index)
        safe_n = max(1, len(s_dates) - evaluator.horizon_days)
        self.safe_dates = s_dates[:safe_n]
        # Forward return strictly inside search_dates (t + H <= search_end)!
        self.search_fwd = (s_close.shift(-evaluator.horizon_days) / s_close - 1.0).reindex(index=self.safe_dates)
        self.search_folds = evaluator.splitter.split(self.safe_dates)

        self.steward_base = np.array(
            [
                float(search_bars["hit_ceiling"].mean()) * 10.0,
                float(search_bars["hit_floor"].mean()) * 10.0,
                float(search_bars["is_halted"].mean()) * 10.0,
                float(s_close.pct_change(fill_method=None).std().mean()) * 20.0,
                min(1.0, len(search_bars) / 50000.0),
                0.5,
            ],
            dtype=float,
        )
        self.fam_ic_sum = {f: 0.04 for f in MECHANISM_FAMILIES}
        self.fam_cnt = {f: 1 for f in MECHANISM_FAMILIES}
        self.seen_raw_fps: set[str] = set()
        self.screen_cache: dict[str, HoloElite | None] = {}
        self.ensemble_cache: dict[str, EvaluationOutcome] = {}
        self.hrm_replay: list[dict[str, Any]] = []
        self.typed_guard_rejections = 0
        self.n_screens = 0

    def _regime_vec(self) -> np.ndarray:
        means = [self.fam_ic_sum[f] / max(1, self.fam_cnt[f]) * 10.0 for f in MECHANISM_FAMILIES]
        cnts = [min(1.0, self.fam_cnt[f] / 25.0) for f in MECHANISM_FAMILIES]
        return np.concatenate([self.steward_base, np.asarray(means), np.asarray(cnts)])

    # ------------------------------------------------------------------ MSSF (Search-Zone Only!)
    @staticmethod
    def _fast_top6_turnover_proxy(sig_df: pd.DataFrame, rebal_step: int = 3, top_k: int = 6) -> float:
        """Exact vectorized Top-K rebalance-boundary turnover proxy on search-zone signals."""
        arr = sig_df.to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, -1e9)
        n_days, n_syms = arr.shape
        k = min(top_k, n_syms)
        if n_days <= rebal_step or k <= 0:
            return 0.20
        rebal_rows = arr[::rebal_step]
        if len(rebal_rows) < 2:
            return 0.20
        top_idx = np.argpartition(rebal_rows, -k, axis=1)[:, -k:]
        # Vectorized membership indicator matrix (n_rebal, n_syms)
        membership = np.zeros_like(rebal_rows, dtype=np.float32)
        rows_ix = np.arange(len(rebal_rows))[:, None]
        membership[rows_ix, top_idx] = 1.0
        overlap = np.sum(membership[1:] * membership[:-1], axis=1) / float(k)
        # 2-way turnover on rebalance day = 2 * (1 - overlap), amortized over `rebal_step` days
        return float(np.mean(2.0 * (1.0 - overlap) / float(rebal_step)))

    def _mssf_calibrate_on_search(self, raw_spec: FactorSpec) -> tuple[FactorSpec, float, int] | None:
        """Nyquist-Matched Multi-Scale Causal State-Space Filter Bank (`MSSF`) on search panel.

        Applies causal hysteresis `ts_mean(., w)` (w in {3, 5} <= prediction horizon H=5) ONLY when
        the raw factor exhibits high rebalance-boundary churn (`est_turn > 0.12`), is not already
        `ts_mean`-wrapped, and retains >= 88% of its raw search-zone Rank IC.
        """
        try:
            raw_mat = evaluate_factor_spec(raw_spec, self.search_panel, verifier=self.verifier)
        except Exception:
            return None
        s_base = raw_mat.reindex(index=self.safe_dates)
        if float(np.nanstd(s_base.to_numpy(dtype=float))) < 1e-8:
            return None

        windows = [int(w) for w in self.cfg.get("mssf_windows", [1, 3, 5])] if self.use_mssf else [1]
        turn_pen = float(self.cfg.get("mssf_turnover_penalty", 0.14))
        core_depth = raw_spec.root.depth()
        already_smoothed = raw_spec.root.name == "ts_mean"

        raw_mu = s_base.mean(axis=1)
        raw_sd = s_base.std(axis=1) + 1e-8
        raw_norm = s_base.sub(raw_mu, axis=0).div(raw_sd, axis=0).fillna(0.0)
        raw_ic_s = compute_rank_ic_series(raw_norm, self.search_fwd)
        if len(raw_ic_s) < 20:
            return None
        raw_mean_ic = float(raw_ic_s.mean())
        raw_sign = 1.0 if raw_mean_ic >= 0.0 else -1.0
        raw_abs_ic = abs(raw_mean_ic)
        raw_turn = self._fast_top6_turnover_proxy(raw_norm * raw_sign)

        # Low-turnover anchors stay crisp at w=1; `ts_mean` roots may still widen slightly if churn > 0.16
        if raw_turn <= 0.11 or (already_smoothed and raw_turn <= 0.16) or not self.use_mssf:
            windows = [1]

        best_choice: tuple[float, int, float] | None = None
        for w in windows:
            if w > 1 and core_depth >= 5:
                continue
            cand_df = (
                s_base
                if w <= 1
                else s_base.rolling(window=w, min_periods=max(1, w // 2)).mean()
            )
            mu = cand_df.mean(axis=1)
            sd = cand_df.std(axis=1) + 1e-8
            norm_df = cand_df.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
            ic_s = compute_rank_ic_series(norm_df, self.search_fwd)
            if len(ic_s) < 20:
                continue
            mean_ic = float(ic_s.mean())
            sign = 1.0 if mean_ic >= 0.0 else -1.0
            abs_ic = abs(mean_ic)
            # Never allow low-pass smoothing to invert the raw economic sign or wash out >25% of IC!
            if w > 1 and (sign != raw_sign or abs_ic < 0.75 * raw_abs_ic):
                continue
            signed_ic_s = ic_s * sign
            fold_ics = [
                float(signed_ic_s.reindex(f.eval_dates).dropna().mean())
                for f in self.search_folds
                if len(signed_ic_s.reindex(f.eval_dates).dropna()) >= 5
            ]
            min_f_ic = min(fold_ics) if fold_ics else abs_ic
            oriented_df = norm_df * sign
            est_turn = self._fast_top6_turnover_proxy(oriented_df)
            score = 0.65 * abs_ic + 0.35 * min_f_ic - turn_pen * est_turn
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, w, sign)

        if best_choice is None:
            return None
        _, w_star, sign_star = best_choice
        node = raw_spec.root
        if w_star > 1:
            node = ASTNode(node_type="op", name="ts_mean", window=w_star, args=[node])
        eff_sign = sign_star
        if sign_star < 0.0 and node.depth() <= 4:
            node = ASTNode(node_type="op", name="neg", args=[node])
            eff_sign = 1.0
        adapted = FactorSpec(
            factor_id=f"{raw_spec.factor_id}_w{w_star}",
            hypothesis_id=raw_spec.hypothesis_id,
            description=f"MSSF(w={w_star}) {raw_spec.description}",
            root=node,
            normalize_cross_section=True,
        )
        return adapted, eff_sign, w_star

    def screen_candidate(self, raw_spec: FactorSpec, family: str, run_id: str) -> HoloElite | None:
        calibrated = self._mssf_calibrate_on_search(raw_spec)
        if calibrated is None:
            return None
        spec, sign, w_star = calibrated
        fp_key = f"{spec.root.structural_fingerprint()}:{int(sign)}"
        if fp_key in self.screen_cache:
            return self.screen_cache[fp_key]

        # Official search-zone evaluation via human-owned SealedEvaluator (fair trial counting!)
        eval_sp = spec
        if sign < 0.0:
            # If depth==5 prevented AST `neg` wrapping, evaluate negated via 1-element weighted ensemble
            out = self.evaluator.evaluate_factor_ensemble(
                specs=[spec],
                weights=[-1.0],
                zone="search",
                candidate_id=spec.factor_id,
                run_id=run_id,
            )
        else:
            out = self.evaluator.evaluate_factor(eval_sp, zone="search", run_id=run_id)
        self.n_screens += 1

        s_df = (
            evaluate_factor_spec(spec, self.search_panel, verifier=self.verifier)
            .reindex(index=self.safe_dates)
            .fillna(0.0)
            * sign
        )
        ic_s = compute_rank_ic_series(s_df, self.search_fwd)
        mean_ic = float(ic_s.mean()) if len(ic_s) else float(out.metrics["rank_ic"])
        fold_ics = [float(fm["rank_ic"]) for fm in out.fold_metrics] or [mean_ic]
        min_fold_ic = float(min(fold_ics))
        sr = float(out.metrics["net_sharpe"])
        turn = float(out.metrics["daily_turnover"])
        icir = float(out.metrics["icir"])

        # Robust search-zone utility: cross-fold minimax IC + friction-penalized Sharpe
        utility = (
            0.55 * mean_ic
            + 0.35 * min_fold_ic
            + 0.020 * np.clip(sr, -1.5, 3.0)
            - 0.16 * turn
        )
        niche = dominant_economic_niche(spec.root)
        elite = HoloElite(
            spec=spec,
            sign=sign,
            smooth_window=w_star,
            family=family,
            niche=niche,
            cell=(niche, horizon_bucket(turn)),
            utility=float(utility),
            rank_ic=mean_ic,
            min_fold_ic=min_fold_ic,
            icir=icir,
            net_sharpe=sr,
            turnover=turn,
            ic_series=ic_s,
            signal_vec=s_df.to_numpy(dtype=float).ravel(),
            signal_df=s_df,
        )
        self.screen_cache[fp_key] = elite
        return elite

    # ------------------------------------------------------------------ Latent Pool Generation
    def _generate_latent_pool(self, r_idx: int, n_latent: int) -> list[tuple[str, FactorSpec]]:
        pool: list[tuple[str, FactorSpec]] = []
        seen_batch: set[str] = set()

        # Round 1 seeds compact domain-grounded causal primitives alongside diffusion & grammar samples
        if r_idx == 1:
            for fam, node in _curated_domain_seeds():
                fp = node.structural_fingerprint()
                if fp not in self.seen_raw_fps and (not self.use_typed_guard or is_typed_stationary_ast(node)):
                    self.seen_raw_fps.add(fp)
                    seen_batch.add(fp)
                    pool.append(
                        (
                            fam,
                            FactorSpec(
                                factor_id=f"holo_seed_{len(pool)}",
                                hypothesis_id="holo_seed",
                                description=f"domain seed ({fam})",
                                root=node,
                            ),
                        )
                    )

        fam_weights = np.asarray(
            [max(0.10, self.fam_ic_sum[f] / max(1, self.fam_cnt[f]) + 0.03) for f in MECHANISM_FAMILIES],
            dtype=float,
        )
        fam_weights /= fam_weights.sum()
        elites = self.archive.all_elites()
        regime_vec = self._regime_vec()
        temp = max(0.42, 0.80 - 0.05 * r_idx)

        for k in range(n_latent * 5):
            if len(pool) >= n_latent:
                break
            fam = MECHANISM_FAMILIES[int(self.rng.choice(len(MECHANISM_FAMILIES), p=fam_weights))]
            fid, hid = f"holo_r{r_idx}_c{k}", f"holoh_r{r_idx}_c{k}"
            u = self.rng.random()
            seed_k = self.seed + r_idx * 6151 + k * 29

            if u < 0.16:
                cand = FactorSpec(
                    factor_id=fid,
                    hypothesis_id=hid,
                    description="grammar exploration",
                    root=self.sampler.random_tree(self.rng, max_depth=4),
                )
            elif len(elites) >= 2 and u < 0.34:
                # Bridge-MDLM Cross-Niche Fusion: fuse 2 distinct elites under a denoised binary operator
                i1, i2 = self.rng.choice(len(elites), size=2, replace=False)
                p_left, p_right = elites[int(i1)].spec.root, elites[int(i2)].spec.root
                while p_left.name in ("neg", "ts_mean") and p_left.args and p_left.depth() >= 3:
                    p_left = p_left.args[0]
                while p_right.name in ("neg", "ts_mean") and p_right.args and p_right.depth() >= 3:
                    p_right = p_right.args[0]
                cand = self.diffusion.bridge_fuse_elites(
                    left_root=p_left,
                    right_root=p_right,
                    family=fam,
                    factor_id=fid,
                    hypothesis_id=hid,
                    temperature=temp,
                    seed=seed_k,
                    use_tree_coords=self.use_tree_diffusion,
                )
            elif elites and u < 0.72:
                t_idx = self.rng.integers(0, len(elites), size=min(3, len(elites)))
                parent = max((elites[int(i)] for i in t_idx), key=lambda e: e.utility)
                base_root = parent.spec.root
                # Unwrap outer MSSF ts_mean / neg so subtree infilling operates on core structural tree
                while base_root.name in ("neg", "ts_mean") and base_root.args and base_root.depth() >= 3:
                    base_root = base_root.args[0]
                base_sp = FactorSpec(
                    factor_id=parent.spec.factor_id,
                    hypothesis_id=parent.spec.hypothesis_id,
                    description=parent.spec.description,
                    root=base_root,
                )
                weak_pos = (
                    self.hrm.identify_weakest_subtree_pos(base_sp, regime_vec)
                    if self.use_holo_hrm
                    else None
                )
                cand = self.diffusion.attribution_guided_infill(
                    parent_spec=base_sp,
                    weak_start_pos=weak_pos,
                    family=fam,
                    factor_id=fid,
                    hypothesis_id=hid,
                    max_depth=4,
                    temperature=temp,
                    seed=seed_k,
                    use_tree_coords=self.use_tree_diffusion,
                )
            else:
                cand = self.diffusion.sample_factor(
                    family=fam,
                    factor_id=fid,
                    hypothesis_id=hid,
                    max_depth=4,
                    temperature=temp + 0.05,
                    seed=seed_k,
                    use_tree_coords=self.use_tree_diffusion,
                )

            fp = cand.root.structural_fingerprint()
            if fp in seen_batch or fp in self.seen_raw_fps:
                continue
            try:
                self.verifier.verify(cand)
            except Exception:
                continue
            if self.use_typed_guard and not is_typed_stationary_ast(cand.root):
                self.typed_guard_rejections += 1
                continue
            seen_batch.add(fp)
            pool.append((fam, cand))
        return pool

    # ------------------------------------------------------------------ SRSS Spectral Portfolio Solver
    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)

    def _select_diverse_basis(self, max_basis: int = 5, max_corr: float = 0.65) -> list[HoloElite]:
        """Select decorrelated positive-IC elites ensuring multi-niche subspace coverage."""
        pos_elites = [e for e in self.archive.all_elites() if e.rank_ic > 0.015 and e.utility > -0.005]
        if not pos_elites:
            return self.archive.all_elites()[: max(1, min(3, len(self.archive.all_elites())))]

        selected: list[HoloElite] = []
        covered_niches: set[str] = set()

        # Pass 1: pick best representative per distinct economic niche if decorrelated
        for e in pos_elites:
            if e.niche in covered_niches:
                continue
            if all(abs(self._corr(e.signal_vec, s.signal_vec)) <= max_corr for s in selected):
                selected.append(e)
                covered_niches.add(e.niche)
            if len(selected) >= max_basis:
                return selected

        # Pass 2: fill remaining subspace slots with highest-utility decorrelated elites
        for e in pos_elites:
            if any(e.fingerprint == s.fingerprint for s in selected):
                continue
            if all(abs(self._corr(e.signal_vec, s.signal_vec)) <= max_corr for s in selected):
                selected.append(e)
            if len(selected) >= max_basis:
                break
        return selected

    def _fast_search_book_surrogate_score(self, basis: list[HoloElite], w: np.ndarray) -> float:
        """Zero-trial analytical search-zone portfolio score capturing true sub-additive Top-6 turnover."""
        comb_df: pd.DataFrame | None = None
        for e, wt in zip(basis, w, strict=True):
            if wt <= 1e-5:
                continue
            part = e.signal_df * float(wt)
            comb_df = part if comb_df is None else (comb_df + part)
        if comb_df is None:
            return -1e9
        ic_s = compute_rank_ic_series(comb_df, self.search_fwd)
        if len(ic_s) < 20:
            return -1e9
        mean_ic = float(ic_s.mean())
        fold_ics = [
            float(ic_s.reindex(f.eval_dates).dropna().mean())
            for f in self.search_folds
            if len(ic_s.reindex(f.eval_dates).dropna()) >= 5
        ]
        min_f_ic = min(fold_ics) if fold_ics else mean_ic
        comb_turn = self._fast_top6_turnover_proxy(comb_df)
        # Concentration HHI regularizer + realistic VN Top-6 turnover drag penalty
        hhi = float(np.sum(w**2))
        return 0.65 * mean_ic + 0.35 * min_f_ic - 0.46 * comb_turn - 0.012 * hhi

    def _solve_bto_srss_candidates(
        self,
        basis: list[HoloElite],
        corr_eta: float = 0.35,
        shrink_delta: float = 0.35,
    ) -> list[np.ndarray]:
        """Boundary-Turnover-Optimal Spectral Subspace Synthesis (`BTO-SRSS`) strictly on search zone.

        Constructs the spectral geodesic between the pure minimax-IC tangent portfolio `w_tangent` and
        the friction-damped anchor `w_anchor`, evaluates true sub-additive Top-6 boundary turnover along
        the simplex geodesic on `self.search_panel` (zero evaluator trials!), and returns the top-2
        distinct weight allocations for exact simulator verification.
        """
        m = len(basis)
        if m == 0:
            return []
        if m == 1 or not self.use_srss:
            return [np.full(m, 1.0 / m, dtype=float)]

        ic_panel = pd.concat([e.ic_series for e in basis], axis=1, sort=False).dropna()
        g_ic = np.asarray(
            [max(0.003, 0.60 * e.rank_ic + 0.40 * max(0.0, e.min_fold_ic)) for e in basis],
            dtype=float,
        )
        turns = np.asarray([e.turnover for e in basis], dtype=float)
        w_anchor = g_ic / np.power(0.035 + turns, 0.85)
        w_anchor /= float(np.sum(w_anchor)) + 1e-12

        if len(ic_panel) < 25:
            return [w_anchor]

        cov_ic = np.cov(ic_panel.to_numpy(dtype=float), rowvar=False)
        if cov_ic.ndim == 0:
            cov_ic = np.array([[float(cov_ic)]])
        avg_var = float(np.mean(np.diag(cov_ic))) + 1e-6

        sig_corr = np.eye(m, dtype=float)
        for i in range(m):
            for j in range(i + 1, m):
                c = abs(self._corr(basis[i].signal_vec, basis[j].signal_vec))
                sig_corr[i, j] = sig_corr[j, i] = c

        # Sub-linear turnover damping (linear ratio 0.04 + turns instead of quadratic explosion!)
        damped_diag = np.diag((0.04 + turns) / 0.08)
        sigma_spec = (
            (1.0 - shrink_delta) * cov_ic
            + shrink_delta * avg_var * np.eye(m)
            + corr_eta * avg_var * (sig_corr**2 - np.eye(m))
            + 0.45 * avg_var * damped_diag
        )
        w_tan = np.clip(np.linalg.solve(sigma_spec + 1e-6 * np.eye(m), g_ic), 0.0, None)
        w_tan = w_tan / (float(np.sum(w_tan)) + 1e-12) if float(np.sum(w_tan)) > 1e-12 else w_anchor.copy()

        # Geodesic simplex candidates + single-factor concentration cap when >=3 orthogonal mechanisms exist
        cap = 0.56 if m >= 3 else 1.0
        candidates_raw: list[np.ndarray] = []
        for lam in (0.0, 0.25, 0.50, 0.70, 0.85, 1.0):
            w = (1.0 - lam) * w_tan + lam * w_anchor
            # Project onto capped simplex [0, cap]
            for _ in range(3):
                over = w > cap
                if not np.any(over) or np.all(over):
                    break
                excess = float(np.sum(w[over] - cap))
                w[over] = cap
                under = ~over
                if float(np.sum(w[under])) > 1e-12:
                    w[under] += excess * (w[under] / float(np.sum(w[under])))
            w = np.where(w >= 0.035, w, 0.0)
            if float(np.sum(w)) <= 1e-12:
                w[int(np.argmax(g_ic))] = 1.0
            w = w / float(np.sum(w))
            candidates_raw.append(w)

        scored = [(self._fast_search_book_surrogate_score(basis, w), w) for w in candidates_raw]
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen_w: list[np.ndarray] = []
        for _, w in scored:
            if all(float(np.max(np.abs(w - prev))) > 0.06 for prev in chosen_w):
                chosen_w.append(w)
            if len(chosen_w) >= 2:
                break
        return chosen_w or [scored[0][1]]

    def _eval_book_on_search(
        self, elites: list[HoloElite], weights: np.ndarray, run_id: str
    ) -> tuple[float, EvaluationOutcome]:
        active = [(e, float(w)) for e, w in zip(elites, weights, strict=True) if w > 1e-5]
        specs = [e.spec for e, _ in active]
        signed_w = [e.sign * w for e, w in active]
        key = "|".join(
            f"{sp.root.canonical_expr()}:{round(sw, 4)}"
            for sp, sw in sorted(zip(specs, signed_w, strict=True), key=lambda x: x[0].root.canonical_expr())
        )
        if key not in self.ensemble_cache:
            self.ensemble_cache[key] = self.evaluator.evaluate_factor_ensemble(
                specs=specs,
                weights=signed_w,
                zone="search",
                candidate_id=f"holo_book_{len(self.ensemble_cache) + 1}",
                run_id=run_id,
            )
        out = self.ensemble_cache[key]
        m = out.metrics
        fold_sr = [float(fm["net_sharpe"]) for fm in out.fold_metrics] or [float(m["net_sharpe"])]
        fold_ic = [float(fm["rank_ic"]) for fm in out.fold_metrics] or [float(m["rank_ic"])]
        # Distributionally robust search-zone portfolio score (zero validation access!)
        score = (
            float(m["net_sharpe"])
            + 0.35 * float(min(fold_sr))
            + 3.5 * float(m["rank_ic"])
            + 1.5 * float(min(fold_ic))
            - 1.6 * float(m["daily_turnover"])
            - 0.08 * float(out.fold_std.get("net_sharpe", 0.0))
        )
        return float(score), out

    # ------------------------------------------------------------------ Main Evolution Loop
    def run(self, run_id: str = "phase6_holo_maveq", score_sealed: bool = False) -> HoloMAVEQResult:
        loop = self.cfg.get("loop", {})
        rounds = int(loop.get("rounds", 7))
        n_latent = int(loop.get("latent_candidates_per_round", 44))
        n_screen = int(loop.get("screen_per_round", 9))
        eps_explore = float(loop.get("ucb_explore_fraction", 0.25))
        max_ens = int(loop.get("max_ensemble_factors", 5))
        max_corr = float(loop.get("max_member_corr", 0.65))
        grpo_epochs = int(loop.get("diffu_grpo_epochs_per_round", 10))

        round_logs: list[HoloRoundLog] = []
        best_champ: tuple[float, int, list[HoloElite], list[float], EvaluationOutcome, EvaluationOutcome] | None = None
        last_val_key: str | None = None
        last_val_out: EvaluationOutcome | None = None

        for r_idx in range(1, rounds + 1):
            regime_vec = self._regime_vec()
            latent = self._generate_latent_pool(r_idx, n_latent)

            # Friction-Aligned Holo-HRM Epistemic UCB Latent Pruning (pure exploration + domain seeds in round 1)
            if r_idx == 1 or not self.use_holo_hrm:
                chosen = latent[:n_screen]
            else:
                ucb_beta = max(0.06, 0.24 - 0.03 * r_idx)
                ranked = sorted(
                    latent,
                    key=lambda item: self.hrm.score_ucb(item[1].root, regime_vec, ucb_beta=ucb_beta)[0],
                    reverse=True,
                )
                n_top = max(1, int(round(n_screen * (1.0 - eps_explore))))
                chosen = ranked[:n_top]
                rem = ranked[n_top:]
                if rem and len(chosen) < n_screen:
                    idx = self.rng.choice(len(rem), size=min(len(rem), n_screen - len(chosen)), replace=False)
                    chosen.extend(rem[int(i)] for i in idx)

            screened_round: list[HoloElite] = []
            inserts = 0
            for fam, raw_cand in chosen:
                self.seen_raw_fps.add(raw_cand.root.structural_fingerprint())
                elite = self.screen_candidate(raw_cand, fam, run_id=f"{run_id}_r{r_idx}")
                if elite is None:
                    continue
                screened_round.append(elite)
                if self.archive.insert(elite):
                    inserts += 1
                self.fam_ic_sum[fam] += max(0.0, elite.rank_ic)
                self.fam_cnt[fam] += 1
                self.hrm_replay.append(
                    {
                        "node": raw_cand.root,
                        "regime_vec": regime_vec,
                        "utility": elite.utility,
                        "turnover": elite.turnover,
                    }
                )

            # Synthesize candidate Spectral Subspace Books on search zone (2-3 compact evaluations/round!)
            basis = self._select_diverse_basis(max_basis=max_ens, max_corr=max_corr)
            if not basis and screened_round:
                basis = [max(screened_round, key=lambda e: e.utility)]

            book_candidates: list[tuple[float, list[HoloElite], np.ndarray, EvaluationOutcome]] = []
            if basis:
                sub_bases = [basis[: min(max_ens, len(basis))]]
                if len(basis) >= 4:
                    sub_bases.append(basis[:3])
                for sub_b in sub_bases:
                    for w_vec in self._solve_bto_srss_candidates(sub_b):
                        sc, s_out = self._eval_book_on_search(sub_b, w_vec, run_id=run_id)
                        book_candidates.append((sc, sub_b, w_vec, s_out))

            if not book_candidates:
                continue

            cur_search_obj, best_b, best_w_arr, best_s_out = max(book_candidates, key=lambda x: x[0])
            active_pairs = [(e, float(w)) for e, w in zip(best_b, best_w_arr, strict=True) if w > 1e-5]
            active_elites = [e for e, _ in active_pairs]
            active_weights = [w for _, w in active_pairs]

            # Counterfactual Subspace Shapley Credit (analytical IC-subspace leave-one-out gain, 0 extra trials!)
            shapley_bonus: dict[str, float] = {}
            if len(active_elites) >= 2:
                comb_sig: pd.DataFrame | None = None
                for e, w in active_pairs:
                    part = e.signal_df * w
                    comb_sig = part if comb_sig is None else (comb_sig + part)
                assert comb_sig is not None
                full_ic = float(compute_rank_ic_series(comb_sig, self.search_fwd).mean())
                for e, w in active_pairs:
                    loo_sig = comb_sig - e.signal_df * w
                    loo_ic = float(compute_rank_ic_series(loo_sig, self.search_fwd).mean())
                    shapley_bonus[e.fingerprint] = max(-0.03, full_ic - loo_ic)

            # Full-history Contrastive Experience Replay for Shapley-diffu-GRPO (top elites vs weakest screened!)
            all_screened = [e for e in self.screen_cache.values() if e is not None]
            all_screened.sort(key=lambda e: e.utility + 1.8 * shapley_bonus.get(e.fingerprint, 0.0), reverse=True)
            top_replay = all_screened[: min(10, len(all_screened))]
            bot_replay = all_screened[-min(8, max(0, len(all_screened) - len(top_replay))):] if len(all_screened) > len(top_replay) else []
            rollouts: list[dict[str, Any]] = []
            for e in top_replay + bot_replay:
                rew = e.utility + 1.8 * shapley_bonus.get(e.fingerprint, 0.0)
                rollouts.append({"family": e.family, "factor_spec": e.spec.model_dump(), "shapley_reward": rew})

            grpo_res = (
                self.diffusion.fit_shapley_diffu_grpo(rollouts, epochs=grpo_epochs)
                if self.use_tree_diffusion
                else {"loss": 0.0, "mean_reward": 0.0}
            )
            hrm_loss = self.hrm.fit_replay(self.hrm_replay[-192:]) if self.use_holo_hrm else 0.0

            # Single scalar validation evaluation per distinct round book
            v_specs = [e.spec for e in active_elites]
            v_signed_w = [e.sign * w for e, w in zip(active_elites, active_weights, strict=True)]
            v_key = "|".join(
                f"{sp.root.canonical_expr()}:{round(sw, 4)}"
                for sp, sw in sorted(zip(v_specs, v_signed_w, strict=True), key=lambda x: x[0].root.canonical_expr())
            )
            if v_key != last_val_key:
                last_val_out = self.evaluator.evaluate_factor_ensemble(
                    specs=v_specs,
                    weights=v_signed_w,
                    zone="validation",
                    candidate_id=f"holo_maveq_r{r_idx}",
                    run_id=run_id,
                )
                last_val_key = v_key
            assert last_val_out is not None
            val_sel = validation_selection_score(last_val_out)
            if best_champ is None or val_sel > best_champ[0]:
                best_champ = (
                    val_sel,
                    r_idx,
                    list(active_elites),
                    list(v_signed_w),
                    last_val_out,
                    best_s_out,
                )

            round_logs.append(
                HoloRoundLog(
                    round_idx=r_idx,
                    latent_pool=len(latent),
                    screened=len(screened_round),
                    archive_coverage=self.archive.coverage,
                    archive_inserts=inserts,
                    ensemble_size=len(active_elites),
                    search_objective=round(cur_search_obj, 4),
                    mean_shapley_reward=float(grpo_res.get("mean_reward", 0.0)),
                    diffu_loss=float(grpo_res.get("loss", 0.0)),
                    hrm_loss=hrm_loss,
                    val_rank_ic=round(float(last_val_out.metrics["rank_ic"]), 5),
                    val_net_sharpe=round(float(last_val_out.metrics["net_sharpe"]), 4),
                    val_selection_score=round(val_sel, 4),
                    evaluator_trials=int(self.evaluator.n_trials_evaluated),
                )
            )

        if best_champ is None:
            raise RuntimeError("Holo-MAVE-Q produced no valid ensemble.")

        _, champ_round, champ_elites, champ_signed_w, champ_val, champ_search = best_champ
        test_metrics: dict[str, Any] = {"note": "sealed_test_not_requested_dev_mode"}
        if score_sealed:
            try:
                test_out = self.evaluator.evaluate_factor_ensemble(
                    specs=[e.spec for e in champ_elites],
                    weights=champ_signed_w,
                    zone="sealed_test",
                    candidate_id="holo_maveq_v3_champion",
                    run_id=run_id,
                    allow_sealed=True,
                )
                test_metrics = dict(test_out.metrics)
            except SealedTestReuseError:
                test_metrics = {"note": "champion_hash_already_scored_in_ledger"}

        total_params = self.diffusion.parameter_count() + self.hrm.parameter_count()
        return HoloMAVEQResult(
            preset=self.preset,
            backend_runtime=f"diff={self.diffusion.backend_runtime}; hrm={self.hrm.backend_runtime}",
            specialist_total_params=total_params,
            round_logs=round_logs,
            champion_round=champ_round,
            champion_expressions=[e.spec.root.canonical_expr() for e in champ_elites],
            champion_weights=[round(float(w), 5) for w in champ_signed_w],
            champion_smoothing=[int(e.smooth_window) for e in champ_elites],
            champion_niches=[e.niche for e in champ_elites],
            champion_search_metrics=dict(champ_search.metrics),
            champion_validation_metrics=dict(champ_val.metrics),
            champion_validation_fold_std=dict(champ_val.fold_std),
            champion_sealed_test_metrics=test_metrics,
            search_evaluations=self.n_screens + len(self.ensemble_cache),
            total_evaluator_trials=int(self.evaluator.n_trials_evaluated),
            archive_coverage=self.archive.coverage,
            typed_guard_rejections=self.typed_guard_rejections,
            validation_daily_returns=champ_val.daily_net_returns,
        )

    def save_checkpoints(self, checkpoint_dir: str | Path) -> dict[str, str]:
        """Persist trained Tree-MDLM and Holo-HRM weights to disk (Kaggle / Colab / Modal)."""
        cdir = Path(checkpoint_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        diff_p = self.diffusion.save_checkpoint(cdir / f"holo_tree_mdlm_{self.preset}.pt")
        hrm_p = self.hrm.save_checkpoint(cdir / f"holo_hrm_{self.preset}.pt")
        return {"tree_mdlm": str(diff_p), "holo_hrm": str(hrm_p)}

    def load_checkpoints(self, checkpoint_dir: str | Path) -> None:
        cdir = Path(checkpoint_dir)
        self.diffusion.load_checkpoint(cdir / f"holo_tree_mdlm_{self.preset}.pt")
        self.hrm.load_checkpoint(cdir / f"holo_hrm_{self.preset}.pt")
