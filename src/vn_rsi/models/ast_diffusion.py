"""Discrete Masked AST Diffusion Model (Tree-MDLM) for Causal Factor Synthesis & Subtree Mutation.

Inspired by MDLM / LLaDA discrete masked diffusion, specialized to our ~70-token closed causal DSL:
- Runs native PyTorch Transformer encoder on GPU/CPU when `torch` is installed (scaling from ~0.1M CPU-fast
  to ~6.4M Kaggle/Colab T4 optimal preset, <0.5 GB VRAM!).
- Provides an automatic lightweight NumPy softmax-neural fallback when `torch` is not installed, so CPU CI
  stays 100% functional without mandatory 2GB PyTorch wheel downloads.
- Supports:
  1. Grammar-constrained parallel unmasking (`sample_factor`) -> 100% valid causal ASTs.
  2. Targeted Subtree Infilling (`infill_mutate_factor`) -> masks weak AST subtree & denoises conditionally.
  3. Online Reward-Weighted ReST Self-Training (`fit_on_trajectories`) on verified `search`-zone trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vn_rsi.dsl.ast_spec import FactorSpec
from vn_rsi.models.ast_vocab import MECHANISM_FAMILIES, ASTVocab

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@dataclass
class ASTDiffusionConfig:
    preset: str = "cpu_fast"  # "cpu_fast" (~0.1M) | "kaggle_t4_optimal" (~6.4M)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 16
    denoise_steps: int = 6
    learning_rate: float = 1e-3
    seed: int = 42

    @classmethod
    def from_preset(cls, preset: str, seed: int = 42) -> ASTDiffusionConfig:
        if preset == "kaggle_t4_optimal":
            return cls(
                preset=preset,
                d_model=256,
                n_layers=8,
                n_heads=8,
                d_ff=1024,
                max_seq_len=16,
                denoise_steps=8,
                learning_rate=3e-4,
                seed=seed,
            )
        return cls(
            preset="cpu_fast",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_len=16,
            denoise_steps=6,
            learning_rate=1e-3,
            seed=seed,
        )


FAMILY_TERMINAL_PRIORS: dict[str, list[str]] = {
    "momentum_liquidity": ["close", "return_1d", "turnover_shock_20d", "volume"],
    "reversal_microstructure": ["close", "turnover_shock_20d", "dist_to_ceiling", "intraday_range", "close_to_open"],
    "value_quality_pit": ["pit_value_quality", "pit_roe", "pit_pb_ratio", "pit_revenue_yoy"],
    "foreign_flow_regime": ["foreign_net_ratio", "fol_room_pct", "pit_value_quality", "close"],
}

FAMILY_OP_PRIORS: dict[str, list[str]] = {
    "momentum_liquidity": ["OP:safe_div", "OP:ts_return:15", "OP:ts_return:10", "OP:ts_std:20", "OP:cs_zscore"],
    "reversal_microstructure": ["OP:neg", "OP:mul", "OP:ts_return:3", "OP:ts_return:5", "OP:cs_zscore"],
    "value_quality_pit": ["OP:add", "OP:cs_zscore", "OP:cs_rank", "OP:ts_mean:5"],
    "foreign_flow_regime": ["OP:add", "OP:ts_mean:5", "OP:ts_mean:10", "OP:cs_zscore"],
}


if HAS_TORCH:

    class _DiscreteASTTransformer(nn.Module):
        """Bidirectional Transformer denoiser predicting clean AST tokens from masked sequences."""

        def __init__(self, vocab_size: int, n_families: int, cfg: ASTDiffusionConfig) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.fam_emb = nn.Embedding(n_families, cfg.d_model)
            self.time_proj = nn.Linear(1, cfg.d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
            )
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, vocab_size)

        def forward(
            self,
            token_ids: torch.Tensor,
            family_ids: torch.Tensor,
            mask_ratio: torch.Tensor,
        ) -> torch.Tensor:
            bsz, seq_len = token_ids.shape
            pos = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, seq_len)
            cond = self.fam_emb(family_ids).unsqueeze(1) + self.time_proj(mask_ratio.unsqueeze(-1)).unsqueeze(1)
            h = self.tok_emb(token_ids) + self.pos_emb(pos) + cond
            h = self.encoder(h)
            return self.head(self.ln_f(h))


class DiscreteASTDiffusion:
    """Wrapper unifying PyTorch GPU/CPU discrete masked diffusion + pure-NumPy fallback."""

    def __init__(self, config: ASTDiffusionConfig | None = None, device: str | None = None) -> None:
        self.config = config or ASTDiffusionConfig()
        self.vocab = ASTVocab(max_seq_len=self.config.max_seq_len)
        self.rng = np.random.default_rng(self.config.seed)
        self.train_steps_completed = 0
        self.last_train_loss: float | None = None

        # Domain-informed prior logits (weakly initializes token preferences before ReST updates)
        self.prior_bias = self._build_prior_bias()

        if HAS_TORCH:
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(self.config.seed)
            self.net: Any = _DiscreteASTTransformer(
                vocab_size=self.vocab.vocab_size,
                n_families=len(MECHANISM_FAMILIES),
                cfg=self.config,
            ).to(self.device)
            self.optimizer: Any = torch.optim.AdamW(self.net.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)
            self.backend_runtime = f"pytorch:{self.device}"
        else:
            # Lightweight NumPy conditional bigram/position parameter tensor for CPU-only zero-torch CI
            self.device = "cpu"
            self.net = None
            self.optimizer = None
            self.np_logits = np.zeros(
                (len(MECHANISM_FAMILIES), self.config.max_seq_len, self.vocab.vocab_size), dtype=float
            )
            self.backend_runtime = "numpy_fallback:cpu"

    def _build_prior_bias(self) -> np.ndarray:
        bias = np.zeros((len(MECHANISM_FAMILIES), self.vocab.vocab_size), dtype=float)
        for fam_idx, fam in enumerate(MECHANISM_FAMILIES):
            for tname in FAMILY_TERMINAL_PRIORS.get(fam, []):
                tid = self.vocab.token_to_id.get(f"TERM:{tname}")
                if tid is not None:
                    bias[fam_idx, tid] += 1.6
            for op_tok in FAMILY_OP_PRIORS.get(fam, []):
                oid = self.vocab.token_to_id.get(op_tok)
                if oid is not None:
                    bias[fam_idx, oid] += 1.5
        return bias

    def parameter_count(self) -> int:
        if HAS_TORCH and self.net is not None:
            return int(sum(p.numel() for p in self.net.parameters()))
        return int(self.np_logits.size)

    def estimate_training_vram_mb(self) -> float:
        """Conservative FP32/AdamW VRAM estimate (weights + grads + 2x optimizer states + activations)."""
        params = self.parameter_count()
        return round((params * 16.0) / (1024.0**2) + 32.0, 2)

    def _predict_logits(self, seq: list[int], family_id: int, mask_ratio: float) -> np.ndarray:
        if HAS_TORCH and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                x = torch.tensor([seq], dtype=torch.long, device=self.device)
                f_id = torch.tensor([family_id], dtype=torch.long, device=self.device)
                mr = torch.tensor([mask_ratio], dtype=torch.float32, device=self.device)
                logits = self.net(x, f_id, mr)[0].detach().cpu().numpy()
            return logits + self.prior_bias[family_id][None, :]
        return self.np_logits[family_id] + self.prior_bias[family_id][None, :]

    def _grammar_unmask_decode(
        self,
        init_seq: list[int],
        family_id: int,
        max_depth: int,
        temperature: float,
        rng: np.random.Generator,
    ) -> list[int]:
        """Iterative grammar-constrained masked diffusion denoising (left-to-right + parallel confidence refinement)."""
        seq = list(init_seq)
        max_body = min(self.config.max_seq_len - 2, 10)
        steps = max(2, self.config.denoise_steps)

        for step in range(steps):
            masked_positions = [i for i in range(1, self.config.max_seq_len - 1) if seq[i] == self.vocab.mask_id]
            if not masked_positions:
                break
            mask_ratio = len(masked_positions) / max(1, self.config.max_seq_len - 2)
            logits = self._predict_logits(seq, family_id, mask_ratio)

            # Unmask a chunk of masked positions in prefix order so grammar validity holds 100%
            chunk = max(1, int(np.ceil(len(masked_positions) / max(1, steps - step))))
            for pos in masked_positions[:chunk]:
                valid_mask = self.vocab.valid_next_token_mask(seq[:pos], max_depth=max_depth, max_body_tokens=max_body)
                pos_logits = logits[pos] / max(0.15, float(temperature))
                pos_logits = np.where(valid_mask, pos_logits, -1e9)
                shifted = pos_logits - np.max(pos_logits)
                probs = np.exp(shifted)
                probs = probs / (np.sum(probs) + 1e-12)
                seq[pos] = int(rng.choice(self.vocab.vocab_size, p=probs))

        # Repair tail deterministically if any slots remain unclosed
        for pos in range(1, self.config.max_seq_len):
            if seq[pos] == self.vocab.mask_id:
                valid_mask = self.vocab.valid_next_token_mask(seq[:pos], max_depth=max_depth, max_body_tokens=max_body)
                valid_ids = np.where(valid_mask)[0]
                seq[pos] = int(valid_ids[0]) if len(valid_ids) > 0 else self.vocab.pad_id
        return seq

    def sample_factor(
        self,
        family: str,
        factor_id: str,
        hypothesis_id: str,
        max_depth: int = 3,
        temperature: float = 0.55,
        seed: int | None = None,
    ) -> FactorSpec:
        """Sample a novel, 100% grammar-valid causal FactorSpec via iterative masked diffusion."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        fam_id = self.vocab.family_to_id(family)
        init_seq = [self.vocab.bos_id] + [self.vocab.mask_id] * (self.config.max_seq_len - 2) + [self.vocab.eos_id]
        final_seq = self._grammar_unmask_decode(init_seq, fam_id, max_depth=max_depth, temperature=temperature, rng=rng)
        return self.vocab.to_factor_spec(final_seq, factor_id=factor_id, hypothesis_id=hypothesis_id, family=family)

    def infill_mutate_factor(
        self,
        parent_spec: FactorSpec,
        family: str,
        factor_id: str,
        hypothesis_id: str,
        max_depth: int = 3,
        temperature: float = 0.5,
        seed: int | None = None,
    ) -> FactorSpec:
        """Subtree Infilling Mutation: mask a subtree of parent AST and denoise conditionally."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        fam_id = self.vocab.family_to_id(family)
        seq = self.vocab.encode_ast(parent_spec.root)

        active_positions = [
            i for i in range(1, len(seq)) if seq[i] not in (self.vocab.eos_id, self.vocab.pad_id, self.vocab.bos_id)
        ]
        if len(active_positions) <= 1:
            return self.sample_factor(family, factor_id, hypothesis_id, max_depth=max_depth, temperature=temperature, seed=seed)

        # Pick a non-root node if available, mask its entire subtree span + trailing suffix for valid re-completion
        pick_pool = active_positions[1:] if len(active_positions) > 1 else active_positions
        start = int(rng.choice(pick_pool))
        s0, s1 = self.vocab.subtree_span(seq, start)
        masked = list(seq)
        for pos in range(s0, min(self.config.max_seq_len - 1, max(s1, s0 + 2))):
            masked[pos] = self.vocab.mask_id

        mutated_seq = self._grammar_unmask_decode(masked, fam_id, max_depth=max_depth, temperature=temperature, rng=rng)
        return self.vocab.to_factor_spec(mutated_seq, factor_id=factor_id, hypothesis_id=hypothesis_id, family=family)

    def fit_on_trajectories(
        self,
        trajectories: list[dict[str, Any]],
        epochs: int = 8,
    ) -> dict[str, float | int]:
        """Online Reward-Weighted Masked Diffusion ReST Self-Training on verified search-zone trajectories."""
        if not trajectories:
            return {"steps": 0, "loss": 0.0}

        seqs: list[list[int]] = []
        fam_ids: list[int] = []
        weights: list[float] = []

        for tr in trajectories:
            try:
                sp = FactorSpec.model_validate(tr["factor_spec"])
                seqs.append(self.vocab.encode_ast(sp.root))
                fam_ids.append(self.vocab.family_to_id(str(tr.get("family", "momentum_liquidity"))))
                ic = float(tr.get("search_rank_ic", 0.02))
                sr = float(tr.get("search_net_sharpe", 0.1))
                weights.append(float(np.clip(1.0 + 10.0 * max(0.0, ic) + 0.5 * max(0.0, sr), 0.25, 5.0)))
            except Exception:
                continue

        if not seqs:
            return {"steps": 0, "loss": 0.0}

        if HAS_TORCH and self.net is not None and self.optimizer is not None:
            self.net.train()
            x_clean = torch.tensor(seqs, dtype=torch.long, device=self.device)
            f_t = torch.tensor(fam_ids, dtype=torch.long, device=self.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
            w_t = w_t / (w_t.mean() + 1e-8)

            last_loss = 0.0
            for _ in range(epochs):
                ratio = float(self.rng.uniform(0.20, 0.75))
                rand_mask = torch.rand(x_clean.shape, device=self.device) < ratio
                # Never mask BOS or PAD
                rand_mask[:, 0] = False
                rand_mask = rand_mask & (x_clean != self.vocab.pad_id)
                if not rand_mask.any():
                    rand_mask[:, 1] = True

                x_noisy = x_clean.clone()
                x_noisy[rand_mask] = self.vocab.mask_id
                mr_t = torch.full((x_clean.shape[0],), ratio, dtype=torch.float32, device=self.device)

                logits = self.net(x_noisy, f_t, mr_t)
                ce = F.cross_entropy(logits.view(-1, self.vocab.vocab_size), x_clean.view(-1), reduction="none")
                ce = ce.view(x_clean.shape)
                per_sample = (ce * rand_mask.float()).sum(dim=1) / rand_mask.float().sum(dim=1).clamp(min=1.0)
                loss = (per_sample * w_t).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                self.train_steps_completed += 1
                last_loss = float(loss.item())

            self.last_train_loss = round(last_loss, 5)
            return {"steps": epochs, "loss": self.last_train_loss, "samples": len(seqs)}

        # Pure-NumPy reward-weighted log-likelihood update fallback
        lr = 0.15
        for _ in range(epochs):
            for seq, fid, wt in zip(seqs, fam_ids, weights, strict=True):
                for pos in range(1, self.config.max_seq_len - 1):
                    tid = seq[pos]
                    if tid == self.vocab.pad_id:
                        break
                    row = self.np_logits[fid, pos] + self.prior_bias[fid]
                    probs = np.exp(row - np.max(row))
                    probs /= probs.sum() + 1e-12
                    grad = -probs
                    grad[tid] += 1.0
                    self.np_logits[fid, pos] += lr * wt * grad / max(1, len(seqs))
            self.train_steps_completed += 1
        self.last_train_loss = 0.0
        return {"steps": epochs, "loss": 0.0, "samples": len(seqs)}

    def save_checkpoint(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if HAS_TORCH and self.net is not None:
            torch.save({"config": self.config.__dict__, "state_dict": self.net.state_dict()}, out)
        else:
            np.savez(out.with_suffix(".npz"), np_logits=self.np_logits)
        return out
