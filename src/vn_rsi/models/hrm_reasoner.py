"""Hierarchical Recursive Reasoning Model (HRM / TRM) for Bi-Level Regime Planning & Candidate Verification.

Inspired by:
- HRM (Sapient Intelligence & Tsinghua, arXiv:2506.21734): Coupled slow high-level (z_H) + fast low-level (z_L) modules.
- TRM (Samsung SAIL Montreal, arXiv:2510.04871): Iterative latent recursive refinement with tiny networks.

Roles inside VN-RSI:
1. Slow Module (z_H): Processes Vietnamese market microstructure regime state (ceiling/floor/halt rates +
   belief memory confidences/ICs) across `n_high_steps` -> outputs outer-loop `planner_focus_weights`,
   recommended `max_ast_depth`, and `preferred_windows`.
2. Fast Module (z_L): Conditioned on z_H and an AST token embedding, iterates `n_low_steps` in latent space
   -> predicts expected search-zone Rank IC (fast neural surrogate verifier before costly backtesting!).
3. Trains online strictly on search-zone observations (zero validation/test leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vn_rsi.models.ast_vocab import MECHANISM_FAMILIES, ASTVocab

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@dataclass
class HRMConfig:
    preset: str = "cpu_fast"  # "cpu_fast" (~0.12M) | "kaggle_t4_optimal" (~3.38M) | "kaggle_max_100m" (~78M)
    d_hidden: int = 64
    n_high_steps: int = 3
    n_low_steps: int = 4
    n_layers_per_block: int = 1
    regime_dim: int = 12  # 4 steward market stats + 4 family confidences + 4 family mean ICs
    max_seq_len: int = 16
    learning_rate: float = 1e-3
    use_amp_fp16: bool = False
    seed: int = 42

    @classmethod
    def from_preset(cls, preset: str, seed: int = 42) -> HRMConfig:
        if preset == "kaggle_max_100m":
            return cls(
                preset=preset,
                d_hidden=768,
                n_high_steps=4,
                n_low_steps=4,
                n_layers_per_block=10,
                regime_dim=12,
                max_seq_len=16,
                learning_rate=1.5e-4,
                use_amp_fp16=True,
                seed=seed,
            )
        if preset == "kaggle_t4_optimal":
            return cls(
                preset=preset,
                d_hidden=256,
                n_high_steps=4,
                n_low_steps=4,
                n_layers_per_block=4,
                regime_dim=12,
                max_seq_len=16,
                learning_rate=3e-4,
                use_amp_fp16=True,
                seed=seed,
            )
        return cls(
            preset="cpu_fast",
            d_hidden=64,
            n_high_steps=3,
            n_low_steps=4,
            n_layers_per_block=1,
            regime_dim=12,
            max_seq_len=16,
            learning_rate=1e-3,
            use_amp_fp16=False,
            seed=seed,
        )


if HAS_TORCH:

    class _ResidualMLPBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(F.gelu(self.fc1(self.ln(x))))

    class _HierarchicalRecursiveNet(nn.Module):
        """Coupled Slow (z_H) + Fast (z_L) Latent Recursive Reasoner."""

        def __init__(self, vocab_size: int, n_families: int, cfg: HRMConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.regime_enc = nn.Sequential(
                nn.Linear(cfg.regime_dim, cfg.d_hidden),
                nn.GELU(),
                nn.Linear(cfg.d_hidden, cfg.d_hidden),
            )
            self.tok_emb = nn.Embedding(vocab_size, cfg.d_hidden)

            # Fast low-level update cell f_L(z_L, z_H, x_ast)
            self.low_cell = nn.GRUCell(cfg.d_hidden * 2, cfg.d_hidden)
            self.low_blocks = nn.ModuleList([_ResidualMLPBlock(cfg.d_hidden) for _ in range(cfg.n_layers_per_block)])

            # Slow high-level update cell f_H(z_H, z_L, x_regime)
            self.high_cell = nn.GRUCell(cfg.d_hidden * 2, cfg.d_hidden)
            self.high_blocks = nn.ModuleList([_ResidualMLPBlock(cfg.d_hidden) for _ in range(cfg.n_layers_per_block)])

            # Heads: (1) Outer-loop family allocation policy, (2) AST surrogate Rank IC predictor
            self.policy_head = nn.Linear(cfg.d_hidden, n_families)
            self.depth_head = nn.Linear(cfg.d_hidden, 2)  # depth 3 vs depth 4
            self.surrogate_ic_head = nn.Linear(cfg.d_hidden, 1)

        def forward(
            self,
            regime_vec: torch.Tensor,
            ast_tokens: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            bsz = regime_vec.shape[0]
            x_reg = self.regime_enc(regime_vec)
            x_ast = self.tok_emb(ast_tokens).mean(dim=1)

            z_h = x_reg.clone()
            z_l = torch.zeros(bsz, self.cfg.d_hidden, device=regime_vec.device)

            # Hierarchical multi-timescale recursion: for each slow step, iterate n_low fast steps
            for _ in range(self.cfg.n_high_steps):
                for _ in range(self.cfg.n_low_steps):
                    low_in = torch.cat([z_h, x_ast], dim=-1)
                    z_l = self.low_cell(low_in, z_l)
                    for blk in self.low_blocks:
                        z_l = blk(z_l)
                high_in = torch.cat([z_l, x_reg], dim=-1)
                z_h = self.high_cell(high_in, z_h)
                for blk in self.high_blocks:
                    z_h = blk(z_h)

            family_logits = self.policy_head(z_h)
            depth_logits = self.depth_head(z_h)
            pred_ic = torch.tanh(self.surrogate_ic_head(z_l)).squeeze(-1) * 0.25
            return family_logits, depth_logits, pred_ic


class HRMReasoner:
    """Hierarchical Recursive Reasoner with online search-zone surrogate & policy updates."""

    def __init__(self, config: HRMConfig | None = None, device: str | None = None) -> None:
        self.config = config or HRMConfig()
        self.vocab = ASTVocab(max_seq_len=self.config.max_seq_len)
        self.train_steps_completed = 0
        self.last_loss: float | None = None

        if HAS_TORCH:
            self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(self.config.seed)
            self.net: Any = _HierarchicalRecursiveNet(
                vocab_size=self.vocab.vocab_size,
                n_families=len(MECHANISM_FAMILIES),
                cfg=self.config,
            ).to(self.device)
            self.optimizer: Any = torch.optim.AdamW(self.net.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)
            self.backend_runtime = f"pytorch:{self.device}"
        else:
            self.device = "cpu"
            self.net = None
            self.optimizer = None
            rng = np.random.default_rng(self.config.seed)
            self.np_w_surrogate = rng.normal(0.0, 0.05, size=(self.vocab.vocab_size + self.config.regime_dim,))
            self.np_w_policy = np.zeros((self.config.regime_dim, len(MECHANISM_FAMILIES)), dtype=float)
            self.backend_runtime = "numpy_fallback:cpu"

    def parameter_count(self) -> int:
        if HAS_TORCH and self.net is not None:
            return int(sum(p.numel() for p in self.net.parameters()))
        return int(self.np_w_surrogate.size + self.np_w_policy.size)

    def estimate_training_vram_mb(self) -> float:
        bytes_per_param = 10.0 if self.config.use_amp_fp16 else 16.0
        return round((self.parameter_count() * bytes_per_param) / (1024.0**2) + 36.0, 2)

    @staticmethod
    def encode_regime_state(
        steward_stats: dict[str, float] | None,
        family_ic: dict[str, float] | None,
        memory_list: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        st = steward_stats or {}
        fic = family_ic or {}
        conf_map = {str(m.get("family")): float(m.get("conf", 0.5)) for m in (memory_list or [])}
        vec = [
            float(st.get("ceiling_hit_rate", 0.03)) * 10.0,
            float(st.get("floor_hit_rate", 0.03)) * 10.0,
            float(st.get("halt_rate", 0.02)) * 10.0,
            min(1.0, float(st.get("search_rows", 20000.0)) / 50000.0),
        ]
        for fam in MECHANISM_FAMILIES:
            vec.append(float(conf_map.get(fam, 0.5)))
        for fam in MECHANISM_FAMILIES:
            vec.append(float(np.clip(fic.get(fam, 0.01) * 10.0, -2.0, 2.0)))
        return np.asarray(vec, dtype=np.float32)

    def score_candidate_ast(self, regime_vec: np.ndarray, ast_tokens: list[int]) -> float:
        """Fast low-level module (z_L) predicts expected search Rank IC of a candidate AST."""
        if HAS_TORCH and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                rv = torch.tensor(regime_vec[None, :], dtype=torch.float32, device=self.device)
                at = torch.tensor([ast_tokens], dtype=torch.long, device=self.device)
                _, _, pred_ic = self.net(rv, at)
                return float(pred_ic[0].item())
        bow = np.zeros(self.vocab.vocab_size, dtype=float)
        for t in ast_tokens:
            if 0 <= t < self.vocab.vocab_size:
                bow[t] += 1.0 / max(1, len(ast_tokens))
        feat = np.concatenate([bow, regime_vec])
        return float(np.tanh(np.dot(self.np_w_surrogate, feat)) * 0.20)

    def propose_outer_policy(
        self,
        regime_vec: np.ndarray,
        family_ic: dict[str, float],
        iteration: int,
    ) -> dict[str, Any]:
        """Slow high-level module (z_H) emits updated planner focus weights & AST depth for the outer loop."""
        dummy_ast = [self.vocab.bos_id, self.vocab.eos_id] + [self.vocab.pad_id] * (self.config.max_seq_len - 2)
        if HAS_TORCH and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                rv = torch.tensor(regime_vec[None, :], dtype=torch.float32, device=self.device)
                at = torch.tensor([dummy_ast], dtype=torch.long, device=self.device)
                fam_logits, depth_logits, _ = self.net(rv, at)
                probs = torch.softmax(fam_logits[0], dim=-1).detach().cpu().numpy()
                rec_depth = 4 if int(torch.argmax(depth_logits[0]).item()) == 1 or iteration >= 1 else 3
        else:
            logits = regime_vec @ self.np_w_policy
            e = np.exp(logits - np.max(logits))
            probs = e / (e.sum() + 1e-12)
            rec_depth = 4 if iteration >= 1 else 3

        # Blend latent neural policy with empirical search-zone family IC evidence (stabilizes tiny-data regime)
        emp = np.array([max(0.05, 0.25 + 2.5 * float(family_ic.get(f, 0.0))) for f in MECHANISM_FAMILIES])
        emp /= emp.sum()
        blended = np.clip(0.45 * probs + 0.55 * emp, 0.10, 0.55)
        blended /= blended.sum()

        win_schedules = [[3, 5, 15], [5, 10, 15], [3, 10, 20], [5, 15, 20]]
        return {
            "planner_focus_weights": {f: round(float(w), 4) for f, w in zip(MECHANISM_FAMILIES, blended, strict=True)},
            "implementer_max_ast_depth": rec_depth,
            "preferred_windows": win_schedules[iteration % len(win_schedules)],
            "best_of_n": 4 if iteration >= 1 else 3,
            "exploration_temperature": round(max(0.22, 0.40 - 0.04 * iteration), 3),
        }

    def train_on_search_observations(
        self,
        observations: list[dict[str, Any]],
        epochs: int = 8,
    ) -> dict[str, float | int]:
        """Update HRM z_L (surrogate IC head) & z_H (family policy head) strictly from search-zone results."""
        if not observations:
            return {"steps": 0, "loss": 0.0}

        regimes = np.stack([np.asarray(o["regime_vec"], dtype=np.float32) for o in observations])
        asts = [list(o["ast_tokens"]) for o in observations]
        ics = np.asarray([float(o["search_rank_ic"]) for o in observations], dtype=np.float32)
        fams = np.asarray([self.vocab.family_to_id(str(o["family"])) for o in observations], dtype=np.int64)
        best_fam_mask = ics >= float(np.median(ics))

        if HAS_TORCH and self.net is not None and self.optimizer is not None:
            self.net.train()
            r_t = torch.tensor(regimes, dtype=torch.float32, device=self.device)
            a_t = torch.tensor(asts, dtype=torch.long, device=self.device)
            ic_t = torch.tensor(ics, dtype=torch.float32, device=self.device)
            f_t = torch.tensor(fams, dtype=torch.long, device=self.device)
            win_t = torch.tensor(best_fam_mask, dtype=torch.bool, device=self.device)

            last_loss = 0.0
            for _ in range(epochs):
                fam_logits, _, pred_ic = self.net(r_t, a_t)
                loss_ic = F.mse_loss(pred_ic, ic_t)
                if win_t.any():
                    loss_pol = F.cross_entropy(fam_logits[win_t], f_t[win_t])
                else:
                    loss_pol = torch.tensor(0.0, device=self.device)
                loss = loss_ic + 0.15 * loss_pol
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                self.train_steps_completed += 1
                last_loss = float(loss.item())

            self.last_loss = round(last_loss, 6)
            return {"steps": epochs, "loss": self.last_loss, "samples": len(observations)}

        # Pure-NumPy gradient update fallback
        lr = 0.05
        for _ in range(epochs):
            for rv, tok_seq, target_ic, fid, is_win in zip(regimes, asts, ics, fams, best_fam_mask, strict=True):
                bow = np.zeros(self.vocab.vocab_size, dtype=float)
                for t in tok_seq:
                    if 0 <= t < self.vocab.vocab_size:
                        bow[t] += 1.0 / max(1, len(tok_seq))
                feat = np.concatenate([bow, rv])
                pred = float(np.tanh(np.dot(self.np_w_surrogate, feat)) * 0.20)
                err = target_ic - pred
                self.np_w_surrogate += lr * err * feat
                if is_win:
                    self.np_w_policy[:, fid] += 0.02 * rv
            self.train_steps_completed += 1
        self.last_loss = 0.0
        return {"steps": epochs, "loss": 0.0, "samples": len(observations)}

    def save_checkpoint(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if HAS_TORCH and self.net is not None:
            torch.save({"config": self.config.__dict__, "state_dict": self.net.state_dict()}, out)
        else:
            np.savez(out.with_suffix(".npz"), w_sur=self.np_w_surrogate, w_pol=self.np_w_policy)
        return out
