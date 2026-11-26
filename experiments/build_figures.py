"""Task 10 CLI: Generate Publication-Grade Vector PDF & SVG Scientific Figures directly from Logged JSONs.

Zero external plotting dependencies required — renders exact vector PDF 1.4 (for LaTeX `\\includegraphics`)
and SVG 1.1 previews directly from `logs/v4_final_arena_trajectories.json`, `logs/experiment_registry.jsonl`,
`logs/v4_ablation_and_ood.json`, and `logs/veteran_reasoning_benchmark.json`:

1. `paper/figures/fig1_equity_drawdown.{pdf,svg}`:
   Cumulative Wealth & Drawdown under Vietnamese T+2.5 + ±7% Price Limits with Causal Regime Shading
   (MAVE-Q v4 vs. RD-Agent(Q), AlphaForge-DE, FinAgent, FinMem, TradingAgents, VN-Index Buy & Hold).
2. `paper/figures/fig2_pareto_sharpe_dsr.{pdf,svg}`:
   Pareto Frontier: Sealed Test Net Sharpe vs. Trial-Deflated DSR (log-scaled trial/VRAM bubble size)
   across MAVE-Q v4, Holo-MAVE-Q v3, Global SOTA Agents, Frozen 70B-671B Frontiers, and 7 Alpha Miners.
3. `paper/figures/fig3_regime_breakdown.{pdf,svg}`:
   4-Regime Performance Decomposition (Bull, Bear/Crash, Sideways, High-Vol Liquidity Crunch) across
   MAVE-Q v4 and top Global SOTA competitors on the Sealed Test Zone.
4. `paper/figures/fig4_ablation_and_veteran.{pdf,svg}`:
   Dual-Panel Evaluation: (Left) 8-Seed Architectural Ablation Net Sharpe & DSR ± SD;
   (Right) 32-Scenario 10-Year Veteran Cognitive Reasoning Accuracy, Risk F1, and Calibration (1 - Brier, 1 - ECE).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _pdf_escape(s: str) -> str:
    # Keep printable ASCII for Helvetica Type1 PDF font
    clean = s.encode("ascii", "replace").decode("ascii")
    return clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class VectorCanvas:
    """Dual-backend publication vector renderer outputting valid PDF 1.4 AND crisp SVG 1.1."""

    def __init__(self, width: float = 840.0, height: float = 460.0) -> None:
        self.w = float(width)
        self.h = float(height)
        self._svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(self.w)}" height="{int(self.h)}" '
            f'viewBox="0 0 {self.w:.1f} {self.h:.1f}" font-family="Helvetica, Arial, sans-serif">',
            f'<rect x="0" y="0" width="{self.w:.1f}" height="{self.h:.1f}" fill="#FFFFFF"/>',
        ]
        self._pdf: list[str] = [
            "1 1 1 rg",
            f"0 0 {self.w:.2f} {self.h:.2f} re f",
        ]

    def _py(self, y: float) -> float:
        return self.h - y

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'width="{w:.2f}"', f'height="{h:.2f}"']
        attrs.append(f'fill="{fill}"' if fill else 'fill="none"')
        if stroke:
            attrs.append(f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"')
        if opacity < 0.999:
            attrs.append(f'fill-opacity="{opacity:.2f}"')
        self._svg.append(f"<rect {' '.join(attrs)}/>")

        py = self._py(y + h)
        if fill:
            r, g, b = _hex_to_rgb01(fill)
            # Blend fill toward white in PDF for subtle shading without ExtGState complexity
            if opacity < 0.999:
                r = 1.0 - opacity * (1.0 - r)
                g = 1.0 - opacity * (1.0 - g)
                b = 1.0 - opacity * (1.0 - b)
            self._pdf.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {py:.2f} {w:.2f} {h:.2f} re f")
        if stroke:
            r, g, b = _hex_to_rgb01(stroke)
            self._pdf.append(
                f"{r:.3f} {g:.3f} {b:.3f} RG {stroke_width:.2f} w {x:.2f} {py:.2f} {w:.2f} {h:.2f} re S"
            )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#CBD5E1",
        stroke_width: float = 1.0,
        dashed: bool = False,
    ) -> None:
        dash_svg = ' stroke-dasharray="4,3"' if dashed else ""
        self._svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_svg}/>'
        )
        r, g, b = _hex_to_rgb01(stroke)
        dash_pdf = "[4 3] 0 d " if dashed else "[] 0 d "
        self._pdf.append(
            f"{dash_pdf}{r:.3f} {g:.3f} {b:.3f} RG {stroke_width:.2f} w "
            f"{x1:.2f} {self._py(y1):.2f} m {x2:.2f} {self._py(y2):.2f} l S [] 0 d"
        )

    def polyline(
        self,
        pts: list[tuple[float, float]],
        stroke: str,
        stroke_width: float = 1.8,
        dashed: bool = False,
    ) -> None:
        if len(pts) < 2:
            return
        pt_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        dash_svg = ' stroke-dasharray="5,3"' if dashed else ""
        self._svg.append(
            f'<polyline fill="none" stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_svg} points="{pt_str}"/>'
        )
        r, g, b = _hex_to_rgb01(stroke)
        dash_pdf = "[5 3] 0 d " if dashed else "[] 0 d "
        cmds = [f"{dash_pdf}{r:.3f} {g:.3f} {b:.3f} RG {stroke_width:.2f} w 1 J 1 j"]
        cmds.append(f"{pts[0][0]:.2f} {self._py(pts[0][1]):.2f} m")
        for x, y in pts[1:]:
            cmds.append(f"{x:.2f} {self._py(y):.2f} l")
        cmds.append("S [] 0 d")
        self._pdf.append(" ".join(cmds))

    def circle(
        self,
        cx: float,
        cy: float,
        rad: float,
        fill: str,
        stroke: str = "#1E293B",
        stroke_width: float = 0.8,
    ) -> None:
        self._svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rad:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" fill-opacity="0.88"/>'
        )
        # 4-segment cubic Bezier circle approximation in PDF
        k = 0.55228475 * rad
        py = self._py(cy)
        rf, gf, bf = _hex_to_rgb01(fill)
        rs, gs, bs = _hex_to_rgb01(stroke)
        path = (
            f"{cx + rad:.2f} {py:.2f} m "
            f"{cx + rad:.2f} {py + k:.2f} {cx + k:.2f} {py + rad:.2f} {cx:.2f} {py + rad:.2f} c "
            f"{cx - k:.2f} {py + rad:.2f} {cx - rad:.2f} {py + k:.2f} {cx - rad:.2f} {py:.2f} c "
            f"{cx - rad:.2f} {py - k:.2f} {cx - k:.2f} {py - rad:.2f} {cx:.2f} {py - rad:.2f} c "
            f"{cx + k:.2f} {py - rad:.2f} {cx + rad:.2f} {py - k:.2f} {cx + rad:.2f} {py:.2f} c "
            "B"
        )
        self._pdf.append(
            f"{rf:.3f} {gf:.3f} {bf:.3f} rg {rs:.3f} {gs:.3f} {bs:.3f} RG {stroke_width:.2f} w {path}"
        )

    def text(
        self,
        x: float,
        y: float,
        label: str,
        size: float = 10.0,
        color: str = "#0F172A",
        anchor: str = "start",
        bold: bool = False,
    ) -> None:
        weight = "bold" if bold else "normal"
        self._svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:.1f}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{_svg_escape(label)}</text>'
        )
        # Approximate text width for anchor alignment in Helvetica
        approx_w = len(label) * size * (0.54 if bold else 0.50)
        tx = x - (approx_w / 2.0 if anchor == "middle" else (approx_w if anchor == "end" else 0.0))
        r, g, b = _hex_to_rgb01(color)
        font_key = "/F2" if bold else "/F1"
        self._pdf.append(
            f"BT {font_key} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
            f"1 0 0 1 {tx:.2f} {self._py(y):.2f} Tm ({_pdf_escape(label)}) Tj ET"
        )

    def save(self, base_path: Path) -> tuple[Path, Path]:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path = base_path.with_suffix(".svg")
        pdf_path = base_path.with_suffix(".pdf")
        svg_path.write_text("\n".join(self._svg + ["</svg>"]) + "\n", encoding="utf-8")

        stream_bytes = ("\n".join(self._pdf) + "\n").encode("latin-1", "replace")
        objs: list[bytes] = [
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
            (
                f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w:.2f} {self.h:.2f}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >> endobj\n"
            ).encode("ascii"),
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
            b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n",
            (
                f"6 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("ascii")
                + stream_bytes
                + b"endstream endobj\n"
            ),
        ]
        buf = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = [0]
        for o in objs:
            offsets.append(len(buf))
            buf.extend(o)
        xref_pos = len(buf)
        buf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
        for off in offsets[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        buf.extend(
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )
        pdf_path.write_bytes(bytes(buf))
        return pdf_path, svg_path


def _load_latest_registry(reg_path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for ln in reg_path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rec = json.loads(ln)
            by_id[str(rec["run_id"])] = rec
    return by_id


def build_fig1_equity_drawdown(traj: dict[str, Any], out_dir: Path) -> list[str]:
    zone_key = "sealed_test" if traj.get("sealed_test", {}).get("models") else "validation"
    zdata = traj[zone_key]
    bm = zdata["benchmark_vnindex"]
    dates = sorted(bm.keys())
    regimes = zdata.get("regimes", {})
    models = zdata["models"]

    series_specs = [
        ("MAVE-Q_v4", "MAVE-Q v4 (Ours)", "#0F766E", 2.4, False),
        ("RD-Agent(Q)", "RD-Agent(Q)", "#2563EB", 1.5, False),
        ("AlphaForge-DE", "AlphaForge-DE", "#7C3AED", 1.4, False),
        ("FinAgent", "FinAgent", "#D97706", 1.4, False),
        ("FinMem", "FinMem", "#64748B", 1.2, True),
        ("VN-Index", "VN-Index Buy & Hold", "#DC2626", 1.5, True),
    ]

    wealth_map: dict[str, np.ndarray] = {}
    dd_map: dict[str, np.ndarray] = {}
    for key, _, _, _, _ in series_specs:
        src = bm if key == "VN-Index" else models.get(key)
        if not src:
            continue
        r = np.asarray([float(src.get(d, 0.0)) for d in dates], dtype=float)
        w = np.cumprod(1.0 + r)
        pk = np.maximum.accumulate(w)
        wealth_map[key] = w
        dd_map[key] = (w / np.maximum(pk, 1e-8)) - 1.0

    cv = VectorCanvas(860.0, 500.0)
    cv.text(430.0, 26.0, "Figure 1: Single-Shot Sealed Test Cumulative Wealth & Drawdown under VN T+2.5 Rules", 13.0, "#0F172A", "middle", True)
    cv.text(430.0, 42.0, f"Zone: {zone_key} ({dates[0]} to {dates[-1]}) | Shaded: Bull (green), Bear/Crunch (red) | Net of 15bps comm + 10bps tax + impact", 9.0, "#475569", "middle")

    x0, x1 = 68.0, 830.0
    # Top panel: Cumulative Wealth
    ty0, ty1 = 62.0, 275.0
    # Bottom panel: Drawdown
    by0, by1 = 310.0, 448.0

    cv.rect(x0, ty0, x1 - x0, ty1 - ty0, fill="#F8FAFC", stroke="#94A3B8", stroke_width=0.9)
    cv.rect(x0, by0, x1 - x0, by1 - by0, fill="#F8FAFC", stroke="#94A3B8", stroke_width=0.9)

    # Regime shading bands
    n = max(1, len(dates) - 1)
    dx = (x1 - x0) / n
    for i, d in enumerate(dates):
        rg = regimes.get(d, "Sideways")
        if rg == "Bull":
            cv.rect(x0 + i * dx, ty0, dx + 0.5, ty1 - ty0, fill="#BBF7D0", opacity=0.22)
        elif rg in ("Bear_Crash", "High_Vol_Liquidity_Crunch"):
            cv.rect(x0 + i * dx, ty0, dx + 0.5, ty1 - ty0, fill="#FECACA", opacity=0.22)
            cv.rect(x0 + i * dx, by0, dx + 0.5, by1 - by0, fill="#FECACA", opacity=0.22)

    all_w = np.concatenate(list(wealth_map.values())) if wealth_map else np.array([1.0, 2.0])
    w_min = min(0.75, float(all_w.min()) * 0.95)
    w_max = float(all_w.max()) * 1.06
    for k in range(5):
        yv = w_min + k * (w_max - w_min) / 4.0
        yy = ty1 - (yv - w_min) / max(w_max - w_min, 1e-6) * (ty1 - ty0)
        cv.line(x0, yy, x1, yy, "#E2E8F0", 0.7, dashed=True)
        cv.text(x0 - 6.0, yy + 3.0, f"{yv:.2f}x", 8.5, "#334155", "end")
    cv.text(18.0, (ty0 + ty1) / 2.0, "Wealth", 9.5, "#0F172A", "middle", True)

    all_dd = np.concatenate(list(dd_map.values())) if dd_map else np.array([-0.5, 0.0])
    dd_min = min(-0.50, float(all_dd.min()) * 1.08)
    for k in range(4):
        yv = dd_min + k * (0.0 - dd_min) / 3.0
        yy = by1 - (yv - dd_min) / max(0.0 - dd_min, 1e-6) * (by1 - by0)
        cv.line(x0, yy, x1, yy, "#E2E8F0", 0.7, dashed=True)
        cv.text(x0 - 6.0, yy + 3.0, f"{yv * 100:.0f}%", 8.5, "#334155", "end")
    cv.text(18.0, (by0 + by1) / 2.0, "Drawdown", 9.0, "#0F172A", "middle", True)

    for key, _, col, sw, dashed in series_specs:
        if key not in wealth_map:
            continue
        w_arr = wealth_map[key]
        d_arr = dd_map[key]
        pts_w = [(x0 + i * dx, ty1 - (float(v) - w_min) / max(w_max - w_min, 1e-6) * (ty1 - ty0)) for i, v in enumerate(w_arr)]
        pts_d = [(x0 + i * dx, by1 - (float(v) - dd_min) / max(0.0 - dd_min, 1e-6) * (by1 - by0)) for i, v in enumerate(d_arr)]
        cv.polyline(pts_w, col, sw, dashed)
        cv.polyline(pts_d, col, sw, dashed)

    # X-axis date ticks
    for idx in np.linspace(0, len(dates) - 1, 6, dtype=int):
        xx = x0 + int(idx) * dx
        cv.text(xx, by1 + 14.0, dates[int(idx)][:7], 8.5, "#334155", "middle")

    # Legend bar at bottom
    lx = x0
    for key, label, col, sw, dashed in series_specs:
        if key not in wealth_map:
            continue
        cv.line(lx, 480.0, lx + 18.0, 480.0, col, sw, dashed)
        cv.text(lx + 23.0, 483.0, label, 8.8, "#0F172A", "start", key == "MAVE-Q_v4")
        lx += 128.0

    p_pdf, p_svg = cv.save(out_dir / "fig1_equity_drawdown")
    return [str(p_pdf), str(p_svg)]


def build_fig2_pareto_sharpe_dsr(runs: dict[str, dict[str, Any]], out_dir: Path) -> list[str]:
    pts: list[tuple[str, float, float, int, str]] = []
    if "phase10_v4_final_arena" in runs:
        p10 = runs["phase10_v4_final_arena"]
        v4_tm = p10["maveq_v4"].get("champion_sealed_test_extended") or p10["maveq_v4"]["champion_sealed_test_metrics"]
        if "net_sharpe" in v4_tm:
            pts.append(("MAVE-Q v4 (Ours)", float(v4_tm["deflated_sharpe_prob"]), float(v4_tm["net_sharpe"]), int(p10["maveq_v4"]["total_evaluator_trials"]), "#0F766E"))
        for gk, gdata in p10.get("global_competitors", {}).items():
            gtm = gdata.get("sealed_test_extended") or gdata.get("sealed_test_metrics", {})
            if "net_sharpe" in gtm:
                pts.append((gk, float(gtm["deflated_sharpe_prob"]), float(gtm["net_sharpe"]), int(gdata["total_evaluator_trials"]), "#2563EB"))

    if "phase7_holo_maveq_combat" in runs:
        p7 = runs["phase7_holo_maveq_combat"]
        v3 = p7["holo_maveq_v3"]
        v3t = v3.get("champion_sealed_test_metrics", {})
        if "net_sharpe" in v3t:
            pts.append(("Holo-MAVE-Q v3", float(v3t["deflated_sharpe_prob"]), float(v3t["net_sharpe"]), int(v3["total_evaluator_trials"]), "#7C3AED"))
        for ck, cdata in p7.get("competitors", {}).items():
            ct = cdata.get("sealed_test_metrics", {})
            if "net_sharpe" in ct:
                short = ck.replace("_alpha_miner", "").replace("_iterative_miner", "").replace("_style_library", "").replace("_dsl_search", "")
                pts.append((short, float(ct["deflated_sharpe_prob"]), float(ct["net_sharpe"]), int(cdata["total_evaluator_trials"]), "#94A3B8"))

    if "phase5_frontier_combat" in runs:
        p5 = runs["phase5_frontier_combat"]
        for fn, ft in p5.get("frontier_sealed_test_scores", {}).items():
            if "net_sharpe" in ft:
                pts.append((fn.split("-")[0] + "-" + fn.split("-")[1], float(ft["deflated_sharpe_prob"]), float(ft["net_sharpe"]), 5, "#D97706"))

    if "phase2_baselines" in runs:
        bm = runs["phase2_baselines"]["sealed_test_baselines"]["vnindex_buy_hold"]
        pts.append(("VN-Index B&H", float(bm["deflated_sharpe_prob"]), float(bm["net_sharpe"]), 1, "#DC2626"))

    cv = VectorCanvas(840.0, 440.0)
    cv.text(420.0, 26.0, "Figure 2: Single-Shot Sealed Test Pareto Frontier (Net Sharpe vs. Trial-Deflated DSR)", 13.0, "#0F172A", "middle", True)
    cv.text(420.0, 42.0, "Bubble radius ~ log(Evaluator Trials); Top-Right = High Risk-Adjusted Return + High Multiple-Testing Credibility", 9.0, "#475569", "middle")

    x0, x1, y0, y1 = 72.0, 800.0, 62.0, 385.0
    cv.rect(x0, y0, x1 - x0, y1 - y0, fill="#F8FAFC", stroke="#94A3B8", stroke_width=0.9)

    srs = [p[2] for p in pts] or [0.0, 2.0]
    y_min = min(-0.2, min(srs) - 0.15)
    y_max = max(2.0, max(srs) + 0.25)

    for k in range(6):
        xv = k * 0.2
        xx = x0 + xv * (x1 - x0)
        cv.line(xx, y0, xx, y1, "#E2E8F0", 0.7, dashed=True)
        cv.text(xx, y1 + 15.0, f"{xv:.1f}", 8.8, "#334155", "middle")
    for k in range(6):
        yv = y_min + k * (y_max - y_min) / 5.0
        yy = y1 - (yv - y_min) / max(y_max - y_min, 1e-6) * (y1 - y0)
        cv.line(x0, yy, x1, yy, "#E2E8F0", 0.7, dashed=True)
        cv.text(x0 - 8.0, yy + 3.0, f"{yv:.2f}", 8.8, "#334155", "end")

    cv.text((x0 + x1) / 2.0, 412.0, "Trial-Deflated Sharpe Ratio Probability (DSR, higher = less multiple-testing inflation)", 9.5, "#0F172A", "middle", True)
    cv.text(22.0, (y0 + y1) / 2.0, "Test Sharpe", 9.5, "#0F172A", "middle", True)

    for name, dsr, sr, trials, col in pts:
        xx = x0 + float(np.clip(dsr, 0.0, 1.0)) * (x1 - x0)
        yy = y1 - (sr - y_min) / max(y_max - y_min, 1e-6) * (y1 - y0)
        rad = 4.5 + 1.8 * math.log1p(max(1, trials))
        is_ours = "MAVE-Q v4" in name
        cv.circle(xx, yy, rad + (2.0 if is_ours else 0.0), col, "#0F172A" if is_ours else "#334155", 1.5 if is_ours else 0.7)
        dx_lbl = -8.0 if dsr > 0.80 else 8.0
        anc = "end" if dsr > 0.80 else "start"
        cv.text(xx + dx_lbl, yy - 5.0 if is_ours else yy + 3.0, f"{name} ({trials}t)", 8.3 if not is_ours else 9.2, "#0F172A", anc, is_ours)

    p_pdf, p_svg = cv.save(out_dir / "fig2_pareto_sharpe_dsr")
    return [str(p_pdf), str(p_svg)]


def build_fig3_regime_breakdown(runs: dict[str, dict[str, Any]], out_dir: Path) -> list[str]:
    p10 = runs.get("phase10_v4_final_arena", {})
    v4_ext = p10.get("maveq_v4", {}).get("champion_sealed_test_extended") or p10.get("maveq_v4", {}).get("champion_validation_extended", {})
    gmap = p10.get("global_competitors", {})

    systems = [
        ("MAVE-Q v4 (Ours)", v4_ext.get("regime_breakdown", {}), "#0F766E"),
        ("RD-Agent(Q)", (gmap.get("RD-Agent(Q)", {}).get("sealed_test_extended") or {}).get("regime_breakdown", {}), "#2563EB"),
        ("AlphaForge-DE", (gmap.get("AlphaForge-DE", {}).get("sealed_test_extended") or {}).get("regime_breakdown", {}), "#7C3AED"),
        ("FinAgent", (gmap.get("FinAgent", {}).get("sealed_test_extended") or {}).get("regime_breakdown", {}), "#D97706"),
        ("FinMem", (gmap.get("FinMem", {}).get("sealed_test_extended") or {}).get("regime_breakdown", {}), "#64748B"),
    ]
    regimes = ["Bull", "Bear_Crash", "Sideways", "High_Vol_Liquidity_Crunch"]
    r_labels = ["Bull Expansion", "Bear / Drawdown", "Sideways", "High-Vol Crunch"]

    cv = VectorCanvas(840.0, 420.0)
    cv.text(420.0, 26.0, "Figure 3: Sealed Test Net Sharpe Decomposition Across 4 Causal Market Regimes", 13.0, "#0F172A", "middle", True)
    cv.text(420.0, 42.0, "Strictly causal benchmark regime classification | Higher bars indicate superior risk-adjusted resilience per regime", 9.0, "#475569", "middle")

    x0, x1, y0, y1 = 68.0, 810.0, 62.0, 355.0
    cv.rect(x0, y0, x1 - x0, y1 - y0, fill="#F8FAFC", stroke="#94A3B8", stroke_width=0.9)

    vals = [
        float(rb.get(rg, {}).get("net_sharpe", 0.0))
        for _, rb, _ in systems
        for rg in regimes
        if rg in rb
    ] or [-2.0, 4.0]
    y_min = min(-1.5, min(vals) - 0.5)
    y_max = max(3.5, max(vals) + 0.6)

    def _y(v: float) -> float:
        return y1 - (v - y_min) / max(y_max - y_min, 1e-6) * (y1 - y0)

    for k in range(6):
        yv = y_min + k * (y_max - y_min) / 5.0
        yy = _y(yv)
        cv.line(x0, yy, x1, yy, "#E2E8F0", 0.7, dashed=True)
        cv.text(x0 - 8.0, yy + 3.0, f"{yv:.1f}", 8.5, "#334155", "end")
    y_zero = _y(0.0)
    cv.line(x0, y_zero, x1, y_zero, "#334155", 1.1)

    group_w = (x1 - x0) / len(regimes)
    bar_w = group_w * 0.14
    for gi, (rg, rlbl) in enumerate(zip(regimes, r_labels, strict=True)):
        gx = x0 + gi * group_w + group_w * 0.14
        cv.text(x0 + (gi + 0.5) * group_w, y1 + 16.0, rlbl, 9.2, "#0F172A", "middle", True)
        for si, (_, rb, col) in enumerate(systems):
            v = float(rb.get(rg, {}).get("net_sharpe", 0.0))
            bx = gx + si * (bar_w + 3.0)
            yv = _y(v)
            top = min(yv, y_zero)
            h_bar = max(1.0, abs(yv - y_zero))
            cv.rect(bx, top, bar_w, h_bar, fill=col, stroke="#1E293B", stroke_width=0.4)

    lx = x0 + 20.0
    for sname, _, col in systems:
        cv.rect(lx, 392.0, 12.0, 10.0, fill=col, stroke="#1E293B", stroke_width=0.5)
        cv.text(lx + 16.0, 400.0, sname, 8.8, "#0F172A", "start", "MAVE-Q v4" in sname)
        lx += 145.0

    p_pdf, p_svg = cv.save(out_dir / "fig3_regime_breakdown")
    return [str(p_pdf), str(p_svg)]


def build_fig4_ablation_and_veteran(ab_data: dict[str, Any], vet_data: dict[str, Any], out_dir: Path) -> list[str]:
    cv = VectorCanvas(880.0, 430.0)
    cv.text(440.0, 26.0, "Figure 4: (Left) 8-Seed Architectural Ablation Study & (Right) 32-Scenario Veteran Reasoning Benchmark", 12.5, "#0F172A", "middle", True)

    # Left panel: 8-seed ablation Net Sharpe & DSR
    lx0, lx1, ly0, ly1 = 56.0, 428.0, 62.0, 355.0
    cv.rect(lx0, ly0, lx1 - lx0, ly1 - ly0, fill="#F8FAFC", stroke="#94A3B8", stroke_width=0.9)
    cv.text((lx0 + lx1) / 2.0, 52.0, "8-Seed Ablations (Validation Net Sharpe & DSR)", 9.5, "#0F172A", "middle", True)

    ab_agg = ab_data.get("ablation_aggregate", {})
    v_order = [
        ("full_maveq_v4", "Full v4"),
        ("no_case_memory", "w/o Mem"),
        ("no_pre_mortem", "w/o PM"),
        ("no_regime_controller", "w/o Ctrl"),
        ("no_zero_val_burn_cv", "w/o 0-Burn"),
        ("no_typed_tree_mdlm", "w/o Tree"),
    ]
    y_max_l = 1.65
    for k in range(5):
        yv = k * 0.4
        yy = ly1 - (yv / y_max_l) * (ly1 - ly0)
        cv.line(lx0, yy, lx1, yy, "#E2E8F0", 0.7, dashed=True)
        cv.text(lx0 - 6.0, yy + 3.0, f"{yv:.1f}", 8.3, "#334155", "end")

    gw = (lx1 - lx0) / len(v_order)
    for i, (vk, vlbl) in enumerate(v_order):
        row = ab_agg.get(vk, {})
        sr = float(row.get("net_sharpe_mean", 0.0))
        sr_sd = float(row.get("net_sharpe_std", 0.0))
        dsr = float(row.get("dsr_mean", 0.0))
        bx = lx0 + i * gw + gw * 0.14
        bw = gw * 0.32
        # Sharpe bar
        hs = max(1.0, (sr / y_max_l) * (ly1 - ly0))
        cv.rect(bx, ly1 - hs, bw, hs, fill="#0F766E" if i == 0 else "#3B82F6", stroke="#1E293B", stroke_width=0.5)
        # Error bar (+/- SD)
        y_hi = ly1 - min(y_max_l, sr + sr_sd) / y_max_l * (ly1 - ly0)
        y_lo = ly1 - max(0.0, sr - sr_sd) / y_max_l * (ly1 - ly0)
        cx_b = bx + bw / 2.0
        cv.line(cx_b, y_hi, cx_b, y_lo, "#0F172A", 1.1)
        # DSR bar
        hd = max(1.0, (dsr / y_max_l) * (ly1 - ly0))
        cv.rect(bx + bw + 3.0, ly1 - hd, bw, hd, fill="#F59E0B", stroke="#1E293B", stroke_width=0.5)
        cv.text(lx0 + (i + 0.5) * gw, ly1 + 14.0, vlbl, 8.2, "#0F172A", "middle", i == 0)

    cv.rect(lx0 + 20.0, 392.0, 12.0, 10.0, fill="#0F766E")
    cv.text(lx0 + 36.0, 400.0, "Val Net Sharpe (±SD)", 8.5, "#0F172A")
    cv.rect(lx0 + 185.0, 392.0, 12.0, 10.0, fill="#F59E0B")
    cv.text(lx0 + 201.0, 400.0, "Val DSR", 8.5, "#0F172A")

    # Right panel: 32-Scenario Veteran Cognitive Benchmark
    rx0, rx1, ry0, ry1 = 480.0, 854.0, 62.0, 355.0
    cv.rect(rx0, ry0, rx1 - rx0, ry1 - ry0, fill="#F8FAFC", stroke="#94A3B8", stroke_width=0.9)
    cv.text((rx0 + rx1) / 2.0, 52.0, "32-Scenario Veteran Reasoning Benchmark (0-1 Scale)", 9.5, "#0F172A", "middle", True)

    s_scores = vet_data.get("benchmark", {}).get("system_scores", {})
    vet_sys = [
        ("MAVE-Q_v4_Veteran", "MAVE-Q v4 Vet", "#0F766E"),
        ("Generic_Rule_Analyst", "Generic Rule", "#2563EB"),
        ("Static_Momentum_Analyst", "Static Mom.", "#7C3AED"),
        ("Naive_FOMO_Agent", "Naive FOMO", "#DC2626"),
    ]
    m_specs = [
        ("Acc", lambda d: float(d.get("decision_accuracy_pct", 0.0)) / 100.0),
        ("Risk F1", lambda d: float(d.get("hidden_risk_f1", 0.0))),
        ("1-Brier", lambda d: max(0.0, 1.0 - float(d.get("brier_score", 0.5)))),
        ("1-ECE", lambda d: max(0.0, 1.0 - float(d.get("ece", 0.5)))),
    ]
    for k in range(6):
        yv = k * 0.2
        yy = ry1 - yv * (ry1 - ry0)
        cv.line(rx0, yy, rx1, yy, "#E2E8F0", 0.7, dashed=True)
        cv.text(rx0 - 6.0, yy + 3.0, f"{yv:.1f}", 8.3, "#334155", "end")

    rgw = (rx1 - rx0) / len(m_specs)
    rbw = rgw * 0.18
    for mi, (mlbl, fn) in enumerate(m_specs):
        cv.text(rx0 + (mi + 0.5) * rgw, ry1 + 14.0, mlbl, 8.8, "#0F172A", "middle", True)
        gx = rx0 + mi * rgw + rgw * 0.10
        for si, (sk, _, col) in enumerate(vet_sys):
            v = fn(s_scores.get(sk, {}))
            hb = max(1.0, v * (ry1 - ry0))
            cv.rect(gx + si * (rbw + 2.5), ry1 - hb, rbw, hb, fill=col, stroke="#1E293B", stroke_width=0.4)

    rlx = rx0
    for _, slbl, col in vet_sys:
        cv.rect(rlx, 392.0, 10.0, 10.0, fill=col)
        cv.text(rlx + 14.0, 400.0, slbl, 8.2, "#0F172A")
        rlx += 94.0

    p_pdf, p_svg = cv.save(out_dir / "fig4_ablation_and_veteran")
    return [str(p_pdf), str(p_svg)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-grade vector PDF & SVG figures from logged runs")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    logs_dir = REPO_ROOT / args.logs_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_latest_registry(logs_dir / "experiment_registry.jsonl")
    traj = json.loads((logs_dir / "v4_final_arena_trajectories.json").read_text(encoding="utf-8"))
    ab_data = json.loads((logs_dir / "v4_ablation_and_ood.json").read_text(encoding="utf-8"))
    vet_data = json.loads((logs_dir / "veteran_reasoning_benchmark.json").read_text(encoding="utf-8"))

    written: list[str] = []
    written.extend(build_fig1_equity_drawdown(traj, out_dir))
    written.extend(build_fig2_pareto_sharpe_dsr(runs, out_dir))
    written.extend(build_fig3_regime_breakdown(runs, out_dir))
    written.extend(build_fig4_ablation_and_veteran(ab_data, vet_data, out_dir))

    print(json.dumps({"status": "OK", "figures_written": written}, indent=2))


if __name__ == "__main__":
    main()
