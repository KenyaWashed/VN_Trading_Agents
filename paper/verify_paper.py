"""Phase 5/11 Verification Script: Verifies 100% of LaTeX citations, cross-refs, inputs, figures, syntax, and log-to-table provenance."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def extract_bib_keys(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))


def extract_cited_keys(tex_text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex_text):
        for k in match.split(","):
            clean = k.strip()
            if clean:
                keys.add(clean)
    return keys


def extract_labels_and_refs(tex_text: str) -> tuple[set[str], set[str]]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex_text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex_text))
    return labels, refs


def check_environment_balance(tex_text: str) -> list[str]:
    stack: list[str] = []
    errors: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex_text):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                errors.append(f"Mismatched \\end{{{env}}} (stack top: {stack[-1] if stack else 'EMPTY'})")
            else:
                stack.pop()
    for leftover in stack:
        errors.append(f"Unclosed \\begin{{{leftover}}}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LaTeX citations, refs, figures, syntax, and log-to-table provenance")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / args.paper_dir
    logs_dir = repo_root / args.logs_dir

    tex_path = paper_dir / "main.tex"
    bib_path = paper_dir / "references.bib"
    prov_path = paper_dir / "generated_provenance.json"
    reg_path = logs_dir / "experiment_registry.jsonl"

    tex_content = tex_path.read_text(encoding="utf-8")
    bib_content = bib_path.read_text(encoding="utf-8")

    # 1. Verify \input{} targets exist and concatenate full document for label/ref checks
    input_targets = re.findall(r"\\input\{([^}]+)\}", tex_content)
    missing_inputs: list[str] = []
    full_tex = tex_content
    for target in input_targets:
        t_path = paper_dir / (target if target.endswith(".tex") else f"{target}.tex")
        if not t_path.exists():
            missing_inputs.append(target)
        else:
            full_tex += "\n" + t_path.read_text(encoding="utf-8")
    if missing_inputs:
        raise SystemExit(f"Input check FAILED! Missing \\input files: {missing_inputs}")

    # 2. Verify \includegraphics{} targets exist
    fig_targets = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_content)
    missing_figs: list[str] = []
    for ftarget in fig_targets:
        fp = paper_dir / ftarget
        if not fp.exists():
            # Allow extensionless includegraphics if .pdf/.png/.svg exists
            if not any((paper_dir / f"{ftarget}{ext}").exists() for ext in (".pdf", ".png", ".svg")):
                missing_figs.append(ftarget)
    if missing_figs:
        raise SystemExit(f"Figure check FAILED! Missing \\includegraphics files: {missing_figs}")

    # 3. Verify LaTeX environment balance + unescaped '&' outside tabular/align + no stray markdown backticks
    env_errors = check_environment_balance(full_tex)
    if env_errors:
        raise SystemExit(f"LaTeX environment balance FAILED: {env_errors}")

    in_amp_env = False
    for lineno, raw_ln in enumerate(tex_content.splitlines(), start=1):
        code_ln = raw_ln.split("%", 1)[0]
        if re.search(r"\\begin\{(tabular|tabular\*|align|align\*|eqnarray|cases)\}", code_ln):
            in_amp_env = True
        if not in_amp_env and re.search(r"(?<!\\)&", code_ln):
            raise SystemExit(f"LaTeX syntax check FAILED: unescaped '&' at main.tex:{lineno}: {raw_ln.strip()}")
        if re.search(r"`[^`\s]", code_ln) and re.search(r"[^`\s]`", code_ln):
            raise SystemExit(f"LaTeX syntax check FAILED: stray markdown backtick at main.tex:{lineno}: {raw_ln.strip()}")
        if re.search(r"\\end\{(tabular|tabular\*|align|align\*|eqnarray|cases)\}", code_ln):
            in_amp_env = False

    # 4. Verify \cite{} keys against references.bib
    bib_keys = extract_bib_keys(bib_content)
    cited_keys = extract_cited_keys(full_tex)
    missing_citations = sorted(cited_keys - bib_keys)
    if missing_citations:
        raise SystemExit(f"Citation check FAILED! Unresolved citation keys: {missing_citations}")

    # 5. Verify \ref{} / \eqref{} keys against \label{}
    labels, refs = extract_labels_and_refs(full_tex)
    missing_refs = sorted(refs - labels)
    if missing_refs:
        raise SystemExit(f"Cross-reference check FAILED! Unresolved \\ref keys: {missing_refs}")

    # 6. Load logged run IDs and recursively extract all numeric values per run_id
    def _collect_floats(node: object, out: list[float]) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            out.append(float(node))
        elif isinstance(node, dict):
            for val in node.values():
                _collect_floats(val, out)
        elif isinstance(node, list):
            for item in node:
                _collect_floats(item, out)

    logged_floats: dict[str, list[float]] = {}
    with reg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                vals: list[float] = []
                _collect_floats(obj, vals)
                logged_floats[str(obj["run_id"])] = vals

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    cells = prov.get("cells", [])
    if not cells:
        raise SystemExit("Provenance check FAILED: zero cells recorded in generated_provenance.json")

    header_markers = (
        "Method &",
        "Ablation Variant &",
        "Model / System &",
        "Variant / OOD Benchmark &",
        "Reasoning System &",
        "Falsified Hypothesis ID &",
        "Candidate Architecture &",
    )
    tables_concat = ""
    for t_file in sorted((paper_dir / "tables").glob("*.tex")):
        content = t_file.read_text(encoding="utf-8")
        tables_concat += "\n" + content
        for ln in content.splitlines():
            if "\\\\" in ln and not any(hm in ln for hm in header_markers):
                if "% run_id=" not in ln:
                    raise SystemExit(f"Table line missing '% run_id=' provenance link in {t_file.name}: {ln}")

    for cell in cells:
        rid = str(cell["run_id"])
        if rid not in logged_floats:
            raise SystemExit(f"Provenance check FAILED: run_id '{rid}' not found in experiment_registry.jsonl")
        fmt_val = str(cell["formatted"])
        if fmt_val not in tables_concat:
            raise SystemExit(f"Provenance check FAILED: formatted value '{fmt_val}' missing from generated .tex tables")
        raw_num = float(cell["raw"])
        if not any(abs(raw_num - v) <= 1e-6 for v in logged_floats[rid]):
            raise SystemExit(f"Provenance mismatch between cell {cell} and parsed numeric log {rid}")

    # 7. Optional PDF compilation if a LaTeX engine is installed on PATH
    pdf_compile_status = "compiler_not_installed_syntax_verified"
    for engine in ("tectonic", "pdflatex", "xelatex"):
        exe = shutil.which(engine)
        if exe:
            cmd = [exe, "main.tex"] if engine == "tectonic" else [exe, "-interaction=nonstopmode", "main.tex"]
            proc = subprocess.run(cmd, cwd=paper_dir, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise SystemExit(f"LaTeX compilation FAILED ({engine}):\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
            pdf_compile_status = f"compiled_ok_with_{engine}"
            break

    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "citations_checked": len(cited_keys),
                "bib_entries_available": len(bib_keys),
                "missing_citations": 0,
                "refs_checked": len(refs),
                "labels_defined": len(labels),
                "inputs_verified": len(input_targets),
                "figures_verified": len(fig_targets),
                "table_cells_verified_against_logs": len(cells),
                "pdf_compilation": pdf_compile_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
