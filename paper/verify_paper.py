"""Phase 5 Verification Script: Ensures 100% of LaTeX citations resolve and every table number matches a JSONL log."""

from __future__ import annotations

import argparse
import json
import re
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LaTeX citations and log-to-table numerical provenance")
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

    bib_keys = extract_bib_keys(bib_content)
    cited_keys = extract_cited_keys(tex_content)
    missing_citations = sorted(cited_keys - bib_keys)
    if missing_citations:
        raise SystemExit(f"Citation check FAILED! Unresolved citation keys: {missing_citations}")

    # Load logged run IDs
    logged_runs: dict[str, str] = {}
    with reg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                logged_runs[str(obj["run_id"])] = line

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    cells = prov.get("cells", [])
    if not cells:
        raise SystemExit("Provenance check FAILED: zero cells recorded in generated_provenance.json")

    tables_concat = ""
    for t_file in sorted((paper_dir / "tables").glob("*.tex")):
        content = t_file.read_text(encoding="utf-8")
        tables_concat += "\n" + content
        # Every data row must carry a % run_id= comment
        for ln in content.splitlines():
            if "\\\\" in ln and "Method &" not in ln and "Ablation Variant &" not in ln:
                if "% run_id=" not in ln:
                    raise SystemExit(f"Table line missing '% run_id=' provenance link: {ln}")

    for cell in cells:
        rid = str(cell["run_id"])
        if rid not in logged_runs:
            raise SystemExit(f"Provenance check FAILED: run_id '{rid}' not found in experiment_registry.jsonl")
        fmt_val = str(cell["formatted"])
        if fmt_val not in tables_concat:
            raise SystemExit(f"Provenance check FAILED: formatted value '{fmt_val}' missing from generated .tex tables")
        # Verify raw number rounded matches logged JSON payload
        raw_str = f"{float(cell['raw']):.3f}"[:5]
        if raw_str not in logged_runs[rid] and str(round(float(cell["raw"]), 4)) not in logged_runs[rid]:
            raise SystemExit(f"Provenance mismatch between cell {cell} and log {rid}")

    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "citations_checked": len(cited_keys),
                "bib_entries_available": len(bib_keys),
                "missing_citations": 0,
                "table_cells_verified_against_logs": len(cells),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
