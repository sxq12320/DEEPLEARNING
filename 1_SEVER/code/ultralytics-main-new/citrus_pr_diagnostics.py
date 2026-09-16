"""Observed PR diagnostics from paired evaluation, without changing AP or old plots.

Only the diagnostic SVG stops at observed recall. Official AP curves/metrics
remain untouched. Every retained point stays in the CSV output. Legacy V9
reports may contain only 1000 sampled thresholds; V10 requests all thresholds.
"""

import csv
import html
import json
from pathlib import Path


def write_pr_diagnostics(report, output=None):
    report = Path(report)
    raw = json.loads(report.read_text(encoding="utf-8"))
    out = Path(output) if output is not None else report.parent / "observed_pr_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    curves = raw["empirical_mask_pr"]
    for mode in ("global", "trustedmask"):
        for cls, curve in curves[mode].items():
            recall, precision = curve["recall"], curve["precision"]
            with (out / f"{mode}_class{cls}_observed.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["confidence", "recall", "precision"])
                writer.writerows(zip(curve["confidence"], recall, precision))
            operating = {}
            for required in (0.85, 0.90, 0.95):
                ids = [i for i, p in enumerate(precision) if p >= required]
                idx = max(ids, key=lambda i: recall[i]) if ids else None
                operating[str(required)] = (
                    None
                    if idx is None
                    else dict(recall=recall[idx], precision=precision[idx], confidence=curve["confidence"][idx])
                )
            summary.append(
                dict(
                    mode=mode,
                    cls=cls,
                    targets=curve["targets"],
                    observed_rmax=curve["rmax"],
                    last_observed_precision=precision[-1] if precision else None,
                    operating_raw=operating,
                    warning="Retained raw points; legacy reports may subsample thresholds. No new AP.",
                )
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # A small vector plot; no zero tail is manufactured beyond measured recall.
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="440" viewBox="0 0 1040 440">',
        '<rect width="1040" height="440" fill="white"/>',
        '<g font-family="Arial,sans-serif" fill="#24354b">',
        '<text x="35" y="26" font-size="18">Observed precision-recall: no padded endpoint; AP unchanged</text>',
    ]
    for column, mode in enumerate(("global", "trustedmask")):
        left, top, width, height = 65 + column * 510, 65, 415, 310
        pieces.append(f'<text x="{left}" y="50" font-size="15">{mode}</text>')
        for tick in range(6):
            v = tick / 5
            x, y = left + width * v, top + height * (1 - v)
            pieces += [
                f'<path d="M {x} {top} v {height} M {left} {y} h {width}" stroke="#e2e8f0" fill="none"/>',
                f'<text x="{x - 8}" y="{top + height + 20}" font-size="11">{v:.1f}</text>',
                f'<text x="{left - 30}" y="{y + 4}" font-size="11">{v:.1f}</text>',
            ]
        for cls, curve in curves[mode].items():
            points = " ".join(
                f"{left + width * r:.2f},{top + height * (1 - p):.2f}"
                for r, p in zip(curve["recall"], curve["precision"])
            )
            pieces.append(f'<polyline points="{points}" fill="none" stroke="#176a9b" stroke-width="1.2"/>')
            pieces.append(
                f'<text x="{left + 8}" y="{top + height - 12}" font-size="12">'
                f"class {html.escape(str(cls))}; Rmax={curve['rmax']:.4f}</text>"
            )
        pieces.append(f'<text x="{left + 130}" y="420" font-size="13">Recall (not confidence)</text>')
    pieces.append("</g></svg>")
    (out / "observed_PR.svg").write_text("\n".join(pieces), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(write_pr_diagnostics(args.report, args.output))
