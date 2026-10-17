"""Render the full-dataset label audit as a per-source-image HTML review page.

Reads per-image JSONs written by ``audit_labels_infer_20261005.py`` plus
``label_selfcheck.csv`` from ``audit_labels_pure_20261005.py``, then produces
one HTML grouped by source image so the user can re-label directly in
X-AnyLabeling against the original JSONs. Four finding types:

- ``missing``: model detections matching no annotation (candidate missed labels);
- ``suspicious``: annotations no model/view proposes anything near (conf floor);
- ``duplicate``: two annotations overlapping >50% (same fruit labelled twice);
- ``geometry``: degenerate / out-of-bounds / implausibly tiny polygons.

Crops are drawn on the ORIGINAL images with existing annotation outlines in
green; candidate boxes yellow with the detection contour in red; suspicious /
duplicate members outlined in red/orange. A checklist CSV mirrors every row.
Read-only: no image, JSON or label file is ever modified.
"""

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def draw_polys(image, polygons, color, thickness):
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.polylines(image, [np.round(pts).astype(np.int32)], True, color, thickness)


def crop_around(image, cx, cy, span, out_size=360):
    h, w = image.shape[:2]
    half = span / 2
    left, top = max(0, int(cx - half)), max(0, int(cy - half))
    right, bottom = min(w, int(cx + half)), min(h, int(cy + half))
    if right - left < span and left > 0:
        left = max(0, right - int(span))
    if bottom - top < span and top > 0:
        top = max(0, bottom - int(span))
    view = image[top:bottom, left:right].copy()
    scale = out_size / max(1, max(right - left, bottom - top))
    return cv2.resize(view, (max(1, int((right - left) * scale)), max(1, int((bottom - top) * scale))),
                      interpolation=cv2.INTER_AREA), (left, top, scale)


def box_center_span(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="LABEL_AUDIT dir with infer/ and selfcheck csv")
    parser.add_argument("--wuxi-root", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.audit / "audit_assets"
    out_dir.mkdir(exist_ok=True)
    images = {p.name: p for d in ("img", "img_2") for p in (args.wuxi_root / d).iterdir() if p.is_file()}

    packets = []
    for f in sorted((args.audit / "infer").glob("audit_*.json")):
        packets.append(json.loads(f.read_text(encoding="utf-8")))
    selfcheck = defaultdict(list)
    sc_path = args.audit / "label_selfcheck.csv"
    if sc_path.exists():
        for r in csv.DictReader(sc_path.open(encoding="utf-8-sig")):
            if r["json"]:
                selfcheck[r["image"]].append(r)

    checklist, blocks = [], []
    totals = Counter()
    n_done = 0
    for p in packets:
        name = p["image"]
        findings = []
        # candidate missing labels, most corroborated first
        for c in sorted(p["candidates"], key=lambda c: (-len(c["votes"]), -c["score"])):
            findings.append(("missing", c))
        # suspicious labels: nothing proposed even at conf 0.001 in any model/view;
        # hard labels: a raw candidate existed but never matched at 0.25 (likely a
        # genuinely difficult fruit, not necessarily a wrong label)
        for g in p["gts"]:
            if g["matched25"]:
                continue
            findings.append(("hard_label" if g["proposed_001"] else "suspicious", g))
        # structural flags from the pure audit
        for s in selfcheck.get(name, []):
            findings.append((s["flag"], s))
        if not findings:
            continue

        src = images.get(name)
        image = cv2.imread(str(src)) if src else None
        polys = []
        if p["json"]:
            data = json.loads(Path(p["json"]).read_text(encoding="utf-8"))
            polys = [s["points"] for s in data.get("shapes", []) if s.get("shape_type") == "polygon"]

        rows_html = []
        centers = []
        for kind, item in findings:
            totals[kind] += 1
            item_id = f"{p['stem']}#{kind}#{totals[kind]}"
            caption = ""
            if kind == "missing":
                cx, cy, span = box_center_span(item["box"])
                span = span * 2.4
            elif kind in ("suspicious", "hard_label"):
                cx, cy, span = box_center_span(item["box"])
                span = span * 2.6
            elif kind == "duplicate_pair":
                i, j = map(int, item["gt_index"].split("&"))
                pts = np.asarray(polys[i] + polys[j], float) if i < len(polys) and j < len(polys) else None
                cx, cy = pts[:, 0].mean() if pts is not None else 0, pts[:, 1].mean() if pts is not None else 0
                span = float(max(pts[:, 0].ptp(), pts[:, 1].ptp()) * 1.8) if pts is not None else 300
            else:
                gi = int(item["gt_index"])
                pts = np.asarray(polys[gi], float) if gi < len(polys) else None
                cx, cy = (pts[:, 0].mean(), pts[:, 1].mean()) if pts is not None else (0, 0)
                span = float(max(pts[:, 0].ptp(), pts[:, 1].ptp()) * 3.0) if pts is not None else 300
            centers.append((cx, cy))
            local_no = len(centers)

            crop_file = ""
            if image is not None:
                span = max(120.0, min(span, 1400.0))
                layer = image.copy()
                draw_polys(layer, polys, (0, 200, 0), max(2, int(span / 160)))
                if kind == "missing":
                    if item.get("poly"):
                        draw_polys(layer, [item["poly"]], (0, 0, 255), max(2, int(span / 200)))
                    x0, y0, x1, y1 = item["box"]
                    cv2.rectangle(layer, (int(x0), int(y0)), (int(x1), int(y1)),
                                  (0, 255, 255), max(2, int(span / 180)))
                    caption = (f"score {item['score']:.2f} · 票数 {item['n_dets']} "
                               f"({' + '.join(html.escape(v) for v in item['votes'])})"
                               f" · 与最近标注 IoU {item['best_iou']:.2f}")
                elif kind in ("suspicious", "hard_label"):
                    gi = item["index"]
                    if gi < len(polys):
                        draw_polys(layer, [polys[gi]], (0, 0, 255), max(3, int(span / 150)))
                    caption = (f"标注#{gi} · raster面积 {item['area_raster']}px · "
                               f"模型最近 IoU {item['best_iou']:.2f} · "
                               f"0.001下限{'有候选但匹配不上' if item['proposed_001'] else '无任何候选'}")
                elif kind == "duplicate_pair":
                    i, j = map(int, item["gt_index"].split("&"))
                    draw_polys(layer, [polys[i]], (0, 0, 255), max(3, int(span / 150)))
                    draw_polys(layer, [polys[j]], (0, 165, 255), max(3, int(span / 150)))
                    caption = f"标注#{i} 与 #{j} 重叠 {float(item['detail'].split('=')[1]):.2f}"
                else:
                    gi = int(item["gt_index"])
                    if gi < len(polys):
                        draw_polys(layer, [polys[gi]], (0, 0, 255), max(3, int(span / 150)))
                    caption = f"标注#{gi} · {html.escape(item['detail'])}"
                view, (left, top, sc) = crop_around(layer, cx, cy, span)
                crop_file = f"audit_assets/c{len(checklist):05d}.jpg"
                cv2.imwrite(str(args.audit / crop_file), view, [cv2.IMWRITE_JPEG_QUALITY, 82])
            checklist.append(dict(id=item_id, image=name, json=p["json"], split=p["split"],
                                  kind=kind, detail=caption.replace("·", "|"), crop=crop_file))
            rows_html.append(
                f"<tr data-kind='{kind}' data-done='0'>"
                f"<td><input type='checkbox' onchange='toggle(this)' data-id='{item_id}'></td>"
                f"<td>#{local_no}</td><td>{html.escape(name)}</td><td>{kind}</td><td>{caption}</td>"
                f"<td>{'<img src=' + repr(crop_file) + '>' if crop_file else '(图像缺失)'}</td></tr>")
        n_done += 1
        overview_file = ""
        if image is not None:
            ow = min(860, image.shape[1])
            osc = ow / image.shape[1]
            overview = cv2.resize(image, (ow, int(image.shape[0] * osc)), interpolation=cv2.INTER_AREA)
            for (kind, _item), (cx, cy) in zip(findings, centers):
                color = {"missing": (0, 0, 255), "suspicious": (255, 0, 255),
                         "hard_label": (255, 0, 0)}.get(kind, (0, 165, 255))
                ox, oy = int(cx * osc), int(cy * osc)
                cv2.circle(overview, (ox, oy), 11, color, 2)
            for n, (cx, cy) in enumerate(centers, 1):
                cv2.putText(overview, str(n), (int(cx * osc) - 8, int(cy * osc) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overview, str(n), (int(cx * osc) - 8, int(cy * osc) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            overview_file = f"audit_assets/ov_{p['stem']}.jpg"
            cv2.imwrite(str(args.audit / overview_file), overview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        blocks.append(
            f"<h3>{html.escape(name)} "
            f"<small>[{p['split']}] 标注 {p['n_gt']} 个 · JSON: {html.escape(Path(p['json']).name) if p['json'] else '无'}</small></h3>"
            + (f"<p>整图定位（红圈=候选漏标 missing，蓝圈=可疑/难例标注，橙圈=重复/几何问题），编号对应下行 # 列：</p>"
               f"<img src='{overview_file}'><br>" if overview_file else "")
            + f"<table border='1' cellpadding='4'><tr><th>已处理</th><th>#</th><th>图片</th><th>类型</th>"
              f"<th>信息</th><th>裁剪图（绿=现有标注）</th></tr>"
            + "".join(rows_html) + "</table>")

    with (args.audit / "relabel_checklist.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "json", "split", "kind", "detail", "crop"])
        writer.writeheader()
        writer.writerows(checklist)

    kinds = " ".join(f"<button onclick=\"filt('{k}')\">{k}({totals[k]})</button>" for k in
                     ("missing", "suspicious", "hard_label", "duplicate_pair", "geometry"))
    page = f"""<meta charset='utf-8'><title>标签审计</title>
<style>body{{font-family:system-ui;margin:12px}}td{{vertical-align:top}}img{{max-width:380px}}
tr[data-done='1']{{opacity:.35}}</style>
<h2>全量标签审计（{len(blocks)} 张图有发现 / 共 {len(packets)} 张）</h2>
<p>类型：missing=模型找到但没标（候选漏标）· suspicious=标注了但 conf 放到 0.001 仍无任何候选（最强嫌疑）·
hard_label=有候选但任何模型/视图都匹配不上（多为难例或标注过松）·
duplicate_pair=两个标注重叠>50%（疑似重复）· geometry=多边形异常。
<b>勾选框仅用于标记已处理（存浏览器），CSV 对照表见 relabel_checklist.csv。</b></p>
<p>{kinds} <button onclick="filt('')">全部</button> <button onclick="exportCSV()">导出勾选进度</button>
<span id="prog"></span></p>
{''.join(blocks)}
<script>
const KEY='label_audit_done';
let done = new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));
function update() {{
  document.querySelectorAll('input[data-id]').forEach(c => c.checked = done.has(c.dataset.id));
  document.querySelectorAll('tr[data-kind]').forEach(t => {{
    const c = t.querySelector('input'); if (c) t.dataset.done = done.has(c.dataset.id) ? '1' : '0';}});
  document.getElementById('prog').textContent = '已处理 ' + done.size + '/' +
    document.querySelectorAll('input[data-id]').length;
}}
function toggle(c) {{ c.checked ? done.add(c.dataset.id) : done.delete(c.dataset.id);
  localStorage.setItem(KEY, JSON.stringify([...done])); update(); }}
function filt(k) {{ document.querySelectorAll('tr[data-kind]').forEach(t =>
  t.style.display = (!k || t.dataset.kind === k) ? '' : 'none'); }}
function exportCSV() {{
  let s = 'id,done\\n'; document.querySelectorAll('input[data-id]').forEach(c =>
    s += c.dataset.id + ',' + (done.has(c.dataset.id) ? 1 : 0) + '\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([s], {{type: 'text/csv'}}));
  a.download = 'label_audit_progress.csv'; a.click(); }}
update();
</script>"""
    (args.audit / "标签审计.html").write_text(page, encoding="utf-8")
    print(f"images_with_findings={len(blocks)} findings={dict(totals)} crops={len(checklist)}")


if __name__ == "__main__":
    main()
