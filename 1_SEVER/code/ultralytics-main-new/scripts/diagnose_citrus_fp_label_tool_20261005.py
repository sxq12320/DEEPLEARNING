"""Click-to-label FP tool (read-only): one self-contained HTML per mode, labels exported as CSV from the browser.

Reads ``fp_<mode>_human_labels.csv`` and the raw crops written by ``diagnose_citrus_fp_gallery_20261005.py``.
Labels are kept in the browser's localStorage and exported with the original columns plus ``human_label`` and
``semantic_status=human_verified``. No image, label or model output is modified.
"""

import argparse
import csv
import json
from pathlib import Path

LABELS = [
    ("unlabeled_fruit", "漏标果", "1"),
    ("leaf", "叶片", "2"),
    ("branch", "枝条", "3"),
    ("highlight", "光斑", "4"),
    ("duplicate", "重复框", "5"),
    ("partial_fruit", "部分果", "6"),
    ("other", "其他/看不清", "7"),
]

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>误检标注 - __MODE__</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#222;color:#fff;padding:10px 16px;z-index:9}
header b{color:#ffd54f} .bar button{margin-left:8px}
.help{background:#fff;margin:12px;padding:12px 16px;border-left:4px solid #1976d2;line-height:1.7}
.row{display:flex;gap:14px;background:#fff;margin:10px 12px;padding:10px;border:3px solid transparent;align-items:center}
.row.cur{border-color:#1976d2} .row.done{background:#e8f5e9}
.row img{width:320px;height:320px;object-fit:contain;background:#000}
.meta{width:150px;font-size:13px;color:#444} .btns{display:flex;flex-direction:column;gap:6px}
.btns button{width:150px;padding:7px;font-size:14px;cursor:pointer;border:1px solid #999;background:#fafafa}
.btns button.on{background:#1976d2;color:#fff;border-color:#1976d2}
</style></head><body>
<header>__TITLE__ ｜ 已标 <b id="n">0</b> / __TOTAL__
<span class="bar"><button onclick="exportCsv()">导出 CSV（标完点这里）</button>
<button onclick="if(confirm('清空本页全部标签？')){localStorage.removeItem(KEY);location.reload()}">清空</button></span></header>
<div class="help">
<b>每一行是什么：</b>模型在这里预测了一个果（分数≥0.25），但这里<b>没有任何标注果</b>能对上，所以被算成误检。<br>
<b>左图</b>：原图，黄框是模型框的位置——<b>只看黄框里实际是什么</b>。<b>右图</b>：红色是模型预测的掩膜（仅供参考）。<br>
<b>怎么标：</b>点右侧按钮，或按数字键 1–7（自动跳到下一张）。↑↓ 键切换行。标签自动保存在浏览器里，关掉页面也不会丢。<br>
<b>判断要点：</b>黄框里确实是一个果（哪怕被叶子挡住一半）→ <b>1 漏标果</b>；叶子、叶背阴影 → 2；
同一个果已经被另一个框框住、这是第二个框 → 5；只框到了一个果的一部分 → 6；看不清 → 7。<br>
<b>标完后：</b>点上方「导出 CSV」，浏览器会下载 <code>__OUT__</code>，把它放回本页所在的文件夹即可。
</div><div id="list"></div>
<script>
const MODE="__MODE__", KEY="fp_label_"+MODE, ROWS=__ROWS__, LABELS=__LABELS__;
let saved=JSON.parse(localStorage.getItem(KEY)||"{}"), cur=0;
function render(){const L=document.getElementById("list");L.innerHTML="";
 ROWS.forEach((r,i)=>{const d=document.createElement("div");d.className="row"+(saved[i]?" done":"")+(i===cur?" cur":"");d.id="r"+i;
  d.innerHTML=`<div class="meta">#${i}<br><b>原图：</b>${r.source||r.image}<br>(${r.image}，正式划分 ${r.split||"?"})<br>分数 ${(+r.score).toFixed(3)}<br>与标注最大IoU ${(+r.max_mask_iou).toFixed(2)}</div>
  <img src="${r.raw}" title="原图（黄框）"><img src="${r.asset}" title="预测掩膜">
  <div class="btns">${LABELS.map(l=>`<button class="${saved[i]===l[0]?'on':''}" onclick="setL(${i},'${l[0]}')">${l[2]} ${l[1]}</button>`).join("")}</div>`;
  d.onclick=()=>{cur=i;mark()};L.appendChild(d)});count()}
function mark(){document.querySelectorAll(".row").forEach((e,i)=>e.classList.toggle("cur",i===cur))}
function count(){document.getElementById("n").textContent=Object.keys(saved).length}
function setL(i,v){saved[i]=v;localStorage.setItem(KEY,JSON.stringify(saved));const d=document.getElementById("r"+i);
 d.classList.add("done");d.querySelectorAll("button").forEach((b,k)=>b.classList.toggle("on",LABELS[k][0]===v));count();
 if(i===cur&&cur<ROWS.length-1){cur++;mark();document.getElementById("r"+cur).scrollIntoView({block:"center"})}}
document.addEventListener("keydown",e=>{const k=LABELS.findIndex(l=>l[2]===e.key);
 if(k>=0){setL(cur,LABELS[k][0]);e.preventDefault()}
 else if(e.key==="ArrowDown"&&cur<ROWS.length-1){cur++;mark();document.getElementById("r"+cur).scrollIntoView({block:"center"});e.preventDefault()}
 else if(e.key==="ArrowUp"&&cur>0){cur--;mark();document.getElementById("r"+cur).scrollIntoView({block:"center"});e.preventDefault()}});
function q(v){v=String(v??"");return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
function exportCsv(){const cols=__COLS__;const miss=ROWS.length-Object.keys(saved).length;
 if(miss&&!confirm("还有 "+miss+" 张没标，仍然导出？"))return;
 const lines=[cols.join(",")].concat(ROWS.map((r,i)=>cols.map(c=>c==="human_label"?q(saved[i]||""):
  c==="semantic_status"?q(saved[i]?"human_verified":"unverified"):q(r.orig[c])).join(",")));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+lines.join("\\n")],{type:"text/csv"}));
 a.download="__OUT__";a.click()}
render();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    for mode, title in (("fine", "切片视图（fine）误检"), ("global", "全图视图（global）误检")):
        source = args.run / f"fp_{mode}_human_labels.csv"
        with source.open(encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns, rows = list(reader.fieldnames), list(reader)
        mapping_file = args.run / f"fp_{mode}_source_map.csv"
        mapping = {}
        if mapping_file.exists():
            with mapping_file.open(encoding="utf-8-sig") as stream:
                mapping = {int(m["row"]): m for m in csv.DictReader(stream)}
            columns += ["source_image", "source_json", "formal_split"]
            for n, r in enumerate(rows):
                m = mapping.get(n, {})
                r.update(source_image=m.get("source_image", ""), source_json=m.get("source_json", ""),
                         formal_split=m.get("formal_split", ""))
        items = [dict(image=r["image"], score=r["score"], max_mask_iou=r["max_mask_iou"], asset=r["asset"],
                      raw=f"assets_raw/raw_{mode}_{n:03d}.jpg", source=Path(r.get("source_image", "")).name,
                      split=r.get("formal_split", ""), orig=r) for n, r in enumerate(rows)]
        missing = [i["raw"] for i in items if not (args.run / i["raw"]).exists()]
        if missing:
            raise FileNotFoundError(f"Run diagnose_citrus_fp_gallery_20261005.py first; missing {missing[:3]}")
        out_name = f"fp_{mode}_human_labels_filled.csv"
        html = (PAGE.replace("__MODE__", mode).replace("__TITLE__", title).replace("__TOTAL__", str(len(items)))
                .replace("__OUT__", out_name)
                .replace("__ROWS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
                .replace("__LABELS__", json.dumps(LABELS, ensure_ascii=False))
                .replace("__COLS__", json.dumps(columns)))
        (args.run / f"fp_{mode}_标注工具.html").write_text(html, encoding="utf-8")
        print(mode, len(items), "->", args.run / f"fp_{mode}_标注工具.html")


if __name__ == "__main__":
    main()
