"""Collect publisher/CVF bibliographic metadata. Never infer experimental validity from metadata."""

import argparse
import html
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


DOIS = {
    "learning_to_zoom_2018": "10.1007/978-3-030-01240-3_4",
    "fovea_2021": "10.1109/ICCV48922.2021.01525",
    "lzu_2023": "10.1109/CVPR52729.2023.00492",
    "esod_2025": "10.1109/TIP.2024.3501853",
    "rfla_2022": "10.1007/978-3-031-20077-9_31",
    "scalekd_detection_2023": "10.1109/CVPR52729.2023.01889",
    "elakrouchi_green_citrus_2025": "10.1016/j.atech.2025.100834",
    "lu_green_citrus_2022": "10.3389/fpls.2022.946154",
}
CVF = {
    "querydet_2022": "CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html",
    "refinemask_2021": "CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html",
    "bpr_2021": "CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html",
    "maskscoring_2019": "../content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html",
    "ranksort_2021": "ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html",
    "copypaste_2021": "CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html",
    "tood_2021": "ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html",
    "set_2025": "CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html",
    "free_kd_2024": "CVPR2024/html/Zhang_FreeKD_Knowledge_Distillation_via_Semantic_Frequency_Prompt_CVPR_2024_paper.html",
    "boundary_iou_2021": "CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html",
    "zoomnet_2022": "CVPR2022/html/Pang_Zoom_in_and_Out_A_Mixed-Scale_Triplet_Network_for_Camouflaged_CVPR_2022_paper.html",
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CitrusBibliographyAudit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def bibtex_from_crossref(key, message):
    """Export fields exactly supplied by Crossref, retaining all date records in JSON."""
    kind = "article" if message.get("type") == "journal-article" else "inproceedings"
    date = message.get("published-print") or message.get("published") or message.get("issued")
    year = date["date-parts"][0][0]
    authors = " and ".join(
        f"{author['family']}, {author.get('given', '')}" if "family" in author else author["name"]
        for author in message.get("author", []))
    fields = dict(title=message["title"][0], author=authors, year=str(year), doi=message["DOI"],
                  url=message["URL"])
    container = message.get("container-title", [])
    if container:
        fields["journal" if kind == "article" else "booktitle"] = container[0]
    for src, dst in (("volume", "volume"), ("issue", "number"), ("page", "pages"), ("article-number", "number")):
        if message.get(src):
            fields[dst] = str(message[src]).replace("-", "--") if dst == "pages" else str(message[src])
    return "@" + kind + "{" + key + ",\n" + "\n".join(
        "  " + field + " = {" + value + "}," for field, value in fields.items()) + "\n}\n"


def collect(job):
    key, kind, source = job
    result = dict(key=key, source=source, kind=kind, status="unverified")
    try:
        if kind == "crossref":
            message = json.loads(fetch(source))["message"]
            result.update(status="verified_metadata", metadata=message,
                          bibtex=bibtex_from_crossref(key, message))
        else:
            document = html.unescape(fetch(source))
            # Older CVF pages put HTML line breaks inside the official citation.
            # Strip presentation markup before extracting the BibTeX, not after export.
            document = re.sub(r"<br\s*/?>", "\n", document, flags=re.IGNORECASE)
            document = re.sub(r"</?[^>]+>", "", document)
            match = re.search(r"@InProceedings\{.*?\n\}", document, re.DOTALL | re.IGNORECASE)
            if not match:
                raise ValueError("No official BibTeX block found")
            bibtex = re.sub(r"(@InProceedings\{)[^,]+,", r"\g<1>" + key + ",", match.group())
            if not all(re.search(r"\b" + field + r"\s*=", bibtex, re.IGNORECASE)
                       for field in ("title", "author", "year")):
                raise ValueError("Incomplete official BibTeX")
            result.update(status="verified_metadata", bibtex=bibtex)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior bibliographic audits")
    jobs = [(key, "crossref", "https://api.crossref.org/works/" + doi) for key, doi in DOIS.items()]
    for key, suffix in CVF.items():
        url = ("https://openaccess.thecvf.com/" + suffix[3:] if suffix.startswith("../")
               else "https://openaccess.thecvf.com/content/" + suffix)
        jobs.append((key, "cvf_bibtex", url))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(collect, jobs))
    args.output.mkdir(parents=True)
    payload = dict(accessed_utc=datetime.now(timezone.utc).isoformat(), sources=rows,
                   boundary="Bibliographic metadata only. Source table/metric/implementation claims require separate checks.")
    (args.output / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "refs.bib").write_text(
        "% Publisher/CVF metadata audited on 2026-10-06. Unverified requests are excluded.\n"
        + "\n".join(row["bibtex"] for row in rows if row["status"] == "verified_metadata"), encoding="utf-8")
    print(json.dumps([dict(key=row["key"], status=row["status"], error=row.get("error")) for row in rows],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
