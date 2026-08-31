# Forensic Integrity Audit Report: Architecture Search & Systematic Literature Review (2026-08-27)

**Audited Work Product**: `E:\mastercode\3_研究生\architecture_search_20260827\`  
**Auditor**: Forensic Integrity Auditor (`auditor_1` / `teamwork_preview_auditor`)  
**Audit Timestamp**: 2026-08-27T07:30:00Z  
**Authoritative Constraints**: `E:\mastercode\.agents\ORIGINAL_REQUEST.md` and `E:\mastercode\AGENTS.md`  
**Integrity Mode**: Benchmark Mode / Strict Academic Forensic Mode (Zero Tolerance for Hallucinations, Fake DOIs, Fake Repositories, or Data Inconsistencies)  
**Definitive Verdict**: **INTEGRITY VIOLATION (REJECTED)**

---

## 1. Executive Summary & Verdict Overview

| Audit Dimension | Core Forensic Check | Status / Result | Key Findings |
| :--- | :--- | :---: | :--- |
| **1. Literature Authenticity** | DOIs, arXiv IDs, authors, venues in `02_search_log.csv`, `03_paper_evidence_matrix.xlsx`, `references.bib` | 🔴 **FAIL (VIOLATION)** | Detected **5 mismatched/hallucinated DOIs** and **fictitious author attributions** (e.g. hijacking medical ECG DOIs for vision tasks, fabricating Piotr Dollar on orchard thinning). |
| **2. Repository Authenticity** | GitHub repository URLs, authors, licenses, modules in `04_repository_evidence_matrix.xlsx` | 🔴 **FAIL (VIOLATION)** | Detected **2 non-existent/synthetic GitHub repository URLs** (`DanFo9/MobileNetV4-PyTorch`, `YOLOv8-Magic/EMA`). |
| **3. Local Empirical Data** | Dataset facts (965 images, 5890 instances, 53.26% small, etc.) vs raw dataset & analysis JSONs | 🟡 **MIXED (PARTIAL PASS)** | Core dataset totals and S00~S09 benchmark metrics match 100% with raw logs, but **metric inconsistencies** exist between `05_current_task_diagnosis.md` and `07_architecture_candidates.md`. |
| **4. Constraint & No-Cheating** | Params $\le 2.85\text{M}$, GFLOPs $\le 10.0\text{G}$, CPU $\le 150\text{ms}$, ban on Mamba/CUDA extensions | 🟢 **PASS (CLEAN)** | Proposed `CitrusB-Seg` (B09) satisfies all 5 hard engineering constraints; Mamba and custom CUDA extensions are explicitly rejected. |
| **5. Deliverable Completeness** | All 13 mandatory files generated with required sections | 🟢 **PASS (CLEAN)** | All 13 required files and helper scripts are present, syntactically valid, and well-structured. |

---

## 2. 5-Component Forensic Audit Report

### 2.1 Component 1: Observation (Direct Forensic Findings & Raw Proof)

#### 2.1.1 Literature Authenticity Forensic Observations

A systematic query across international bibliographic indices (CrossRef, IEEE Xplore, Elsevier ScienceDirect, Springer, arXiv, DBLP) revealed critical authenticity discrepancies:

| Artifact & Location | Cited Entity in Work Product | Cited Identifier / DOI | True Real-World Entity at this DOI / Identifier | Severity |
| :--- | :--- | :--- | :--- | :---: |
| `references.bib` (lines 37-46) & `03_paper_evidence_matrix.xlsx` (row 177) | Title: *Large Separable Kernel Attention for Medical and Natural Image Segmentation*<br>Authors: Lau, Kin Kwan and Meng, Yanda<br>Journal: *CMPB* 2023 | `10.1016/j.cmpb.2023.107775` | **Title: "Deep neural network technique for automated detection of ADHD and CD using ECG signal"**<br>Authors: P. Rajasekhar et al.<br>Journal: *Computer Methods and Programs in Biomedicine*, Vol. 241, 2023.<br>*(Real LSKA paper is "Large Separable Kernel Attention: Rethinking the Large Kernel Attention Design in CNN" by Kin Wai Lau, Lai-Man Po, Yasar Abbas Ur Rehman, arXiv:2309.01439, published in Expert Systems with Applications, Vol. 236, 2024)* | 🔴 **Critical Violation (Hijacked DOI)** |
| `02_search_log.csv` (line 52, Row 52) | Title: *CamoFormer: Masked Separable Attention for Camouflaged Object Detection*<br>Authors: Y. Yin; X. Zhang; Y. Sun; C. Gao; S. Ding<br>Venue: *Pattern Recognition* 2024 | `10.1016/j.patcog.2024.110321` | **Title: "Learning with incomplete labels of multisource datasets for ECG classification"**<br>Authors: Qince Li, Yang Liu, Ze Zhang, Jun Liu, Yongfeng Yuan, Kuanquan Wang, Runnan He<br>Journal: *Pattern Recognition*, Vol. 150, 110321, June 2024. | 🔴 **Critical Violation (Hijacked DOI)** |
| `02_search_log.csv` (line 99, Row 99) | Title: *Immature Green Apple Instance Segmentation in Orchard Using Deep Learning*<br>Authors: H. Kang; C. Chen<br>Venue: *COMPAG* 2020 | `10.1016/j.compag.2020.105456` | **Title: "An optimized dense convolutional neural network model for disease recognition and classification in corn leaf"**<br>Authors: Abdul Waheed et al.<br>Journal: *Computers and Electronics in Agriculture*, Vol. 175, 105456, August 2020. | 🔴 **Critical Violation (Hijacked DOI)** |
| `02_search_log.csv` (line 100, Row 100) | Title: *Real-Time Detection and Localization of Immature Citrus Fruit for Robotic Thinning*<br>Authors: **R. Xiong; C. Zheng; Y. Mao; X. Liang; P. Dollar**<br>Venue: *COMPAG* 2021 | `10.1016/j.compag.2021.106237` | **Title: "3D global mapping of large-scale unstructured orchard integrating eye-in-hand stereo vision and SLAM"**<br>Authors: Mingyou Chen, Yunchao Tang, Xiaojun Zou, Zhibin Huang, Hao Zhou, S. Chen<br>Journal: *Computers and Electronics in Agriculture*, Vol. 187, 106237, August 2021.<br>*(Piotr Dollar / P. Dollar was fabricated as co-author on agricultural citrus thinning)* | 🔴 **Critical Violation (Hijacked DOI & Fictitious Authors)** |
| `02_search_log.csv` (line 101, Row 101) | Title: *High-Precision Occluded Apple Instance Segmentation in Dense Orchards (DaSNet-v2)*<br>Authors: H. Kang; C. Chen<br>Venue: *COMPAG* 2022 | `10.1016/j.compag.2022.107058` | **Title: "Force distribution of thumb-index finger power-grasp during stable fruit grasp control"**<br>Author: Xiaojing Chen<br>Journal: *Computers and Electronics in Agriculture*, Vol. 198, 107058, July 2022. | 🔴 **Critical Violation (Hijacked DOI)** |
| `02_search_log.csv` (line 13, Row 13) | Title: *Dot Distance for Tiny Object Detection in Aerial Images*<br>Venue: *IEEE GRSL* | `10.1109/LGRS.2021.3068644` | True Paper was published in **CVPRW 2021** (IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops) by Chang Xu, Jinwang Wang, Wen Yang, Lei Yu. | 🟡 **Moderate Error (Mismatched Venue/DOI)** |

#### 2.1.2 Repository Authenticity Forensic Observations

An audit of the 14 GitHub repository URLs in `04_repository_evidence_matrix.xlsx` (and `build_excel_matrices.py`) revealed:

| Repo ID | Module / Concept | Cited Repository URL | Empirical Audit Result | Authenticity Status |
| :--- | :--- | :--- | :--- | :---: |
| **R01** | StarNet | `https://github.com/ma-xu/Rewrite-the-Stars` | Official CVPR 2024 repository by Xu Ma et al. | 🟢 Authentic |
| **R02** | MobileNetV4 | `https://github.com/DanFo9/MobileNetV4-PyTorch` | Non-existent / synthetic repository URL. Danfeng Qin is the first author, but no such official GitHub account/repo exists. Legitimate implementations exist under `d-li14/mobilenetv4.pytorch` and Google AutoML. | 🔴 **Violation (Synthetic URL)** |
| **R03** | RepNCSPELAN (YOLOv9) | `https://github.com/WongKinYiu/yolov9` | Official ECCV 2024 repository by Chien-Yao Wang et al. | 🟢 Authentic |
| **R04** | PointRend | `https://github.com/facebookresearch/detectron2` | Official FAIR Detectron2 repository. | 🟢 Authentic |
| **R05** | BiFPN | `https://github.com/google/automl/tree/master/efficientdet` | Official Google AutoML repository. | 🟢 Authentic |
| **R06** | Dynamic Snake Conv | `https://github.com/YaoleiQi/DSCNet` | Official ICCV 2023 repository by Yaolei Qi et al. | 🟢 Authentic |
| **R07** | EMA Attention | `https://github.com/YOLOv8-Magic/EMA` | Synthetic URL. "YOLOv8-Magic" is a common keyword for CSDN Chinese technical blog tutorials, not a verified organization repository. Official EMA code is published under author repos. | 🔴 **Violation (Synthetic URL)** |
| **R08** | DCNv4 | `https://github.com/OpenGVLab/DCNv4` | Official CVPR 2024 OpenGVLab repository. | 🟢 Authentic |
| **R09** | Boundary IoU | `https://github.com/bowenc0221/boundary-iou-api` | Official CVPR 2021 FAIR repository by Bowen Cheng. | 🟢 Authentic |
| **R10** | LSKA | `https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention` | Official repository by Kin Wai Lau (Steven Lau). | 🟢 Authentic |
| **R11** | BMask R-CNN | `https://github.com/hustvl/BMaskR-CNN` | Official ECCV 2020 repository by HUST VL. | 🟢 Authentic |
| **R12** | QueryDet | `https://github.com/ChenhongyiYang/QueryDet-PyTorch` | Official CVPR 2022 repository by Chenhongyi Yang. | 🟢 Authentic |
| **R13** | DySample | `https://github.com/tiny-smart/dysample` | Official ICCV 2023 repository by Zhenda Liu et al. | 🟢 Authentic |
| **R14** | SCSegamba | `https://github.com/Karl1109/SCSegamba` | Official CVPR 2025 repository by Karl et al. | 🟢 Authentic |

#### 2.1.3 Local Empirical Data Forensic Observations

Cross-checking cited figures against local dataset artifacts (`data/orange_yolo_grouped_dedup_20260820/README.md`, `1_SEVER/results/_analysis/_analysis_20260824_network_redesign/dataset_difficulty/summary.json`, `1_SEVER/results/RESULTS_INDEX.csv`, `1_SEVER/results/S_series/grouped_clean_300ep/20260827_S_RESULTS_TO_B_V2.md`):

1. **Dataset Split & Instance Counts**:
   - Total Images: **965** (Train: 676, Val: 193, Test: 96) $\rightarrow$ **100% MATCH**
   - Total Instances: **5,890** (Train: 4,120, Val: 1,181, Test: 589) $\rightarrow$ **100% MATCH**
   - Cleaned Duplicates: **7 instances removed** $\rightarrow$ **100% MATCH**
   - COCO Small Area ($<1024\text{ px}^2$): **3,137 instances (53.26%)** $\rightarrow$ **100% MATCH**
   - Low Color Contrast ($\Delta E_{\text{Lab}} < 10$): **675 instances (11.46%)** $\rightarrow$ **100% MATCH**
   - Heavy Color Contrast ($\Delta E_{\text{Lab}} < 15$): **2,415 instances (41.00%)** $\rightarrow$ **100% MATCH**

2. **Cross-Document Metric Divergence**:
   - In `05_current_task_diagnosis.md` (lines 36-44), the statistics match `summary.json` exactly:
     - Solidity $< 0.85$: **1,037 instances (17.61%)**
     - Nearest Neighbor Gap $\le 4\text{ px}$: **2,082 instances (35.35%)**
     - Scale Ratio Mean: **24.30x** (Median: 7.22, P90: 60.03)
   - However, in `07_architecture_candidates.md` (lines 18-20), old preliminary figures are still cited:
     - Cited "Solidity $< 0.85$ 占 22.99%" (diverges from 17.61%)
     - Cited "间距 $\le 4\text{px}$ 占 11.10%" (diverges from 35.35%)
     - Cited "单图面积比均值 19.46x" (diverges from 24.30x)

3. **S-Series Benchmark Accuracy**:
   All cited S00~S09 accuracy values in `05_current_task_diagnosis.md`, `07_architecture_candidates.md`, and `08_final_architecture_recommendation.md` match `RESULTS_INDEX.csv` and `20260827_S_RESULTS_TO_B_V2.md` to within 4 decimal places:
   - S00 Reference: Mask AP50 = 0.7859, Mask AP50-95 = 0.6074, P = 0.8663, R = 0.7138
   - S01 RepContext: Mask AP50 = 0.7894, Mask AP50-95 = 0.6124, P = 0.8588, R = 0.7265
   - S02 LSKA: Mask AP50 = 0.7791, Mask AP50-95 = 0.6074, P = 0.8885, R = 0.7020
   - S04 Lite Head: Mask AP50 = 0.7899, Mask AP50-95 = 0.6150, P = 0.8974, R = 0.7155
   - S09 Dense Topology: Mask AP50 = 0.7843, Mask AP50-95 = 0.6162, P = 0.9143, R = 0.6868

#### 2.1.4 Engineering Constraints & No-Cheating Observations

- **Parameter Budget**: B09 `CitrusB-Seg` is 2.697M (Budget: $\le 2.85\text{M}$) $\rightarrow$ **PASS** ($-5.4\%$ margin).
- **Compute Budget**: B09 `CitrusB-Seg` is 9.45 GFLOPs (Budget: $\le 10.0\text{G}$) $\rightarrow$ **PASS** ($-5.5\%$ margin).
- **CPU Latency**: B09 `CitrusB-Seg` measures 146.6 ms / median 147.43 ms (Budget: $\le 150\text{ms}$) $\rightarrow$ **PASS**.
- **GPU Latency**: B09 `CitrusB-Seg` measures 6.8 ms (Budget: $\le 8.0\text{ms}$) $\rightarrow$ **PASS**.
- **Pretrained Transferability**: 96.4% COCO weight retention (Constraint: $\ge 95.0\%$) $\rightarrow$ **PASS**.
- **Forbidden Operator Ban**: Mamba / Selective Scan and custom CUDA C++ extensions (DCNv3/v4) are explicitly categorized as "STRICTLY REJECTED" in `04_repository_evidence_matrix.xlsx` and `06_negative_results_and_risks.md` $\rightarrow$ **PASS**.
- **Task Scope Compliance**: Pure RGB instance segmentation, strictly excluding RGB-D, amodal, OBB, and robotic pose heads $\rightarrow$ **PASS**.

---

### 2.2 Component 2: Logic Chain (Forensic Reasoning to Conclusion)

```
[Observation 1]: references.bib cites DOI 10.1016/j.cmpb.2023.107775 as LSKA paper.
         │
         ▼
[Cross-Check]: Querying CrossRef/ScienceDirect shows DOI 10.1016/j.cmpb.2023.107775 is an ECG/ADHD paper.
         │
         ▼
[Deduction 1]: The bibliographic entry references a hijacked/hallucinated DOI.
         │
         ▼
[Observation 2]: 02_search_log.csv rows 52, 99, 100, 101 cite DOIs pointing to corn leaf, grasp control, SLAM, and ECG papers.
         │
         ▼
[Cross-Check]: Querying COMPAG and Pattern Recognition indices confirms title/author mismatches and fabricated co-authorship (P. Dollar).
         │
         ▼
[Deduction 2]: Search log contains multiple hallucinated DOIs and fabricated author attributions.
         │
         ▼
[Observation 3]: 04_repository_evidence_matrix.xlsx contains synthetic URLs (DanFo9/MobileNetV4-PyTorch, YOLOv8-Magic/EMA).
         │
         ▼
[Rule Enforcement]: ORIGINAL_REQUEST.md explicitly states: "不得虚构论文、DOI 或 GitHub。"
                    Integrity Forensics mandate: "Trust NOTHING — verify EVERYTHING. If ANY check fails, your verdict is INTEGRITY VIOLATION and you MUST reject the work product."
         │
         ▼
[Final Assessment]: Despite stellar local empirical data accuracy (S00~S09, dataset facts) and compliant architectural design (CitrusB-Seg), the presence of fabricated DOIs, fake URLs, and literature hallucinations constitutes a direct, uncompromised INTEGRITY VIOLATION.
```

---

### 2.3 Component 3: Caveats & Scope Boundaries

1. **No Evidence of Fabricated Test Results**: The local empirical experiment tables, S-series checkpoints, and dataset statistics were NOT fabricated. They accurately reflect genuine training runs in `1_SEVER/results/`.
2. **Quality of Architectural Proposals**: The proposed `CitrusB-Seg` (B09) architecture is technically sound, adheres to all parameter/compute constraints, uses legitimate structural reparameterization mathematics, and does NOT cheat with facade code.
3. **Root Cause of Violation**: The integrity violation is isolated to **external literature metadata generation and search log curation** (hallucinated DOIs and synthetic GitHub URLs produced during rapid literature synthesis).

---

### 2.4 Component 4: Conclusion & Remediation Directives

#### Definitive Audit Verdict
**INTEGRITY VIOLATION — REJECTED**

#### Mandatory Remediation Requirements Before Re-submission:
1. **Fix DOIs in `references.bib`**:
   - Replace `@article{lau2023large}` DOI `10.1016/j.cmpb.2023.107775` with the authentic publication: *Expert Systems with Applications*, 2024, Vol. 236, or `arXiv:2309.01439`.
2. **Scrub and Correct `02_search_log.csv`**:
   - Fix Row 52 (CamoFormer) DOI from `10.1016/j.patcog.2024.110321` to authentic citation.
   - Fix Row 99 (Immature Green Apple) DOI from `10.1016/j.compag.2020.105456` to authentic citation.
   - Fix Row 100 (Immature Citrus Thinning) by removing fabricated author "P. Dollar" and correcting DOI `10.1016/j.compag.2021.106237`.
   - Fix Row 101 (DaSNet-v2) DOI from `10.1016/j.compag.2022.107058` to authentic citation.
   - Fix Row 13 (Dot Distance) venue to `CVPRW 2021`.
3. **Scrub and Correct `04_repository_evidence_matrix.xlsx`**:
   - Replace `https://github.com/DanFo9/MobileNetV4-PyTorch` with legitimate repository (e.g. `https://github.com/d-li14/mobilenetv4.pytorch` or Google AutoML).
   - Replace `https://github.com/YOLOv8-Magic/EMA` with authentic EMA repository.
4. **Harmonize Statistical Figures in `07_architecture_candidates.md`**:
   - Update lines 18-20 to reflect the audited dataset facts (Solidity $<0.85$: 17.61%, Gap $\le 4\text{px}$: 35.35%, Scale Ratio: 24.30x).

---

### 2.5 Component 5: Independent Verification Method

Any peer auditor or supervisor can independently verify all findings in this report using the following protocol:

1. **Verify Mismatched DOIs**:
   - Query DOI Resolver: `https://doi.org/10.1016/j.cmpb.2023.107775` $\rightarrow$ Resolves to "Deep neural network technique for automated detection of ADHD and CD using ECG signal".
   - Query DOI Resolver: `https://doi.org/10.1016/j.patcog.2024.110321` $\rightarrow$ Resolves to "Learning with incomplete labels of multisource datasets for ECG classification".
   - Query DOI Resolver: `https://doi.org/10.1016/j.compag.2021.106237` $\rightarrow$ Resolves to Chen et al., "3D global mapping of large-scale unstructured orchard integrating eye-in-hand stereo vision and SLAM" (confirming absence of Piotr Dollar).
2. **Verify Dataset Metrics**:
   - Inspect `E:\mastercode\1_SEVER\results\_analysis\_analysis_20260824_network_redesign\dataset_difficulty\summary.json` to verify the 5,890 instance breakdown.
3. **Verify S-Series Metrics**:
   - Inspect lines 83-92 of `E:\mastercode\1_SEVER\results\RESULTS_INDEX.csv` to confirm exact S00~S09 values.
