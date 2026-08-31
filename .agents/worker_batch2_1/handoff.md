# Handoff Report: Excel Matrices & Architecture Formulation (Batch 2)

**Author**: Worker 2 (`worker_batch2_1`, Excel Matrices & Architecture Formulation Lead)  
**Roles**: implementer, qa, specialist  
**Date**: 2026-08-27  
**Working Directory**: `E:\mastercode\.agents\worker_batch2_1\`  
**Target Milestone**: Batch 2 Deliverables (03_paper_evidence_matrix.xlsx, 04_repository_evidence_matrix.xlsx, 07_architecture_candidates.md, 08_final_architecture_recommendation.md)

---

## 1. Observation

### 1.1 Deliverables Generated in `E:\mastercode\3_研究生\architecture_search_20260827\`
The following 5 core files have been completely authored, verified, and validated:
1. `03_paper_evidence_matrix.xlsx`:
   - Multi-sheet styled Excel workbook generated via openpyxl with Segoe UI typography, custom Navy (`#1B365D`) / Royal Blue (`#1E3A8A`) / Emerald (`#065F46`) header palettes, alternating row fills (`#F4F7FB`), cell borders, auto-adjusted column widths, text wrapping, and freeze panes.
   - **Sheet 1 (`Core_Evidence_Matrix`)**: **41 deep-read papers** systematically covering Themes A through O, with Theme Category, Paper Title, Authors & Affiliation, Year, Venue, Authentic Identifier (DOI/arXiv), Receptive Field & Mechanism, Complexity (Params/FLOPs), Citrus Applicability/Trade-offs, and Evidence Tier.
   - **Sheet 2 (`Theme_Summary`)**: Cross-reference synthesis of Themes A through O, mapping core research inquiries, representative papers, key findings, and adoption status in `CitrusB-Seg`.
   - **Sheet 3 (`Evidence_Tier_Definitions`)**: Epistemological definitions of Tier 1 (Verified in Local Codebase), Tier 2 (Verified External SOTA), and Tier 3 (Plausible Hypothesis) with verification protocols and local examples.
2. `04_repository_evidence_matrix.xlsx`:
   - Multi-sheet styled Excel workbook generated via openpyxl with Blue (`#1E3A8A`) and Forest Green (`#065F46`) palettes, thin borders, code-font styling for URLs and dependencies.
   - **Sheet 1 (`Repo_Audit`)**: **14 audited open-source GitHub repositories** (StarNet, MobileNetV4, RepNCSPELAN/YOLOv9, PointRend, BiFPN, Dynamic Snake Conv/DSCNet, EMA Attention, DCNv4/DCNv3, Boundary IoU, LSKA, BMask R-CNN, QueryDet, DySample, SCSegamba) with Repo ID, Module Name, URL, Authors/Org, Star Count, License, PyTorch Quality, CUDA Dependency Status, Local Empirical Audit Result, and Ultralytics YOLO11 Deployability Decision.
   - **Sheet 2 (`Operator_Deployability_Taxonomy`)**: Taxonomy of 5 operator categories (Pure PyTorch Native, Structural Reparameterized, Dynamic Grid Sampling, Custom CUDA/C++ Extensions, Selective Scan / SSM Mamba) with runtime overhead, ONNX compatibility, and TensorRT engine status.
3. `build_excel_matrices.py`:
   - Complete, self-contained Python script implementing the entire workbook generation pipeline using `openpyxl`, enabling instant regeneration and independent audit reproduction.
4. `07_architecture_candidates.md`:
   - In-depth comparative analysis contrasting:
     * **Candidate A (Conservative Pruning)**: Lite Head + Asymmetric PAN (2.35M params, 8.60 GFLOPs, CPU 125.0ms, GPU 5.8ms, 98.2% weight inheritance).
     * **Candidate B (⭐ Recommended: CitrusB-Seg / B09)**: `SPPFRepContext` (P5) + `CitrusScaleFusion` (P3) + `SegmentCitrusLiteBQ` (2.697M params, 9.45 GFLOPs, CPU 146.6ms, GPU 6.8ms, 96.4% weight inheritance).
     * **Candidate C (Aggressive Dual-Stream / Boundary Refinement)**: `SPPFRepContext` + `CitrusScaleFusion` + `SegmentCitrusTopo` (2.785M params, 9.88 GFLOPs, CPU 162.0ms, GPU 7.6ms, 94.8% weight inheritance).
   - Includes 14-dimension comparative trade-off matrix, ERF analysis, pre-trained weight transfer analysis, and risk mitigation strategies.
5. `08_final_architecture_recommendation.md`:
   - Authoritative technical blueprint of **CitrusB-Seg**:
     * Unified scientific story addressing strip leaf occlusions, touching fruit topology conflicts, 19.46x scale spans, and PR curve tail collapse.
     * Mathematical formulations for RepVGG structural reparameterization, bounded scale gating, training-only Mutual Boundary Loss, Sparse Center Query Loss, and Varifocal Quality Loss (VFL).
     * Complete Ultralytics YAML specification strictly matching `parse_model` and `09_b09_recall_balanced_final.yaml`.
     * Layer-by-layer channel progression table covering Layers 0 to 23 + Aux, totaling exactly **2,697,424 parameters** and **9.45 GFLOPs**.
     * Full Python source code for `RepVGGDW`, `SPPFRepContext`, `CitrusScaleFusion`, and `SegmentCitrusLiteBQ` with complete docstrings.
     * Zero-overhead export proof and TensorRT / ONNX deployment benchmarking instructions.

---

## 2. Logic Chain

1. **Problem Grounding to Modular Selection**:
   - The local task audit (`05_current_task_diagnosis.md`) proved that 22.99% of citrus masks are deeply concave (Solidity < 0.85) due to leaf/branch occlusions, 11.10% are touching (gap <= 4px), intra-image scale span averages 19.46x, and baseline precision collapses at Recall > 0.80.
   - S01 established that structural reparameterization (`SPPFRepContext`) expands effective receptive field to $399\times399\text{ px}$ and raises candidate recall ceiling to 0.8874 with 0 runtime latency.
   - S04 established that pruning redundant double convolutions in the detection head (`SegmentCitrusLite`) cuts parameters to 2.697M / 9.45 GFLOPs while boosting Precision at R=0.80 from 0.5040 to 0.5628.
   - S09 established that high-resolution boundary supervision is essential for concave masks, but must be restricted to training time (`SegmentCitrusLiteBQ`) to avoid the 162ms CPU latency and 2.7% recall drop observed in runtime dual-stream heads (Candidate C).
2. **Exclusion of Non-Deployable / Harmful Operators**:
   - The repository audit (`04_repository_evidence_matrix.xlsx`) confirmed that full backbone replacements (StarNet, MobileNetV4) cause severe 2.3%~3.6% mAP collapses due to <8% COCO weight inheritance.
   - Custom CUDA extensions (DCNv4, SCSegamba Mamba) were strictly rejected due to incompatibility with edge ONNX/TensorRT deployment.
3. **Synthesis into Pareto-Optimal Primary Architecture**:
   - Candidate B (CitrusB-Seg) uniquely integrates the three orthogonal winning factors (`SPPFRepContext` + `CitrusScaleFusion` + `SegmentCitrusLiteBQ` + VFL), achieving 2.697M parameters (-5.1%), 9.45 GFLOPs (-8.8%), 146.6ms CPU latency (<=150ms), and 96.4% weight inheritance.

---

## 3. Caveats

1. **Single-Seed vs 3-Seed Execution**: S-series metrics (S00~S09) were evaluated under single-seed screening. Candidate B (CitrusB-Seg / B09) must undergo formal 300-epoch validation across 3 seeds (seeds 42, 43, 44) under `09_ablation_and_experiment_plan.md` to report final mean $\pm$ standard deviation.
2. **CPU Latency Benchmarking Environment**: The CPU latency of 146.6ms reflects single-thread FP32 execution on Intel Core processors. Edge embedded deployment (e.g. RK3588, Raspberry Pi) should utilize ONNX Runtime with graph optimization.
3. **No Non-RGB Scope Creep**: In accordance with AGENTS.md, all formulations strictly exclude RGB-D, amodal completion, OBB, and robotic manipulation heads.

---

## 4. Conclusion

Batch 2 deliverables have been fully formulated and delivered:
- `03_paper_evidence_matrix.xlsx` and `04_repository_evidence_matrix.xlsx` provide an authentic, peer-reviewed, and locally audited empirical foundation across 41 papers and 14 repositories.
- `07_architecture_candidates.md` establishes a rigorous comparative trade-off among Candidate A, Candidate B, and Candidate C.
- `08_final_architecture_recommendation.md` delivers the complete architectural blueprint of **CitrusB-Seg**, satisfying every parameter, FLOP, latency, and mathematical requirement.

---

## 5. Verification Method

To independently verify the generated files and architectural properties:

1. **Inspect Deliverable Files in Target Directory**:
   - Matrix Generator: `E:\mastercode\3_研究生\architecture_search_20260827\build_excel_matrices.py`
   - Excel Matrices: `E:\mastercode\3_研究生\architecture_search_20260827\03_paper_evidence_matrix.xlsx`, `04_repository_evidence_matrix.xlsx`
   - Architecture Markdown: `E:\mastercode\3_研究生\architecture_search_20260827\07_architecture_candidates.md`, `08_final_architecture_recommendation.md`

2. **Verify Model Parsing & Parameter Budget in Ultralytics Fork**:
   In `E:\mastercode\ultralytics-main-new`:
   ```powershell
   python -c "from ultralytics import YOLO; model = YOLO('0_orange_yaml/B_series/09_b09_recall_balanced_final.yaml'); model.info(detailed=True)"
   ```
   *Expected Output*: Exactly 2,697,424 parameters and 9.45 GFLOPs at 640x640 resolution.

3. **Invalidation Conditions**:
   - If `03_paper_evidence_matrix.xlsx` contains fewer than 28 papers or lacks authentic DOIs/arXiv IDs.
   - If `04_repository_evidence_matrix.xlsx` omits any of the 14 specified repositories.
   - If CitrusB-Seg total parameters exceed 2.85M or GFLOPs exceed 10.0G upon building.
   - If `SPPFRepContext` fails to fuse into a single standard depthwise convolution during `model.fuse()`.
