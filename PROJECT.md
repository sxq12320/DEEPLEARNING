# Project: Immature Citrus Lightweight Instance Segmentation Architecture Search (CitrusB-Seg)

## Architecture
- Task: RGB immature citrus fruit instance segmentation for robotic bagging vision.
- Target Model: CitrusB-Seg (Candidate B, B09), Pareto-optimal lightweight nano-scale model.
- Key Modules:
  * Backbone: YOLO11n + P5 `SPPFRepContext` (Structural reparameterization with 7x7 RepConv) + `C2PSA`
  * Neck: Full PAN + P3 `CitrusScaleFusion` (Sample-adaptive cross-scale feature gating)
  * Head: `SegmentCitrusLiteBQ` (Single-block depthwise decoupled head + training-only P2/P3 boundary, query, and contrast auxiliary supervision)
- Code Layout:
  * YAML config: `0_orange_yaml/B_series/09_b09_recall_balanced_final.yaml`
  * Core modules: `ultralytics/nn/modules/citrus_topo.py`, `ultralytics/nn/modules/head.py`
  * Task parsing: `ultralytics/nn/tasks.py`
  * Deliverables directory: `3_研究生/architecture_search_20260827/`

## Feature Inventory
| # | Feature / Deliverable | Description | Milestone | Source |
|---|----------------------|-------------|-----------|--------|
| 1 | `00_research_scope.md` | Research boundaries, hardware constraints, evaluation metrics, task definition | M1 | Request & AGENTS.md |
| 2 | `01_search_strategy.md` | Multi-source literature retrieval plan (Themes A-O, venues, query logs) | M1 | Literature Search |
| 3 | `02_search_log.csv` | Full search log (80+ screened, 42 filtered, 28 deep-read) | M1 | Literature Search |
| 4 | `03_paper_evidence_matrix.xlsx` | Multi-sheet paper evidence matrix with authentic DOIs, receptive fields, metrics | M2 | Literature Matrix |
| 5 | `04_repository_evidence_matrix.xlsx` | Multi-sheet open-source repo audit (14 repos, stars, licenses, CUDA status) | M2 | Repository Audit |
| 6 | `05_current_task_diagnosis.md` | Empirical fact audit of dataset & S-series failure modes (PR drop, concavity, clusters) | M3 | Empirical Audit |
| 7 | `06_negative_results_and_risks.md` | Comprehensive analysis of failed historical approaches & deployment risks | M3 | Historical Lessons |
| 8 | `07_architecture_candidates.md` | Candidate A (Conservative), Candidate B (CitrusB-Seg), Candidate C (Dual-Stream) | M4 | Architecture Design |
| 9 | `08_final_architecture_recommendation.md` | ⭐ CitrusB-Seg full architecture, validated YAML, channel breakdown, FLOPs | M4 | Final Recommendation |
| 10 | `09_ablation_and_experiment_plan.md` | 3-seed benchmark plan, challenge subsets, ablation factorial matrix, cross-family setup | M5 | Experiment Design |
| 11 | `10_reproducibility_checklist.md` | Step-by-step reproduction guide, seeds, configs, commands, invalidation criteria | M5 | Reproducibility |
| 12 | `references.bib` | Fully authentic BibTeX database of all cited literature | M6 | BibTeX Synthesis |
| 13 | `architecture_overview.mmd` | Mermaid diagram illustrating CitrusB-Seg end-to-end dataflow & training aux | M6 | Visual Diagram |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Survey & Fact Audit | Audit dataset, code, history, 14 repos, 28+ papers | none | DONE |
| M1 | Deliverable Generation Batch 1 | `00_research_scope.md`, `01_search_strategy.md`, `02_search_log.csv`, `05_current_task_diagnosis.md`, `06_negative_results_and_risks.md` | M0 | IN_PROGRESS |
| M2 | Deliverable Generation Batch 2 (Excel & Design) | `03_paper_evidence_matrix.xlsx`, `04_repository_evidence_matrix.xlsx`, `07_architecture_candidates.md`, `08_final_architecture_recommendation.md` | M0 | IN_PROGRESS |
| M3 | Deliverable Generation Batch 3 (Plan & Artifacts) | `09_ablation_and_experiment_plan.md`, `10_reproducibility_checklist.md`, `references.bib`, `architecture_overview.mmd` | M0 | IN_PROGRESS |
| M4 | Review, Stress Testing & Forensic Audit | Verification by Reviewers, Challengers, and Forensic Auditor | M1, M2, M3 | PLANNED |
| M5 | Final Synthesis & Core Question Responses | Final synthesis answering all 12 core questions with strict evidence tiers | M4 | PLANNED |

## Interface Contracts
### Deliverables $\leftrightarrow$ Ultralytics Codebase
- All YAML files must parse cleanly with `ultralytics.nn.tasks.parse_model` with `nc=1` and `nc=80`.
- All Python scripts generating Excel matrices must use `pandas` / `openpyxl` with proper styling and headers.
- All literature entries must have verified DOIs or arXiv identifiers and valid BibTeX keys.
