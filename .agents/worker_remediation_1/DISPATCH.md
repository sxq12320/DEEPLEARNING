# DISPATCH

## 2026-08-27T07:30:30Z

You are the Remediation Worker (teamwork_preview_worker).

MANDATORY FIRST STEP:
Read the authoritative user request at:
E:\mastercode\.agents\ORIGINAL_REQUEST.md
Also read repository rules in:
E:\mastercode\AGENTS.md
Read the full forensic audit failure evidence at:
E:\mastercode\.agents\auditor_1\handoff.md

MANDATORY INTEGRITY DIRECTIVE:
The Forensic Integrity Auditor detected specific authenticity violations. You MUST remediate every single identified issue across the deliverable files in E:\mastercode\3_研究生\architecture_search_20260827\:

1. Fix references.bib:
   - Fix @article{lau2023large}: Update title to "Large Separable Kernel Attention: Rethinking the Large Kernel Attention Design in CNN", authors to "Lau, Kin Wai and Po, Lai-Man and Rehman, Yasar Abbas Ur", journal to "Expert Systems with Applications", volume="236", pages="121359", year="2024", doi="10.1016/j.eswa.2023.121359", or arXiv="2309.01439".

2. Fix 02_search_log.csv:
   - Row 13 (Dot Distance): Update venue to CVPRW 2021 (IEEE/CVF CVPR Workshops 2021), DOI 10.1109/CVPRW53098.2021.00192.
   - Row 52 (CamoFormer): Update DOI to authentic arXiv:2401.07728 / Pattern Recognition citation.
   - Row 99 (Immature Green Apple): Update DOI to authentic COMPAG citation (e.g. Kang & Chen, CompAg Vol. 173, 105377, 2020, DOI: 10.1016/j.compag.2020.105377).
   - Row 100 (Immature Citrus Fruit): Remove fabricated author "P. Dollar". Update author list and citation to authentic agricultural paper (e.g., Rong et al., Computers and Electronics in Agriculture, Vol. 182, 106035, 2021, DOI: 10.1016/j.compag.2021.106035).
   - Row 101 (DaSNet-v2): Update DOI to authentic COMPAG citation (Kang & Chen, CompAg Vol. 191, 106556, 2021, DOI: 10.1016/j.compag.2021.106556).

3. Fix 04_repository_evidence_matrix.xlsx, 03_paper_evidence_matrix.xlsx, and build_excel_matrices.py:
   - In build_excel_matrices.py:
     * Replace R02 MobileNetV4 URL with "https://github.com/d-li14/mobilenetv4.pytorch" and author "Danfeng Qin / Google Research (Port by D-Li14)".
     * Replace R07 EMA Attention URL with "https://github.com/Gus-Code/EMA-attention-module" and author "Daliang Ouyang et al.".
     * Update LSKA paper details in Sheet 1 of 03_paper_evidence_matrix.xlsx to ESWA 2024 / arXiv:2309.01439.
   - Re-run build_excel_matrices.py to regenerate both 03_paper_evidence_matrix.xlsx and 04_repository_evidence_matrix.xlsx with 100% authentic metadata.

4. Harmonize 07_architecture_candidates.md:
   - In lines 18-20, update the cited statistics to match the audited 5,890-instance values in 05_current_task_diagnosis.md (Solidity < 0.85: 17.61%, Gap <= 4px: 35.35%, Scale Ratio Mean: 24.30x).

5. Verify all changes and ensure zero remaining synthetic URLs, zero mismatched DOIs, and perfect consistency across all 13 deliverables.

Write your handoff report to E:\mastercode\.agents\worker_remediation_1\handoff.md and send a completion message.
