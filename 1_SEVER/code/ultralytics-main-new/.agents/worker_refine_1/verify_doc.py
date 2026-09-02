import os, sys

with open('E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md', 'r', encoding='utf-8') as f:
    text = f.read()

print('Document length (chars):', len(text))
print('Line count:', len(text.splitlines()))

checks = [
    ('Gate 1: Dry-Run & Envelopes', 'Gate 1: Dry-Run YAML Build & Capacity Envelope Gate' in text),
    ('Gate 2: CUDA Sync Requirement', 'torch.cuda.synchronize()' in text),
    ('Gate 3: Warmup-Aware Stability', 'Gate 3: Warmup-Aware 3-Epoch Smoke Convergence Gate' in text),
    ('Gate 4: Screening Protocol', 'Gate 4: 50-Epoch Fast Screening Gate' in text),
    ('E_merge: Asymmetric Ground-Truth Recall', '\\text{Cov}(M_i^{\\text{pred}}, M_j^{\\text{gt}})' in text),
    ('E_split: Fragment Purity & Min Area', 'Fragment Purity (Precision)' in text and '|c_{j,k} \\cap M_j^{\\text{gt}}| \\ge 0.05 |M_j^{\\text{gt}}|' in text),
    ('Delta Solidity: Pixel Summation', 'Pixel-Summation & True-Positive Evaluation' in text and '\\sum_{u=1}^H \\sum_{v=1}^W \\mathbb{I}(M_i^{\\text{pred}}(u, v) > 0.5)' in text),
    ('AP_tiny: COCO areaRng Hook', 'areaRng = [0, 256]' in text),
    ('Hard Cap G07 <= 3.200M', 'Params}(\\text{G07}) \\le 3.200\\text{ M}' in text),
    ('Intermediate Envelopes G00-G06', 'G03 (PID Tri-Branch Only)' in text and '[3.156\\text{ M}, 3.284\\text{ M}]' in text),
]

all_passed = True
for name, passed in checks:
    status = 'PASS' if passed else 'FAIL'
    if not passed:
        all_passed = False
    print(f'[{status}] {name}')

if not all_passed:
    sys.exit(1)
print('ALL 10 REFINEMENT CHECKS PASSED PERFECTLY!')
