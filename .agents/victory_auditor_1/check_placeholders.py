import os, glob, re, sys
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r'E:\mastercode\3_研究生\architecture_search_20260827'
files = sorted(glob.glob(os.path.join(base_dir, '*.*')))

placeholder_patterns = [
    r'\bTODO\b',
    r'\bTBD\b',
    r'\bFIXME\b',
    r'\[placeholder\]',
    r'\[待填\]',
    r'\[待定\]'
]

print("=== CHECKING FOR UNRESOLVED PLACEHOLDERS IN DELIVERABLES ===")
for f in files:
    fname = os.path.basename(f)
    if fname.endswith('.xlsx') or fname.endswith('.pyc'):
        continue
    with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
    
    found_placeholders = []
    for lno, line in enumerate(lines, 1):
        for pat in placeholder_patterns:
            if re.search(pat, line, re.IGNORECASE):
                found_placeholders.append((lno, line.strip()))
    
    if found_placeholders:
        print(f"[WARN] {fname}: {len(found_placeholders)} potential placeholders:")
        for lno, ltext in found_placeholders[:3]:
            print(f"   Line {lno}: {ltext}")
    else:
        print(f"[CLEAN] {fname:42s} : {len(lines):4d} lines, 0 placeholders")
