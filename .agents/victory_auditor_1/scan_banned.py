import os, re, glob

base_dir = r'E:\mastercode\3_研究生\architecture_search_20260827'
all_files = glob.glob(os.path.join(base_dir, '*'))

banned_patterns = {
    'Fake DOI 107775': r'107775',
    'Fake DOI 110321': r'110321',
    'Fake DOI 105456': r'105456',
    'Fake DOI 106237': r'106237',
    'Fake DOI 107058': r'107058',
    'Fake Repo DanFo9': r'DanFo9',
    'Fake Repo YOLOv8-Magic': r'YOLOv8-Magic',
    'Obsolete stat 22.99%': r'22\.99',
    'Obsolete stat 11.10%': r'11\.10',
    'Obsolete stat 19.46x': r'19\.46'
}

print("=== SCANNING FOR FORBIDDEN / OBSOLETE PATTERNS ===")
findings = {}
for name, pat in banned_patterns.items():
    matches = []
    for fpath in all_files:
        if os.path.isdir(fpath): continue
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                for line_no, line in enumerate(fp, 1):
                    if re.search(pat, line):
                        matches.append((fname, line_no, line.strip()))
        except Exception as e:
            pass
    findings[name] = matches
    if matches:
        print(f"[FOUND] {name}: {len(matches)} occurrences")
        for m in matches[:3]:
            print(f"   {m[0]}:{m[1]} -> {m[2]}")
    else:
        print(f"[CLEAN] {name}: 0 matches")
