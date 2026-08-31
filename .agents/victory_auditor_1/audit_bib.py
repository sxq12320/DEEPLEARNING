import re

bib_path = r'E:\mastercode\3_研究生\architecture_search_20260827\references.bib'
with open(bib_path, 'r', encoding='utf-8') as f:
    bib_text = f.read()

entries = re.findall(r'@(\w+)\s*\{([^,]+),([\s\S]*?)(?=\n@|\Z)', bib_text)
print(f'Total bib entries parsed: {len(entries)}')

keys = []
dois = []
arxivs = []
titles = []
for entry_type, cite_key, body in entries:
    k = cite_key.strip()
    keys.append(k)
    doi_m = re.search(r'doi\s*=\s*[\"{]([^\"}]+)[\"}]', body, re.IGNORECASE)
    arxiv_m = re.search(r'(?:arxiv|eprint)\s*=\\s*[\"{]([^\"}]+)[\"}]', body, re.IGNORECASE)
    title_m = re.search(r'title\s*=\s*[\"{]([^\"}]+)[\"}]', body, re.IGNORECASE)
    
    t = title_m.group(1) if title_m else 'NO TITLE'
    titles.append((k, t))
    if doi_m:
        dois.append((k, doi_m.group(1).strip()))
    if arxiv_m:
        arxivs.append((k, arxiv_m.group(1).strip()))

print(f'Parsed {len(keys)} entries, {len(dois)} DOIs, {len(arxivs)} arXiv records.')
for k, d in dois:
    print(f'  [DOI] {k:25s} -> {d}')
for k, a in arxivs:
    print(f'  [arXiv] {k:25s} -> {a}')
