import re

bib_path = r'E:\mastercode\3_研究生\architecture_search_20260827\references.bib'
with open(bib_path, 'r', encoding='utf-8') as f:
    bib_text = f.read()

entries = re.findall(r'@(\w+)\s*\{([^,]+),([\s\S]*?)(?=\n@|\Z)', bib_text)

print(f"Total entries: {len(entries)}")
for i, (entry_type, cite_key, body) in enumerate(entries):
    title = re.search(r'title\s*=\s*[\"{]([\s\S]*?)[\"}]', body)
    author = re.search(r'author\s*=\s*[\"{]([\s\S]*?)[\"}]', body)
    year = re.search(r'year\s*=\s*[\"{](\d+)[\"}]', body)
    doi = re.search(r'doi\s*=\s*[\"{]([\s\S]*?)[\"}]', body)
    journal = re.search(r'(?:journal|booktitle)\s*=\s*[\"{]([\s\S]*?)[\"}]', body)
    note = re.search(r'note\s*=\s*[\"{]([\s\S]*?)[\"}]', body)
    
    t_str = title.group(1).replace('\n', ' ') if title else 'N/A'
    a_str = author.group(1).replace('\n', ' ') if author else 'N/A'
    y_str = year.group(1) if year else 'N/A'
    d_str = doi.group(1) if doi else (note.group(1) if note else 'N/A')
    j_str = journal.group(1).replace('\n', ' ') if journal else 'N/A'
    
    print(f"{i+1:02d}. [{cite_key.strip()}] ({y_str})")
    print(f"    Title:   {t_str}")
    print(f"    Authors: {a_str[:60]}...")
    print(f"    Venue:   {j_str}")
    print(f"    ID:      {d_str}")
