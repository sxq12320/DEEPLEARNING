import re, yaml

doc_path = r'E:\mastercode\3_研究生\architecture_search_20260827\08_final_architecture_recommendation.md'
with open(doc_path, 'r', encoding='utf-8') as f:
    text = f.read()

yaml_blocks = re.findall(r'```yaml([\s\S]*?)```', text)
print(f"Found {len(yaml_blocks)} YAML code blocks in 08_final_architecture_recommendation.md")

for idx, y_str in enumerate(yaml_blocks):
    try:
        parsed = yaml.safe_load(y_str)
        print(f"[OK] YAML block {idx+1} parsed successfully!")
        print("  Keys:", list(parsed.keys()))
        if 'backbone' in parsed:
            print(f"  Backbone layers: {len(parsed['backbone'])}")
        if 'head' in parsed:
            print(f"  Head layers: {len(parsed['head'])}")
    except Exception as e:
        print(f"[FAIL] YAML block {idx+1} parsing failed: {e}")
