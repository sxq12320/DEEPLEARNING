import json, os

fact_path = r'E:\mastercode\3_研究生\architecture_search_20260827\dataset_facts_audit.json'
if os.path.exists(fact_path):
    with open(fact_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("=== DATASET FACTS AUDIT JSON ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
else:
    print("dataset_facts_audit.json not found!")
