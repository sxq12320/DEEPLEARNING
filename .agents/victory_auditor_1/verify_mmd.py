import sys
sys.stdout.reconfigure(encoding='utf-8')

mmd_path = r'E:\mastercode\3_研究生\architecture_search_20260827\architecture_overview.mmd'
with open(mmd_path, 'r', encoding='utf-8') as f:
    mmd_text = f.read()

lines = mmd_text.splitlines()
print(f"Total lines in architecture_overview.mmd: {len(lines)}")
print("First 15 lines:")
for l in lines[:15]:
    print(" ", l)

print("\nLast 15 lines:")
for l in lines[-15:]:
    print(" ", l)
