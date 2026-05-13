import os
import re

target_dir = r'E:\mastercode\1.coding\0_segment'

arg_pattern = re.compile(r'^([ \t]*)参数(?:说明)?\s*:', re.MULTILINE)
ret_pattern = re.compile(r'^([ \t]*)返回值(?:说明)?\s*:', re.MULTILINE)
raise_pattern = re.compile(r'^([ \t]*)异常(?:说明)?\s*:', re.MULTILINE)
yield_pattern = re.compile(r'^([ \t]*)生成(?:说明)?\s*:', re.MULTILINE)

changed_files = 0
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            new_content = arg_pattern.sub(r'\1Args:', content)
            new_content = ret_pattern.sub(r'\1Returns:', new_content)
            new_content = raise_pattern.sub(r'\1Raises:', new_content)
            new_content = yield_pattern.sub(r'\1Yields:', new_content)

            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f'Updated {filepath}')
                changed_files += 1

print(f'Total files updated: {changed_files}')
