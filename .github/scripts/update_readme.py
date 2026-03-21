import os
import re
from datetime import datetime

def update_readme():
    title = os.environ.get("ISSUE_TITLE", "")
    number = os.environ.get("ISSUE_NUMBER", "")
    url = os.environ.get("ISSUE_URL", "")
    created_at = os.environ.get("ISSUE_CREATED_AT", "")

    # 格式化日期
    date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%Y-%m-%d")

    # 新博客条目
    new_entry = f"- [{title}]({url}) - {date}\n"

    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    # 在 <!-- BLOG_LIST --> 标记之后插入新条目
    marker = "<!-- BLOG_LIST -->"
    if marker not in content:
        print(f"Marker '{marker}' not found in README.md")
        return

    updated = content.replace(marker, marker + "\n" + new_entry, 1)

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"Added entry: {title} (#{number})")

if __name__ == "__main__":
    update_readme()