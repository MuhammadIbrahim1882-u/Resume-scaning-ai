"""Refreshes the demo data embedded in index.html (used when the page is opened without the server).
Run after changing sample_data/ or the scoring code:  python build_demo.py
"""
import json
import re
from pathlib import Path

from screener import rank_resumes

root = Path(__file__).parent
base = root / "sample_data"
jd = (base / "job_description.txt").read_text(encoding="utf-8")
resumes = [{"filename": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted((base / "resumes").glob("*.txt"))]
data = {"job_description": jd, "results": rank_resumes(jd, resumes), "skipped": []}

block = '<script id="demo-data">\nwindow.SAMPLE_DEMO = ' + json.dumps(data, indent=1).replace("</", "<\\/") + ';\n</script>'
page = root / "index.html"
html = page.read_text(encoding="utf-8")
html, n = re.subn(r'<script id="demo-data">.*?</script>', lambda m: block, html, count=1, flags=re.S)
assert n == 1, "demo-data block not found in index.html"
page.write_text(html, encoding="utf-8")
print("Updated demo data in index.html")
