"""Render Follow_up_Results_Seeds_and_Protocols.md to a PDF (A4, tables wrapped to page width) with Python-Markdown + headless Google Chrome.
   pip install markdown && python export_results_pdf.py"""
import subprocess, tempfile
from pathlib import Path
import markdown

root = Path(__file__).resolve().parent
html = markdown.markdown((root / "FOLLOWUP_RESULTS_SHAREABLE.md").read_text(), extensions=["tables"])
import re
html = re.sub(r"<table>(.*?)</table>", lambda m: ("<table class='long'>" if m.group(1).count("<tr>") > 20 else "<table>") + m.group(1) + "</table>", html, flags=re.S)
css = """@page{size:A4;margin:14mm}body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;margin:0}
table{border-collapse:collapse;margin:6px 0 16px;width:100%;font-size:8.5pt;break-inside:avoid}
thead{display:table-header-group}table.long{break-inside:auto}h1{page-break-before:always}h1:first-of-type{page-break-before:avoid}h2,h3{break-after:avoid}p{margin:6px 0}th,td{border:1px solid #bbb;padding:3px 5px;text-align:left;vertical-align:top}
th{background:#f0f0f0}tr{page-break-inside:avoid}code{font-size:9pt}h1{font-size:17pt;border-bottom:1px solid #999;padding-bottom:3px}h2{font-size:14pt;margin-top:18px}h3{font-size:12pt}"""
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "r.html"; f.write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>")
    out = root / "Follow_up_Results_Seeds_and_Protocols.pdf"
    subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", f"file://{f}"], check=True, capture_output=True)
print("wrote", out)
