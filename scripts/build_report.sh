#!/usr/bin/env bash
# docs/report.md -> docs/report.pdf. Needs a python3 with the `markdown` package and LibreOffice.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/build
python3 - <<'EOF'
import pathlib
import markdown
src = pathlib.Path("docs/report.md").read_text(encoding="utf-8")
body = markdown.markdown(src, extensions=["tables"])
import re
import struct


def sized(m):  # LibreOffice squashes images given only a width, so set both from the PNG header
    path = pathlib.Path("figures") / m.group(1)
    w, h = struct.unpack(">II", path.read_bytes()[16:24])
    return f'width="480" height="{round(480 * h / w)}" src="{path.resolve().as_uri()}"'


body = re.sub(r'src="\.\./figures/([^"]+)"', sized, body)
css = pathlib.Path("docs/report.css").read_text(encoding="utf-8")
html = f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>'
pathlib.Path("docs/build/report.html").write_text(html, encoding="utf-8")
EOF
soffice --headless --infilter="HTML (StarWriter)" --convert-to pdf:writer_pdf_Export --outdir docs/build docs/build/report.html >/dev/null 2>&1
mv docs/build/report.pdf docs/report.pdf
pdfinfo docs/report.pdf | grep Pages
