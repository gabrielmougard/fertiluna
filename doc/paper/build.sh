#!/usr/bin/env bash
# Build the FertiLuna technical report.
#
#   1. Render every PlantUML (.puml) diagram to SVG.
#   2. Convert each SVG to a vector PDF (rsvg-convert) for clean \includegraphics.
#   3. Compile main.tex with latexmk (pdflatex + biber).
#
# Requirements: plantuml, rsvg-convert (librsvg), a TeX distribution with biber.
#
# The real-screenshot overlays in figures/overlays/overlay-{1..4}.png are produced
# by the classical-CV pipeline and committed alongside this report. To regenerate
# them from the four repo-root Premom screenshots:
#
#   ( cd ../../model && ./.venv/bin/python -m fertiluna_vision_cv.cli assess \
#       --images ../real-screen-1.png ../real-screen-2.png \
#                ../real-screen-3.png ../real-screen-4.png --out /tmp/cv-assess )
#   for n in 1 2 3 4; do
#     sips -Z 1500 "/tmp/cv-assess/real-screen-${n}_overlay.png" \
#          --out "figures/overlays/overlay-${n}.png"
#   done
set -euo pipefail
cd "$(dirname "$0")"

# Make TinyTeX visible if it is installed in the user home.
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"

echo "==> [1/3] Rendering PlantUML diagrams to SVG"
for f in puml/*.puml; do
  echo "    - $f"
  plantuml -tsvg -o "../figures" "$f"
done

echo "==> [2/3] Converting SVG figures to PDF"
for f in figures/*.svg; do
  rsvg-convert -f pdf -o "${f%.svg}.pdf" "$f"
  echo "    - ${f%.svg}.pdf"
done

echo "==> [3/4] Compiling main.tex"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

echo "==> [4/4] Compressing PDF for the website (public/doc/fertiluna-paper.pdf)"
SITE_PDF="../../public/doc/fertiluna-paper.pdf"
mkdir -p "$(dirname "$SITE_PDF")"
if command -v gs >/dev/null 2>&1; then
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.6 -dPDFSETTINGS=/ebook \
     -dNOPAUSE -dQUIET -dBATCH -dDetectDuplicateImages=true \
     -dColorImageResolution=150 -dGrayImageResolution=150 \
     -dMonoImageResolution=300 -sOutputFile="$SITE_PDF" main.pdf
  echo "    - $(du -h "$SITE_PDF" | cut -f1) -> $SITE_PDF"
else
  echo "    (ghostscript not found; copying uncompressed PDF instead)"
  cp main.pdf "$SITE_PDF"
fi

echo "==> Done: $(pwd)/main.pdf"
