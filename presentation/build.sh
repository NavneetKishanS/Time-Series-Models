#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# SVG → PDF conversion (idempotent: skip if PDF is newer than SVG)
CONV=""
if command -v rsvg-convert &>/dev/null; then CONV="rsvg"; fi
if [ -z "$CONV" ] && command -v inkscape &>/dev/null; then CONV="inkscape"; fi
if [ -z "$CONV" ] && command -v cairosvg &>/dev/null; then CONV="cairosvg"; fi
if [ -z "$CONV" ]; then echo "ERROR: no SVG converter found (rsvg-convert / inkscape / cairosvg)"; exit 1; fi

mkdir -p figures
for svg in svgs/*.svg; do
  name=$(basename "$svg" .svg)
  pdf="figures/${name}.pdf"
  if [ ! -f "$pdf" ] || [ "$svg" -nt "$pdf" ]; then
    echo "Converting: $name"
    case "$CONV" in
      rsvg)     rsvg-convert -f pdf -o "$pdf" "$svg" ;;
      inkscape) inkscape --export-type=pdf --export-filename="$pdf" "$svg" ;;
      cairosvg) cairosvg "$svg" -o "$pdf" ;;
    esac
  fi
done

# LaTeX compile (two passes for cross-references)
echo "Compiling LaTeX (pass 1)..."
xelatex -interaction=nonstopmode main.tex > /dev/null
echo "Compiling LaTeX (pass 2)..."
xelatex -interaction=nonstopmode main.tex > /dev/null

echo ""
echo "Done → presentation/main.pdf"
