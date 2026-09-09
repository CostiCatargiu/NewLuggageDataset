#!/usr/bin/env bash
# Regenerate fig1.png / fig2.png from the TikZ sources, for the DOCX build and
# for \figpngtrue. Requires pdflatex + poppler-utils.
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import re
s=open('luggage_paper.tex').read(); i=s.find('\\begin{document}')
pre=[('\\documentclass[border=6pt]{standalone}' if l.strip().startswith('\\documentclass') else l)
     for l in s[:i].split('\n') if not any(p in l for p in ('geometry}','balance','stfloats','hyperref'))]
pre='\n'.join(pre)
# fig 1: the tikzpicture inside the \iffigpng ... \else ... \fi block
body=s[s.find('\\else',s.find('\\iffigpng')):]
a=body.find('\\begin{tikzpicture}'); b=body.find('\\end{tikzpicture}')+len('\\end{tikzpicture}')
open('_f1.tex','w').write(pre+'\n\\begin{document}\n'+body[a:b]+'\n\\end{document}\n')
# fig 2
f=open('fig_arch.tex').read()
a=f.find('\\begin{tikzpicture}'); b=f.find('\\end{tikzpicture}')+len('\\end{tikzpicture}')
open('_f2.tex','w').write(pre+'\n\\begin{document}\n'+f[a:b].replace('\\eqref{eq:gate}','{(8)}')+'\n\\end{document}\n')
PY
for n in 1 2; do
  pdflatex -interaction=nonstopmode _f$n.tex >/dev/null 2>&1
  pdftoppm -r 200 -png -singlefile _f$n.pdf fig$n
  echo "fig$n.png"
done
rm -f _f?.tex _f?.pdf _f?.aux _f?.log 2>/dev/null || true
