#!/usr/bin/env bash
# Build luggage_paper.docx and luggage_paper.doc from the LaTeX source.
#
# Why this is not a plain "pandoc luggage_paper.tex -o x.docx":
#   1. The TikZ figures cannot survive the trip, so the \iffigpng switch is
#      flipped to \figpngtrue and the fig1.png / fig2.png renders are used.
#      Run make_figs.sh first if the TikZ has changed.
#   2. \input{...} is inlined, because pandoc does not follow it.
#   3. The confusion matrices are included as PDF for LaTeX; Word cannot embed
#      PDF, so the .png twins are substituted.
#   4. pandoc does NOT resolve \cite against a hand-written thebibliography:
#      it silently DROPS every citation and leaks the "{99}" width argument.
#      So citations are pre-resolved to [n] here, numbered by \bibitem order,
#      which is exactly the order LaTeX numbers them in the PDF.
#
# Requires: pandoc, libreoffice (for the legacy .doc), python3.
set -e
cd "$(dirname "$0")"

python3 - <<'PY'
import re

bib  = open('bibliography.tex').read()
keys = re.findall(r'\\bibitem\{([^}]+)\}', bib)
num  = {k: i + 1 for i, k in enumerate(keys)}

src = open('luggage_paper.tex').read().replace('\\figpngfalse', '\\figpngtrue', 1)

# ---- unwrap \twocolumn[...] -----------------------------------------------
# The title, author, affiliation and the WHOLE ABSTRACT live inside
# \twocolumn[ \begin{@twocolumnfalse} ... ], which pandoc does not understand:
# it silently discards the entire bracketed argument. Every Word export made
# before this fix was missing its title block and abstract. Unwrap it, and
# turn the centred title into \title/\author so pandoc emits a real title block.
i = src.find('\\twocolumn[')
if i != -1:
    d, j = 0, i + len('\\twocolumn')
    while j < len(src):
        if src[j] == '[':
            d += 1
        elif src[j] == ']':
            d -= 1
            if d == 0:
                break
        j += 1
    block = src[i + len('\\twocolumn['):j]
    rest  = src[j + 1:]

    block = block.replace('\\begin{@twocolumnfalse}', '').replace('\\end{@twocolumnfalse}', '')

    def grab(pattern, text):
        m = re.search(pattern, text, re.S)
        return (m.group(1).strip() if m else None), (text.replace(m.group(0), '', 1) if m else text)

    title, block = grab(r'\{\\LARGE\\bfseries\s*(.*?)\\par\}', block)
    author, block = grab(r'\{\\large\s*(.*?)\\par\}', block)
    affil, block = grab(r'\{\\small\s*(.*?)\\par\}', block)

    def flat(s):
        return re.sub(r'\s+', ' ', re.sub(r'\\\\(\[[^\]]*\])?', ' ', s or '')).strip()

    head = ''
    if title:
        head += '\\title{%s}\n' % flat(title)
        who = flat(author)
        if affil:
            who += r'\and ' + flat(affil)   # \\ collapses; \and puts it on its own line
        head += '\\author{%s}\n\\maketitle\n' % who
    # whatever is left of the block (the abstract) is kept verbatim
    block = re.sub(r'\\begin\{center\}|\\end\{center\}|\\vspace\{[^}]*\}', '', block)
    # a real "Abstract" heading: as an environment pandoc emits the body with
    # no label at all, which reads as an unmarked lead paragraph
    block = block.replace('\\begin{abstract}', '\\section*{Abstract}').replace('\\end{abstract}', '')
    src = src[:i] + head + block + rest

def expand(t):
    return re.sub(r'^\\input\{([^}]+)\}\s*$',
                  lambda m: expand(open(m.group(1) + ('' if m.group(1).endswith('.tex')
                                                      else '.tex')).read()),
                  t, flags=re.M)
src = expand(src)
src = src.replace('_cm.pdf', '_cm.png')          # Word cannot embed PDF

# ---- citations: \cite{a,b} -> [3], [7] -------------------------------------
missing = []
def cite(m):
    out = []
    for k in m.group(1).split(','):
        k = k.strip()
        if k not in num:
            missing.append(k)
            out.append('[?]')
        else:
            out.append('[%d]' % num[k])
    return ', '.join(out)
src, n_cite = re.subn(r'\\cite\{([^}]+)\}', cite, src)
if missing:
    raise SystemExit('unknown citation keys: %s' % sorted(set(missing)))

# ---- bibliography: numbered paragraphs instead of a leaked "{99}" ----------
src = re.sub(r'\\begin\{thebibliography\}\{[^}]*\}\s*(\\small)?', '', src)
src = src.replace('\\end{thebibliography}', '')
src = re.sub(r'\\bibitem\{([^}]+)\}', lambda m: '\n\n[%d]~' % num[m.group(1)], src)

open('/tmp/flat.tex', 'w').write(src)
print('  %d bibitems, %d \\cite commands resolved, %d images'
      % (len(keys), n_cite, len(re.findall(r'\\includegraphics', src))))
PY

pandoc /tmp/flat.tex -f latex -t docx \
  --resource-path=.:./TrainingGrafs:./Confusion_original_improved \
  -o luggage_paper.docx
echo "  luggage_paper.docx"

rm -rf /tmp/loprof && mkdir -p /tmp/loprof
soffice --headless -env:UserInstallation=file:///tmp/loprof \
        --convert-to doc:"MS Word 97" --outdir . luggage_paper.docx >/dev/null 2>&1
echo "  luggage_paper.doc"
