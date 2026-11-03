"""Post-process latexdiff output so the diff PDF compiles:
- text moved between sections appears once as deleted and once as added, so its \\label{} would be defined twice
  (and deleted equations become align*, where labels are not allowed): drop labels inside deleted blocks;
- latexdiff sometimes comments out the closing brace of a \\DIFdel{ (a deleted \\item heading in bold): close any
  \\DIFdel{ that is still open at the end of its line.

    python scripts/revision/diff_fix_labels.py paper/diff.tex
"""

import re
import sys


def close_open_difdel(line):
    out, i = line, 0
    for m in re.finditer(r"\\DIFdel\{", line):
        d, j = 1, m.end()
        while j < len(line) and d > 0:
            c = line[j]
            if c == "\\":
                j += 2
                continue
            if c == "%":
                break
            d += c == "{"
            d -= c == "}"
            j += 1
        if d > 0:  # unclosed on this line: close it before the comment that swallowed the brace
            k = line.find("%", m.end())
            out = line[:k] + "}" * d + line[k:] if k >= 0 else line + "}" * d
            break
    return out


p = sys.argv[1]
out, inside = [], False
for line in open(p).read().split("\n"):
    if r"\DIFdelbegin" in line:
        inside = True
    if inside:
        line = re.sub(r"\\label\{[^}]*\}", "", line)
    if "MBLOCKRIGHTBRACE" in line and r"\DIFdel{" in line:
        line = close_open_difdel(line)
    if r"\DIFdelend" in line:
        inside = False
    out.append(line)
open(p, "w").write("\n".join(out))
