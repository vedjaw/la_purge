# Paper sources

| file | what it is |
|------|------------|
| `purge_tmlr.tex`, `purge_tmlr.pdf` | the current manuscript (TMLR) |
| `purge_aaai.tex`, `purge_aaai_supplementary.tex` (+ PDFs) | an earlier, shorter version |

Build the current manuscript from this directory. Figures are read from
`../images/` and the bibliography from `../references.bib`.

```bash
pdflatex purge_tmlr.tex
bibtex   purge_tmlr
pdflatex purge_tmlr.tex
pdflatex purge_tmlr.tex
```

`tmlr.sty` and `tmlr.bst` are not included; get them from
<https://github.com/JmlrOrg/tmlr-style-file>. Without them the source falls back
to a plain `article` layout with `plainnat`, which is how the bundled PDF was
built, so its page count differs from the TMLR layout.

Text added in the latest revision is wrapped in `\added{...}` and shows in
orange by default. The bundled PDF is built without the marking:

```bash
pdflatex "\def\finalversion{}\input{purge_tmlr}"
```
