# Paper sources

| file | what it is |
|------|------------|
| `excise_tmlr.tex`, `excise_tmlr.pdf` | the current manuscript (TMLR) |
| `purge_aaai.tex`, `purge_aaai_supplementary.tex` (+ PDFs) | an earlier, shorter version, from when the method was called PURGE |

Build the current manuscript from this directory. Figures are read from
`../images/` and the bibliography from `../references.bib`.

```bash
pdflatex excise_tmlr.tex
bibtex   excise_tmlr
pdflatex excise_tmlr.tex
pdflatex excise_tmlr.tex
```

The bundled PDF uses the official TMLR style files, `tmlr.sty` and `tmlr.bst`,
which are not included; get them from
<https://github.com/JmlrOrg/tmlr-style-file> and put them in this directory.
Without them the source falls back to a plain `article` layout with `plainnat`,
and the page count differs from the TMLR layout.

Text changed in the latest revision (in response to the review of 31 October
2026) is wrapped in `\rev{...}` and shows in blue in the build above. Text from
the round before is wrapped in `\added{...}` and now prints black; build with
`"\def\showpreviousround{}\input{excise_tmlr}"` to see it in orange. The
bundled PDF is built without any marking:

```bash
pdflatex -jobname=excise_tmlr "\def\finalversion{}\input{excise_tmlr}"
bibtex   excise_tmlr
pdflatex -jobname=excise_tmlr "\def\finalversion{}\input{excise_tmlr}"
pdflatex -jobname=excise_tmlr "\def\finalversion{}\input{excise_tmlr}"
```
