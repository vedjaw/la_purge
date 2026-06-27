# Paper sources

Compile locally (requires a LaTeX install + `acl.sty`):

```bash
pdflatex purge_arxiv_final.tex
bibtex purge_arxiv_final
pdflatex purge_arxiv_final.tex
pdflatex purge_arxiv_final.tex
```

Pre-built PDF: `purge_arxiv_final.pdf`

Figures live in `../images/`. Compile from the repo root (recommended) or set `\graphicspath{{../images/}}`.
