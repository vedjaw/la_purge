# Paper sources

Compile locally (requires a LaTeX install + `acl.sty`):

```bash
pdflatex purge_arxiv_final.tex
bibtex purge_arxiv_final
pdflatex purge_arxiv_final.tex
pdflatex purge_arxiv_final.tex
```

Pre-built PDF: `purge_arxiv_final.pdf`

Figures are referenced from `../assets/figures/` and `../images/` — copy paper figures into `assets/figures/` before compiling from this directory, or compile from the repo root with `\graphicspath{{../assets/figures/}}`.
