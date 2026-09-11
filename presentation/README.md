# Energy Fingerprints pitch deck

A compact 16:9 LaTeX/Beamer deck for the Energy Data Hackdays pitch.

## Build

From this directory, run:

```sh
latexmk -pdf main.tex
```

Or compile `main.tex` twice with `pdflatex`. The deck uses only standard Beamer and TikZ packages; it deliberately does not depend on a separately installed theme.

## Updating the placeholders

Every asset slide contains an editable `confusionmatrix` command. Replace the four `TBD` values with the held-out validation counts in this order:

```tex
\\confusionmatrix{TP}{FN}{FP}{TN}
```

Use a held-out set, label the split and threshold in the caption, and remove any metric whose ground truth is not yet adequate. This matters especially for battery, heat-pump, and “other device” classifiers.

`main.tex` also marks narrative and visual placeholders with `TBD` so they can be found quickly:

```sh
rg 'TBD' main.tex
```
