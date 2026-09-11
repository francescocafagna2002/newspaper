# Energy Fingerprints pitch deck

A compact 16:9 LaTeX/Beamer deck for the Energy Data Hackdays pitch.

## Build

From this directory, run:

```sh
latexmk -pdf main.tex
```

Or compile `main.tex` twice with `pdflatex`. The deck uses only standard Beamer and TikZ packages; it deliberately does not depend on a separately installed theme.

### If `sourcesans.sty` or `ly1enc.def` is not found

Debian/Ubuntu TeX Live installs often omit both. With no root access, install them
into your personal tree (`~/texmf`) instead:

```sh
cd "$(mktemp -d)"
curl -sLO https://mirrors.ctan.org/install/fonts/sourcesans.tds.zip
curl -sLO https://mirrors.ctan.org/install/fonts/psfonts/ly1.tds.zip
for z in sourcesans ly1; do unzip -q -o "$z.tds.zip" -d ~/texmf; done
mktexlsr ~/texmf
updmap-user --enable Map=SourceSansThree.map
```

`ly1` is needed because `sourcesans` loads the LY1 encoding. Without the `updmap-user`
step the deck still compiles but the Source Sans glyphs do not embed.

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
