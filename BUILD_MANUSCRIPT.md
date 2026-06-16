# Building manuscript.pdf

The phenomenology paper source is `manuscript.tex` at the repo root.

## Requirements

- LaTeX distribution with `pdflatex` (TeX Live, MiKTeX, or MacTeX)
- Run **twice** for references/table of contents

## Build

```bash
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Output: `manuscript.pdf` (gitignored; build locally).

### Windows (MiKTeX)

```powershell
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Install MiKTeX if missing: https://miktex.org/download

## Before submission

1. Cross-check claims against `knowledge/wiki/synthesis/manuscript-ledger-alignment.md`
2. Verify frozen diagnostics: `./scripts/reproduce_phenomenology_tranche.sh`
3. Bundle artifacts: `./scripts/bundle_submission_artifacts.sh`

## Canonical protocol

`knowledge/wiki/synthesis/survivor-protocol-preregistered.md`

## Methodology export template

`knowledge/wiki/synthesis/phenomenology-methodology-export.md`
