# Makefile for LaTeX manuscript

MAIN = manuscript
BIB = references

.PHONY: all clean pdf view

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIB).bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).bcf $(MAIN).run.xml
	rm -f $(MAIN).pdf

view: pdf
	open $(MAIN).pdf 2>/dev/null || xdg-open $(MAIN).pdf 2>/dev/null || evince $(MAIN).pdf
