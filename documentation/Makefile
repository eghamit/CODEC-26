# Build the biosensor documentation PDF
main.pdf: main.tex figures/*.png
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.out *.toc

.PHONY: clean
