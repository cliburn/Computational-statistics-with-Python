report.pdf: report.tex table.tex heatmap.png
	pdflatex report
	pdflatex report
	pdflatex report

table.tex: cases.csv ctrls.csv
	python prepare_results.py

heatmap.png: cases.csv ctrls.csv
	python prepare_results.py

cases.csv: 
	python prepare_data.py

ctrls.csv: 
	python prepare_data.py

.PHONY: all clean

all: report.pdf

clean:
	rm -f *csv *png *aux *log *png table.tex *pytxcode *pdf
