TGT        = presentation

LFLAGS     = 
PFLAGS     = 
M4FLAGS    = -I /usr/share/circuit_macros/ pgf.m4

TBL   = tbl
MAKE  = make
LATEX = pdflatex
PIC   = dpic
M4    = m4
GP    = gnuplot
RM    = /bin/rm


IMG_M4 := $(wildcard *.m4)
IMG_TEX := $(patsubst %.m4, %.tex, ${IMG_M4})
GRPH_GP := $(wildcard *.gnuplot)
GRPH_TEX := $(patsubst %.gnuplot, %.gnuplot.tex, ${GRPH_GP})

all: ../img pdf

pdf: $(TGT).tex $(GRPH_TEX) $(IMG_TEX)
	$(LATEX) $<

%.tex: %.m4
	$(M4) $(M4FLAGS) $^ | $(PIC) -g > $@

%.tex: %.gnuplot
	$(GP) $^

imgs:
	make -C img 

grafs:
	make -C graf

clean:
	$(RM) -f *.aux $(IMG_TEX) *.idx *.log *.out $(TGT).pdf
	make -C img clean
	make -C graf clean

edit:
	vim makefile

view:
	zathura $(TGT).pdf >&/dev/null & 
