# file:        Makefile
# author:      Andrea Vedaldi
# description: Build everything

NAME               := blocks
VER                := 0.1.1
DIST                = $(NAME)-$(VER)

# CLFAGS options added to environment
# MEX    environment has precedence
# CC     environment has precedence

MEX                ?= mex
CC                 ?= CC

# --------------------------------------------------------------------
#                                                                Flags
# --------------------------------------------------------------------

# generic flags
CFLAGS           += -I. -Wall -gstabs+ -O3
#CFLAGS           += -Wno-overlength-strings
#CFLAGS           += -Wno-variadic-macros
CFLAGS           += -Wno-unused-function 
CFLAGS           += -Wno-long-long
LDFLAGS          +=
MEX_CFLAGS        =  -g -O  CFLAGS='$$CFLAGS $(CFLAGS)'

# Determine on the flight the system we are running on
Darwin_PPC_ARCH    := mac
Darwin_i386_ARCH   := mci
Linux_i386_ARCH    := glx
Linux_i686_ARCH    := glx
Linux_x86_64_ARCH  := g64
Linux_unknown_ARCH := glx

ARCH             := $($(shell echo `uname -sm` | tr \  _)_ARCH)

mac_BINDIR       := bin/mac
mac_CFLAGS       := -Wno-variadic-macros -D__BIG_ENDIAN__
mac_LDFLAGS      := 
mac_MEX_CFLAGS   := 
mac_MEX_SUFFIX   := mexmac

mci_BINDIR       := bin/maci
mci_CFLAGS       := -Wno-variadic-macros -D__LITTLE_ENDIAN__ -gstabs+
mci_LDFLAGS      :=
mci_MEX_CFLAGS   :=
mci_MEX_SUFFIX   := mexmaci

glx_BINDIR       := bin/glx
glx_CFLAGS       := -D__LITTLE_ENDIAN__ -std=c99
glx_LDFLAGS      := -lm
glx_MEX_CFLAGS   :=
glx_MEX_SUFFIX   := mexglx

g64_BINDIR       := bin/g64
g64_CFLAGS       := -D__LITTLE_ENDIAN__ -std=c99 -fPIC
g64_LDFLAGS      := -lm
g64_MEX_CFLAGS   :=
g64_MEX_SUFFIX   := mexa64

CFLAGS           += $($(ARCH)_CFLAGS)
LDFLAGS          += $($(ARCH)_LDFLAGS)
MEX_SUFFIX       := $($(ARCH)_MEX_SUFFIX)
MEX_CFLAGS       += $($(ARCH)_MEX_CFLAGS)
BINDIR           := $($(ARCH)_BINDIR)
BINDIST          := $(DIST)-bin

.PHONY : all
all : all-mex

# this is to make directories
.PRECIOUS: %/.dirstamp
%/.dirstamp :	
	mkdir -p $(dir $@)
	echo "I'm here" > $@

# --------------------------------------------------------------------
#                                                      Build MEX files
# --------------------------------------------------------------------

mex_src := $(shell find generics -name "*.c")
mex_tgt := $(mex_src:.c=.$(MEX_SUFFIX))

.PHONY: all-mex
all-mex : $(mex_tgt)

%.$(MEX_SUFFIX) : %.c 
	@echo "   MX '$<' ==> '$@'"
	@$(MEX) $(MEX_CFLAGS) $< -output $(@:.$(MEX_SUFFIX)=)

# --------------------------------------------------------------------
#                                                       Clean and dist
# --------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f  `find . -name '*~'`
	rm -f  `find . -name '.DS_Store'`
	rm -f  `find . -name '.gdb_history'`

.PHONY: distclean
distclean: clean
	rm -f  `find . -name "*.mexmac"`
	rm -f  `find . -name "*.mexmaci"`
	rm -f  `find . -name "*.mexglx"`
	rm -f  `find . -name "*.mexa64"`
	rm -f  `find . -name "*.mexw32"`
	rm -f  `find . -name "*.dll"`
	rm -f  $(NAME)-*.tar.gz

.PHONY: dist
dist: d := $(notdir $(CURDIR)) 
dist:
	git archive --format=zip --prefix=blocks/ v$(VER) > blocks-$(VER).zip 

.PHONY: autorights
autorights: distclean
	autorights \
	  . \
	  --recursive    \
	  --verbose \
	  --template bsds \
	  --years 2009   \
	  --authors "Brian Fulkerson and Andrea Vedaldi" \
	  --holders "Brian Fulkerson and Andrea Vedaldi" \
	  --program "Blocks"
