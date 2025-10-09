
LDFLAGS+=-ltiff -lpng -ljpeg
LDFLAGS+=-lfftw3 -lfftw3f -lm

OPTIM+=-O3 -march=native
DEBUG=-g -Wall -Wextra
CFLAGS+=${OPTIM} ${DEBUG}
CXXFLAGS+=${OPTIM} ${DEBUG} -std=c++11 -pedantic
LDFLAGS+=${OPTIM} ${DEBUG}

# The following conditional statement appends "-std=gnu99" to CFLAGS when the
# compiler does not define __STDC_VERSION__.  The idea is that many older
# compilers are able to compile standard C when given that option.
# This hack seems to work for all versions of gcc, clang and icc.
CVERSION = $(shell $(CC) -dM -E - < /dev/null | grep __STDC_VERSION__)
ifeq ($(CVERSION),)
CFLAGS+=-std=gnu99
endif

# use OpenMP only if not clang
ifeq ($(shell $(CXX) -v 2>&1 | grep -c "clang"), 0)
OPTIM+=-fopenmp
LDFLAGS+=-lfftw3_omp -lfftw3f_omp
endif

#FLT_FLAGS=-Dflt=float -Dflt_fmt="\"%f\"" -DNUM_SINGLE
FLT_FLAGS=-Dflt=double -Dflt_fmt="\"%lf"\"

CFLAGS+=${FLT_FLAGS}
CXXFLAGS+=${FLT_FLAGS}

CXXFLAGS+=-DFFTW_HAS_THREADS

all: main

TV=tvdeconv_20120607/basic.o tvdeconv_20120607/tvreg.o
CFLAGS+=-DTVREG_DECONV=1

OBJ=main.o \
	angleSet.o \
	iio.o \
	${TV}

main: ${OBJ}
	$(CXX) -o $@ $^ ${LDFLAGS} ${OPTIM}

iio.o: iio.c

clean:
	rm -f ${OBJ}

