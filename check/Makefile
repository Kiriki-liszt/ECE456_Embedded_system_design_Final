CC = gcc
#CC=path of your cross compiler
CFLAGS = -Wall -g -O2

LIBS = -lm -lrt
LDFLAGS = ${LIBS}

.PHONY: all seq clean

all: seq

seq: bin_chk

bin_chk: bin_chk.o
	${CC} $^ -o $@ ${LDFLAGS}


clean:
	rm -f bin_chk.o bin_chk


# build commands
# make clean && make && ./bin_chk