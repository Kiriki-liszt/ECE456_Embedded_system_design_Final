CC = gcc
#CC=path of your cross compiler
CFLAGS = -Wall -g -O0 -fopenmp -mfloat-abi=softfp -mfpu=neon

LIBS = -lm -lrt
LDFLAGS = ${LIBS}


all: seq

.PHONY: all seq clean


seq: recognition_seq

recognition_seq: recognition_seq.o main.o
	${CC} -fopenmp -mfloat-abi=softfp -mfpu=neon $^ -o $@ ${LDFLAGS}


clean:
	rm -f recognition_seq.o main.o recognition_seq 


# build commands
# make clean && make && ./recognition_seq medium.network rrr.out