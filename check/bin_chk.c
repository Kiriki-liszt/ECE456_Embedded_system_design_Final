
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "recognition.h"


int main(void) {
    char a[IMG_COUNT+1][13]={'\0', }, b[IMG_COUNT+1][13]={'\0',};
	FILE *io_file;

	register int i;

	io_file = fopen("result2.out", "rb");
	fread(a[0], 5*sizeof(char), 1, io_file); 
	for (i = 0; i < IMG_COUNT; i++) {
		fread(a[i+1], 12*sizeof(char), 1, io_file); 
	}
	fclose(io_file);

	io_file = fopen("ddd1.out", "rb");
	fread(b[0], 5*sizeof(char), 1, io_file); 
	for (i = 0; i < IMG_COUNT; i++) {
		fread(b[i+1], 12*sizeof(char), 1, io_file); 
	}
	fclose(io_file);
	
	
	for(int i = 0; i < IMG_COUNT+1; i++) {
		if( strcmp(a[i], b[i]) ) {
			printf("%d: %s != %s \n", i, a[i], b[i]);
		}
	}
	

	return 0;
}
