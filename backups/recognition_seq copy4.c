#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

#include <arm_neon.h>
#include <pthread.h>

#define sigmoid(x) (1 / (1 + exp(-x)))

#define IMG_COUNT_pthread 12500
#define half_img 392
#define half_size 256
#define fixed_depth 3
#define fixed_size 512
#define power_size 262144
#define sss1 262656			// size * size + size
#define sss2 525312			// 2 * ( size * size + size )
#define si 401408			// size * IMAGE_SIZE
#define sis 401920			// size * IMAGE_SIZE + size
#define ds 5120				// digit_count * size
#define sd 1536				// size * depth
#define SD 1024 			// size * (depth-1)

//	ndk-build clean && ndk-build && adb push libs/arm64-v8a/recognition_seq /data/local && adb shell chmod +x /data/local/recognition_seq


typedef struct{
	float 			* images;
	float 			* network;
	int 			* labels;
	float 			* confidences;
	unsigned int	img_start;
}args;

void * func(void* arguments) {				//optimized dnn

	args* data = (args*) arguments;

	unsigned int 	img_start 		= (data->img_start) * IMG_COUNT_pthread;
	float 	* input 		= data->images + img_start * IMG_SIZE;
	float 	* network 		= data->network;
	int 	* labels 		= data->labels;
	float 	* confidences	= data->confidences;

	register unsigned int i, x, y, SX;
	float *hidden_layers, **weights, **biases;
	register float sum;

	float32x4_t Avec, Bvec;
	float32x4_t SSUM;

	hidden_layers	= (float *) malloc(__SIZEOF_FLOAT__ * sd);
	weights			= (float **)malloc(__SIZEOF_POINTER__ * (fixed_depth + 1));
	biases			= (float **)malloc(__SIZEOF_POINTER__ * (fixed_depth + 1));

	// Set pointers for weights and biases

	// 1. Input layer	
	weights[0] = network;
	biases[0]  = weights[0] + si;

	// 2. Hidden layers
	weights[1] = network + sis;
	biases[1]  = weights[1] + power_size;

	weights[2] = network + sis + sss1;
	biases[2]  = weights[2] + power_size;

	// 3. Output layer	
	weights[fixed_depth] = network + sis + sss2;
	biases[fixed_depth]  = weights[fixed_depth] + ds;

	float * wghts0 = weights[0], * wghts1 = weights[1], * wghts2 = weights[2], * wghts3 = weights[3];
	float output[DIGIT_COUNT];
	
	// Recognize numbers
	for(i = img_start; i < img_start+IMG_COUNT_pthread; i++)
	{
		// From the input layer to the first hidden layer
		SX = 0;
		// why not loop termination? -> cache hot ratio!
		for(x = 0; x < fixed_size; x++)
		{
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < IMG_SIZE; y+=4)
			{
				Avec = vld1q_f32(input + y);
				Bvec = vld1q_f32(wghts0 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum = biases[0][x] + vaddvq_f32(SSUM);
			hidden_layers[x] = sigmoid(sum);
			SX += IMG_SIZE;
		}
		// Between hidden layers
		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < fixed_size; y+=4)
			{	
				Avec = vld1q_f32(hidden_layers + y);
				Bvec = vld1q_f32(wghts1 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum = biases[1][x] + vaddvq_f32(SSUM);
			hidden_layers[fixed_size + x] = sigmoid(sum);
			SX += fixed_size;
		}
		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < fixed_size; y+=4)
			{	
				Avec = vld1q_f32(hidden_layers + fixed_size + y);
				Bvec = vld1q_f32(wghts2 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum = biases[2][x] + vaddvq_f32(SSUM);
			hidden_layers[SD + x] = sigmoid(sum);
			SX += fixed_size;
		}
		// From the last hidden layer to the output layer
				// Find the answer
		float max = 0;
		unsigned int label = 0;
		
		SX = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < fixed_size; y+=4)
			{
				Avec = vld1q_f32(hidden_layers + SD + y);
				Bvec = vld1q_f32(wghts3 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum = biases[fixed_depth][x] + vaddvq_f32(SSUM);
			output[x] = sigmoid(sum);
			SX += fixed_size;
			if(output[x] > max)
			{
				label = x;
				max = output[x];
			}
		}
		
		// Store the result
		confidences[i] = max;
		labels[i] = label;

		input += IMG_SIZE;
	}
	pthread_exit(NULL);
}


void recognition(float * images, float * network, /*int depth, int size, */ int * labels, float * confidences)
{
	pthread_t pid[4];	//pid
	args arguments[4];

	for(unsigned int img_divide = 0 ; img_divide < 4; img_divide++) {
		arguments[img_divide].img_start = img_divide;
		arguments[img_divide].images = images;
		arguments[img_divide].network = network;
		arguments[img_divide].labels = labels;
		arguments[img_divide].confidences = confidences;
		pthread_create( &pid[img_divide]/*pid*/ , NULL, &func , (void *)&arguments[img_divide] );	//multiple, like 4
	}

	for(unsigned int img_divide = 0 ; img_divide < 4; img_divide++) {
		pthread_join( pid[img_divide],  NULL );
	}
}
