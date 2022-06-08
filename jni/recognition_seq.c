#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>
#include <arm_neon.h>

#define sigmoid(x) (1 / (1 + exp(-x)))
#define half_img 392
#define half_size 256
#define fixed_depth 3
#define fixed_size 512
#define power_size 262144
#define sss1 262656		// size * size + size
#define sss2 525312		// 2 * ( size * size + size )
#define si 401408		// size * IMAGE_SIZE
#define sis 401920		// size * IMAGE_SIZE + size
#define ds 5120			// digit_count * size
#define sd 1536			// size * depth
#define SD 1024 		// size * (depth-1)

//	ndk-build clean && ndk-build && adb push libs/arm64-v8a/recognition_seq /data/local && adb shell chmod +x /data/local/recognition_seq


void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
	register unsigned int i, x, y, SX;
	float *hidden_layers, **weights, **biases;
	register float sum;

	float32x4_t Avec, Bvec;
	float32x4_t SSUM;

	hidden_layers	= (float *) malloc(sizeof(float) * sd);
	weights			= (float **)malloc(sizeof(float *) * (fixed_depth + 1));
	biases			= (float **)malloc(sizeof(float *) * (fixed_depth + 1));

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


	float * input = images;
	float * wghts0 = weights[0], * wghts1 = weights[1], * wghts2 = weights[2], * wghts3 = weights[3];

	// Recognize numbers
	for(i = 0; i < IMG_COUNT; i++)
	{
		//float * input = images + IMG_SIZE * i;
		float output[DIGIT_COUNT];

		// From the input layer to the first hidden layer
		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			sum = biases[0][x];
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < IMG_SIZE; y+=4)
			{
				Avec = vld1q_f32(input + y);
				Bvec = vld1q_f32(wghts0 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);

			}
			sum += vaddvq_f32(SSUM);
			hidden_layers[x] = sigmoid(sum);
			SX += IMG_SIZE;
		}

		// Between hidden layers
		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			sum = biases[1][x];
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < fixed_size; y+=4)
			{	
				Avec = vld1q_f32(hidden_layers + y);
				Bvec = vld1q_f32(wghts1 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum += vaddvq_f32(SSUM);
			hidden_layers[fixed_size + x] = sigmoid(sum);
			SX += fixed_size;
		}

		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			sum = biases[2][x];
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < fixed_size; y+=4)
			{	
				Avec = vld1q_f32(hidden_layers + fixed_size + y);
				Bvec = vld1q_f32(wghts2 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum += vaddvq_f32(SSUM);
			hidden_layers[SD + x] = sigmoid(sum);
			SX += fixed_size;
		}

		// From the last hidden layer to the output layer
		SX = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			sum = biases[fixed_depth][x];
			SSUM = vdupq_n_f32(0.0);
			for(y = 0; y < fixed_size; y+=4)
			{
				Avec = vld1q_f32(hidden_layers + SD + y);
				Bvec = vld1q_f32(wghts3 + SX + y);
				SSUM = vmlaq_f32(SSUM, Avec, Bvec);
			}
			sum += vaddvq_f32(SSUM);
			output[x] = sigmoid(sum);
			SX += fixed_size;
		}


		// Find the answer
		float max = 0;
		int label = 0;
		
		for(x = 0; x < DIGIT_COUNT; x++)
		{
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
}

