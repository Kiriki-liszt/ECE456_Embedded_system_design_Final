//	ndk-build clean && ndk-build && adb push libs/arm64-v8a/project_basis /data/local && adb shell chmod +x /data/local/project_basis

#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

#include <arm_neon.h>

#define sigmoid(x) (1 / (1 + exp(-x)))

void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
	register	unsigned	int		i, j, x, y;	// register unsigned int
				unsigned	int		size_x;		// code motion & reduction in strength
							float	*hidden_layers, *temp, **weights, **biases;

	hidden_layers 	= (float *)malloc(sizeof(float) * size * depth);
	weights			= (float **)malloc(sizeof(float *) * (depth + 1));
	biases			= (float **)malloc(sizeof(float *) * (depth + 1));

	// Set pointers for weights and biases
	// 1. Input layer
	weights[0] = network;
	biases[0] = weights[0] + size * IMG_SIZE;

	// 2. Hidden layers
	// layer 1
	weights[1] = network + (size * IMG_SIZE + size);
	biases[1] = weights[1] + size * size;
	// layer 2
	weights[2] = network + (size * IMG_SIZE + size) + (size * size + size);
	biases[2] = weights[2] + size * size;

	// 3. Output layer
	weights[depth] = weights[depth - 1] + size * size + size;
	biases[depth] = weights[depth] + DIGIT_COUNT * size;

	float * input = images;						// reduction in strength

	// Recognize numbers
	for(i = 0; i < IMG_COUNT; i++)
	{
		// From the input layer to the first hidden layer
		size_x = 0;
		for(x = 0; x < size; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			SSUM  = vdupq_n_f32(0.0);
			for(y = 0; y < IMG_SIZE; y += 4)
			{
				Avec	= vld1q_f32(input + y);
				Bvec	= vld1q_f32(weights[0] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
			}
			float sum = biases[0][x] + vaddvq_f32(SSUM);
			hidden_layers[x] = sigmoid(sum);
			size_x += IMG_SIZE;
		}

		// Between hidden layers
		// layer 1
		size_x = 0;
		for(x = 0; x < size; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			SSUM  = vdupq_n_f32(0.0);
			for(y = 0; y < size; y += 4)
			{
				Avec	= vld1q_f32(hidden_layers + y);
				Bvec	= vld1q_f32(weights[1] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
			}
			float sum = biases[1][x] + vaddvq_f32(SSUM);
			hidden_layers[size + x] = sigmoid(sum);
			size_x += size;
		}
		// layer 2
		size_x = 0;
		for(x = 0; x < size; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			SSUM  = vdupq_n_f32(0.0);
			for(y = 0; y < size; y += 4)
			{
				Avec	= vld1q_f32(hidden_layers + size + y);
				Bvec	= vld1q_f32(weights[2] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
			}
			float sum = biases[2][x] + vaddvq_f32(SSUM);
			hidden_layers[size * 2 + x] = sigmoid(sum);
			size_x += size;
		}
		
		// From the last hidden layer to the output layer
		// Find the answer
		// Loop jamming
		float output[DIGIT_COUNT];
		float max = 0;
		int label = 0;
		size_x = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			SSUM  = vdupq_n_f32(0.0);
			for(y = 0; y < size; y += 4)
			{
				Avec	= vld1q_f32(hidden_layers + size * (depth-1) + y);
				Bvec	= vld1q_f32(weights[3] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
			}
			float sum = biases[depth][x] + vaddvq_f32(SSUM);
			output[x] = sigmoid(sum);
			size_x += size;

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
