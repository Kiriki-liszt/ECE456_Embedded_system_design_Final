//	ndk-build clean && ndk-build && adb push libs/arm64-v8a/project_basis /data/local && adb shell chmod +x /data/local/project_basis

#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

#include <arm_neon.h>

#define sigmoid(x) (1 / (1 + exp(-x)))

// #define macros
#define size_512 512
#define depth_3 3
#define size_depth 1536					// size * depth
#define size_depth_m1 1024				// size * (depth-1)
#define size_DIGIT_COUNT 5120			// size * DIGIT_COUNT
#define size_size 262144				// size * size
#define size_size_addsize 262656		// size * size + size
#define size_size_addsize_x2 525312		// 2 * ( size * size + size )
#define size_IMG_SIZE 401408			// size * IMAGE_SIZE
#define size_IMG_SIZE_addsize 401920	// size * IMAGE_SIZE + size
#define sof 6144			// __SIZEOF_FLOAT__ * size * depth
#define sofp 32				// __SIZEOF_POINTER__ * (depth + 1)

void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
	register	unsigned	int		i, x, y;	// register unsigned int
				unsigned	int		size_x;		// code motion & reduction in strength
							float	*hidden_layers, *temp, **weights, **biases;

	hidden_layers 	= (float * )malloc(sof);
	weights			= (float **)malloc(sofp);
	biases			= (float **)malloc(sofp);

	// Set pointers for weights and biases
	// 1. Input layer
	weights[0] = network;
	biases[0]  = weights[0] + size_IMG_SIZE;

	// 2. Hidden layers
	// layer 1
	weights[1] = network + size_IMG_SIZE_addsize;
	biases[1]  = weights[1] + size_size;
	// layer 2
	weights[2] = network + size_IMG_SIZE_addsize + size_size_addsize;
	biases[2]  = weights[2] + size_size;

	// 3. Output layer
	weights[depth_3] = network + size_IMG_SIZE_addsize + size_size_addsize_x2;
	biases[depth_3]  = weights[depth_3] + size_DIGIT_COUNT;

	float * wghts0 = weights[0], * wghts1 = weights[1], * wghts2 = weights[2], * wghts3 = weights[3];
	float * input = images;

	// Recognize numbers
	for(i = 0; i < IMG_COUNT; i++)
	{
		// From the input layer to the first hidden layer
		size_x = 0;
		for(x = 0; x < size_512; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			float32x4_t Avec2, Bvec2, SSUM2;			// NEON unroll
			SSUM  = vdupq_n_f32(0.0);
			SSUM2 = vdupq_n_f32(0.0);
			for(y = 0; y < IMG_SIZE; y += 8)
			{
				Avec	= vld1q_f32(input + y);
				Bvec	= vld1q_f32(wghts0 + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(input + y + 4);
				Bvec2	= vld1q_f32(wghts0 + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM2, Avec2, Bvec2);
			}
			float sum = biases[0][x] + vaddvq_f32(SSUM) + vaddvq_f32(SSUM2);
			hidden_layers[x] = sigmoid(sum);
			size_x += IMG_SIZE;
		}

		// Between hidden layers
		// layer 1
		size_x = 0;
		for(x = 0; x < size_512; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			float32x4_t Avec2, Bvec2, SSUM2;			// NEON unroll
			SSUM  = vdupq_n_f32(0.0);
			SSUM2 = vdupq_n_f32(0.0);
			for(y = 0; y < size_512; y += 8)
			{
				Avec	= vld1q_f32(hidden_layers + y);
				Bvec	= vld1q_f32(wghts1 + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(hidden_layers + y + 4);
				Bvec2	= vld1q_f32(wghts1 + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM, Avec, Bvec);
			}
			float sum = biases[1][x] + vaddvq_f32(SSUM) + vaddvq_f32(SSUM2);
			hidden_layers[size_512 + x] = sigmoid(sum);
			size_x += size_512;
		}
		// layer 2
		size_x = 0;
		for(x = 0; x < size_512; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			float32x4_t Avec2, Bvec2, SSUM2;			// NEON unroll
			SSUM  = vdupq_n_f32(0.0);
			SSUM2 = vdupq_n_f32(0.0);
			for(y = 0; y < size_512; y += 8)
			{
				Avec	= vld1q_f32(hidden_layers + size_512 + y);
				Bvec	= vld1q_f32(wghts2 + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(hidden_layers + size_512 + y + 4);
				Bvec2	= vld1q_f32(wghts2 + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM, Avec, Bvec);
			}
			float sum = biases[2][x] + vaddvq_f32(SSUM) + vaddvq_f32(SSUM2);
			hidden_layers[size_depth_m1 + x] = sigmoid(sum);
			size_x += size_512;
		}
		
		// From the last hidden layer to the output layer
		// Find the answer
		// Loop jamming
		float output[DIGIT_COUNT];
		size_x = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			float32x4_t Avec2, Bvec2, SSUM2;			// NEON unroll
			SSUM  = vdupq_n_f32(0.0);
			SSUM2 = vdupq_n_f32(0.0);
			for(y = 0; y < size; y += 8)
			{
				Avec	= vld1q_f32(hidden_layers + size_depth_m1 + y);
				Bvec	= vld1q_f32(wghts3 + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(hidden_layers + size_depth_m1 + y + 4);
				Bvec2	= vld1q_f32(wghts3 + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM2, Avec2, Bvec2);
			}
			float sum = biases[depth_3][x] + vaddvq_f32(SSUM) + vaddvq_f32(SSUM2);
			output[x] = sigmoid(sum);
			size_x += size_512;

			if(output[x] > confidences[i])
			{
				label[i] = x;
				confidences[i] = output[x];
			}
		}
		input += IMG_SIZE;
	}
}
