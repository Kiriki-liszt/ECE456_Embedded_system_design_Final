//	ndk-build clean && ndk-build && adb push libs/arm64-v8a/recognition_seq /data/local && adb shell chmod +x /data/local/recognition_seq

#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

#define sigmoid(x) (1 / (1 + exp(-x)))

void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
	register int 	i, j, x, y, size_x;
			float 	*hidden_layers, *temp, **weights, **biases;

	hidden_layers	= (float *)malloc(sizeof(float) * size * depth);
	weights 		= (float **)malloc(sizeof(float *) * (depth + 1));
	biases 			= (float **)malloc(sizeof(float *) * (depth + 1));

	// Set pointers for weights and biases
	// 1. Input layer
	weights[0] = network;
	biases[0]  = weights[0] + size * IMG_SIZE;

	// 2. Hidden layers
	// layer 1
	weights[1] = network + (size * IMG_SIZE + size);
	biases[1]  = weights[1] + size * size;
	// layer 2
	weights[2] = network + (size * IMG_SIZE + size) + (size * size + size);
	biases[2]  = weights[2] + size * size;

	// 3. Output layer
	weights[depth] = weights[depth - 1] + size * size + size;
	biases[depth]  = weights[depth] + DIGIT_COUNT * size;

	float * input = images;

	// Recognize numbers
	for(i = 0; i < IMG_COUNT; i++)
	{
		float output[DIGIT_COUNT];

		// From the input layer to the first hidden layer
		size_x = 0;
		for(x = 0; x < size; x++)
		{
			float sum = 0;
			for(y = 0; y < IMG_SIZE; y++)
			{
				sum += input[y] * weights[0][size_x + y];
			}
			sum += biases[0][x];
			hidden_layers[x] = sigmoid(sum);
			size_x += IMG_SIZE;
		}

		// Between hidden layers
		// layer 1
		size_x = 0;
		for(x = 0; x < size; x++)
		{
			float sum = 0;
			for(y = 0; y < size; y++)
			{
				sum += hidden_layers[y] * weights[1][size_x + y];
			}
			sum += biases[1][x];
			hidden_layers[size + x] = sigmoid(sum);
			size_x += size;
		}
		// layer 2
		size_x = 0;
		for(x = 0; x < size; x++)
		{
			float sum = 0;
			for(y = 0; y < size; y++)
			{
				sum += hidden_layers[size + y] * weights[2][size_x + y];
			}
			sum += biases[2][x];
			hidden_layers[size * 2 + x] = sigmoid(sum);
			size_x += size;
		}

		
		// From the last hidden layer to the output layer
		size_x = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			float sum = 0;
			for(y = 0; y < size; y++)
			{
				sum += hidden_layers[size * (depth-1) + y] * weights[depth][size_x + y];
			}
			sum += biases[depth][x];
			output[x] = sigmoid(sum);
			size_x += size;

			if(output[x] > confidences[i])
			{
				labels[i]= x;
				confidences[i] = output[x];
			}
		}
		input += IMG_SIZE;		// reduction in stregnth
	}
}
