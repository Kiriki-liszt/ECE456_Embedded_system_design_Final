#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

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

/*
void in_2_first(float * input, float ** weights, float * hidden_layers, float ** biases) {
	// From the input layer to the first hidden layer
	float sum;
	int IX = 0;
	for(int x = 0; x < fixed_size; x++)
	{
		sum = 0;
		IX = IMG_SIZE * x;
		for(int y = 0; y < IMG_SIZE; y++)
		{
			sum += input[y] * weights[0][IX + y];
		}
		sum += biases[0][x];
		hidden_layers[x] = sigmoid(sum);
		//IX += IMG_SIZE;
	}
}

void between(float ** weights, float * hidden_layers, float ** biases) {
	float sum;
	int SJ = 0, SX = 0, SJJ = fixed_size;
	for(int j = 1; j < fixed_depth; j++)
	{
		//SX = 0;
		SJ = fixed_size * (j-1);
		SJJ = fixed_size * j;
		for(int x = 0; x < fixed_size; x++)
		{
			sum = 0;
			SX = fixed_size * x;
			for(int y = 0; y < fixed_size; y++)
			{	// 이전 레이어는 계속 반복 : x가 없음 // 근데 weights는 새로운 세트로 바뀜 
				sum += hidden_layers[SJ + y] * weights[j][SX + y];
			}
			sum += biases[j][x];
			hidden_layers[SJJ + x] = sigmoid(sum);
			//SX += fixed_size;
		}
		//SJ += fixed_size;
		//SJJ += fixed_size;
	}
}

void last_2_out(float * output, float ** weights, float * hidden_layers, float ** biases) {
	float sum;
	int SX = 0;
	
	for(int x = 0; x < DIGIT_COUNT; x++)
	{
		sum = 0;
		SX = fixed_size * x;
		for(int y = 0; y < fixed_size; y++)
		{
			sum += hidden_layers[SD + y] * weights[fixed_depth][SX + y];
		}
		sum += biases[fixed_depth][x];
		output[x] = sigmoid(sum);
		//SX += fixed_size;
	}
}
*/

void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
	register unsigned int i, x, y, SX;
	float *hidden_layers, *temp, **weights, **biases, sum, sum2;

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

	// Recognize numbers
	for(i = 0; i < IMG_COUNT; i++)
	{
		//float * input = images + IMG_SIZE * i;
		float output[DIGIT_COUNT];


		// From the input layer to the first hidden layer
		// in_2_first(input, weights, hidden_layers, biases);

		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			sum = biases[0][x];
			sum2 = 0;
			for(y = 0; y < IMG_SIZE; y+=2)
			{
				sum += input[y] * weights[0][SX + y];
				sum2 += input[y+1] * weights[0][SX + y+1];
			}
			sum += sum2;
			hidden_layers[x] = sigmoid(sum);
			SX += IMG_SIZE;
		}

		// Between hidden layers
		// between(weights, hidden_layers, biases);

		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			sum = biases[1][x];
			sum2 = 0;
			for(y = 0; y < fixed_size; y+=2)
			{	
				sum += hidden_layers[y] * weights[1][SX + y];
				sum2 += hidden_layers[y+1] * weights[1][SX + y+1];
			}
			sum += sum2;
			hidden_layers[fixed_size + x] = sigmoid(sum);
			SX += fixed_size;
		}

		SX = 0;
		for(x = 0; x < fixed_size; x++)
		{
			sum = biases[2][x];
			sum2 = 0;
			for(y = 0; y < fixed_size; y+=2)
			{	
				sum += hidden_layers[fixed_size + y] * weights[2][SX + y];
				sum2 += hidden_layers[fixed_size + y+1] * weights[2][SX + y+1];
			}
			sum += sum2;
			hidden_layers[SD + x] = sigmoid(sum);
			SX += fixed_size;
		}

		// From the last hidden layer to the output layer
		// last_2_out(output, weights, hidden_layers, biases);

		SX = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			sum = biases[fixed_depth][x];
			sum2 = 0;
			for(y = 0; y < fixed_size; y+=2)
			{
				sum += hidden_layers[SD + y] * weights[fixed_depth][SX + y];
				sum2 += hidden_layers[SD + y+1] * weights[fixed_depth][SX + y+1];
			}
			sum += sum2;
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

