//	ndk-build clean && ndk-build && adb push libs/arm64-v8a/recognition_seq /data/local && adb shell chmod +x /data/local/recognition_seq

#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

#include <arm_neon.h>					// NEON
#include <pthread.h>					// pthread

#define sigmoid(x) (1 / (1 + exp(-x)))

// #define macros
#define size_512				512
#define depth_3					3
#define size_depth				1536	// size * depth
#define size_depth_m1			1024	// size * (depth-1)
#define size_DIGIT_COUNT		5120	// size * DIGIT_COUNT
#define size_size				262144	// size * size
#define size_size_addsize		262656	// size * size + size
#define size_size_addsize_x2	525312	// 2 * ( size * size + size )
#define size_IMG_SIZE			401408	// size * IMAGE_SIZE
#define size_IMG_SIZE_addsize	401920	// size * IMAGE_SIZE + size
#define sof_size_depth			6144	// __SIZEOF_FLOAT__ * size * depth
#define sofp_depth_add1			32		// __SIZEOF_POINTER__ * (depth + 1)

#define thd_num 8
#define IMG_COUNT_pthread 6250

typedef struct{
	float 			* images;
	float 			**weights;
	float			**biases;
	int 			* labels;
	float 			* confidences;
	int				img_start;
} args;

void * pthd_func(void* arguments) {

	args* data = (args*) arguments;

	int				img_start 		= (data->img_start) * IMG_COUNT_pthread;
	float 			* input 		= data->images + img_start * IMG_SIZE;
	float 			**weights 		= data->weights;
	float 			**biases 		= data->biases;
	int 			* labels 		= data->labels;
	float 			* confidences	= data->confidences;

	register	int 	i, x, y, size_x;
				float 	* hidden_layers;
	hidden_layers	= (float * )malloc(sof_size_depth);

	// Recognize numbers
	for(i = img_start; i < img_start+IMG_COUNT_pthread; i++)
	{
		float output[DIGIT_COUNT];

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
				Bvec	= vld1q_f32(weights[0] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(input + y + 4);
				Bvec2	= vld1q_f32(weights[0] + size_x + y + 4);
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
				Bvec	= vld1q_f32(weights[1] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(hidden_layers + y + 4);
				Bvec2	= vld1q_f32(weights[1] + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM2, Avec2, Bvec2);
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
				Bvec	= vld1q_f32(weights[2] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(hidden_layers + size_512 + y + 4);
				Bvec2	= vld1q_f32(weights[2] + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM2, Avec2, Bvec2);
			}
			float sum = biases[2][x] + vaddvq_f32(SSUM) + vaddvq_f32(SSUM2);
			hidden_layers[size_depth_m1 + x] = sigmoid(sum);
			size_x += size_512;
		}

		
		// From the last hidden layer to the output layer
		size_x = 0;
		for(x = 0; x < DIGIT_COUNT; x++)
		{
			float32x4_t Avec,  Bvec,  SSUM;				// NEON
			float32x4_t Avec2, Bvec2, SSUM2;			// NEON unroll
			SSUM  = vdupq_n_f32(0.0);
			SSUM2 = vdupq_n_f32(0.0);
			for(y = 0; y < size_512; y += 8)
			{
				Avec	= vld1q_f32(hidden_layers + size_depth_m1 + y);
				Bvec	= vld1q_f32(weights[3] + size_x + y);
				SSUM	= vmlaq_f32(SSUM, Avec, Bvec);
				Avec2	= vld1q_f32(hidden_layers + size_depth_m1 + y + 4);
				Bvec2	= vld1q_f32(weights[3] + size_x + y + 4);
				SSUM2	= vmlaq_f32(SSUM2, Avec2, Bvec2);
			}
			float sum = biases[depth_3][x] + vaddvq_f32(SSUM) + vaddvq_f32(SSUM2);
			output[x] = sigmoid(sum);
			size_x += size_512;

			if(output[x] > confidences[i])
			{
				labels[i]= x;
				confidences[i] = output[x];
			}
		}
		input += IMG_SIZE;		// reduction in stregnth
	}
	pthread_exit(NULL);
}


void recognition(float * images, float * network, /*int depth, int size,*/ int * labels, float * confidences)
{
	float 	**weights, **biases;

	weights 		= (float **)malloc(sofp_depth_add1);
	biases 			= (float **)malloc(sofp_depth_add1);

	// Set pointers for weights and biases
	// 1. Input layer
	weights[0] = network;
	biases[0]  = weights[0] + size_IMG_SIZE;

	// 2. Hidden layers
	// layer 1
	weights[1] = network + size_IMG_SIZE_addsize;
	biases[1]  = weights[1] + size_size;
	// layer 2
	weights[2] = weights[1] + size_size_addsize;
	biases[2]  = weights[2] + size_size;

	// 3. Output layer
	weights[depth_3] = weights[depth_3 - 1] + size_size_addsize;
	biases[depth_3]  = weights[depth_3] + size_DIGIT_COUNT;

	pthread_t pid[thd_num];	//pid
	args arguments[thd_num];
	int img_divide;

	for (img_divide = 0 ; img_divide < thd_num; img_divide++) {
		arguments[img_divide].img_start		= img_divide;
		arguments[img_divide].images		= images;
		arguments[img_divide].biases		= biases;
		arguments[img_divide].weights		= weights;
		arguments[img_divide].labels		= labels;
		arguments[img_divide].confidences	= confidences;
		pthread_create( &pid[img_divide]/*pid*/ , NULL, &pthd_func, (void *)&arguments[img_divide] ); 
	}

	for (img_divide = 0 ; img_divide < thd_num; img_divide++) {
		pthread_join( pid[img_divide],  NULL );
	}
}
