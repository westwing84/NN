/****************************
3����1�o�͂̃j���[�����l�b�g���[�N

****************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LAYER 2		//NN�̑w��
#define NUM 4		//NN�̑f�q��
#define TDTMAX 6	//���t�f�[�^�Q�̐�
#define INPUT 3		//���͂̌�
#define OUTPUT 1	//�o�͂̌�

double sigmoid(double s);
void learning(double t_in[], double t_out[],
	int input, int output, int layer, int elenum,
	double epsilon
);

static double*** omega;	//�d��

int main(void) {
	
	int layer = LAYER, elenum = NUM, input = INPUT, output = OUTPUT;
	double epsilon = 0.2;
	double t_in[TDTMAX][INPUT] = { {0,0,0},{1,0,1},{1,1,1},{1,1,0},{1,0,0},{0,0,1} };	//���t�f�[�^����
	double t_out[TDTMAX][OUTPUT] = { {0},{0},{1},{0},{1},{1} };							//���t�f�[�^�o��

	//omega�̃f�[�^�̈�m��
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum, sizeof(double));
		for (int j = 0; j < elenum; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//�d�݂̏����l�𗐐��ɂ�茈��
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = rand() / (RAND_MAX + 1);
			}
		}
	}

	learning(t_in[0], t_out[0], input, output, layer, elenum, epsilon);

	//�̈�̉��
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(omega[i][j]);
		}
		free(omega[i]);
	}
	free(omega);
	
	return 0;
}

//�V�O���C�h�֐�
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}

//���t�f�[�^���w�K�����C�d��omega�����肷��Depsilon�͊w�K���D
void learning(double t_in[], double t_out[], 
	int input, int output, int layer, int elenum, 
	double epsilon
) {
	double*** x = (double***)calloc(layer, sizeof(double));				//�e�j���[�����ւ̓���
	double** u = (double**)calloc(layer - 1, sizeof(double));			//�e�j���[��������̏o��
	double* y = (double*)calloc(output, sizeof(double));				//NN�̏o��
	
	//�̈�m��
	for (int i = 0; i < layer; i++) {
		x[i] = (double**)calloc(elenum, sizeof(double));
		if (i < layer - 1) u[i] = (double*)calloc(elenum, sizeof(double));
		for (int j = 0; j < elenum; j++) {
			x[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//���͑w�����1�w�ւ̓`�B
	for (int i = 0; i < input; i++) {
		for (int j = 0; j < elenum; j++) {
			x[0][i][j] = t_in[i];
		}
	}
	for (int i = 0; i < elenum; i++) {
		for (int j = 0; j < elenum; j++) {
			u[0][j] += t_in[i] * omega[0][i][j];
		}
	}

	//���ԑw�̓`�B(3�w�ȏ�̏ꍇ�̂�)
	if (layer >= 3) {
		for (int i = 1; i < layer - 1; i++) {
			for (int j = 0; j < elenum; j++) {
				for (int k = 0; k < elenum; k++) {
					u[i - 1][j] = sigmoid(u[i - 1][j]);
					x[i][j][k] = u[i - 1][j];
					u[i][k] += x[i][j][k] * omega[i][j][k];
				}
			}
		}
	}

	//��(layer-1)�w����o�͑w(��layer�w)�ւ̓`�B��NN�̏o�͂̌v�Z

	for (int j = 0; j < elenum; j++) {
		for (int k = 0; k < output; k++) {
			u[layer - 2][j] = sigmoid(u[layer - 2][j]);
			x[layer - 1][j][k] = u[layer - 2][j];
			y[k] += x[layer - 1][j][k] * omega[layer - 1][j][k];
		}
	}
	for (int k = 0; k < output; k++) {
		y[k] = sigmoid(y[k]);
		printf("%lf\n", y[k]);
	}

	//�̈�̉��
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(x[i][j]);
		}
		free(x[i]);
		if (i < layer - 1) free(u[i]);
	}
	free(x);
	free(u);
	free(y);
}