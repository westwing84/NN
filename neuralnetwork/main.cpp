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

//�֐��̃v���g�^�C�v�錾
double sigmoid(double s);
double diff(double (*func)(double, double), double x, double t);
double loss_func(double x, double t);
void learning(double t_in[], double t_out[],
	int input, int output, int layer, int elenum,
	double epsilon
);

static double*** omega;	//�d��

//main�֐�
int main(void) {
	
	int layer = LAYER, elenum = NUM, input = INPUT, output = OUTPUT;
	double epsilon = 0.2;
	double t_in[TDTMAX][INPUT] = { {0,0,0},{1,0,1},{1,1,1},{1,1,0},{1,0,0},{0,0,1} };	//���t�f�[�^����
	double t_out[TDTMAX][OUTPUT] = { {0},{0},{1},{0},{1},{1} };							//���t�f�[�^�o��

	//omega�̃f�[�^�̈�m��
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum + 1, sizeof(double));
		for (int j = 0; j < elenum + 1; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}
	//�ϐ�omega�̐���
	//omega[i][j][k]: i�w�ڂ�j�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̎}�̏d�݁D
	//0�w�ڂ͓��͑w�C0�Ԗڂ̃j���[�����̓o�C�A�X�Ƃ���D


	//�d�݂̏����l�𗐐��ɂ�茈��
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = (double)rand() / (double)(RAND_MAX + 1);
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

//2�ϐ��֐�func��x�ɂ���������W�������߂�֐�
double diff(double (*func)(double, double), double x, double t) {
	double delta = 1e-4;
	return (func(x + delta, t) - func(x - delta, t)) / (2 * delta);
}

//�����֐�
//x: �j���[�����ւ̓��́Ct: �o�͂̋��t�f�[�^
double loss_func(double x, double t) {
	return pow(sigmoid(x) - t, 2);
}

//���t�f�[�^���w�K�����C�d��omega�����肷��Depsilon�͊w�K���D
void learning(double t_in[], double t_out[], 
	int input, int output, int layer, int elenum, 
	double epsilon
) {
	double*** x = (double***)calloc(layer, sizeof(double));				//�e�j���[�����ւ̓���
	double** u = (double**)calloc(layer - 1, sizeof(double));			//�e�j���[��������̏o��
	double* y = (double*)calloc(output, sizeof(double));				//NN�̏o��
	double b = 1;														//�o�C�A�X
	double error;														//�덷
	double** dLdx = (double**)calloc(layer - 1, sizeof(double));		//�t�`�d
	double dLdx_sum = 0;
	
	//�̈�m��
	for (int i = 0; i < layer; i++) {
		x[i] = (double**)calloc(elenum, sizeof(double));
		if (i < layer - 1) u[i] = (double*)calloc(elenum, sizeof(double));
		for (int j = 0; j < elenum; j++) {
			x[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}
	for (int i = 0; i < layer - 1; i++) {
		dLdx[i] = (double*)calloc(elenum, sizeof(double));
	}

	//���͑w�����1�w�ւ̓`�B
	for (int i = 0; i < input; i++) {
		for (int j = 0; j < elenum; j++) {
			x[0][i][j] = t_in[i];
		}
	}
	for (int i = 0; i < elenum + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) u[0][j] += b * omega[0][i][j];
			else u[0][j] += t_in[i - 1] * omega[0][i][j];
		}
	}

	//���ԑw�̓`�B(3�w�ȏ�̏ꍇ�̂�)
	if (layer >= 3) {
		for (int i = 1; i < layer - 1; i++) {
			for (int j = 0; j < elenum + 1; j++) {
				for (int k = 0; k < elenum; k++) {
					if (j == 0) u[i][k] += b * omega[i][j][k];
					else {
						u[i - 1][j - 1] = sigmoid(u[i - 1][j - 1]);
						x[i][j - 1][k] = u[i - 1][j - 1];
						u[i][k] += x[i][j - 1][k] * omega[i][j][k];
					}
				}
			}
		}
	}

	//��(layer-1)�w����o�͑w(��layer�w)�ւ̓`�B��NN�̏o�͂̌v�Z
	for (int j = 0; j < elenum + 1; j++) {
		for (int k = 0; k < output; k++) {
			if (j == 0) y[k] += b * omega[layer - 1][j][k];
			else {
				u[layer - 2][j - 1] = sigmoid(u[layer - 2][j - 1]);
				x[layer - 1][j - 1][k] = u[layer - 2][j - 1];
				y[k] += x[layer - 1][j - 1][k] * omega[layer - 1][j][k];
			}
		}
	}
	for (int k = 0; k < output; k++) {
		y[k] = sigmoid(y[k]);
	}


	//�덷�t�`�d�ɂ��d�݂̍X�V
	//�o�͑w
	for (int i = 0; i < elenum + 1; i++) {
		for (int j = 0; j < output; j++) {
			if (i == 0) error = 2 * b * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
			else error = 2 * x[layer - 1][i - 1][j] * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
			omega[layer - 1][i][j] -= epsilon * error;
			if (i > 0) dLdx[layer - 2][j] = 2 * omega[layer - 1][i][j] * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
		}
	}

	//���ԑw
	for (int i = layer - 2; i > 0;i--) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				for (int l = 0; l < elenum; l++) {
					dLdx_sum += dLdx[i][l];
				}
				if (j == 0) error = b * u[i][k] * (1 - u[i][k]) * dLdx_sum;
				else error = x[i][j - 1][k] * u[i][k] * (1 - u[i][k]) * dLdx_sum;
				omega[i][j][k] -= epsilon * error;
				dLdx_sum = 0;
				dLdx[i - 1][k] *= omega[i][j][k] * u[i][k] * (1 - u[i][k]);
			}
		}
	}

	//���͑w
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			for (int k = 0; k < elenum; k++) {
				dLdx_sum += dLdx[0][k];
			}
			if (i == 0) error = b * u[0][j] * (1 - u[0][j]) * dLdx_sum;
			else error = x[0][i - 1][j] * u[0][j] * (1 - u[0][j]) * dLdx_sum;
			omega[0][i][j] -= epsilon * error;
		}
		
	}
	
	//�̈�̉��
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(x[i][j]);
		}
		free(x[i]);
		if (i < layer - 1) free(u[i]);
	}
	for (int i = 0; i < layer - 1; i++) {
		free(dLdx[i]);
	}
	free(x);
	free(u);
	free(y);
	free(dLdx);
}