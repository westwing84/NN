/****************************
3����1�o�͂̃j���[�����l�b�g���[�N

****************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#define LAYER 3		//NN�̑w��
#define NUM 4		//NN�̑f�q��
#define TDTMAX 6	//���t�f�[�^�Q�̐�
#define INPUT 3		//���͂̌�
#define OUTPUT 1	//�o�͂̌�
#define EPSILON 0.1	//�w�K��
#define LEARNING_TIMES 1e7	//�w�K�񐔂̏��

//�֐��̃v���g�^�C�v�錾
double sigmoid(double s);
double learning(
	double t_in[],		//���͋��t�f�[�^
	double t_out[],		//�o�͋��t�f�[�^
	double y[],			//NN�̏o��
	int input,			//NN�̓��͂̌�
	int output,			//NN�̏o�͂̌�
	int layer,			//NN�̑w��
	int elenum,			//�e�w�ɂ�����j���[�����̌�
	double epsilon		//�w�K��
);
void transmission(double in[], double y[], int input, int output, int layer, int elenum);

//�ϐ��錾
static double*** omega;	//�d��
//omega[i][j][k]: i�w�ڂ�j�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̎}�̏d�݁D
//�e�v�f���́Comega[layer][elenum+1][elenum]�Delenum+1�Ƃ��Ă���̂̓o�C�A�X���܂�ł��邽�߁D
//0�w�ڂ͓��͑w�D�܂��Comega[i][0][k]�̓o�C�A�X�ł���D

static double*** x;		//�e�j���[�����ւ̓��́D�o�C�A�X�������D
//x[i][j][k]: i�w�ڂ�(j+1)�Ԗڂ̃j���[��������(i+1)�w�ڂ�(k+1)�Ԗڂ̃j���[�����ւ̓��́D
//�e�v�f���́Cx[layer][elenum][elenum]�D

static double** u;		//�e�j���[��������̏o��
//u[i][j]: (i+1)�w�ڂ�(j+1)�Ԗڂ̃j���[�����̏o�́D
//�e�v�f���́Cu[layer-1][elenum]�D

static double** dLdx;	//�t�`�d
//dLdx[i][j]: (i+2)�w�ڂ�(j+1)�Ԗڂ̃j���[�����̋t�`�d�o�́D
//�e�v�f���́CdLdx[layer-1][elenum]�D

//main�֐�
int main(void) {
	
	int layer = LAYER, elenum = NUM, input = INPUT, output = OUTPUT;
	double epsilon = EPSILON;
	double* dt_in, *dt_out;
	double* y;		//NN�̏o��
	double error;	//�덷
	int command;
	double t_in[TDTMAX][INPUT];
	double t_out[TDTMAX][OUTPUT];

	//���t�f�[�^�t�@�C���I�[�v��
	ifstream ifs("data.csv");
	if (!ifs) {
		printf("���t�f�[�^�t�@�C�����J���܂���ł����D\n");
		return 0;
	}

	//���t�f�[�^��t_in��t_out�ɓǂݍ���
	string str;
	for (int i = 0; getline(ifs, str); i++) {
		string tmp;
		stringstream stream;
		stream << str;
		for (int j = 0; getline(stream, tmp, ','); j++) {
			if (j < INPUT) t_in[i][j] = atof(tmp.c_str());
			else t_out[i][j - INPUT] = atof(tmp.c_str());
		}
	}
	
	//�e�p�����[�^���L�[�{�[�h�������
	printf("���t�f�[�^���j���[�����l�b�g���[�N�Ɋw�K�����܂��D\n�w�K������͂��Ă�������: ");
	scanf_s("%lf", &epsilon);
	printf("�j���[�����l�b�g���[�N�̑w��: ");
	scanf_s("%d", &layer);
	printf("�e�w�̑f�q��: ");
	scanf_s("%d", &elenum);

	//omega�̃f�[�^�̈�m��
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum + 1, sizeof(double));
		for (int j = 0; j < elenum + 1; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//�̈�m��
	x = (double***)calloc(layer, sizeof(double));
	u = (double**)calloc(layer - 1, sizeof(double));
	dLdx = (double**)calloc(layer - 1, sizeof(double));
	y = (double*)calloc(output, sizeof(double));
	dt_in = (double*)calloc(input, sizeof(double));
	dt_out = (double*)calloc(output, sizeof(double));
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

	//�d�݂̏����l��-1�`1�̗����ɂ�茈��
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = (double)rand() / (double)(RAND_MAX + 1) * 2 - 1;
			}
		}
	}
	
	
	//�w�K
	printf("���t�f�[�^���w�K���Ă��܂��D\n");
	for (int i = 0; i < LEARNING_TIMES; i++) {
		error = 0;
		for (int j = 0; j < TDTMAX; j++) {
			error += learning(t_in[j], t_out[j], y, input, output, layer, elenum, epsilon);
		}
		error /= TDTMAX;
		//if (i % 100000 == 0) printf("%lf\n", error);
		if (error < 1e-3) break;
	}
	printf("�w�K���������܂����D\n");
	command = 2;
	while (command != 0) {
		printf("0: �I���C1: �f�[�^����\n�R�}���h����͂��Ă�������: ");
		scanf_s("%d", &command);
		switch (command)
		{
		case 0:
			printf("�I�����܂��D\n");
			break;

		case 1:
			printf("�j���[�����l�b�g���[�N�ւ̓��͂��s���܂��D�f�[�^��%d���͂��Ă��������D\n", input);
			for (int i = 0; i < input; i++) {
				scanf_s("%lf", &dt_in[i]);
			}
			transmission(dt_in, dt_out, input, output, layer, elenum);
			printf("�o�͂�\n");
			for (int i = 0; i < output; i++) {
				printf("%lf ", dt_out[i]);
			}
			printf("\n");
			break;

		default:
			printf("������x���͂��Ă��������D\n");
			break;
		}
	}
	

	//�̈�̉��
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(omega[i][j]);
		}
		free(omega[i]);
	}
	free(omega);
	free(y);

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
	free(dLdx);
	free(dt_in);
	free(dt_out);
	
	return 0;
}


//�V�O���C�h�֐�
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}


/**********************************
���t�f�[�^���w�K�����C�d��omega�����肷��D
�߂�l�͋��t�f�[�^�Əo�͂̌덷�D
***********************************/
double learning(
	double t_in[],		//���͋��t�f�[�^
	double t_out[],		//�o�͋��t�f�[�^
	double y[],			//NN�̏o��
	int input,			//NN�̓��͂̌�
	int output,			//NN�̏o�͂̌�
	int layer,			//NN�̑w��
	int elenum,			//�e�w�ɂ�����j���[�����̌�
	double epsilon		//�w�K��
) {
	double b = 1;				//�o�C�A�X�ɑ΂������
	double error, error_out;	//�덷
	double dLdx_sum = 0;		//�t�`�d�̘a

	//���`�d�ɂ�苳�t����t_in�ɑ΂���o��y���v�Z����
	transmission(t_in, y, input, output, layer, elenum);

	//�o�͂̌덷�̌v�Z
	error_out = 0;
	for (int i = 0; i < output; i++) {
		error_out += pow(y[i] - t_out[i], 2);
	}

	//�덷�t�`�d�@�ɂ��d�݂̍X�V
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
	for (int i = layer - 2; i > 0; i--) {
		for (int j = 0; j < elenum; j++) {
			dLdx_sum += dLdx[i][j];
		}
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				if (j == 0) error = b * u[i][k] * (1 - u[i][k]) * dLdx_sum;
				else error = x[i][j - 1][k] * u[i][k] * (1 - u[i][k]) * dLdx_sum;
				omega[i][j][k] -= epsilon * error;
				if (j > 0) dLdx[i - 1][j - 1] = omega[i][j][k] * u[i][k] * (1 - u[i][k]) * dLdx[i][k];
			}
		}
		dLdx_sum = 0;
	}

	//���͑w
	for (int k = 0; k < elenum; k++) {
		dLdx_sum += dLdx[0][k];
	}
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) error = b * u[0][j] * (1 - u[0][j]) * dLdx_sum;
			else error = x[0][i - 1][j] * u[0][j] * (1 - u[0][j]) * dLdx_sum;
			omega[0][i][j] -= epsilon * error;
		}
	}

	return error_out;
}


//���`�d�ɂ��NN�̏o��y�𓾂�֐�
void transmission(double in[], double y[], int input, int output, int layer, int elenum) {
	double b = 1;	//�o�C�A�X
	//�j���[��������̏o��u����ёS�̂̏o��y�̏�����
	for (int i = 0; i < layer - 1; i++) {
		for (int j = 0; j < elenum; j++) {
			u[i][j] = 0;
		}
	}
	for (int i = 0; i < output; i++) {
		y[i] = 0;
	}

	//���͑w�����1�w�ւ̓`�B
	for (int i = 0; i < input; i++) {
		for (int j = 0; j < elenum; j++) {
			x[0][i][j] = in[i];
		}
	}
	for (int i = 0; i < input + 1; i++) {
		for (int j = 0; j < elenum; j++) {
			if (i == 0) u[0][j] += b * omega[0][i][j];
			else u[0][j] += in[i - 1] * omega[0][i][j];
		}
	}

	//���ԑw�̓`�B(3�w�ȏ�̏ꍇ�̂�)
	for (int i = 1; i < layer - 1; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			if (j > 0) u[i - 1][j - 1] = sigmoid(u[i - 1][j - 1]);
			for (int k = 0; k < elenum; k++) {
				if (j == 0) u[i][k] += b * omega[i][j][k];
				else {
					x[i][j - 1][k] = u[i - 1][j - 1];
					u[i][k] += x[i][j - 1][k] * omega[i][j][k];
				}
			}
		}
	}

	//��(layer-1)�w����o�͑w(��layer�w)�ւ̓`�B
	for (int j = 0; j < elenum + 1; j++) {
		if (j > 0) u[layer - 2][j - 1] = sigmoid(u[layer - 2][j - 1]);
		for (int k = 0; k < output; k++) {
			if (j == 0) y[k] += b * omega[layer - 1][j][k];
			else {
				x[layer - 1][j - 1][k] = u[layer - 2][j - 1];
				y[k] += x[layer - 1][j - 1][k] * omega[layer - 1][j][k];
			}
		}
	}
	for (int k = 0; k < output; k++) {
		y[k] = sigmoid(y[k]);
	}
}