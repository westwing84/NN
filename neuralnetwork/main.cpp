/****************************
3入力1出力のニューラルネットワーク

****************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LAYER 2		//NNの層数
#define NUM 4		//NNの素子数
#define TDTMAX 6	//教師データ群の数
#define INPUT 3		//入力の個数
#define OUTPUT 1	//出力の個数

double sigmoid(double s);
void learning(double t_in[], double t_out[],
	int input, int output, int layer, int elenum,
	double epsilon
);

static double*** omega;	//重み

int main(void) {
	
	int layer = LAYER, elenum = NUM, input = INPUT, output = OUTPUT;
	double epsilon = 0.2;
	double t_in[TDTMAX][INPUT] = { {0,0,0},{1,0,1},{1,1,1},{1,1,0},{1,0,0},{0,0,1} };	//教師データ入力
	double t_out[TDTMAX][OUTPUT] = { {0},{0},{1},{0},{1},{1} };							//教師データ出力

	//omegaのデータ領域確保
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum, sizeof(double));
		for (int j = 0; j < elenum; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//重みの初期値を乱数により決定
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = rand() / (RAND_MAX + 1);
			}
		}
	}

	learning(t_in[0], t_out[0], input, output, layer, elenum, epsilon);

	//領域の解放
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum; j++) {
			free(omega[i][j]);
		}
		free(omega[i]);
	}
	free(omega);
	
	return 0;
}

//シグモイド関数
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}

//教師データを学習させ，重みomegaを決定する．epsilonは学習率．
void learning(double t_in[], double t_out[], 
	int input, int output, int layer, int elenum, 
	double epsilon
) {
	double*** x = (double***)calloc(layer, sizeof(double));				//各ニューロンへの入力
	double** u = (double**)calloc(layer - 1, sizeof(double));			//各ニューロンからの出力
	double* y = (double*)calloc(output, sizeof(double));				//NNの出力
	
	//領域確保
	for (int i = 0; i < layer; i++) {
		x[i] = (double**)calloc(elenum, sizeof(double));
		if (i < layer - 1) u[i] = (double*)calloc(elenum, sizeof(double));
		for (int j = 0; j < elenum; j++) {
			x[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//入力層から第1層への伝達
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

	//中間層の伝達(3層以上の場合のみ)
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

	//第(layer-1)層から出力層(第layer層)への伝達とNNの出力の計算

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

	//領域の解放
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