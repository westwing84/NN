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

//関数のプロトタイプ宣言
double sigmoid(double s);
double diff(double (*func)(double, double), double x, double t);
double loss_func(double x, double t);
void learning(double t_in[], double t_out[],
	int input, int output, int layer, int elenum,
	double epsilon
);

static double*** omega;	//重み

//main関数
int main(void) {
	
	int layer = LAYER, elenum = NUM, input = INPUT, output = OUTPUT;
	double epsilon = 0.2;
	double t_in[TDTMAX][INPUT] = { {0,0,0},{1,0,1},{1,1,1},{1,1,0},{1,0,0},{0,0,1} };	//教師データ入力
	double t_out[TDTMAX][OUTPUT] = { {0},{0},{1},{0},{1},{1} };							//教師データ出力

	//omegaのデータ領域確保
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum + 1, sizeof(double));
		for (int j = 0; j < elenum + 1; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}
	//変数omegaの説明
	//omega[i][j][k]: i層目のj番目のニューロンから(i+1)層目の(k+1)番目のニューロンへの枝の重み．
	//0層目は入力層，0番目のニューロンはバイアスとする．


	//重みの初期値を乱数により決定
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = (double)rand() / (double)(RAND_MAX + 1);
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

//2変数関数funcのxにおける微分係数を求める関数
double diff(double (*func)(double, double), double x, double t) {
	double delta = 1e-4;
	return (func(x + delta, t) - func(x - delta, t)) / (2 * delta);
}

//損失関数
//x: ニューロンへの入力，t: 出力の教師データ
double loss_func(double x, double t) {
	return pow(sigmoid(x) - t, 2);
}

//教師データを学習させ，重みomegaを決定する．epsilonは学習率．
void learning(double t_in[], double t_out[], 
	int input, int output, int layer, int elenum, 
	double epsilon
) {
	double*** x = (double***)calloc(layer, sizeof(double));				//各ニューロンへの入力
	double** u = (double**)calloc(layer - 1, sizeof(double));			//各ニューロンからの出力
	double* y = (double*)calloc(output, sizeof(double));				//NNの出力
	double b = 1;														//バイアス
	double error;														//誤差
	double** dLdx = (double**)calloc(layer - 1, sizeof(double));		//逆伝播
	double dLdx_sum = 0;
	
	//領域確保
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

	//入力層から第1層への伝達
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

	//中間層の伝達(3層以上の場合のみ)
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

	//第(layer-1)層から出力層(第layer層)への伝達とNNの出力の計算
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


	//誤差逆伝播による重みの更新
	//出力層
	for (int i = 0; i < elenum + 1; i++) {
		for (int j = 0; j < output; j++) {
			if (i == 0) error = 2 * b * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
			else error = 2 * x[layer - 1][i - 1][j] * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
			omega[layer - 1][i][j] -= epsilon * error;
			if (i > 0) dLdx[layer - 2][j] = 2 * omega[layer - 1][i][j] * y[j] * (y[j] - t_out[j]) * (1 - y[j]);
		}
	}

	//中間層
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

	//入力層
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
	
	//領域の解放
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