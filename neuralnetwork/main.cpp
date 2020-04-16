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
#define LEARNING_TIMES 10000000	//学習回数の上限

//関数のプロトタイプ宣言
double sigmoid(double s);
double learning(
	double t_in[],		//入力教師データ
	double t_out[],		//出力教師データ
	double y[],			//NNの出力
	int input,			//NNの入力の個数
	int output,			//NNの出力の個数
	int layer,			//NNの層数
	int elenum,			//各層におけるニューロンの個数
	double epsilon		//学習率
);
void transmission(double in[], double y[], int input, int output, int layer, int elenum);

//変数宣言
static double*** omega;	//重み
//omega[i][j][k]: i層目のj番目のニューロンから(i+1)層目の(k+1)番目のニューロンへの枝の重み．
//各要素数は，omega[layer][elenum+1][elenum]．elenum+1としているのはバイアスに対する重みも含んでいるため．
//0層目は入力層，0番目のニューロンはバイアスとする．

static double*** x;		//各ニューロンへの入力．バイアスを除く．
//x[i][j][k]: i層目の(j+1)番目のニューロンから(i+1)層目の(k+1)番目のニューロンへの入力．
//各要素数は，x[layer][elenum][elenum]．

static double** u;		//各ニューロンからの出力
//u[i][j]: (i+1)層目の(j+1)番目のニューロンの出力．
//各要素数は，u[layer-1][elenum]．

static double** dLdx;	//逆伝播
//dLdx[i][j]: (i+2)層目の(j+1)番目のニューロンの逆伝播出力．
//各要素数は，dLdx[layer-1][elenum]．

//main関数
int main(void) {
	
	int layer = LAYER, elenum = NUM, input = INPUT, output = OUTPUT;
	double epsilon = 0.1;		//学習率
	double* dt_in, *dt_out;
	double* y;		//NNの出力
	double error;	//誤差
	int command;
	double t_in[TDTMAX][INPUT] = { {0,0,0},{1,0,1},{1,1,1},{1,1,0},{1,0,0},{0,0,1} };
	double t_out[TDTMAX][OUTPUT] = { {0},{0},{1},{0},{1},{1} };
	
	//omegaのデータ領域確保
	omega = (double***)calloc(layer, sizeof(double));
	for (int i = 0; i < layer; i++) {
		omega[i] = (double**)calloc(elenum + 1, sizeof(double));
		for (int j = 0; j < elenum + 1; j++) {
			omega[i][j] = (double*)calloc(elenum, sizeof(double));
		}
	}

	//領域確保
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

	//重みの初期値を-1〜1の乱数により決定
	srand((unsigned)time(NULL));
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < elenum + 1; j++) {
			for (int k = 0; k < elenum; k++) {
				omega[i][j][k] = (double)rand() / (double)(RAND_MAX + 1);
			}
		}
	}
	
	
	//学習
	printf("教師データを学習しています．\n");
	for (int i = 0; i < LEARNING_TIMES; i++) {
		error = 0;
		for (int j = 0; j < TDTMAX; j++) {
			error += learning(t_in[j], t_out[j], y, input, output, layer, elenum, epsilon);
		}
		if (i % 1000 == 0) printf("%lf\n", error);
		if (error < 1e-5) break;
	}
	printf("学習が完了しました．\n");
	command = 2;
	while (command != 0) {
		printf("0: 終了，1: データ入力\nコマンドを入力してください: ");
		scanf_s("%d", &command);
		switch (command)
		{
		case 0:
			printf("終了します．\n");
			break;

		case 1:
			printf("ニューラルネットワークへの入力を行います．データを%d個入力してください．\n", input);
			for (int i = 0; i < input; i++) {
				scanf_s("%lf", &dt_in[i]);
			}
			transmission(dt_in, dt_out, input, output, layer, elenum);
			printf("出力は\n");
			for (int i = 0; i < output; i++) {
				printf("%lf ", dt_out[i]);
			}
			printf("\n");
			break;

		default:
			printf("もう一度入力してください．\n");
			break;
		}
	}
	

	//領域の解放
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


//シグモイド関数
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}


/**********************************
教師データを学習させ，重みomegaを決定する．
戻り値は教師データと出力の誤差．
***********************************/
double learning(
	double t_in[],		//入力教師データ
	double t_out[],		//出力教師データ
	double y[],			//NNの出力
	int input,			//NNの入力の個数
	int output,			//NNの出力の個数
	int layer,			//NNの層数
	int elenum,			//各層におけるニューロンの個数
	double epsilon		//学習率
) {
	double b = 1;				//バイアス
	double error, error_out;	//誤差
	double dLdx_sum = 0;		//逆伝播の和

	//順伝播により出力yを計算する
	transmission(t_in, y, input, output, layer, elenum);

	//出力の誤差の計算
	error_out = 0;
	for (int i = 0; i < output; i++) {
		error_out += pow(y[i] - t_out[i], 2);
	}

	//誤差逆伝播法による重みの更新
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

	//入力層
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


//順伝播によりNNの出力yを得る関数
void transmission(double in[], double y[], int input, int output, int layer, int elenum) {
	double b = 1;	//バイアス
	//ニューロンからの出力uおよび全体の出力yの初期化
	for (int i = 0; i < layer - 1; i++) {
		for (int j = 0; j < elenum; j++) {
			u[i][j] = 0;
		}
	}
	for (int i = 0; i < output; i++) {
		y[i] = 0;
	}

	//入力層から第1層への伝達
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

	//中間層の伝達(3層以上の場合のみ)
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

	//第(layer-1)層から出力層(第layer層)への伝達
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