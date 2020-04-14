#include <stdio.h>
#include <math.h>

double sigmoid(double s);

int main(void) {

	return 0;
}

//シグモイド関数
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}

