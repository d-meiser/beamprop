#include <complex.h>

typedef double complex Amplitude;

struct Field {
	Amplitude *ampitude;
	int m;
	int n;
};


int main(int argn, char **argv)
{
	(void)argn;
	(void)argv;
	return 0;
}

