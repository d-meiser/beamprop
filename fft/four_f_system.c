#include <fft.h>
#include <stdlib.h>
#include <math.h>


void build_file_name(const char *base, int i, const char *suffix,
		     size_t n, char *file_name)
{
	snprintf(file_name, n, "%s_%d%s", base, i, suffix);
}

int main(int argn, char **argv)
{
	(void)argn;
	(void)argv;

	srand(100);

	struct Field field;
	int m = 128;
	int n = m;
	double xmax = 1.0;
	double ymax = xmax;
	double limits[4] = {-xmax, xmax, -ymax, ymax};
	FieldCreate(&field, m, n, limits);

	double wx = 5.0e-2;
	double wy = 8.0e-2;
	struct GaussianCtx gctx = {0.0, wx, 0.0, wy};
	FieldFill(&field, FieldGaussian, &gctx);
	FieldWriteIntensitiesToFile(&field, "initial_state.dat");

	double lambda = 1.0e-2;
	double k0 = 2.0 * M_PI / lambda;
	double dz = 1.0e0;
	int num_steps = 10;
	for (int i = 0; i < num_steps; ++i) {
		FieldPropagate(&field, k0, dz);
		char fn[512];
		build_file_name("field", i, ".dat", 512, fn);
		FieldWriteIntensitiesToFile(&field, fn);
	}

	FieldDestroy(&field);
	return 0;
}

