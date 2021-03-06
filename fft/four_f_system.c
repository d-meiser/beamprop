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

	struct Field2D field;
	int m = 2048;
	int n = m;
	double xmax = 1.0;
	double ymax = xmax;
	double limits[4] = {-xmax, xmax, -ymax, ymax};
	Field2DCreate(&field, m, n, limits);

	Field2DFillConstant(&field, 10.0);

	double lambda = 2.5e-3;
	double k0 = 2.0 * M_PI / lambda;
	double f = 5.0;
	double aperture_radius = 0.7 * xmax;

	Field2DSphericalAperture(&field, aperture_radius);
	Field2DAmplitudeGrating(&field, 0.2, 0.1);
	Field2DThinLens(&field, k0, -f);
	char file_name[1000];
	build_file_name("field", 0, ".dat", 1000, file_name);
	Field2DWriteIntensitiesToFile(&field, "field_0.dat");

	Field2DPropagate(&field, k0, 0.1 * f);
	build_file_name("field", 1, ".dat", 1000, file_name);
	Field2DWriteIntensitiesToFile(&field, file_name);

	Field2DPropagate(&field, k0, 0.9 * f);
	build_file_name("field", 2, ".dat", 1000, file_name);
	Field2DWriteIntensitiesToFile(&field, file_name);

	Field2DPropagate(&field, k0, f);
	build_file_name("field", 3, ".dat", 1000, file_name);
	Field2DWriteIntensitiesToFile(&field, file_name);

	Field2DThinLens(&field, k0, -f);
	Field2DPropagate(&field, k0, 3 * f);
	build_file_name("field", 4, ".dat", 1000, file_name);
	Field2DWriteIntensitiesToFile(&field, file_name);

	Field2DDestroy(&field);
	return 0;
}

