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
	int m = 1024;
	int n = m;
	double xmax = 10.0;
	double ymax = xmax;
	double limits[4] = {-xmax, xmax, -ymax, ymax};
	FieldCreate(&field, m, n, limits);

	FieldFillConstant(&field, 12.0);
	FieldWriteIntensitiesToFile(&field, "field_0.dat");

	double lambda = 1.0e-1;
	double k0 = 2.0 * M_PI / lambda;
	double f = 4.0;
	double aperture_radius = 0.5 * xmax;

	FieldSphericalAperture(&field, aperture_radius);
	FieldThinLens(&field, k0, f);
	FieldWriteIntensitiesToFile(&field, "field_0.dat");

	double dz = 0.02 * f;
	for (int i = 0; i < 10; ++i) {
		FieldPropagate(&field, k0, dz);
		char file_name[1000];
		build_file_name("field", i + 1, ".dat", 1000, file_name);
		FieldWriteIntensitiesToFile(&field,file_name);
	}

	/*
	FieldPropagate(&field, k0, f);
	FieldWriteIntensitiesToFile(&field, "field_2.dat");

	FieldThinLens(&field, k0, f);
	FieldPropagate(&field, k0, 2 * f);
	FieldWriteIntensitiesToFile(&field, "field_3.dat");
	*/

	FieldDestroy(&field);
	return 0;
}

