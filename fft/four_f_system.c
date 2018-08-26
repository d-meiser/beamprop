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
	double xmax = 1.0;
	double ymax = xmax;
	double limits[4] = {-xmax, xmax, -ymax, ymax};
	FieldCreate(&field, m, n, limits);

	FieldFillConstant(&field, 10.0);
	FieldWriteIntensitiesToFile(&field, "field_0.dat");

	double lambda = 1.0e-2;
	double k0 = 2.0 * M_PI / lambda;
	double f = 5.0;
	double aperture_radius = 0.7 * xmax;

	FieldSphericalAperture(&field, aperture_radius);
	FieldThinLens(&field, k0, -f);
	char file_name[1000];
	build_file_name("field", 0, ".dat", 1000, file_name);
	FieldWriteIntensitiesToFile(&field, "field_0.dat");

	FieldPropagate(&field, k0, 0.1 * f);
	build_file_name("field", 1, ".dat", 1000, file_name);
	FieldWriteIntensitiesToFile(&field, file_name);

	FieldPropagate(&field, k0, 0.9 * f);
	build_file_name("field", 2, ".dat", 1000, file_name);
	FieldWriteIntensitiesToFile(&field, file_name);

	FieldPropagate(&field, k0, f);
	build_file_name("field", 3, ".dat", 1000, file_name);
	FieldWriteIntensitiesToFile(&field, file_name);

	FieldThinLens(&field, k0, -f);
	FieldPropagate(&field, k0, 3 * f);
	build_file_name("field", 4, ".dat", 1000, file_name);
	FieldWriteIntensitiesToFile(&field, file_name);

	FieldDestroy(&field);
	return 0;
}

