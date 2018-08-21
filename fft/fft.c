#include <complex.h>
#include <stdlib.h>
#include <string.h>


typedef double complex Amplitude;


/** Scalar field on uniform cartesian grid */
struct Field {
	/** 2D array of field amplitudes */
	Amplitude *amplitude;
	/** size along x */
	int m;
	/** size along y */
	int n;
	/** limits of domain along x and y */
	double limits[4];
	/** padded length of a row to achieve alignment of each row */
	int padded_n;
};

#define MIN_ALIGNMENT 128

void FieldCreate(struct Field *field, int m, int n, double limits[static 4])
{
	static const int padding = MIN_ALIGNMENT / sizeof(Amplitude);
	field->padded_n = (n + padding - 1) / padding;
	field->padded_n *= padding;
	field->amplitude = aligned_alloc(MIN_ALIGNMENT,
		m * field->padded_n * sizeof(*field->amplitude));
	field->m = m;
	field->n = n;
	memcpy(field->limits, limits, 4 * sizeof(*limits));
}

typedef Amplitude(*AnalyticalField)(double x, double y, void *ctx);
void FieldFill(struct Field *field, AnalyticalField f, void *ctx)
{
	asm("# Start of FieldFill\n");
	double xmin = field->limits[0];
	double dx = (field->limits[1] - field->limits[0]) / (field->m - 1);
	double ymin = field->limits[2];
	double dy = (field->limits[3] - field->limits[2]) / (field->n - 1);
	for (int i = 0; i < field->m; ++i) {
		double x =  xmin + i * dx;
		for (int j = 0; j < field->n; ++j) {
			double y =  ymin + j * dy;
			field->amplitude[i * field->padded_n + j] =
				f(x, y, ctx);
		}
	}
	asm("# End of FieldFill\n");
}

void FieldDestroy(struct Field *field)
{
	free(field->amplitude);
}

static void FieldFillConstant(struct Field *field, Amplitude a)
{
	for (int i = 0; i < field->m * field->n; ++i) {
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			row[j] = a;
		}
	}
}


int main(int argn, char **argv)
{
	(void)argn;
	(void)argv;

	struct Field in;
	int m = 128;
	int n = m;
	double xmax = 1.0;
	double ymax = xmax;
	double limits[4] = {-xmax, xmax, -ymax, ymax};
	FieldCreate(&in, m, n, limits);
	Amplitude a = 1.0 + I * 0.1;
	asm("# Start fill with constant field");
	FieldFillConstant(&in, a);
	asm("# End fill with constant field");

	FieldDestroy(&in);
	return 0;
}

