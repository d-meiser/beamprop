#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fftw3.h>


typedef double complex Amplitude;


/** Scalar field on uniform cartesian grid */
struct Field {
	/** 2D array of field amplitudes */
	Amplitude *amplitude;
	/** 2D array of fourier transformed field amplitudes */
	Amplitude *fourier_amplitude;
	/** size along x */
	int m;
	/** size along y */
	int n;
	/** limits of domain along x and y */
	double limits[4];
	/** padded length of a row to achieve alignment of each row */
	int padded_n;
	/** FFT plan for this field */
	fftw_plan plan_forward;
	fftw_plan plan_backward;
};

#define MIN_ALIGNMENT 128

static int compute_padded_n(int n)
{
	static const int alignment = MIN_ALIGNMENT / sizeof(Amplitude);
	int padded_n = (n + alignment - 1) / alignment;
	padded_n *= alignment;
	assert(padded_n >= n);
	return padded_n;
}

/** Allocates an aligned 2D array of amplitudes
 *
 * n has to be a multiple of (MIN_ALIGNMENT / sizeof(Amplitude)). The allocated
 * memory has the property that the beginning of each row is aligned on a
 * MIN_ALIGNMENT boundary.
 */
static Amplitude *allocate_2D_array(int m, int n)
{
	assert(n % (MIN_ALIGNMENT / sizeof(Amplitude)) == 0);
	Amplitude *array = aligned_alloc(MIN_ALIGNMENT,
		m * n * sizeof(Amplitude));
	return array;
}

void FieldCreate(struct Field *field, int m, int n, double limits[static 4])
{
	field->m = m;
	field->n = n;
	field->padded_n = compute_padded_n(n);
	field->amplitude = allocate_2D_array(m, field->padded_n);
	field->fourier_amplitude = allocate_2D_array(m, field->padded_n);
	memcpy(field->limits, limits, 4 * sizeof(*limits));

	int dims[2] = {field->m, field->n};
	int inembed[2] = {field->m, field->padded_n};
	int onembed[2] = {field->m, field->padded_n};
	int istride = 1;
	int ostride = 1;
	int idist = 0;
	int odist = 0;
	field->plan_forward = fftw_plan_many_dft(
		2, dims, 1,
		field->amplitude, inembed, istride, idist,
		field->fourier_amplitude, onembed, ostride, odist,
		FFTW_FORWARD, FFTW_ESTIMATE);
	field->plan_backward = fftw_plan_many_dft(
		2, dims, 1,
		field->fourier_amplitude, onembed, ostride, odist,
		field->amplitude, inembed, istride, idist,
		FFTW_BACKWARD, FFTW_ESTIMATE);
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
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			double y =  ymin + j * dy;
			row[j] = f(x, y, ctx);
		}
	}
	asm("# End of FieldFill\n");
}

void FieldDestroy(struct Field *field)
{
	free(field->amplitude);
	free(field->fourier_amplitude);
	fftw_destroy_plan(field->plan_forward);
	fftw_destroy_plan(field->plan_backward);
}

static void FieldFillConstant(struct Field *field, Amplitude a)
{
	for (int i = 0; i < field->m; ++i) {
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

