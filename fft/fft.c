#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


typedef double complex Amplitude;
#define MY_EPS 1.0e-9


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

struct Field FieldCopy(struct Field *field)
{
	struct Field copy;
	FieldCreate(&copy, field->m, field->n, field->limits);

	assert(copy.m == field->m);
	assert(copy.n == field->n);
	assert(copy.padded_n == field->padded_n);

	memcpy(copy.amplitude, field->amplitude,
		field->m * field->padded_n * sizeof(*field->amplitude));
	memcpy(copy.fourier_amplitude, field->fourier_amplitude,
		field->m * field->padded_n * sizeof(*field->fourier_amplitude));

	return copy;
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

void FieldTransform(struct Field *field, int sign)
{
	if (sign == FFTW_FORWARD) {
		fftw_execute(field->plan_forward);
	} else if (sign == FFTW_BACKWARD) {
		fftw_execute(field->plan_backward);
	} else {
		assert(0);
	}
}

void FieldFillConstant(struct Field *field, Amplitude a)
{
	for (int i = 0; i < field->m; ++i) {
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			row[j] = a;
		}
	}
}

struct Field build_some_field()
{
	struct Field field;
	int m = 128;
	int n = m;
	double xmax = 1.0;
	double ymax = xmax;
	double limits[4] = {-xmax, xmax, -ymax, ymax};
	FieldCreate(&field, m, n, limits);
	return field;
}

static double my_drand(double min, double max)
{
	return min + (max - min) * drand48();
}

static Amplitude noise(double x, double y, void *ctx)
{
	(void)x;
	(void)y;
	(void)ctx;
	return my_drand(-0.5, 0.5) + I * my_drand(-0.5, 0.5);
}

static void test_delta_function_transforms_to_constant()
{
	struct Field field = build_some_field();
	FieldFillConstant(&field, 0.0 + I * 0.0);
	field.amplitude[0] = 1.0;
	FieldTransform(&field, FFTW_FORWARD);

	assert(cabs(field.fourier_amplitude[0] -
		field.fourier_amplitude[1]) < MY_EPS);
	assert(cabs(field.fourier_amplitude[0] -
		field.fourier_amplitude[1 * field.padded_n]) < MY_EPS);
	assert(cabs(field.fourier_amplitude[3 * field.padded_n + 5] -
		field.fourier_amplitude[8 * field.padded_n + 10]) < MY_EPS);

	FieldDestroy(&field);
}

static void test_constant_transforms_to_delta_function()
{
	struct Field field = build_some_field();
	FieldFillConstant(&field, 3.0 + I * 2.0);
	FieldTransform(&field, FFTW_FORWARD);

	assert(cabs(field.fourier_amplitude[0]) > MY_EPS);
	assert(cabs(field.fourier_amplitude[1]) < MY_EPS);
	assert(cabs(field.fourier_amplitude[1 * field.padded_n]) < MY_EPS);
	assert(cabs(field.fourier_amplitude[5 * field.padded_n + 20]) < MY_EPS);

	FieldDestroy(&field);
}

static void test_inverse_transform()
{
	struct Field field = build_some_field();
	FieldFill(&field, noise, 0);
	struct Field field_copy = FieldCopy(&field);

	FieldTransform(&field, FFTW_FORWARD);
	FieldTransform(&field, FFTW_BACKWARD);

	assert(cabs(field_copy.amplitude[0] -
		field.amplitude[0] / (field.m * field.n)) < MY_EPS);
	assert(cabs(field_copy.amplitude[3] -
		field.amplitude[3] / (field.m * field.n)) < MY_EPS);
	assert(cabs(field_copy.amplitude[200] -
		field.amplitude[200] / (field.m * field.n)) < MY_EPS);

	FieldDestroy(&field_copy);
	FieldDestroy(&field);
}

/** Compute i-th frequency for uniform sampling

    @param i Index of wave number. Assumed to be in [0, n)
    @param n Number of samples
    @param l Length of sampling interval
    @return The i-th frequency
*/
static double ki(int i, int n, double l)
{
	double dk = 2.0 * M_PI / l;
	if (i > n / 2) i -= n;
	return i * dk;
}

static void test_ki()
{
	assert(fabs(ki(0, 10, 1.0)) < MY_EPS);
	assert(ki(1, 10, 1.0) > 0.0);
	assert(ki(9, 10, 1.0) < 0.0);
	double lx = 3.0;
	assert(fabs(fabs(ki(16, 32, lx)) - 16 * 2 * M_PI / lx) < MY_EPS);
}

void FieldPropagate(struct Field *field, double k_0, double dz)
{
	FieldTransform(field, FFTW_FORWARD);

	const double lx = field->limits[1] - field->limits[0];
	const double ly = field->limits[3] - field->limits[2];
	for (int i = 0; i > field->m; ++i) {
		const double kx = ki(i, field->m, lx);
		Amplitude *row = __builtin_assume_aligned(
			field->fourier_amplitude + i * field->padded_n,
			MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			const double ky = ki(j, field->n, ly);
			double k_par = (kx * kx + ky * ky) / (2.0 * k_0);
			row[j] *= cexp(I * k_par * dz);
		}
	}

	FieldTransform(field, FFTW_BACKWARD);
}

int main(int argn, char **argv)
{
	(void)argn;
	(void)argv;

	srand(100);

	test_delta_function_transforms_to_constant();
	test_constant_transforms_to_delta_function();
	test_inverse_transform();
	test_ki();

	struct Field field = build_some_field();
	Amplitude a = 1.0 + I * 0.1;
	asm("# Start fill with constant field");
	FieldFillConstant(&field, a);
	asm("# End fill with constant field");

	FieldDestroy(&field);
	return 0;
}

