#include <fft.h>
#include <assert.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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

static double compute_dx(double xmin, double xmax, int n)
{
	return (xmax - xmin) / (n - 1.0);
}

void FieldFill(struct Field *field, AnalyticalField f, void *ctx)
{
	asm("# Start of FieldFill\n");
	double xmin = field->limits[0];
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	double ymin = field->limits[2];
	double dy = compute_dx(field->limits[2], field->limits[3], field->n);
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


double ki(int i, int n, double l)
{
	double dk = 2.0 * M_PI / l;
	if (i > n / 2) i -= n;
	return i * dk;
}

void FieldPropagate(struct Field *field, double k_0, double dz)
{
	FieldTransform(field, FFTW_FORWARD);

	const double lx = field->limits[1] - field->limits[0];
	const double ly = field->limits[3] - field->limits[2];
	for (int i = 0; i < field->m; ++i) {
		const double kx = ki(i, field->m, lx);
		Amplitude *row = __builtin_assume_aligned(
			field->fourier_amplitude + i * field->padded_n,
			MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			const double ky = ki(j, field->n, ly);
			double k_par = (kx * kx + ky * ky) / (2.0 * k_0);
			row[j] *= cexp(I * k_par * dz) / (field->m * field->n);
		}
	}

	FieldTransform(field, FFTW_BACKWARD);
}

Amplitude FieldGaussian(double x, double y, void *ctx)
{
	struct GaussianCtx *gctx = ctx;
	x -= gctx->mu_x;
	y -= gctx->mu_y;
	return 1.0 / (2.0 * M_PI * gctx->sigma_x * gctx->sigma_y) * exp(
		-0.5 * x * x / (gctx->sigma_x * gctx->sigma_x)
		-0.5 * y * y / (gctx->sigma_y * gctx->sigma_y));
}

void FieldWriteIntensities(struct Field *field, FILE *f)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	double dy = compute_dx(field->limits[2], field->limits[3], field->n);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			double y = field->limits[2] + j * dy;
			fprintf(f, "%lf %lf %lf %lf  ",
				x, y, creal(row[j]), cimag(row[j]));
		}
		fprintf(f, "\n");
	}
}

void FieldWriteIntensitiesToFile(struct Field *field,
					const char *filename)
{
	FILE *f = fopen(filename, "w");
	FieldWriteIntensities(field, f);
	fclose(f);
}

void FieldSphericalAperture(struct Field *field, double radius)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	double dy = compute_dx(field->limits[2], field->limits[3], field->n);
	double radius_squared = radius * radius;
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			double y = field->limits[2] + j * dy;
			if (x * x + y * y > radius_squared) {
				row[j] = 0.0;
			}
		}
	}
}

void FieldThinLens(struct Field *field, double k0, double f)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	double dy = compute_dx(field->limits[2], field->limits[3], field->n);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			double y = field->limits[2] + j * dy;
			double phase = -0.5 * k0 * (x*x + y*y) / f;
			double c = cos(phase);
			double s = sin(phase);
			row[j] *= c + I * s;
		}
	}
}
