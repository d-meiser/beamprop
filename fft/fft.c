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

void Field2DCreate(struct Field2D *field, int m, int n, double limits[static 4])
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

struct Field2D Field2DCopy(struct Field2D *field)
{
	struct Field2D copy;
	Field2DCreate(&copy, field->m, field->n, field->limits);

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

void Field2DFill(struct Field2D *field, AnalyticalField2D f, void *ctx)
{
	asm("# Start of Field2DFill\n");
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
	asm("# End of Field2DFill\n");
}

void Field2DDestroy(struct Field2D *field)
{
	free(field->amplitude);
	free(field->fourier_amplitude);
	fftw_destroy_plan(field->plan_forward);
	fftw_destroy_plan(field->plan_backward);
}

void Field2DTransform(struct Field2D *field, int sign)
{
	if (sign == FFTW_FORWARD) {
		fftw_execute(field->plan_forward);
	} else if (sign == FFTW_BACKWARD) {
		fftw_execute(field->plan_backward);
	} else {
		assert(0);
	}
}

void Field2DFillConstant(struct Field2D *field, Amplitude a)
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

void Field2DPropagate(struct Field2D *field, double k_0, double dz)
{
	Field2DTransform(field, FFTW_FORWARD);

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

	Field2DTransform(field, FFTW_BACKWARD);
}

Amplitude Field2DGaussian(double x, double y, void *ctx)
{
	struct Gaussian2DCtx *gctx = ctx;
	x -= gctx->mu_x;
	y -= gctx->mu_y;
	return 1.0 / (2.0 * M_PI * gctx->sigma_x * gctx->sigma_y) * exp(
		-0.5 * x * x / (gctx->sigma_x * gctx->sigma_x)
		-0.5 * y * y / (gctx->sigma_y * gctx->sigma_y));
}

void Field2DWriteIntensities(struct Field2D *field, FILE *f)
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

void Field2DWriteIntensitiesToFile(struct Field2D *field,
					const char *filename)
{
	FILE *f = fopen(filename, "w");
	Field2DWriteIntensities(field, f);
	fclose(f);
}

void Field2DSphericalAperture(struct Field2D *field, double radius)
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

void Field2DThinLens(struct Field2D *field, double k0, double f)
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

FFT_EXPORT void Field2DAmplitudeGrating(struct Field2D *field, double pitch, double w)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		x = x - floor(x / pitch) * pitch;
		assert(x >= 0);
		assert(x <= pitch);
		double mask = 0.0;
		if (x < 0.5 * dx) {
			mask = 0.5 + x / dx;
		} else if (x >= 0.5 * dx && x < w - 0.5 * dx) {
			mask = 1.0;
		} else if (x >= w - 0.5 * dx && x < w + 0.5 * dx) {
			mask = 0.5 - (x - w) / dx;
		} else if (x >= pitch - 0.5 * dx) {
			mask = 0.5 + (x - pitch) / dx;
		}
		Amplitude *row = __builtin_assume_aligned(
			field->amplitude + i * field->padded_n, MIN_ALIGNMENT);
		for (int j = 0; j < field->n; ++j) {
			row[j] *= mask;
		}
	}
}

void Field1DCreate(struct Field1D *field, int m, double limits[static 2])
{
	field->m = m;
	field->amplitude = aligned_alloc(MIN_ALIGNMENT, m * sizeof(Amplitude));
	field->fourier_amplitude =
		aligned_alloc(MIN_ALIGNMENT, m * sizeof(Amplitude));
	memcpy(field->limits, limits, 2 * sizeof(*limits));

	int dims[1] = {field->m};
	int inembed[1] = {field->m};
	int onembed[1] = {field->m};
	int istride = 1;
	int ostride = 1;
	int idist = 0;
	int odist = 0;
	field->plan_forward = fftw_plan_many_dft(
		1, dims, 1,
		field->amplitude, inembed, istride, idist,
		field->fourier_amplitude, onembed, ostride, odist,
		FFTW_FORWARD, FFTW_ESTIMATE);
	field->plan_backward = fftw_plan_many_dft(
		1, dims, 1,
		field->fourier_amplitude, onembed, ostride, odist,
		field->amplitude, inembed, istride, idist,
		FFTW_BACKWARD, FFTW_ESTIMATE);
}

struct Field1D Field1DCopy(struct Field1D *field)
{
	struct Field1D copy;
	Field1DCreate(&copy, field->m, field->limits);

	assert(copy.m == field->m);

	memcpy(copy.amplitude, field->amplitude,
		field->m * sizeof(*field->amplitude));
	memcpy(copy.fourier_amplitude, field->fourier_amplitude,
		field->m * sizeof(*field->fourier_amplitude));

	return copy;
}

void Field1DFill(struct Field1D *field, AnalyticalField1D f, void *ctx)
{
	asm("# Start of Field1DFill\n");
	double xmin = field->limits[0];
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	Amplitude *row = __builtin_assume_aligned(field->amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		double x =  xmin + i * dx;
		row[i] = f(x, ctx);
	}
	asm("# End of Field1DFill\n");
}

void Field1DDestroy(struct Field1D *field)
{
	free(field->amplitude);
	free(field->fourier_amplitude);
	fftw_destroy_plan(field->plan_forward);
	fftw_destroy_plan(field->plan_backward);
}

void Field1DTransform(struct Field1D *field, int sign)
{
	if (sign == FFTW_FORWARD) {
		fftw_execute(field->plan_forward);
	} else if (sign == FFTW_BACKWARD) {
		fftw_execute(field->plan_backward);
	} else {
		assert(0);
	}
}

void Field1DFillConstant(struct Field1D *field, Amplitude a)
{
	Amplitude *row = __builtin_assume_aligned(field->amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		row[i] = a;
	}
}


void Field1DPropagate(struct Field1D *field, double k_0, double dz)
{
	Field1DTransform(field, FFTW_FORWARD);

	const double lx = field->limits[1] - field->limits[0];
	Amplitude *row = __builtin_assume_aligned(
		field->fourier_amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		const double kx = ki(i, field->m, lx);
		double k_par = kx * kx / (2.0 * k_0);
		row[i] *= cexp(I * k_par * dz) / field->m;
	}

	Field1DTransform(field, FFTW_BACKWARD);
}

Amplitude Field1DGaussian(double x, void *ctx)
{
	struct Gaussian1DCtx *gctx = ctx;
	x -= gctx->mu_x;
	return 1.0 / sqrt(2.0 * M_PI * gctx->sigma_x) * exp(
		-0.5 * x * x / (gctx->sigma_x * gctx->sigma_x));
}

void Field1DWriteIntensities(struct Field1D *field, FILE *f)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	Amplitude *row = __builtin_assume_aligned(
		field->amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		fprintf(f, "%lf %lf %lf  ", x, creal(row[i]), cimag(row[i]));
	}
}

void Field1DWriteIntensitiesToFile(struct Field1D *field,
				   const char *filename)
{
	FILE *f = fopen(filename, "w");
	Field1DWriteIntensities(field, f);
	fclose(f);
}

void Field1DSphericalAperture(struct Field1D *field, double radius)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	double radius_squared = radius * radius;
	Amplitude *row = __builtin_assume_aligned(
		field->amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		if (x * x > radius_squared) {
			row[i] = 0.0;
		}
	}
}

void Field1DThinLens(struct Field1D *field, double k0, double f)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	Amplitude *row = __builtin_assume_aligned(
		field->amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		double phase = -0.5 * k0 * x*x / f;
		double c = cos(phase);
		double s = sin(phase);
		row[i] *= c + I * s;
	}
}

FFT_EXPORT void Field1DAmplitudeGrating(struct Field1D *field, double pitch, double w)
{
	double dx = compute_dx(field->limits[0], field->limits[1], field->m);
	Amplitude *row = __builtin_assume_aligned(
		field->amplitude, MIN_ALIGNMENT);
	for (int i = 0; i < field->m; ++i) {
		double x = field->limits[0] + i * dx;
		x = x - floor(x / pitch) * pitch;
		assert(x >= 0);
		assert(x <= pitch);
		double mask = 0.0;
		if (x < 0.5 * dx) {
			mask = 0.5 + x / dx;
		} else if (x >= 0.5 * dx && x < w - 0.5 * dx) {
			mask = 1.0;
		} else if (x >= w - 0.5 * dx && x < w + 0.5 * dx) {
			mask = 0.5 - (x - w) / dx;
		} else if (x >= pitch - 0.5 * dx) {
			mask = 0.5 + (x - pitch) / dx;
		}
		row[i] *= mask;
	}
}
