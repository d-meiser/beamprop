#ifndef BEAMPROP_FFT_H
#define BEAMPROP_FFT_H

#include <complex.h>
#include <stdio.h>
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

void FieldCreate(struct Field *field, int m, int n, double limits[static 4]);
void FieldDestroy(struct Field *field);
struct Field FieldCopy(struct Field *field);
typedef Amplitude(*AnalyticalField)(double x, double y, void *ctx);
void FieldFill(struct Field *field, AnalyticalField f, void *ctx);
void FieldTransform(struct Field *field, int sign);
void FieldPropagate(struct Field *field, double k_0, double dz);

struct GaussianCtx {
	double mu_x;
	double sigma_x;
	double mu_y;
	double sigma_y;
};
Amplitude FieldGaussian(double x, double y, void *ctx);
void FieldWriteIntensities(struct Field *field, FILE *f);
void FieldWriteIntensitiesToFile(struct Field *field, const char *filename);

#endif
