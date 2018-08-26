#ifndef BEAMPROP_FFT_H
#define BEAMPROP_FFT_H

#include <fft_export.h>
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

FFT_EXPORT void FieldCreate(struct Field *field, int m, int n, double limits[static 4]);
FFT_EXPORT void FieldDestroy(struct Field *field);
FFT_EXPORT struct Field FieldCopy(struct Field *field);
typedef Amplitude(*AnalyticalField)(double x, double y, void *ctx);
FFT_EXPORT void FieldFill(struct Field *field, AnalyticalField f, void *ctx);
FFT_EXPORT void FieldTransform(struct Field *field, int sign);
FFT_EXPORT void FieldPropagate(struct Field *field, double k_0, double dz);

struct GaussianCtx {
	double mu_x;
	double sigma_x;
	double mu_y;
	double sigma_y;
};
FFT_EXPORT Amplitude FieldGaussian(double x, double y, void *ctx);
FFT_EXPORT void FieldFillConstant(struct Field *field, Amplitude a);
FFT_EXPORT void FieldWriteIntensities(struct Field *field, FILE *f);
FFT_EXPORT void FieldWriteIntensitiesToFile(struct Field *field, const char *filename);
/** Compute i-th frequency for uniform sampling

    @param i Index of wave number. Assumed to be in [0, n)
    @param n Number of samples
    @param l Length of sampling interval
    @return The i-th frequency
*/
FFT_EXPORT double ki(int i, int n, double l);
FFT_EXPORT void FieldSphericalAperture(struct Field *field, double radius);
FFT_EXPORT void FieldThinLens(struct Field *field, double k0, double f);

#endif
