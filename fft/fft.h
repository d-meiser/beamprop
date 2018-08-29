#ifndef BEAMPROP_FFT_H
#define BEAMPROP_FFT_H

#include <fft_export.h>
#include <complex.h>
#include <stdio.h>
#include <fftw3.h>


typedef double complex Amplitude;
/** Scalar field on uniform cartesian grid */
struct Field2D {
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

FFT_EXPORT void Field2DCreate(struct Field2D *field, int m, int n, double limits[static 4]);
FFT_EXPORT void Field2DDestroy(struct Field2D *field);
FFT_EXPORT struct Field2D Field2DCopy(struct Field2D *field);
typedef Amplitude(*AnalyticalField2D)(double x, double y, void *ctx);
FFT_EXPORT void Field2DFill(struct Field2D *field, AnalyticalField2D f, void *ctx);
FFT_EXPORT void Field2DTransform(struct Field2D *field, int sign);
FFT_EXPORT void Field2DPropagate(struct Field2D *field, double k_0, double dz);

struct GaussianCtx {
	double mu_x;
	double sigma_x;
	double mu_y;
	double sigma_y;
};
FFT_EXPORT Amplitude Field2DGaussian(double x, double y, void *ctx);
FFT_EXPORT void Field2DFillConstant(struct Field2D *field, Amplitude a);
FFT_EXPORT void Field2DWriteIntensities(struct Field2D *field, FILE *f);
FFT_EXPORT void Field2DWriteIntensitiesToFile(struct Field2D *field, const char *filename);
/** Compute i-th frequency for uniform sampling

    @param i Index of wave number. Assumed to be in [0, n)
    @param n Number of samples
    @param l Length of sampling interval
    @return The i-th frequency
*/
FFT_EXPORT double ki(int i, int n, double l);
FFT_EXPORT void Field2DSphericalAperture(struct Field2D *field, double radius);
FFT_EXPORT void Field2DThinLens(struct Field2D *field, double k0, double f);
FFT_EXPORT void Field2DAmplitudeGrating(struct Field2D *field, double pitch, double w);

#endif
