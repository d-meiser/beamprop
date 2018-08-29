#ifndef BEAMPROP_FFT_H
#define BEAMPROP_FFT_H

#include <fft_export.h>
#include <complex.h>
#include <stdio.h>
#include <fftw3.h>


typedef double complex Amplitude;

/** Compute i-th frequency for uniform sampling

    @param i Index of wave number. Assumed to be in [0, n)
    @param n Number of samples
    @param l Length of sampling interval
    @return The i-th frequency
*/
FFT_EXPORT double ki(int i, int n, double l);


/* 2D API */
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

struct Gaussian2DCtx {
	double mu_x;
	double sigma_x;
	double mu_y;
	double sigma_y;
};
FFT_EXPORT Amplitude Field2DGaussian(double x, double y, void *ctx);
FFT_EXPORT void Field2DFillConstant(struct Field2D *field, Amplitude a);
FFT_EXPORT void Field2DWriteIntensities(struct Field2D *field, FILE *f);
FFT_EXPORT void Field2DWriteIntensitiesToFile(struct Field2D *field, const char *filename);
FFT_EXPORT void Field2DSphericalAperture(struct Field2D *field, double radius);
FFT_EXPORT void Field2DThinLens(struct Field2D *field, double k0, double f);
FFT_EXPORT void Field2DAmplitudeGrating(struct Field2D *field, double pitch, double w);

/* 1D API */
/** Scalar field on uniform cartesian grid */
struct Field1D {
	/** 1D array of field amplitudes */
	Amplitude *amplitude;
	/** 1D array of fourier transformed field amplitudes */
	Amplitude *fourier_amplitude;
	/** size along x */
	int m;
	double limits[2];
	/** FFT plan for this field */
	fftw_plan plan_forward;
	fftw_plan plan_backward;
};

FFT_EXPORT void Field1DCreate(struct Field1D *field, int m, double limits[static 2]);
FFT_EXPORT void Field1DDestroy(struct Field1D *field);
FFT_EXPORT struct Field1D Field1DCopy(struct Field1D *field);
typedef Amplitude(*AnalyticalField1D)(double x, void *ctx);
FFT_EXPORT void Field1DFill(struct Field1D *field, AnalyticalField1D f, void *ctx);
FFT_EXPORT void Field1DTransform(struct Field1D *field, int sign);
FFT_EXPORT void Field1DPropagate(struct Field1D *field, double k_0, double dz);

struct Gaussian1DCtx {
	double mu_x;
	double sigma_x;
};
FFT_EXPORT Amplitude Field1DGaussian(double x, void *ctx);
FFT_EXPORT void Field1DFillConstant(struct Field1D *field, Amplitude a);
FFT_EXPORT void Field1DWriteIntensities(struct Field1D *field, FILE *f);
FFT_EXPORT void Field1DWriteIntensitiesToFile(struct Field1D *field, const char *filename);
FFT_EXPORT void Field1DSphericalAperture(struct Field1D *field, double radius);
FFT_EXPORT void Field1DThinLens(struct Field1D *field, double k0, double f);
FFT_EXPORT void Field1DAmplitudeGrating(struct Field1D *field, double pitch, double w);
#endif
