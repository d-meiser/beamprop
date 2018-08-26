#include <fft.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


#define MY_EPS 1.0e-9


static struct Field build_some_field()
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

static void test_ki()
{
	assert(fabs(ki(0, 10, 1.0)) < MY_EPS);
	assert(ki(1, 10, 1.0) > 0.0);
	assert(ki(9, 10, 1.0) < 0.0);
	assert(fabs(fabs(ki(16, 32, 3.0)) - 16 * 2 * M_PI / 3.0) < MY_EPS);
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
	return 0;
}
