#pragma once

#include <string>

extern "C" {
#include "iio.h"
}

template <typename T>
T* iio_read_image(const std::string& fname, int *w, int *h, int *pd);

template <>
float* iio_read_image(const std::string& fname, int *w, int *h, int *pd)
{
    return iio_read_image_float_vec((char*) fname.c_str(), w, h, pd);
}

template <>
double* iio_read_image(const std::string& fname, int *w, int *h, int *pd)
{
    return iio_read_image_double_vec((char*) fname.c_str(), w, h, pd);
}

template <typename T>
void iio_write_image(const std::string& filename, T *x, int w, int h, int pd);

template <>
void iio_write_image(const std::string& filename, float *x, int w, int h, int pd)
{
    iio_write_image_float_vec((char*) filename.c_str(), x, w, h, pd);
}

template <>
void iio_write_image(const std::string& filename, double *x, int w, int h, int pd)
{
    iio_write_image_double_vec((char*) filename.c_str(), x, w, h, pd);
}


