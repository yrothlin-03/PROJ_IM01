#include <iostream>
#include <string>
#include <omp.h>

#include "args.hxx"
#include "image.hpp"
#include "iio.hpp"

#include "options.hpp"
#include "estimateKernel.hpp"
#include "deconvBregman.hpp"

static options parse_args(int argc, char** argv)
{
    args::ArgumentParser parser("Recovering the blur kernel from natural image statistics: An analysis of the Goldstein-Fattal method");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<flt> compensationFactor(parser, "alpha", "factor of the compensation filter", {'a', "alpha"}, 2.1);
    args::ValueFlag<int> seed(parser, "seed", "set the seed to a predefined value", {"seed"}, -1);
    args::ValueFlag<flt> finalDeconvolutionWeight(parser, "lambda", "regularization weight for the final deconvolution", {'l', "lambda"}, flt(3000));
    args::ValueFlag<flt> intermediateDeconvolutionWeight(parser, "lambda2", "regularization weight for the kernel evaluation", {"lambda2"}, flt(3000));
    args::ValueFlag<int> Nouter(parser, "Nouter", "number of iterations of the support", {'i', "Nouter"}, 3);
    args::ValueFlag<int> Ninner(parser, "Ninner", "number of iterations of the phase retrieval", {'p', "Ninner"}, 300);
    args::ValueFlag<int> Ntries(parser, "Ntries", "number of tries of the phase retrieval", {'t', "Ntries"}, 30);
    args::ValueFlag<bool> medianFilter(parser, "medianFilter", "apply the median filtering to the autocorrelations", {'m', "median"}, true);
    args::Positional<std::string> input(parser, "input", "input blurry image file", args::Options::Required);
    args::Positional<int> kernelSize(parser, "kernelSize", "kernel size (should be odd)", args::Options::Required);
    args::Positional<std::string> out_kernel(parser, "out_kernel", "kernel output file", args::Options::Required);
    args::Positional<std::string> out_deconv(parser, "out_deconv", "deconv output file", args::Options::Required);

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cout << parser;
        exit(0);
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    } catch (const args::ValidationError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }

    if (args::get(kernelSize) % 2 == 0) {
        std::cerr << "Error: kernelSize (argument 2) has to be odd." << std::endl;
        exit(1);
    }

    options opts;
    opts.Ninner = args::get(Ninner);
    opts.Nouter = args::get(Nouter);
    opts.Ntries = args::get(Ntries);
    opts.medianFilter = args::get(medianFilter);
    opts.compensationFactor = args::get(compensationFactor);
    opts.finalDeconvolutionWeight = args::get(finalDeconvolutionWeight);
    opts.intermediateDeconvolutionWeight = args::get(intermediateDeconvolutionWeight);
    opts.seed = args::get(seed);
    opts.input = args::get(input);
    opts.kernelSize = args::get(kernelSize);
    opts.out_kernel = args::get(out_kernel);
    opts.out_deconv = args::get(out_deconv);
    return opts;
}

int main(int argc, char** argv)
{
    img_t<flt>::use_threading(omp_get_max_threads());

    struct options opts = parse_args(argc, argv);

    if (opts.seed != -1) {
        srand(opts.seed);
    } else {
        srand(time(0));
    }

    // read the input image
    int w, h, d;
    flt* data = iio_read_image<flt>(opts.input, &w, &h, &d);
    img_t<flt> img(w, h, d, data);
    free(data);

    // normalize the image between 0 and 1
    flt max = 0.;
    for (int i = 0; i < img.size; i++)
        max = std::max(max, img[i]);
    for (int i = 0; i < img.size; i++)
        img[i] /= max;

    // estimate the kernel (call Algorithm 1 of the paper)
    img_t<flt> kernel;
    estimateKernel(kernel, img, opts.kernelSize, opts);

    // save the estimated kernel
    iio_write_image(opts.out_kernel, &kernel[0], kernel.w, kernel.h, kernel.d);

    // deconvolve the blurry image using the estimated kernel
    img_t<flt> result;
    img_t<flt> tapered;
    img_t<flt> deconv;
    pad_and_taper(tapered, img, kernel);
    deconvBregman(deconv, tapered, kernel, 20, opts.finalDeconvolutionWeight);
    unpad(result, deconv, kernel);

    // clamp the result and restore the original range
    for (int i = 0; i < result.size; i++)
        result[i] = std::max(std::min(flt(1.), result[i]), flt(0.));
    for (int i = 0; i < result.size; i++)
        result[i] *= max;

    // save the deblurred image
    iio_write_image(opts.out_deconv, &result[0], result.w, result.h, result.d);

    return EXIT_SUCCESS;
}

