#ifndef MATRIX_MANAGER_H
#define MATRIX_MANAGER_H

#include <complex>
#include <vector>
#include "TaskScheduler.h"

class MatrixManager
{
public:
    static void matrixXmatrixMKLfull(const std::complex<float> *matrixA,
                                     const std::complex<float> *matrixB,
                                     std::complex<float> *result,
                                     const int &rowSize, const int &colSize,
                                     const int &midSize, const int &lda, const int &ldb, const int &ldc,
                                     char transA, char transB);

    static double matrixXmatrixMKLPT(const std::complex<float> *matrixA,
                                     const std::complex<float> *matrixB,
                                     std::complex<float> *result, const int &rowSize, const int &colSize,
                                     const int &midSize, const int &lda, const int &ldb, const int &ldc,
                                     char transA, char transB, int numRepeat);

    static void vectorPlusEqualVector_MKL(std::complex<float> *v1, const int &size, const std::complex<float> *v2);

    // Reduction and normalization of the result matrix
    static void reduceAndNormalize(std::complex<float> *result, int resultSize, int numBins,
                                   std::vector<std::complex<float>> &tempBuffer,
                                   TaskScheduler &scheduler);
};

#endif // MATRIX_MANAGER_H
