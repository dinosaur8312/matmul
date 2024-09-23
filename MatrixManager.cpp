#include "MatrixManager.h"
#include <mkl.h>
#include <iostream>

void MatrixManager::matrixXmatrixMKLfull(const std::complex<float> *matrixA,
                                         const std::complex<float> *matrixB,
                                         std::complex<float> *result,
                                         const int &rowSize, const int &colSize,
                                         const int &midSize, const int &lda, const int &ldb, const int &ldc,
                                         char transA, char transB)
{
    if (rowSize <= 0 || colSize <= 0 || midSize <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
        return;

    const std::complex<float> alpha(1, 0);
    const std::complex<float> beta(0, 0);

    cgemm(&transA, &transB, &rowSize, &colSize, &midSize, (MKL_Complex8 *)&alpha,
          (MKL_Complex8 *)matrixA, &lda, (MKL_Complex8 *)matrixB, &ldb,
          (MKL_Complex8 *)&beta, (MKL_Complex8 *)result, &ldc);
}

double MatrixManager::matrixXmatrixMKLPT(const std::complex<float> *matrixA,
                                         const std::complex<float> *matrixB,
                                         std::complex<float> *result, const int &rowSize, const int &colSize,
                                         const int &midSize, const int &lda, const int &ldb, const int &ldc,
                                         char transA, char transB, int numRepeat)
{
    const std::complex<float> alpha(1, 0);
    const std::complex<float> beta(0, 0);
    double time_st = dsecnd();
    
    for (int i = 0; i < numRepeat; i++)
    {
        cgemm(&transA, &transB, &rowSize, &colSize, &midSize, (MKL_Complex8 *)&alpha,
              (MKL_Complex8 *)matrixA, &lda, (MKL_Complex8 *)matrixB, &ldb,
              (MKL_Complex8 *)&beta, (MKL_Complex8 *)result, &ldc);
    }
    
    double time_end = dsecnd();
    double time_avg = (time_end - time_st) / numRepeat;
    double gflop = (2.0 * rowSize * colSize * midSize) * 1E-9;
    
    return gflop / time_avg;
}

void MatrixManager::vectorPlusEqualVector_MKL(std::complex<float> *v1, const int &size, const std::complex<float> *v2)
{
    int incx = 1;
    int incy = 1;
    float alpha = +1;
    int complexSize = size * 2; // Treat complex as two floats

    saxpy(&complexSize, &alpha, (float *)v2, &incx, (float *)v1, &incy);
}


void MatrixManager::reduceAndNormalize(std::complex<float> *result, int resultSize, int numBins,
                                       std::vector<std::complex<float>> &tempBuffer,
                                       TaskScheduler &scheduler)
{
    // Step 1: Clear temporary buffer
    memset(&tempBuffer[0], 0, resultSize * sizeof(std::complex<float>));

    // Step 2: Sum results from each task managed by the scheduler into tempBuffer
    for (int bin = 0; bin < numBins; ++bin)
    {
        scheduler.addTaskResultsToBuffer(&tempBuffer[0], resultSize);
    }

    // Step 3: Find the maximum absolute value in the temp buffer
    float max = 0;
    for (int i = 0; i < resultSize; ++i)
    {
        float val = std::abs(tempBuffer[i]);
        if (val > max)
            max = val;
    }

    // Step 4: Normalize the result
    for (int i = 0; i < resultSize; ++i)
    {
        result[i] = tempBuffer[i] / max;
    }
}
