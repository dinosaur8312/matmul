#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include <complex>
#include <vector>
#include <mkl.h>
#include <set>
#include "Task.h"

class BufferManager {
public:
    BufferManager();
    ~BufferManager();

    // Prepare buffers based on the maximum matrix sizes from the task pool
    void prepareInternalBuffers(int matrixSize, int numRHS, const std::multiset<Task> &taskPool);

    // Clear the output and matmat buffers
    void clearBuffers(int matrixSize, int numRHS);

    // Getters for the buffers (for use in matrix multiplication tasks)
    std::complex<float>* getInputBuffer();
    std::complex<float>* getOutputBuffer();
    std::complex<float>* getInternalBuffer();
    std::complex<float>* getMatmatBuffer();

private:
    int m_OutputbufferSize;
    std::complex<float> *m_inputBuffer;
    std::complex<float> *m_outputBuffer;
    std::complex<float> *m_internalBuffer;
    std::complex<float> *m_matmat;
};

#endif // BUFFER_MANAGER_H

