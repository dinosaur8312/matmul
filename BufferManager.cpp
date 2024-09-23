#include "BufferManager.h"
#include <iostream>
#include <algorithm>

BufferManager::BufferManager() : m_OutputbufferSize(0), m_inputBuffer(nullptr), 
                                 m_outputBuffer(nullptr), m_internalBuffer(nullptr), m_matmat(nullptr) {}

BufferManager::~BufferManager() {
    if (m_inputBuffer) mkl_free(m_inputBuffer);
    if (m_outputBuffer) mkl_free(m_outputBuffer);
    if (m_internalBuffer) mkl_free(m_internalBuffer);
    if (m_matmat) mkl_free(m_matmat);
}

void BufferManager::prepareInternalBuffers(int matrixSize, int numRHS, const std::multiset<Task> &taskPool) {
    int maxSrcBuffer = -1;
    int maxTestBuffer = -1;
    int maxRankBuffer = -1;

    // Find the maximum buffer sizes from the tasks
    for (const auto &task : taskPool) {
        maxSrcBuffer = std::max(maxSrcBuffer, task.N());
        maxTestBuffer = std::max(maxTestBuffer, task.M());
        maxRankBuffer = std::max(maxRankBuffer, task.R());
    }

    // Allocate the buffers
    m_inputBuffer = static_cast<std::complex<float>*>(mkl_malloc(numRHS * maxSrcBuffer * sizeof(std::complex<float>), 128));
    m_outputBuffer = static_cast<std::complex<float>*>(mkl_malloc(numRHS * maxTestBuffer * sizeof(std::complex<float>), 128));
    m_internalBuffer = static_cast<std::complex<float>*>(mkl_malloc(numRHS * maxRankBuffer * sizeof(std::complex<float>), 128));
    m_matmat = static_cast<std::complex<float>*>(mkl_malloc(numRHS * matrixSize * sizeof(std::complex<float>), 128));

    // Print allocated buffer sizes for debugging
    std::cout << "Allocated input buffer of size " << numRHS << " x " << maxSrcBuffer << std::endl;
    std::cout << "Allocated output buffer of size " << numRHS << " x " << maxTestBuffer << std::endl;
    std::cout << "Allocated internal buffer of size " << numRHS << " x " << maxRankBuffer << std::endl;
    std::cout << "Allocated matmat buffer of size " << numRHS << " x " << matrixSize << std::endl;

    // Store the size of the output buffer
    m_OutputbufferSize = maxTestBuffer;
}

void BufferManager::clearBuffers(int matrixSize, int numRHS) {
    memset(m_outputBuffer, 0, sizeof(std::complex<float>) * m_OutputbufferSize * numRHS);
    memset(m_matmat, 0, sizeof(std::complex<float>) * matrixSize * numRHS);
}

std::complex<float>* BufferManager::getInputBuffer() { return m_inputBuffer; }
std::complex<float>* BufferManager::getOutputBuffer() { return m_outputBuffer; }
std::complex<float>* BufferManager::getInternalBuffer() { return m_internalBuffer; }
std::complex<float>* BufferManager::getMatmatBuffer() { return m_matmat; }
