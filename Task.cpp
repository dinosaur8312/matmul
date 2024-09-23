#include "Task.h"
#include <mkl.h>
#include <cmath>
#include <iostream>

Task::Task(const int &m, const int &n, const int &r, const int &id)
    : m_M(m), m_N(n), m_R(r), m_id(id) {
    m_cost = (r == 0) ? (m * n) : (r * (m + n));
    m_size = m;
    m_size *= n;
}

Task::Task() : m_M(0), m_N(0), m_R(0), m_cost(0), m_id(-1), m_size(0) {}

const int &Task::M() const { return m_M; }
const int &Task::N() const { return m_N; }
const int &Task::R() const { return m_R; }
const long int &Task::size() const { return m_size; }

bool Task::operator<(const Task &other) const {
    return m_cost < other.m_cost;
}

void Task::setUp(int matrixSize) {
    // Allocate matrices
    if (m_R == 0) {
        m_pDense = (std::complex<float> *)mkl_malloc(m_M * m_N * sizeof(std::complex<float>), 128);
        // Fill matrix entries for dense matrices
    } else {
        m_Qmat = (std::complex<float> *)mkl_malloc(m_M * m_R * sizeof(std::complex<float>), 128);
        m_Rmat = (std::complex<float> *)mkl_malloc(m_R * m_N * sizeof(std::complex<float>), 128);
        // Fill matrix entries for factorized matrices
    }
}

int Task::d_createBasisMap(int index, bool isSource, int maxSize) {
    long long hash = (2 * m_id + 23);
    hash *= index;
    hash *= index;

    long long t2 = m_id;
    t2 *= m_id;
    t2 += 45;
    t2 *= index;

    hash += t2;
    hash += 43;
    if (isSource) {
        hash *= 100;
        hash += 1423;
    }
    return (int)(hash % maxSize);
}

std::complex<float> Task::d_matrixEntry(int rIndex, int cIndex, int maxLevel, bool local) {
    std::complex<float> result(0, 0);
    for (int i = 0; i < 10; i++) {
        long long hash = (cIndex + 1001) * rIndex + i * 500;
        if (!local) {
            hash *= 100;
            hash += 1423;
        }
        result += std::complex<float>(hash % maxLevel, (hash + 1) % maxLevel);  // Example calculation
    }
    return result;
}

void Task::matmat(const std::complex<float> *globalB, std::complex<float> *globalCPerBin,
                  std::complex<float> *localB, std::complex<float> *localC,
                  std::complex<float> *localMat, int nRHS, int matrixSize) {
    for (int j = 0; j < nRHS; ++j)
        for (int i = 0; i < m_N; ++i)
            localB[i + m_N * j] = globalB[m_srcBasisMap[i] + matrixSize * j];

    // Use m_pDense or m_Qmat, m_Rmat for matrix operations
    if (m_R == 0) {
        // Matrix multiplication for dense matrices
    } else {
        // Matrix multiplication for factorized matrices
    }
}

bool Task::isEmpty() const {
    return m_id == -1;
}
