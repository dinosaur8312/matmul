#ifndef TASK_H
#define TASK_H

#include <vector>
#include <complex>

class Task {
public:
    Task(const int &m, const int &n, const int &r, const int &id);
    Task();

    // Accessor methods
    const int &M() const;
    const int &N() const;
    const int &R() const;
    double cost() const;
    const long int &size() const;

    // Task execution methods
    void setUp(int matrixSize);
    void matmat(const std::complex<float> *globalB,
                std::complex<float> *globalCPerBin,
                std::complex<float> *localB,
                std::complex<float> *localC,
                std::complex<float> *localMat,
                int nRHS,
                int matrixSize);
    bool operator<(const Task &other) const;

    // Create basis map and matrix entry
    int d_createBasisMap(int index, bool isSource, int maxSize);
    std::complex<float> d_matrixEntry(int rIndex, int cIndex, int maxLevel, bool local);
    bool isEmpty() const;

private:
    int m_M;   // Rows
    int m_N;   // Columns
    int m_R;   // Rank
    int m_cost; // Task cost
    int m_id;  // Task ID
    long int m_size;

    std::vector<int> m_srcBasisMap;
    std::vector<int> m_sinkBasisMap;

    std::complex<float> *m_pDense, *m_Qmat, *m_Rmat;
};

#endif
