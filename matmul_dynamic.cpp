#include <iostream>
#include <fstream>
#include <memory>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <complex>
#include <vector>
#include <queue>
#include <mutex>
#include <set>
#include <omp.h>
#include <mkl.h>

// Global task pool and mutex for synchronization
std::queue<Task> taskPool; // Shared task pool
std::mutex taskMutex;      // Mutex to synchronize access to the task pool

// Function to get the next available task from the pool
bool getNextTask(Task &task)
{
    std::lock_guard<std::mutex> lock(taskMutex);
    if (!taskPool.empty())
    {
        task = taskPool.front();
        taskPool.pop();
        return true;
    }
    return false; // No tasks left
}

// Matrix multiplication using MKL (unchanged)
void matrixXmatrixMKLfull(const std::complex<float> *matrixA,
                          const std::complex<float> *matrixB,
                          std::complex<float> *result,
                          const int &rowSize, const int &colSize,
                          const int &midSize,
                          const int &lda, const int &ldb, const int &ldc,
                          char transA, char transB)
{
    if (rowSize <= 0 || colSize <= 0 || midSize <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
        return;

    const std::complex<float> alpha(1, 0);
    const std::complex<float> beta(0, 0);
    int least_lda = (transA == 'N') ? rowSize : midSize;
    int true_k = (transA == 'N') ? midSize : rowSize;
    int least_ldb = (transB == 'N') ? midSize : colSize;
    int true_n = (transB == 'N') ? colSize : midSize;

    cgemm(&transA, &transB, &least_lda, &true_n, &true_k, (MKL_Complex8 *)&alpha,
          (MKL_Complex8 *)matrixA, &lda, (MKL_Complex8 *)matrixB, &ldb,
          (MKL_Complex8 *)&beta, (MKL_Complex8 *)result, &ldc);
}

double matrixXmatrixMKLPT(const std::complex<float> *matrixA,
                          const std::complex<float> *matrixB,
                          std::complex<float> *result,
                          const int &rowSize, const int &colSize,
                          const int &midSize,
                          const int &lda, const int &ldb, const int &ldc,
                          char transA, char transB, int numRepeat)
{
    if (rowSize <= 0 || colSize <= 0 || midSize <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
        return 0.0;

    const std::complex<float> alpha(1, 0);
    const std::complex<float> beta(0, 0);
    int least_lda = rowSize;
    int true_k = midSize;
    int true_n = colSize;

    double time_st = dsecnd();
    for (auto i = 0; i < numRepeat; i++)
    {
        cgemm(&transA, &transB, &least_lda, &true_n, &true_k, (MKL_Complex8 *)&alpha,
              (MKL_Complex8 *)matrixA, &lda, (MKL_Complex8 *)matrixB, &ldb,
              (MKL_Complex8 *)&beta, (MKL_Complex8 *)result, &ldc);
    }
    double time_end = dsecnd();
    double time_avg = (time_end - time_st) / numRepeat;
    double gflop = (2.0 * least_lda * true_n * true_k) * 1E-9;

    return gflop / time_avg;
}

void vectorPlusEqualVector_MKL(std::complex<float> *v1, const int &size, const std::complex<float> *v2)
{
    int incx = 1;
    int incy = 1;
    float alpha = +1;
    int complexSize = size * 2;

    saxpy(&complexSize, &alpha, (float *)v2, &incx, (float *)v1, &incy);
}




// Dynamic task scheduling within matrix multiplication
void dynamicMatmatScheduler(std::vector<std::complex<float>> &globalB,
                            std::vector<std::complex<float>> &globalCPerBin,
                            int matrixSize, int numRHS)
{
    // Use OpenMP or threads to parallelize task execution
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        Task task;
        while (getNextTask(task))
        { // Dynamically fetch tasks
            // Thread will now process the fetched task
            task.matmat(globalB.data(), globalCPerBin.data(), nullptr, nullptr, nullptr, numRHS, matrixSize);
        }
    }
}

// Parse MIL file and build tasks
void parseMILFile(const std::string &MILFile, const std::string &filter, TaskScheduler &scheduler)
{
    std::fstream fMil;
    fMil.open(MILFile, std::fstream::in);
    std::string thisLine = "";
    int numTasks = 0;
    std::set<int> includedMILs;

    while (!fMil.eof())
    {
        std::getline(fMil, thisLine);
        std::vector<std::string> words = SplitStringOnWhiteSpace(thisLine);
        if (words.size() < 7)
            continue;

        if (!filter.empty() && filter != words[3])
            continue;

        int index = atoi(words[2].c_str());
        int M = atoi(words[4].c_str());
        int N = atoi(words[5].c_str());
        int R = atoi(words[6].c_str());

        if (includedMILs.find(index) == includedMILs.end())
        {
            includedMILs.insert(index);
        }
        else
        {
            continue;
        }

        Task task(M, N, R, numTasks);
        scheduler.addTask(task);
        ++numTasks;
    }

    fMil.close();
}

// Add all tasks to the shared task pool
void addToTaskPool(std::multiset<Task> &allTasks)
{
    std::lock_guard<std::mutex> lock(taskMutex); // Ensure only one thread modifies the pool
    for (const auto &task : allTasks)
    {
        taskPool.push(task);
    }
}

// Parse MIL file, add tasks, and run matrix multiplication with dynamic scheduling
void runDynamicScheduler(const std::string &MILFile, int numThreads, int numRepeat, int numRHS, const std::string &filter)
{
    // Parse MIL file and build tasks
    std::multiset<Task> allTasks;
    parseMILFileAndBuildTasks(MILFile, allTasks, filter);

    // Add tasks to the task pool
    addToTaskPool(allTasks);

    // Setup matrices and buffers (globalB, globalCPerBin, etc.)
    int matrixSize = 1000; // Example matrix size, adjust as necessary
    std::vector<std::complex<float>> globalB(matrixSize * numRHS);
    std::vector<std::complex<float>> globalCPerBin(matrixSize * numRHS);

    // Start dynamic task scheduling and processing
    dynamicMatmatScheduler(globalB, globalCPerBin, matrixSize, numRHS);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <MILFile> <numThreads> <numRepeat> <numRHS>" << std::endl;
        return 1;
    }

    std::string MILFile = argv[1];
    int numThreads = std::atoi(argv[2]);
    int numRepeat = std::atoi(argv[3]);
    int numRHS = std::atoi(argv[4]);
    std::string filter = argv[5];

    // Run the dynamic scheduler
    runDynamicScheduler(MILFile, numThreads, numRepeat, numRHS, filter);

    return 0;
}

class Task
{
public:
    Task(const int &m, const int &n, const int &r, const int &id) : m_M(m), m_N(n), m_R(r), m_id(id)
    {
        m_cost = (r == 0) ? (m * n) : (r * (m + n));
        m_size = m;
        m_size *= n;
    }

    const int &M() const { return m_M; }
    const int &N() const { return m_N; }
    const int &R() const { return m_R; }
    bool operator<(const Task &other) const
    {
        return m_cost < other.m_cost;
    }
    const long int &size() const { return m_size; }
    int d_createBasisMap(int index, bool isSource, int maxSize)
    {
        long long hash = (2 * m_id + 23);
        hash *= index;
        hash *= index;

        long long t2 = m_id;
        t2 *= m_id;
        t2 += 45;
        t2 *= index;

        hash += t2;
        hash += 43;
        if (isSource)
        {
            hash *= 100;
            hash += 1423;
        }
        return (int)(hash % maxSize);
    }
    std::complex<float> d_matrixEntry(int rIndex, int cIndex,
                                      const TreeNode *tHead,
                                      int maxLevel, bool local)
    {
        std::complex<float> result;
        for (int ii = 0; ii < 10; ii++)
        {
            long long hash = (cIndex + 1001) * rIndex + ii * 500;
            if (!local)
            {
                hash *= 100;
                hash += 1423;
            }
            long max = (long)std::pow(2, maxLevel);
            int tIndex = (int)(hash % max);
            std::complex<float> val = tHead->getVal(tIndex, maxLevel);
            if (std::abs(val) > 1e-31)
                val /= std::sqrt(std::abs(val));
            result += val;
        }
        return result;
    }

    void matmat(const std::complex<float> *globalB,
                std::complex<float> *globalCPerBin,
                std::complex<float> *localB,
                std::complex<float> *localC,
                std::complex<float> *localMat,
                int nRHS,
                int matrixSize)
    {
        for (int j = 0; j < nRHS; ++j)
            for (int i = 0; i < m_N; ++i)
            {
                localB[i + m_N * j] = globalB[m_srcBasisMap[i] + matrixSize * j];
            }
        if (m_R == 0)
        {
            matrixXmatrixMKLfull(&m_pDense[0], &localB[0], &localC[0], m_M, nRHS, m_N, m_M, m_N, m_M, 'N', 'N');
        }
        else
        {
            matrixXmatrixMKLfull(&m_Rmat[0], &localB[0], &localMat[0], m_R, nRHS, m_N, m_R, m_N, m_R, 'N', 'N');
            matrixXmatrixMKLfull(&m_Qmat[0], &localMat[0], &localC[0], m_M, nRHS, m_R, m_M, m_R, m_M, 'N', 'N');
        }
        for (int j = 0; j < nRHS; ++j)
            for (int i = 0; i < m_M; ++i)
            {
                globalCPerBin[m_sinkBasisMap[i] + matrixSize * j] += localC[i + m_M * j];
            }
    }

    void setUp(int matrixSize, const TreeNode *tHead, const int &tLevel)
    {
        m_srcBasisMap.resize(m_N);
        //  printf("\t\t\tSetting up %d source basis maps\n", m_N);
        for (int ii = 0; ii < m_N; ++ii)
        {
            m_srcBasisMap[ii] = d_createBasisMap(ii, true, matrixSize);
            // printf("Source basis map %d: %d\n", ii, m_srcBasisMap[ii]);
        }

        //  printf("\t\t\tSetting up %d sink basis maps\n", m_M);
        m_sinkBasisMap.resize(m_M);
        for (int ii = 0; ii < m_M; ++ii)
        {
            m_sinkBasisMap[ii] = d_createBasisMap(ii, false, matrixSize);
            // printf("Sink basis map %d: %d\n", ii, m_sinkBasisMap[ii]);
        }

        if (m_R == 0)
        {
            m_pDense = (std::complex<float> *)mkl_malloc(m_M * m_N * sizeof(std::complex<float>), 128);
            for (int ii = 0; ii < m_M; ii++)
            {
                for (int jj = 0; jj < m_N; jj++)
                    m_pDense[ii + jj * m_M] = d_matrixEntry(m_sinkBasisMap[ii],
                                                            m_srcBasisMap[jj],
                                                            tHead, tLevel, true);
            }
        }
        else
        {
            m_Qmat = (std::complex<float> *)mkl_malloc(m_M * m_R * sizeof(std::complex<float>), 128);
            m_Rmat = (std::complex<float> *)mkl_malloc(m_R * m_N * sizeof(std::complex<float>), 128);

            for (int ii = 0; ii < m_M; ++ii)
            {
                for (int jj = 0; jj < m_R; ++jj)
                    m_Qmat[ii + jj * m_M] = d_matrixEntry(m_sinkBasisMap[ii],
                                                          jj,
                                                          tHead, tLevel, false);
            }
            for (int ii = 0; ii < m_R; ++ii)
            {
                for (int jj = 0; jj < m_N; ++jj)
                    m_Rmat[ii + jj * m_R] = d_matrixEntry(m_srcBasisMap[jj], ii, tHead, tLevel, false);
            }
        }
    }

    double cost() const
    {
        return (m_R != 0) ? 1.0 * m_R * (m_M + m_N) : 1.0 * m_M * m_N;
    }

private:
    Task() { ; }
    int m_M;
    int m_N;
    int m_R;
    int m_cost;
    int m_id;
    long int m_size;
    std::vector<int> m_srcBasisMap;
    std::vector<int> m_sinkBasisMap;

    std::complex<float> *m_pDense, *m_Qmat, *m_Rmat;
};
