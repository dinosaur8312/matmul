#ifndef TASK_SCHEDULER_H
#define TASK_SCHEDULER_H

#include <complex>
#include <vector>
#include <string>
#include "Task.h"
#include "BufferManager.h"

class TaskScheduler
{
public:
    TaskScheduler(const std::string &MILFile, const std::string &filter, int numThreads, int matrixSize, int numRHS);

    void addTask(const Task &task);
    Task getTask();
    void dynamicMatmatScheduler(std::vector<std::complex<float>> &globalB,
                                std::vector<std::complex<float>> &globalCPerBin,
                                int matrixSize, int numRHS);
    void addTaskResultsToBuffer(std::complex<float> *buffer, int bufferSize);

private:
    void parseMILFile(const std::string &MILFile, const std::string &filter);

    std::multiset<Task> taskPool;
    std::vector<std::complex<float>> taskResults;
    BufferManager bufferManager;
    int currentTaskIndex;
};

#endif // TASK_SCHEDULER_H
