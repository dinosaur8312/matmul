#include "TaskScheduler.h"
#include "Utilities.h"
#include <fstream>
#include <iostream>
#include <set>

TaskScheduler::TaskScheduler(const std::string &MILFile, const std::string &filter, int numThreads, int matrixSize, int numRHS)
    : currentTaskIndex(0), bufferManager(matrixSize, numRHS) {
    // Parse the file and add tasks to the task pool
    parseMILFile(MILFile, filter);
    
    // Allocate necessary buffers after task pool initialization
    bufferManager.prepareInternalBuffers(taskPool, numThreads);
}

void TaskScheduler::parseMILFile(const std::string &MILFile, const std::string &filter) {
    std::fstream fMil;
    fMil.open(MILFile, std::fstream::in);
    std::string thisLine = "";
    int numTasks = 0;
    std::set<int> includedMILs;

    while (!fMil.eof()) {
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

        if (includedMILs.find(index) == includedMILs.end()) {
            includedMILs.insert(index);
        } else {
            continue;
        }

        Task task(M, N, R, numTasks);
        addTask(task);
        ++numTasks;
    }

    fMil.close();
}

void TaskScheduler::addTask(const Task &task) {
    taskPool.insert(task);
}

Task TaskScheduler::getTask() {
    auto it = taskPool.begin();
    Task task = *it;
    taskPool.erase(it);
    return task;
}

void TaskScheduler::dynamicMatmatScheduler(std::vector<std::complex<float>> &globalB,
                                           std::vector<std::complex<float>> &globalCPerBin,
                                           int matrixSize, int numRHS) {
    #pragma omp parallel
    {
        while (!taskPool.empty()) {
            Task task = getTask();
            bufferManager.processTask(task, globalB, globalCPerBin);
        }
    }
}

void TaskScheduler::addTaskResultsToBuffer(std::complex<float> *buffer, int bufferSize) {
    bufferManager.reduceResults(buffer, bufferSize);
}
