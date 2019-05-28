#include <iostream>

#include <nccl.h>

#include <execinfo.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdexcept>

#include <unistd.h>

#pragma once

/**
 * @brief simple utility function to print an array of floats
 * that lives on the host.
 */
template<typename T>
void print(T *arr, int size, std::string name, cudaStream_t s) {

float *res = (T*)malloc(size*sizeof(T));
CUDA_CHECK(cudaMemcpyAsync(res, arr, size*sizeof(T), cudaMemcpyDeviceToHost, s));
CUDA_CHECK(cudaStreamSynchronize(s));

std::cout << name << " = [";
for(int i = 0; i < size; i++) {
    std::cout << res[i] << " ";

    if(i < size-1)
        std::cout << ", ";
    }

    std::cout << "]" << std::endl;
    free(res);
}

/**
 * @brief simple utility function to initialize all the items in a device
 * array to the given value.
 */
template<typename T>
void init_dev_arr(T *devArr, int size, T value, cudaStream_t s) {
    T *h_init = (T*)malloc(size * sizeof(T));
    for(int i = 0; i < size; i++)
        h_init[i] = value;
    CUDA_CHECK(cudaMemcpyAsync(devArr, h_init, size*sizeof(T), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    free(h_init);
}

/**
 * @brief simple utility function to verify all the items in a device
 * array equal the given value.
 */
template<typename T>
bool verify_dev_arr(T *devArr, int size, T value, cudaStream_t s) {

    bool ret = true;

    T *h_init = (T*)malloc(size * sizeof(T));
    CUDA_CHECK(cudaMemcpyAsync(h_init, devArr, size*sizeof(T), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    for(int i = 0; i < size; i++)
        if(h_init[i] != value)
            ret = false;

    free(h_init);

    return ret;
}
