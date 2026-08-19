#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: cuda_peer_copy_probe SRC_GPU DST_GPU SIZE_MIB REPEATS\n";
        return 2;
    }
    const int source_device = std::atoi(argv[1]);
    const int destination_device = std::atoi(argv[2]);
    const size_t size_bytes =
        static_cast<size_t>(std::strtoull(argv[3], nullptr, 10)) * 1024 * 1024;
    const int repeats = std::atoi(argv[4]);

    int can_access = 0;
    check(cudaDeviceCanAccessPeer(&can_access, destination_device, source_device),
          "cudaDeviceCanAccessPeer");
    if (!can_access) {
        throw std::runtime_error("destination cannot access source peer");
    }

    unsigned char* source = nullptr;
    unsigned char* destination = nullptr;
    check(cudaSetDevice(source_device), "cudaSetDevice(source)");
    check(cudaMalloc(&source, size_bytes), "cudaMalloc(source)");
    check(cudaMemset(source, 73, size_bytes), "cudaMemset(source)");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(source)");

    check(cudaSetDevice(destination_device), "cudaSetDevice(destination)");
    check(cudaMalloc(&destination, size_bytes), "cudaMalloc(destination)");
    check(cudaMemset(destination, 0, size_bytes), "cudaMemset(destination)");
    cudaError_t enable_status = cudaDeviceEnablePeerAccess(source_device, 0);
    if (enable_status != cudaSuccess &&
        enable_status != cudaErrorPeerAccessAlreadyEnabled) {
        check(enable_status, "cudaDeviceEnablePeerAccess");
    }

    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t end;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&end), "cudaEventCreate(end)");

    std::vector<float> elapsed_ms;
    for (int iteration = 0; iteration < repeats + 2; ++iteration) {
        check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        check(cudaMemcpyPeerAsync(destination, destination_device, source,
                                  source_device, size_bytes, stream),
              "cudaMemcpyPeerAsync");
        check(cudaEventRecord(end, stream), "cudaEventRecord(end)");
        check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, start, end),
              "cudaEventElapsedTime");
        if (iteration >= 2) elapsed_ms.push_back(milliseconds);
    }

    std::vector<unsigned char> host(size_bytes);
    check(cudaMemcpy(host.data(), destination, size_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(verification)");
    if (!std::all_of(host.begin(), host.end(),
                     [](unsigned char value) { return value == 73; })) {
        throw std::runtime_error("verification failed");
    }
    std::sort(elapsed_ms.begin(), elapsed_ms.end());
    const float median_ms = elapsed_ms[elapsed_ms.size() / 2];
    const double gib_per_second =
        (static_cast<double>(size_bytes) / (1024.0 * 1024.0 * 1024.0)) /
        (median_ms / 1000.0);
    std::cout << "CUDA_PEER_COPY source=" << source_device
              << " destination=" << destination_device
              << " size_mib=" << argv[3] << " median_ms=" << median_ms
              << " gib_per_second=" << gib_per_second << " verified=true\n";

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    cudaFree(destination);
    cudaSetDevice(source_device);
    cudaFree(source);
    return 0;
}
