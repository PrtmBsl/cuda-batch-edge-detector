// Copyright 2026 GPU Specialization Capstone Project
// Batch GPU Image Processing Pipeline (Grayscale -> Gaussian Blur -> Sobel)
// Styled according to the Google C++ Style Guide.

#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

// CUDA thread block dimensions and shared-memory halo radii.
constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;
constexpr int kGaussRadius = 2;  // 5x5 Gaussian kernel
constexpr int kSobelRadius = 1;  // 3x3 Sobel kernel

// 5x5 Gaussian filter weights (sum = 256).
__constant__ float kDeviceGaussianKernel[25] = {
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
    4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
    4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f};

constexpr float kHostGaussianKernel[25] = {
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
    4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
    4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f};

struct CliOptions {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  std::string csv_path = "data/output/benchmark_results.csv";
  bool verify_cpu = true;
};

struct BenchmarkRecord {
  std::string filename;
  int width;
  int height;
  float cpu_ms;
  float gpu_kernel_ms;
  float gpu_total_ms;
  int max_pixel_diff;
};

// Macro wrapper for strict CUDA runtime error checking.
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                                  \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// Kernel 1: Converts 1-, 3-, or 4-channel uint8 image into single-channel gray.
__global__ void RgbToGrayscaleKernel(const uint8_t* __restrict__ input,
                                     uint8_t* __restrict__ gray, int width,
                                     int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int pixel_idx = y * width + x;
  if (channels == 1) {
    gray[pixel_idx] = input[pixel_idx];
  } else {
    int base = pixel_idx * channels;
    float r = static_cast<float>(input[base]);
    float g = static_cast<float>(input[base + 1]);
    float b = static_cast<float>(input[base + 2]);
    float lum = 0.299f * r + 0.587f * g + 0.114f * b;
    gray[pixel_idx] = static_cast<uint8_t>(fminf(fmaxf(lum, 0.0f), 255.0f));
  }
}

// Kernel 2: 5x5 Gaussian Blur utilizing 2D Shared Memory Tiling with Halo.
__global__ void GaussianBlurSharedKernel(const uint8_t* __restrict__ input,
                                         uint8_t* __restrict__ output,
                                         int width, int height) {
  constexpr int kSharedW = kBlockDimX + 2 * kGaussRadius;
  constexpr int kSharedH = kBlockDimY + 2 * kGaussRadius;
  __shared__ float tile[kSharedH][kSharedW];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int total_shared_elems = kSharedW * kSharedH;
  int threads_per_block = blockDim.x * blockDim.y;

  // Cooperatively load tile + halo apron into shared memory with clamping.
  for (int i = tid; i < total_shared_elems; i += threads_per_block) {
    int sy = i / kSharedW;
    int sx = i % kSharedW;
    int gx = min(max(blockIdx.x * blockDim.x + sx - kGaussRadius, 0), width - 1);
    int gy = min(max(blockIdx.y * blockDim.y + sy - kGaussRadius, 0), height - 1);
    tile[sy][sx] = static_cast<float>(input[gy * width + gx]);
  }
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float sum = 0.0f;
  int k_idx = 0;
  for (int dy = -kGaussRadius; dy <= kGaussRadius; ++dy) {
    for (int dx = -kGaussRadius; dx <= kGaussRadius; ++dx) {
      float val = tile[threadIdx.y + kGaussRadius + dy]
                      [threadIdx.x + kGaussRadius + dx];
      sum += val * kDeviceGaussianKernel[k_idx++];
    }
  }
  output[y * width + x] = static_cast<uint8_t>(fminf(fmaxf(sum, 0.0f), 255.0f));
}

// Kernel 3: 3x3 Sobel Edge Magnitude utilizing 2D Shared Memory Tiling.
__global__ void SobelEdgeSharedKernel(const uint8_t* __restrict__ input,
                                      uint8_t* __restrict__ output, int width,
                                      int height) {
  constexpr int kSharedW = kBlockDimX + 2 * kSobelRadius;
  constexpr int kSharedH = kBlockDimY + 2 * kSobelRadius;
  __shared__ float tile[kSharedH][kSharedW];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int total_shared_elems = kSharedW * kSharedH;
  int threads_per_block = blockDim.x * blockDim.y;

  for (int i = tid; i < total_shared_elems; i += threads_per_block) {
    int sy = i / kSharedW;
    int sx = i % kSharedW;
    int gx = min(max(blockIdx.x * blockDim.x + sx - kSobelRadius, 0), width - 1);
    int gy = min(max(blockIdx.y * blockDim.y + sy - kSobelRadius, 0), height - 1);
    tile[sy][sx] = static_cast<float>(input[gy * width + gx]);
  }
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int cy = threadIdx.y + kSobelRadius;
  int cx = threadIdx.x + kSobelRadius;

  float gx = -tile[cy - 1][cx - 1] + tile[cy - 1][cx + 1] -
             2.0f * tile[cy][cx - 1] + 2.0f * tile[cy][cx + 1] -
             tile[cy + 1][cx - 1] + tile[cy + 1][cx + 1];

  float gy = -tile[cy - 1][cx - 1] - 2.0f * tile[cy - 1][cx] -
             tile[cy - 1][cx + 1] + tile[cy + 1][cx - 1] +
             2.0f * tile[cy + 1][cx] + tile[cy + 1][cx + 1];

  float mag = sqrtf(gx * gx + gy * gy);
  output[y * width + x] = static_cast<uint8_t>(fminf(fmaxf(mag, 0.0f), 255.0f));
}

// Serial CPU reference pipeline for speedup benchmarking and correctness check.
void RunCpuPipeline(const uint8_t* input, uint8_t* output, int width,
                    int height, int channels) {
  std::vector<uint8_t> gray(width * height);
  std::vector<uint8_t> blurred(width * height);

  // Stage 1: Grayscale
  for (int i = 0; i < width * height; ++i) {
    if (channels == 1) {
      gray[i] = input[i];
    } else {
      int base = i * channels;
      float lum = 0.299f * input[base] + 0.587f * input[base + 1] +
                  0.114f * input[base + 2];
      gray[i] = static_cast<uint8_t>(std::min(std::max(lum, 0.0f), 255.0f));
    }
  }

  // Stage 2: 5x5 Gaussian Blur
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      int k_idx = 0;
      for (int dy = -kGaussRadius; dy <= kGaussRadius; ++dy) {
        for (int dx = -kGaussRadius; dx <= kGaussRadius; ++dx) {
          int sx = std::min(std::max(x + dx, 0), width - 1);
          int sy = std::min(std::max(y + dy, 0), height - 1);
          sum += static_cast<float>(gray[sy * width + sx]) *
                 kHostGaussianKernel[k_idx++];
        }
      }
      blurred[y * width + x] =
          static_cast<uint8_t>(std::min(std::max(sum, 0.0f), 255.0f));
    }
  }

  // Stage 3: 3x3 Sobel Edge Detection
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto sample = [&](int sx, int sy) -> float {
        int cx = std::min(std::max(sx, 0), width - 1);
        int cy = std::min(std::max(sy, 0), height - 1);
        return static_cast<float>(blurred[cy * width + cx]);
      };
      float gx = -sample(x - 1, y - 1) + sample(x + 1, y - 1) -
                 2.0f * sample(x - 1, y) + 2.0f * sample(x + 1, y) -
                 sample(x - 1, y + 1) + sample(x + 1, y + 1);
      float gy = -sample(x - 1, y - 1) - 2.0f * sample(x, y - 1) -
                 sample(x + 1, y - 1) + sample(x - 1, y + 1) +
                 2.0f * sample(x, y + 1) + sample(x + 1, y + 1);
      float mag = std::sqrt(gx * gx + gy * gy);
      output[y * width + x] =
          static_cast<uint8_t>(std::min(std::max(mag, 0.0f), 255.0f));
    }
  }
}

bool HasImageExtension(const std::string& filename) {
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) return false;
  std::string ext = filename.substr(dot_pos);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
          ext == ".tga");
}

std::vector<std::string> ListInputImages(const std::string& dir_path) {
  std::vector<std::string> files;
  DIR* dir = opendir(dir_path.c_str());
  if (!dir) {
    std::cerr << "Error: Cannot open input directory: " << dir_path << "\n";
    return files;
  }
  struct dirent* entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name != "." && name != ".." && HasImageExtension(name)) {
      files.push_back(name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

void PrintUsage(const char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  --input <dir>     Directory containing input images "
               "(default: data/input)\n"
            << "  --output <dir>    Directory for processed output images "
               "(default: data/output)\n"
            << "  --csv <file>      Path to save CSV benchmark log "
               "(default: data/output/benchmark_results.csv)\n"
            << "  --no-verify       Skip CPU reference benchmark\n"
            << "  --help            Display this help message\n";
}

CliOptions ParseArguments(int argc, char** argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      opts.input_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      opts.output_dir = argv[++i];
    } else if (arg == "--csv" && i + 1 < argc) {
      opts.csv_path = argv[++i];
    } else if (arg == "--no-verify") {
      opts.verify_cpu = false;
    } else if (arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  return opts;
}

BenchmarkRecord ProcessSingleImage(const std::string& filename,
                                   const CliOptions& opts) {
  std::string full_in_path = opts.input_dir + "/" + filename;
  int width = 0;
  int height = 0;
  int channels = 0;
  uint8_t* h_input =
      stbi_load(full_in_path.c_str(), &width, &height, &channels, 0);
  if (!h_input) {
    std::cerr << "Failed to load image: " << full_in_path << "\n";
    std::exit(EXIT_FAILURE);
  }

  size_t raw_bytes = static_cast<size_t>(width) * height * channels;
  size_t gray_bytes = static_cast<size_t>(width) * height;
  std::vector<uint8_t> h_gpu_output(gray_bytes);

  // Allocate GPU device buffers.
  uint8_t* d_input = nullptr;
  uint8_t* d_gray = nullptr;
  uint8_t* d_blur = nullptr;
  uint8_t* d_edges = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, raw_bytes));
  CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
  CUDA_CHECK(cudaMalloc(&d_blur, gray_bytes));
  CUDA_CHECK(cudaMalloc(&d_edges, gray_bytes));

  // Create CUDA events for accurate hardware timing.
  cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
  CUDA_CHECK(cudaEventCreate(&start_total));
  CUDA_CHECK(cudaEventCreate(&stop_total));
  CUDA_CHECK(cudaEventCreate(&start_kernel));
  CUDA_CHECK(cudaEventCreate(&stop_kernel));

  dim3 block(kBlockDimX, kBlockDimY);
  dim3 grid((width + kBlockDimX - 1) / kBlockDimX,
            (height + kBlockDimY - 1) / kBlockDimY);

  CUDA_CHECK(cudaEventRecord(start_total));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, raw_bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaEventRecord(start_kernel));
  RgbToGrayscaleKernel<<<grid, block>>>(d_input, d_gray, width, height,
                                        channels);
  GaussianBlurSharedKernel<<<grid, block>>>(d_gray, d_blur, width, height);
  SobelEdgeSharedKernel<<<grid, block>>>(d_blur, d_edges, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop_kernel));

  CUDA_CHECK(cudaMemcpy(h_gpu_output.data(), d_edges, gray_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop_total));
  CUDA_CHECK(cudaEventSynchronize(stop_total));

  float gpu_kernel_ms = 0.0f;
  float gpu_total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, start_kernel, stop_kernel));
  CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms, start_total, stop_total));

  // Optional CPU verification & baseline benchmark.
  float cpu_ms = 0.0f;
  int max_diff = 0;
  if (opts.verify_cpu) {
    std::vector<uint8_t> h_cpu_output(gray_bytes);
    cudaEvent_t cpu_start, cpu_stop;
    CUDA_CHECK(cudaEventCreate(&cpu_start));
    CUDA_CHECK(cudaEventCreate(&cpu_stop));
    CUDA_CHECK(cudaEventRecord(cpu_start));
    RunCpuPipeline(h_input, h_cpu_output.data(), width, height, channels);
    CUDA_CHECK(cudaEventRecord(cpu_stop));
    CUDA_CHECK(cudaEventSynchronize(cpu_stop));
    CUDA_CHECK(cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_stop));
    CUDA_CHECK(cudaEventDestroy(cpu_start));
    CUDA_CHECK(cudaEventDestroy(cpu_stop));

    for (size_t i = 0; i < gray_bytes; ++i) {
      int diff = std::abs(static_cast<int>(h_gpu_output[i]) -
                          static_cast<int>(h_cpu_output[i]));
      max_diff = std::max(max_diff, diff);
    }
  }

  // Write processed PNG output image.
  size_t dot_pos = filename.find_last_of('.');
  std::string stem =
      (dot_pos == std::string::npos) ? filename : filename.substr(0, dot_pos);
  std::string out_path = opts.output_dir + "/" + stem + "_edges.png";
  if (!stbi_write_png(out_path.c_str(), width, height, 1, h_gpu_output.data(),
                      width)) {
    std::cerr << "Warning: Failed to write output image: " << out_path << "\n";
  }

  // Cleanup resources.
  stbi_image_free(h_input);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gray));
  CUDA_CHECK(cudaFree(d_blur));
  CUDA_CHECK(cudaFree(d_edges));
  CUDA_CHECK(cudaEventDestroy(start_total));
  CUDA_CHECK(cudaEventDestroy(stop_total));
  CUDA_CHECK(cudaEventDestroy(start_kernel));
  CUDA_CHECK(cudaEventDestroy(stop_kernel));

  return BenchmarkRecord{filename,      width,        height,  cpu_ms,
                         gpu_kernel_ms, gpu_total_ms, max_diff};
}

}  // namespace

int main(int argc, char** argv) {
  CliOptions opts = ParseArguments(argc, argv);

  int device_id = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  std::cout << "=== CUDA Batch Image Processing Pipeline ===\n"
            << "GPU Device: " << prop.name << " (SM " << prop.major << "."
            << prop.minor << ", Global Mem: "
            << (prop.totalGlobalMem / (1024 * 1024)) << " MB)\n\n";

  std::vector<std::string> images = ListInputImages(opts.input_dir);
  if (images.empty()) {
    std::cerr << "No supported images found in " << opts.input_dir << "\n";
    return EXIT_FAILURE;
  }

  mkdir(opts.output_dir.c_str(), 0755);

  std::vector<BenchmarkRecord> records;
  records.reserve(images.size());

  std::cout << std::left << std::setw(24) << "Image" << std::setw(14)
            << "Resolution" << std::setw(12) << "CPU (ms)" << std::setw(14)
            << "GPU Kern(ms)" << std::setw(14) << "GPU Tot(ms)" << std::setw(12)
            << "Speedup" << "MaxDiff\n";
  std::cout << std::string(96, '-') << "\n";

  for (const auto& img : images) {
    BenchmarkRecord rec = ProcessSingleImage(img, opts);
    records.push_back(rec);
    float speedup =
        (rec.gpu_kernel_ms > 0.0f) ? (rec.cpu_ms / rec.gpu_kernel_ms) : 0.0f;
    std::string res =
        std::to_string(rec.width) + "x" + std::to_string(rec.height);
    std::cout << std::left << std::setw(24) << rec.filename << std::setw(14)
              << res << std::fixed << std::setprecision(2) << std::setw(12)
              << rec.cpu_ms << std::setw(14) << rec.gpu_kernel_ms
              << std::setw(14) << rec.gpu_total_ms << std::setw(12) << speedup
              << rec.max_pixel_diff << "\n";
  }

  // Export CSV log file for execution artifact verification.
  std::ofstream csv_file(opts.csv_path);
  if (csv_file.is_open()) {
    csv_file << "filename,width,height,pixels,cpu_ms,gpu_kernel_ms,"
                "gpu_total_ms,kernel_speedup,total_speedup,max_pixel_diff\n";
    for (const auto& r : records) {
      long pixels = static_cast<long>(r.width) * r.height;
      float k_speedup =
          (r.gpu_kernel_ms > 0.0f) ? (r.cpu_ms / r.gpu_kernel_ms) : 0.0f;
      float t_speedup =
          (r.gpu_total_ms > 0.0f) ? (r.cpu_ms / r.gpu_total_ms) : 0.0f;
      csv_file << r.filename << "," << r.width << "," << r.height << ","
               << pixels << "," << r.cpu_ms << "," << r.gpu_kernel_ms << ","
               << r.gpu_total_ms << "," << k_speedup << "," << t_speedup << ","
               << r.max_pixel_diff << "\n";
    }
    std::cout << "\nSaved benchmark artifacts to: " << opts.csv_path << "\n";
  } else {
    std::cerr << "Warning: Could not write CSV file to " << opts.csv_path
              << "\n";
  }

  return EXIT_SUCCESS;
}
