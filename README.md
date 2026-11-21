# CUDA Batch Image Processing & Sobel Edge Detection Pipeline

## Project Overview
This capstone project implements a high-throughput, GPU-accelerated batch image processing pipeline using NVIDIA CUDA C++. Given an input directory of images, the application executes a three-stage computer vision preprocessing workflow across all images in a single batch run, validates numerical accuracy against a serial CPU baseline, and exports processed edge maps alongside a CSV benchmark report.

## GPU Architecture & CUDA Kernels
1. **RGB-to-Grayscale Conversion (`RgbToGrayscaleKernel`)**: Maps each output pixel to a 2D CUDA thread (`16x16` thread blocks) to compute ITU-R BT.601 luminance (`0.299R + 0.587G + 0.114B`) in global memory.
2. **Tiled 5×5 Gaussian Blur (`GaussianBlurSharedKernel`)**: Uses `__constant__` memory for filter weights and cooperatively loads `20x20` image tiles (including a 2-pixel halo apron) into low-latency `__shared__` memory with boundary clamping to eliminate redundant global memory reads.
3. **Tiled 3×3 Sobel Edge Detection (`SobelEdgeSharedKernel`)**: Loads `18x18` shared-memory tiles (1-pixel halo) to compute horizontal ($G_x$) and vertical ($G_y$) intensity gradients and outputs edge magnitude $\sqrt{G_x^2 + G_y^2}$.

## Prerequisites & Installation
* **Hardware**: NVIDIA GPU (Compute Capability 5.0+)
* **Software**: CUDA Toolkit (`nvcc`), `g++` (C++17), `make`, `curl`
* **Image I/O**: Uses single-header `stb_image.h` and `stb_image_write.h` (automatically downloaded via `make setup`).

## Build & Run Instructions
```bash
# 1. Download stb_image headers & compile the CUDA executable
make

# 2. Run the automated end-to-end batch script
chmod +x run.sh
./run.sh
```

### Command-Line Interface (CLI) Arguments
```bash
./batch_edge_detector --input <input_dir> --output <output_dir> --csv <csv_path> [--no-verify] [--help]
```
* `--input <dir>`: Directory containing `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.tga` images (default: `data/input`).
* `--output <dir>`: Destination directory for `*_edges.png` outputs (default: `data/output`).
* `--csv <file>`: Output path for execution timing and speedup metrics (default: `data/output/benchmark_results.csv`).
* `--no-verify`: Disables the serial CPU reference pass when running pure GPU throughput benchmarks.

## Key Engineering Lessons & Future Improvements
* **Shared Memory Halo Tiling**: Moving 5×5 neighborhood reads from global memory into `__shared__` memory reduced redundant memory transactions by ~25× per pixel block, yielding substantial kernel speedups on 1080p and 4K resolutions while maintaining `<= 1` uint8 rounding difference versus the CPU reference.
* **Host-to-Device Transfer Overhead**: Profiling with `cudaEvent_t` revealed that PCIe `cudaMemcpy` calls account for a large share of total wall-clock runtime on smaller images.
* **Next Steps**: Implement **CUDA Streams** with pinned host memory (`cudaMallocHost`) to overlap `cudaMemcpyAsync` transfers of image $N+1$ with kernel execution on image $N$.
