# ECE408/CS483 Final Project

## Team "upper-bound"
| Name | netID | UIN | Affiliation |
| ------------- |
| Haohang Huang | hhuang81 | 672130161 | UIUC on-campus |
| Jiayi Luo | jiayil5 | 676472385| UIUC on-campus |
| Hanhaotian Liu | hl14 | 656316093| UIUC on-campus |

## Table of Contents

* [Milestone 1: Due 3/7/2019 @5pm](#milestone-1)
* [Milestone 2: Due 3/14/2019 @5pm](#milestone-2)
* [Milestone 3: Due 4/4/2019 @5pm](#milestone-3)
* [Milestone 4: Due 4/18/2019 @5pm](#milestone-4)
* [Final Submission: Due 5/2/2019 @5pm (Reading Day)](#final-submission)

## Milestone 1

Due March 7 @ 5pm

### Time profile

A list of kernels that collectively consume more than 90% of the program time:

| Kernels | Time Percentage |
| ------------- |
| CUDA memcpy HtoD | 38.92% |
| cudnn::detail::implicit_convolve_sgemm | 20.80% |
| volta_cgemm_64x32_tn | 12.15% |
| op_generic_tensor_kernel | 7.20% |
| fft2d_c2r_32x32 | 5.87% |
| volta_sgemm_128x128_tn | 5.75% |
| cudnn::detail::pooling_fw_4d_kernel | 4.62% |
| fft2d_r2c_32x32 | 3.81% |

A list of API calls that collectively consume more than 90% of the program time:

| API calls | Time Percentage |
| ------------- |
| cudaStreamCreateWithFlags | 42.73% |
| cudaMemGetInfo | 33.95% |
| cudaFree | 21.13% |

Difference between kernels and API calls:
* Kernels are the user-implemented functions on GPU device with `__global__` declaration specifier. These functions will be executed by all threads in parallel. Each thread will be assigned a built-in variable (stored in its register) when executing the kernel functions.
* API calls are implemented and provided by the CUDA, whose names usually start with `cu*()` or `cuda*()`. They consist a minimal set of extensions to the C language and a runtime CUDA library. API calls include driver APIs and runtime APIs.

<!--
Kernel functions are compiled by Device Just-In-Time Compiler. API functions are compiled by Host C Compiler/Linker. (Note: this is actually not right, some runtime APIs can also be called in kernel functions, e.g. one kernel can launch another kernel).
-->

### Standard Output

Output of rai running MXNet on CPU:
~~~bash
Loading fashion-mnist data... done
Loading model... done
New Inference
EvalMetric: {'accuracy': 0.8236}
~~~

Output of rai running MXNet on GPU:
~~~bash
Loading fashion-mnist data... done
Loading model... done
New Inference
EvalMetric: {'accuracy': 0.8236}
~~~

As can be seen, the output accuracy are identical for CPU and GPU version.

### Run time

Program run time (in second):

| Hardware | user | system | elapsed |
| ------------- |
| CPU | 9.12 | 3.58 | 0:05.25 |
| GPU | 4.24 | 3.03 | 0:04.12 |

<!--
In Linux manual, the default format time string is:

%Uuser %Ssystem %Eelapsed %PCPU (%Xtext+%Ddata %Mmax)k
%Iinputs+%Ooutputs (%Fmajor+%Rminor)pagefaults %Wswaps

%U     Total number of CPU-seconds that the process spent in user mode.
%S     Total number of CPU-seconds that the process spent in kernel mode.
%E     Elapsed real time (in [hours:]minutes:seconds).
%P     Percentage of the CPU that this job got, computed as (%U + %S) / %E.
-->

As can be seen, the user time and elapsed real time of GPU version are both shorter than CPU version.

<div style="page-break-after: always;"></div>

## Milestone 2

Due March 14 @ 5pm

### CPU Implementation

A CPU version has been implemented in `new-forward.h` and tested execution time and correctness on different data size.

### Program Execution Time

| Data Size | user | system | elapsed |
| ------------- |
| 100 | 3.11 | 2.67 | 0:01.03 |
| 1000 | 4.39 | 2.78 | 0:01.92 |
| 10000 | 14.32 | 4.39 | 0:10.60 |

### Operator Time
| Data Size | Op time 1 | Op time 2 |
| ------------- |
| 100 | 0.031691 | 0.067879 |
| 1000 | 0.223601 | 0.686639 |
| 10000 | 2.158946 | 6.900892 |

### Correctness

| Data Size | Correctness  |
|-------------|
| 100       | 0.84 |
| 1000      | 0.852 |
| 10000     | 0.8397 |

<div style="page-break-after: always;"></div>

## Milestone 3

Due April 4 @ 5pm

### GPU Implementation

A GPU version has been implemented in `ece408_src/new-forward.cuh` and tested execution time and correctness on different data size. As can be seen, the GPU version brings significant performance improvements over the CPU version.

### Program Execution Time

| Data Size | user | system | elapsed |
| ------------- |
| 100 | 4.24 | 3.14 | 0:04.12 |
| 1000 | 4.34 | 3.42 | 0:04.21 |
| 10000 | 4.49 | 3.17 | 0:04.32 |

### Operator Time
| Data Size | Op time 1 | Op time 2 |
| ------------- |
| 100 | 0.000113 | 0.000240 |
| 1000 | 0.000894 | 0.002468 |
| 10000 | 0.008905 | 0.024344 |

### Correctness

| Data Size | Correctness  |
|-------------|
| 100       | 0.84 |
| 1000      | 0.852 |
| 10000     | 0.8397 |

### Performance Results

`nvprof` and NVVP are used to profile the program. Note that only the results from the default data size (10000) are presented in the report for conciseness.

#### `nvprof`

A list of kernels that collectively consume more than 90% of the program time:

| Kernels | Time Percentage |
| ------------- |
| mxnet::op::forward_kernel | 57.89% |
| CUDA memcpy HtoD | 28.49% |
| volta_sgemm_32x128_tn | 4.31% |
| mshadow::cuda::MapPlanLargeKernel | 4.17% |

A list of API calls that collectively consume more than 90% of the program time:

| API calls | Time Percentage |
| ------------- |
| cudaStreamCreateWithFlags | 49.20% |
| cudaMemGetInfo | 29.69% |
| cudaFree | 18.78% |

#### NVVP

The NVVP profile is shown below. Our `::forward_kernel` has `77%` importance of the total compute time.

![M3_NVVP](report_figures/m3_nvvp.jpg)

<!--
How to NVVP?
# Follow the tutorial and install NVVP on EWS.
open by: ~/software/cuda-10.0/bin/nvvp &

# Run nvprof command in yaml
- nvprof -o timeline.nvvp python m3.1.py # output profile for nvvp, download it from the AWS link
- nvprof --kernels "::forward:1" --analysis-metrics -o forward1_analysis.nvprof python m3.1.py
- nvprof --kernels "::forward:2" --analysis-metrics -o forward1_analysis.nvprof python m3.1.py

# Follow the AWS link and download file
unzip by: tar xf <filename.tar.gz>

# NVVP: File > Import > Nvprof > Single process
open timeline.nvvp file as timeline data
open *analysis.nvprof as event/metrics data

# Don't know how to save figure; Don't know how analysis.nvprof work

-->

<!--
<div style="page-break-after: always;"></div>

## Milestone 4

Due April 18 @ 5pm

| Deliverables |
| ------------ |
| Everything from Milestone 3 |
| Implement three GPU optimizations |
| Report: Describe the optimization |
| Report: demonstrate `nvprof` profiling the execution |
| Report: use NVVP to analyze your optimization |
| Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=m4` to mark your job for grading |

### 4.1 Add three GPU Optimization

For this milestone, you should attempt at least three GPU optimizations (see [optimizations](#optimizations)).

Describe the optimizations in your `report.pdf`.

### 4.2 Performance Analysis with `nvprof` and NVVP

Use the NVIDIA Visual Profiler and your analysis information to describe the effect that your optimizations had on the performance of your convolution.
If possible, you should try to separate the effect of each optimization in your analysis.

Use

    rai -p <project folder> --queue rai_amd64_ece408 --submit=m4

to submit your project folder.

## Final Submission

Due May 2 @ 5pm

| Deliverables |
| ------------ |
| Everything from Milestone 4 |
| Implement final GPU optimizations |
| Report: Describe and analyze the optimizations |
| Report: demonstrate `nvprof` profiling the execution |
| Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=final` to mark your job for grading |

### Optimized Layer

Optimize your GPU convolution (see [optimizations](#optimizations)).

Your implementation must work with `rai -p <project-folder> --queue rai_amd64_ece408 --submit=final`.
This means all your source files must be in `ece408_src`, and your implementation must work when they are copied to `src/operator/custom` in the MXNet tree, and `make` is invoked on the MXNet tree.
This is done in the provided `rai_build.yml`.
Likewise, the provided `final.py` provides an example of the script that will be used to time your implementation.

All of your code for this and the later milestones must be executed between `auto start = ...` and `auto end = ...` in `new-inl.h`.
The easiest way to ensure this is that all of your code should be in `forward()` or called by `forward()` from `new-forward.cuh` or `new-forward.h`.
Do not modify any timing-related code.

Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=final` to submit your project folder.

### Final Report

You've been building this final report through all the milestones.
Keep the content from the earlier milestones, but be sure to include the following:

* Your team name
* Your team member names
* your netids
* your UINs

The final report should include at least the following information for each optimization

1. **Optimization Approach and Results**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use nvprof and NVVP to justify your explanation.
    * Any external references used during identification or development of the optimization
    * How  your team organized and divided up this work.
2. **References** (as needed)
3. **(Optional) Suggestions for Improving Next Year**
-->
