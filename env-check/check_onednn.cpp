/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example weights_decompression_matmul.cpp
/// > Annotated version: @ref weights_decompression_matmul_cpp
///
/// @page weights_decompression_matmul_cpp_short
/// C++ API example demonstrating how one can use
/// [MatMul](@ref dev_guide_matmul) with compressed weights.
///
/// Concepts:
/// - Asymmetric quantization
///   - Scales: dnnl::primitive_attr::set_scales()
///   - Zero points: dnnl::primitive_attr::set_zero_points()
/// - [Operation fusion](@ref dev_guide_attributes_post_ops)
/// - Create primitive once, use multiple times
/// - Weights pre-packing: use #dnnl::memory::format_tag::any
///
/// @page weights_decompression_matmul_cpp MatMul Tutorial: weights decompression
/// @copydetails weights_decompression_matmul_cpp_short
///
/// Assumptions:
/// 1. The shape of the weights (matrix \f$B(K, N)\f$) is known in advance, the
///    data type is `int8_t` and shifted from 0 (i.e. the zero point is not 0).
/// 2. The source matrix \f$A\f$ and destination matrix \f$C\f$ have floating
///    point data type.
/// 3. Scaling (re-quantization) factor specified at run-time only.
///
/// Since the shape of weights is known in advance, the MatMul weights can be
/// created with format tag #dnnl::memory::format_tag::any to enable the library
/// to choose the most appropriate layout for best performance.
///
/// @warning
/// The format tag #dnnl::memory::format_tag::any doesn't work for memory
/// descriptors that have one or more unknown dimensions and/or strides.
///
/// @include weights_decompression_matmul.cpp

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"


using namespace dnnl;
using fp16 = sycl::half;
#define QK 64
// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();
    // printf("size %d\n", size);

    if (!handle) throw std::runtime_error("handle is nullptr.");

    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, handle, size).wait();
            }
        }
        return;
    }

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

namespace {

void init_vector(std::vector<int8_t> &v, int K, int N) {

    for(int i=0; i<N; i++){
        for(int j=0; j<K; j++){
            if(i==j){
                v[i*K+j]=1;
            }
        }
    }
}

} // namespace

int number_of_runs = 1;

// Create a MatMul primitive descriptor for the following op:
// C_f16 = A_f16 * (B_s8 - zp_B) * sc_B[:]
//
// Here:
// - Matrices A and C are of f16 data type.
// - The B matrix is stored as int8_t, its zero point is zp_B, and all its
//   dimensions are known. This matrix can be a matrix of compressed weights
//   in an MLP topology.
// - The weights scaling values are not known at the primitive creation time.
matmul::primitive_desc matmul_pd_create(
        int64_t M, int64_t N, int64_t K, int64_t G, const engine &eng) {

    memory::desc a_md({M, K}, memory::data_type::f16, {K, 1}); // M x K layout
    memory::desc b_md({K, N}, memory::data_type::s8, memory::format_tag::ba);
    memory::desc c_md({M, N}, memory::data_type::f16, {N, 1}); // M x N layout

    // Create attributes and indicate that the alpha and zero points are
    // runtime parameters
    primitive_attr attr;
    // Set scales with multiple scales along K and N dimensions and with groups along K.
    attr.set_scales(DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1), {G, 1}, memory::data_type::f32);
    // Set a single zero point with s8 data type.
    // attr.set_zero_points(
    //         DNNL_ARG_WEIGHTS, /* mask */ 0, {}, memory::data_type::s8);
    // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
    // integral primitives (in this example, matmul).
    attr.set_fpmath_mode(fpmath_mode::f16, true);

    // Create a MatMul primitive descriptor
    return matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
}

void prepare_input(memory &A_f16_mem, memory &sc_B_mem, memory &zp_B_mem) {
    int64_t M = A_f16_mem.get_desc().get_dims()[0];
    int64_t N = sc_B_mem.get_desc().get_dims()[0];
    int64_t K = A_f16_mem.get_desc().get_dims()[1];
    int64_t NUM_G = sc_B_mem.get_desc().get_dims()[1];

    std::vector<fp16> A_f16(M * K, 1);

    std::vector<float> sc_B(NUM_G * N);

    for(int i=0; i<NUM_G; i++){
        for(int j=0; j<N; j++){
            sc_B[i*N+j]=j*NUM_G+i+1;
        }
    }

    // int8_t zp_B = 0;

    write_to_dnnl_memory(A_f16.data(), A_f16_mem);
    // write_to_dnnl_memory(&zp_B, zp_B_mem);
    write_to_dnnl_memory(sc_B.data(), sc_B_mem);
}

void infer(const matmul &matmul_p, int64_t M, int64_t N, int64_t K, int64_t G,
        const memory &B_s8_mem, const engine &eng) {
    // input of the current layer / operation
    memory A_f16_mem({{M, K}, memory::data_type::f16, {K, 1}}, eng);
    // De-quantization parameters (eg. Scale and Shift)
    const int64_t n_groups = K / G;
    memory sc_B_mem({{N, n_groups}, memory::data_type::f32, {1, N}}, eng);
    memory zp_B_mem({{1}, memory::data_type::s8, {1}}, eng);

    // the function below fills dnnl::memory with some values
    // these memories, typically, come from the previous layers / operations
    // with meaningful data inside
    prepare_input(A_f16_mem, sc_B_mem, zp_B_mem);

    // output - no initialization required
    memory C_f16_mem({{M, N}, memory::data_type::f16, {N, 1}}, eng);

    stream s(eng);
    for (int run = 0; run < number_of_runs; ++run)
        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_f16_mem}, {DNNL_ARG_WEIGHTS, B_s8_mem},
                        {DNNL_ARG_DST, C_f16_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_B_mem},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                                zp_B_mem}});
    s.wait();

    std::vector<fp16> c_cpu(M * N);
    auto sycl_queue
        = dnnl::sycl_interop::get_queue(s);
    sycl_queue.memcpy(c_cpu.data(), (fp16*)C_f16_mem.get_data_handle(), M * N * sizeof(fp16)).wait();
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            printf("%f ", static_cast<float>(c_cpu[i*N+j]));
        }
        printf("\n");
    }
}

void weights_decompression_matmul(engine::kind engine_kind) {
    engine eng(engine_kind, 0);

    const int64_t K = 256;
    const int64_t N = 256;
    const int64_t M = 2;
    // Quantization Group size for scales. Must be divisible by 32.
    const int64_t G = 64;

    auto matmul_pd = matmul_pd_create(M, N, K, G, eng);

    // Original weights stored as float in a known format
    std::vector<int8_t> B_s8(K * N);

    // Pre-packed weights stored as int8_t
    memory B_s8_mem(matmul_pd.weights_desc(), eng);
    init_vector(B_s8, K, N);
    {
        stream s(eng);
        // memory B_f16_mem(
        //         {{K, N}, memory::data_type::f16, memory::format_tag::ab}, eng);
        write_to_dnnl_memory(B_s8.data(), B_s8_mem);
        // reorder(B_f16_mem, B_s8_mem).execute(s, B_f16_mem, B_s8_mem);
        s.wait();

        // std::vector<int8_t> B_s8_cpu(K * N);
        // auto sycl_queue
        //     = dnnl::sycl_interop::get_queue(s);
        // sycl_queue.memcpy(B_s8_cpu.data(), (int8_t*)B_s8_mem.get_data_handle(), K *N).wait();
        // for(int i=0; i<N; i++){
        //     for(int j=0; j<K; j++){
        //         printf("%d ", B_s8_cpu[i*N+j]);
        //     }
        //     printf("\n");
        // }
    }

    matmul matmul_p(matmul_pd);

    infer(matmul_p, M, N, K, G, B_s8_mem, eng);
}

int main(int argc, char **argv) {
    engine::kind engine_kind = dnnl::engine::kind::gpu;
    // GPU is not supported
    // if (engine_kind != engine::kind::cpu) return 0;
    weights_decompression_matmul(engine_kind);
}
