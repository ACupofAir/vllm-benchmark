/*
 * source /opt/intel/oneapi/2025.0/oneapi-vars.sh
 *
 * dpcpp dg2_cpy.cpp -o dg2_cpy.exe -DXE_PLUS
 *
 * export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
 * export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
 *
 * ./dg2_cpy.exe
 */
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

uint16_t pattern_counter = 0xa770;
#define DG2_NUM (2)
typedef uint32_t pattern_t;
using message_t = sycl::vec<uint32_t, 4>;

#define SG_SZ (16)                /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
#define LS_SZ (sizeof(message_t)) /* load/store byte size per work-item */

// 2 Cards
#define LL256_BUF_SIZE (32 * 1024 * 1024)
#define GATHER_BUF_OFFSET (LL256_BUF_SIZE / 2)
static void *host_bufs[DG2_NUM]; /* host shared buf */
static void *peer_bufs[DG2_NUM]; /* shared buf on peer side */

void dg2_init(sycl::queue &q, int q_idx, bool is_p2p)
{
    if (host_bufs[q_idx] == nullptr)
    {
        void *host_buf = nullptr;

        size_t buf_size = LL256_BUF_SIZE;
        constexpr size_t pagesize = 4096;
        if (is_p2p)
        {
            host_buf = sycl::aligned_alloc_device(pagesize, buf_size, q);
        }
        else
        {
            host_buf = sycl::aligned_alloc_host(pagesize, buf_size, q);
        }

        host_bufs[q_idx] = host_buf;
        peer_bufs[q_idx] = host_buf;
    }
}

void usm_memcpy(void *dst, const void *src, size_t count, sycl::queue &q_dst, sycl::queue &q_src)
{
    void *usm_buf = sycl::malloc_host(count * sizeof(int32_t), q_dst.get_context());
    q_src.memcpy(usm_buf, src, count * sizeof(int32_t)).wait();
    q_dst.memcpy(dst, usm_buf, count * sizeof(int32_t)).wait();
}

void print_host_buffer(void *org_host_buf, int idx, int N)
{
    int32_t *host_buf = (int32_t *)org_host_buf;
    const int nums_per_line = 16;
    printf("Print result on device %d:\n", idx);
    for (int i = 0; i < N / nums_per_line; i++)
    {
        printf("line %d: ", i);
        for (int j = 0; j < nums_per_line; j++)
        {
            printf("%x ", host_buf[i * nums_per_line + j]);
        }
        printf("\n");
    }
}

int main()
{
    auto Devs = sycl::platform(sycl::gpu_selector_v).get_devices(sycl::info::device_type::gpu);

    if (Devs.size() < 2)
    {
        std::cout << "Cannot test P2P capabilities, at least two devices are "
                     "required, exiting."
                  << std::endl;
        return 0;
    }

    std::vector<sycl::queue> Queues;
    std::transform(Devs.begin(), Devs.end(), std::back_inserter(Queues),
                   [](const sycl::device &D)
                   { return sycl::queue{D}; });

    const int N = 128;
    const bool isp2p = false;
    Devs[0].ext_oneapi_enable_peer_access(Devs[1]);

    std::vector<int> input(N);
    std::iota(input.begin(), input.end(), 0);

    int *dev0_ptr = sycl::malloc_device<int32_t>(N, Queues[0]);
    int *dev1_ptr = sycl::malloc_device<int32_t>(N, Queues[1]);

    Queues[0].memcpy(dev0_ptr, &input[0], N * sizeof(int32_t)); // dev1_ptr should be all zero

    dg2_init(Queues[0], 0, isp2p);
    dg2_init(Queues[1], 1, isp2p);

    printf("Before cpy:\n");
    Queues[0].memcpy(host_bufs[0], dev0_ptr, N * sizeof(int32_t)).wait();
    Queues[1].memcpy(host_bufs[1], dev1_ptr, N * sizeof(int32_t)).wait();

    for (int i = 0; i < 2; i++)
    {
        print_host_buffer(host_bufs[i], i, N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    Queues[0].wait();
    Queues[1].wait();
    
    usm_memcpy(dev1_ptr, dev0_ptr, N, Queues[0], Queues[1]);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Copy time: %.3f ms\n", elapsed_ms);

    printf("After cpy:\n");
    Queues[0].memcpy(host_bufs[0], dev0_ptr, N * sizeof(int32_t)).wait();
    Queues[1].memcpy(host_bufs[1], dev1_ptr, N * sizeof(int32_t)).wait();

    for (int i = 0; i < 2; i++)
    {
        print_host_buffer(host_bufs[i], i, N);
    }
}