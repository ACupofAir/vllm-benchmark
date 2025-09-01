/*
 * source /opt/intel/oneapi/2025.0/oneapi-vars.sh
 *
 * B60:
 * dpcpp dg2_allreduce.cpp -o dg2_allreduce.o -DXE_PLUS
 * A770:
 * dpcpp dg2_allreduce.cpp -o dg2_allreduce.o
 *
 * export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
 * export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
 * 
 * ./dg2_allreduce.o
 */
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

enum ggml_type
{
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_COUNT = 39,
};

uint16_t pattern_counter = 0xa770;
#define DG2_NUM (2)
typedef uint32_t pattern_t;
using message_t = sycl::vec<uint32_t, 4>;

#define SG_SZ (16)                /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
#define LS_SZ (sizeof(message_t)) /* load/store byte size per work-item */

#define __LscLoadUnCached(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.uc.uc   (M1, 16)  %0:d64  flat[%1]:a64" : "=rw"(var) : "rw"(addr) : "memory")
#define __LscLoadCached(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.ca.ca   (M1, 16)  %0:d64  flat[%1]:a64" : "=rw"(var) : "rw"(addr) : "memory")
#define __LscLoadUnCachedVec(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.uc.uc   (M1, 16)  %0:d32x4  flat[%1]:a64" : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "rw"(addr) : "memory")
#define __LscLoadCachedVec(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.ca.ca   (M1, 16)  %0:d32x4  flat[%1]:a64" : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "rw"(addr) : "memory")
#define __LscLoadL3CachedVec(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.uc.ca   (M1, 16)  %0:d32x4  flat[%1]:a64" : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "rw"(addr) : "memory")

#define __LscStoreUnCached(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.uc.uc  (M1, 16)  flat[%0]:a64  %1:d64" : : "rw"(addr), "rw"(var) : "memory")
#define __LscStoreCached(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.ca.ca  (M1, 16)  flat[%0]:a64  %1:d64" : : "rw"(addr), "rw"(var) : "memory")
#define __LscStoreUnCachedVec(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.uc.uc  (M1, 16)  flat[%0]:a64  %1:d32x4" : : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "memory")
#define __LscStoreCachedVec(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.ca.ca  (M1, 16)  flat[%0]:a64  %1:d32x4" : : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "memory")

#define LscLoadCached     __LscLoadCachedVec
#define LscLoadUnCached   __LscLoadUnCachedVec
#define LscLoadL3Cached   __LscLoadL3CachedVec
#define LscStoreCached    __LscStoreCachedVec
#define LscStoreUnCached  __LscStoreUnCachedVec


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

template <typename T>
static inline message_t _sum(message_t dst, message_t src)
{
    using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
    return sycl::bit_cast<message_t>(sycl::bit_cast<math_t>(dst) + sycl::bit_cast<math_t>(src));
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
static inline message_t sum(message_t dst, message_t src, const ggml_type &dtype)
{
    message_t data;

    switch (dtype)
    {
    case GGML_TYPE_F16:
        data = _sum<sycl::half>(dst, src);
        break;

    case GGML_TYPE_F32:
        data = _sum<float>(dst, src);
        break;

    case GGML_TYPE_I32:
        data = _sum<int>(dst, src);
        break;

    default:
        /* following code will hurt performance */
        // sycl::ext::oneapi::experimental::printf("Unknow dtype!\n");
        break;
    }

    return data;
}

static inline void sync_data(char *src, message_t &data, int lid, pattern_t pattern)
//[NEW-CODE]
{
    size_t sz = sizeof(message_t);
    auto   sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    
    int retry_count = 0;
    const int max_retries = 1;  // 增加重试次数
    
    do {
        LscLoadUnCached(data, src + lid * sz);

        if(lid==3) {
            sycl::ext::oneapi::experimental::printf("[RECV] pattern=0x%x, data[3]=0x%x from addr %p\n", 
                    pattern, data[3], src+lid*sz);  // rank用0占位
        } 
        
        bool pattern_match = !((lid == 3) && (data[3] != pattern)) &&
                            !((lid == 7) && (data[3] != pattern)) &&
                            !((lid == 11) && (data[3] != pattern)) &&
                            !((lid == 15) && (data[3] != pattern));
        
        if (!pattern_match) {
            retry_count++;
            if (retry_count > max_retries) {
                if (lid==3) {
                    sycl::ext::oneapi::experimental::printf("[TIMEOUT] pattern mismatch: pattern=0x%x, data[3]=0x%x from addr %p\n", 
                            pattern, data[3], src+lid*sz);
                }
                break;
            } 
        } else {
            if (lid==3) {
                sycl::ext::oneapi::experimental::printf("[SUCCESS] pattern matched: pattern=0x%x, data[3]=0x%x from addr %p\n", 
                        pattern, data[3], src+lid*sz);
            }
        }

    } while (sycl::any_of_group(sg, ((lid == 3) && (data[3] != pattern)) || 
                                    ((lid == 7) && (data[3] != pattern)) ||
                                    ((lid == 11) && (data[3] != pattern)) ||
                                    ((lid == 15) && (data[3] != pattern))));
}
//[ORG-CODE]
//{
//    size_t sz = sizeof(message_t);
//    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
//
//    do {
//#if defined(XE_PLUS)
//       LscLoadL3Cached(data, src + lid * sz);
//#else
//        LscLoadUnCached(data, src + lid * sz);
//#endif
//
//    } while (sycl::any_of_group(sg, ((lid ==  3) && (data[3] != pattern)) ||
//                                    ((lid ==  7) && (data[3] != pattern)) ||
//                                    ((lid == 11) && (data[3] != pattern)) ||
//                                    ((lid == 15) && (data[3] != pattern))));
//}
//
// Make some explain on simd instructions:
// "mov (M1, 1) %0(1, 15)<1> %0(3, 7)<0;1,0>\n"
// We need to maske it clear that this simd instruction is at subgroup level
// instread of work item level
// for a subgroup, it maintains a group of 16*4 uint32_t vec register, and they are viewed as 16*vec4-32bit registers
// (3, 7) means the 3rd 32bit element of the 7th vec4-32bit register
// (1, 15) means the 1st 32bit element of the 15th vec4-32bit register
//
// line 0: 0 1 2 3 4 5 6 7 8 9 a b c d e a7710000 
// line 1: 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e a7710000 
// line 2: 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e a7710000 
// line 3: 30 31 32 33 34 35 36 37 38 39 3a 3b f 1f 2f a7710000 
// 
// so this means mov '0f' to the position showed in line 3

// Also we can tell the difference of register between BMG and ARC
// for bmg it is a vec4-32bit register
// for arc it is a vec8-16bit register, right?
// "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
// move the 6th 16bit element of 7th vec8-16bit to the 3rf 16bit element of 7th vec8-16bit
// seems not right...
static inline void shuffle_data(message_t &data)
{
#if defined(XE_PLUS)
    __asm__ __volatile__("mov (M1, 1) %0(0, 15)<1> %0(3, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(1, 15)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(2, 15)<1> %0(3, 11)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : );
#else

    __asm__ __volatile__("mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : );
#endif
}

static inline void insert_pattern(message_t &data, pattern_t pattern)
{
#if defined(XE_PLUS)
    __asm__ __volatile__("mov (M1, 1) %0(3, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 15)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : "rw"(pattern));
#else
    __asm__ __volatile__("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : "rw"(pattern));
#endif
}

static inline void restore_data(message_t &data)
{
#if defined(XE_PLUS)
    __asm__ __volatile__("mov (M1, 1) %0(3, 3)<1> %0(0, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(1, 15)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 11)<1> %0(2, 15)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : );
#else
    __asm__ __volatile__("mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : );
#endif
}
#endif

static inline void send(char *next, char *src, int lid, int req_workitems, const ggml_type &dtype, int rank,
                        pattern_t pattern, size_t left_size)
{
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    if ((lid < req_workitems) && (lid * sz < left_size))
        LscLoadCached(data, src + lid * sz);

    shuffle_data(data);
    insert_pattern(data, pattern);

    // FIXME:ADD-BEG 添加：确认发送操作
    if (lid == 3) {
        void * write_addr = next + lid * sz;
        sycl::ext::oneapi::experimental::printf("[SEND] Rank %d lid %d sending pattern 0x%x to addr %p, data[3]=0x%x\n", 
                                               rank, lid, pattern, write_addr, data[3]);
    }
    // FIXME:ADD-END
    LscStoreUnCached(next + lid * sz, data);
#endif
}

static inline void recv_reduce_send(char *dst, char *next, char *src, int lid, int req_workitems,
                                    const ggml_type &dtype, int rank, pattern_t pattern, size_t left_size)
{
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);
    message_t *dst_buf = (message_t *)dst;

    sync_data(src, data, lid, pattern);
    restore_data(data);

    if ((lid < req_workitems) && (lid * sz < left_size))
        data = sum(dst_buf[lid], data, dtype);

    shuffle_data(data);
    insert_pattern(data, pattern);
    LscStoreUnCached(next + lid * sz, data);
#endif
}

static inline void recv_reduce_copy_send(char *dst, char *next, char *src, int lid, int req_workitems,
                                         const ggml_type &dtype, int rank, pattern_t pattern, size_t left_size)
{
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);
    message_t *dst_buf = (message_t *)dst;

    sync_data(src, data, lid, pattern);
    restore_data(data);

    if ((lid < req_workitems) && (lid * sz < left_size))
        data = sum(dst_buf[lid], data, dtype);

    if ((lid < req_workitems) && (lid * sz < left_size))
        LscStoreUnCached(dst + lid * sz, data);

    shuffle_data(data);
    insert_pattern(data, pattern);
    LscStoreUnCached(next + lid * sz, data);
#endif
}

static inline void recv_copy_send(char *dst, char *next, char *src, int lid, int req_workitems,
                                  const ggml_type& dtype, int rank, pattern_t pattern,size_t left_size)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    sync_data(src, data, lid, pattern);
    LscStoreUnCached(next + lid * sz, data);

    restore_data(data);

    if ((lid < req_workitems) && (lid * sz < left_size))
        LscStoreUnCached(dst + lid * sz, data);
    #endif
}

static inline void recv(char *dst, char *src, int lid, int req_workitems,
                        const ggml_type& dtype, int rank, pattern_t pattern,size_t left_size)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    /* copy reduced data from peer */
    sync_data(src, data, lid, pattern);

    restore_data(data);

    if ((lid < req_workitems) && (lid * sz < left_size))
        LscStoreUnCached(dst + lid * sz, data);
    #endif
}


void dg2_ll256_allreduce(const void *src, void *dst, size_t count, const int world_rank, const int world_size,
                         sycl::queue &q, ggml_type dtype)
{
    // std::cout << "enter " << __func__ << ", rank: " << world_rank <<  ", count: " << count << std::endl;

    size_t dt_sz;
    switch (dtype)
    {
    case GGML_TYPE_F16:
        dt_sz = sizeof(sycl::half);
        break;
    case GGML_TYPE_F32:
        dt_sz = sizeof(float);
        break;
    case GGML_TYPE_I32:
        dt_sz = sizeof(int32_t);
        break;
    default:
        assert(false and "Unsupported allreduce data type.");
    }
    char *recv_buf = static_cast<char *>(dst);
    char *send_buf = static_cast<char *>(const_cast<void *>(src));

    /*
     * Intel(R) Arc(TM) A770 Graphics:
     *   Number Of Slices:                       1
     *   Number Of Subslices Per Slice:          32
     *   Number Of EU Per Subslice:              16
     *   Number Of Threads Per EU:               8
     *   Total EU Count:                         512
     *   Physical EU SIMD Width:                 8
     */

    /* 64-byte load/store granularity to HBM, Maximum 128-byte payload can be used by EU store */
    /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
    size_t sg_sz = SG_SZ;

    size_t l_sz = 1 * sg_sz;
    size_t g_sz = 512 * l_sz;

    /* To avoid pattern not changed when "iters" is 1 */
    pattern_t pattern_prefix = (pattern_counter + 1) << 16;

    q.submit([&](auto& h) {
        using namespace sycl::ext::intel::experimental::esimd;

        int local_world_rank = world_rank;
        int local_world_size = world_size;

        int next_rank = (local_world_rank + 1) % local_world_size;
        char *local_host_buf = (char *)host_bufs[local_world_rank];

        char *local_peer_bufs[DG2_NUM];
        for (int i = 0; i < world_size; i++)
            local_peer_bufs[i] = (char *)peer_bufs[i];

        /*
         * In a single subgroup:
         *   a> 1 dedicated work-item to manage a LS_SZ-byte pattern.
         *   b> other work-items to process data, and each of them handle a LS_SZ-byte data.
         */
        auto default_subgroup_capacity = sg_sz * LS_SZ;  /* bytes: data and pattern  processed by 1 subgroup */
        auto default_workgroup_capacity = l_sz * LS_SZ;  /* bytes: data and patterns processed by 1 workgroup */
        //auto default_total_capacity = g_sz * LS_SZ;      /* bytes: data and patterns processed by all workgroups in 1 iteration */

        /* In a single workgroup, the available work-items to process data, excluding work-items for patterns */
        auto workgroup_available_items = l_sz - (l_sz / sg_sz);
        auto total_available_items = (g_sz / l_sz) * workgroup_available_items;

        auto subgroup_capacity = LS_SZ * (sg_sz - 1);                  /* bytes: data processed by 1 subgroup */
        auto workgroup_capacity = LS_SZ * workgroup_available_items;   /* bytes: data processed by 1 workgroup */
        auto total_capacity = (g_sz / l_sz) * workgroup_capacity;      /* bytes: data processed by all workgroups in 1 iteration */

        /* div up */
        int iters = (count * dt_sz + (local_world_size * total_available_items * LS_SZ - 1)) / (local_world_size * total_available_items * LS_SZ);

        //sycl::ext::oneapi::experimental::printf("------> rank: %d, group num: %ld, loop count: %zu\n", local_world_rank, g_sz / l_sz, iters);

        h.parallel_for(sycl::nd_range<1>(g_sz, l_sz), [=] (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
            int idx = 0;
            size_t offset = 0;
            size_t offset_with_pattern = 0;

            auto group_id = item.get_group_linear_id();
            auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto sg_id = sg.get_group_id()[0];
            auto sg_lid = sg.get_local_id()[0];

            for (int i = 0; i < iters; i++) {
                // FIXME:ADD-BEG
                if (group_id == 0 && sg_id == 0 && sg_lid == 0) {
                    sycl::ext::oneapi::experimental::printf("[AllReduce] Rank %d starting kernel, iters=%d\n",
                                                            local_world_rank, i);
                }
                // FIXME:ADD-END
                pattern_t pattern = pattern_prefix + i;

                auto base = local_world_size * (i        * total_capacity  +
                                                group_id * workgroup_capacity  +
                                                sg_id    * subgroup_capacity);
                auto base_with_pattern = local_world_size * (/* i        * default_total_capacity  + */
                                                             group_id * default_workgroup_capacity  +
                                                             sg_id    * default_subgroup_capacity);

                auto finished = i * total_capacity * local_world_size;   /* bytes */
                auto unreduced = count * dt_sz - finished;               /* bytes */

                auto req_workitems = sg_sz - 1;                /* required work-items exclude 1 work-item for pattern */
                auto chunk_sz = req_workitems * LS_SZ;         /* LS_SZ bytes per work-item */
                auto chunk_with_pattern = sg_sz * LS_SZ;       /* aligned to 256B */

                /* items will be assigned to each rank */
                auto per_rank_items = (unreduced + (local_world_size * LS_SZ - 1)) / (local_world_size * LS_SZ);
                auto req_workgroups = (per_rank_items + (workgroup_available_items - 1)) / workgroup_available_items;
                auto req_subgroups = 0;

                if (req_workgroups >= g_sz/l_sz) {
                    req_workgroups = g_sz/l_sz;
                } else {
                    if (group_id == (req_workgroups - 1)) {
                        req_subgroups = (per_rank_items + (sg_sz - 1)) / (sg_sz - 1);

                        /* (req_subgroups % (l_sz/sg_sz) - 1) equals to the final subgroup id in a workgroup */
                        /* Note:  req_subgroups % (l_sz/sg_sz) might be 0 */
                        if (((req_subgroups % (l_sz/sg_sz)) == 0) || (sg_id == (req_subgroups % (l_sz/sg_sz) - 1))) {
                            if ((per_rank_items % (sg_sz - 1)) != 0) {
                                /* FIXME: */
                                req_workitems = per_rank_items % (sg_sz - 1);
                                chunk_sz = req_workitems * LS_SZ;    /* LS_SZ bytes per work-item */
                            }
                        }
                    }
                }

                if (group_id < req_workgroups) {
                    // step 1: push data to next GPU
                    {
                        offset = base + local_world_rank * chunk_sz;
                        offset_with_pattern = base_with_pattern + local_world_rank * chunk_with_pattern;

                        char *next = local_peer_bufs[next_rank];

                        size_t left_size = count * dt_sz - offset;
                        // FIXME:START
                        if (group_id == 0 && sg_id == 0 && sg_lid == 0) {
                            sycl::ext::oneapi::experimental::printf("[AllReduce] Rank %d step 1\n", local_world_rank);
                        }
                        // FIXME:END
                        send(next + offset_with_pattern, send_buf + offset, sg_lid, req_workitems, dtype, local_world_rank, pattern,left_size);
                    }

                    // step 2: reduce and copy to next GPU
                    for (int j = 2; j < local_world_size; j++) {
                        idx = (local_world_rank + local_world_size + 1 - j) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;
                        char *next = local_peer_bufs[next_rank];

                        size_t left_size = count * dt_sz - offset;
                        // FIXME:START
                        if (group_id == 0 && sg_id == 0 && sg_lid == 0) {
                            sycl::ext::oneapi::experimental::printf("[AllReduce] Rank %d step 2\n", local_world_rank);
                        }
                        // FIXME:END
                        recv_reduce_send(recv_buf + offset, next + offset_with_pattern, src + offset_with_pattern,
                                         sg_lid, req_workitems, dtype, local_world_rank, pattern,left_size);
                    }

                    // step 3: reduce this buffer and data, which will produce the final
                    // result that we store in this data and push to the next GPU
                    {
                        idx = (local_world_rank + 1) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;
                        char *next = local_peer_bufs[next_rank];

                        size_t left_size = count * dt_sz - offset;
                        // FIXME:START
                        if (group_id == 0 && sg_id == 0 && sg_lid == 0) {
                            sycl::ext::oneapi::experimental::printf("[AllReduce] Rank %d step 3\n", local_world_rank);
                        }
                        // FIXME:END
                        recv_reduce_copy_send(recv_buf + offset, next + GATHER_BUF_OFFSET + offset_with_pattern, src + offset_with_pattern,
                                              sg_lid, req_workitems, dtype, local_world_rank, pattern,left_size);
                    }

                    // step 4: copy to next GPU
                    for (int j = 1; j < local_world_size - 1; ++j) {
                        idx = (local_world_rank + local_world_size + 1 - j) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = GATHER_BUF_OFFSET + base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;
                        char *next = local_peer_bufs[next_rank];

                        size_t left_size = count * dt_sz - offset;
                        // FIXME:START
                        if (group_id == 0 && sg_id == 0 && sg_lid == 0) {
                            sycl::ext::oneapi::experimental::printf("[AllReduce] Rank %d step 4\n", local_world_rank);
                        }
                        // FIXME:END
                        recv_copy_send(recv_buf + offset, next + offset_with_pattern, src + offset_with_pattern,
                                       sg_lid, req_workitems, dtype, local_world_rank, pattern,left_size);
                    }

                    // step 5: Make final copy from buffer to dest
                    {
                        idx = (local_world_rank + 2) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = GATHER_BUF_OFFSET + base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;

                        size_t left_size = count * dt_sz - offset;
                        // FIXME: START
                        if (group_id == 0 && sg_id == 0 && sg_lid == 0) {
                            sycl::ext::oneapi::experimental::printf("[AllReduce] Rank %d step 5\n", local_world_rank);
                        }
                        // FIXME: END
                        recv(recv_buf + offset, src + offset_with_pattern, sg_lid, req_workitems, dtype, local_world_rank, pattern,left_size);
                    }
                }
            }
        });
    });
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
    ////////////////////////////////////////////////////////////////////////

    // if (!Devs[0].ext_oneapi_can_access_peer(
    //         Devs[1], sycl::ext::oneapi::peer_access::access_supported))
    // {
    //     std::cout << "P2P access is not supported by devices, exiting."
    //               << std::endl;
    //     return 0;
    // }

    const int N = 128;
    const bool isp2p = false;
    Devs[0].ext_oneapi_enable_peer_access(Devs[1]);

    std::vector<int> input(N);
    std::iota(input.begin(), input.end(), 0);

    int *dev0_ptr = sycl::malloc_device<int32_t>(N, Queues[0]);
    int *dev1_ptr = sycl::malloc_device<int32_t>(N, Queues[1]);

    Queues[0].memcpy(dev0_ptr, &input[0], N * sizeof(int32_t));


    dg2_init(Queues[0], 0, isp2p);
    dg2_init(Queues[1], 1, isp2p);

    printf("Before allreduce:\n");
    Queues[0].memcpy(host_bufs[0], dev0_ptr, N * sizeof(int32_t)).wait();
    Queues[1].memcpy(host_bufs[1], dev1_ptr, N * sizeof(int32_t)).wait();
    for (int i = 0; i < 2; i++)
    {
        print_host_buffer(host_bufs[i], i, N);
    }

    dg2_ll256_allreduce(dev0_ptr, dev0_ptr, N, 0, 2, Queues[0], GGML_TYPE_I32);
    dg2_ll256_allreduce(dev1_ptr, dev1_ptr, N, 1, 2, Queues[1], GGML_TYPE_I32);

    Queues[0].wait();
    Queues[1].wait();

    printf("After wait:\n");
    for (int i = 0; i < 2; i++)
    {
        print_host_buffer(host_bufs[i], i, N);
    }

    Queues[0].memcpy(host_bufs[0], dev0_ptr, N * sizeof(int32_t)).wait();
    Queues[1].memcpy(host_bufs[1], dev1_ptr, N * sizeof(int32_t)).wait();

    Queues[0].wait();
    Queues[1].wait();

    printf("After allreduce:\n");
    for (int i = 0; i < 2; i++)
    {
        print_host_buffer(host_bufs[i], i, N);
    }
}
