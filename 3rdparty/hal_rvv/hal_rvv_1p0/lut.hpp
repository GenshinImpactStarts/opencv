// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/utility.hpp>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_lut
#define cv_hal_lut cv::cv_hal_rvv::lut

// inline static void lutGather(const uchar* src_data,
//                              vuint8m4_t lut_v0,
//                              vuint8m4_t lut_v1,
//                              vuint8m4_t lut_v2,
//                              vuint8m4_t lut_v3,
//                              uchar* dst_data)
// {
//     asm("vle8.v      v28, (%0)\n"
//         "vmsltu.vx   v0, v28, %6\n"
//         "vrgather.vv v24, %1, v28, v0.t\n"
//         ""
//         "vsub.vx     v28, v28, %6\n"
//         "vmsltu.vx   v0, v28, %6\n"
//         "vrgather.vv v20, %2, v28, v0.t\n"
//         "vmerge.vvm  v24, v24, v20, v0\n"
//         ""
//         "vsub.vx     v28, v28, %6\n"
//         "vmsltu.vx   v0, v28, %6\n"
//         "vrgather.vv v20, %3, v28, v0.t\n"
//         "vmerge.vvm  v24, v24, v20, v0\n"
//         ""
//         "vsub.vx     v28, v28, %6\n"
//         "vmsltu.vx   v0, v28, %6\n"
//         "vrgather.vv v20, %4, v28, v0.t\n"
//         "vmerge.vvm  v24, v24, v20, v0\n"
//         "vse8.v      v24, (%5)"
//         :
//         : "r"(src_data), "vr"(lut_v0), "vr"(lut_v1), "vr"(lut_v2), "vr"(lut_v3), "r"(dst_data),
//         "r"(64) : "memory", "v0", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
//         "v29", "v30", "v31"
//     );
//     // vuint8m4_t src_v, dst_v;
//     // vbool2_t mask;
//     // src_v = __riscv_vle8_v_u8m4(src_data, vl);
//     // mask = __riscv_vmsltu(src_v, 64, vl);
//     // dst_v = __riscv_vrgather(mask, lut_v0, src_v, vl);

//     // src_v = __riscv_vsub(src_v, 64, vl);
//     // mask = __riscv_vmsltu(src_v, 64, vl);
//     // dst_v = __riscv_vmerge(dst_v, __riscv_vrgather(mask, lut_v1, src_v, vl), mask, vl);

//     // src_v = __riscv_vsub(src_v, 64, vl);
//     // mask = __riscv_vmsltu(src_v, 64, vl);
//     // dst_v = __riscv_vmerge(dst_v, __riscv_vrgather(mask, lut_v2, src_v, vl), mask, vl);

//     // src_v = __riscv_vsub(src_v, 64, vl);
//     // mask = __riscv_vmsltu(src_v, 64, vl);
//     // dst_v = __riscv_vmerge(dst_v, __riscv_vrgather(mask, lut_v3, src_v, vl), mask, vl);

//     // __riscv_vse8(dst_data, dst_v, vl);
// }

// for vl 128
// static void lut_u8m4x4(const uchar* src_data,
//                        vuint8m4_t lut_v0,
//                        vuint8m4_t lut_v1,
//                        vuint8m4_t lut_v2,
//                        vuint8m4_t lut_v3,
//                        uchar* dst_data,
//                        size_t len)
// {
//     size_t vl = __riscv_vsetvlmax_e8m4();
//     for (; len >= vl; len -= vl, src_data += vl, dst_data += vl)
//     {
//         lutGather(src_data, lut_v0, lut_v1, lut_v2, lut_v3, dst_data);
//     }
//     for (; len > 0; len -= vl, src_data += vl, dst_data += vl)
//     {
//         vl = __riscv_vsetvl_e8m4(len);
//         lutGather(src_data, lut_v0, lut_v1, lut_v2, lut_v3, dst_data);
//     }
// }

// inline vuint8m4x2_t lut_gather_u8c2(vuint8m4x2_t lut_v, vuint8m4x2_t src_v, size_t vl)
// {
//     auto src0 = __riscv_vget_u8m4(src_v, 0);
//     auto src1 = __riscv_vget_u8m4(src_v, 1);
//     auto lut0 = __riscv_vget_u8m4(lut_v, 0);
//     auto lut1 = __riscv_vget_u8m4(lut_v, 1);
//     auto dst0 = __riscv_vrgather(lut0, src0, vl);
//     auto dst1 = __riscv_vrgather(lut1, src1, vl);
//     return __riscv_vcreate_v_u8m4x2(dst0, dst1);
// }

// for large vlen
// inline void lut_u8c2(const uchar* src_data, vuint8m4x2_t lut_v, uchar* dst_data, size_t len)
// {
//     size_t vl = __riscv_vsetvlmax_e8m4();
//     size_t step = vl * 2;

//     for (; len >= vl; len -= vl, src_data += step, dst_data += step)
//     {
//         auto src_v = __riscv_vlseg2e8_v_u8m4x2(src_data, vl);
//         auto dst_v = lut_gather_u8c2(lut_v, src_v, vl);
//         __riscv_vsseg2e8(dst_data, dst_v, vl);
//     }
//     for (; len > 0; len -= vl, src_data += step, dst_data += step)
//     {
//         vl = __riscv_vsetvl_e8m4(len);
//         auto src_v = __riscv_vlseg2e8_v_u8m4x2(src_data, vl);
//         auto dst_v = lut_gather_u8c2(lut_v, src_v, vl);
//         __riscv_vsseg2e8(dst_data, dst_v, vl);
//     }
// }

// cache each cn
// inline void
//     lut_u8cx(const uchar* src_data, vuint8m8_t lut_v, uchar* dst_data, size_t len, size_t cn)
// {
//     size_t vl = __riscv_vsetvlmax_e8m8();
//     size_t step = vl * cn;

//     for (; len >= vl; len -= vl, src_data += step, dst_data += step)
//     {
//         auto src_v = __riscv_vlse8_v_u8m8(src_data, static_cast<ptrdiff_t>(cn), vl);
//         auto dst_v = __riscv_vrgather(lut_v, src_v, vl);
//         __riscv_vsse8(dst_data, static_cast<ptrdiff_t>(cn), dst_v, vl);
//     }
//     for (; len > 0; len -= vl, src_data += step, dst_data += step)
//     {
//         vl = __riscv_vsetvl_e8m8(len);
//         auto src_v = __riscv_vlse8_v_u8m8(src_data, static_cast<ptrdiff_t>(cn), vl);
//         auto dst_v = __riscv_vrgather(lut_v, src_v, vl);
//         __riscv_vsse8(dst_data, static_cast<ptrdiff_t>(cn), dst_v, vl);
//     }
// }

// use vluxei8 load directly
// template <typename T>
// struct LUTLoad
// {
//     static inline const T* loadTable(const uchar* lut_data)
//     {
//         return reinterpret_cast<const T*>(lut_data);
//     }

//     static inline void lut(const uchar* src, const T* lut, T* dst, size_t len)
//     {
//         size_t vl = __riscv_vsetvlmax_e8m8() / sizeof(T);
//         for (; len >= vl; len -= vl, src += vl, dst += vl)
//         {
//             auto src_v = __riscv_vle8_v_u8m8(src, vl);
//             auto dst_v = __riscv_vluxei8_v_u8m8(lut, src_v, vl);
//             __riscv_vse8(dst, dst_v, vl);
//         }
//         for (; len > 0; len -= vl, src += vl, dst += vl)
//         {
//             vl = __riscv_vsetvl_e8m8(len);
//             auto src_v = __riscv_vle8_v_u8m8(src, vl);
//             auto dst_v = __riscv_vluxei8_v_u8m8(lut, src_v, vl);
//             __riscv_vse8(dst, dst_v, vl);
//         }
//     }
// };

// for vl 128
// class LUTParallelBodyU8M4x4 : public cv::ParallelLoopBody
// {
// public:
//     const uchar* src;
//     const uchar* lut;
//     uchar* dst;
//     size_t src_step;
//     size_t dst_step;
//     size_t width;

//     LUTParallelBodyU8M4x4(const uchar* src_data,
//                           size_t src_step,
//                           const uchar* lut_data,
//                           uchar* dst_data,
//                           size_t dst_step,
//                           size_t width) :
//         src(src_data), lut(lut_data), dst(dst_data), src_step(src_step), dst_step(dst_step),
//         width(width)
//     {
//     }

//     void operator()(const cv::Range& range) const CV_OVERRIDE
//     {
//         auto lut_v0 = __riscv_vle8_v_u8m4(lut, 64);
//         auto lut_v1 = __riscv_vle8_v_u8m4(lut + 64, 64);
//         auto lut_v2 = __riscv_vle8_v_u8m4(lut + 128, 64);
//         auto lut_v3 = __riscv_vle8_v_u8m4(lut + 192, 64);
//         auto src_data = src + range.start * src_step;
//         auto dst_data = dst + range.start * dst_step;
//         size_t h = range.size();
//         size_t w = width;
//         if (w * 1 == src_step && w * 1 == dst_step)
//         {
//             w = w * h;
//             h = 1;
//         }
//         for (; h; h--, src_data += src_step, dst_data += dst_step)
//             lut_u8m4x4(src_data, lut_v0, lut_v1, lut_v2, lut_v3, dst_data, w);
//     }

// private:
//     LUTParallelBodyU8M4x4(const LUTParallelBodyU8M4x4&);
//     LUTParallelBodyU8M4x4& operator=(const LUTParallelBodyU8M4x4&);
// };

// need vlen >= 256
struct LUTCacheU8
{
    using ElemType = uchar;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline vuint8m8_t loadTable(const ElemType* lut_data)
    {
        return __riscv_vle8_v_u8m8(lut_data, 256);
    }

    static inline size_t setvl(size_t len)
    {
        return __riscv_vsetvl_e8m8(len);
    }

    static inline size_t setvlmax()
    {
        return __riscv_vsetvlmax_e8m8();
    }

    static inline void lut(const uchar* src, vuint8m8_t lut_v, ElemType* dst, size_t vl)
    {
        auto src_v = __riscv_vle8_v_u8m8(src, vl);
        auto dst_v = __riscv_vrgather(lut_v, src_v, vl);
        __riscv_vse8(dst, dst_v, vl);
    }
};

// need vlen >= 512
struct LUTCacheU16
{
    using ElemType = uint16_t;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline vuint16m8_t loadTable(const ElemType* lut_data)
    {
        return __riscv_vle16_v_u16m8(lut_data, 256);
    }

    static inline size_t setvl(size_t len)
    {
        return __riscv_vsetvl_e16m8(len);
    }

    static inline size_t setvlmax()
    {
        return __riscv_vsetvlmax_e16m8();
    }

    static inline void lut(const uchar* src, vuint16m8_t lut_v, ElemType* dst, size_t vl)
    {
        auto src_v = __riscv_vzext_vf2(__riscv_vle8_v_u8m4(src, vl), vl);
        auto dst_v = __riscv_vrgather(lut_v, src_v, vl);
        __riscv_vse16(dst, dst_v, vl);
    }
};

// need vlen >= 1024
struct LUTCacheU32
{
    using ElemType = uint32_t;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline vuint32m8_t loadTable(const ElemType* lut_data)
    {
        return __riscv_vle32_v_u32m8(lut_data, 256);
    }

    static inline size_t setvl(size_t len)
    {
        return __riscv_vsetvl_e32m8(len);
    }

    static inline size_t setvlmax()
    {
        return __riscv_vsetvlmax_e32m8();
    }

    static inline void lut(const uchar* src, vuint32m8_t lut_v, ElemType* dst, size_t vl)
    {
        auto src_v = __riscv_vzext_vf2(__riscv_vle8_v_u8m2(src, vl), vl);
        auto dst_v = __riscv_vrgatherei16(lut_v, src_v, vl);
        __riscv_vse32(dst, dst_v, vl);
    }
};

template <typename LUT_TYPE>
class LUTParallelBody : public cv::ParallelLoopBody
{
    using ElemType = typename LUT_TYPE::ElemType;

public:
    const uchar* src_data;
    const uchar* lut_data;
    uchar* dst_data;
    size_t src_step;
    size_t dst_step;
    size_t width;

    LUTParallelBody(const uchar* src_data,
                    size_t src_step,
                    const uchar* lut_data,
                    uchar* dst_data,
                    size_t dst_step,
                    size_t width) :
        src_data(src_data), lut_data(lut_data), dst_data(dst_data), src_step(src_step),
        dst_step(dst_step), width(width)
    {
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        auto src = src_data + range.start * src_step;
        auto dst = dst_data + range.start * dst_step;
        size_t h = range.size();
        size_t w = width;
        if (w == src_step && w * LUT_TYPE::elem_size == dst_step)
        {
            w = w * h;
            h = 1;
        }
        auto lut = LUT_TYPE::loadTable((ElemType*)lut_data);
        size_t maxlv = LUT_TYPE::setvlmax();
        for (; h; h--, src += src_step, dst += dst_step)
        {
            size_t vl = maxlv;
            size_t l = w;
            auto s = src;
            auto d = (ElemType*)dst;
            for (; l >= vl; l -= vl, s += vl, d += vl)
            {
                LUT_TYPE::lut(s, lut, d, vl);
            }
            for (; l > 0; l -= vl, s += vl, d += vl)
            {
                vl = LUT_TYPE::setvl(l);
                LUT_TYPE::lut(s, lut, d, vl);
            }
        }
    }

private:
    LUTParallelBody(const LUTParallelBody&);
    LUTParallelBody& operator=(const LUTParallelBody&);
};

inline int lut(const uchar* src_data,
               size_t src_step,
               [[maybe_unused]] size_t src_type,
               const uchar* lut_data,
               size_t lut_channel_size,
               size_t lut_channels,
               uchar* dst_data,
               size_t dst_step,
               int width,
               int height)
{
    if (width <= 0 || height <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    size_t w = width;
    size_t h = height;
    size_t vlen = __riscv_vsetvlmax_e8m8();
    if (lut_channels == 1)
    {
        w *= CV_MAT_CN(src_type);

        if (lut_channel_size == 1 && vlen >= 256)
        {
            LUTParallelBody<LUTCacheU8> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            Range all(0, height);
            if (w * h >= (1 << 18))
                cv::parallel_for_(all, body);
            else
                body(all);
            return CV_HAL_ERROR_OK;
        }
        else if (lut_channel_size == 2 && vlen >= 512)
        {
            LUTParallelBody<LUTCacheU16> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            Range all(0, height);
            if (w * h >= (1 << 18))
                cv::parallel_for_(all, body);
            else
                body(all);
            return CV_HAL_ERROR_OK;
        }
        else if (lut_channel_size == 4 && vlen >= 1024)
        {
            LUTParallelBody<LUTCacheU32> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            Range all(0, height);
            if (w * h >= (1 << 18))
                cv::parallel_for_(all, body);
            else
                body(all);
            return CV_HAL_ERROR_OK;
        }
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
    // for vl 128
    // {
    //     LUTParallelBodyU8M4x4 body(src_data, src_step, lut_data, dst_data, dst_step, w);
    //     Range all(0, height);
    //     if (w * h >= (1 << 18))
    //         cv::parallel_for_(all, body);
    //     else
    //         body(all);
    //     return CV_HAL_ERROR_OK;
    // }

    // cache each cn
    // if (lut_channel_size == 1 && __riscv_vsetvlmax_e8m8() >= 256)
    // {
    //     for (size_t cn = 0; cn < lut_channels; cn++)
    //     {
    //         auto lut = __riscv_vlse8_v_u8m8(lut_data + cn, cn, 256);
    //         auto s = src_data + cn;
    //         auto d = dst_data + cn;
    //         for (int i = 0; i < height; i++, s += src_step, d += dst_step)
    //             lut_u8cx(s, lut, d, width, lut_channels);
    //     }
    //     return CV_HAL_ERROR_OK;
    // }
}


}}  // namespace cv::cv_hal_rvv
