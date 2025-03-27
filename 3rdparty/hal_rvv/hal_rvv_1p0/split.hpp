// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED
#define OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED

#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_split8u
#define cv_hal_split8u cv::cv_hal_rvv::split
#undef cv_hal_split16u
#define cv_hal_split16u cv::cv_hal_rvv::split
#undef cv_hal_split32s
#define cv_hal_split32s cv::cv_hal_rvv::split
#undef cv_hal_split64s
#define cv_hal_split64s cv::cv_hal_rvv::split

template <typename T>
inline int split(const T* src, T** dst, int _len, int cn)
{
    size_t vl;
    if (__builtin_expect(cn == 1, 0))
    {
        T* dst0 = dst[0];
        for (size_t len = _len; len > 0; len -= vl)
        {
            vl = RVV<T, LMUL_8>::setvl(len);
            auto v0 = RVV<T, LMUL_8>::vload(src, vl);
            RVV<T, LMUL_8>::vstore(dst0, v0, vl);
            src += vl;
            dst0 += vl;
        }
    }
    else if (__builtin_expect(cn == 2, 1))
    {
        T *dst0 = dst[0], *dst1 = dst[1];
        for (size_t len = _len; len > 0; len -= vl)
        {
            vl = RVV<T, LMUL_4>::setvl(len);
            auto v = RVV<T, LMUL_4>::template vload_seg<2>(src, vl);
            auto v0 = RVV<T, LMUL_4>::template vget<0>(v);
            auto v1 = RVV<T, LMUL_4>::template vget<1>(v);
            RVV<T, LMUL_4>::vstore(dst0, v0, vl);
            RVV<T, LMUL_4>::vstore(dst1, v1, vl);
            src += vl * cn;
            dst0 += vl, dst1 += vl;
        }
    }
    else if (__builtin_expect(cn == 3, 1))
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        for (size_t len = _len; len > 0; len -= vl)
        {
            vl = RVV<T, LMUL_2>::setvl(len);
            auto v = RVV<T, LMUL_2>::template vload_seg<3>(src, vl);
            auto v0 = RVV<T, LMUL_2>::template vget<0>(v);
            auto v1 = RVV<T, LMUL_2>::template vget<1>(v);
            auto v2 = RVV<T, LMUL_2>::template vget<2>(v);
            RVV<T, LMUL_2>::vstore(dst0, v0, vl);
            RVV<T, LMUL_2>::vstore(dst1, v1, vl);
            RVV<T, LMUL_2>::vstore(dst2, v2, vl);
            src += vl * cn;
            dst0 += vl, dst1 += vl, dst2 += vl;
        }
    }
    else if (__builtin_expect(cn == 4, 1))
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        for (size_t len = _len; len > 0; len -= vl)
        {
            vl = RVV<T, LMUL_2>::setvl(len);
            auto v = RVV<T, LMUL_2>::template vload_seg<4>(src, vl);
            auto v0 = RVV<T, LMUL_2>::template vget<0>(v);
            auto v1 = RVV<T, LMUL_2>::template vget<1>(v);
            auto v2 = RVV<T, LMUL_2>::template vget<2>(v);
            auto v3 = RVV<T, LMUL_2>::template vget<3>(v);
            RVV<T, LMUL_2>::vstore(dst0, v0, vl);
            RVV<T, LMUL_2>::vstore(dst1, v1, vl);
            RVV<T, LMUL_2>::vstore(dst2, v2, vl);
            RVV<T, LMUL_2>::vstore(dst3, v3, vl);
            src += vl * cn;
            dst0 += vl, dst1 += vl, dst2 += vl, dst3 += vl;
        }
    }
    // else if (cn == 5)
    // {
    //     T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4];
    //     for (size_t len = _len; len > 0; len -= vl)
    //     {
    //         vl = RVV<T, LMUL_1>::setvl(len);
    //         auto v = RVV<T, LMUL_1>::template vload_seg<5>(src, vl);
    //         auto v0 = RVV<T, LMUL_1>::template vget<0>(v);
    //         auto v1 = RVV<T, LMUL_1>::template vget<1>(v);
    //         auto v2 = RVV<T, LMUL_1>::template vget<2>(v);
    //         auto v3 = RVV<T, LMUL_1>::template vget<3>(v);
    //         auto v4 = RVV<T, LMUL_1>::template vget<4>(v);
    //         RVV<T, LMUL_1>::vstore(dst0, v0, vl);
    //         RVV<T, LMUL_1>::vstore(dst1, v1, vl);
    //         RVV<T, LMUL_1>::vstore(dst2, v2, vl);
    //         RVV<T, LMUL_1>::vstore(dst3, v3, vl);
    //         RVV<T, LMUL_1>::vstore(dst4, v4, vl);
    //         src += vl * cn;
    //         dst0 += vl, dst1 += vl, dst2 += vl, dst3 += vl, dst4 += vl;
    //     }
    // }
    // else if (cn == 6)
    // {
    //     T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4], *dst5 = dst[5];
    //     for (size_t len = _len; len > 0; len -= vl)
    //     {
    //         vl = RVV<T, LMUL_1>::setvl(len);
    //         auto v = RVV<T, LMUL_1>::template vload_seg<6>(src, vl);
    //         auto v0 = RVV<T, LMUL_1>::template vget<0>(v);
    //         auto v1 = RVV<T, LMUL_1>::template vget<1>(v);
    //         auto v2 = RVV<T, LMUL_1>::template vget<2>(v);
    //         auto v3 = RVV<T, LMUL_1>::template vget<3>(v);
    //         auto v4 = RVV<T, LMUL_1>::template vget<4>(v);
    //         auto v5 = RVV<T, LMUL_1>::template vget<5>(v);
    //         RVV<T, LMUL_1>::vstore(dst0, v0, vl);
    //         RVV<T, LMUL_1>::vstore(dst1, v1, vl);
    //         RVV<T, LMUL_1>::vstore(dst2, v2, vl);
    //         RVV<T, LMUL_1>::vstore(dst3, v3, vl);
    //         RVV<T, LMUL_1>::vstore(dst4, v4, vl);
    //         RVV<T, LMUL_1>::vstore(dst5, v5, vl);
    //         src += vl * cn;
    //         dst0 += vl, dst1 += vl, dst2 += vl, dst3 += vl, dst4 += vl, dst5 += vl;
    //     }
    // }
    // else if (cn == 7)
    // {
    //     T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4], *dst5 = dst[5], *dst6 = dst[6];
    //     for (size_t len = _len; len > 0; len -= vl)
    //     {
    //         vl = RVV<T, LMUL_1>::setvl(len);
    //         auto v = RVV<T, LMUL_1>::template vload_seg<7>(src, vl);
    //         auto v0 = RVV<T, LMUL_1>::template vget<0>(v);
    //         auto v1 = RVV<T, LMUL_1>::template vget<1>(v);
    //         auto v2 = RVV<T, LMUL_1>::template vget<2>(v);
    //         auto v3 = RVV<T, LMUL_1>::template vget<3>(v);
    //         auto v4 = RVV<T, LMUL_1>::template vget<4>(v);
    //         auto v5 = RVV<T, LMUL_1>::template vget<5>(v);
    //         auto v6 = RVV<T, LMUL_1>::template vget<6>(v);
    //         RVV<T, LMUL_1>::vstore(dst0, v0, vl);
    //         RVV<T, LMUL_1>::vstore(dst1, v1, vl);
    //         RVV<T, LMUL_1>::vstore(dst2, v2, vl);
    //         RVV<T, LMUL_1>::vstore(dst3, v3, vl);
    //         RVV<T, LMUL_1>::vstore(dst4, v4, vl);
    //         RVV<T, LMUL_1>::vstore(dst5, v5, vl);
    //         RVV<T, LMUL_1>::vstore(dst6, v6, vl);
    //         src += vl * cn;
    //         dst0 += vl, dst1 += vl, dst2 += vl, dst3 += vl, dst4 += vl, dst5 += vl, dst6 += vl;
    //     }
    // }
    // else if (cn == 8)
    // {
    //     T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4], *dst5 = dst[5], *dst6 = dst[6], *dst7 = dst[7];
    //     for (size_t len = _len; len > 0; len -= vl)
    //     {
    //         vl = RVV<T, LMUL_1>::setvl(len);
    //         auto v = RVV<T, LMUL_1>::template vload_seg<8>(src, vl);
    //         auto v0 = RVV<T, LMUL_1>::template vget<0>(v);
    //         auto v1 = RVV<T, LMUL_1>::template vget<1>(v);
    //         auto v2 = RVV<T, LMUL_1>::template vget<2>(v);
    //         auto v3 = RVV<T, LMUL_1>::template vget<3>(v);
    //         auto v4 = RVV<T, LMUL_1>::template vget<4>(v);
    //         auto v5 = RVV<T, LMUL_1>::template vget<5>(v);
    //         auto v6 = RVV<T, LMUL_1>::template vget<6>(v);
    //         auto v7 = RVV<T, LMUL_1>::template vget<7>(v);
    //         RVV<T, LMUL_1>::vstore(dst0, v0, vl);
    //         RVV<T, LMUL_1>::vstore(dst1, v1, vl);
    //         RVV<T, LMUL_1>::vstore(dst2, v2, vl);
    //         RVV<T, LMUL_1>::vstore(dst3, v3, vl);
    //         RVV<T, LMUL_1>::vstore(dst4, v4, vl);
    //         RVV<T, LMUL_1>::vstore(dst5, v5, vl);
    //         RVV<T, LMUL_1>::vstore(dst6, v6, vl);
    //         RVV<T, LMUL_1>::vstore(dst7, v7, vl);
    //         src += vl * cn;
    //         dst0 += vl, dst1 += vl, dst2 += vl, dst3 += vl, dst4 += vl, dst5 += vl, dst6 += vl, dst7 += vl;
    //     }
    // }
    else
    {
        // for (size_t k = 0; k < cn; ++k)
        // {
        //     const T* src0 = src + k;
        //     T* dst0 = dst[k];
        //     for (size_t len = _len; len > 0; len -= vl)
        //     {
        //         vl = RVV<T, LMUL_8>::setvl(len);
        //         auto v0 = RVV<T, LMUL_8>::vload_stride(src0, cn * sizeof(T), vl);
        //         RVV<T, LMUL_8>::vstore(dst0, v0, vl);
        //         src0 += vl * cn;
        //         dst0 += vl;
        //     }
        // }
        auto k = cn;
        for (; k >= 4; k -= 4)
        {
            const T* src0 = src;
            // T *dst0 = dst[0], *dst1 = dst[1];
            T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
            for (size_t len = _len; len > 0; len -= vl)
            {
                vl = RVV<T, LMUL_2>::setvl(len);
                auto v = RVV<T, LMUL_2>::template vload_sseg<4>(src0, cn * sizeof(T), vl);
                auto v0 = RVV<T, LMUL_2>::template vget<0>(v);
                auto v1 = RVV<T, LMUL_2>::template vget<1>(v);
                auto v2 = RVV<T, LMUL_2>::template vget<2>(v);
                auto v3 = RVV<T, LMUL_2>::template vget<3>(v);
                RVV<T, LMUL_2>::vstore(dst0, v0, vl);
                RVV<T, LMUL_2>::vstore(dst1, v1, vl);
                RVV<T, LMUL_2>::vstore(dst2, v2, vl);
                RVV<T, LMUL_2>::vstore(dst3, v3, vl);
                src0 += vl * cn;
                // dst0 += vl, dst1 += vl;
                dst0 += vl, dst1 += vl, dst2 += vl, dst3 += vl;
            }
            src += 2, dst += 2;
        }
        if (k == 1)
        {
            T* dst0 = dst[0];
            for (size_t len = _len; len > 0; len -= vl)
            {
                vl = RVV<T, LMUL_8>::setvl(len);
                auto v0 = RVV<T, LMUL_8>::vload_stride(src, cn * sizeof(T), vl);
                RVV<T, LMUL_8>::vstore(dst0, v0, vl);
                src += vl * cn;
                dst0 += vl;
            }
        }
        else if (k == 2)
        {
            T *dst0 = dst[0], *dst1 = dst[1];
            for (size_t len = _len; len > 0; len -= vl)
            {
                vl = RVV<T, LMUL_4>::setvl(len);
                auto v = RVV<T, LMUL_4>::template vload_sseg<2>(src, cn * sizeof(T), vl);
                auto v0 = RVV<T, LMUL_4>::template vget<0>(v);
                auto v1 = RVV<T, LMUL_4>::template vget<1>(v);
                RVV<T, LMUL_4>::vstore(dst0, v0, vl);
                RVV<T, LMUL_4>::vstore(dst1, v1, vl);
                src += vl * cn;
                dst0 += vl, dst1 += vl;
            }
        }
        else if (k == 3)
        {
            T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
            for (size_t len = _len; len > 0; len -= vl)
            {
                vl = RVV<T, LMUL_2>::setvl(len);
                auto v = RVV<T, LMUL_2>::template vload_sseg<3>(src, cn * sizeof(T), vl);
                auto v0 = RVV<T, LMUL_2>::template vget<0>(v);
                auto v1 = RVV<T, LMUL_2>::template vget<1>(v);
                auto v2 = RVV<T, LMUL_2>::template vget<2>(v);
                RVV<T, LMUL_2>::vstore(dst0, v0, vl);
                RVV<T, LMUL_2>::vstore(dst1, v1, vl);
                RVV<T, LMUL_2>::vstore(dst2, v2, vl);
                src += vl * cn;
                dst0 += vl, dst1 += vl, dst2 += vl;
            }
        }
    }
    return CV_HAL_ERROR_OK;
}


// void f()
// {
//     split((uint8_t*)0, (uint8_t**)0, 0, 0);
// }

// template <size_t cn, typename T>
// inline void split_(const T* src, T** dst, size_t len)
// {
//     using RVV_T = RVV<T, LMUL_8>;
//     size_t vl;
//     for (size_t cur = 0; cur < len; cur += vl, src += vl * cn)
//     {
//         vl = RVV_T::setvl(len - cur);
// #pragma unroll
//         for (size_t i = 0; i < cn; i++)
//         {
//             auto v = RVV_T::vload_stride(src + i, cn * sizeof(T), vl);
//             RVV_T::vstore(dst[i] + cur, v, vl);
//         }
//     }
// }

// template <typename T>
// inline int split(const T* src, T** dst, int _len, int cn)
// {
//     switch (cn) {
//     case 1:
//         split_<1>(src, dst, _len);
//         break;
//     case 2:
//         split_<2>(src, dst, _len);
//         break;
//     case 3:
//         split_<3>(src, dst, _len);
//         break;
//     case 4:
//         split_<4>(src, dst, _len);
//         break;
//     case 5:
//         split_<5>(src, dst, _len);
//         break;
//     case 6:
//         split_<6>(src, dst, _len);
//         break;
//     case 7:
//         split_<7>(src, dst, _len);
//         break;
//     case 8:
//         split_<8>(src, dst, _len);
//         break;
//     default:
//         return CV_HAL_ERROR_NOT_IMPLEMENTED;
//     }
//     return CV_HAL_ERROR_OK;
// }

// inline int split8u(const uchar* src, uchar** dst, int len, int cn)
// {
//     int vl = 0;
//     if (cn == 1)
//     {
//         uchar* dst0 = dst[0];
//         for (int i = 0; i < len; i += vl)
//         {
//             vl = __riscv_vsetvl_e8m8(len - i);
//             __riscv_vse8_v_u8m8(dst0 + i, __riscv_vle8_v_u8m8(src + i, vl), vl);
//         }
//     }
//     else if (cn == 2)
//     {
//         uchar *dst0 = dst[0], *dst1 = dst[1];
//         for (int i = 0; i < len; i += vl)
//         {
//             vl = __riscv_vsetvl_e8m4(len - i);
//             vuint8m4x2_t seg = __riscv_vlseg2e8_v_u8m4x2(src + i * cn, vl);
//             __riscv_vse8_v_u8m4(dst0 + i, __riscv_vget_v_u8m4x2_u8m4(seg, 0), vl);
//             __riscv_vse8_v_u8m4(dst1 + i, __riscv_vget_v_u8m4x2_u8m4(seg, 1), vl);
//         }
//     }
//     else if (cn == 3)
//     {
//         uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
//         for (int i = 0; i < len; i += vl)
//         {
//             vl = __riscv_vsetvl_e8m2(len - i);
//             vuint8m2x3_t seg = __riscv_vlseg3e8_v_u8m2x3(src + i * cn, vl);
//             __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x3_u8m2(seg, 0), vl);
//             __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x3_u8m2(seg, 1), vl);
//             __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x3_u8m2(seg, 2), vl);
//         }
//     }
//     else if (cn == 4)
//     {
//         uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
//         for (int i = 0; i < len; i += vl)
//         {
//             vl = __riscv_vsetvl_e8m2(len - i);
//             vuint8m2x4_t seg = __riscv_vlseg4e8_v_u8m2x4(src + i * cn, vl);
//             __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 0), vl);
//             __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 1), vl);
//             __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 2), vl);
//             __riscv_vse8_v_u8m2(dst3 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 3), vl);
//         }
//     }
//     else
//     {
//         int k = 0;
//         for (; k <= cn - 4; k += 4)
//         {
//             uchar *dst0 = dst[k], *dst1 = dst[k + 1], *dst2 = dst[k + 2], *dst3 = dst[k + 3];
//             for (int i = 0; i < len; i += vl)
//             {
//                 vl = __riscv_vsetvl_e8m2(len - i);
//                 vuint8m2x4_t seg = __riscv_vlsseg4e8_v_u8m2x4(src + k + i * cn, cn, vl);
//                 __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 0), vl);
//                 __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 1), vl);
//                 __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 2), vl);
//                 __riscv_vse8_v_u8m2(dst3 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 3), vl);
//             }
//         }
//         for (; k < cn; ++k)
//         {
//             uchar* dstK = dst[k];
//             for (int i = 0; i < len; i += vl)
//             {
//                 vl = __riscv_vsetvl_e8m2(len - i);
//                 vuint8m2_t seg = __riscv_vlse8_v_u8m2(src + k + i * cn, cn, vl);
//                 __riscv_vse8_v_u8m2(dstK + i, seg, vl);
//             }
//         }
//     }
//     return CV_HAL_ERROR_OK;
// }

}}  // namespace cv::cv_hal_rvv
#endif
