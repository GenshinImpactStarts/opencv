// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_GEMM_HPP_INCLUDED
#define OPENCV_HAL_RVV_GEMM_HPP_INCLUDED

#include <riscv_vector.h>

#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_gemm32f
#define cv_hal_gemm32f cv::cv_hal_rvv::gemm
#undef cv_hal_gemm64f
#define cv_hal_gemm64f cv::cv_hal_rvv::gemm
// #undef cv_hal_gemm32fc
// #define cv_hal_gemm32fc cv::cv_hal_rvv::gemmc
// #undef cv_hal_gemm64fc
// #define cv_hal_gemm64fc cv::cv_hal_rvv::gemmc

template <typename T>
class GemmBuf
{
public:
    T* buf;

    GemmBuf(size_t len)
    {
        if (len * sizeof(T) > 1024)
            buf = new T[len];
        else
            buf = _buf;
    }

    ~GemmBuf()
    {
        if (buf != _buf)
            delete[] buf;
    }

private:
    T _buf[1024 / sizeof(T)];
};

#define gemmNoTransKernelImpl(eew, lmul, w_or_d) \
    __asm__( \
        "vsetvli      zero, %[vl], " #eew ", " #lmul ", ta, ma\n\t" \
        "vl" #eew ".v v24, (%[pb])\n\t" \
        "add          %[pb], %[pb], %[pb_step]\n\t" \
        "vmv.v.i      %[d0], 0\n\t" \
        "vmv.v.i      %[d1], 0\n\t" \
        "vmv.v.i      %[d2], 0\n\t" \
        "vmv.v.i      %[d3], 0\n\t" \
        "fl" #w_or_d " ft0, (%[pa0])\n\t" \
        "fl" #w_or_d " ft1, (%[pa1])\n\t" \
        "fl" #w_or_d " ft2, (%[pa2])\n\t" \
        "fl" #w_or_d " ft3, (%[pa3])\n\t" \
        "addi         t6, %[n], -2\n\t" \
        "bge          zero, t6, 2f\n\t" \
        "1:\n\t" \
        "vl" #eew ".v v28, (%[pb])\n\t" \
        "add          %[pb], %[pb], %[pb_step]\n\t" \
        "fl" #w_or_d " ft4, %[size1](%[pa0])\n\t" \
        "vfmacc.vf    %[d0], ft0, v24\n\t" \
        "fl" #w_or_d " ft5, %[size1](%[pa1])\n\t" \
        "vfmacc.vf    %[d1], ft1, v24\n\t" \
        "fl" #w_or_d " ft6, %[size1](%[pa2])\n\t" \
        "vfmacc.vf    %[d2], ft2, v24\n\t" \
        "fl" #w_or_d " ft7, %[size1](%[pa3])\n\t" \
        "vfmacc.vf    %[d3], ft3, v24\n\t" \
        "vl" #eew ".v v24, (%[pb])\n\t" \
        "add          %[pb], %[pb], %[pb_step]\n\t" \
        "fl" #w_or_d " ft0, %[size2](%[pa0])\n\t" \
        "vfmacc.vf    %[d0], ft4, v28\n\t" \
        "fl" #w_or_d " ft1, %[size2](%[pa1])\n\t" \
        "vfmacc.vf    %[d1], ft5, v28\n\t" \
        "fl" #w_or_d " ft2, %[size2](%[pa2])\n\t" \
        "vfmacc.vf    %[d2], ft6, v28\n\t" \
        "fl" #w_or_d " ft3, %[size2](%[pa3])\n\t" \
        "vfmacc.vf    %[d3], ft7, v28\n\t" \
        "addi         %[pa0], %[pa0], %[size2]\n\t" \
        "addi         %[pa1], %[pa1], %[size2]\n\t" \
        "addi         %[pa2], %[pa2], %[size2]\n\t" \
        "addi         %[pa3], %[pa3], %[size2]\n\t" \
        "addi         t6, t6, -2\n\t" \
        "blt          zero, t6, 1b\n\t" \
        "2:\n\t" \
        "vfmacc.vf    %[d0], ft0, v24\n\t" \
        "vfmacc.vf    %[d1], ft1, v24\n\t" \
        "vfmacc.vf    %[d2], ft2, v24\n\t" \
        "vfmacc.vf    %[d3], ft3, v24\n\t" \
        "bnez         t6, 3f\n\t" \
        "vl" #eew ".v v24, (%[pb])\n\t" \
        "fl" #w_or_d " ft0, %[size1](%[pa0])\n\t" \
        "vfmacc.vf    %[d0], ft0, v24\n\t" \
        "fl" #w_or_d " ft1, %[size1](%[pa1])\n\t" \
        "vfmacc.vf    %[d1], ft1, v24\n\t" \
        "fl" #w_or_d " ft2, %[size1](%[pa2])\n\t" \
        "vfmacc.vf    %[d2], ft2, v24\n\t" \
        "fl" #w_or_d " ft3, %[size1](%[pa3])\n\t" \
        "vfmacc.vf    %[d3], ft3, v24\n\t" \
        "3:\n\t" \
        : [pa0] "+r"(pa0), \
          [pa1] "+r"(pa1), \
          [pa2] "+r"(pa2), \
          [pa3] "+r"(pa3), \
          [pb] "+r"(pb), \
          [d0] "=&vr"(d0), \
          [d1] "=&vr"(d1), \
          [d2] "=&vr"(d2), \
          [d3] "=&vr"(d3) \
        : [vl] "r"(vl), \
          [n] "r"(n), \
          [pb_step] "r"(vl * sizeof(T)), \
          [size1] "i"(sizeof(T)), \
          [size2] "i"(2 * sizeof(T)) \
        : "t6", \
          "ft0", \
          "ft1", \
          "ft2", \
          "ft3", \
          "ft4", \
          "ft5", \
          "ft6", \
          "ft7", \
          "v24", \
          "v25", \
          "v26", \
          "v27", \
          "v28", \
          "v29", \
          "v30", \
          "v31")

// template <typename RVV_T,
//           typename T = typename RVV_T::ElemType,
//           typename VT = typename RVV_T::VecType>
// inline void gemmNoTransKernel(
//     const T* pa0,
//     const T* pa1,
//     const T* pa2,
//     const T* pa3,
//     const T* pb,
//     size_t vl,
//     size_t n,
//     VT& d0,
//     VT& d1,
//     VT& d2,
//     VT& d3)
// {
//     T a0, a1, a2, a3, a4, a5, a6, a7;
//     typename RVV_T::VecType b0, b1;
//     d0 = RVV_T::vmv(0.0, vl);
//     d1 = RVV_T::vmv(0.0, vl);
//     d2 = RVV_T::vmv(0.0, vl);
//     d3 = RVV_T::vmv(0.0, vl);
//     b0 = RVV_T::vload(pb, vl);
//     pb += vl;
//     a0 = pa0[0], a1 = pa1[0], a2 = pa2[0], a3 = pa3[0];
//     size_t in = n;
//     for (; in > 2; in -= 2)
//     {
//         b1 = RVV_T::vload(pb, vl);
//         pb += vl;
//         __asm__ volatile("");
//         a4 = pa0[1];
//         d0 = __riscv_vfmacc(d0, a0, b0, vl);
//         __asm__ volatile("");
//         a5 = pa1[1];
//         d1 = __riscv_vfmacc(d1, a1, b0, vl);
//         __asm__ volatile("");
//         a6 = pa2[1];
//         d2 = __riscv_vfmacc(d2, a2, b0, vl);
//         __asm__ volatile("");
//         a7 = pa3[1];
//         d3 = __riscv_vfmacc(d3, a3, b0, vl);
//         __asm__ volatile("");
//         b0 = RVV_T::vload(pb, vl);
//         pb += vl;
//         __asm__ volatile("");
//         a0 = pa0[2];
//         d0 = __riscv_vfmacc(d0, a4, b1, vl);
//         __asm__ volatile("");
//         a1 = pa1[2];
//         d1 = __riscv_vfmacc(d1, a5, b1, vl);
//         __asm__ volatile("");
//         a2 = pa2[2];
//         d2 = __riscv_vfmacc(d2, a6, b1, vl);
//         __asm__ volatile("");
//         a3 = pa3[2];
//         d3 = __riscv_vfmacc(d3, a7, b1, vl);
//         __asm__ volatile("");
//         pa0 += 2, pa1 += 2, pa2 += 2, pa3 += 2;
//     }
//     d0 = __riscv_vfmacc(d0, a0, b0, vl);
//     d1 = __riscv_vfmacc(d1, a1, b0, vl);
//     d2 = __riscv_vfmacc(d2, a2, b0, vl);
//     d3 = __riscv_vfmacc(d3, a3, b0, vl);
//     if (in == 2)
//     {
//         b0 = RVV_T::vload(pb, vl);
//         d0 = __riscv_vfmacc(d0, pa0[1], b0, vl);
//         d1 = __riscv_vfmacc(d1, pa1[1], b0, vl);
//         d2 = __riscv_vfmacc(d2, pa2[1], b0, vl);
//         d3 = __riscv_vfmacc(d3, pa3[1], b0, vl);
//     }
// }

template <typename RVV_T,
          typename T = typename RVV_T::ElemType,
          typename VT = typename RVV_T::VecType>
inline void gemmNoTransKernel(
    const T* pa0,
    const T* pa1,
    const T* pa2,
    const T* pa3,
    const T* pb,
    size_t vl,
    size_t n,
    VT& d0,
    VT& d1,
    VT& d2,
    VT& d3);

template <>
inline void gemmNoTransKernel<RVV_F32M4>(
    const float* pa0,
    const float* pa1,
    const float* pa2,
    const float* pa3,
    const float* pb,
    size_t vl,
    size_t n,
    vfloat32m4_t& d0,
    vfloat32m4_t& d1,
    vfloat32m4_t& d2,
    vfloat32m4_t& d3)
{
    using T = RVV_F32M4::ElemType;
    gemmNoTransKernelImpl(e32, m4, w);
}

template <>
inline void gemmNoTransKernel<RVV_F64M4>(
    const double* pa0,
    const double* pa1,
    const double* pa2,
    const double* pa3,
    const double* pb,
    size_t vl,
    size_t n,
    vfloat64m4_t& d0,
    vfloat64m4_t& d1,
    vfloat64m4_t& d2,
    vfloat64m4_t& d3)
{
    using T = RVV_F64M4::ElemType;
    gemmNoTransKernelImpl(e64, m4, d);
}

template <typename T>
inline void gemmNoTrans(
    const T* src1,
    size_t _src1_step,
    const T* src2,
    size_t _src2_step,
    T alpha,
    const T* src3,
    size_t _src3_step,
    T beta,
    T* dst,
    size_t _dst_step,
    int _m,
    int _n,
    int _k,
    bool transC)
{
    using RVV_T = RVV<T, LMUL_4>;
    size_t vl;
    size_t m = _m, n = _n, k = _k;
    size_t src1_step = _src1_step / sizeof(T), src2_step = _src2_step / sizeof(T),
           src3_step = _src3_step / sizeof(T), dst_step = _dst_step / sizeof(T);
    GemmBuf<T> buf(_n * RVV_T::setvlmax());

    for (size_t ik = 0; ik < k; ik += vl)
    {
        vl = RVV_T::setvl(k - ik);
        for (size_t in = 0; in < n; ++in)
            RVV_T::vstore(buf.buf + in * vl, RVV_T::vload(src2 + ik + in * src2_step, vl), vl);

        size_t im = 0;
        for (; im + 4 <= m; im += 4)
        {
            const T* pa0 = src1 + im * src1_step;
            const T* pa1 = src1 + (im + 1) * src1_step;
            const T* pa2 = src1 + (im + 2) * src1_step;
            const T* pa3 = src1 + (im + 3) * src1_step;
            T* pb = buf.buf;
            const T* pc = transC ? src3 + ik * src3_step + im : src3 + im * src3_step + ik;
            T* pd = dst + im * dst_step + ik;
            typename RVV_T::VecType d0, d1, d2, d3;
            gemmNoTransKernel<RVV_T>(pa0, pa1, pa2, pa3, pb, vl, n, d0, d1, d2, d3);
            d0 = __riscv_vfmul(d0, alpha, vl);
            d1 = __riscv_vfmul(d1, alpha, vl);
            d2 = __riscv_vfmul(d2, alpha, vl);
            d3 = __riscv_vfmul(d3, alpha, vl);
            if (src3)
            {
                if (!transC)
                {
                    d0 = __riscv_vfmacc(d0, beta, RVV_T::vload(pc + 0 * src3_step, vl), vl);
                    d1 = __riscv_vfmacc(d1, beta, RVV_T::vload(pc + 1 * src3_step, vl), vl);
                    d2 = __riscv_vfmacc(d2, beta, RVV_T::vload(pc + 2 * src3_step, vl), vl);
                    d3 = __riscv_vfmacc(d3, beta, RVV_T::vload(pc + 3 * src3_step, vl), vl);
                }
                else
                {
                    d0 = __riscv_vfmacc(d0, beta, RVV_T::vload_stride(pc + 0, _src3_step, vl), vl);
                    d1 = __riscv_vfmacc(d1, beta, RVV_T::vload_stride(pc + 1, _src3_step, vl), vl);
                    d2 = __riscv_vfmacc(d2, beta, RVV_T::vload_stride(pc + 2, _src3_step, vl), vl);
                    d3 = __riscv_vfmacc(d3, beta, RVV_T::vload_stride(pc + 3, _src3_step, vl), vl);
                }
            }
            RVV_T::vstore(pd + 0 * dst_step, d0, vl);
            RVV_T::vstore(pd + 1 * dst_step, d1, vl);
            RVV_T::vstore(pd + 2 * dst_step, d2, vl);
            RVV_T::vstore(pd + 3 * dst_step, d3, vl);
        }
        size_t res = m - im;
        if (res)
        {
            const T* pa0 = src1 + im * src1_step;
            const T* pa1 = res > 1 ? src1 + (im + 1) * src1_step : pa0;
            const T* pa2 = res > 2 ? src1 + (im + 2) * src1_step : pa0;
            const T* pa3 = res > 3 ? src1 + (im + 3) * src1_step : pa0;
            T* pb = buf.buf;
            const T* pc = transC ? src3 + ik * src3_step + im : src3 + im * src3_step + ik;
            T* pd = dst + im * dst_step + ik;
            typename RVV_T::VecType d0, d1, d2, d3;
            gemmNoTransKernel<RVV_T>(pa0, pa1, pa2, pa3, pb, vl, n, d0, d1, d2, d3);
            d0 = __riscv_vfmul(d0, alpha, vl);
            d1 = __riscv_vfmul(d1, alpha, vl);
            d2 = __riscv_vfmul(d2, alpha, vl);
            d3 = __riscv_vfmul(d3, alpha, vl);
            if (src3)
            {
                if (transC)
                {
                    const T* pc0 = pc;
                    const T* pc1 = res > 1 ? pc + 1 : pc;
                    const T* pc2 = res > 2 ? pc + 2 : pc;
                    const T* pc3 = res > 3 ? pc + 3 : pc;
                    d0 = __riscv_vfmacc(d0, beta, RVV_T::vload_stride(pc0, _src3_step, vl), vl);
                    d1 = __riscv_vfmacc(d1, beta, RVV_T::vload_stride(pc1, _src3_step, vl), vl);
                    d2 = __riscv_vfmacc(d2, beta, RVV_T::vload_stride(pc2, _src3_step, vl), vl);
                    d3 = __riscv_vfmacc(d3, beta, RVV_T::vload_stride(pc3, _src3_step, vl), vl);
                }
                else
                {
                    const T* pc0 = pc;
                    const T* pc1 = res > 1 ? pc + 1 * src3_step : pc;
                    const T* pc2 = res > 2 ? pc + 2 * src3_step : pc;
                    const T* pc3 = res > 3 ? pc + 3 * src3_step : pc;
                    d0 = __riscv_vfmacc(d0, beta, RVV_T::vload(pc0, vl), vl);
                    d1 = __riscv_vfmacc(d1, beta, RVV_T::vload(pc1, vl), vl);
                    d2 = __riscv_vfmacc(d2, beta, RVV_T::vload(pc2, vl), vl);
                    d3 = __riscv_vfmacc(d3, beta, RVV_T::vload(pc3, vl), vl);
                }
            }
            T* pd0 = pd;
            T* pd1 = res > 1 ? pd + 1 * dst_step : pd;
            T* pd2 = res > 2 ? pd + 2 * dst_step : pd;
            T* pd3 = res > 3 ? pd + 3 * dst_step : pd;
            RVV_T::vstore(pd0, d0, vl);
            RVV_T::vstore(pd1, d1, vl);
            RVV_T::vstore(pd2, d2, vl);
            RVV_T::vstore(pd3, d3, vl);
        }
    }
}

template <typename T>
inline void gemmTransA(
    const T* src1,
    size_t _src1_step,
    const T* src2,
    size_t _src2_step,
    T alpha,
    const T* src3,
    size_t _src3_step,
    T beta,
    T* dst,
    size_t _dst_step,
    int _m,
    int _n,
    int _k,
    bool transC)
{
    using RVV_T = RVV<T, LMUL_4>;
    size_t vl;
    size_t m = _m, n = _n, k = _k;
    size_t src1_step = _src1_step / sizeof(T), src2_step = _src2_step / sizeof(T),
           src3_step = _src3_step / sizeof(T), dst_step = _dst_step / sizeof(T);
    GemmBuf<T> buf(_n * RVV_T::setvlmax());
}

template <typename T>
inline int
    gemm(const T* src1,
         size_t _src1_step,
         const T* src2,
         size_t _src2_step,
         T alpha,
         const T* src3,
         size_t _src3_step,
         T beta,
         T* dst,
         size_t _dst_step,
         int _m,
         int _n,
         int _k,
         int flags)
{
    if (!_m || !_n || !_k)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    bool transA = flags & CV_HAL_GEMM_1_T;
    bool transB = flags & CV_HAL_GEMM_2_T;
    bool transC = flags & CV_HAL_GEMM_3_T;
    if (!transA && !transB)
    {
        gemmNoTrans(
            src1,
            _src1_step,
            src2,
            _src2_step,
            alpha,
            src3,
            _src3_step,
            beta,
            dst,
            _dst_step,
            _m,
            _n,
            _k,
            transC);
    }
    // else if (transA && !transB)
    // {
    //     gemmTransA(
    //         src1,
    //         _src1_step,
    //         src2,
    //         _src2_step,
    //         alpha,
    //         src3,
    //         _src3_step,
    //         beta,
    //         dst,
    //         _dst_step,
    //         _m,
    //         _n,
    //         _k,
    //         transC);
    // }
    else
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

inline int gemmc(
    const float* src1,
    size_t src1_step,
    const float* src2,
    size_t src2_step,
    float alpha,
    const float* src3,
    size_t src3_step,
    float beta,
    float* dst,
    size_t dst_step,
    int m,
    int n,
    int k,
    int flags)
{
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}  // namespace cv::cv_hal_rvv

#endif  // OPENCV_HAL_RVV_GEMM_HPP_INCLUDED
