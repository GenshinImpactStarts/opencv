// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <cmath>
#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_log32f
#define cv_hal_log32f cv::cv_hal_rvv::log32f
// #undef cv_hal_log64f
// #define cv_hal_log64f cv::cv_hal_rvv::log64f

namespace detail {

static constexpr union
{
    float f;
    int32_t i;
} log_half = {0.5f};

static constexpr float log_SQRTHF_fp32 = 0.707106781186547524f;
static constexpr float log_q1_fp32 = -2.12194440E-4f;
static constexpr float log_q2_fp32 = 0.693359375f;
static constexpr float log_p0_fp32 = 7.0376836292E-2f;
static constexpr float log_p1_fp32 = -1.1514610310E-1f;
static constexpr float log_p2_fp32 = 1.1676998740E-1f;
static constexpr float log_p3_fp32 = -1.2420140846E-1f;
static constexpr float log_p4_fp32 = 1.4249322787E-1f;
static constexpr float log_p5_fp32 = -1.6668057665E-1f;
static constexpr float log_p6_fp32 = 2.0000714765E-1f;
static constexpr float log_p7_fp32 = -2.4999993993E-1f;
static constexpr float log_p8_fp32 = 3.3333331174E-1f;

static constexpr int32_t log_inv_mant_mask_s32 = ~0x7f800000;

}  // namespace detail

inline int log32f(const float* src, float* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m4(len);
        auto in = __riscv_vle32_v_f32m4(src, vl);
        auto ux = __riscv_vreinterpret_i32m4(in);
        auto emm0 = __riscv_vsra(ux, 23, vl);

        ux = __riscv_vand(ux, detail::log_inv_mant_mask_s32, vl);
        ux = __riscv_vor(ux, detail::log_half.i, vl);
        auto x = __riscv_vreinterpret_f32m4(ux);

        emm0 = __riscv_vsub(emm0, 0x7f, vl);
        auto e = __riscv_vfcvt_f_x_v_f32m4(emm0, vl);

        e = __riscv_vfadd(e, 1.0, vl);

        auto mask = __riscv_vmflt(x, detail::log_SQRTHF_fp32, vl);
        auto tmp = __riscv_vfsub(x, 1.0, vl);
        e = __riscv_vfsub_mu(mask, e, e, 1.0, vl);
        x = __riscv_vfadd_mu(mask, x, tmp, x, vl);

        auto z = __riscv_vfmul(x, x, vl);

        auto y = __riscv_vfadd(__riscv_vfmul(x, detail::log_p0_fp32, vl), detail::log_p1_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p2_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p3_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p4_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p5_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p6_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p7_fp32, vl);
        y = __riscv_vfadd(__riscv_vfmul(y, x, vl), detail::log_p8_fp32, vl);
        y = __riscv_vfmul(y, x, vl);
        y = __riscv_vfmul(y, z, vl);

        y = __riscv_vfmacc(y, detail::log_q1_fp32, e, vl);

        y = __riscv_vfnmsac(y, 0.5, z, vl);

        x = __riscv_vfadd(x, y, vl);
        x = __riscv_vfmacc(x, detail::log_q2_fp32, e, vl);

        x = __riscv_vfmerge(x, -INFINITY, __riscv_vmfeq(in, 0.0, vl), vl);
        x = __riscv_vfmerge(x, NAN, __riscv_vmflt(in, 0.0, vl), vl);
        x = __riscv_vfmerge(x, INFINITY, __riscv_vmfeq(in, INFINITY, vl), vl);

        __riscv_vse32(dst, x, vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
