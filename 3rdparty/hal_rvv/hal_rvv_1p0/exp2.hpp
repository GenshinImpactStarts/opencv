// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"
#include "hal_rvv_1p0/div.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_exp32f
#define cv_hal_exp32f cv::cv_hal_rvv::exp32f
#undef cv_hal_exp64f
#define cv_hal_exp64f cv::cv_hal_rvv::exp64f

namespace detail {

static constexpr double exp_log2e = 1.44269504088896340736;  // log_2(e) == 1 / ln(2)

static constexpr int exp32f_power_min = -127;
static constexpr int exp32f_power_max = 128;
static constexpr int exp64f_power_min = -1023;
static constexpr int exp64f_power_max = 1024;

static constexpr size_t exp32f_power_shift = 23;
static constexpr size_t exp64f_power_shift = 52;

// Taylor expansion coefficients for exp(x / log2(e)) == exp(ln(2) x) == 2^x
static constexpr float exp32f_p7 = 0.000015252733804;
static constexpr float exp32f_p6 = 0.000154035303934;
static constexpr float exp32f_p5 = 0.001333355814643;
static constexpr float exp32f_p4 = 0.009618129107628;
static constexpr float exp32f_p3 = 0.055504108664822;
static constexpr float exp32f_p2 = 0.240226506959101;
static constexpr float exp32f_p1 = 0.693147180559945;
static constexpr float exp32f_p0 = 1.000000000000000;

// Pade approximant coefficients for exp(x / log2(e)) == exp(ln(2) x) == 2^x
static constexpr double exp64f_p6 = 1.667048743872468e-07;
static constexpr double exp64f_p5 = 1.010118041396094e-05;
static constexpr double exp64f_p4 = 2.914584578069235e-04;
static constexpr double exp64f_p3 = 0.005045828060438;
static constexpr double exp64f_p2 = 0.054596933399796;
static constexpr double exp64f_p1 = 0.346573590279973;
static constexpr double exp64f_p0 = 1.000000000000000;

// Taylor expansion
template <typename RVV_T,
          typename T = typename RVV_T::ElemType,
          typename VT = typename RVV_T::VecType>
inline T exp32f(T x, size_t vl, VT p0, VT p1, VT p2, VT p3, VT p4, VT p5)
{
    x = __riscv_vfmul(x, exp_log2e, vl);

    auto pow_i = cv::cv_hal_rvv::RVV_ToInt<RVV_T>::cast(x, vl);
    auto clamped_i = __riscv_vmax(__riscv_vmin(pow_i, exp32f_power_max, vl), exp32f_power_min, vl);
    auto pow_val = RVV_T::reinterpret(
        __riscv_vsll(__riscv_vadd(clamped_i, -exp32f_power_min, vl), exp32f_power_shift, vl));

    x = __riscv_vfsub(x, RVV_T::cast(pow_i, vl), vl);

    auto base = __riscv_vfadd(__riscv_vfmul(x, exp32f_p7, vl), exp32f_p6, vl);
    base = __riscv_vfmadd(base, x, p5, vl);
    base = __riscv_vfmadd(base, x, p4, vl);
    base = __riscv_vfmadd(base, x, p3, vl);
    base = __riscv_vfmadd(base, x, p2, vl);
    base = __riscv_vfmadd(base, x, p1, vl);
    base = __riscv_vfmadd(base, x, p0, vl);

    // in case base = -inf but pow = 0
    return __riscv_vfmul(pow_val, base, vl);
    // return __riscv_vfmul_mu(__riscv_vmfne(pow_val, 0.0, vl), pow_val, pow_val, base, vl);
}

// Pade approximant
template <typename RVV_T,
          typename T = typename RVV_T::ElemType,
          typename VT = typename RVV_T::VecType>
inline T exp64f(T x, size_t vl, VT p0, VT p1, VT p2)
{
    x = __riscv_vfmul(x, exp_log2e, vl);

    auto pow_i = cv::cv_hal_rvv::RVV_ToInt<RVV_T>::cast(x, vl);
    auto clamped_i = __riscv_vmax(__riscv_vmin(pow_i, exp64f_power_max, vl), exp64f_power_min, vl);
    auto pow_val = RVV_T::reinterpret(
        __riscv_vsll(__riscv_vadd(clamped_i, -exp64f_power_min, vl), exp64f_power_shift, vl));

    x = __riscv_vfsub(x, RVV_T::cast(pow_i, vl), vl);
    auto x2 = __riscv_vfmul(x, x, vl);

    auto even = __riscv_vfadd(__riscv_vfmul(x2, exp64f_p6, vl), exp64f_p4, vl);
    even = __riscv_vfmadd(even, x2, p2, vl);
    even = __riscv_vfmadd(even, x2, p0, vl);

    auto odd = __riscv_vfadd(__riscv_vfmul(x2, exp64f_p5, vl), exp64f_p3, vl);
    odd = __riscv_vfmadd(odd, x2, p1, vl);
    odd = __riscv_vfmul(odd, x, vl);

    auto base =
        detail::div<RVV_T, 3>(__riscv_vfadd(even, odd, vl), __riscv_vfsub(even, odd, vl), vl);

    // in case base = -inf but pow = 0
    return __riscv_vfmul(pow_val, base, vl);
    // return __riscv_vfmul_mu(__riscv_vmfne(pow_val, 0.0, vl), pow_val, pow_val, base, vl);
}

}  // namespace detail

inline int exp32f(const float* src, float* dst, int _len)
{
    using RVV_T = RVV_F32M2;
    size_t vl = RVV_T::setvlmax();
    auto p0 = RVV_T::vmv(detail::exp32f_p0, vl);
    auto p1 = RVV_T::vmv(detail::exp32f_p1, vl);
    auto p2 = RVV_T::vmv(detail::exp32f_p2, vl);
    auto p3 = RVV_T::vmv(detail::exp32f_p3, vl);
    auto p4 = RVV_T::vmv(detail::exp32f_p4, vl);
    auto p5 = RVV_T::vmv(detail::exp32f_p5, vl);
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = RVV_T::setvl(len);
        auto x = RVV_T::vload(src, vl);
        RVV_T::vstore(dst, detail::exp32f<RVV_T>(x, vl, p0, p1, p2, p3, p4, p5), vl);
    }

    return CV_HAL_ERROR_OK;
}

inline int exp64f(const double* src, double* dst, int _len)
{
    using RVV_T = RVV_F64M2;
    size_t vl = RVV_T::setvlmax();
    auto p0 = RVV_T::vmv(detail::exp64f_p0, vl);
    auto p1 = RVV_T::vmv(detail::exp64f_p1, vl);
    auto p2 = RVV_T::vmv(detail::exp64f_p2, vl);
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = RVV_T::setvl(len);
        auto x = RVV_T::vload(src, vl);
        RVV_T::vstore(dst, detail::exp64f<RVV_T>(x, vl, p0, p1, p2), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
