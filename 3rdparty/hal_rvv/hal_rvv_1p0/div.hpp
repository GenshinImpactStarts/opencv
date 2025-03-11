// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv { namespace detail {

template <typename RVV_T, size_t iter_times, typename T = typename RVV_T::VecType>
inline T div(T a, T b, size_t vl)
{
    auto res = __riscv_vfrec7(b, vl);
#pragma unroll
    for (size_t i = 0; i < iter_times; i++)
    {
        auto tmp = RVV_T::vmv(2.0, vl);
        tmp = __riscv_vfnmsac(tmp, b, res, vl);
        res = __riscv_vfmul(res, tmp, vl);
    }
    return __riscv_vfmul(a, res, vl);
}

}}}  // namespace cv::cv_hal_rvv::detail
