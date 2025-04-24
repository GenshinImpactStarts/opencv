// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {
using namespace perf;

CV_ENUM(FlagType, 0, GEMM_1_T, GEMM_2_T, GEMM_3_T, GEMM_1_T | GEMM_2_T, GEMM_2_T | GEMM_3_T)

typedef tuple<Size, FlagType, MatType> GemmParams;
typedef TestBaseWithParam<GemmParams> GemmFixture;

PERF_TEST_P(
    GemmFixture,
    Gemm,
    ::testing::Combine(::testing::Values(Size(640, 640), Size(1280, 1280), Size(1920, 1920)),
                       ::testing::Values(FlagType(0)),
                       ::testing::Values(CV_32FC1, CV_64FC1)))
    // ::testing::Combine(::testing::Values(Size(640, 640), Size(1280, 1280), Size(1920, 1920)),
    //                    FlagType::all(),
    //                    ::testing::Values(CV_32FC1, CV_32FC2, CV_64FC1, CV_64FC2)))
{
    GemmParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int flags = get<1>(params);
    const int type = get<2>(params);

    UMat src1(srcSize, type), src2(srcSize, type), src3(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, src3).out(dst);
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);
    randu(src3, -10.0f, 10.0f);

    TEST_CYCLE() cv::gemm(src1, src2, 0.6, src3, 1.5, dst, flags);

    SANITY_CHECK_NOTHING();
}

}}  // namespace opencv_test
