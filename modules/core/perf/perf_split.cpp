#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<Size, MatType, int> Size_Depth_Channels_t;
typedef perf::TestBaseWithParam<Size_Depth_Channels_t> Size_Depth_Channels;

PERF_TEST_P( Size_Depth_Channels, split,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8U, CV_16S, CV_32F, CV_64F),
                 testing::Values(2, 3, 4)
             )
           )
{
    Size sz = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int channels = get<2>(GetParam());

    Mat m(sz, CV_MAKETYPE(depth, channels));
    randu(m, 0, 255);

    vector<Mat> mv;
    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) split(m, (vector<Mat>&)mv);

    SANITY_CHECK(mv, 2e-5);
}

PERF_TEST_P( Size_Depth_Channels, split2,
             testing::Combine
             (
                 testing::Values(Size(1280, 720)),
                 testing::Values(CV_8U, CV_16S, CV_32F, CV_64F),
                //  testing::Values(4, 5, 6, 7, 8)
                 testing::Values(9, 10, 11, 12)
             )
           )
{
    Size sz = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int channels = get<2>(GetParam());

    Mat m(sz, CV_MAKETYPE(depth, channels));
    randu(m, 0, 255);

    vector<Mat> mv;
    TEST_CYCLE() split(m, (vector<Mat>&)mv);

    SANITY_CHECK_NOTHING();
}

} // namespace
