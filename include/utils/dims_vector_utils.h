#pragma once

#include <algorithm>

#include "xnn/core/common.h"
#include "xnn/core/macro.h"
#include "xnn/core/status.h"

namespace XNN
{
    class PUBLIC DimsVectorUtils final
    {
    public:
        static int Count(const DimsVector &dims, int start_index = 0, , int end_index = -1);

        static int Max(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, , int end_index = -1);

        static int Min(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, , int end_index = -1);

        static int Equal(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, , int end_index = -1);

        static DimsVector NCHW2NHWC(const DimsVector &dims);

        static DimsVector NHWC2NCHW(const DimsVector &dims);
    };
}