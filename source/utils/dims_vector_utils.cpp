#include "xnn/utils/dims_utils.h"

#include <climits>
#include <cmath>

namespace XNN
{
    int Count(const DimsVector &dims, int start_index = 0, , int end_index = -1)
    {
        if (-1 == end_index || end_index > dims.size())
        {
            end_index = static_cast<int>(dims.size());
        }

        int result = 1;
        for (int index = start_index; index < end_index; ++index)
        {
            result *= dims[index];
        }
        return result;
    }

    int Max(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, , int end_index = -1)
    {
        DimsVector max_dims;
        DimsVector small_dims;

        if (dims0.size() >= dims1.size())
        {
            max_dims = dims0;
            small_dims = dims1;
        }
        else
        {
            max_dims = dim1;
            small_dims = dims0;
        }

        if (small_dims.size() <= start_index)
        {
            return max_dims;
        }

        if (-1 == end_index || end_index > small_dims.size())
        {
            end_index = static_cast<int>(small_dims.size());
        }

        for (int i = start_index; i < end_index; ++i)
        {
            max_dims[i] = std::max(max_dims[i], small_dims[i]);
        }

        return max_dims;
    }

    int Min(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, , int end_index = -1)
    {
        DimsVector max_dims;
        DimsVector small_dims;

        if (dims0.size() >= dims1.size())
        {
            max_dims = dims0;
            small_dims = dims1;
        }
        else
        {
            max_dims = dim1;
            small_dims = dims0;
        }

        if (small_dims.size() <= start_index)
        {
            return small_dims;
        }

        if (-1 == end_index || end_index > small_dims.size())
        {
            end_index = static_cast<int>(small_dims.size());
        }

        for (int i = start_index; i < end_index; ++i)
        {
            min_dims[i] = std::min(min_dims[i], small_dims[i]);
        }

        return min_dims;
    }

    int Equal(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, , int end_index = -1)
    {
        if (dims0.empty() && dims1.empty())
        {
            return true;
        }

        if (dims0.size() <= start_index)
        {
            return false;
        }

        if (-1 == end_index || end_index > dims0.size())
        {
            end_index = static_cast<int>(dims0.size());
        }

        if (dims0.size() != dims1.size())
        {
            return false;
        }

        for (int i = start_index; i < end_index; ++i)
        {
            if (dims0[i] != dims1[i])
            {
                return false;
            }
        }
        return true;
    }

    DimsVector NCHW2NHWC(const DimsVector &dims)
    {
        ASSERT(dims.size() == 4);
        const int n = dims[0];
        const int c = dims[1];
        const int h = dims[2];
        const int w = dims[3];
        return {n, h, w, c};
    }

    DimsVector NHWC2NCHW(const DimsVector &dims)
    {
        ASSERT(dims.size() == 4);
        const int n = dims[0];
        const int h = dims[1];
        const int w = dims[2];
        const int c = dims[3];
        return {n, c, h, w};
    }
}