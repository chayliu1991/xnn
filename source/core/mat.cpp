
#include "xnn/core/mat.h"

#include "xnn/core/abstract_device.h"
#include "xnn/utils/dims_utils.h"

namespace XNN
{

    Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims, void *data)
    {
        dims_ = dims;
        data_alloc_ = nullptr;

        device_type_ = device_type;
        mat_type_ = mat_type;
        data_ = data;
    }

    Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims)
    {
        dims_ = shape_dims;
        auto device = GetDevice(device_type);
        ASSERT(device != nullptr);

        int count = DimsVectorUtils::Count(dims);
        if (count < 0)
        {
            LOGE("Mat::Mat has invalid dims with count < 0\n");
        }
        ASSERT(count >= 0);

        device_type_ = device_type;
        mat_type_ = mat_type;
        void *data_alloc = nullptr;
        auto status = device->Allocate(&data_alloc, mat_type, dims);
        if (status == XNN_OK)
        {
            data_alloc_ = std::shared_ptr<void>(data_alloc, [=](void *p)
                                                {
                                                    auto device = GetDevice(device_type);
                                                    if (device)
                                                    {
                                                        device->Free(p);
                                                    } });
            data_ = data_alloc_.get();
        }
        else
        {
            data_ = nullptr;
            data_alloc_ = nullptr;
        }
    }

    // empty mat
    Mat(DeviceType device_type, MatType mat_type)
    {
        device_type_ = device_type;
        mat_type_ = mat_type;
        data_ = nullptr;
        data_alloc_ = nullptr;
    }

    Mat::~Mat()
    {
        data_alloc_ = nullptr;
        data_ = nullptr;
    }

    DeviceType Mat::GetDeviceType() const
    {
        return device_type_;
    }

    DeviceType Mat::GetMatType() const
    {
        return mat_type_;
    }

    void *Mat::GetData() const
    {
        return data_;
    }

    DimsVector Mat::GetDims() const
    {
        return dims_;
    }

    int Mat::GetDim(int index) const
    {
        if ((index >= 0) && index < (int)dims_.size())
        {
            return dims_[index];
        }
        else
        {
            return 0;
        }
    }

    int Mat::GetBatch() const
    {
        return GetDim(0);
    }

    int Mat::GetChannel() const
    {
        return GetDim(1);
    }

    int Mat::GetHeight() const
    {
        return GetDim(2);
    }

    int Mat::GetWidth() const
    {
        return GetDim(3);
    }

}