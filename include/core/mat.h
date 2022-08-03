#pragma once

#include <map>
#include <memory>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace XNN
{
    typedef enum
    {
        INVALID = -1,
        // bgr or rgb: uint8
        N8UC3 = 0x00,
        // bgra or rgba: uint8
        N8UC4 = 0x01,
        // gray: uint8
        NGRAY = 0x10,
        // YUV420SP, YYYYVUVUVU
        NNV21 = 0x11,
        // YUV420SP, YYYYUVUVUV
        NNV12 = 0x12,
        // NCDi[0-4]: float
        NCHW_FLOAT = 0x20,
        // NCDi[0-4]: int32
        NC_INT32 = 0x21,

        // RESERVED FOR INTERNAL TEST USE
        RESERVED_BFP16_TEST = 0x200,
        RESERVED_FP16_TEST = 0x201,
        RESERVED_INT8_TEST = 0x202,
    } PUBLIC MatType;

    class PUBLIC Mat final
    {
    public:
        ~Mat();

        Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims, void *data);

        Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims);

        // empty mat
        Mat(DeviceType device_type, MatType mat_type);

        DEPRECATED("use Mat(DeviceType, MatType, DimsVector, void*) instead")
        Mat(DeviceType device_type, MatType mat_type, void *data) : Mat(device_type, mat_type, {1, 0, 0, 0}, data){};

    public:
        DeviceType GetDeviceType() const;
        MatType GetMatType() const;
        void *GetData() const;
        int GetBatch() const;
        int GetChannel() const;
        int GetHeight() const;
        int GetWidth() const;
        int GetDim(int index) const;
        DimsVector GetDims() const;

    private:
        Mat(){};

    protected:
        DeviceType device_type_ = DEVICE_NATIVE;
        MatType mat_type_ = INVALID;
        void *data_ = nullptr;
        DimsVector dims_ = {};

    private:
        std::shared_ptr<void> data_alloc_ = nullptr;
    };

    using MatMap = std::map<std::string, std::shared_ptr<Mat>>;
}

#pragma warning(pop)