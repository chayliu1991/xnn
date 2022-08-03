#pragma once

#include "xnn/core/abstract_layer_acc.h"
#include "xnn/core/blob.h"
#include "xnn/core/common.h"
#include "xnn/core/context.h"
#include "xnn/core/layer_type.h"
#include "xnn/core/status.h"
#include "xnn/memory_manager/blob_memory_size_info.h"
#include "xnn/utils/blob_converter.h"

namespace XNN
{
    struct ImplementedPrecision final
    {
        bool fp32_implemented = false;
        bool fp16_implemented = false;
        bool bfp16_implemented = false;
    };

    struct ImplementedLayout final
    {
        std::vector<DataFormat> layouts;
    };

    class AbstractDevice
    {
    public:
        explicit AbstractDevice(DeviceType);

        virtual ~AbstractDevice();

        virtual BlobMemorySizeInfo Calculate(BlobDesc &desc) = 0;

        virtual Status Allocate(void **handle, MatType mat_type, DimsVectr dims) = 0;

        virtual Status Allocate(void **handle, BlobMemorySizeInfo &size_info) = 0;

        virtual Status Free(void *handle) = 0;

        virtual Status CopyToDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) = 0;

        virtual Status CopyFromDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) = 0;

        virtual AbstractLayerAcc *CreateLayerAcc(LayerType type) = 0;

        virtual Context *CreateContext(int device_id) = 0;

        virtual std::shared_ptr<const ImplementdPrecision> GetImplementedPrecision(LayerType type);

        virtual std::shared_ptr<const ImplementedLayout> GetImplementedLayout(LayerType type);

        DeviceType GetDeviceType();

        virtual NetworkType ConverAutoworkType() = 0;

    private:
        DeviceType device_type_;
    };

    std::map<DeviceType, std::shared_ptr<AbstractDevice>> &GetGlobalDeviceMap();

    AbstractDevice *GetDevice(DeviceType type);

    template <typename T>
    class TypeDeviceRegister final
    {
    public:
        explicit TypeDeviceRegister(DeviceType type)
        {
            auto &device_map = GetGlobalDeviceMap();
            if (device_map.find(type) == device.end())
            {
                device_map[type] = std::shared_ptr<T>(new T(type));
            }
        }
    };
}