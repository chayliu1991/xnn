#pragma once

#include <cstring>
#include <map>
#include <memory>

#include "xnn/core/abstract_device.h"

namespace XNN
{
    class CpuDevice final : public AbstractDevice
    {
    public:
        explicit CpuDevice(DeviceType device_type);

        ~CpuDevice();

        virtual BlobMemorySizeInfo Calculate(BlobDesc &desc) override;

        virtual Status Allocate(void **handle, MatType mat_type, DimsVectr dims) override;

        virtual Status Allocate(void **handle, BlobMemorySizeInfo &size_info) override;

        virtual Status Free(void *handle) override;

        virtual Status CopyToDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) override;

        virtual Status CopyFromDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) override;

        virtual AbstractLayerAcc *CreateLayerAcc(LayerType type) override;

        virtual Context *CreateContext(int device_id) override;

        virtual NetworkType ConverAutoworkType() override;

        static Status RegisterLayerAccCreater(LayerType type, LayerAccCreator *creator);

    private:
        static std::map<LayerType, std::shared_ptr<LayerAccCreator>> &GetLayerCreatorMap();
    };

    template <typename T>
    class CpuTypeLayerAccRegister
    {
    public:
        explicit CpuTypeLayerAccRegister(LayerType type)
        {
            CpuDevice::RegisterLayerAccCreater(type, new T());
        }
    };
}
