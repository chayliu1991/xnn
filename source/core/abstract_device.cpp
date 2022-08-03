
#include "xnn/core/abstract_device.h"

#include <map>
#include <mutex>

namespace XNN
{
    AbstractDevice::AbstractDevice(DeviceType device_type) : device_type_(device_type)
    {
    }

    AbstractDevice::~AbstractDevice()
    {
    }

    DeviceType AbstractDevice::GetDeviceType()
    {
        return device_type_;
    }

    Status AbstractDevice::Allocate(BlobHandle *handle, BlobMemorySizeInfo &size_info)
    {
        void *data = nullptr;

        auto status = Allocate(&data, size_info);
        if (status != XNN_OK)
        {
            return status;
        }

        handle->base = data;
        handle->bytes_offset = 0;

        return XNN_OK;
    }

    std::shared_ptr<const ImplementdPrecision> AbstractDevice::GetImplementedPrecision(LayerType type)
    {
        return std::make_shared<ImplementdPrecision>();
    }

    std::shared_ptr<const ImplementedLayout> AbstractDevice::GetImplementedLayout(LayerType type)
    {
        return std::make_shared<ImplementedLayout>();
    }

    AbstractDevice *GetDevice(DeviceType type)
    {
        return GetGlobalDeviceMap()[type].get();
    }

    std::map<DeviceType, std::shared_ptr<AbstractDevice>> &GetGlobalDeviceMap()
    {
        static std::once_flag once;
        static std::shared_ptr<std::map<DeviceType, std::shared_ptr<AbstractDevice>>> device_map;
        std::call_once(once, []()
                       { device_map.reset(new std::map<DeviceType, std::shared_ptr<AbstractDevice>>); });
        return *device_map;
    }
}