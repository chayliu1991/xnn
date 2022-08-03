#include "xnn/device/cpu/cpu_device.h"
#include "xnn/device/cpu/cpu_context.h"
#include "xnn/utils/blob_memory_size_utils.h"

namespace XNN
{

    CpuDevice::CpuDevice(DeviceType device_type) : AbstractDevice(device_type)
    {
    }

    CpuDevice::~CpuDevice()
    {
    }

    BlobMemorySizeInfo CpuDevice::Calculate(BlobDesc &desc)
    {
        return Calculate1DMemorySize(desc);
    }

    Status CpuDevice::Allocate(void **handle, MatType mat_type, DimsVectr dims)
    {
        BlobDesc desc;
        desc.dims = dims;
        desc.device_type = DEVICE_NAIVE;

        if (mat_type == NCHW_FLOAT ||
            mat_type == RESERVED_BFP16_TEST || mat_type == RESERVED_INT8_TEST || mat_type == RESERVED_FP16_TEST)
        {
            desc.data_type = DATA_TYPE_FLOAT;
            desc.data_format = DATA_FORMAT_NCHW;
            auto size_info = Calculate(desc);
            return Allocate(handle, size_info);
        }
        else if (mat_type == N8UC3 || mat_type == NGRAY || mat_type == NNV21 || mat_type == NNV12)
        {
            desc.data_type = DATA_TYPE_INT8;
            desc.data_format = DATA_FORMAT_NCHW;
            auto size_info = Calculate(desc);
            return Allocate(handle, size_info);
        }
        else if (mat_type == N8UC4)
        {
            BlobMemorySizeInfo size_info;
            int count = desc.dims[0] * desc.dims[2] * desc.dims[3];
            size_info.data_type = DATA_TYPE_INT8;
            size_info.dims.push_back(count);
            return Allocate(handle, size_info);
        }
        else if (mat_type == NC_INT32)
        {
            auto size_info = Calculate(desc);
            size_info.data_type = DATA_TYPE_INT32;
            return Allocate(handle, size_info);
        }
        else
        {
            LOGE("CpuDevice dont support mat_type:%d\n", mat_type);
            return Status(XNNERR_PARAM_ERR, "cpu dont support mat_type");
        }
    }

    Status CpuDevice::Allocate(void **handle, BlobMemorySizeInfo &size_info)
    {
        if (handle)
        {
            auto size = GetBlobMemoryBytesSize(size_info);
            if (size > 0)
            {
                *handle = malloc(size);
                if (*handle && size > 0)
                {
                    memset(*handle, 0, size);
                }
            }
            else if (size == 0)
            {
                *handle = nullptr;
            }
            else
            {
                return Status(XNNERR_PARAM_ERR, "CpuDevice::Allocate malloc bytes size < 0");
            }
        }

        return XNN_OK;
    }

    Status CpuDevice::Free(void *handle)
    {
        if (handle)
        {
            free(handle);
        }
        return XNN_OK;
    }

    Status CpuDevice::CopyToDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue)
    {
        auto size_info = Calculate(desc);
        size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

        memcpy(reinterpret_cast<char *>(dst->base) + dst->bytes_offset,
               reinterpret_cast<char *>(src->base) + src->bytes_offset, size_in_bytes);

        return XNN_OK;
    }

    Status CpuDevice::CopyFromDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue)
    {
        auto size_info = Calculate(desc);
        size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

        memcpy(reinterpret_cast<char *>(dst->base) + dst->bytes_offset,
               reinterpret_cast<char *>(src->base) + src->bytes_offset, size_in_bytes);

        return XNN_OK;
    }

    AbstractLayerAcc *CpuDevice::CreateLayerAcc(LayerType type)
    {
        auto &layer_creator_map = GetLayerCreatorMap();
        if (layer_creator_map.count(type) > 0)
        {
            return layer_creator_map[type]->CreateLayerAcc(type);
        }
        return nullptr;
    }

    Context *CpuDevice::CreateContext(int device_id)
    {
        return new CpuContext();
    }

    NetworkType CpuDevice::ConverAutoworkType()
    {
        return NETWORK_TYPE_DEFAULT;
    }

    Status CpuDevice::RegisterLayerAccCreater(LayerType type, LayerAccCreator *creator)
    {
        GetLayerCreatorMap()[type] = std::shared_ptr<LayerAccCreator>(creator);
    }

    std::map<LayerType, std::shared_ptr<LayerAccCreator>> &CpuDevice::GetLayerCreatorMap()
    {
        static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
        return layer_creator_map;
    }

    TypeDeviceRegister<CpuDevice> g_cpu_device_register(DEVICE_NAIVE);
}