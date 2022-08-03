#include <iomanip>
#include <sstream>

#include "xnn/core/abstract_device.h"
#include "xnn/core/blob_impl.h"
#include "xnn/memory_manager/blob_memory_size_info.h"
#include "xnn/utils/data_flag_utils.h"

namespace XNN
{
    BlobImpl::BlobImpl(BlobDesc desc) : desc_(desc), alloc_memory_(false)
    {
    }

    BlobImpl::BlobImpl(BlobDesc desc, bool alloc_memory) : desc_(desc), alloc_memory_(false)
    {
        if (alloc_memory)
        {
            auto device = GetDevice(desc.device_type);

            if (device != nullptr)
            {
                BlobMemorySizeInfo size_info = device->Calculate(desc);
                device->Allocate(&handle_.base, size_info);
            }
        }
    }

    BlobImpl::BlobImpl(BlobDesc desc, BlobHandle handle) : desc_(desc), handle_(handle), alloc_memory_(false)
    {
    }

    BlobImpl::~BlobImpl()
    {
        if (alloc_memory_ && handle_.base != nullptr)
        {
            auto device = GetDevice(desc.device_type);
            if (device != nullptr)
            {
                device->Free(handle_.base);
            }
        }
    }

    BlobDesc &BlobImpl::GetBlobDesc()
    {
        return desc_;
    }

    void BlobImpl::SetBlobDesc(BlobDesc desc)
    {
        desc_ = desc;
    }

    BlobHandle BlobImpl::GetHandle()
    {
        return handle_;
    }

    void BlobImpl::SetHandle(BlobHandle handle)
    {
        if (alloc_memory_)
        {
            auto device = GetDevice(desc.device_type);
            if (device != nullptr)
            {
                device->Free(handle_.base);
            }
        }

        handle_ = handle;
        alloc_memory_ = false;
    }

    bool BlobImpl::NeedAllocateInForward()
    {
        return DataFlagUtils::AllocateInForward(flag_);
    }

    bool BlobImpl::IsConstant()
    {
        return DataFlagUtils::ChangeStatus(flag_) > 0;
    }

    int BlobImpl::GetFlag()
    {
        return flag_;
    }

    void BlobImpl::SetFlag(int flag)
    {
        flag_ = flag;
    }

}