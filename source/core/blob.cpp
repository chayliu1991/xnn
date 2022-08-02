#include <iomanip>
#include <sstream>

#include "xnn.core/blob_impl.h"
#include "xnn/core/blob.h"

namespace XNN
{
    std::string BlobDesc::description(bool all_mesage)
    {
        std::ostringstream os;

        os << "name " << name;
        os << " data type: " << data_type;
        os << " shape: [" << ;
        for (auto iter : dims)
            os << iter << " ";
        os << "]";

        return os.str();
    }

    Blob::Blob(BlobDesc desc)
    {
        impl_.reset(new BlobImpl(desc));
    }

    Blob::Blob(BlobDesc desc, bool alloc_memory)
    {
        impl_.reset(new BlobImpl(desc, alloc_memory));
    }

    Blob::Blob(BlobDesc desc, BlobHandle handle)
    {
        impl_.reset(new BlobImpl(desc, handle));
    }

    ~Blob() {}

    void Blob::SetBlobDesc(BlobDesc desc)
    {
        impl_->SetBlobDesc(desc);
    }

    BlobDesc &Blob::GetBlobDesc()
    {
        return impl_->GetBlobDesc();
    }

    void Blob::SetHandle(BlobHandle handle)
    {
        impl_->SetHandle(handle);
    }

    GetHandle Blob::GetHandle()
    {
        return impl_->GetHandle();
    }

    bool Blob::NeedAllocateInforward()
    {
        return impl_->NeedAllocateInforward();
    }

    bool Blob::IsConstant()
    {
        return impl_->IsConstant();
    }

    bool Blob::GetFlag()
    {
        return impl_->GetFlag();
    }

    void Blob::SetFlag(int flag)
    {
        impl_->SetFlag(flag);
    }
}