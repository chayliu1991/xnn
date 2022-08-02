#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "xnn/core/blob.h"
#include "xnn/core/common.h"
#include "xnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace XNN
{
    class PUBLIC BlobImpl final
    {
        explicit BlobImpl(BlobDesc desc);

        BlobImpl(BlobDesc desc, bool alloc_memory);

        BlobImpl(BlobDesc desc, BlobHandle handle);

        virtual ~BlobImpl();

        BlobDesc &GetBlobDesc();
        void SetBlobDesc(BlobDesc desc);

        BlobHandle GetHandle();
        void SetHandle(BlobHandle handle);

        bool NeedAllocateInForward();

        bool IsConstant();

        int GetFlag();
        void SetFlag(int flag);

    private:
        BlobDesc desc_;
        BlobHandle handle_;
        bool alloc_memory_;

        //@ 0: data alwalys change
        //@ 1: data change if shape differ
        //@ 2: data never change
        int flag_ = DATA_FLAG_CHANGE_ALWAYS;
    };
}

#pragma warning(pop)