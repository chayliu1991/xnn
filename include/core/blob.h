#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "xnn/core/common.h"
#include "xnn/core/macro.h"
#include "xnn/utils/dims_vector_utils.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namepace XNN
{
    //@brief BlobDesc blob data info
    struct PUBLIC BlobDesc final
    {
        //@ device_type describes device cpu, gpu, ...
        DeviceType device_type = DEVICE_NAIVE;
        //@ data_type describes data precision fp32, in8, ...
        DataType data_type = DATA_TYPE_FLOAT;
        //@ data_format describes data order nchw, nhwc, ...
        DataFormat data_format = DATA_FORMAT_AUTO;
        //@ DimsVector describes data dims
        DimsVector dims;
        //@ name describes the blob name
        std::string name = "";

        std::string description(bool all_message = false);
    };

    struct PUBLIC BlobHandle final
    {
        void *base = nullptr;
        uint64_t bytes_offset = 0;
    };

    class BlobImpl;

    class PUBLIC Blob final
    {
    public:
        explicit Blob(BlobDesc desc);

        Blob(BlobDesc desc, bool alloc_memory);

        Blob(BlobDesc desc, BlobHandle handle);

        virtual ~Blob();

        BlobDesc &GetBlobDesc();
        void SetBlobDesc(BlobDesc desc);

        BlobHandle GetHandle();
        void SetHandle(BlobHandle handle);

        bool NeedAllocateInForward();

        bool IsConstant();

        int GetFlag();
        int SetFlag(int flag);

    private:
        std::unique_ptr<BlobImpl> impl_;
    };

    using InputShapeMap = std::map<std::string, DimsVector>;
    using InputDataTypeMap = std::map<std::string, DataType>;

    using BlobMap = std::map<std::string, Blob *>;
}

#pragma warning(pop)