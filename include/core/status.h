#pragma once

#include <memory>
#include <string>
#include <vector>

#include "xnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace XNN
{
    enum StatusCode
    {
        XNN_OK = 0x0,

        //@  param errcode
        XNNERR_PARAM_ERR = 0x1000,
        XNNERR_INVALID_NETCFG = 0x1002,
        XNNERR_INVALID_LAYERCFG = 0x1003,
        XNNERR_NULL_PARAM = 0x1004,
        XNNERR_INVALID_GROUP = 0x1005,
        XNNERR_INVALID_AXIS = 0x1006,

        //@  network errcode
        XNNERR_NET_ERR = 0x2000,
        XNNERR_UNSUPPORT_NET = 0x2001,

        //@  layer errcode
        XNNERR_LAYER_ERR = 0x3000,
        XNNERR_UNKNOWN_LAYER = 0x3001,
        XNNERR_CREATE_LAYER = 0x3002,
        XNNERR_INIT_LAYER = 0x3003,
        XNNERR_INVALID_DATA = 0x3004,
        XNNERR_ELT_UNSUP_OP = 0x3005,

        //@  model errcode
        XNNERR_MODEL_ERR = 0x4000,
        XNNERR_INVALID_MODEL = 0x4001,
        XNNERR_FIND_MODEL = 0x4002,

        //@  instance errcode
        XNNERR_INST_ERR = 0x5000,
        XNNERR_MAXINST_COUNT = 0x5001,
        XNNERR_ALLOC_INSTANCE = 0x5002,
        XNNERR_INVALID_INSTANCE = 0x5003,
        XNNERR_CONTEXT_ERR = 0x5004,

        //@  common errcode
        XNNERR_COMMON_ERROR = 0x6000,
        XNNERR_OUTOFMEMORY = 0x6001,
        XNNERR_INVALID_INPUT = 0x6002,
        XNNERR_FIND_RESOURCE = 0x6003,
        XNNERR_NO_RESULT = 0x6004,
        XNNERR_LOAD_MODEL = 0x6005,
        XNNERR_PACK_MODEL = 0x6006,
        XNNERR_SET_CPU_AFFINITY = 0x6007,
        XNNERR_OPEN_FILE = 0x6008,

        //@  forward memory error
        XNNERR_NOT_SUPPORT_SET_FORWARD_MEM = 0x8000,
        XNNERR_FORWARD_MEM_NOT_SET = 0x8001,
        XNNERR_SHARED_MEMORY_FORWARD_NOT_SAME_THREAD = 0x8003,
        XNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT = 0x8004,

        //@  device
        XNNERR_DEVICE_NOT_SUPPORT = 0x9000,
        XNNERR_DEVICE_LIBRARY_LOAD = 0x9001,
        XNNERR_DEVICE_CONTEXT_CREATE = 0x9002,
        XNNERR_DEVICE_INVALID_COMMAND_QUEUE = 0x9003,
        XNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT = 0x9004,

        //@  OpenCL
        XNNERR_OPENCL_FINISH_ERROR = 0xA000,
        XNNERR_OPENCL_API_ERROR = 0xA001,
        XNNERR_OPENCL_RUNTIME_ERROR = 0xA002,
        XNNERR_OPENCL_ACC_INIT_ERROR = 0xA003,
        XNNERR_OPENCL_ACC_RESHAPE_ERROR = 0xA004,
        XNNERR_OPENCL_ACC_FORWARD_ERROR = 0xA005,
        XNNERR_OPENCL_KERNELBUILD_ERROR = 0xA006,
        XNNERR_OPENCL_MEMALLOC_ERROR = 0xA007,
        XNNERR_OPENCL_MEMMAP_ERROR = 0xA008,
        XNNERR_OPENCL_MEMUNMAP_ERROR = 0xA009,
        XNNERR_OPENCL_UNSUPPORT_ERROR = 0xA00A,

        //@  SNPE
        XNNERR_SNPE_API_ERROR = 0xB001,

        //@  Atlas
        XNNERR_ATLAS_RUNTIME_ERROR = 0xC001,
        XNNERR_ATLAS_TIMEOUT_ERROR = 0xC002,
        XNNERR_ATLAS_MALLOC_ERROR = 0xC003,
        XNNERR_ATLAS_GRAPH_INIT_ERROR = 0xC004,

        //@  Hiai
        XNNERR_HIAI_API_ERROR = 0xD001,

        //@  Huawei NPU
        XNNERR_NPU_LOAD_ERROR = 0xE000,
        XNNERR_NPU_UNSUPPORT_ERROR = 0xE001,
        XNNERR_NPU_HIAI_API_ERROR = 0xE002,

        //@  Cuda
        XNNERR_CUDA_TENSORRT_ERROR = 0xF001,
        XNNERR_CUDA_SYNC_ERROR = 0xF002,
        XNNERR_CUDA_MEMCPY_ERROR = 0xF003,
        XNNERR_CUDA_KERNEL_LAUNCH_ERROR = 0xF004,

        //@  XNN CONVERT
        XNN_CONVERT_OK = 0x10000,
        XNNERR_CONVERT_UNSUPPORT_LAYER = 0x10001,
        XNNERR_CONVERT_GENERATE_MODEL = 0x10002,
        XNNERR_CONVERT_INVALID_MODEL = 0x10003,
        XNNERR_CONVERT_UNSUPPORT_PASS = 0x10004,
        XNNERR_CONVERT_OPTIMIZE_ERROR = 0x10005,

        //@  Quantize
        XNNERR_QUANTIZE_ERROR = 0x20001,

        //@  Apple NPU
        XNNERR_IOS_VERSION_ERROR = 0x30001,
        XNNERR_ANE_CLEAN_ERROR = 0x30002,
        XNNERR_ANE_SAVE_MODEL_ERROR = 0x30003,
        XNNERR_ANE_COMPILE_MODEL_ERROR = 0x30004,
        XNNERR_ANE_EXECUTOR_ERROR = 0x30005,
        XNNERR_COREML_VERSION_ERROR = 0x30006,

    };

    class PUBLIC Status final
    {
    public:
        ~Status();
        Status(int code = XNN_OK, std::string &message = "OK");

        Status &operator(int code);

        bool operator==(int code);
        bool operator!=(int code);

        operator int();
        operator bool();

        std::string description();

    private:
        int code_ = 0;
        std::string message_ = "";
    };
}
#pragma warning(pop)