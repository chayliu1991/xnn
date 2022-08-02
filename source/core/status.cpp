#include "xnn/core/status.h"

namespace XNN
{
    std::string StatusGetDefaultMessage(int code)
    {
        switch (code)
        {
        case XNNERR_INVALID_NETCFG:
            return "invalid net config, proto or model is invalid";
        case XNNERR_SET_CPU_AFFINITY:
            return "failed to set cpu affinity";
        case XNNERR_DEVICE_NOT_SUPPORT:
            return "device is nil or unsupported";
        case XNNERR_DEVICE_CONTEXT_CREATE:
            return "context is nil or created failed";
        default:
            return "";
        }
    }

    Status::Status(int code, std::string &message) : code_(code),
                                                     message_((message != "OK" && message.length() > 0) ? message : StatusGetDefaultMessage(code))
    {
    }

    Status::~Status()
    {
        code_ = 0;
        message_ = "";
    }

    Status &Status::operator=(int code)
    {
        code_ = code;
        message_ = StatusGetDefaultMessage(code);
        return *this;
    }

    bool Status::operator==(int code)
    {
        return code_ == code;
    }

    bool Status::operator==(int code)
    {
        return code_ != code;
    }

    Status::operator int()
    {
        return code_;
    }

    Status::operator bool()
    {
        return code_ == XNN_OK;
    }

    std::string Status::description()
    {
        std::ostringstream os;
        os << "code: 0x" << std::uppercase << std::setfill('0') << std::setw(4) << std::hex << code_
           << " msg: " << message_;
        return os.str();
    }
}