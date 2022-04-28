#ifndef CNET_IMGLOGGER_HPP
#define CNET_IMGLOGGER_HPP

#include <string>
#include "cnet_common.hpp"

namespace Cnet
{

    class ImageLogger
    {
    private:

        std::string _location;

    public:

        ImageLogger(const std::string& location)
        {
            _location = location;
        }

        void log_image(MatrixRm *matrix, const std::string& cat, const std::string& name, const size_t iter = 0)
        {
            std::string out_path = _location;
            if (iter != 0)
            {
                out_path += "/" + std::to_string(iter);
            }
            out_path += "/" + cat;
            if (!std::experimental::filesystem::exists(out_path)) {
                std::experimental::filesystem::create_directories(out_path);
            }
            save_image(matrix, out_path + "/" + name);
        }
    };

}

#endif //CNET_IMGLOGGER_HPP
