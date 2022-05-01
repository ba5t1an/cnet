#ifndef CNET_LOGGER_HPP
#define CNET_CNET_LOGGER_HPP

#include "memory"
#include "mutex"

namespace Cnet
{

    enum LogType {
        DEBUG = 0,
        INFO = 1,
        WARN = 2,
        ERROR = 3
    };

    class Logger
    {
    private:
        std::unique_ptr<Logger> _logger;
        std::mutex _mutex;

        Logger()
        {
            _logger = nullptr;
        };
        /**
        * copy constructor for the Logger class.
        */
        Logger(const Logger&){};             // copy constructor is private
        /**
        * assignment operator for the Logger class.
        */
        Logger& operator=(const Logger&){ return *this; };  // assignment operator is private

    public:

        static std::unique_ptr<Logger>& get_logger()
        {
            if (!_logger){
                _logger = std::unique_ptr<Logger>(new Logger());
            }
            return _logger;
        }
    };
}


#endif //CNET_CNET_LOGGER_HPP
