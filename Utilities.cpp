#include "Utilities.h"
#include <sstream>

namespace Utils
{
    std::vector<std::string> SplitStringOnWhiteSpace(const std::string &input)
    {
        std::vector<std::string> result;
        std::istringstream istr(input);
        std::string data;
        while (istr >> data)
        {
            result.push_back(data);
        }
        return result;
    }
}
