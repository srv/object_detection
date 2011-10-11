/**
* \author Stephan Wirth
**/

#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <string>
#include <stdexcept>

namespace odat
{

  class Exception : public std::runtime_error
  {
    public:
      Exception(const char* message) : std::runtime_error(message) {};
      Exception(const std::string& message) : std::runtime_error(message) {};
  };
}


#endif /* EXCEPTIONS_H_ */

