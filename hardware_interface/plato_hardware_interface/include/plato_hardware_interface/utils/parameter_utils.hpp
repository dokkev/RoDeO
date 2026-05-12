#ifndef PLATO_HARDWARE_INTERFACE__PARAMETER_UTILS_HPP_
#define PLATO_HARDWARE_INTERFACE__PARAMETER_UTILS_HPP_

#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>

namespace plato_hardware_interface::utils
{

inline bool parse_bool_parameter(const std::string & value, const char * parameter_name)
{
  if (value == "true" || value == "True" || value == "1") {
    return true;
  }

  if (value == "false" || value == "False" || value == "0") {
    return false;
  }

  throw std::invalid_argument(
          std::string("Hardware parameter '") + parameter_name +
          "' must be one of: true, false, 1, 0");
}

inline std::chrono::microseconds parse_nonnegative_microseconds_parameter(
  const std::string & value,
  const char * parameter_name)
{
  long long parsed = 0;
  try {
    parsed = std::stoll(value);
  } catch (const std::exception &) {
    throw std::invalid_argument(
            std::string("Hardware parameter '") + parameter_name +
            "' must be a non-negative integer (microseconds)");
  }

  if (parsed < 0) {
    throw std::invalid_argument(
            std::string("Hardware parameter '") + parameter_name +
            "' must be >= 0 (microseconds)");
  }

  return std::chrono::microseconds(parsed);
}

inline double parse_nonnegative_double_parameter(
  const std::string & value,
  const char * parameter_name)
{
  double parsed = 0.0;
  try {
    parsed = std::stod(value);
  } catch (const std::exception &) {
    throw std::invalid_argument(
            std::string("Hardware parameter '") + parameter_name +
            "' must be a non-negative number");
  }

  if (!std::isfinite(parsed) || parsed < 0.0) {
    throw std::invalid_argument(
            std::string("Hardware parameter '") + parameter_name +
            "' must be a finite value >= 0");
  }

  return parsed;
}

}  // namespace plato_hardware_interface::utils

#endif  // PLATO_HARDWARE_INTERFACE__PARAMETER_UTILS_HPP_
