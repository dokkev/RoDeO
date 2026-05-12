#ifndef PLATO_HARDWARE_INTERFACE__USEFUL_FUNCTIONS_HPP_
#define PLATO_HARDWARE_INTERFACE__USEFUL_FUNCTIONS_HPP_


#include <cmath>

namespace Plato{

    /// @brief PI
    constexpr float PI = 3.141592f;

    /// @brief Convert degrees to radians
    /// @param degrees angle in degree
    /// @return angle in radians
    constexpr float deg2rad(float degrees){
        return degrees * PI / 180.0f;
    }

    /// @brief Convert radians to degrees
    /// @param radians input angle in radians
    /// @return output angle in degrees
    constexpr float rad2deg(float radians){
        return radians * 180.0f / PI;
    }

    /// @brief Check if two float numbers are almost equal
    /// @param a first float number
    /// @param b second float number
    /// @param epsilon tolerance value with default value 1e-5
    /// @return true if the two numbers are almost equal
    inline bool almost_equal(float a, float b, float epsilon = 1e-5f) {
        return std::fabs(a - b) < epsilon;
    }
} // namespace Plato



#endif // PLATO_HARDWARE_INTERFACE__USEFUL_FUNCTIONS_HPP_