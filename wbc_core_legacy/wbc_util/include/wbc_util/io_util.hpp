/**
 * @file wbc_core/wbc_util/include/wbc_util/io_util.hpp
 * @brief Doxygen documentation for io_util module.
 */
#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef THIS_COM
#define THIS_COM ""
#endif

namespace color {
constexpr int kRed = 0;
constexpr int kBoldRed = 1;
constexpr int kGreen = 2;
constexpr int kBoldGreen = 3;
constexpr int kYellow = 4;
constexpr int kBoldYellow = 5;
constexpr int kBlue = 6;
constexpr int kBoldBlue = 7;
constexpr int kMagneta = 8;
constexpr int kBoldMagneta = 9;
constexpr int kCyan = 10;
constexpr int kBoldCyan = 11;
} // namespace color

namespace util {

void SaveVector(const Eigen::VectorXd& vec, std::string name,
                bool b_param = false);
void SaveVector(double* vec, std::string name, int size, bool b_param = false);
void SaveVector(const std::vector<double>& vec, std::string name,
                bool b_param = false);
void SaveValue(double value, std::string name, bool b_param = false);
void CleaningFile(std::string file_name, std::string& ret_file, bool b_param);

void PrettyConstructor(const int& num_tab, const std::string& name);
void ColorPrint(const int& color, const std::string& name,
                bool line_change = true);

void PrettyPrint(Eigen::VectorXd const& vv, std::ostream& os,
                 std::string const& title, std::string const& prefix = "",
                 bool nonl = false);
void PrettyPrint(Eigen::MatrixXd const& mm, std::ostream& os,
                 std::string const& title, std::string const& prefix = "",
                 bool vecmode = false, bool nonl = false);
void PrettyPrint(Eigen::Quaternion<double> const& qq, std::ostream& os,
                 std::string const& title, std::string const& prefix = "",
                 bool nonl = false);
void PrettyPrint(Eigen::Vector3d const& vv, std::ostream& os,
                 std::string const& title, std::string const& prefix = "",
                 bool nonl = false);
void PrettyPrint(const std::vector<double>& vec, const char* title);
void PrettyPrint(const std::vector<int>& vec, const char* title);

std::string PrettyString(Eigen::VectorXd const& vv);
std::string PrettyString(Eigen::MatrixXd const& mm,
                         std::string const& prefix = "");
std::string PrettyString(double vv);

} // namespace util
