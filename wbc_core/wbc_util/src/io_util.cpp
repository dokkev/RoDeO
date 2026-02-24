#include "wbc_util/io_util.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>

namespace util
{
namespace {
std::list<std::string> gs_file_name_string;
}

void SaveVector(const Eigen::VectorXd & vec_, std::string name_, bool b_param)
{
  std::string file_name;
  util::CleaningFile(name_, file_name, b_param);

  std::ofstream savefile(file_name.c_str(), std::ios::app);
  for (int i(0); i < vec_.rows(); ++i) {
    savefile << vec_(i) << "\t";
  }
  savefile << "\n";
  savefile.flush();
}

void SaveValue(double _value, std::string _name, bool b_param)
{
  std::string file_name;
  util::CleaningFile(_name, file_name, b_param);
  std::ofstream savefile(file_name.c_str(), std::ios::app);

  savefile << _value << "\n";
  savefile.flush();
}

void SaveVector(double *_vec, std::string _name, int size, bool b_param)
{
  std::string file_name;
  util::CleaningFile(_name, file_name, b_param);
  std::ofstream savefile(file_name.c_str(), std::ios::app);

  for (int i(0); i < size; ++i) {
    savefile << _vec[i] << "\t";
  }
  savefile << "\n";
  savefile.flush();
}

void SaveVector(
  const std::vector<double> & _vec, std::string _name,
  bool b_param)
{
  std::string file_name;
  util::CleaningFile(_name, file_name, b_param);
  std::ofstream savefile(file_name.c_str(), std::ios::app);
  for (std::size_t i = 0; i < _vec.size(); ++i) {
    savefile << _vec[i] << "\t";
  }
  savefile << "\n";
  savefile.flush();
}

void CleaningFile(
  std::string _file_name, std::string & _ret_file,
  bool b_param)
{
  if (b_param) {
    _ret_file += THIS_COM;
  } else {
    _ret_file += THIS_COM "experiment_data/";
  }

  _ret_file += _file_name;
  _ret_file += ".txt";

  std::list<std::string>::iterator iter =
    std::find(gs_file_name_string.begin(), gs_file_name_string.end(), _file_name);
  if (gs_file_name_string.end() == iter) {
    gs_file_name_string.push_back(_file_name);
    remove(_ret_file.c_str());
  }
}

void PrettyConstructor(const int & _num_tab, const std::string & _name)
{
  int color;
  util::ColorPrint(color::kBoldRed, "|", false);
  std::string content = " ";
  int space_to_go(0);
  if (_num_tab != 0) {
    for (int i = 0; i < _num_tab; ++i) {
      content += "    ";
    }
    content = content + "||--" + _name;
    switch (_num_tab) {
      case 1:
        color = color::kBoldGreen;
        break;
      case 2:
        color = color::kBoldYellow;
        break;
      case 3:
        color = color::kBoldBlue;
        break;
      case 4:
        color = color::kBoldMagneta;
        break;
      default:
        assert(false);
    }
  } else {
    content += _name;
    color = color::kBoldRed;
  }
  space_to_go = 78 - content.length();
  // std::cout << space_to_go << std::endl;
  for (int i = 0; i < space_to_go; ++i) {
    content += " ";
  }
  util::ColorPrint(color, content, false);
  util::ColorPrint(color::kBoldRed, "|");
}

void ColorPrint(const int & _color, const std::string & _name, bool line_change)
{
  switch (_color) {
    case color::kRed:
      printf("\033[0;31m");
      break;
    case color::kBoldRed:
      printf("\033[1;31m");
      break;
    case color::kGreen:
      printf("\033[0;32m");
      break;
    case color::kBoldGreen:
      printf("\033[1;32m");
      break;
    case color::kYellow:
      printf("\033[0;33m");
      break;
    case color::kBoldYellow:
      printf("\033[1;33m");
      break;
    case color::kBlue:
      printf("\033[0;34m");
      break;
    case color::kBoldBlue:
      printf("\033[1;34m");
      break;
    case color::kMagneta:
      printf("\033[0;35m");
      break;
    case color::kBoldMagneta:
      printf("\033[1;35m");
      break;
    case color::kCyan:
      printf("\033[0;36m");
      break;
    case color::kBoldCyan:
      printf("\033[1;36m");
      break;
    default:
      std::cout << "No Such Color" << std::endl;
      exit(0);
  }
  if (line_change) {
    printf("%s\n", _name.c_str());
  } else {
    printf("%s", _name.c_str());
  }
  printf("\033[0m");
}

void PrettyPrint(
  Eigen::VectorXd const & vv, std::ostream & os,
  std::string const & title, std::string const & prefix,
  bool nonl)
{
  PrettyPrint((Eigen::MatrixXd const &)vv, os, title, prefix, true, nonl);
}

void PrettyPrint(
  Eigen::MatrixXd const & mm, std::ostream & os,
  std::string const & title, std::string const & prefix,
  bool vecmode, bool nonl)
{
  char const *nlornot("\n");
  if (nonl) {
    nlornot = "";
  }
  if (!title.empty()) {
    os << title << nlornot;
  }
  if ((mm.rows() <= 0) || (mm.cols() <= 0)) {
    os << prefix << " (empty)" << nlornot;
  } else {
    // if (mm.cols() == 1) {
    //   vecmode = true;
    // }

    if (vecmode) {
      if (!prefix.empty()) {
        os << prefix;
      }
      for (int ir(0); ir < mm.rows(); ++ir) {
        os << PrettyString(mm.coeff(ir, 0));
      }
      os << nlornot;

    } else {
      for (int ir(0); ir < mm.rows(); ++ir) {
        if (!prefix.empty()) {
          os << prefix;
        }
        for (int ic(0); ic < mm.cols(); ++ic) {
          os << PrettyString(mm.coeff(ir, ic));
        }
        os << nlornot;
      }
    }
  }
}
void PrettyPrint(
  Eigen::Quaternion<double> const & qq, std::ostream & os,
  std::string const & title, std::string const & prefix,
  bool nonl)
{
  PrettyPrint(qq.coeffs(), os, title, prefix, true, nonl);
}
void PrettyPrint(
  Eigen::Vector3d const & vv, std::ostream & os,
  std::string const & title, std::string const & prefix,
  bool nonl)
{
  PrettyPrint((Eigen::MatrixXd const &)vv, os, title, prefix, true, nonl);
}
void PrettyPrint(const std::vector<double> & _vec, const char *title)
{
  std::printf("%s: ", title);
  for (std::size_t i = 0; i < _vec.size(); ++i) {
    std::printf("% 6.4f, \t", _vec[i]);
  }
  std::printf("\n");
}

void PrettyPrint(const std::vector<int> & _vec, const char *title)
{
  std::printf("%s: ", title);
  for (std::size_t i = 0; i < _vec.size(); ++i) {
    std::printf("%d, \t", _vec[i]);
  }
  std::printf("\n");
}
std::string PrettyString(Eigen::VectorXd const & vv)
{
  std::ostringstream os;
  PrettyPrint(vv, os, "", "", true);
  return os.str();
}

std::string PrettyString(Eigen::MatrixXd const & mm, std::string const & prefix)
{
  std::ostringstream os;
  PrettyPrint(mm, os, "", prefix);
  return os.str();
}

std::string PrettyString(double vv)
{
  constexpr int buflen = 32;
  char buf[buflen];
  memset(buf, 0, sizeof(buf));
  snprintf(buf, buflen - 1, "% 6.6f  ", vv);
  std::string str(buf);
  return str;
}

} // namespace util
