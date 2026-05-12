#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "joint_impedance_controller/impedance_handler.hpp"

namespace
{

class TempDir
{
public:
  TempDir()
  {
    const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
    path_ = std::filesystem::temp_directory_path() / ("impedance_handler_test_" + stamp);
    std::filesystem::create_directories(path_);
  }

  ~TempDir()
  {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  std::string file_path(const std::string & name) const
  {
    return (path_ / name).string();
  }

  void write_file(const std::string & name, const std::string & content) const
  {
    const auto full_path = path_ / name;
    std::filesystem::create_directories(full_path.parent_path());
    std::ofstream out(full_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out << content;
    ASSERT_TRUE(out.good());
  }

private:
  std::filesystem::path path_;
};

void expect_vectors_near(
  const std::vector<double> & actual,
  const std::vector<double> & expected,
  double tolerance = 1e-12)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
  }
}

}  // namespace

TEST(ImpedanceHandlerTest, InterpolatesFromMinMaxPreset)
{
  TempDir temp_dir;
  temp_dir.write_file(
    "impedance_preset.yaml",
    R"(impedance_preset:
  min:
    stiffness: [0.0, 0.0, 0.0, 0.0]
    damping: [0.0, 0.0, 0.0, 0.0]
  max:
    stiffness: [2.0, 4.0, 6.0, 8.0]
    damping: [0.2, 0.4, 0.6, 0.8]
)");

  joint_impedance_controller::ImpedanceHandler handler(
    4, temp_dir.file_path("impedance_preset.yaml"));

  std::string error;
  ASSERT_TRUE(handler.set_level(5.0, &error)) << error;

  const auto gains = handler.gains();
  expect_vectors_near(gains.stiffness, std::vector<double>({1.0, 2.0, 3.0, 4.0}));
  expect_vectors_near(gains.damping, std::vector<double>({0.1, 0.2, 0.3, 0.4}));
  EXPECT_EQ(handler.anchor_levels(), std::vector<double>({0.0, 10.0}));
}

TEST(ImpedanceHandlerTest, RejectsLegacyNumericAnchorFormat)
{
  TempDir temp_dir;
  temp_dir.write_file(
    "impedance_preset.yaml",
    R"(impedance_preset:
  0.0:
    stiffness: [0.0, 0.0]
    damping: [0.0, 0.0]
  10.0:
    stiffness: [10.0, 20.0]
    damping: [1.0, 2.0]
)");

  EXPECT_THROW(
    {
      joint_impedance_controller::ImpedanceHandler handler(
        2, temp_dir.file_path("impedance_preset.yaml"));
    },
    std::runtime_error);
}
