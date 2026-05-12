#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace joint_impedance_controller
{

struct ImpedanceGains
{
  std::vector<double> stiffness;
  std::vector<double> damping;
};

class ImpedanceHandler
{
public:
  explicit ImpedanceHandler(
    int joint_count,
    std::string preset_yaml_path);

  bool set_level(
    double level,
    std::string * error_out = nullptr);

  void set_custom_gains(
    const std::vector<double> & stiffness,
    const std::vector<double> & damping);

  ImpedanceGains gains() const;

  double active_level() const;

  std::string active_label() const;

  std::vector<double> anchor_levels() const;

private:
  bool load_presets(std::string * error_out);
  ImpedanceGains interpolate_gains(double level) const;

  std::vector<double> sanitize_vector(
    const std::vector<double> & in,
    double fallback) const;

  struct LevelAnchor
  {
    double level;
    ImpedanceGains gains;
  };

  const int joint_count_;
  const std::string preset_yaml_path_;

  mutable std::mutex mutex_;
  std::vector<LevelAnchor> anchors_;
  std::vector<double> available_anchor_levels_;
  std::optional<double> active_level_;
  bool using_custom_gains_{false};
  ImpedanceGains gains_;
};

}  // namespace joint_impedance_controller
