#ifndef __TSID_CONFIG_HH__
#define __TSID_CONFIG_HH__

// Minimal config for the extracted wbc_core TSID-style build.
// Symbol visibility handling can be refined later if this package is exported
// across shared-library boundaries on Windows.
#define TSID_DLLAPI

// wbc_core requires proxsuite, so expose the TSID ProxQP solver enum.
#define TSID_WITH_PROXSUITE

#endif  // __TSID_CONFIG_HH__
