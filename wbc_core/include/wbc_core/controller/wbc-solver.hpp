//
// Copyright (c) 2026
//
// Compatibility header for the final-form WBMC core.
//
// The implementation truth now lives in `wbmc.hpp/cpp`. This header preserves
// the historic path and type aliases while the codebase converges on the new
// WBMC-centered architecture.
//

#ifndef __wbc_controller_wbc_solver_hpp__
#define __wbc_controller_wbc_solver_hpp__

#include "wbc_core/controller/wbmc.hpp"

namespace tsid {

using WBCSolution = WBMCSolution;
using WBCSolver = WBMC;

}  // namespace tsid

#endif  // ifndef __wbc_controller_wbc_solver_hpp__
