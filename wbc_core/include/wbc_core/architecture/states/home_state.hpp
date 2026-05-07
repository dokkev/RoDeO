//
// Copyright (c) 2026
//
// Home state: identical to Initialize but typically stay_here=true.
//

#ifndef WBC_CORE_ARCHITECTURE_STATES_HOME_STATE_HPP_
#define WBC_CORE_ARCHITECTURE_STATES_HOME_STATE_HPP_

#include "wbc_core/architecture/states/initialize_state.hpp"

namespace wbc {

// Home is functionally identical to Initialize (ramp to target_jpos, then hold).
using HomeState = InitializeState;

}  // namespace wbc

#endif
