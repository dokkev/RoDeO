# Review Pass 05 - 상태 흐름/소유권 정합성

## 범위
- `control_architecture`, `state_machine`, `state_provider`

## Findings

### Warning
- `rpc_ros/wbc_core/wbc_architecture/src/interface/control_architecture.cpp:327`  
  `Step()`에서 `sp_->current_time_`, `sp_->nominal_jpos_`를 갱신합니다.
- `rpc_ros/wbc_core/wbc_fsm/include/wbc_fsm/interface/state_machine.hpp:161`  
  동시에 `StateMachine::UpdateStateTime()`도 `sp_->current_time_`, `sp_->state_`, `sp_->is_first_visit_`를 갱신합니다. 작성 주체가 2곳이라 추론 난이도가 높습니다.

### Suggestion
- `rpc_ros/wbc_core/wbc_fsm/include/wbc_fsm/interface/state_machine.hpp:145`  
  FSM 메타데이터(`state_`, `prev_state_`, `is_first_visit_`)의 단일 writer를 FSMHandler로 고정하고, ControlArchitecture는 읽기 전용으로 두는 구조가 가독성/안전성에 유리합니다.
- `rpc_ros/wbc_core/wbc_robot_system/include/wbc_robot_system/state_provider.hpp:95`  
  StateProvider 공개 멤버 직접 접근 대신 setter/getter 또는 context update API로 수렴 권장.

## 개선 제안
1. SP write ownership을 문서가 아니라 코드로 고정(단일 update entrypoint).
2. `StateProvider`는 공개 필드 축소 후 `ApplyFsmContext(...)` 같은 명시 API 제공.
3. ControlArchitecture/FSMHandler 간 상태 동기화 경계를 인터페이스로 분리.

## Self-feedback
- 기능 버그보다 장기 유지보수 리스크 중심의 패스였습니다.
- 다음 패스에서 API/네이밍 일관성 문제를 집중 정리합니다.
