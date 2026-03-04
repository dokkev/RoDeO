# Review Pass 02 - RT 경로 예외/로그 I/O

## 범위
- `wbc_fsm`, `wbc_architecture`, `wbc_trajectory`

## Findings

### Critical
- `rpc_ros/wbc_core/wbc_architecture/src/interface/control_architecture.cpp:502`  
  `UpdateModel()`에서 입력 크기 불일치 시 `throw`를 사용합니다. update loop에서 예외 unwind가 발생하면 지연이 비결정적으로 커집니다.
- `rpc_ros/wbc_core/wbc_architecture/src/interface/control_architecture.cpp:521`  
  floating-base quaternion invalid 시 `throw` 처리. RT에서는 safe command fallback으로 변환하는 것이 안전합니다.

### Warning
- `rpc_ros/wbc_core/wbc_fsm/include/wbc_fsm/fsm_handler.hpp:151`  
  `UpdateImpl()` 전이 실패 시 `std::cerr` 출력이 들어갑니다. state 전이 오류가 반복되면 loop 인접 경로에서 I/O 블로킹 가능성이 있습니다.
- `rpc_ros/wbc_core/wbc_trajectory/src/joint_integrator.cpp:64`  
  `Integrate()`에서 미초기화 경고를 `std::cerr`로 출력합니다. 주기 함수 내 문자열 I/O는 피해야 합니다.

### Suggestion
- `rpc_ros/wbc_core/wbc_fsm/include/wbc_fsm/interface/state_machine.hpp:124`  
  `SetParameters()` 내부 경고를 runtime 단계와 분리하고 configure 단계에서만 emit 하도록 명시 분리 권장.

## 개선 제안
1. RT 경로의 `throw`는 `return false + safe fallback` 패턴으로 통일.
2. `std::cerr/std::cout`는 `RCLCPP_*_THROTTLE` 또는 비-RT diagnostic channel로 이관.
3. 오류 latch 플래그와 함께 1회 로깅(`*_ONCE`)을 기본화.

## Self-feedback
- 예외 자체의 합리성(초기화 단계 vs RT 단계)은 일부 케이스에서 더 세분화가 필요합니다.
- 다음 패스에서 release 빌드에서 사라지는 assert 리스크를 별도로 검증합니다.
