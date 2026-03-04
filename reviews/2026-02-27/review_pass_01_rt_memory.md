# Review Pass 01 - RT 메모리/할당

## 범위
- `wbc_solver`, `wbc_robot_system`, `wbc_architecture`

## Findings

### Critical
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:120`  
  `FindConfiguration()`에서 `stacked_contact_jacobian_`, `stacked_contact_jdot_qdot_`, `stacked_contact_op_cmd_`를 매 틱 `resize()` 합니다. contact topology가 고정인 경우 RT 힙 할당이 반복됩니다.
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:248`  
  `BuildContactMtxVect()`에서 `Jc_`, `JcDotQdot_`, `Uf_vec_`를 매 호출 `resize()` 합니다.
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:282`  
  `GetDesiredReactionForce()`에서 `des_rf_.resize(total_force_dim)`를 반복합니다.

### Warning
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:337`  
  QP 내부 버퍼는 캐시되어 있으나 dimension 변경 시 다중 `resize()`가 발생합니다. contact 수가 런타임에 변하면 스파이크가 생깁니다.
- `rpc_ros/wbc_core/wbc_robot_system/src/pinocchio_robot_system.cpp:299`  
  `UpdateState()`가 매 틱 `std::map::find`를 2회/관절 수행하여 `O(N log N)` 문자열 비교 비용을 유발합니다.

### Suggestion
- `rpc_ros/wbc_core/wbc_robot_system/include/wbc_robot_system/state_provider.hpp:121`  
  contact를 `unordered_map<string,...>`로 유지하고 RT에서 `find()`를 수행합니다. 링크 수가 고정이면 index 기반 배열로 전환하는 편이 캐시 친화적입니다.

## 개선 제안
1. contact/force 최대 크기 기준으로 solver 버퍼를 생성자에서 1회 pre-alloc.
2. contact topology 변경은 비-RT 구간에서만 허용하고, RT에서는 `setZero/head` 방식으로 재사용.
3. `PinocchioRobotSystem::UpdateState()`는 이름 기반 map 대신 ordered vector 또는 precomputed index map 입력 API 추가.

## Self-feedback
- 이번 패스는 힙 할당과 컨테이너 접근 비용에 집중했고, 수치 안정성 검토는 제외했습니다.
- 다음 패스에서 예외/로그 I/O가 RT 경로에 남았는지 분리 검토가 필요합니다.
