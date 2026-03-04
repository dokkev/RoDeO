# Review Pass 08 - Solver 수치 안정성/실패 복구

## 범위
- `wbc_solver/wbic.cpp`

## Findings

### Critical
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:370`  
  `solve_quadprog` 호출 실패는 catch/finite 검사로 감지하지만, 반복 상한/시간 상한 제어가 노출되지 않습니다. worst-case 실행시간을 bounding하기 어렵습니다.

### Warning
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:146`  
  `JtPre`, `JtPre_dyn`는 태스크 루프마다 새 임시 행렬을 만듭니다. 반복된 dense 연산 중 캐시 미스/할당 부담이 커질 수 있습니다.
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:415`  
  `GetSolution()`에서 `trq_ = ...` 대입은 매번 전체 재평가를 수행합니다. `noalias`와 사전분해 활용으로 비용 절감 여지가 있습니다.

### Suggestion
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:396`  
  cost logging 항목(`delta_qddot_cost_`, `delta_rf_cost_`, `Xc_ddot_cost_`)을 debug telemetry로 노출하면 이상치 조기 감지가 쉽습니다.

## 개선 제안
1. solver 실행시간 upper bound를 위한 옵션(최대 반복/타임아웃) 노출.
2. 태스크 루프 임시 행렬을 멤버 버퍼로 추가 pre-alloc.
3. 실패 시 fallback torque 정책(hold/zero/last-safe)을 명시적으로 선택 가능하게 구성.

## Self-feedback
- 수학적 정확도 비교보다 실행시간/복구 관점에 집중했습니다.
- 다음 패스에서 ROS2 controller wiring 가정과 인터페이스 매핑 리스크를 검토합니다.
