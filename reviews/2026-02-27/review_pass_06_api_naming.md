# Review Pass 06 - API/네이밍 일관성

## 범위
- `robot_system`, `control_architecture`, `rpc_example`

## Findings

### Warning
- `rpc_ros/wbc_core/wbc_robot_system/include/wbc_robot_system/interface/robot_system.hpp:64`  
  `GetX()` 계열과 레거시 별칭(`GetQIdx`, `GetRobotComPos`)이 동시에 존재해 API 표면이 큽니다.
- `rpc_ros/wbc_core/wbc_architecture/include/wbc_architecture/interface/control_architecture.hpp:172`  
  `GetFsmHandler()/GetStateProvider()/CurrentTime()/ControlDt()` 등 naming 스타일이 혼용됩니다.

### Suggestion
- `rpc_ros/wbc_core/wbc_robot_system/include/wbc_robot_system/interface/robot_system.hpp:266`  
  `b_fixed_base_`, `b_print_info_` 레거시 public alias는 이미 deprecate 되었으므로 제거 로드맵을 문서화 권장.
- `rpc_ros/wbc_core/rpc_example/include/rpc_example/example_state_provider.hpp:27`  
  `b_ee_contact_` 같은 헝가리안 prefix는 신규 코드와 컨벤션 충돌 가능성이 있습니다.

## 개선 제안
1. API naming rule을 패키지 전역으로 결정 (`GetX` 또는 `x()` 중 하나).
2. deprecate API 제거 target release를 명시.
3. 신규 코드에서 `b_` prefix 금지 lint 규칙 도입.

## Self-feedback
- 실동작 영향이 즉시 크지 않아 우선순위는 낮지만, 누적 비용은 높은 항목들입니다.
- 다음 패스에서 YAML 파서/컴파일러의 검증 강도를 점검합니다.
