# Review Pass 09 - ROS2 Controller Wiring/가정

## 범위
- `optimo_controller`, `optimo_hardware_interface`

## Findings

### Warning
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:120`  
  `on_activate()`에서 interface 개수만 확인하고 이름 정렬 검증을 하지 않습니다. 순서가 바뀐 하드웨어 구현과 연결되면 silent miswiring 가능성이 있습니다.
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:179`  
  `update()`가 state interface를 인덱스 기반으로 직접 읽습니다. 이름 기반 매칭 캐시가 없으면 재사용성/안전성이 떨어집니다.
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:221`  
  command write도 동일한 인덱스 가정에 의존합니다.

### Suggestion
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:62`  
  `state_interface_configuration()`이 `command_interface_configuration()`를 그대로 반환합니다. 현재는 맞지만 의도를 주석으로 더 명확히 남기면 유지보수에 유리합니다.

## 개선 제안
1. activate 시 interface name->index 매핑을 1회 검증하고 캐시.
2. update에서는 캐시 인덱스만 사용해 RT 비용을 유지.
3. miswiring 감지를 위한 startup self-check 로그 추가.

## Self-feedback
- 기능이 minimal skeleton인 점을 감안해 구조적 리스크 위주로 평가했습니다.
- 다음 패스에서 테스트/문서 커버리지와 우선순위 실행계획을 통합합니다.
