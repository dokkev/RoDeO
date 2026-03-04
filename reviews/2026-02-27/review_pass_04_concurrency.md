# Review Pass 04 - 동시성/데이터 레이스

## 범위
- `task_reference_provider`, `state_provider`, `optimo_controller`

## Findings

### Critical
- `rpc_ros/wbc_core/wbc_architecture/include/wbc_architecture/task_reference_provider.hpp:76`  
  `SetTaskReference()`/`GetTaskReference()`는 동기화 없이 같은 `std::optional<TaskReference>`를 읽고 씁니다. non-RT writer + RT reader 동시 접근 시 데이터 레이스 위험이 있습니다.

### Warning
- `rpc_ros/wbc_core/wbc_robot_system/include/wbc_robot_system/state_provider.hpp:151`  
  contact map 값 overwrite API는 thread-safe하지 않습니다. 호출 정책(단일 RT writer)을 강제하지 않으면 race 가능성이 있습니다.
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:255`  
  `WriteEstimatorContactsNonRt()`에서 벡터 전체 복사 스냅샷을 만들어 버퍼 write. 고주파 sensor callback에서 복사 비용이 커질 수 있습니다.

### Suggestion
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:267`  
  `readFromRT()`로 최신 샘플만 읽는 구조는 적절하나, producer side overwrite 정책(최신값만 유지)와 capacity 전략을 문서화 권장.

## 개선 제안
1. TaskReferenceProvider를 `RealtimeBuffer<std::optional<TaskReference>>` 기반으로 변경.
2. StateProvider는 “RT thread only write” 규칙을 주석이 아니라 타입/API로 강제.
3. estimator snapshot은 고정 크기 구조체 또는 pre-allocated ring buffer로 축소.

## Self-feedback
- 잠재 race를 코드 패턴으로 분류했으며, 실제 호출 스레드 모델 전체 추적은 추가 확인이 필요합니다.
- 다음 패스에서 상태 흐름(SP/FSM/ControlArchitecture write ownership) 정합성을 집중 점검합니다.
