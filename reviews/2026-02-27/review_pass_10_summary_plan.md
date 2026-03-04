# Review Pass 10 - 종합 정리/실행 우선순위

## 범위
- `rpc_ros` 전체(소스/헤더)

## 핵심 이슈 Top 5

### Critical
- `rpc_ros/wbc_core/wbc_solver/src/wbic.cpp:120`  
  RT 루프 내 가변 `resize()` 다수.
- `rpc_ros/wbc_core/wbc_util/include/wbc_util/yaml_util.hpp:26`  
  라이브러리 레벨 `std::exit` 호출.

### Warning
- `rpc_ros/wbc_core/wbc_architecture/include/wbc_architecture/task_reference_provider.hpp:76`  
  TaskReference 접근 동기화 부재(잠재 race).
- `rpc_ros/controller/optimo_controller/src/optimo_controller.cpp:120`  
  interface name 매핑 검증 없이 index 가정.
- `rpc_ros/wbc_core/wbc_formulation/src/contact_constraint.cpp:81`  
  release에서 사라지는 assert 기반 입력 검증.

## 개선 로드맵
1. 1주차: WBIC 버퍼 pre-alloc + RT assert/throw 정리.
2. 2주차: TaskReferenceProvider/StateProvider thread model 명시 및 API 보강.
3. 3주차: controller interface index cache + name validation.
4. 4주차: parser error model 통일(`exit` 제거) + 테스트 보강.

## Self-feedback
- 10회 리뷰를 통해 RT 안전성, 동시성, API 유지보수성 이슈를 분리해서 식별했습니다.
- 아직 미완료인 부분은 “정량 성능 측정(주기 지터, 할당 횟수)”이며, 다음 단계에서 벤치마크를 붙여 검증해야 합니다.
