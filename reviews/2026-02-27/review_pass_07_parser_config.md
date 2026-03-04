# Review Pass 07 - YAML 파싱/컴파일러 견고성

## 범위
- `wbc_util/yaml_util`, `wbc_util/yaml_parser`, `wbc_runtime_config`

## Findings

### Critical
- `rpc_ros/wbc_core/wbc_util/include/wbc_util/yaml_util.hpp:26`  
  필수 파라미터 누락 시 `std::exit(EXIT_FAILURE)`를 호출합니다. 라이브러리 코드에서 프로세스 강제 종료는 상위 복구 전략을 차단합니다.
- `rpc_ros/wbc_core/wbc_util/include/wbc_util/yaml_util.hpp:53`  
  벡터 파라미터도 동일하게 `std::exit` 사용.

### Warning
- `rpc_ros/wbc_core/wbc_architecture/src/wbc_runtime_config.cpp:281`  
  `ParseTaskPool()`가 파싱/검증/객체생성을 한 함수에서 처리해 분기 복잡도가 큽니다.
- `rpc_ros/wbc_core/wbc_architecture/src/wbc_runtime_config.cpp:206`  
  `ParseConstraintPool()`과 `ParseGlobalConstraints()`가 유사한 validation 흐름을 반복합니다.

### Suggestion
- `rpc_ros/wbc_core/wbc_architecture/src/wbc_runtime_config.cpp:386`  
  state-machine transition graph 검증은 잘 되어 있으나, cycle/terminal state 정책까지 명시하면 운용 중 디버깅이 쉬워집니다.

## 개선 제안
1. `yaml_util`의 `std::exit`를 예외 또는 `expected<T, Error>` 패턴으로 교체.
2. 파싱 함수를 `validate -> decode -> instantiate` 단계로 분리.
3. YAML schema 문서와 파서 에러 메시지를 1:1 매핑.

## Self-feedback
- parser 품질은 이미 개선된 편이지만, error-handling 정책은 아직 라이브러리 관점에서 강경합니다.
- 다음 패스에서 solver 수치 안정성과 실패 복구 동작을 집중 분석합니다.
