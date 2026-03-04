# Review Pass 03 - assert 의존성과 Release 안전성

## 범위
- `wbc_formulation`, `wbc_trajectory`, `wbc_util`

## Findings

### Critical
- `rpc_ros/wbc_core/wbc_formulation/src/contact_constraint.cpp:81`  
  `PointContact::SetParameters()`가 gain 차원 검증을 `assert`에 의존합니다. `-DNDEBUG`에서 검증이 제거됩니다.
- `rpc_ros/wbc_core/wbc_formulation/src/contact_constraint.cpp:235`  
  `SurfaceContact::SetParameters()`도 동일 패턴.

### Warning
- `rpc_ros/wbc_core/wbc_trajectory/include/wbc_trajectory/interpolation.hpp:352`  
  Bezier 계열 입력 구간 검증을 `assert(x>=0 && x<=1)`로 처리해 release에서 무방비.
- `rpc_ros/wbc_core/wbc_trajectory/src/math_util.cpp:398`  
  `ClampVector/HStack/VStack` 차원 검증이 assert-only입니다.
- `rpc_ros/wbc_core/wbc_util/src/io_util.cpp:118`  
  switch default에서 `assert(false)` 사용. release에서는 조용히 잘못된 경로를 통과할 수 있습니다.

## 개선 제안
1. 외부 입력/설정 검증은 `throw std::invalid_argument` 또는 명시 `false` 반환으로 전환.
2. math 유틸은 디버그 assert + release guard를 병행.
3. assert가 필요한 내부 불변식은 주석으로 “internal invariant only”를 명시.

## Self-feedback
- assert 제거 범위를 전부 확정하지는 않았고, 입력 경계가 외부 노출인 지점 위주로 분류했습니다.
- 다음 패스에서 concurrency와 data race 가능성을 별도로 점검합니다.
