# wbc_core 코드 리뷰

날짜: 2026-06-15

## 수정 메모

- inequality/bound `MotionObjective`는 public schema를 나누지 않고 `IDHQP`
  내부에서 hard feasibility constraint로 조립하도록 수정했다.
- contact motion snapshot은 `Jc * qddot = motion_rhs` 형태로 desired
  acceleration을 보존하도록 수정했다.
- active list가 없을 때 task/contact 등록 순서를 보존하도록 registry 내부
  order vector를 추가했다.
- `addOperationalTask()`, `addTaskSpaceBias()` legacy shim은 제거하고 explicit
  `addTask(task, level, weight)`만 남겼다.
- runtime dimension validation은 적용하지 않는다. 1-2 kHz hot path에서 매 step
  전체 schema를 재검사하지 않고, assembly/task/contact 생성 책임과 debug assert,
  focused test로 계약을 지킨다.

범위: `wbc_core` 현재 변경분 중심. `control_architecture`, `wbc_ros`,
robot-specific package는 직접 리뷰 범위에서 제외하되, `wbc_core` API를
소비하는 테스트 경로는 증거로만 확인했다.

## 총평

현재 방향은 좋다. `IDProblem`/`IDSolution`을 formulation schema로 좁히고,
`IDHQP`가 command integration을 하지 않게 만든 점은 controller와
formulation 경계를 분명하게 한다. `JointTorqueLimits`와
`JointTorqueLimitConstraint`로 이름을 바꾼 것도 joint-space ID 문제라는
의미에 맞다.

다만 아직 solver contract가 완전히 닫히지 않았다. 특히 `TaskMotion`이
inequality를 낼 수 있는데 `IDHQP`는 모든 motion objective를 soft task로
넣고, inner QP는 inequality cost를 지원하지 않는다. 이 경로는 실제 solve
실패로 이어질 수 있다. 또한 `IDProblem` 입력 dimension 검증이 대부분
`assert`에 기대고 있어 release build에서 잘못된 문제 입력을 안전하게
거절하지 못한다.

## Findings

### P1. Inequality `TaskMotion`을 objective로 넣으면 solver가 실패한다

위치:

- `wbc_core/include/wbc_core/formulations/hqp/blocks/motion-constraint-block.hpp:69`
- `wbc_core/src/controller/id-hqp.cpp:289`
- `wbc_core/src/solvers/solver-HQP-eiquadprog.cpp:122`
- `wbc_core/include/wbc_core/tasks/task-capture-point-inequality.hpp:31`

`MotionConstraintBlock`은 equality뿐 아니라 inequality/bound objective도
build할 수 있게 되어 있다. 그런데 `IDHQP::assembleHierarchy()`는 모든
`problem.motion_objectives`를 `addTask()`로 넣는다. 현재 eiquadprog inner
solver는 cost level에서 inequality를 만나면 `"Inequalities in the cost
function are not implemented yet"` 경로로 실패한다.

이건 API와 solver capability가 불일치하는 문제다. `TaskCapturePointInequality`
같은 `TaskMotion` 파생 task는 registry를 통해 자연스럽게
`MotionObjective`가 될 수 있으므로, 실제 사용자가 넣는 순간 solve가 깨질
수 있다.

권장 수정:

- hard feasibility로 들어가야 하는 inequality는 `IDBase::addConstraint()`로
  들어가도록 `IDProblem` schema를 분리한다.
- soft objective로 허용할 수 있는 항은 equality/least-squares로 제한하고,
  `validateHierarchy()`나 별도 `validateProblem()`에서 inequality objective를
  명시적으로 reject한다.
- 테스트에 inequality `TaskMotion` 하나를 추가해서 현재 정책이 reject인지,
  hard constraint인지 명확히 고정한다.

### P1. `IDProblem` dimension validation이 `assert` 중심이라 release에서 안전하지 않다

위치:

- `wbc_core/src/controller/id-hqp.cpp:143`
- `wbc_core/src/controller/id-hqp.cpp:158`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/joint-torque-limit-block.hpp:45`
- `wbc_core/src/controller/id-hqp.cpp:483`

`IDHQP::solve()`는 `validateInput()` 전에 `beginCycle()`을 호출한다.
`beginCycle()`은 `problem.qddot_ref` 크기를 `assert`로만 확인하고 바로
`m_qddotRefCurrent`에 복사한다. release build에서 잘못된 크기가 들어오면
내부 vector가 잘못 resize된 뒤 이후 block build나 torque recovery에서
dimension mismatch가 발생할 수 있다.

같은 문제가 `joint_torque_limits.lower/upper`, `h_ext`, contact matrix/vector
크기에도 있다. 대부분 block 내부 `assert`만 있고, public `IDProblem` 입력에
대한 fail-fast validation이 없다.

권장 수정:

- `beginCycle()` 전에 `validateProblemDimensions(problem)`를 호출한다.
- `qddot_ref.size() == nv`, `h_ext.size() == nv`,
  `joint_torque_limits.{lower,upper}.size() == na`, contact `Jc/T/Uf` 크기를
  runtime check로 검사한다.
- 실패 시 assert가 아니라 `IDSolution.success=false`로 안전하게 반환하고,
  이전 cycle의 vector size를 오염시키지 않게 한다.

### P2. Contact motion snapshot이 desired acceleration을 버린다

위치:

- `wbc_core/include/wbc_core/controller/base/id-problem-registry.hpp:340`
- `wbc_core/include/wbc_core/controller/base/id-problem-registry.hpp:343`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/contact-acceleration-block.hpp:49`
- `wbc_core/src/tasks/task-se3-equality.cpp:188`

`TaskSE3Equality`는 contact motion constraint vector를 `a_des - drift`로
만든다. 그런데 registry는
`getDesiredAcceleration() - motion_cst.vector()`로 `Jcdot_qdot`만 복원해서
`ContactConstraintData`에 저장한다. 이후 `ContactConsistencyConstraint`는
`Jc * qddot = -Jcdot_qdot`만 강제한다.

결과적으로 contact task의 `a_des` 성분, 즉 contact reference error를 줄이기
위한 PD correction은 hard contact consistency에서 사라진다. "활성 contact는
완전히 stationary constraint로만 취급한다"가 의도라면 괜찮지만, 그 경우
`ContactBase::computeMotionConstraint()`와 `TaskSE3Equality` reference/Kp/Kd를
계산하는 경로가 혼란스럽다. 반대로 contact reference correction을 기대한다면
현재 구현은 tracking을 하지 않는다.

권장 수정:

- stationary contact가 명확한 정책이면 `ContactConstraintData` 필드를
  `Jcdot_qdot` 대신 `contact_drift`처럼 명명하고, registry에서
  `computeMotionConstraint()`의 desired term을 의도적으로 버린다는 주석을
  둔다.
- contact task acceleration을 반영해야 한다면 block RHS를
  `motion_cst.vector()` 기반으로 구성하도록 바꾼다.
- nonzero contact desired acceleration/reference error가 있는 테스트를 추가한다.

### P2. Floating-base contact dynamics와 torque-limit contact coupling 테스트가 부족하다

위치:

- `wbc_core/src/controller/id-hqp.cpp:272`
- `wbc_core/src/controller/id-hqp.cpp:487`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/joint-torque-limit-block.hpp:58`
- `wbc_core/test/test-wbc-solver-semantics.cpp:57`

현재 wbc_core solver tests는 contact force용 fixture에서 `T = Matrix::Zero(0, 1)`
형태를 주로 사용한다. 이 경우 `lambda` decision은 friction bound 테스트에는
도움이 되지만, `Jc^T * T * lambda`가 dynamics/torque recovery에 실제로
결합되는 경로는 거의 검증하지 못한다.

특히 이번 변경은 `tau_sol` recovery와 `JointTorqueLimitConstraint`가 모두
`Jc^T * T * lambda` 항을 사용한다. floating-base dynamics hard constraint와
joint torque limit inequality가 contact force와 동시에 맞물리는 테스트가
없으면 sign convention regression을 놓치기 쉽다.

권장 테스트:

- free-flyer robot + 실제 `ContactPoint` 또는 `Contact6d` snapshot으로
  `FloatingBaseDynamicsConstraint` residual을 검증한다.
- nonzero `T`, nonzero `lambda_sol` 상태에서 `tau_sol =
  M*qddot+h-Jc^T*T*lambda-h_ext`가 맞는지 검증한다.
- 같은 조건에서 joint torque bound가 contact force term을 포함해 작동하는지
  검증한다.

### P3. active list가 없을 때 task/contact 순서가 nondeterministic하다

위치:

- `wbc_core/include/wbc_core/controller/base/id-problem-registry.hpp:283`
- `wbc_core/include/wbc_core/controller/base/id-problem-registry.hpp:305`

active task/contact names가 비어 있으면 registry는 `std::unordered_map`을
순회해서 `IDProblem`을 만든다. 이 경우 objective 순서와 contact 순서가
결정적이지 않다. 내부 torque recovery는 name 기반 layout을 사용하므로
대부분 안전하지만, 외부에서 `lambda_sol`의 순서를 해석하거나 로그를 비교할
때 순서가 흔들릴 수 있다.

권장 수정:

- registry에 등록 순서 vector를 별도로 유지한다.
- 또는 active list 없이 build하는 overload는 테스트/legacy 용도로만 제한하고
  runtime path에서는 항상 state의 explicit active list를 요구한다.

### P3. `controller/base`에 legacy semantic shim이 아직 남아 있다

위치:

- `wbc_core/include/wbc_core/controller/base/id-problem-registry.hpp:49`
- `wbc_core/include/wbc_core/controller/base/id-problem-registry.hpp:54`

`addOperationalTask()`와 `addTaskSpaceBias()`는 각각 level 1, level 2로 매핑되는
legacy shim이다. 지금 방향은 "모든 문제는 HQP, level은 명시"에 가깝기 때문에
base layer에 semantic 이름을 계속 두면 controller/formulation 경계가 다시
흐려질 수 있다.

권장 수정:

- 내부 테스트도 `addTask(task, level, weight)`로 옮긴다.
- legacy shim은 필요하면 별도 compatibility header로 빼거나 제거한다.

## 긍정적인 변경

- `IDSolution`이 `qddot_ref`, `delta_qddot_sol`, `qddot_sol`, `lambda_sol`,
  `tau_sol`만 갖게 된 것은 naming 문서와 잘 맞다.
- `JointTorqueLimits`와 `JointTorqueLimitConstraint`로 정리한 것은 actuator
  driver가 아니라 model-side joint torque bound라는 의미를 분명히 한다.
- `MotionObjective`와 `ContactConstraintData`를 task/contact 쪽으로 옮긴 것은
  formulation schema가 과도하게 primitive를 소유하던 문제를 줄인다.
- `IDHierarchyPolicy` 제거 후 fixed hierarchy rule을 `IDHQP` 내부에 둔 것은
  현재 정책에서는 더 단순하다.

## 확인한 정적 체크

- `git diff --check`: 통과
- 새 untracked header no-index whitespace check: 통과
- `colcon`: 현재 Windows/WSL 환경에 command가 없어 실행하지 못함

## 다음 우선순위

1. 실제 colcon 환경에서 `wbc_core`와 `control_architecture` 테스트를 실행해 새
   HQP 조립 정책을 확인한다.
2. runtime dimension validation 대신 config/runtime assembly 단계의 dimension
   contract 테스트를 늘린다.
3. 새 task/constraint를 추가할 때 equality는 soft objective, inequality/bound는
   hard feasibility constraint라는 규칙을 유지한다.
