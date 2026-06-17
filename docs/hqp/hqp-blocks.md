# HQP Blocks

이 문서는 `wbc_core`의 inverse-dynamics HQP 구현에서 "HQP block"이 무엇이고,
각 block이 어떤 matrix/vector를 만들어 `HQPData`에 들어가는지 설명한다.

`docs/task/*.md` 문서가 task/contact의 물리적 의미를 설명한다면, 이 문서는 그
결과가 solver가 먹는 선형 constraint로 바뀌는 구현 레이어를 설명한다.

대상 구현:

- `wbc_core/include/wbc_core/formulations/hqp/hqp-block-base.hpp`
- `wbc_core/include/wbc_core/formulations/hqp/hqp-build-context.hpp`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/*.hpp`
- `wbc_core/src/controller/id-hqp.cpp`
- `wbc_core/include/wbc_core/solvers/hqp-data-utils.hpp`
- `wbc_core/src/solvers/hqp-data-utils.cpp`

## HQP Block이 하는 일

HQP solver가 최종적으로 받는 것은 task 객체나 contact 객체가 아니다. Solver는 각
level마다 선형 equality, inequality, bound constraint를 담은 `HQPData`를 받는다.

```text
HQPData
  level 0: hard constraints
  level 1: high priority objectives
  level 2: lower priority objectives
  ...
```

`HQPBlock`은 runtime state와 problem snapshot을 읽어서, solver가 바로 사용할 수
있는 `ConstraintBase` 객체를 갱신하는 작은 builder다.

```cpp
class HQPBlock {
 public:
  virtual void build(const HQPBuildContext& ctx) = 0;
  const std::shared_ptr<math::ConstraintBase>& constraint() const;
};
```

각 block은 다음 책임을 가진다.

- 자신의 `ConstraintBase`를 소유한다.
- 매 tick `build(ctx)`에서 matrix, RHS, bounds를 다시 채운다.
- controller가 정한 level과 weight로 `HQPData`에 들어간다.

Block은 solver를 직접 호출하지 않는다. Block은 `HQPData`에 들어갈 재료만 만든다.

## Decision Vector

현재 `IDHQP`는 delta-form inverse dynamics를 사용한다.

```text
x = [delta_qddot; lambda]

qddot = qddot_ref + delta_qddot
```

각 기호는 다음 뜻이다.

| 기호 | 의미 |
| --- | --- |
| `delta_qddot` | reference acceleration에서 얼마나 수정할지 |
| `lambda` | stacked contact force decision variable |
| `nv` | generalized velocity dimension |
| `na` | actuated joint dimension |
| `nvFloat` | floating-base dimension, 보통 0 또는 6 |
| `lambdaDim` | 전체 contact force decision dimension |
| `qpDim()` | `nv + lambdaDim` |

모든 HQP block matrix는 column 수가 `ctx.qpDim()`이어야 한다.

```text
cols = nv + lambdaDim
```

`qddot`에 걸리는 항은 보통 left block에 들어간다.

```text
[ A_qddot  0 ] * [delta_qddot; lambda]
```

contact force에 걸리는 항은 `nv` 이후 column에 들어간다.

```text
[ A_qddot  A_lambda ] * [delta_qddot; lambda]
```

## HQPBuildContext

`HQPBuildContext`는 block들이 필요로 하는 per-cycle 입력 묶음이다. Controller가 한
tick의 robot dynamics, contact snapshot, optional limits를 모아서 채운다.

중요 필드는 다음과 같다.

| 필드 | 쓰는 block | 의미 |
| --- | --- | --- |
| `M` | dynamics, torque limit | mass matrix |
| `h` | dynamics, torque limit | nonlinear effects |
| `nv`, `na`, `nvFloat`, `lambdaDim` | all blocks | dimension |
| `contactInfos` | dynamics, torque limit | per-contact `Jc`, `T`, lambda offset |
| `Jc`, `contact_motion_rhs` | contact acceleration | stacked contact motion data |
| `Uf`, `uf_lb`, `uf_ub` | friction | stacked force inequality data |
| `h_ext` | dynamics, torque limit | external generalized force contribution |
| `tau_lb`, `tau_ub` | torque limit | joint torque bounds |
| `qddot_ref` | delta-form blocks | reference acceleration |

`IDHQP::buildHqpContext()`가 이 context를 만든다.

```text
ctx.M = mass(q)
ctx.h = nonlinearEffects(q, v)
ctx.nv = robot.nv()
ctx.na = robot.na()
ctx.lambdaDim = stacked_contact_lambda_dim
ctx.qddot_ref = problem.qddot_ref
```

Contact가 있으면 `contactInfos`에 contact별 `Jc`, `T`, `lambdaOffset`,
`lambdaDim`도 들어간다. 이 offset이 있어야 여러 contact의 `lambda` column을
올바른 위치에 넣을 수 있다.

## Block Lifecycle

`IDHQP::solve()`에서 block은 다음 순서로 사용된다.

```text
beginCycle(problem)
updateRobotModel()
prepareSolveWorkspace(problem)
buildHardConstraints(problem)
buildRegularizationBlocks(problem)
buildObjectiveBlocks(problem)
assembleHierarchy(problem)
resizeSolverFromHQPData()
solve
decodeSolution
```

각 단계의 역할은 다음과 같다.

- `prepareSolveWorkspace`: contacts를 stack하고 어떤 block이 필요한지 flag를 세운다.
- `buildHqpContext`: block들이 공유할 `HQPBuildContext`를 만든다.
- `buildHardConstraints`: dynamics/contact/friction/torque-limit block을 build한다.
- `buildRegularizationBlocks`: `delta_qddot`, `lambda` regularization block을 build한다.
- `buildObjectiveBlocks`: runtime motion objective와 joint acceleration objective를 build한다.
- `assembleHierarchy`: build된 constraint들을 `HQPData` level에 넣는다.

Block object는 `IDHQP` member로 살아 있다. 매 tick 새로 만드는 대신 내부 constraint를
resize하고 값을 덮어쓴다.

## Level 배치

현재 hierarchy 정책은 `IDHQP::assembleHierarchy()`가 소유한다.

### Level 0

Level 0은 hard feasibility 영역이다.

```text
floating-base dynamics
contact acceleration consistency
friction cone
joint torque limits
motion inequalities / bounds
```

Level 0 constraint는 task weight가 아니라 feasibility로 취급된다. Motion objective가
inequality나 bound라면 `MotionConstraintBlock`으로 build한 뒤 level 0에 들어간다.

### Positive Objective Levels

Equality motion objective와 joint acceleration objective는 양수 level에 들어간다.

```text
level >= 1
```

`IDHQP::validateHierarchy()`는 equality objective level이 0이면 invalid로 본다.

### Regularization Level

Regularization은 가장 깊은 user objective 다음 level에 들어간다.

```text
regLevel = problem.maxObjectiveLevel() + 1
```

현재 regularization block은 다음 두 가지다.

```text
delta_qddot -> 0
lambda -> 0
```

`lambda` regularization은 contact force가 있고 `w_lambda > 0`일 때만 들어간다.

## FloatingBaseDynamicsConstraint

Floating-base robot은 base DOF에 직접 actuator torque를 줄 수 없다. 그래서
floating-base dynamics는 hard equality로 들어간다.

기본 식은 다음과 같다.

```text
Sf * (M * qddot - Jc^T * T * lambda) = -Sf * h
```

delta-form에서는 다음 식으로 바뀐다.

```text
Sf * (M * delta_qddot - Jc^T * T * lambda)
  = -Sf * (M * qddot_ref + h)
```

Block matrix는 다음 형태다.

```text
rows = nvFloat
cols = nv + lambdaDim

A.leftCols(nv) = M.topRows(nvFloat)
A.lambda_cols  = -Jc^T.topRows(nvFloat) * T
b              = -M.topRows(nvFloat) * qddot_ref - h.head(nvFloat)
```

`h_ext`가 있으면 RHS에 더한다.

```text
b += h_ext.head(nvFloat)
```

`nvFloat == 0`이면 block은 `0 x qpDim` equality로 resize하고 빠진다.

## ContactConsistencyConstraint

Active contact는 contact frame이나 contact point가 원하는 acceleration 관계를
만족하도록 hard equality를 만든다.

```text
Jc * qddot = contact_motion_rhs
```

delta-form에서는 다음과 같다.

```text
Jc * delta_qddot = contact_motion_rhs - Jc * qddot_ref
```

Block matrix는 다음 형태다.

```text
rows = Jc.rows()
cols = nv + lambdaDim

A.leftCols(nv) = Jc
A.lambda_cols  = 0
b              = contact_motion_rhs - Jc * qddot_ref
```

Contact force `lambda`는 contact acceleration equation에 직접 들어가지 않는다.
`lambda`는 dynamics와 torque-limit block에서 coupling된다.

## FrictionConeConstraint

Friction block은 contact force decision variable만 제한한다.

```text
uf_lb <= Uf * lambda <= uf_ub
```

전체 decision vector 기준 matrix는 다음 형태다.

```text
rows = Uf.rows()
cols = nv + lambdaDim

A.leftCols(nv) = 0
A.block(0, nv, rows, lambdaDim) = Uf
lb = uf_lb
ub = uf_ub
```

이 block은 `delta_qddot` column을 건드리지 않는다. Contact force가 friction cone과
normal force bound 안에 있는지만 본다.

## JointTorqueLimitConstraint

입력 primitive는 단순하다.

```text
tau_lb <= tau <= tau_ub
```

하지만 HQP block에서는 torque를 decision variable `x = [delta_qddot; lambda]`에
대한 inequality로 바꿔야 한다.

현재 부호 convention은 다음과 같다.

```text
M * qddot + h = S^T * tau + Jc^T * T * lambda + h_ext
```

actuated joint rows만 보면:

```text
tau = M_joint * qddot + h_joint - Jc_joint^T * T * lambda - h_ext_joint
```

delta-form을 대입하면:

```text
tau =
  M_joint * delta_qddot
  - Jc_joint^T * T * lambda
  + h_joint
  + M_joint * qddot_ref
  - h_ext_joint
```

constant 항을 bound 쪽으로 옮기면 block은 다음 matrix/bounds를 만든다.

```text
rows = na
cols = nv + lambdaDim

A.leftCols(nv) = M.bottomRows(na)
A.lambda_cols  = -Jc^T.bottomRows(na) * T

c  = h.tail(na) + M.bottomRows(na) * qddot_ref - h_ext.tail(na)
lb = tau_lb - c
ub = tau_ub - c
```

Contact가 없으면 `A.lambda_cols`가 비어 있어서 훨씬 단순하다.

```text
tau_lb - c <= M_joint * delta_qddot <= tau_ub - c
```

Contact가 있으면 per-contact `lambdaOffset`을 사용해서 각 contact의 column block을
채운다.

```text
A.block(0, nv + lambdaOffset, na, contactLambdaDim)
  = -Jc_contact^T.bottomRows(na) * T_contact
```

이 block에서 가장 조심할 부분은 세 가지다.

- contact force 부호
- `qddot_ref`를 bound constant로 옮기는 것
- 여러 contact의 `lambda` column offset

## MotionConstraintBlock

`MotionConstraintBlock`은 `MotionObjective`를 solver constraint로 바꾼다.

Equality objective:

```text
J * qddot = a_des
```

delta-form:

```text
J * delta_qddot = a_des - J * qddot_ref
```

Matrix form:

```text
A.leftCols(nv) = J
A.lambda_cols  = 0
b              = a_des - J * qddot_ref
```

Inequality 또는 bound objective:

```text
lower <= J * qddot <= upper
```

delta-form:

```text
lower - J*qddot_ref <= J * delta_qddot <= upper - J*qddot_ref
```

`MotionConstraintBlock`은 objective type에 따라 내부 constraint object를
`ConstraintEquality` 또는 `ConstraintInequality`로 바꿀 수 있다.

```text
ensureEquality()
ensureInequality()
```

이 점 때문에 block consumer는 build 이후 `block.constraint()`를 다시 읽어야 한다.

## JointAccelerationBias

Joint acceleration objective는 full generalized acceleration target을 선호하게 만든다.

```text
qddot = qddot_bias
```

delta-form에서는:

```text
delta_qddot = qddot_bias - qddot_ref
```

Matrix form:

```text
A.leftCols(nv) = I
A.lambda_cols  = 0
b              = qddot_bias - qddot_ref
```

`qddot_bias`가 null이면 RHS는 zero다.

## AccelerationRegularization

Acceleration regularization은 solved correction을 작게 유지한다.

```text
delta_qddot -> 0
```

Matrix form:

```text
A.leftCols(nv) = I
A.lambda_cols  = 0
b              = 0
```

주석에는 `||qddot|| -> 0`도 적혀 있지만, 현재 `IDHQP` context는 항상
`qddot_ref`를 제공하므로 실제 solve variable 기준으로는 `delta_qddot` regularization에
가깝다.

## ContactForceRegularization

Contact force regularization은 contact force decision variable을 작게 유지한다.

```text
lambda -> 0
```

Matrix form:

```text
A.leftCols(nv) = 0
A.block(0, nv, lambdaDim, lambdaDim) = I
b = 0
```

`lambdaDim == 0`이면 `0 x qpDim` equality로 resize한다. `IDHQP`는 contact force가
있고 `w_lambda > 0`일 때만 이 block을 hierarchy에 추가한다.

## HQPData Assembly

Block이 만든 constraint는 `solvers::hqp::addTerm()`으로 `HQPData`에 들어간다.

```cpp
addTerm(hqp_data, level, weight, block.constraint());
```

`HQPData`의 각 element는 다음 형태다.

```text
ConstraintLevel = vector<pair<weight, ConstraintBasePtr>>
HQPData         = vector<ConstraintLevel>
```

`solvers::hqp::dimensions()`는 모든 level을 훑으며 solver resize에 필요한 값을 센다.

```text
variables    = first non-null constraint cols()
equalities   = sum(rows of equality constraints)
inequalities = sum(rows of inequality or bound constraints)
```

이 값이 바뀌면 `IDHQP`는 solver를 다시 만들고 resize한다.

## Empty Blocks

여러 block은 비활성 조건에서 `0 x qpDim` constraint로 resize한다.

예:

- fixed-base robot의 floating-base dynamics block
- contact가 없는 contact consistency block
- `lambdaDim == 0`인 friction 또는 lambda regularization block
- torque limits가 disabled인 torque-limit block

다만 `IDHQP::assembleHierarchy()`는 대부분 workspace flag를 보고 필요한 block만
추가한다. Empty block resize는 block 자체의 방어적 동작이자 shape 일관성을 위한
구현이다.

## Block을 추가할 때 체크할 것

새 HQP block을 추가할 때는 다음을 먼저 정해야 한다.

1. 이 block은 hard feasibility인가, weighted objective인가?
2. Equality인가 inequality인가?
3. Matrix column 수가 `nv + lambdaDim`인가?
4. `qddot_ref`가 있을 때 RHS 또는 bounds를 어떻게 shift해야 하는가?
5. Contact force `lambda`를 쓰면 per-contact offset을 올바르게 반영하는가?
6. `h_ext`와 contact force 부호 convention을 기존 dynamics/torque recovery와 맞췄는가?
7. 비활성 조건에서 block을 skip할지, `0 x qpDim`으로 만들지 정했는가?
8. Solver resize에 필요한 equality/inequality row 수가 deterministic한가?

특히 delta-form block에서는 원래 식을 먼저 `qddot = qddot_ref + delta_qddot`로
펼친 뒤, `qddot_ref`가 곱해진 항을 RHS나 bounds로 옮기는 순서가 가장 안전하다.

## 현재 테스트가 고정하는 부분

`wbc_core/test/test_math_validation.cpp`는 model 없이 다음 block 수학을 직접 검증한다.

- motion equality delta-form RHS
- motion inequality bound shift
- contact acceleration delta-form RHS
- friction cone matrix placement
- floating-base dynamics contact/delta signs
- joint torque limit contact/delta signs

이 테스트들은 robot URDF fixture 없이 block-level 수식을 고정한다. HQP block 수식을
바꿀 때는 먼저 이 테스트가 어떤 sign과 matrix placement를 기대하는지 확인하는 것이
좋다.
