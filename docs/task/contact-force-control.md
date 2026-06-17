# Contact and Force Control

이 문서는 `wbc_core`에서 contact와 contact force가 inverse-dynamics HQP 문제로
들어가는 흐름을 설명한다.

`TaskSE3Equality`처럼 하나의 task가 error를 만들고 constraint를 내보내는 경우와
달리, contact control은 여러 조각이 함께 움직인다. active contact는 motion
constraint, force decision variable, friction constraint, dynamics coupling을
동시에 만든다. 그 위에 force tracking 또는 CoP 같은 force task가 추가될 수 있다.

대상 구현:

- `wbc_core/include/wbc_core/contacts/contact-base.hpp`
- `wbc_core/include/wbc_core/contacts/contact-constraint-data.hpp`
- `wbc_core/include/wbc_core/contacts/contact-level.hpp`
- `wbc_core/include/wbc_core/contacts/contact-point.hpp`
- `wbc_core/include/wbc_core/contacts/contact-6d.hpp`
- `wbc_core/include/wbc_core/tasks/task-contact-force.hpp`
- `wbc_core/include/wbc_core/tasks/task-contact-force-equality.hpp`
- `wbc_core/include/wbc_core/tasks/task-cop-equality.hpp`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/contact-acceleration-block.hpp`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/friction-cone-block.hpp`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/floating-base-dynamics-block.hpp`
- `wbc_core/include/wbc_core/formulations/hqp/blocks/joint-torque-limit-block.hpp`
- `wbc_core/src/controller/id-hqp.cpp`

## 큰 그림

contact가 없을 때 inverse dynamics solver는 주로 joint acceleration `qddot`을
찾는다. contact가 active가 되면 solver decision vector에 contact force variable
`lambda`가 추가된다.

```text
x = [qddot; lambda]
```

현재 `IDHQP`는 reference acceleration을 중심으로 delta-form을 사용한다. 그래서
실제 solver variable은 다음처럼 보는 것이 더 정확하다.

```text
x = [delta_qddot; lambda]

qddot = qddot_ref + delta_qddot
```

contact control의 핵심은 다음 네 가지다.

1. contact point나 contact frame이 움직이지 않도록 acceleration constraint를 만든다.
2. contact force를 `lambda`라는 decision variable로 둔다.
3. friction cone과 normal force limit으로 `lambda`의 feasible range를 제한한다.
4. dynamics와 torque recovery에서 `Jc^T * T * lambda`가 들어간다.

즉 contact는 "힘을 직접 계산해서 더하는 기능"이 아니다. Contact는 solver가
`qddot`과 `lambda`를 함께 찾을 수 있도록 문제 구조를 확장한다.

## 주요 기호

| 기호 | 의미 | 코드에서 주로 대응되는 것 |
| --- | --- | --- |
| `qddot` | generalized acceleration | `IDSolution::qddot_sol` |
| `delta_qddot` | reference acceleration 기준 correction | `IDSolution::delta_qddot_sol` |
| `lambda` | contact force decision variable | `IDSolution::lambda_sol` |
| `Jc` | stacked contact Jacobian | `ContactConstraintData::Jc` |
| `motion_rhs` | contact acceleration RHS | `ContactConstraintData::motion_rhs` |
| `T` | `lambda`를 physical force/wrench로 바꾸는 matrix | `ContactConstraintData::T` |
| `Uf` | friction/force inequality matrix | `ContactConstraintData::Uf` |
| `uf_lb`, `uf_ub` | friction/force inequality bounds | `ContactConstraintData::uf_lb`, `uf_ub` |
| `M` | mass matrix | `RobotSystem::mass(data)` |
| `h` | nonlinear effects | `RobotSystem::nonLinearEffects(data)` |
| `h_ext` | external generalized wrench contribution | `IDProblem::h_ext` |

## Contact는 Task 하나가 아니다

`TaskSE3Equality`는 motion task다. 입력 pose와 현재 pose를 비교해서 다음 형태의
constraint를 만든다.

```text
J_task * qddot = a_des - drift
```

반면 contact는 적어도 다음 세 종류의 solver-facing 정보를 만든다.

```text
Jc * qddot = motion_rhs
uf_lb <= Uf * lambda <= uf_ub
f_contact = T * lambda
```

이 세 식은 서로 다른 역할을 한다.

- `Jc * qddot = motion_rhs`: contact motion consistency
- `uf_lb <= Uf * lambda <= uf_ub`: friction cone, normal force, force bounds
- `f_contact = T * lambda`: dynamics와 force task에서 쓸 physical force/wrench 매핑

그래서 contact 관련 코드를 읽을 때는 "어떤 task가 force를 만든다"보다 "active
contact가 solver의 decision space와 hard constraints를 확장한다"로 생각하는 편이
좋다.

## ContactBase

`ContactBase`는 contact primitive가 solver에 제공해야 하는 공통 interface다.

중요한 함수는 다음과 같다.

```cpp
virtual unsigned int n_motion() const = 0;
virtual unsigned int n_force() const = 0;

virtual const ConstraintBase& computeMotionConstraint(...) = 0;
virtual const ConstraintInequality& computeForceTask(...) = 0;
virtual const Matrix& getForceGeneratorMatrix() const = 0;
virtual const ConstraintEquality& computeForceRegularizationTask(...) = 0;
```

각 함수의 의미는 다음과 같다.

- `n_motion()`: contact motion constraint row 수
- `n_force()`: 이 contact가 추가하는 `lambda` 차원
- `computeMotionConstraint()`: `Jc * qddot = motion_rhs`에 들어갈 `Jc`와 RHS 생성
- `computeForceTask()`: friction/normal force inequality 생성
- `getForceGeneratorMatrix()`: `T` 생성
- `computeForceRegularizationTask()`: contact-local force reference/regularization 생성

현재 `IDProblemRegistry::snapshotContact()`는 contact에서 다음 값을 뽑아
`ContactConstraintData`로 옮긴다.

```cpp
out.Jc = motion_cst.matrix();
out.motion_rhs = motion_cst.vector();
out.T = contact.getForceGeneratorMatrix();
out.Uf = force_cst.matrix();
out.uf_lb = force_cst.lowerBound();
out.uf_ub = force_cst.upperBound();
```

주의할 점이 하나 있다. `computeForceRegularizationTask()`는 현재 snapshot 단계에서
호출되지만, `ContactConstraintData`에는 per-contact force regularization constraint가
저장되지 않는다. 현재 `IDHQP` solve path는 global `lambda` regularization block을
사용한다.

```text
lambda -> 0
```

따라서 contact-local force reference를 solver objective로 넣고 싶다면, 현재
`IDProblem` schema에 그 objective가 어떻게 들어갈지 먼저 명확히 해야 한다.

## Contact 종류

### ContactPoint

`ContactPoint`는 3D point contact다.

```text
n_force = 3
T = I_3
lambda = f_point
```

motion 쪽은 underlying `TaskSE3Equality`를 사용하지만 mask를 position-only로 둔다.

```text
mask = [1, 1, 1, 0, 0, 0]
```

즉 point contact는 contact point의 linear acceleration을 제약하고, orientation은
제약하지 않는다.

force inequality는 접선 방향 두 개와 normal direction으로 friction pyramid와 normal
force bound를 만든다.

```text
lb <= B * f_point <= ub
```

여기서 마지막 row는 normal force bound다.

```text
f_min <= n^T * f_point <= f_max
```

### Contact6d

`Contact6d`는 보통 foot contact처럼 finite support polygon을 갖는 6D contact를
표현한다. 구현상 4개의 contact point force를 사용한다.

```text
n_force = 12
lambda = [f_0; f_1; f_2; f_3]
```

각 `f_i`는 3D force다. `T`는 네 point force를 6D wrench로 모은다.

```text
f_linear = f_0 + f_1 + f_2 + f_3
tau      = p_0 x f_0 + p_1 x f_1 + p_2 x f_2 + p_3 x f_3

wrench = T * lambda
```

코드에서는 `Contact6d::updateForceGeneratorMatrix()`가 이 `T`를 만든다.

```cpp
m_forceGenMat.block<3, 3>(0, i * 3).setIdentity();
m_forceGenMat.block<3, 3>(3, i * 3) = pinocchio::skew(contact_point_i);
```

motion 쪽은 full 6D `TaskSE3Equality`를 사용한다.

### ContactTwoFramePositions

`ContactTwoFramePositions`는 두 frame 사이를 position-only로 묶는 contact다. 구현
주석상 ball joint를 흉내내는 용도에 가깝다.

```text
n_force = 3
T = I_3
```

rotation은 unconstrained로 두고 linear part만 사용한다.

## ContactConstraintData

`ContactConstraintData`는 per-tick contact snapshot이다. `ContactBase` 객체 자체를
solver에 넘기지 않고, 그 tick에 필요한 행렬과 벡터만 복사해서 넘긴다.

```cpp
struct ContactConstraintData {
  std::string name;
  Matrix Jc;
  Vector motion_rhs;
  Matrix T;
  Matrix Uf;
  Vector uf_lb;
  Vector uf_ub;
};
```

이 구조는 `IDProblem` 안에 들어간다.

```cpp
std::vector<ContactConstraintData> contacts;
```

`IDHQP`는 이 vector를 받아서 contact들을 stack한다.

```text
Jc_stack =
[ Jc_0 ]
[ Jc_1 ]
[ ...  ]

lambda =
[ lambda_0 ]
[ lambda_1 ]
[ ...      ]
```

각 contact의 `lambda`가 전체 `lambda`에서 어디에 들어가는지는
`contactLayout[name].lambdaOffset`과 `lambdaDim`으로 추적한다.

## Contact Motion Constraint

active contact는 hard equality constraint를 만든다.

```text
Jc * qddot = motion_rhs
```

delta-form에서는 solver variable이 `delta_qddot`이므로 RHS가 바뀐다.

```text
Jc * (qddot_ref + delta_qddot) = motion_rhs

Jc * delta_qddot = motion_rhs - Jc * qddot_ref
```

`ContactConsistencyConstraint`가 이 식을 만든다.

```text
[ Jc  0 ] * [delta_qddot; lambda] = motion_rhs - Jc * qddot_ref
```

contact acceleration constraint는 level 0 hard constraint로 들어간다. 즉 solver는
가능하면 이 식을 반드시 만족해야 한다.

## Friction Cone Constraint

contact force는 아무 방향이나 아무 크기로 나올 수 없다. 접촉면에서는 보통
normal force와 tangential force 사이에 friction cone 관계가 있다.

간단히 말하면 contact force `f`를 접촉면 normal 방향 성분과 접선 방향 성분으로
나눴을 때, 접선 방향 힘이 너무 크면 발이나 손끝이 미끄러진다.

```text
f_n = n^T * f
f_t = f - n * f_n

||f_t|| <= mu * f_n
```

- `n`: contact normal
- `f_n`: normal force
- `f_t`: tangential force
- `mu`: friction coefficient

3D force space에서 이 feasible set을 그리면 normal 방향으로 열린 cone처럼 생긴다.
그래서 friction cone이라고 부른다.

문제는 일반적인 cone 식 `||f_t|| <= mu * f_n`이 선형 inequality가 아니라는 점이다.
현재 HQP/QP 조립은 선형 constraint를 기대하므로, `wbc_core`는 cone을 네 면짜리
friction pyramid로 근사한다.

```text
        true friction cone
             /\
            /  \

        linearized pyramid
            /\
           /__\
```

수식으로는 접촉면 위의 서로 직교하는 두 tangent direction `t1`, `t2`를 만들고,
각 tangent 방향의 양/음 방향 힘을 제한한다.

```text
 t1^T * f <= mu * n^T * f
-t1^T * f <= mu * n^T * f
 t2^T * f <= mu * n^T * f
-t2^T * f <= mu * n^T * f
```

이 네 줄은 모두 다음과 같은 선형 inequality로 바꿀 수 있다.

```text
( t1 - mu * n)^T * f <= 0
(-t1 - mu * n)^T * f <= 0
( t2 - mu * n)^T * f <= 0
(-t2 - mu * n)^T * f <= 0
```

normal force 자체도 너무 작거나 커지지 않도록 따로 제한한다.

```text
f_min <= n^T * f <= f_max
```

`wbc_core`에서는 이 관계를 다음 선형 inequality로 저장한다.

```text
uf_lb <= Uf * lambda <= uf_ub
```

`FrictionConeConstraint`는 이 식을 전체 decision vector에 맞게 배치한다.

```text
[ 0  Uf ] * [delta_qddot; lambda] between [uf_lb, uf_ub]
```

`ContactPoint`와 `Contact6d`는 contact normal과 friction coefficient로 `Uf`,
`uf_lb`, `uf_ub`를 만든다. `Contact6d`는 4개의 point force 각각에 friction pyramid
제약을 적용하고, 전체 normal force bound도 같이 넣는다.

### ContactPoint 구현

`ContactPoint::updateForceInequalityConstraints()`는 3D point force `f`에 대한
5-row inequality를 만든다.

```text
n_in  = 4 * 1 + 1
n_var = 3 * 1
B     = zeros(5, 3)
lb    = [-inf, -inf, -inf, -inf, f_min]
ub    = [0,     0,     0,     0,     f_max]
```

먼저 normal에 수직인 tangent basis를 만든다.

```cpp
t1 = m_contactNormal.cross(Vector3::UnitX());
if (t1.norm() < 1e-5) {
  t1 = m_contactNormal.cross(Vector3::UnitY());
}
t2 = m_contactNormal.cross(t1);
t1.normalize();
t2.normalize();
```

`contactNormal`이 `UnitX`와 거의 평행이면 cross product가 너무 작아지므로, fallback
axis로 `UnitY`를 쓴다. 그 다음 네 개의 friction pyramid row를 채운다.

```cpp
B.block<1, 3>(0, 0) = (-t1 - m_mu * m_contactNormal).transpose();
B.block<1, 3>(1, 0) = ( t1 - m_mu * m_contactNormal).transpose();
B.block<1, 3>(2, 0) = (-t2 - m_mu * m_contactNormal).transpose();
B.block<1, 3>(3, 0) = ( t2 - m_mu * m_contactNormal).transpose();
```

첫 네 row는 upper bound가 0이고 lower bound가 매우 작은 값이다. 즉 사실상 다음만
강제한다.

```text
B.row(i) * f <= 0
```

마지막 row는 normal force bound다.

```cpp
B.block<1, 3>(4, 0) = m_contactNormal.transpose();
ub(4) = m_fMax;
lb(4) = m_fMin;
```

그래서 최종 constraint는 다음과 같다.

```text
lb <= B * f <= ub
```

`ContactPoint`에서는 `T = I_3`이므로 `lambda`와 3D point force `f`를 거의 같은
값으로 봐도 된다.

### Contact6d 구현

`Contact6d`는 4개의 point force를 묶어서 6D wrench를 만든다.

```text
lambda = [f_0; f_1; f_2; f_3]
```

따라서 friction pyramid row도 각 point force마다 반복된다.

```cpp
for (int i = 1; i < 4; i++) {
  B.block<4, 3>(4 * i, 3 * i) = B.topLeftCorner<4, 3>();
}
```

마지막 row는 네 point force의 normal component 합을 제한한다.

```cpp
B.block<1, 3>(n_in - 1, 0) = m_contactNormal.transpose();
B.block<1, 3>(n_in - 1, 3) = m_contactNormal.transpose();
B.block<1, 3>(n_in - 1, 6) = m_contactNormal.transpose();
B.block<1, 3>(n_in - 1, 9) = m_contactNormal.transpose();

ub(n_in - 1) = m_fMax;
lb(n_in - 1) = m_fMin;
```

즉 `Contact6d`의 force inequality는 다음을 동시에 말한다.

- 각 contact point force는 friction pyramid 안에 있어야 한다.
- 네 point force의 normal force 합은 `[f_min, f_max]` 안에 있어야 한다.

## Dynamics Coupling

floating-base inverse dynamics에서 contact force는 dynamics equality에 직접
들어간다.

```text
Sf * (M * qddot - Jc^T * T * lambda) = -Sf * h
```

여기서 `Sf`는 floating-base row만 고르는 selection이라고 보면 된다. floating-base
DOF는 직접 actuator torque로 제어할 수 없기 때문에, base dynamics는 hard equality로
맞춰야 한다.

delta-form에서는 다음 식이 된다.

```text
Sf * (M * delta_qddot - Jc^T * T * lambda)
  = -Sf * (M * qddot_ref + h)
```

external generalized wrench `h_ext`가 있으면 RHS에 더해진다.

```text
RHS = -Sf * (M * qddot_ref + h) + Sf * h_ext
```

`FloatingBaseDynamicsConstraint`가 이 block을 만든다.

## Joint Torque Limit과 Contact Force

joint torque limit도 contact force와 coupling된다. 현재 구현의 주석 기준 식은
다음과 같다.

```text
tau = M_joint * qddot + h_joint - Jc_joint^T * T * lambda - h_ext_joint

tau_lb <= tau <= tau_ub
```

delta-form에서는 constant 항에 `M_joint * qddot_ref`가 포함된다.

```text
M_joint * delta_qddot - Jc_joint^T * T * lambda
  between [tau_lb - c, tau_ub - c]

c = h_joint + M_joint * qddot_ref - h_ext_joint
```

`JointTorqueLimitConstraint`가 이 inequality를 만든다. 이 block을 읽을 때 중요한
점은 torque bound가 단순히 `qddot`만 제한하는 것이 아니라 contact force decision
variable `lambda`도 같이 제한한다는 것이다.

## Force Regularization

contact force는 hard constraints만으로는 하나로 정해지지 않을 수 있다. 예를 들어
여러 contact가 같은 motion constraint를 만족할 수 있으면 force 분배가 여러 개일 수
있다.

현재 `IDHQP`는 regularization level에 다음 objective를 추가할 수 있다.

```text
lambda -> 0
```

코드에서는 `ContactForceRegularization` block이 다음 식을 만든다.

```text
[ 0  I ] * [delta_qddot; lambda] = 0
```

이 objective는 보통 가장 낮은 우선순위의 regularization level에 들어간다.

주의할 점:

- 이 regularization은 physical force tracking이 아니다.
- 목적은 feasible solution 중 작은 `lambda`를 선호하게 만드는 것이다.
- contact-specific `m_forceRegTask`와 현재 `IDHQP`의 global lambda regularization은
  같은 것이 아니다.

## TaskContactForceEquality

`TaskContactForceEquality`는 reference force와 measured external force를 비교해서
force tracking equality를 만드는 task다.

현재 구현은 6D force/wrench reference를 기준으로 한다.

```text
force_error = f_ref - f_ext

f_cmd =
  f_ref
  + Kp * force_error
  + Kd * (f_ref_dot - f_ext_dot)
  + Ki * integral_error
```

그리고 associated contact의 force generator matrix를 constraint matrix로 사용한다.

```text
T * lambda = f_cmd
```

구현상 `TaskContactForceEquality`는 active contact list를 받아서 associated contact가
현재 active인지 확인한다. associated contact가 active가 아니면 constraint를 새로
계산하지 않고 현재 `m_constraint`를 그대로 반환한다. 따라서 inactive contact의 force
task는 상위 assembly에서 objective로 넣지 않는 것이 안전하다.

주의할 점:

- 현재 constructor는 `m_constraint(name, 6, 12)`로 시작한다.
- 따라서 현재 형태는 `Contact6d`처럼 `T`가 6x12인 contact에 가장 자연스럽다.
- `ContactPoint`처럼 3D force contact에 같은 task를 바로 쓰려면 dimension contract를
  다시 확인해야 한다.
- 현재 `IDProblemRegistry`의 public registration path는 `TaskMotion` 중심이다.
  force task를 final `IDProblem` objective로 넣는 별도 lane은 아직 명확하지 않다.

그래서 지금 코드를 기준으로는 `TaskContactForceEquality`를 "개념과 legacy-compatible
type은 존재하지만, final IDHQP assembly와의 연결은 별도 설계가 필요한 영역"으로
읽는 것이 안전하다.

## TaskCopEquality

`TaskCopEquality`는 contact force들을 이용해서 center of pressure 조건을 만드는
force task다. 특정 contact 하나에 붙기보다 active contact set 전체를 본다.

이 task는 active contact list를 받아 전체 force vector 차원을 센다.

```text
n = sum(contact.n_force())
```

그리고 각 contact point 위치와 ground normal, reference CoP를 사용해서 matrix
column을 채운다.

```text
M.middleCols(contact_lambda_offset + 3 * point_index, 3)
  = (p_world - cop_ref) * normal^T
```

직관적으로는 "contact force들이 만드는 moment가 원하는 CoP 기준으로 맞도록" 하는
constraint다.

주의할 점:

- `TaskCopEquality`는 active contacts가 필요하다.
- contact point 위치는 현재 robot frame pose를 통해 world frame으로 변환된다.
- `m_contact_name`은 empty string이며, 여러 contact를 대상으로 한다.
- 이 task도 현재 final `IDProblemRegistry` force-objective lane과의 연결 상태를
  확인해야 한다.

## IDHQP에서 들어가는 순서

현재 `IDHQP::solve()`의 흐름은 크게 다음과 같다.

```text
beginCycle(problem)
updateRobotModel()
prepareSolveWorkspace(problem)
buildHardConstraints(problem)
buildRegularizationBlocks(problem)
buildObjectiveBlocks(problem)
assembleHierarchy(problem)
solve HQP
decodeSolution(problem, hqpSol)
computeModelTorque(problem)
```

contact 관련 block은 다음 우선순위로 들어간다.

### Level 0 hard constraints

```text
floating-base dynamics
contact acceleration consistency
friction cone
joint torque limits
motion inequalities / bounds
```

### Positive objective levels

```text
motion equality objectives
joint acceleration objectives
```

### Regularization level

```text
delta_qddot -> 0
lambda -> 0
```

즉 contact feasibility는 낮은 priority objective가 아니라 hard feasibility에 가깝다.
반대로 force preference나 force tracking은 objective로 들어가야 한다.

## Torque Recovery

solver가 끝나면 `IDHQP`는 다음을 decode한다.

```text
delta_qddot_sol = x.head(nv)
lambda_sol      = x.tail(lambdaDim)
qddot_sol       = qddot_ref + delta_qddot_sol
```

그 다음 model torque를 복구한다.

```text
tau_full = M * qddot_sol + h
tau_full -= h_ext
tau_full -= Jc^T * T * lambda_sol

tau_sol = actuated_tail(tau_full)
```

여기서 `tau_sol`은 final hardware command가 아니다. `tau_sol`은 inverse dynamics에서
나온 model torque이며, command builder 쪽에서 feedforward torque로 쓰인다.

## Contact와 Force Task를 구분하는 법

다음 질문을 던지면 코드를 읽을 때 덜 헷갈린다.

1. 이 코드는 contact가 active일 때 반드시 만족해야 하는 조건인가?
   - 그렇다면 보통 `ContactConstraintData`, contact acceleration, friction, dynamics,
     torque-limit 쪽이다.

2. 이 코드는 어떤 force를 선호하거나 추종하게 만드는가?
   - 그렇다면 force regularization, force tracking, CoP objective 쪽이다.

3. 이 코드는 physical force/wrench를 다루는가, solver variable `lambda`를 다루는가?
   - physical force/wrench라면 `T * lambda` 매핑을 확인해야 한다.

4. 이 contact는 3D point force인가, 6D wrench-producing contact인가?
   - `n_force()`와 `getForceGeneratorMatrix()` shape를 먼저 확인해야 한다.

## 흔한 오해

### lambda는 항상 physical force가 아니다

`ContactPoint`에서는 `T = I`라서 `lambda`를 3D force로 봐도 거의 맞다. 하지만
`Contact6d`에서는 `lambda`가 네 contact point force를 이어 붙인 12D vector이고,
physical 6D wrench는 `T * lambda`다.

### Contact motion constraint와 force tracking은 다르다

contact motion constraint는 접촉이 미끄러지거나 떨어지지 않도록 acceleration
관계를 강제한다.

```text
Jc * qddot = motion_rhs
```

force tracking은 특정 force나 wrench를 선호하게 만든다.

```text
T * lambda = f_cmd
```

둘은 같은 contact에서 나올 수 있지만 같은 constraint가 아니다.

### Contact force는 dynamics에서 부호가 중요하다

현재 구현 기준 dynamics와 torque recovery는 contact term을 다음 방향으로 사용한다.

```text
M * qddot + h = S^T * tau + Jc^T * T * lambda + h_ext

tau = M_joint * qddot + h_joint - Jc_joint^T * T * lambda - h_ext_joint
```

부호를 바꾸면 contact force가 robot을 밀어주는 방향과 robot torque가 보상해야 하는
방향이 뒤집힌다. contact 관련 테스트는 이 부호를 직접 고정해야 한다.

### Normal force bound와 friction cone은 frame convention에 민감하다

`ContactPoint`와 `Contact6d`는 `contactNormal`로 friction inequality를 만든다.
normal 방향이 뒤집히면 `f_min`, `f_max`의 의미도 같이 뒤집힌다. config나 robot
description에서 normal direction을 바꿀 때는 friction inequality 결과를 함께
확인해야 한다.

## 현재 구현 상태와 문서화할 다음 지점

현재 final IDHQP path에서 확실히 연결되어 있는 것은 다음이다.

- active contact snapshot
- stacked `Jc`, `motion_rhs`, `T`, `Uf`, `uf_lb`, `uf_ub`
- contact acceleration hard constraint
- friction hard inequality
- floating-base dynamics contact coupling
- joint torque limit contact coupling
- global `lambda` regularization
- `lambda_sol` decode와 model torque recovery

아직 문서나 설계가 더 필요한 지점은 다음이다.

- `TaskContactForceEquality`와 `TaskCopEquality`를 final `IDProblem` objective로
  어떤 schema에 넣을지
- per-contact `forceRegTask`를 유지할지, global `lambda` regularization으로 통일할지
- 3D force task와 6D wrench task를 같은 API로 다룰지 분리할지
- force objective의 hierarchy level과 weight override를 state machine에서 어떻게
  표현할지

이 문서는 contact feasibility와 dynamics coupling을 먼저 고정한다. force tracking
task wiring은 별도 설계가 정리되면 이어서 업데이트하는 것이 좋다.
