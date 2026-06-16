# TaskSE3Equality

이 문서는 `wbc_core`의 `TaskSE3Equality`가 SE(3) task-space error를
어떻게 만들고, 그 error가 inverse-dynamics HQP 문제의 equality constraint로
어떻게 들어가는지 설명한다.

대상 구현:

- `wbc_core/include/wbc_core/tasks/task-se3-equality.hpp`
- `wbc_core/src/tasks/task-se3-equality.cpp`
- `wbc_core/src/math/lie_group/se3.cpp`

## 이 Task가 하는 일

`TaskSE3Equality`는 로봇 모델 안의 특정 frame을 원하는 3D pose로 추종시키는
motion task다. 여기서 pose는 위치와 자세를 합친 값이다.

예를 들어 end-effector frame을 원하는 위치와 자세로 보내고 싶다면, 이 task는
현재 frame pose와 목표 frame pose의 차이를 계산하고, 그 차이가 줄어드는 방향의
task acceleration을 만든다. 그 다음 solver가 풀 수 있도록 다음 형태의 식을
만든다.

```text
J_task(q) * qddot = a_des - drift
```

각 항의 의미는 다음과 같다.

- `qddot`: solver가 찾을 joint acceleration
- `J_task(q)`: 현재 frame의 6D Jacobian
- `a_des`: pose error와 velocity error를 줄이기 위해 task가 원하는 6D acceleration
- `drift`: 현재 속도 때문에 생기는 frame acceleration 항, 대략 `Jdot * qdot` 역할

즉 task는 직접 torque를 만들지 않는다. task는 "이 frame이 이런 acceleration을
가지려면 joint acceleration이 이 식을 만족해야 한다"는 constraint를 만든다.

## Lie Group과 Lie Algebra

`TaskSE3Equality`를 읽을 때 `SE(3)`, `SO(3)`, `se(3)`, `so(3)`, `log3`, `log6`
같은 말이 계속 나온다. 이 용어들은 "pose와 rotation을 일반 벡터처럼 빼면 안
되기 때문에, 제어에 쓸 수 있는 error 벡터로 바꾸는 방법"을 설명하기 위해 필요하다.

먼저 큰 그림은 다음과 같다.

| 이름 | 의미 | 코드에서 주로 대응되는 것 |
| --- | --- | --- |
| `SO(3)` | 3D rotation들의 공간 | `Eigen::Matrix3d`, `SE3::rotation()` |
| `SE(3)` | 3D rigid-body pose들의 공간 | `pinocchio::SE3` |
| `so(3)` | rotation 주변의 작은 회전, angular error/velocity 공간 | `Eigen::Vector3d`, `pinocchio::log3(...)` 결과 |
| `se(3)` | pose 주변의 작은 6D motion, twist/error/velocity 공간 | `pinocchio::Motion`, 6D vector |

대문자 `SO(3)`, `SE(3)`는 실제 rotation 또는 pose 자체를 뜻한다. 소문자
`so(3)`, `se(3)`는 그 rotation 또는 pose 근처에서 "얼마나 움직여야 하는가"를
나타내는 error, velocity, acceleration 쪽 공간이라고 보면 된다.

여기서 중요한 구분이 있다. `SO(3)`와 `SE(3)`는 실제 transformation이다. 반면
`so(3)`와 `se(3)`는 transformation 자체가 아니라 transformation의 "작은 변화량",
"접선 벡터", 또는 "생성자(generator)"에 가깝다. 그래서 `so(3)`나 `se(3)` 값을
그대로 pose transform처럼 점에 곱해서 좌표를 바꾸는 식으로 쓰면 안 된다. 실제
transformation으로 쓰려면 `exp3`, `exp6`를 통해 `SO(3)`, `SE(3)` 값으로 올려야
한다.

```text
so(3) --exp3--> SO(3)
se(3) --exp6--> SE(3)

SO(3) --log3--> so(3)
SE(3) --log6--> se(3)
```

### Group이란 무엇인가

group은 어떤 값들을 합성할 수 있고, 되돌릴 수 있고, 아무것도 안 하는 identity가
있는 집합이다. rotation과 pose는 모두 group으로 볼 수 있다.

rotation 예시는 다음과 같다.

```text
R_total = R1 * R2
```

`R2`만큼 돌린 뒤 `R1`만큼 더 돌린 결과도 여전히 rotation이다. 그리고 `R`의 inverse
rotation도 존재한다.

pose도 마찬가지다.

```text
T_total = T1 * T2
```

`T2` 변환을 적용한 뒤 `T1` 변환을 적용한 결과도 여전히 pose transform이다. 그리고
`T`의 inverse transform도 존재한다.

### Lie Group이란 무엇인가

Lie group은 group인데, 그 안에서 값이 연속적으로 변할 수 있는 구조다.

로봇 자세는 갑자기 순간이동하듯 바뀌는 값이 아니라, 아주 작은 변화들을 이어 붙여
움직인다. 그래서 rotation과 pose는 다음처럼 생각할 수 있다.

```text
현재 rotation + 아주 작은 회전 변화 -> 다음 rotation
현재 pose     + 아주 작은 6D motion  -> 다음 pose
```

이 "아주 작은 변화"를 다룰 수 있기 때문에 Lie group이 제어와 로봇 동역학에서
많이 쓰인다.

### SO(3): 3D Rotation의 공간

`SO(3)`는 3D rotation matrix들의 공간이다. 수식으로는 다음 조건을 만족하는 3x3
matrix들의 집합이다.

```text
SO(3) = { R | R^T * R = I, det(R) = 1 }
```

말로 풀면 다음 뜻이다.

- `R^T * R = I`: 축들이 서로 직교하고 길이가 1이다.
- `det(R) = 1`: 좌표계를 뒤집는 reflection이 아니라 순수 rotation이다.

rotation matrix는 9개의 숫자로 보이지만, 실제 자유도는 3개뿐이다. 그래서 두
rotation matrix를 element-wise로 그냥 빼면 "어느 축으로 얼마나 돌아야 하는가"라는
3D error가 바로 나오지 않는다.

### SE(3): 3D Pose의 공간

`SE(3)`는 3D rigid-body transform들의 공간이다. 위치 translation과 자세 rotation을
함께 가진다.

수식에서는 보통 다음 4x4 matrix로 쓴다.

```text
T = [ R  p ]
    [ 0  1 ]
```

- `R`은 `SO(3)` rotation이다.
- `p`는 3D translation이다.

Pinocchio의 `pinocchio::SE3`가 이 값을 표현한다.

```cpp
pinocchio::SE3 T;
T.rotation();     // SO(3) rotation R
T.translation();  // translation p
```

`TaskSE3Equality`의 현재 frame pose `oMi`와 목표 pose `m_M_ref`는 둘 다
`pinocchio::SE3`다. 즉 task가 비교하는 원본 값은 6D vector가 아니라 SE(3) pose다.

### Lie Algebra란 무엇인가

Lie algebra는 Lie group 근처의 작은 변화를 벡터처럼 다룰 수 있게 만든 공간이다.

rotation이나 pose 자체는 일반 벡터 공간이 아니다. 예를 들어 rotation matrix 두
개를 더하거나 빼면 결과가 더 이상 정상적인 rotation matrix가 아닐 수 있다. 하지만
제어기는 보통 다음과 같은 벡터 연산을 하고 싶어 한다.

```text
a_des = Kp * error + Kd * velocity_error + a_ref
```

그러려면 pose 차이를 `error`라는 벡터로 바꿔야 한다. 이때 group의 값을 algebra의
값으로 내려주는 함수가 `log`다.

```text
rotation error:  SO(3) -> so(3)  by log3
pose error:      SE(3) -> se(3)  by log6
```

반대로 algebra의 작은 변화량을 다시 group 값으로 올리는 함수가 `exp`다.

```text
small rotation vector -> SO(3) rotation  by exp3
6D twist vector       -> SE(3) pose      by exp6
```

여기서 `log`는 로그 출력(log file)이 아니라 수학의 logarithm map이다.

### so(3): Rotation Error와 Angular Velocity의 공간

`so(3)`는 `SO(3)` 주변의 작은 회전을 표현하는 공간이다. 수학적으로는 3x3
skew-symmetric matrix로 쓰지만, 로봇 코드에서는 보통 3D vector로 다룬다.

```text
omega = [wx, wy, wz]
```

이 벡터는 "어느 축으로 얼마나 회전해야 하는가" 또는 "어느 축으로 얼마나 빠르게
돌고 있는가"를 표현한다.

수학적으로 `so(3)` 원소는 다음처럼 skew-symmetric matrix로도 표현된다.

```text
hat(omega) = [  0  -wz   wy ]
             [ wz    0  -wx ]
             [-wy   wx    0 ]
```

하지만 이 matrix는 rotation matrix가 아니다. `R^T * R = I`, `det(R) = 1` 조건을
만족하는 `SO(3)` 값이 아니기 때문이다. 이 matrix는 rotation을 "만들어내는"
생성자다. 실제 rotation은 `exp3(omega)` 또는 `exp(hat(omega))`를 통해 얻는다.

```text
R = exp(hat(omega))  // R is in SO(3)
```

Pinocchio에서는 상대 rotation matrix를 `log3`에 넣으면 이 3D vector가 나온다.

```cpp
Eigen::Vector3d orientation_error = pinocchio::log3(R_error);
```

`TaskSE3Equality`의 orientation error는 이 방식으로 만든다.

```cpp
error.angular() = pinocchio::log3(transform_error.rotation());
```

### se(3): Pose Error와 6D Motion의 공간

`se(3)`는 `SE(3)` 주변의 작은 rigid-body motion을 표현하는 공간이다. 로봇 제어
코드에서는 보통 6D vector로 다룬다.

```text
motion = [ linear_x, linear_y, linear_z, angular_x, angular_y, angular_z ]
```

Pinocchio의 `pinocchio::Motion`은 이 6D motion을 표현하는 타입이다. 이 프로젝트의
task 코드에서는 `Motion::linear()`와 `Motion::angular()`를 채우고, `toVector()`로
6D vector를 얻는다.

수학적으로 `se(3)` 원소는 다음과 같은 4x4 matrix로도 쓸 수 있다.

```text
hat(xi) = [ hat(omega)  v ]
          [     0       0 ]
```

여기서 `xi = [v; omega]`다. 이 matrix도 homogeneous transformation matrix가 아니다.
실제 `SE(3)` transform은 마지막 행이 `[0 0 0 1]`이고 왼쪽 위가 rotation matrix여야
한다. 반면 `hat(xi)`의 왼쪽 위는 skew-symmetric matrix이고 오른쪽 아래는 `0`이다.
즉 `se(3)`는 transform 자체가 아니라 transform을 만들어내는 6D motion generator다.

```text
T = exp(hat(xi))  // T is in SE(3)
```

```cpp
pinocchio::Motion error;
error.linear() = position_error;
error.angular() = orientation_error;

Eigen::VectorXd e = error.toVector();
```

`log6`를 쓰면 SE(3) transform 전체를 한 번에 `se(3)`의 6D motion으로 바꾼다.

```text
se3_error = log6(T_error)
```

하지만 현재 `TaskSE3Equality` 구현은 `log6`를 쓰지 않는다. 현재 구현은 SE(3)
상대 transform을 만든 뒤, translation은 그대로 쓰고 rotation에만 `log3`를 적용해
`pinocchio::Motion`을 채운다.

```text
current implementation:
  position_error    = translation(T_error)
  orientation_error = log3(rotation(T_error))
```

따라서 이 문서에서 말하는 `se(3)` 또는 6D motion은 "항상 `log6` 결과"라는 뜻이
아니다. 6D motion 형식은 같지만, 그 6D error를 만드는 방법은 현재 방식과 `log6`
방식이 다를 수 있다.

## 수학 표기와 코드 구현

이 절은 같은 개념을 LaTeX 수식과 C++ 코드에서 각각 어떻게 표현하는지 정리한다.
실제 `TaskSE3Equality` 구현은 `hat` matrix나 matrix exponential을 직접 구현하지
않고, Pinocchio가 제공하는 `SE3`, `Motion`, `log3`, `log6`, `exp3`, `exp6`를 쓴다.

### SO(3)와 so(3)

3D rotation matrix의 공간은 다음처럼 쓴다.

$$
SO(3) =
\left\{
R \in \mathbb{R}^{3 \times 3}
\mid
R^\top R = I,\ \det(R) = 1
\right\}
$$

rotation error나 angular velocity는 보통 3D vector로 들고 다닌다.

$$
\omega =
\begin{bmatrix}
\omega_x \\
\omega_y \\
\omega_z
\end{bmatrix}
\in \mathbb{R}^3
$$

수학적으로 `so(3)` 원소는 이 3D vector를 skew-symmetric matrix로 바꾼 값이다.

$$
\widehat{\omega} =
\begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
\in \mathfrak{so}(3)
$$

`so(3)`에서 `SO(3)`로 올라갈 때는 exponential map을 쓴다.

$$
R = \exp(\widehat{\omega})
$$

반대로 rotation matrix에서 angular error vector를 얻을 때는 logarithm map을 쓴다.

$$
\omega = \log(R)^\vee
$$

여기서 `vee`는 skew-symmetric matrix를 다시 3D vector로 펴는 연산이다. 코드에서는
보통 `hat`/`vee`를 직접 만들지 않고 Pinocchio API를 호출한다.

```cpp
Eigen::Matrix3d R_error = transform_error.rotation();

// SO(3) -> so(3), returned as a 3D vector.
Eigen::Vector3d omega = pinocchio::log3(R_error);

// so(3) 3D vector -> SO(3).
Eigen::Matrix3d R = pinocchio::exp3(omega);
```

현재 `TaskSE3Equality`의 orientation error도 이 구현을 쓴다.

```cpp
error.angular() = pinocchio::log3(transform_error.rotation());
```

수식으로는 다음과 같다.

$$
e_R = \log(R_{err})^\vee
$$

### SE(3)와 se(3)

3D pose 또는 rigid-body transform은 다음처럼 쓴다.

$$
T =
\begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix}
\in SE(3)
$$

여기서 각 항은 다음 의미다.

$$
R \in SO(3), \qquad p \in \mathbb{R}^3
$$

코드에서는 `pinocchio::SE3`가 이 값을 표현한다.

```cpp
pinocchio::SE3 T;

Eigen::Matrix3d R = T.rotation();
Eigen::Vector3d p = T.translation();
```

`se(3)` 원소는 6D vector로 다룬다.

$$
\xi =
\begin{bmatrix}
v \\
\omega
\end{bmatrix}
\in \mathbb{R}^6
$$

여기서 `v`는 linear part, `omega`는 angular part다. matrix 형태로 쓰면 다음과 같다.

$$
\widehat{\xi} =
\begin{bmatrix}
\widehat{\omega} & v \\
0 & 0
\end{bmatrix}
\in \mathfrak{se}(3)
$$

다시 강조하면, 이 `\widehat{\xi}`는 homogeneous transformation matrix가 아니다.
실제 transform은 `exp`를 거쳐야 한다.

$$
T = \exp(\widehat{\xi})
$$

반대로 `SE(3)` transform에서 6D motion vector를 얻는 식은 다음이다.

$$
\xi = \log(T)^\vee
$$

Pinocchio 코드에서는 `se(3)`의 6D 값을 `pinocchio::Motion`으로 표현한다.

```cpp
pinocchio::Motion xi;

Eigen::Vector3d v = xi.linear();
Eigen::Vector3d omega = xi.angular();
Eigen::Matrix<double, 6, 1> xi_vec = xi.toVector();
```

full `log6` 방식을 쓰면 코드는 다음처럼 된다.

```cpp
pinocchio::SE3 T_error = current.actInv(desired);

// SE(3) -> se(3), returned as pinocchio::Motion.
pinocchio::Motion xi_error = pinocchio::log6(T_error);

// se(3) -> SE(3).
pinocchio::SE3 T_from_xi = pinocchio::exp6(xi_error);
```

수식으로는 다음이다.

$$
T_{err} = T_{current}^{-1} T_{desired}
$$

$$
\xi_{err} =
\log(T_{err})^\vee =
\begin{bmatrix}
\rho \\
\phi
\end{bmatrix}
$$

여기서 `rho`는 `log6`의 linear component이고, `phi`는 angular component다.

### 현재 TaskSE3Equality의 Error 구현

현재 구현은 full `log6`를 쓰지 않는다. 먼저 현재 pose와 목표 pose의 상대 transform을
만든다.

```cpp
const pinocchio::SE3 transform_error = current.actInv(desired);
```

수식으로는 다음이다.

$$
T_{err} =
T_{current}^{-1} T_{desired}
=
\begin{bmatrix}
R_{err} & p_{err} \\
0 & 1
\end{bmatrix}
$$

그 다음 translation part는 그대로 쓰고, rotation part만 `log3`로 바꾼다.

```cpp
error.linear() = transform_error.translation();
error.angular() = pinocchio::log3(transform_error.rotation());
```

수식으로는 다음이다.

$$
e_{split} =
\begin{bmatrix}
p_{err} \\
\log(R_{err})^\vee
\end{bmatrix}
$$

이 값이 `pinocchio::Motion`에 저장되고, 이후 `toVector()`로 6D vector가 된다.

```cpp
m_p_error_vec = m_p_error.toVector();
```

full `log6` 방식이었다면 수식은 다음처럼 바뀐다.

$$
e_{log6} =
\log(T_{err})^\vee
=
\begin{bmatrix}
\rho \\
\phi
\end{bmatrix}
$$

그리고 코드는 대략 다음처럼 된다.

```cpp
pinocchio::Motion error = pinocchio::log6(transform_error);
```

둘 다 6D vector를 만들지만, linear part가 다르다.

$$
e_{split,linear} = p_{err}
$$

$$
e_{log6,linear} = \rho
$$

일반적으로 회전 error가 있으면 `p_{err}`와 `rho`는 같은 의미가 아니다. 그래서 현재
방식에서 `log6` 방식으로 바꾸면 task error 정의가 바뀐다.

### 왜 Translation과 Orientation Error를 따로 계산하는가

현재 구현이 translation과 orientation을 따로 계산하는 이유는 단순히 코드가 쉬워서가
아니다. 이 task의 제어 의미를 명확하게 유지하기 위한 선택이다.

상대 transform은 다음처럼 만든다.

$$
T_{err} =
T_{current}^{-1} T_{desired}
=
\begin{bmatrix}
R_{err} & p_{err} \\
0 & 1
\end{bmatrix}
$$

여기서 `p_err`는 이미 3D vector다.

$$
p_{err} \in \mathbb{R}^3
$$

즉 현재 frame 기준에서 목표 frame origin이 어디에 있는지를 바로 나타낸다. 그래서
현재 task convention에서는 이 값을 position error로 그대로 쓸 수 있다.

$$
e_p = p_{err}
$$

반면 `R_err`는 3D vector가 아니라 rotation matrix다.

$$
R_{err} \in SO(3)
$$

rotation matrix는 숫자 9개로 보이지만 실제로는 일반 벡터 공간의 값이 아니다.
따라서 `R_desired - R_current` 같은 matrix 뺄셈을 orientation error로 쓰면
"어느 축으로 얼마나 회전해야 하는가"가 명확하지 않다. 그래서 `log3`를 통해
rotation을 3D tangent vector로 바꾼다.

$$
e_R = \log(R_{err})^\vee
$$

현재 구현의 6D pose error는 다음과 같다.

$$
e_{split} =
\begin{bmatrix}
e_p \\
e_R
\end{bmatrix}
=
\begin{bmatrix}
p_{err} \\
\log(R_{err})^\vee
\end{bmatrix}
$$

코드에서는 다음 두 줄이 이 식에 해당한다.

```cpp
error.linear() = transform_error.translation();
error.angular() = pinocchio::log3(transform_error.rotation());
```

이 방식의 실용적인 장점은 다음과 같다.

- position row의 의미가 `x/y/z 방향으로 얼마나 이동해야 하는가`로 직접 읽힌다.
- orientation row의 의미가 `roll/pitch/yaw 축 계열로 얼마나 회전해야 하는가`에 가까운
  angular error로 읽힌다.
- 6D mask의 의미가 명확하다. position만 켜면 `p_err`만 제어하고, orientation만 켜면
  `log3(R_err)`만 제어한다.
- `Kp`, `Kd` gain도 position 3축과 orientation 3축으로 나누어 이해하기 쉽다.

반대로 `log6`를 쓰면 다음처럼 full SE(3) error를 한 번에 만든다.

$$
e_{log6} =
\log(T_{err})^\vee
=
\begin{bmatrix}
\rho \\
\phi
\end{bmatrix}
$$

이 방식은 "현재 pose에서 목표 pose로 가는 하나의 screw/twist motion"을 error로
보는 관점이다. 이 관점도 수학적으로 타당하다. 다만 이 경우 linear part인 `rho`는
단순히 `p_err`와 같은 뜻이 아니다. rotation error와 결합된 SE(3) logarithm의
linear component다.

그래서 현재 방식은 다음 의도를 가진다.

```text
position control    -> relative translation을 직접 줄인다.
orientation control -> relative rotation을 log3 angular error로 줄인다.
```

`log6` 방식은 다음 의도에 더 가깝다.

```text
full pose control -> relative transform 전체를 하나의 6D twist error로 줄인다.
```

따라서 translation/orientation을 따로 계산하는 것은 "수학을 덜 쓴 방식"이라기보다,
position task와 orientation task의 의미, mask, gain tuning을 분리해서 유지하는 control
law 선택이다.

### SE3 저장 벡터와 Error 벡터는 다르다

헷갈리기 쉬운 지점은 `SE3` pose를 저장하기 위한 벡터와 task error 벡터가 모두
`Eigen::Vector` 형태로 보인다는 점이다. 하지만 두 벡터의 의미는 다르다.

`SE3`는 실제 transform이다. 수식으로는 TF matrix처럼 쓸 수 있다.

$$
T =
\begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix}
\in SE(3)
$$

코드에서는 이 transform을 raw 4x4 matrix로 들고 다니지 않고 `pinocchio::SE3`로
들고 다닌다.

```cpp
pinocchio::SE3 T;
Eigen::Matrix3d R = T.rotation();
Eigen::Vector3d p = T.translation();
```

반면 `se3ToVector`의 12D vector는 reference pose를 저장하거나 표시하기 위한
프로젝트 내부 표현이다.

```cpp
se3ToVector(m_M_ref, m_p_ref);
se3ToVector(oMi, m_p);
```

수식으로 보면 다음 정보를 한 줄로 담은 것이다.

$$
x_{SE3-storage} =
\begin{bmatrix}
p \\
\operatorname{vec}(R)
\end{bmatrix}
\in \mathbb{R}^{12}
$$

이 값은 pose 저장 형식이지 task error가 아니다. rotation matrix 9개를 그대로 담고
있기 때문에, 이 벡터를 빼서 orientation error로 쓰면 안 된다.

task error는 6D motion 형식이다.

$$
e =
\begin{bmatrix}
e_{linear} \\
e_{angular}
\end{bmatrix}
\in \mathbb{R}^{6}
$$

현재 구현에서는 이 6D error를 다음처럼 만든다.

$$
e =
\begin{bmatrix}
p_{err} \\
\log(R_{err})^\vee
\end{bmatrix}
$$

코드로는 `pinocchio::Motion`에 linear/angular part를 채운 뒤 `toVector()`로 solver가
먹을 수 있는 6D vector로 바꾼다.

```cpp
error.linear() = transform_error.translation();
error.angular() = pinocchio::log3(transform_error.rotation());

m_p_error_vec = m_p_error.toVector();
```

정리하면 다음과 같다.

| 값 | 차원 | 의미 | 코드 |
| --- | --- | --- | --- |
| `pinocchio::SE3` | transform object | 실제 pose/TF transform | `oMi`, `m_M_ref` |
| SE3 storage vector | 12D | pose 저장/표시용 `[p; vec(R)]` | `se3ToVector`, `vectorToSE3` |
| task error vector | 6D | controller가 줄이려는 tangent error | `pinocchio::Motion::toVector()` |
| solver RHS | task dim | mask 적용 후 HQP에 들어가는 vector | `m_constraint.vector()` |

따라서 "SE3는 TF matrix인가?"에는 "수학적으로는 그렇고, 코드에서는
`pinocchio::SE3` 객체로 표현한다"가 답이다. 하지만 "error는 그냥 flatten array인가?"
에는 "아니다"가 답이다. error는 TF matrix를 펼친 값이 아니라, relative transform에서
뽑아낸 6D tangent vector다.

### Desired Acceleration과 Constraint 구현

현재 local frame 모드에서 velocity error와 desired acceleration은 다음 코드로 만든다.

```cpp
m_v_error = m_wMl.actInv(m_v_ref) - v_frame;

m_a_des = m_Kp.cwiseProduct(m_p_error_vec) +
          m_Kd.cwiseProduct(m_v_error.toVector()) +
          m_wMl.actInv(m_a_ref).toVector();
```

수식으로 쓰면 다음과 같다.

$$
e_v = v_{ref}^{local} - v_{current}^{local}
$$

$$
a_{des} =
K_p \odot e_{pose}
+ K_d \odot e_v
+ a_{ref}^{local}
$$

여기서 `\odot`는 element-wise multiplication이다. 즉 6D 각 row마다 gain을 따로
곱한다.

마지막으로 task는 HQP에 넣을 equality constraint를 만든다.

```cpp
m_constraint.matrix().row(idx) = m_J.row(i);
m_constraint.vector().row(idx) = (m_a_des - drift).row(i);
```

수식은 다음이다.

$$
J(q)\ddot{q} = a_{des} - a_{drift}
$$

frame acceleration 관점에서는 다음 식을 만족시키려는 것이다.

$$
a_{task} = J(q)\ddot{q} + a_{drift}
$$

따라서 `a_task = a_des`가 되도록 solver 오른쪽 항에 `a_des - a_drift`를 넣는다.

## Transformation Matrix와 Pinocchio SE3

수식에서는 3D pose를 보통 4x4 homogeneous transformation matrix로 쓴다.

```text
T = [ R  p ]
    [ 0  1 ]
```

- `R`: 3x3 rotation matrix. frame의 자세를 표현한다.
- `p`: 3x1 translation vector. frame의 위치를 표현한다.

이 matrix는 local frame의 점 `x_local`을 world frame의 점 `x_world`로 바꿀 수
있다.

```text
x_world = R * x_local + p
```

Pinocchio에서는 이 4x4 matrix를 직접 들고 다니기보다 `pinocchio::SE3` 타입으로
관리한다. `SE3` 안에는 같은 정보가 `rotation()`과 `translation()`으로 들어 있다.

```cpp
pinocchio::SE3 T;
T.rotation();     // R
T.translation();  // p
```

코드에서 현재 frame pose는 다음처럼 가져온다.

```cpp
SE3 oMi;
m_robot.framePosition(data, m_frame_id, oMi);
```

`oMi`는 Pinocchio 표기 관례에 가까운 이름이다. `o`는 world/origin frame, `i`는
현재 frame을 뜻한다고 보면 된다. 즉 `oMi`는 "frame i의 pose를 world frame에서
본 값"이다.

목표 pose는 `m_M_ref`에 저장된다.

```cpp
vectorToSE3(ref.pos, m_M_ref);
```

reference trajectory의 `pos`는 12차원 벡터다. 4x4 matrix 전체 16개를 저장하지
않는 이유는 마지막 행 `[0 0 0 1]`이 항상 고정이기 때문이다. 그래서 이 저장
형식은 다음 정보만 가진다.

```text
reference pos = [ translation 3개 ; rotation matrix 계수 9개 ]
```

이 벡터와 `pinocchio::SE3` 사이의 변환은 `se3ToVector`, `vectorToSE3`가 맡는다.
이 helper들은 수학을 새로 구현하는 함수가 아니라, 이 프로젝트에서 쓰는 저장
형식과 Pinocchio의 `SE3` 타입을 연결하는 변환 함수다.

## 현재 Pose Error를 만드는 방법

현재 pose를 `T_current`, 목표 pose를 `T_desired`라고 하자. task가 먼저 만드는
것은 두 pose의 절대 차이가 아니라 "현재 frame에서 봤을 때 목표 frame이 어디에
있는가"이다.

수식으로는 다음이다.

```text
T_error = inverse(T_current) * T_desired
```

코드에서는 이 계산을 직접 inverse/multiply로 쓰지 않고 Pinocchio API로 쓴다.

```cpp
const pinocchio::SE3 transform_error = current.actInv(desired);
```

`current.actInv(desired)`는 `current.inverse() * desired`와 같은 의미의 상대
변환을 만든다고 보면 된다. 이렇게 만든 `T_error`는 "현재 frame 기준의 목표
pose"다.

현재 구현의 error 생성은 `poseError`에 있다.

```cpp
const pinocchio::SE3 transform_error = current.actInv(desired);
error.linear() = transform_error.translation();
error.angular() = pinocchio::log3(transform_error.rotation());
```

즉 현재 구현은 pose error를 다음처럼 만든다.

```text
T_error = inverse(T_current) * T_desired

position_error    = translation(T_error)
orientation_error = log3(rotation(T_error))

pose_error = [ position_error ; orientation_error ]
```

중요한 점은 현재 구현이 `T_error` 전체에 `log6`를 적용하지 않는다는 것이다.
translation은 상대 변환의 translation을 그대로 쓰고, orientation만 `log3`로
3차원 회전 error로 바꾼다.

## 왜 Rotation에는 log를 쓰는가

위치 error는 직관적이다. 현재 위치가 `(1, 0, 0)`이고 목표 위치가 `(2, 0, 0)`이면
대략 `+1m`만큼 이동하면 된다.

하지만 자세는 단순 뺄셈으로 다루면 안 된다. rotation matrix나 quaternion은
일반적인 3D 벡터가 아니라 "회전"이라는 공간 위의 값이다. 두 rotation matrix를
그냥 빼면 다음 질문에 답하기 어렵다.

```text
현재 자세에서 목표 자세로 가려면 어느 축으로 몇 radian 회전해야 하는가?
```

`log3`는 이 질문에 답하는 함수다. 상대 rotation `R_error`를 3차원 벡터로 바꾼다.
이 벡터는 "어느 축으로 얼마나 회전해야 하는지"를 담는 rotation vector다.

```text
orientation_error = log3(R_error)
```

작은 회전에서는 이 값이 우리가 직관적으로 생각하는 angular error처럼 동작한다.
예를 들어 z축으로 0.1 rad만큼 더 돌아야 한다면 대략 다음과 비슷한 값이 나온다.

```text
[0, 0, 0.1]
```

그래서 task-space PD 제어식에서 orientation error를 velocity/acceleration과 같은
6D motion vector 안에 넣을 수 있다.

## 현재 방식과 log6 방식의 차이

SE(3) pose error를 만드는 대표적인 방식은 크게 두 가지로 볼 수 있다.

현재 구현은 translation과 rotation을 분리해서 본다.

```text
T_error = inverse(T_current) * T_desired

e_current = [ translation(T_error) ;
              log3(rotation(T_error)) ]
```

`log6` 방식은 `T_error` 전체를 한 번에 Lie algebra의 6D motion vector로 바꾼다.

```text
T_error = inverse(T_current) * T_desired

e_log6 = log6(T_error)
       = [ rho ;
           phi ]
```

여기서 `phi`는 rotation 쪽 error이고, 현재 방식의 `log3(rotation(T_error))`와
같은 계열의 값이다. 하지만 `rho`는 단순히 `translation(T_error)`와 항상 같지
않다. `log6`의 translation 부분은 rotation error와 함께 "현재 pose에서 목표
pose로 가는 하나의 6D twist"를 만들기 위해 계산된다.

직관적으로 말하면 다음 차이다.

| 방식 | 직관 | translation error 의미 | orientation error 의미 |
| --- | --- | --- | --- |
| 현재 방식 | 위치와 자세 error를 각각 만든 뒤 6D 벡터로 합친다. | 상대 TF의 translation을 그대로 쓴다. | 상대 rotation에 `log3`를 적용한다. |
| `log6` 방식 | 현재 pose에서 목표 pose로 가는 하나의 SE(3) motion을 만든다. | rotation과 결합된 twist의 linear 부분이다. | 같은 SE(3) twist의 angular 부분이다. |

그래서 `poseError`를 단순히 `log6`로 바꾸는 것은 이름만 바꾸는 refactor가 아니다.
task의 error 정의가 바뀐다. error 정의가 바뀌면 gain tuning, mask 해석, 기존
테스트 기준도 같이 달라질 수 있다.

## log6를 쓰면 항상 Position과 Orientation을 같이 제어하는가

그렇지는 않다. 이 task에는 mask가 있다.

6D vector의 순서는 다음이다.

```text
[ linear_x, linear_y, linear_z, angular_x, angular_y, angular_z ]
```

mask는 이 6개 중 어떤 row를 solver constraint에 넣을지 고른다.

```text
[1, 1, 1, 0, 0, 0]  -> position row만 사용
[0, 0, 0, 1, 1, 1]  -> orientation row만 사용
[1, 1, 1, 1, 1, 1]  -> full SE3 task
```

따라서 `log6`를 쓰더라도 mask로 position row만 고를 수는 있다. 다만 그
position row의 의미가 현재 방식과 달라진다.

- 현재 방식의 position row: `translation(T_error)`
- `log6` 방식의 position row: `log6(T_error)`의 linear component

`T_error`에 rotation error가 있으면 `log6`의 linear component는 단순 translation과
달라질 수 있다. 그래서 position만 켠 mask에서도 controller 반응이 달라질 수 있다.

## Local Frame과 Local World-Oriented Frame

`TaskSE3Equality`는 계산 결과를 어느 frame 기준으로 표현할지 선택할 수 있다.

기본값은 local frame이다.

```cpp
m_local_frame = true;
```

### Local frame

local frame 모드에서는 Jacobian, velocity error, desired acceleration을 현재 frame
기준으로 둔다.

```cpp
m_robot.frameJacobianLocal(data, m_frame_id, m_J);
poseError(oMi, m_M_ref, m_p_error);

m_p_error_vec = m_p_error.toVector();
m_v_error = m_wMl.actInv(m_v_ref) - v_frame;

m_a_des = Kp * position_error
        + Kd * velocity_error
        + reference_acceleration;
```

여기서 `m_wMl`은 현재 frame의 rotation만 가진 SE3다.

```cpp
m_wMl.rotation(oMi.rotation());
```

translation은 사용하지 않고 rotation만 사용한다. 목적은 reference velocity와
acceleration을 현재 frame 기준으로 돌려서 비교하기 위함이다.

### Local world-oriented frame

`useLocalFrame(false)`를 호출하면 local world-oriented frame을 쓴다. 이 모드는
origin은 현재 frame 쪽을 따르되, 축 방향은 world와 정렬된 표현이라고 이해하면
된다.

코드에서는 local frame에서 얻은 error, velocity, drift, Jacobian을 `m_wMl`의 action
matrix로 회전시킨다.

```cpp
m_p_error_vec = m_wMl.toActionMatrix() * m_p_error.toVector();
m_v_error = m_v_ref - m_wMl.act(v_frame);
m_drift = m_wMl.act(m_drift);
m_J = m_wMl.toActionMatrix() * m_J;
```

이렇게 하면 solver에 들어가는 row들이 local frame 표현이 아니라 world 축과 맞춘
표현이 된다.

## Desired Acceleration과 HQP Constraint

pose error와 velocity error가 만들어지면 task는 desired acceleration을 만든다.

```text
a_des = Kp .* pose_error + Kd .* velocity_error + a_ref
```

`.*`는 element-wise 곱이다. 6D 각 축마다 다른 gain을 줄 수 있다는 뜻이다.

그 다음 실제 solver constraint는 다음처럼 만들어진다.

```cpp
m_constraint.matrix().row(idx) = m_J.row(i);
m_constraint.vector().row(idx) = (m_a_des - drift).row(i);
```

즉 solver에는 다음 식이 들어간다.

```text
J_task(q) * qddot = a_des - drift
```

왜 `drift`를 빼는지 이해하려면 frame acceleration 관계를 보면 된다.

```text
task_acceleration = J_task(q) * qddot + drift
```

원하는 것은 `task_acceleration = a_des`이므로 solver가 풀 수 있는 형태로 옮기면
다음이 된다.

```text
J_task(q) * qddot = a_des - drift
```

`getAcceleration(dv)`가 다음처럼 구현된 이유도 같다.

```cpp
return m_constraint.matrix() * dv + m_drift_masked;
```

constraint matrix에 `dv`, 즉 solver가 찾은 joint acceleration을 곱하고 drift를 다시
더하면 실제 task acceleration을 복원할 수 있다.

## 수식과 코드의 대응

| 수식/개념 | 코드 |
| --- | --- |
| 현재 frame pose `T_current` | `m_robot.framePosition(data, m_frame_id, oMi)` |
| 목표 frame pose `T_desired` | `m_M_ref` |
| 상대 pose `T_error = inverse(T_current) * T_desired` | `current.actInv(desired)` |
| 상대 translation | `transform_error.translation()` |
| 상대 rotation | `transform_error.rotation()` |
| rotation error | `pinocchio::log3(transform_error.rotation())` |
| 6D pose error | `m_p_error.toVector()` |
| frame velocity | `m_robot.frameVelocity(data, m_frame_id, v_frame)` |
| drift acceleration | `m_robot.frameClassicAcceleration(data, m_frame_id, m_drift)` |
| local frame Jacobian | `m_robot.frameJacobianLocal(data, m_frame_id, m_J)` |
| solver 식의 왼쪽 | `m_constraint.matrix() = selected rows of m_J` |
| solver 식의 오른쪽 | `m_constraint.vector() = selected rows of (m_a_des - drift)` |


## 간단한 실행 흐름

한 control tick에서 `TaskSE3Equality::compute()`는 대략 다음 순서로 동작한다.

1. Pinocchio data에서 현재 frame pose, velocity, drift를 읽는다.
2. 현재 pose `oMi`와 목표 pose `m_M_ref`로 상대 transform을 만든다.
3. 상대 transform에서 translation error와 `log3` rotation error를 만든다.
4. velocity error를 만든다.
5. PD 식으로 desired task acceleration `a_des`를 만든다.
6. Jacobian row와 `a_des - drift` row를 mask에 맞게 골라 equality constraint를 만든다.
7. ID-HQP solver가 이 constraint를 포함해서 `qddot`을 푼다.

핵심은 이 task가 "pose를 직접 맞추는 함수"가 아니라는 점이다. 이 task는 현재
pose와 목표 pose의 차이를 6D acceleration 목표로 바꾸고, 그 목표를 HQP가 풀 수
있는 선형 constraint 형태로 제공한다.
