1. 최종 마스터 플랜
북극성

최종적으로는 시스템을 이렇게 해석해야 해:

YAML / external config
  -> compiled config compile
  -> runtime assembly
  -> FSM / handlers / managers update references
  -> registry/adapters assemble WBMCStepInput
  -> WBMC solve
  -> command adapter
  -> actuator interface

여기서 가장 중요한 원칙은 4개야.

원칙 A. TSID는 primitive provider

TSID task/contact/HQP solver는 계속 써도 돼.
하지만 **WBMC semantics의 owner는 TSID가 아니라 너의 WBMC**여야 해.

즉 TSID 객체는 직접적인 제어 철학이 아니라:

Jacobian / drift / contact basis / friction cone
generic task/contact primitive
generic HQP numerical backend

만 제공.

원칙 B. state machine은 behavior scheduler

FSM은

어떤 state가 active인지
어떤 reference를 갱신할지
어떤 task/contact를 켤지

까지만 알아야 해.

FSM이 알면 안 되는 것:

physics vs operational vs bias hierarchy 의미
strict-priority semantics
solver 내부 level policy
hard/soft 의미론 재정의
원칙 C. registry는 binding layer

WBMCRegistry는 아주 유용한 migration shim인데, 최종형에서도 살아남으려면 역할이 분명해야 해.

registry의 일:

task/contact primitive 보관
role 기반 분류
active set filtering
snapshot 변환
nominal/bounds/external wrench 같은 runtime context를 WBMCStepInput으로 assemble

registry가 하면 안 되는 일:

solver 논리 자체 결정
bias 생성
state transition 결정
hierarchy semantics 재정의
원칙 D. solver output과 actuator command는 분리

WBMC의 본질적인 출력은:

delta_qddot
qddot
tau
lambda

이거지, q, qdot가 아니야.

따라서 장기적으로는

WBMC = optimization result producer
CommandAdapter = actuator-friendly command generator

로 나누는 게 맞아.

최종 디렉토리/모듈 목표

현재 구조를 완전히 갈아엎을 필요는 없지만, 개념상 아래처럼 수렴시키는 게 좋아.

wbc_core/
  include/wbc_core/
    wbmc/
      wbmc.hpp
      wbmc-step-input.hpp
      wbmc-solution.hpp
      wbmc-hierarchy-policy.hpp
      blocks/
      bias/
      nominal/
    adapters/
      command-adapter.hpp
    runtime/
      compiled_config.hpp
      runtime-assembler.hpp
      config_loader.hpp
      runtime_config.hpp
    architecture/
      control_architecture.hpp
      fsm_handler.hpp
      state_machine.hpp
      states/
    primitives/
      tasks/
      contacts/
      robots/
      solvers/

지금 당장 폴더를 이렇게 다 바꾸라는 뜻은 아니고, 책임 분리 방향을 저렇게 잡으라는 뜻이야.

2. 단계별 실행 플랜
Phase 0. Semantic freeze

먼저 해야 할 건 “이 컨트롤러가 무엇인가”를 얼려버리는 거야.

목표

WBMC의 의미론을 더 이상 흔들리지 않게 고정.

해야 할 일
WBMC.md를 단일 source of truth로 유지
아래를 명시적으로 고정
decision variable = [delta_qddot, lambda]
qddot = qddot_nom + delta_qddot
Level 0 = Physics
Level 1 = Operational
Level 2 = Bias
Level 3 = Regularization
support contact force는 outcome임을 고정
object dynamics / dexterous force optimization은 v1 non-goal 유지
완료 조건
새 코드 추가 시 이 정의와 충돌하는 API/주석/이름이 없을 것
Phase 1. CompiledConfig / assembly 분리

현재 ConfigCompiler가 너무 많은 걸 하고 있으니 제일 먼저 여기부터 정리.

현재 문제

ConfigCompiler가 지금 대충 다 함:

YAML resolve
task/contact parsing
state machine parsing
constraints parsing
controller gains parsing
runtime object assembly
FSM initialization

이러면 future extension이 들어올수록 폭발함.

목표 구조
1) ConfigLoader

역할:

YAML 로드
external file resolve
path normalize
2) ConfigCompiler

역할:

YAML -> plain CompiledConfig
task/contact/state/global constraint/controller params만 정적 기술

여기서는 TSID 객체 생성 금지.

3) RuntimeAssembler

역할:

compiled config + robot + shared services를 사용해서
task/contact 객체 생성
registry 등록
FSM 생성
nominal provider binding
torque bounds binding
왜 중요하냐

이렇게 해야 나중에

YAML schema 교체
JSON/TOML 지원
planner-generated compiled config
test fixture 기반 synthetic runtime

가 쉬워져.

완료 조건
ConfigCompiler가 compile 단계 전용으로 유지됨
compiled config 단계와 runtime assembly 단계가 분리됨
parsing unit test와 assembly unit test가 독립적으로 존재
Phase 2. State semantics 정리
현재 문제

StateConfig 안에 task_names, task_weights, task_priorities가 있는데, WBMC에서는 task_priorities가 거의 의미론적으로 위험해.

또 YAML 키가 task_hierarchy라서 state가 solver hierarchy를 직접 정의하는 느낌을 줌.

목표

state는 strict-priority hierarchy를 정의하지 말고, active set selection만 하게 만든다.

바꿔야 할 것

task_hierarchy -> 다음 중 하나로 변경 추천:

active_tasks
task_selection
tasks

그리고 state config는 이런 느낌으로 단순화:

states:
  - id: 1
    name: home
    tasks:
      - name: com_task
        weight: 10.0
      - name: ee_task
        weight: 100.0
    contacts:
      - name: left_foot
      - name: right_foot
제거/축소 추천
task_priorities
state별 solver hierarchy override

priority 개념이 정말 필요하면 오직 operational layer 내부 ordering 같은 아주 제한된 문맥에서만 별도 필드로.

완료 조건
state는 active task/contact와 state params만 가짐
solver hierarchy는 WBMCHierarchyPolicy 또는 fixed policy가 소유
Phase 3. Registry 역할 고정
현재 문제

WBMCRegistry는 지금 매우 유용하지만, 방심하면 “작은 control architecture”가 되기 쉬워.

목표

registry를 binding + snapshot assembly layer로 고정.

허용 기능
task/contact pool 등록
motion role (operational, bias) 분류
active set filtering
task weight override
contact selection
nominal provider hookup
torque bounds hookup
external wrench hookup
WBMCStepInput 생성
금지 기능
state transition logic
task reference generator 로직 자체
predictive bias 생성
planner logic
solver hierarchy edit
control mode semantics redefinition
추천 추가

registry 자체에 comment/doc로 contract 명시:

// WBMCRegistry is a runtime binding and snapshot assembly layer.
// It does not own solver semantics, bias generation, or state transitions.
완료 조건
registry header와 tests에서 이 contract가 분명함
registry가 reference generation logic까지 먹지 않음
Phase 4. Command adapter 분리
현재 문제

ControlArchitecture::Step()에서 solve 후 바로

tau
적분된 qdot
적분된 q

를 같이 만들고 있음.

이건 practical하지만 architecture purity 관점에서 애매해.

목표

ControlArchitecture는 WBMCSolution까지만 책임지고, actuator-facing command는 별도 adapter가 생성.

추천 구조
struct LowLevelCommand {
  Vector tau;
  Vector q;
  Vector qdot;
};

class CommandAdapter {
 public:
  LowLevelCommand fromSolution(
      const WBMCSolution& sol,
      const RobotState& state,
      double dt);
};
장점
pure torque mode / hybrid mode / fallback mode 분리 쉬움
future actuator interface 확장 쉬움
solver unit test와 command generation unit test 분리 가능
완료 조건
ControlArchitecture에서 직접 q, qdot 적분하지 않음
command shaping test가 따로 생김
Phase 5. Nominal / Bias pipeline 강화
현재 상태

이미 NominalAccelerationProvider와 BiasReference 구조가 있어서 굉장히 좋음.

목표

WBMC는 nominal/bias를 소비만 하고, 생성하지는 않게 유지.

추천 확장
ZeroNominalProvider
PostureNominalProvider
PlannerNominalProvider
StateDependentNominalMux
PredictiveBiasProvider는 나중에 optional
핵심 원칙
nominal은 “예상/선호 acceleration”
bias는 “remaining subspace selector”
operational task는 “must do now”

이 3개를 섞지 않는다.

완료 조건
posture task와 nominal source의 역할이 문서/테스트에서 분명
“posture bias가 operational처럼 쓰이는” config가 금지되거나 경고됨
Phase 6. Primitive layer와 semantic layer 분리
현재 문제

repo 안에 아직 TSID-style general library와 final WBMC core가 함께 살아 있음.

목표

최소한 개념적으로 두 층을 확실히 분리.

Primitive layer
robot wrapper
TSID task/contact types
generic constraint classes
generic HQP solver implementations
WBMC semantic layer
WBMCStepInput
WBMC
WBMCSolution
hierarchy policy
registry/adapters
bias/nominal interfaces
완료 조건
include dependency가 한 방향으로만 흐름
primitives -> adapters -> wbmc/runtime -> architecture
반대 방향 include 없음
3. 구현 체크리스트

아래는 실제로 PR 단위로 쪼개기 좋은 체크리스트야.

A. Semantic contract
 [x] WBMC.md와 실제 API가 일치한다
 [x] WBMCStepInput이 solver semantics의 단일 입력 계약이다
 [x] WBMCSolution이 solver output의 단일 truth다
 [x] decision variable에 tau가 다시 들어가지 않는다
 [x] qddot_nom 없는 경우 zero fallback이 유지된다
 [x] hierarchy가 strict ordered 아니면 안전 실패한다
B. Runtime/config split
 [x] YAML loading/resolution이 별도 모듈로 분리된다
 [x] YAML -> spec compile 단계가 별도 모듈로 분리된다
 [x] spec -> runtime object assembly가 별도 모듈로 분리된다
 [x] parsing code가 TSID 객체 생성을 직접 하지 않는다
 [x] assembly code가 parsing 정책을 다시 해석하지 않는다
 [x] ConfigCompiler가 thin facade가 되거나 제거된다
C. State machine boundaries
 [x] state YAML에서 task_hierarchy 이름을 제거/alias 처리한다
 [x] state는 active tasks/contacts만 정의한다
 [x] state가 solver hierarchy level을 직접 정하지 않는다
 [x] task_priorities를 제거하거나 deprecated 처리한다
 [x] state transition과 task reference update 책임이 분리된다
 [x] state code가 WBMC 내부 semantics를 직접 호출하지 않는다
D. Registry boundaries
 [x] registry가 task/contact binding만 담당한다
 [x] registry가 snapshot assembly만 담당한다
 [x] registry가 bias generation을 하지 않는다
 [x] registry가 planner logic을 갖지 않는다
 [x] active task filtering이 role semantics를 깨지 않는다
 [x] missing task/contact name에 대해 deterministic error behavior가 있다
E. Solver/core
 [x] physics level이 hard feasibility만 포함한다
 [x] operational level이 직접 motion objectives만 포함한다
 [x] bias level이 solution selection only임이 유지된다
 [x] regularization level이 항상 최하위다
 [x] torque limit가 hard inequality다
 [x] contact-free cycle이 자연스럽게 동작한다
 [x] support contact consistency residual이 작다
 [x] bias가 operational task를 깨지 않는다
F. Command/output path
 [x] WBMCSolution과 actuator command를 분리한다
 [x] command adapter가 별도 클래스/모듈이 된다
 [x] torque-only mode를 지원한다
 [x] position/velocity helper command는 optional이다
 [x] solve failure시 safe fallback이 명확하다
 [x] 이전 명령 hold 정책이 문서화된다
G. Naming / API hygiene
 [x] task_hierarchy naming 정리
 [x] posture_task vs bias semantics가 명확하다
 [x] operational_task naming이 코드 전반에서 일관적이다
 [x] TSID legacy naming이 WBMC semantics를 오염시키지 않는다
 [x] deprecated alias는 명확히 주석 처리된다
 [x] public API에서 “old WBIC interpretation” 흔적이 줄어든다
4. 테스트 플랜

이제 제일 중요한 부분.
지금 테스트는 semantic solver test가 이미 꽤 좋다.
하지만 앞으로는 세 층으로 나눠서 봐야 해.

Layer 1. Solver semantic tests

이건 지금 있는 test-wbc-solver-semantics.cpp 계열을 더 강화하는 거야.

목적

WBMC 수학 의미가 안 깨졌는지 확인.

반드시 있어야 할 테스트
1) Nominal effect
zero nominal이면 baseline correction 동작
좋은 nominal이면 ||delta_qddot|| 감소
나쁜 nominal이어도 feasibility로 복귀
2) Operational dominates bias
operational tracking 정확도 유지
bias가 operational residual을 키우지 않음
redundant axis에서만 bias가 작동
3) Physics dominates all
torque bounds against aggressive bias
contact consistency against conflicting operational request
friction constraint against impossible load
4) Contact mode transitions
contact on/off 시 dimension change 안전
lambda dimension zero/nonzero transition 안전
contact-free mode에서 unnecessary block skip
5) Failure safety
invalid hierarchy
dimension mismatch
impossible constraints
solver fail 시 returned solution reset/hold behavior
pass criteria
모든 residual/inequality 위반이 tolerance 내
failure case에서 garbage output 없음
Layer 2. Architecture boundary tests

이건 지금 가장 부족한 부분이야.

목적

config/runtime/fsm/registry/solver 경계가 깨지지 않는지 확인.

테스트 묶음
A. YAML parsing tests
valid minimal config parse
missing required key throws
unknown task/contact type throws
deprecated field alias accepted with warning
malformed state config deterministic fail
B. Spec compilation tests
YAML의 task/contact/state count가 spec count와 일치
state의 active task names가 canonicalized됨
unknown reference name이 compile time에 잡힘
C. Runtime assembly tests
spec -> task/contact objects 생성 성공
role classification 정확
torque bounds hook-up 정확
nominal provider injection 정확
D. FSM boundary tests
FSM이 state transition만 결정
solver hierarchy 변경 권한 없음
state 변경 시 active tasks/contacts만 바뀜
E. Registry tests
operational task만 operational vector로 감
bias task만 bias vector로 감
weight override 반영 정확
missing active task name 처리 명확
active contact filtering 정확
pass criteria
잘못된 config는 가능한 한 compile/assembly 단계에서 빨리 실패
runtime solve 단계까지 늦게 터지는 config가 줄어듦
Layer 3. Integration tests
목적

실제 controller pipeline이 정상 동작하는지 보기.

추천 시나리오
Scenario 1. Fixed-base minimal robot
1~2 DoF toy robot
posture bias + one operational task
no contact
expected qddot/tau analytically 검증 가능

이건 가장 중요해.
Romeo 같은 큰 모델보다 먼저 toy model integration test가 있어야 디버깅이 편해.

Scenario 2. Floating-base with support contact
simple floating-base test robot
single support contact
base stabilization / simple EE task
contact consistency + torque recovery 검증
Scenario 3. FSM transition scenario
initialize -> home -> teleop
task set 전환
contact set 유지/변경
solve 성공 지속 확인
Scenario 4. Bad config rejection
잘못된 task name
잘못된 contact name
duplicate names
invalid hierarchy
compile/assembly 단계에서 fail 확인
Scenario 5. Command adapter
WBMCSolution에서 torque-only command
optional q/qdot integration command
solve failure hold behavior
pass criteria
한 cycle만이 아니라 여러 cycle 연속 실행에도 안정
state transition 시 crash / dimension mismatch / stale pointer 없음
5. 추천 테스트 매트릭스

이건 네가 CI에 넣기 좋은 형태야.

축 1: Robot type
fixed-base toy
floating-base toy
full humanoid model
축 2: Contact mode
no contact
single contact
multi-contact
축 3: Task mix
operational only
operational + bias
bias only
impossible operational
축 4: Nominal
zero nominal
good nominal
bad nominal
축 5: Runtime path
direct step input
registry assembled input
full YAML + FSM + control architecture

이렇게 교차하면 꽤 탄탄해져.

6. “제대로 구현됐는지” 판정 기준

이건 아주 중요해서 따로 적을게.

아키텍처적으로 성공

다음 질문에 모두 “예”여야 해.

Q1.

WBMC solve에 들어가는 모든 정보가 WBMCStepInput으로 설명 가능한가?
예여야 함.

Q2.

state machine을 갈아껴도 solver semantics가 안 바뀌는가?
예여야 함.

Q3.

TSID task/contact 구현을 바꿔도 WBMC의 level semantics는 유지되는가?
예여야 함.

Q4.

bias를 꺼도 controller가 정상 동작하는가?
예여야 함.

Q5.

nominal provider를 다른 걸로 갈아껴도 solver core는 안 바뀌는가?
예여야 함.

Q6.

contact가 0개일 때도 특수 hack 없이 자연히 동작하는가?
예여야 함.

Q7.

solve 결과를 actuator command로 바꾸는 계층이 solver 밖에 있는가?
최종적으로 예여야 함.

수학적으로 성공
Q1.

physics residual이 tolerance 안인가

Q2.

operational task residual이 bias 유무와 무관하게 유지되는가

Q3.

torque bounds 위반이 없는가

Q4.

contact consistency residual이 작은가

Q5.

bad nominal이어도 feasible correction으로 복귀하는가

소프트웨어적으로 성공
Q1.

잘못된 YAML이 초기에 fail 하는가

Q2.

duplicate name / missing reference가 deterministic하게 잡히는가

Q3.

FSM/state/ref update와 solver code가 독립 테스트 가능한가

Q4.

새 task/contact type 추가 시 config parser, registry, solver를 모두 뜯지 않아도 되는가

7. 추천 구현 순서

실제 작업 순서는 이렇게 가는 게 제일 덜 아프다.

PR 1

task_hierarchy -> tasks 또는 active_tasks rename
task_priorities deprecated
관련 parser/test 수정

PR 2

ConfigCompiler 분해
ConfigLoader / ConfigCompiler / RuntimeAssembler 도입

PR 3

CommandAdapter 도입
ControlArchitecture에서 direct q/qdot integration 제거

PR 4

registry contract 문서화 + registry boundary tests 추가

PR 5

toy model integration tests 추가
full architecture tests를 Romeo 의존에서 일부 분리

PR 6

public include 구조 정리
primitive vs WBMC semantic dependency 정리

8. 개인적으로 가장 먼저 고칠 3개

내가 네 코드베이스 owner라면 바로 이 3개부터 한다.

첫째

task_hierarchy naming 제거
이건 작아 보여도 의미론 오염이 커.

둘째

ConfigCompiler 분리
이건 미래 복잡도를 결정하는 진짜 핵심.

셋째

ControlArchitecture의 command shaping 분리
solver purity와 actuator integration을 나누는 순간 구조가 훨씬 선명해짐.

9. 구현 감사 보고 (2026-04-14)

검증 방식
- 코드 정합성 점검: 핵심 파일 grep/수동 확인
- 테스트 검증: `test_formulation_wbmc`, `test_architecture`, `test_wbc_solver_semantics`

요약
- 체크리스트 A~G는 현재 코드 기준으로 구현/검증 완료 상태로 판단.
- 2026-04-14 기준 회귀 테스트 3/3 pass.

근거 스냅샷
- A. Semantic contract
  - `WBMCStepInput`/`WBMCSolution` 단일 입출력 계약: `include/wbc_core/controller/wbmc-step-input.hpp`, `include/wbc_core/controller/wbmc-solution.hpp`
  - `qddot_nom` fallback + strict hierarchy fail-safe + `tau` 복원: `src/controller/wbmc.cpp`
- B. Runtime/config split
  - `ConfigLoader`/`ConfigCompiler`/`RuntimeAssembler` 분리:
    - `include/wbc_core/runtime/config_loader.hpp`
    - `include/wbc_core/runtime/config_compiler.hpp`
    - `include/wbc_core/runtime/config_validator.hpp`
    - `include/wbc_core/runtime/compiled_config.hpp`
    - `include/wbc_core/runtime/runtime_assembler.hpp`
- C. State machine boundaries
  - 상태별 solver hierarchy override 거부:
    - `src/runtime/config_compiler.cpp`
  - state unknown task/contact 조기 실패:
    - `src/runtime/runtime_assembler.cpp`
  - 관련 테스트:
    - `test/test-architecture.cpp` (`ConfigCompiler_RejectsStateHierarchyOverride`, `ConfigCompiler_StateUnknown*Throws`)
- D. Registry boundaries
  - registry 계약 주석 + deterministic unknown-name 에러:
    - `include/wbc_core/controller/wbmc-registry.hpp`
  - 역할 보존 테스트:
    - `test/test-wbc-solver-semantics.cpp` (`RegistryPreservesOperationalAndBiasRoleSemantics`)
- E. Solver/core
  - 물리/운용/bias/regularization strict level 분리:
    - `src/controller/wbmc.cpp`
  - contact optional activation/regularization 조건부:
    - `src/controller/wbmc.cpp`
  - 관련 테스트:
    - `test/test-wbc-solver-semantics.cpp` (`ContactConsistency*`, `TorqueBounds*`, `RegularizationNeverOverrides*`)
- F. Command/output path
  - `CommandAdapter` 분리 + torque-only 모드:
    - `include/wbc_core/adapters/command-adapter.hpp`
  - solve/adapter 실패 시 이전 명령 hold:
    - `src/architecture/control_architecture.cpp`
  - 관련 테스트:
    - `test/test-architecture.cpp` (`CommandAdapter_TorqueOnlyMode*`)
- G. Naming/API hygiene
  - preferred naming: `bias_task`, `tasks`
  - deprecated alias 유지: `posture_task`, `task_hierarchy`
  - 근거:
    - `src/runtime/runtime_assembler.cpp`
    - `src/runtime/config_compiler.cpp`
    - `controller/optimo_controller/config/task_list.yaml`
    - `controller/optimo_controller/config/state_machine.yaml`

남은 리스크/후속(체크리스트 외)
- 문서 4~7장의 장기 플랜 항목(예: toy model integration 확대, include dependency one-way 정적 검증 자동화)은 별도 작업으로 남아 있음.

9.1 2차 리뷰 반영 결과 (2026-04-14)

이번 보강에서 실제 반영된 항목
- CompiledConfig pure-typed 전환
  - `TaskSpec { YAML::Node }`, `ContactSpec { YAML::Node }` 제거
  - 명시적 typed 필드로 전환:
    - `include/wbc_core/runtime/compiled_config.hpp`
  - compiler 단계에서 typed compiled config 생성:
    - `src/runtime/config_compiler.cpp`
- StateConfig legacy 축소
  - `task_priorities` 제거
  - canonical task name heuristic 필드(`ee_pos_name` 등) 제거
  - 관련 로직 제거:
    - `include/wbc_core/runtime/runtime_config.hpp`
    - `src/runtime/runtime_assembler.cpp`
- ForceTask silent skip 제거
  - WBMC v1에서 `ForceTask` 입력 시 explicit 예외 처리:
    - `src/runtime/runtime_assembler.cpp`
- CommandAdapter 계약 명확화
  - mode별 동작/실패 시 contract 주석 강화:
    - `include/wbc_core/adapters/command-adapter.hpp`

추가 테스트 근거
- `ConfigCompiler_ParseMinimal` (typed field 검증 강화)
- `RuntimeAssembler_UnsupportedForceTaskThrows`
- 기존 경계 테스트 유지:
  - state hierarchy override reject
  - unknown task/contact deterministic throw

아직 남은 핵심 리스크 (체크리스트 외)
- `RuntimeAssembler::InitializeFsm()`가 여전히 다기능(배선/제약/FSM 생성) 집중
- `WBMCRegistry` 구현이 header에 크게 존재(유지보수/컴파일 비용 리스크)
- `include/wbc_core` vs `include/tsid` 이중 public include tree 정리 필요
