# Plato Hardware Interface Refactor Plan

## Goal

Restructure Plato so the Plato-specific architecture is conceptually only:

1. `PlatoCANProtocol`
2. `PlatoHardwareInterface`
3. `PlatoHand`
4. `FiveBarLinkage`

And the reusable pieces shared by Plato and Aristo live in `can_hardware_common`.

The design target is:

- Plato and Aristo differ mainly in CAN protocol behavior and transmission behavior.
- ROS 2 only talks in joint space.
- the hand class coordinates the full pipeline and owns the canonical runtime data through `Robot`.
- transport, conversion, and data ownership are explicit and easy to reason about.
- the final structure should improve readability, simplicity, and robustness, not preserve current file boundaries.

## Core Idea

The reusable abstraction is not "Plato runtime" or "device".

The reusable abstraction is:

- one common `Robot` data model
- one common hand orchestration pattern
- one pluggable CAN protocol
- one pluggable transmission model

That means the architectural pattern for both robots should be:

- `Robot` in `can_hardware_common`
- common hand core in `can_hardware_common`
- robot-specific CAN protocol in each robot package
- robot-specific transmission in each robot package
- robot-specific ROS hardware adapter in each robot package

## Target Package Ownership

### `can_hardware_common`

This package should own only generic reusable pieces:

- `Robot`
- `Robot::JointCommand`
- `Robot::JointState`
- `Robot::ActuatorCommand`
- `Robot::ActuatorState`
- actuator command mode enum
- validity / readiness bookkeeping
- common hand base or hand core
- `CanBusManager`
- optional abstract protocol / transmission concepts if they help reuse

This package should not own:

- Plato CAN ids
- Aristo CAN ids
- five-bar linkage math
- robot-specific actuator limits
- Plato or Aristo startup policies

### `plato_hardware_interface`

This package should own only Plato-specific pieces:

- `PlatoCANProtocol`
- `PlatoHand`
- `PlatoHardwareInterface`
- `FiveBarLinkage`
- Plato-specific actuator config
- Plato-specific linkage config
- hardware YAML files under `plato_hardware_interface/config`
- Plato config loading and validation code
- Plato-specific sign conventions and command policies

### `aristo_hardware_interface`

This package should mirror the same pattern:

- `AristoCANProtocol`
- `AristoHand`
- `AristoHardwareInterface`
- Aristo-specific transmission

### `plato_bringup`

This package should own Plato ROS-facing bringup config:

- ros2_control controller parameter YAML
- controller manager / broadcaster YAML
- Plato joint impedance controller YAML
- launch-time ROS wiring

This package should not own:

- actuator hardware YAML
- linkage geometry YAML
- Plato hardware parsing and validation logic

### `aristo_bringup`

This package should own Aristo ROS-facing bringup config:

- ros2_control controller parameter YAML
- controller manager / broadcaster YAML
- Aristo joint impedance controller YAML
- launch-time ROS wiring

### `joint_impedance_controller`

This package should own the shared joint-space impedance controller used by both Plato and Aristo:

- the `JointImpedanceController` plugin
- the generic controller parameter schema
- the command/state topic contract
- generic command validation and PD/feedforward logic

This package should not own:

- Plato or Aristo joint names
- robot-specific gains
- CAN protocol details
- transmission math
- hardware-specific branching

## Non-ROS Hardware Config

Actuator config and linkage parameters are hardware package data, not ROS parameters.

That means:

- they should live under `plato_hardware_interface/config`
- they should be installed with the package and resolved via `ament_index_cpp`
- they should be loaded once during initialization, not during the control loop
- they should not be represented as ROS node params unless there is a clear runtime tuning need

For Plato specifically, the package-owned hardware config files should be:

- `config/plato_actuators.yaml`
- `config/plato_actuator_offsets.yaml`
- `config/plato_linkage.yaml`

This is separate from ROS-facing controller config, which should live in `plato_bringup/config`,
for example:

- `plato_bringup/config/plato2_joint_impedance_controller.yaml`

That controller YAML is ROS configuration.

The actuator, actuator-offset, and linkage YAML files are hardware description/configuration for the Plato package itself.

`plato_actuators.yaml` and `plato_linkage.yaml` should contain fixed hardware constants.

`plato_actuator_offsets.yaml` is different:

- it is still package-owned hardware config
- but it is expected to be recalibrated over time for zeroing
- runtime zeroing flows should update this file, not rewrite the fixed actuator constants YAML
- the simplest integration path is a ros2_control hardware param exposed as a launch arg such as
  `zeroing:=true`, which requests one-shot software zeroing on startup once full actuator feedback
  becomes available

## Recommended Plato Config Integration

The cleanest design is to parse both hardware YAML files into one aggregate Plato config object before constructing the runtime objects.

Suggested shape:

```cpp
struct PlatoHandConfig {
  std::vector<PlatoActuatorConfig> actuators;
  FiveBarLinkageConfig linkage;
};
```

Recommended ownership:

- `plato_hardware_interface` owns `PlatoHandConfig`
- `plato_hardware_interface` owns the YAML loader for that config
- `PlatoHand` receives a fully parsed `PlatoHandConfig`
- `PlatoHardwareInterface` only triggers config loading during initialization

Preferred loading pattern:

1. resolve the package share directory with `ament_index_cpp`
2. load `config/plato_actuators.yaml`
3. load `config/plato_actuator_offsets.yaml`
4. load `config/plato_linkage.yaml`
5. validate names, sizes, CAN ids, limits, offsets, and linkage fields
6. merge actuator constants with actuator offsets
7. construct one `PlatoHandConfig`
8. pass that config into `PlatoHand`

This keeps file I/O and YAML parsing out of the runtime/control code and gives `PlatoHand` typed config instead of raw file paths.

## Shared Joint Impedance Controller

The same `joint_impedance_controller` plugin should be reusable for both Plato and Aristo.

The controller should remain purely joint-space and hardware-agnostic.

That means:

- it consumes joint-space state interfaces only
- it writes joint-space command interfaces only
- it does not know about actuator ids, CAN frames, or transmission details
- it does not branch on robot type

All robot-specific behavior should be handled by:

- the robot hardware interface
- the robot hand/transmission implementation
- per-robot controller YAML in bringup

## Controller/Hardware Contract

To make one controller work for both robots, both hardware interfaces must export the same joint-space ros2_control contract.

Required state interfaces per joint:

- `position`
- `velocity`
- `effort`

Required command interfaces per joint:

- `position`
- `velocity`
- `effort`
- `stiffness`
- `damping`

This is the key contract boundary.

Above this boundary:

- the controller works only in joint space

Below this boundary:

- Plato and Aristo are free to use different actuator protocols and transmission models

In other words, the hardware layer must absorb robot-specific differences so the controller does not have to.

## Controller Semantics

The controller command model should stay aligned with `Robot::JointCommand`.

Conceptually, the controller operates on:

- desired joint position
- desired joint velocity
- feedforward effort
- joint stiffness
- joint damping

This should match:

- the `ImpedanceCommands` topic payload
- the joint command interfaces exported by the hardware
- the joint-space command representation held by `Robot`

Keeping those three representations structurally aligned is important for readability and for avoiding glue code.

## `compute_impedance_torque` Strategy

The purpose of `compute_impedance_torque` is to select whether the shared controller should add a controller-side impedance torque term on top of the commanded impedance target:

- directly in the motor driver / lower hardware layer
- or in the shared ROS controller by converting the impedance target into a torque command

This should remain a bringup-level choice rather than becoming robot-specific branching inside the controller source.

Meaning:

- if `compute_impedance_torque=true`, the controller computes
  `tau = effort_ff + Kp * pos_error + Kd * vel_error`
  and writes the resulting torque through the joint `effort` command interface
- when `compute_impedance_torque=true`, the controller should still pass down the desired impedance values `position`, `velocity`, `stiffness`, and `damping`
- if `compute_impedance_torque=false`, the controller only passes the target impedance values down to the hardware layer, meaning the lower layer is responsible for interpreting `position`, `velocity`, `effort`, `stiffness`, and `damping`

Per-robot interpretation then becomes:

- Plato needs a torque-producing path because the motor driver exposes torque control rather than native impedance control
- for Plato, `compute_impedance_torque=true` should currently be treated as the required/default mode
- with the current joint command contract, Plato cannot safely infer whether the controller already converted impedance into torque if both `effort` and `stiffness`/`damping` are present
- therefore the current Plato hardware path should treat `effort` as the final torque command and only forward `stiffness`/`damping` as descriptive values unless an explicit mode signal is added
- if true hardware-side impedance is needed for Plato later, the command contract should grow an explicit mode/interface rather than relying on implicit interpretation
- Aristo can treat this as optional, because its lower actuator layer may be able to consume impedance targets more directly
- Aristo can therefore choose either controller-side PD torque or hardware-pass-through impedance depending on actuator protocol semantics and desired control allocation

The important design rule is:

- the controller exposes the same behavior to both robots
- the per-robot mode selection and default live in bringup YAML
- the controller source code does not contain Plato-specific or Aristo-specific branches

## Controller Config Ownership

Because the controller is shared, robot-specific joint lists and gains should live in bringup, not in the controller package.

That means:

- `joint_impedance_controller` owns the generic parameter schema only
- `plato_bringup` owns Plato controller YAML
- `aristo_bringup` owns Aristo controller YAML

Expected bringup-side files:

- `plato_bringup/config/plato_joint_impedance_controller.yaml`
- `aristo_bringup/config/plato2_joint_impedance_controller.yaml`

The Aristo controller file name may still reflect older naming.

That naming cleanup is secondary to the architectural rule: controller package generic, bringup package robot-specific.

## Controller Robustness Expectations

For shared use across Plato and Aristo, the controller/hardware boundary should have explicit startup and validity behavior.

Recommended contract:

- hardware exports `NaN` joint state until real feedback is available
- controller refuses PD torque on joints without finite position/velocity feedback
- hardware does not send zero commands just because the controller has not yet latched a real command
- command interface shape stays identical across Plato and Aristo even if one robot internally ignores or remaps part of it

This keeps the shared controller simple while preserving safe startup behavior on both robots.

## Config Loader Structure

The current separate actuator and linkage loader files are acceptable, but the target architecture should present one higher-level entry point to the rest of the system.

Preferred external API:

```cpp
PlatoHandConfig load_default_plato_hand_config();
PlatoHandConfig load_plato_hand_config(const std::string & config_dir);
```

Internally, that loader can still delegate to smaller helpers:

- `load_plato_actuator_configs(...)`
- `load_plato_linkage_config(...)`

But `PlatoHardwareInterface` and `PlatoHand` should not need to coordinate two independent config-loading paths themselves.

This simplifies initialization and makes the Plato package config boundary explicit.

## Final Plato-Side Class Picture

The Plato package should ideally expose only these four main concepts:

### `PlatoCANProtocol`

Responsibilities:

- encode actuator commands into CAN frames
- decode CAN frames into actuator-space state
- implement Plato-specific actuator command semantics
- own low-level CAN layout and protocol constants

It should not:

- know ROS interfaces
- know five-bar kinematics
- own the canonical `Robot`

### `FiveBarLinkage`

Responsibilities:

- convert actuator space to joint space for the Plato linkage
- convert joint command to actuator command for the linkage-driven joints
- validate linkage state
- own five-bar configuration
- compute transmission effects like position and torque amplification

It should not:

- talk to CAN
- export ROS interfaces

### `PlatoHand`

Responsibilities:

- own `Robot`
- own `CanBusManager`
- own Plato actuator channels or protocol-facing actuator state
- own `PlatoCANProtocol`
- own `FiveBarLinkage`
- receive typed Plato hardware config at construction
- perform the read/write update order
- coordinate enable/disable/reset behavior

This is the main runtime object for Plato.

It should be the place where:

- transport comes in
- actuator-space state is updated
- transmission maps to joint space
- joint commands are mapped back to actuator-space commands
- CAN frames are emitted

So `PlatoHand` replaces the need for separately naming "runtime" and "device" if that naming does not buy clarity.

`PlatoHand` should not discover config files on its own deep inside runtime methods.

It should be given typed, already-validated configuration during construction.

### `PlatoHardwareInterface`

Responsibilities:

- thin ros2_control adapter only
- export command interfaces from `PlatoHand::robot().joint_commands()`
- export state interfaces from `PlatoHand::robot().joint_states()`
- forward lifecycle and read/write calls to `PlatoHand`

It should not:

- know CAN frame formats
- know five-bar math
- know actuator-level conversion details

## Common Reusable Structure

The common layer should make Plato and Aristo look structurally identical.

Preferred generic shape:

```cpp
namespace can_hardware_common {

class Robot;

template<typename Protocol, typename Transmission>
class CanHandBase {
public:
  Robot & robot();
  const Robot & robot() const;

  void enable();
  void disable();
  void read();
  void write();

protected:
  Robot robot_;
  CanBusManager can_bus_manager_;
  Protocol protocol_;
  Transmission transmission_;
};

}  // namespace can_hardware_common
```

Then:

- `PlatoHand` is a thin concrete specialization of the common hand structure
- `AristoHand` is another specialization with different protocol/transmission

If templates turn out to be awkward, the same idea can be implemented with composition instead:

- `PlatoHand` owns a reusable common hand core object
- the common hand core calls protocol/transmission objects

The key point is not template vs composition.

The key point is:

- hand orchestration is shared
- protocol is pluggable
- transmission is pluggable

## Robot Role

`Robot` should not be a thin wrapper.

It should be the canonical runtime data model.

It should own:

- joint-space command buffers
- joint-space state buffers
- actuator-space command buffers
- actuator-space state buffers
- validity state
- command mode state
- helper APIs for reset, invalidate, readiness, and view access

Suggested `Robot` responsibilities:

- `resize(num_joints, num_actuators)`
- `reset_joint_commands_to_nan()`
- `reset_actuator_commands()`
- `invalidate_joint_states()`
- `invalidate_actuator_states()`
- `has_complete_joint_command()`
- `has_complete_actuator_feedback()`
- `joint_commands()`
- `joint_states()`
- `actuator_commands()`
- `actuator_states()`

`Robot` should not:

- know about CAN transport
- know about five-bar math
- know about Plato or Aristo

## Data Model

ROS only interacts with joint space.

Hardware only interacts with actuator space.

So the common data model should explicitly hold both.

### Joint Space

For ROS-facing data:

- position
- velocity
- effort feedforward
- stiffness
- damping

### Actuator Space

For hardware-facing data:

- position
- velocity
- effort
- stiffness
- damping
- command mode

Suggested command modes:

- `disabled`
- `position`
- `torque`
- `impedance`

For Plato:

- thumb roll and yaw actuators are `position`
- MCP/PIP actuators are `torque`

For Aristo:

- likely `impedance` for all actuators

This is exactly why command mode should be part of the common actuator-space command model.

## PlatoHand Update Order

The hand should define the full update pipeline.

### Read Path

1. poll CAN
2. decode incoming frames into actuator-space state
3. write actuator feedback into `robot.actuator_states()`
4. run `FiveBarLinkage` conversion into `robot.joint_states()`
5. ros2_control reads joint-space state

### Write Path

1. ros2_control writes joint commands into `robot.joint_commands()`
2. validate joint command completeness
3. run `FiveBarLinkage` conversion into `robot.actuator_commands()`
4. `PlatoCANProtocol` encodes actuator commands into CAN frames
5. send frames

This is the cleanest mental model:

- `Robot` owns data
- `FiveBarLinkage` owns conversion math
- `PlatoCANProtocol` owns byte-level hardware protocol
- `PlatoHand` owns orchestration

## What To Avoid

Avoid introducing extra named layers unless they materially simplify the design.

That means:

- no separate `PlatoRuntime` if it is just another name for `PlatoHand`
- no separate `PlatoDevice` if it only splits transport from hand orchestration without making reuse clearer
- no separate public `Actuator` abstraction if a lighter protocol/channel representation is clearer

The target is not "more classes".

The target is:

- fewer concepts
- sharper ownership
- easier reasoning

## Actuator Representation

Current `Actuator` code should not constrain the final architecture.

The preferred end state is probably not a heavyweight standalone actuator class per joint.

Better options:

### Option A: Keep small actuator channel objects inside `PlatoHand`

Each channel holds:

- config
- last feedback
- status
- maybe a few helper methods

`PlatoCANProtocol` performs actual encode/decode.

### Option B: Protocol owns stateless encode/decode and `PlatoHand` owns raw channel data

This is even flatter:

- channel structs hold data only
- protocol converts data to and from frames

Preferred direction:

- lean toward flatter channel data + protocol logic
- avoid unnecessary heap-owned protocol objects per actuator
- keep the public architecture focused on the four main Plato-side classes

## ROS 2 Boundary

`PlatoHardwareInterface` should be nearly trivial.

It should:

- create `PlatoHand`
- load Plato hardware YAML config once during initialization
- size the `Robot` buffers
- export pointers into `robot.joint_states()` and `robot.joint_commands()`
- call `hand.read()` and `hand.write()`
- handle lifecycle enable/disable/reset

It should not:

- inspect actuator-space data
- implement transmission
- implement protocol details
- contain raw YAML parsing logic beyond invoking the package loader
- own controller parameter YAML

It must, however, export the full joint command/state interface contract expected by the shared `joint_impedance_controller`.

## Cross-Robot Reuse

The intended reusable pattern is:

- same `Robot`
- same hand orchestration structure
- different protocol
- different transmission
- same joint impedance controller at the ROS layer

So the only robot-specific differences should be:

- CAN protocol / actuator command semantics
- transmission / linkage model
- robot-specific config
- robot-specific controller YAML in bringup

That is the core reuse goal.

## Recommended End State

### Common Layer

In `can_hardware_common`:

- `Robot`
- common hand base or hand core
- `CanBusManager`
- common enums and utility types

### Plato Layer

In `plato_hardware_interface`:

- `PlatoCANProtocol`
- `PlatoHand`
- `PlatoHardwareInterface`
- `FiveBarLinkage`

### Aristo Layer

In `aristo_hardware_interface`:

- `AristoCANProtocol`
- `AristoHand`
- `AristoHardwareInterface`
- Aristo transmission

### Shared Controller Layer

In `joint_impedance_controller`:

- one shared controller plugin
- no robot-specific branching
- one generic parameter schema

## Migration Plan

### Phase 1: Strengthen `Robot` in `can_hardware_common`

- make `Robot` the authoritative holder of joint and actuator spaces
- add command mode and validity support
- keep the API convenient for ROS pointer export

### Phase 2: Introduce common hand orchestration in `can_hardware_common`

- add `CanHandBase` or an equivalent reusable hand core
- make it responsible for the generic read/write order
- keep protocol and transmission pluggable

### Phase 3: Refactor Plato around the four-class model

- add one aggregate Plato hardware config loader for actuator and linkage YAML
- keep hardware YAML in `plato_hardware_interface/config`
- pass typed config into `PlatoHand`
- collapse current hand/runtime/device concerns into `PlatoHand`
- move protocol behavior into `PlatoCANProtocol`
- move all five-bar conversion into `FiveBarLinkage`
- make `PlatoHardwareInterface` thin

### Phase 4: Apply the same pattern to Aristo

- create `AristoCANProtocol`
- plug in Aristo transmission
- reuse the common hand structure and `Robot`

### Phase 5: Standardize the shared controller contract

- keep `joint_impedance_controller` generic and joint-space only
- make both Plato and Aristo export the same joint command/state interfaces
- move robot-specific joint lists and controller gains fully into bringup YAML
- keep robot-specific control-mode selection as configuration, not controller branching

## File Layout Proposal

### `can_hardware_common`

- `include/can_hardware_common/robot.hpp`
- `include/can_hardware_common/can_hand_base.hpp` or equivalent
- `include/can_hardware_common/command_mode.hpp` if needed

### `plato_hardware_interface`

- `config/plato_actuators.yaml`
- `config/plato_actuator_offsets.yaml`
- `config/plato_linkage.yaml`
- `include/plato_hardware_interface/plato_can_protocol.hpp`
- `src/plato_can_protocol.cpp`
- `include/plato_hardware_interface/plato_hand_config.hpp`
- `include/plato_hardware_interface/utils/actuator_config_loader.hpp`
- `src/utils/actuator_config_loader.cpp`
- `include/plato_hardware_interface/utils/actuator_offset_loader.hpp`
- `src/utils/actuator_offset_loader.cpp`
- `include/plato_hardware_interface/utils/linkage_config_loader.hpp`
- `src/utils/linkage_config_loader.cpp`
- `include/plato_hardware_interface/utils/plato_hand_config_loader.hpp`
- `src/utils/plato_hand_config_loader.cpp`
- `include/plato_hardware_interface/plato_hand.hpp`
- `src/plato_hand.cpp`
- `include/plato_hardware_interface/five_bar_linkage.hpp`
- `src/five_bar_linkage.cpp`
- `include/plato_hardware_interface/plato_hardware_interface.hpp`
- `src/plato_hardware_interface.cpp`

### `plato_bringup`

- `config/plato2_joint_impedance_controller.yaml`
- `config/plato_joint_impedance_controller.yaml`
- other ROS controller / broadcaster YAML
- launch files that bind the hardware interface into ROS

### `aristo_bringup`

- `config/plato2_joint_impedance_controller.yaml`
- Aristo bringup launch files and scripts

### `joint_impedance_controller`

- `include/joint_impedance_controller/joint_impedance_controller.hpp`
- `src/joint_impedance_controller.cpp`
- `src/joint_impedance_controller_parameters.yaml`

### `aristo_hardware_interface`

- `include/aristo_hardware_interface/aristo_can_protocol.hpp`
- `src/aristo_can_protocol.cpp`
- `include/aristo_hardware_interface/aristo_hand.hpp`
- `src/aristo_hand.cpp`
- Aristo transmission files
- `include/aristo_hardware_interface/aristo.hpp`
- `src/aristo.cpp`

## Summary

The clean target is:

- common `Robot`
- common hand orchestration
- robot-specific protocol
- robot-specific transmission
- thin ROS adapter

For Plato specifically, the conceptual public architecture should reduce to:

- `PlatoCANProtocol`
- `PlatoHardwareInterface`
- `PlatoHand`
- `FiveBarLinkage`

That is the structure most likely to improve readability, simplicity, reuse, and robustness at the same time.
