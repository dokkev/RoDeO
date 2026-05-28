# Plato2 firmware

## Dynamixel CAN protocol draft

MCU 하나가 여러 Dynamixel servo를 관리한다. CAN ID는 MCU board ID로 쓰고,
payload 첫 바이트로 command를 구분한다.

Command frame:

```text
DATA[0] = command
DATA[1] = servo_id
DATA[2..] = payload
```

Lifecycle command:

```text
ENABLE:  DATA[0] = 0x91, DATA[1] = servo_id
DISABLE: DATA[0] = 0x92, DATA[1] = servo_id
```

Lifecycle response:

```text
DATA[0] = echoed command
DATA[1] = servo_id
DATA[2] = result, 0x00 success, 0x01 failure
```

Position command:

```text
DATA[0] = 0x95
DATA[1] = servo_id
DATA[2..5] = float32 goal position, little-endian, radians
DATA[6..7] = uint16 current limit, little-endian, milliamps
```

Position response:

```text
DATA[0] = 0x95
DATA[1] = servo_id
DATA[2] = result, 0x00 success, 0x01 failure, 0x02 motor disabled
DATA[3..4] = uint16 packed position
DATA[5..6] = packed velocity bits
DATA[6..7] = packed torque bits
```
