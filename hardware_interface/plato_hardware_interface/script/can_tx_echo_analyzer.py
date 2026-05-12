#!/usr/bin/env python3
"""
Monitor PCAN TX timing using echo frames and print per-ID frequency/jitter stats.

Typical use:
  1) Start hardware bringup in another terminal.
  2) Run this script:
       python3 can_tx_echo_analyzer.py --duration 10
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


# PCANBasic constants (from /usr/include/PCANBasic.h)
PCAN_USBBUS1 = 0x51
PCAN_BAUD_1M = 0x0014
PCAN_NONEBUS = 0x00
PCAN_ERROR_OK = 0x00000
PCAN_ERROR_QRCVEMPTY = 0x00020
PCAN_MESSAGE_ECHO = 0x20
PCAN_ALLOW_ECHO_FRAMES = 0x2C


class TPCANMsg(ctypes.Structure):
    _fields_ = [
        ("ID", ctypes.c_uint32),
        ("MSGTYPE", ctypes.c_ubyte),
        ("LEN", ctypes.c_ubyte),
        ("DATA", ctypes.c_ubyte * 8),
    ]


class TPCANTimestamp(ctypes.Structure):
    _fields_ = [
        ("millis", ctypes.c_uint32),
        ("millis_overflow", ctypes.c_uint16),
        ("micros", ctypes.c_uint16),
    ]


@dataclass
class FrameEvent:
    t: float
    can_id: int
    opcode: int


def _format_error(lib: ctypes.CDLL, status: int) -> str:
    buf = ctypes.create_string_buffer(256)
    # 0x09 = English
    if lib.CAN_GetErrorText(status, 0x09, buf) == PCAN_ERROR_OK:
        return buf.value.decode(errors="replace")
    return f"PCAN error 0x{status:X}"


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    idx = int(round((len(values) - 1) * q))
    return sorted(values)[idx]


def _print_stats(
    events: List[FrameEvent],
    expected_period_ms: float,
    burst_ratio: float,
) -> None:
    per_key: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for ev in events:
        per_key[(ev.can_id, ev.opcode)].append(ev.t)

    if not per_key:
        print("No matching TX echo frames captured.")
        return

    burst_threshold = (expected_period_ms / 1000.0) * burst_ratio
    print("\n=== TX Echo Timing Stats ===")
    print(f"Expected period: {expected_period_ms:.3f} ms, burst threshold: < {burst_threshold*1000.0:.3f} ms")
    print("ID/opcode         count   avg_hz   dt_min(ms) dt_p50(ms) dt_p95(ms) dt_max(ms) burst_cnt")

    for (can_id, opcode), ts in sorted(per_key.items()):
        ts.sort()
        if len(ts) < 2:
            avg_hz = 0.0
            dt = []
        else:
            dt = [b - a for a, b in zip(ts[:-1], ts[1:])]
            duration = ts[-1] - ts[0]
            avg_hz = (len(ts) - 1) / duration if duration > 0 else 0.0

        if dt:
            dt_ms = [x * 1000.0 for x in dt]
            burst_cnt = sum(1 for x in dt if x < burst_threshold)
            print(
                f"0x{can_id:02X}/0x{opcode:02X}     "
                f"{len(ts):5d}   {avg_hz:7.2f}   "
                f"{min(dt_ms):9.3f} {statistics.median(dt_ms):9.3f} "
                f"{_percentile(dt_ms, 0.95):9.3f} {max(dt_ms):9.3f} "
                f"{burst_cnt:9d}"
            )
        else:
            print(
                f"0x{can_id:02X}/0x{opcode:02X}     "
                f"{len(ts):5d}   {avg_hz:7.2f}   "
                f"{float('nan'):9.3f} {float('nan'):9.3f} "
                f"{float('nan'):9.3f} {float('nan'):9.3f} "
                f"{0:9d}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze PCAN TX timing via echo frames.")
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds.")
    parser.add_argument(
        "--ids",
        default="11,12,13,14,15,16,17,18",
        help="Comma-separated TX IDs in hex or decimal (default: 11..18 hex).",
    )
    parser.add_argument(
        "--opcodes",
        default="93,95",
        help="Comma-separated opcode bytes in hex or decimal (default: 0x93,0x95).",
    )
    parser.add_argument(
        "--expected-period-ms",
        type=float,
        default=10.0,
        help="Expected command period per actuator (ms).",
    )
    parser.add_argument(
        "--burst-ratio",
        type=float,
        default=0.6,
        help="Burst threshold ratio. interval < expected_period*ratio counted as burst.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate periodic TX in this script and measure echo timing.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=100.0,
        help="Per-ID TX frequency used in --generate mode.",
    )
    parser.add_argument(
        "--tx-opcode",
        default="0x93",
        help="Opcode byte to send in --generate mode (default: 0x93).",
    )
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Call CAN_Uninitialize(PCAN_NONEBUS) before init. Use only when channel is stale.",
    )
    args = parser.parse_args()

    try:
        ids = {int(x.strip(), 0) for x in args.ids.split(",") if x.strip()}
        opcodes = {int(x.strip(), 0) for x in args.opcodes.split(",") if x.strip()}
    except ValueError as exc:
        print(f"Invalid --ids/--opcodes: {exc}", file=sys.stderr)
        return 2

    lib = ctypes.CDLL("libpcanbasic.so")
    lib.CAN_Initialize.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
    lib.CAN_Initialize.restype = ctypes.c_uint32
    lib.CAN_Uninitialize.argtypes = [ctypes.c_uint16]
    lib.CAN_Uninitialize.restype = ctypes.c_uint32
    lib.CAN_SetValue.argtypes = [ctypes.c_uint16, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32]
    lib.CAN_SetValue.restype = ctypes.c_uint32
    lib.CAN_Read.argtypes = [ctypes.c_uint16, ctypes.POINTER(TPCANMsg), ctypes.POINTER(TPCANTimestamp)]
    lib.CAN_Read.restype = ctypes.c_uint32
    lib.CAN_Write.argtypes = [ctypes.c_uint16, ctypes.POINTER(TPCANMsg)]
    lib.CAN_Write.restype = ctypes.c_uint32
    lib.CAN_GetErrorText.argtypes = [ctypes.c_uint32, ctypes.c_uint16, ctypes.c_char_p]
    lib.CAN_GetErrorText.restype = ctypes.c_uint32

    if args.force_reset:
        lib.CAN_Uninitialize(PCAN_NONEBUS)
    st = lib.CAN_Initialize(PCAN_USBBUS1, PCAN_BAUD_1M)
    if st != PCAN_ERROR_OK:
        print(
            f"CAN_Initialize failed: {_format_error(lib, st)}\n"
            "Hint: another process may already own USBBUS1. Stop bringup or run with --generate only "
            "while channel is free.",
            file=sys.stderr,
        )
        return 1

    try:
        echo_on = ctypes.c_uint32(1)
        st = lib.CAN_SetValue(
            PCAN_USBBUS1,
            PCAN_ALLOW_ECHO_FRAMES,
            ctypes.byref(echo_on),
            ctypes.sizeof(echo_on),
        )
        if st != PCAN_ERROR_OK:
            print(f"Warning: failed to enable echo frames: {_format_error(lib, st)}")

        # Drain queue before capture.
        msg = TPCANMsg()
        ts = TPCANTimestamp()
        while lib.CAN_Read(PCAN_USBBUS1, ctypes.byref(msg), ctypes.byref(ts)) == PCAN_ERROR_OK:
            pass

        tx_opcode = int(args.tx_opcode, 0) & 0xFF
        print("Capturing TX echo frames...")
        print(f"IDs: {[hex(x) for x in sorted(ids)]}, opcodes: {[hex(x) for x in sorted(opcodes)]}")
        print(f"Duration: {args.duration:.2f}s")
        if args.generate:
            print(f"Generate mode: enabled, target_hz={args.target_hz:.2f}, tx_opcode=0x{tx_opcode:02X}")

        events: List[FrameEvent] = []
        t_end = time.monotonic() + args.duration
        period = 1.0 / args.target_hz if args.target_hz > 0 else 0.0
        next_tx: Dict[int, float] = {can_id: time.monotonic() for can_id in ids}
        tx_sent: Dict[int, int] = defaultdict(int)

        while time.monotonic() < t_end:
            now = time.monotonic()

            if args.generate and period > 0.0:
                for can_id in sorted(ids):
                    if now < next_tx[can_id]:
                        continue
                    tx = TPCANMsg()
                    tx.ID = can_id
                    tx.MSGTYPE = 0x00
                    tx.LEN = 8
                    tx.DATA[0] = tx_opcode
                    for i in range(1, 8):
                        tx.DATA[i] = 0

                    wst = lib.CAN_Write(PCAN_USBBUS1, ctypes.byref(tx))
                    if wst == PCAN_ERROR_OK:
                        tx_sent[can_id] += 1
                    next_tx[can_id] += period
                    if next_tx[can_id] < now - period:
                        next_tx[can_id] = now + period

            had_frame = False
            while True:
                msg = TPCANMsg()
                ts = TPCANTimestamp()
                rst = lib.CAN_Read(PCAN_USBBUS1, ctypes.byref(msg), ctypes.byref(ts))
                if rst == PCAN_ERROR_OK:
                    had_frame = True
                    is_echo = (msg.MSGTYPE & PCAN_MESSAGE_ECHO) != 0
                    if not is_echo or msg.LEN < 1:
                        continue
                    can_id = int(msg.ID)
                    opcode = int(msg.DATA[0])
                    if can_id in ids and opcode in opcodes:
                        events.append(FrameEvent(t=time.monotonic(), can_id=can_id, opcode=opcode))
                    continue
                if rst == PCAN_ERROR_QRCVEMPTY:
                    break
                print(f"CAN_Read error: {_format_error(lib, rst)}", file=sys.stderr)
                return 1

            if not had_frame:
                time.sleep(0.0005)

        _print_stats(
            events=events,
            expected_period_ms=args.expected_period_ms,
            burst_ratio=args.burst_ratio,
        )
        if args.generate:
            print("\nGenerated TX count by ID:")
            for can_id in sorted(ids):
                print(f"  ID 0x{can_id:02X}: {tx_sent[can_id]}")
    finally:
        lib.CAN_Uninitialize(PCAN_USBBUS1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
