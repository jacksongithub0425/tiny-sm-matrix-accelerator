# tiny-sm-matrix-accelerator

A study-scale, SM-inspired compute core on a Zynq-7020, built around a single
Tensor-Core-like INT8 MMA unit (`mma.m8n8k16.s8.s8.s32`). The goal is not to
reproduce a commercial GPU or win a TOPS benchmark. The goal is to implement
and measure the micro-architectural mechanisms that make GPU-style matrix
acceleration efficient: warp scheduling, a banked register file with operand
collection, banked/swizzled shared memory, scoreboard-based issue control, and
CP.ASYNC-inspired global-to-shared tile prefetch.

## What this is

- A single SM with a single MMA pipe targeting Xilinx XC7Z020.
- INT8 multiply, INT32 accumulate, modeled on the Volta/Turing fragment shape.
- Two warps of eight lanes each, scheduled round-robin with a real scoreboard.
- A 32-bit fixed-width ISA, a Python assembler, and hand-written GEMM kernels.
- Measured: achieved GOPS, MMA-pipe utilization, stall reasons, bank conflicts.

## What this is not

- Not CUDA-compatible. The ISA is original; only the *shape* mirrors PTX.
- Not a GPGPU. No virtual memory, no caches, no branch divergence stack in v1.
- Not a multi-SM design. One SM is enough to study the mechanisms honestly.
- Not a peak-performance accelerator. A pure systolic array would beat it on
  GOPS/DSP; that is the wrong metric for this project.

## Current Status

This repository starts as an architectural specification and implementation
plan. The hardware will be implemented incrementally. Features listed as V1
target are planned milestones, not all implemented on day one.

### Milestone Plan

| Milestone | Goal                                                | Status      |
|-----------|-----------------------------------------------------|-------------|
| M0        | README + architecture specification                 | In progress |
| M1        | `mma_8x8x16_int8.sv` datapath + testbench           | Planned     |
| M2        | Python/numpy golden model + randomized tests        | Planned     |
| M3        | 2-warp scheduler + simple instruction issue         | Planned     |
| M4        | Banked register file + operand collector            | Planned     |
| M5        | Banked/swizzled shared memory                       | Planned     |
| M6        | Scoreboard stall tracking                           | Planned     |
| M7        | CP.ASYNC-inspired double-buffered prefetch          | Planned     |
| M8        | PYNQ PS/PL demo                                     | Planned     |
| M9        | Performance/roofline report                         | Planned     |

## Target and scope

| Item            | Value                                            |
|-----------------|--------------------------------------------------|
| Board           | PYNQ-Z1 / Arty Z7-020 (Xilinx XC7Z020)           |
| Datatype        | INT8 x INT8, INT32 accumulate                    |
| MMA fragment    | `m8n8k16.s8.s8.s32` (8x16 * 16x8 -> 8x8 INT32)   |
| Warps per SM    | 2                                                |
| Lanes per warp  | 8                                                |
| MMA pipes       | 1                                                |
| Shared memory   | 16 KB, 8 banks, swizzled                         |
| Register file   | 32 x 32-bit per lane, 4 banks, operand collector |
| Host link       | AXI HP from Zynq PS                              |

Resource budget (planned versus XC7Z020 available):

| Resource | Planned (approx) | Available | Headroom |
|----------|------------------|-----------|----------|
| DSP48E1  | ~80              | 220       | 2.7x     |
| BRAM36   | ~20              | 140       | 7x       |
| LUT      | ~30,000          | 53,200    | 1.7x     |
| FF       | ~20,000          | 106,400   | 5x       |

The MMA unit alone is 64 INT8 MACs = 64 DSPs; the rest of the DSP budget
covers address generation and the cp.async engine. The headroom on BRAM and FF
is intentional and leaves room for a second warp pair or a second MMA pipe in a
follow-up version.

## Architecture overview

```
  PS (ARM Cortex-A9, PYNQ)
        |
        | AXI-Lite (control)   AXI HP (DDR <-> SMEM via cp.async)
        v                       ^
  +-----------------------------+-----------------------------+
  |                       tiny_sm_top                         |
  |                                                           |
  |   +-------------+    +----------------+   +-----------+   |
  |   | fetch /     |--->| warp scheduler |-->| decode /  |   |
  |   | i-buffer    |    | (RR + scbd)    |   | issue     |   |
  |   +-------------+    +----------------+   +-----+-----+   |
  |                                                 |         |
  |                                                 v         |
  |                                       +-------------------+
  |                                       | operand collector |
  |                                       +---------+---------+
  |                                                 |          |
  |     +----------------+    +----------------+    |          |
  |     | regfile_banked |<-->| scoreboard     |<---+          |
  |     | (4 banks)      |    +----------------+    |          |
  |     +-------+--------+                          |          |
  |             |                                   |          |
  |             v                                   v          |
  |     +----------------+              +----------------------+
  |     | mma_8x8x16     |              | ldst_unit            |
  |     | (64 DSP, INT8) |              | (LDS / STS / LDG /   |
  |     +----------------+              |  STG)                |
  |                                     +-----------+----------+
  |                                                 |          |
  |                                                 v          |
  |                                     +----------------------+
  |                                     | smem_banked + swizzle|
  |                                     | (16 KB, 8 banks)     |
  |                                     +-----------+----------+
  |                                                 ^
  |                                                 |
  |                                     +-----------+----------+
  |                                     | cp_async (DDR->SMEM) |
  |                                     +----------------------+
  +-----------------------------------------------------------+
```

Block by block, with the GPU mechanism each one mirrors:

- **fetch / i-buffer.** Per-warp instruction fetch and a small per-warp
  instruction buffer. Mirrors the front-end of a real SM, where fetch is
  decoupled from issue.
- **warp scheduler.** Round-robin across the two warps, with a stall-on-scoreboard
  policy. Holds per-warp PC and per-warp active mask. Greedy-then-oldest is
  listed as a future variant.
- **decode / issue.** Decodes the 32-bit instruction, checks scoreboard, issues
  to the operand collector or the LD/ST unit.
- **operand collector.** Resolves register-file bank conflicts before dispatch
  to the MMA pipe, mirroring the high-level role of operand collection in a
  real SM.
- **regfile_banked.** 32 x 32-bit per lane, 4 banks. Banking is what makes the
  operand collector a non-trivial piece.
- **mma_8x8x16.** The Tensor-Core analogue. 64 INT8 MACs, INT32 accumulate, one
  fragment per issue. See section below.
- **ldst_unit.** Implements `LDS`, `STS`, `LDG`, `STG`. Drives the swizzle on
  the SMEM side.
- **smem_banked + swizzle.** 16 KB shared memory across 8 banks. Swizzle
  function chosen so MMA operand loads are bank-conflict free.
- **cp_async.** DMA engine from DDR (via AXI HP) into SMEM, runs in parallel
  with MMA. This is a CP.ASYNC-inspired prefetch engine. It is not
  PTX-compatible, but it studies the same idea: overlapping global-to-shared
  tile movement with MMA execution.
- **scoreboard.** Tracks in-flight LDS / cp.async / MMA / STG, gates issue.

## The MMA instruction

`MMA D, A, B, C` computes `D = A * B + C` where:

- `A` is a 8x16 INT8 fragment (held distributed across the 8 lanes of a warp).
- `B` is a 16x8 INT8 fragment (likewise).
- `C` and `D` are 8x8 INT32 fragments.

This is the same shape as Turing's `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`.
The first implementation maps the 64 logical INT8 MAC lanes conservatively onto
DSP48E1 resources, prioritizing correctness and timing closure over DSP packing.
DSP packing and more aggressive pipeline optimization are left as future work.
Initial target: fixed-latency pipeline, with latency and throughput reported
after synthesis/timing closure.

## ISA

A 32-bit fixed-width encoding, ten opcodes in v1. Full encoding in
[docs/isa_spec.md](docs/isa_spec.md); the assembler lives at
[asm/asm.py](asm/asm.py).

| Mnemonic   | Meaning                              | Notes                          |
|------------|--------------------------------------|--------------------------------|
| `LDG`      | Load global -> register              | Synchronous, blocking          |
| `STG`      | Store register -> global             | Synchronous                    |
| `CP.ASYNC` | Async copy global -> shared          | Non-blocking, scoreboarded     |
| `LDS`      | Load shared -> register              | Goes through swizzle           |
| `STS`      | Store register -> shared             | Goes through swizzle           |
| `MMA`      | Tensor-core fragment multiply-add    | 8x8x16 INT8 -> INT32           |
| `BAR`      | Warp barrier                         | Also drains cp.async queue     |
| `BRA`      | Branch on per-warp condition         | No divergence stack in v1      |
| `EXIT`     | Retire warp                          | Also raises a host interrupt   |
| `NOP`      | No-op                                | Useful for stall studies       |

## Warp execution model

Two warps, eight lanes each, single PC per warp, one active mask per warp.
The scheduler picks the next warp round-robin and only issues if the
scoreboard says no source is in-flight. `BAR` drains the cp.async queue and
synchronizes the two warps. `BRA` is uniform-only in v1: all eight lanes of a
warp must agree on the branch. The reconvergence stack and per-lane divergence
are listed as roadmap items.

## Memory hierarchy

- **DDR.** Reached through AXI HP from the PS. In the optimized GEMM path,
  `CP.ASYNC` is the primary path for moving A/B tiles from DDR into SMEM.
  `LDG` and `STG` remain available for scalar setup, debug, and result
  movement.
- **Shared memory.** 16 KB, 8 banks, swizzled. The swizzle function is chosen
  to make the access pattern of the MMA operand loads (8 lanes, 16-element row)
  bank-conflict free. The verification suite measures the conflict rate with
  the swizzle disabled, so the win is reported, not asserted.
- **Register file.** 32 x 32-bit per lane, 4 banks. The operand collector
  resolves the bank conflicts that a 3-source instruction (`MMA D,A,B,C`) can
  produce.

## Repository layout

```
tiny-sm-matrix-accelerator/
  README.md
  docs/
    architecture.md         block-level micro-arch reference
    isa_spec.md             bit-level instruction encoding
    microarch_notes.md      design choices, dead ends, why
    verification_plan.md    test strategy, coverage goals
    perf_report.md          measured numbers, roofline, plots
  rtl/
    tiny_sm_top.sv
    fetch_decode.sv
    warp_sched.sv
    regfile_banked.sv
    operand_collector.sv
    mma_8x8x16_int8.sv
    ldst_unit.sv
    smem_banked.sv
    scoreboard.sv
    cp_async.sv
  isa/
    isa_spec.md             (mirror of docs/ for in-place reading)
    opcodes.md
  asm/
    asm.py                  Python assembler
    kernels/
      gemm_128x128.s
      gemm_256x256.s
  tb/
    tb_mma_8x8x16.sv
    tb_smem_banked.sv
    tb_warp_sched.sv
    tb_tiny_sm_top.sv
  golden/
    mma_ref.py              numpy reference for one fragment
    gemm_ref.py             numpy reference for full GEMM
    swizzle_ref.py          reference swizzle for SMEM checks
  sim/
    cocotb_runner.py
    waves/                  GTKWave / Surfer save files
  vivado/
    create_project.tcl
    constraints.xdc
  sw/
    pynq_runner.py          PYNQ notebook host driver
  results/
    sim_logs/
    waveforms/
    utilization_reports/
    roofline/
```

Files that do not yet exist are planned, not promised; this README is the
contract for what each one will hold.

## Build and run

### Simulation

cocotb runs against either Verilator (fast) or Vivado xsim (closer to synth).

```
cd sim
python cocotb_runner.py --sim verilator --tb tb_tiny_sm_top
```

A passing run writes a waveform under `results/waveforms/` and a stall-reason
summary to `results/sim_logs/`.

### Synthesis

```
cd vivado
vivado -mode batch -source create_project.tcl
vivado -mode batch -source build.tcl
```

Outputs land in `results/utilization_reports/` and a bitstream under
`vivado/build/`.

### On board

```
# from a PYNQ-Z1
jupyter notebook sw/pynq_runner.py
```

The runner allocates contiguous buffers for A, B, C, D in DDR, assembles the
chosen kernel, loads the bitstream, kicks off the SM, reads back D, and
compares against `golden/gemm_ref.py`.

## Verification plan

Detailed in [docs/verification_plan.md](docs/verification_plan.md). Top level:

- Per-block directed tests for `mma_8x8x16`, `smem_banked`, `regfile_banked`,
  `warp_sched`, `scoreboard`, `cp_async`.
- Randomized MMA fragment tests against `golden/mma_ref.py`.
- End-to-end GEMM with M=N=K in {64, 128, 256, 512}, with and without
  cp.async, with and without swizzle.
- Scoreboard counters (issued / stalled / stall-reason histogram) checked in
  every end-to-end run; a regression in stall mix fails the test.

## Performance methodology

This is the section the project lives or dies on. Every number reported in
[docs/perf_report.md](docs/perf_report.md) is measured on real hardware (or in
post-synth simulation when on-board measurement is impossible) and compared
against an explicit ceiling.

What is measured:

- Achieved INT8 GOPS versus the DSP-roofline peak at the synthesized clock.
- MMA-pipe utilization (fraction of cycles the MMA issues a fragment).
- Stall-reason breakdown: RF bank conflict, SMEM bank conflict, scoreboard
  hold, cp.async-not-ready, instruction-buffer empty.
- Achieved SMEM bandwidth versus 8-bank peak.
- Achieved DDR bandwidth via AXI HP versus the channel peak.

What is swept:

- GEMM shape: M = N = K in {64, 128, 256, 512}.
- cp.async on / off (cp.async off forces synchronous LDG, which exposes the
  DDR latency the async engine is hiding).
- Swizzle on / off (swizzle off should regress to a measurable bank-conflict
  rate; the gap is the win attributable to swizzling).

The result is a roofline plot, a stall-reason stacked bar across the sweep,
and a single-page summary suitable for a portfolio.

## Roadmap

Beyond v1, in roughly this order of value:

- Second MMA pipe, dual-issue from the warp scheduler.
- FP16 multiply with FP32 accumulate, sharing the DSP fabric.
- INT4 mode for double-throughput on the same DSPs.
- Per-lane branch divergence with a reconvergence stack.
- Two SMs sharing the AXI HP port, with arbitration.
- Greedy-then-oldest warp scheduler as an A/B against round-robin.

Explicit non-goals, now and later:

- A CUDA front-end. The ISA stays original.
- Virtual memory or address translation.
- Hardware-managed caches. The memory hierarchy is software-managed by design.

## References

- NVIDIA Volta, Turing, and Ampere architecture whitepapers.
- PTX ISA, the `mma.sync` section.
- GPGPU-Sim and accel-sim, for the modeling vocabulary used in
  `docs/microarch_notes.md`.
- Eyeriss and Gemmini, as systolic-array comparison points for the perf report.
- Xilinx UG479 (DSP48E1) for the MAC pipeline details.
