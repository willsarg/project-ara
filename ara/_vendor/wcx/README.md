<div align="center">

# wcx-suite

**Will's CUDA Suite** — a VRAM/stress bench for local CUDA inference on NVIDIA GPUs.

*Find each model's safe context ceiling without exhausting the card.*

</div>

---

wcx-suite is the CUDA sibling of [wmx-suite](https://github.com/willsarg/wmx-suite). Same
discipline, other side of the hardware fence: it finds the **safe context ceiling** for a
model+context on **your** GPU by extrapolating from measurements taken below the hardware
wall — never by probing into it.

On Apple Silicon the wall is the unified-memory working set; here it's the GPU's **VRAM**.
A CUDA over-allocation usually raises an out-of-memory error and kills the *workload* rather
than the machine — but on a GPU that also drives a display, exhausting VRAM can freeze the
desktop or trip a driver reset. The rule holds: stay under the wall.

## Quick start

```bash
uv sync
uv run wcx-suite system                 # GPU, VRAM total/free, the wall, and the safe budget
uv run wcx-suite calibrate              # measure this card's cold-start VRAM overhead (once per machine)
uv run wcx-suite characterize <hf_id>   # safe probe → fitted safe context ceiling, stored
```

`characterize` ramps context under the engine and **refuses before it ever loads past the
VRAM wall**, then fits and stores the safe ceiling — the same predict-don't-probe discipline
as wmx-suite. Footprint is read in binary **GiB** to match the wall, and weight/KV-cache
quantization (int4/int8/fp8 weights, hqq KV) is wired through the loader.

## Status

A functional measurement engine, not barebones anymore: `system` + `calibrate` +
`characterize`, plus governed **generate** and capability **benchmark** workers that
[project-ara](https://github.com/willsarg/project-ara) drives out-of-process
(`python -m wcx_suite.generate` / `wcx_suite.benchmark`). Both apply each model's **own chat
template** so template-strict instruct models (e.g. Gemma) behave correctly. **186 tests pass.**

## Where it fits

This is the engine [project-ara](https://github.com/willsarg/project-ara) installs on NVIDIA
hardware (`ara install --engine wcx`), the way wmx-suite is its Apple-Silicon engine.

## License

[Apache 2.0](./LICENSE) © Will Sarg
