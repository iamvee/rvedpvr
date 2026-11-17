# rvedpvr

Passive biventricular inflation simulations for the right-ventricular end-diastolic
pressure–volume relation. One run inflates one rat heart geometry in FEniCSx and records
the pressure–volume curve and the wall geometry at every load step; the batch tools run
thousands of such runs as parallel Docker containers.

## Layout

```
src/            the solver (run inside the solver image)
  edpvr.py        command line and the sequence of one simulation
  abaqus_mesh.py  reader for the node/elem/set/surf text meshes
  fem_mesh.py     DOLFINx mesh construction and boundary tagging
  morph.py        cavity scaling, translation and septal shift of a base geometry
  fibers.py       rule-based helical fiber field
  material.py     function spaces and the strain-energy densities
  solver.py       residual, boundary conditions, Newton solver, volumetric error
  loading.py      cavity-volume integrals, surface tracking, the pressure ramp
  outputs.py      geometry records and the files each run writes
  thickness.py    transmural wall thickness and its apex-to-base bands
  measure.py      surface areas, radii, curvature, extents
  geom_filter.py  anatomical acceptance windows for a probed geometry
mesh/mesh_files/  the 20 base geometries (ids 1, 21, ..., 381)
designs/runqueue.json   every simulation of the study, one entry per run
results/<run_id>/       the numeric record of every simulation
plan.py           draws a candidate design from STUDY_CONFIG.yaml
screen.py         probes every distinct geometry, applies the windows, writes a run queue
run.py            solves a queue in parallel containers; resumable, retrying, shardable
aggregate.py      one row per finished run -> results/summary.csv
merge_results.py  merges results/ from several machines solving one queue
config.py         loads STUDY_CONFIG.yaml
STUDY_CONFIG.yaml machine budget, loading protocol, material, sweep bands, cohort design
docker/           solver image (pinned DOLFINx) and runner image (Python + Docker CLI)
run_study.sh/.bat one command: build images, solve the queue, aggregate
CITATIONS.md      software and methods to cite
```

## Requirements

Docker (Docker Desktop on Windows/macOS, Docker Engine on Linux). No host Python is
needed.

## Base geometries

Each geometry is four plain-text files, `<id>_node.txt`, `<id>_elem.txt`, `<id>_set.txt`,
`<id>_surf.txt`, 1-indexed: node coordinates (mm); linear tetrahedra (Abaqus C3D4); node
sets `top_nodes` (basal plane), `topNode_X_max_min_nodes`, `topNode_Z_max_min_nodes`
(in-plane pins), `epi_nodes`, `LV_nodes`, `RV_nodes`; surfaces `epi_surf`, `top_surf`,
`LV_surf`, `RV_surf` as element–face pairs. `src/abaqus_mesh.py` reads them. The morphs
applied per simulation (cavity scaling, translation, septal shift) are in `src/morph.py`.

## One simulation

```bash
docker build -t rvedpvr-fenics -f docker/solver.Dockerfile .
docker run --rm -v "$PWD:/work" -w /work rvedpvr-fenics \
    python src/edpvr.py --geom 1 --lvedp 30 --rvedp 10 --steps 20 --no-index
```

Geometry morph: `--f_lv --f_rv` scale each cavity, `--shift_lv --shift_rv` translate them
along the LV→RV axis, `--sept` moves the RV endocardium toward the free wall. Fibers:
`--endo --epi` helix angles (deg). Material: `--material ho` (default; `--a --b --af --bf`,
kPa) or `--material fung` (`--c --B1 --B2 --B3 --K`). Formulation: `--formulation
compressible` (default, `--kappa` MPa, `--u-order 1|2`), `incompressible` (P2/P1) or
`hybrid` (P1/DG0, `--pstab`). `--probe` measures the reference geometry and stops.
`python src/edpvr.py --help` lists everything.

## The study

`designs/runqueue.json` lists every simulation of the study: its `run_id`, arm, geometry
and the full solver parameter set. `results/<run_id>/` holds the numeric record each of
them produced (files below). `run.py` skips a run whose record is present, so

```bash
./run_study.sh            # (Windows: run_study.bat)
```

solves only what is missing; delete `results/` to re-solve everything. `./run_study.sh 50`
solves the first 50 queued runs, `./run_study.sh plan` builds a new queue without solving.
`--shard i/N` or `--range A-B` split one queue across machines, which must all hold the
same `designs/runqueue.json`; `merge_results.py` combines their `results/`.

A new design is drawn from `STUDY_CONFIG.yaml` by `plan.py` (three arms: Latin-hypercube
material and shape draws at each pressure ratio of the protocol; every base geometry at
the reference material over a ratio sweep; wider bands over every factor) and screened by
`screen.py`, which probes each distinct morphed geometry once (`edpvr.py --probe`), rejects
shapes outside the windows of `src/geom_filter.py` or with inverted elements, and writes a
seeded, shuffled queue with a hash. `run.py` prints the loaded settings.

## Outputs

Each run writes `results/<run_id>/`. Lengths mm, areas mm², volumes mm³, pressures mmHg.

| File | Content |
|---|---|
| `params.json` | every input parameter, plus `morph_quality` (inverted elements, minimum volume ratio) |
| `pv.csv` | `P_lv, P_rv, V_lv, V_rv, V_epi` per converged load step, row 0 unloaded |
| `measurements.json` | converged/requested steps, myocardial volume drift, wall geometry at the reference and final states |
| `measurements_ts.csv` | the same wall geometry at every step: thickness (mean, std, five apex-to-base bands) of septum, LV and RV free wall, areas, radii, curvature |
| `frames.npz`, `fibers.npz` | deforming surfaces and the fiber field; written only without `--no-frames` |
| `probe.json` | with `--probe` only: reference-geometry measurements |

Wall thickness is the transmural distance along each endocardial vertex's outward normal
to the first surface met; a vertex belongs to the septum when that surface is the other
endocardium and to a free wall when it is the epicardium.

`python aggregate.py` writes `results/summary.csv`, one row per run: the parameters, the
pressure ratio, convergence, and the reference and final thickness, area, volume and
radius values.
