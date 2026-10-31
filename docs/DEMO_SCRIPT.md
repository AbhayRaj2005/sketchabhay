# 5-minute demo + judge Q&A

**Before the demo (10 min):** `python scripts/check_env.py`; run one sample end-to-end; open `python app.py`; keep `samples/outputs/` as backup.

1. (30 s) Problem + one-line pitch: "Any sketch → real 3D file, on any laptop, GPU optional."
2. (60 s) Draw live on the canvas (a bottle / lamp / ring). Add prompt "blue glass bottle". Generate.
3. (60 s) Rotate in the 3D viewer; point at the Fidelity score, watertight flag, timing; open the `.glb` in Blender.
4. (45 s) Judge hands you a fresh sketch / uploads a photo of paper → same pipeline (proves nothing is pre-generated).
5. (45 s) Toggle *Holes → nested* on a ring/racket: through-hole appears. Explain Euler-number logic.
6. (45 s) Show `run_report.md`: hardware, stage timings, fallbacks. Explain tiered backends and the drift guard.
7. (15 s) Limitations slide (single view hides the back; inflate is blob-like; GPU path recommended).

**Likely questions**
* *Is it just a plane?* No: validator checks depth ratio, watertightness, Euler characteristic.
* *How do you keep the object like the sketch?* ControlNet scribble + candidate re-ranking by silhouette IoU + geometry guard.
* *What if there is no GPU in the evaluation room?* `inflate` backend, ~2 s, always valid GLB.
* *What do you download / which services?* docs/MODELS.md; nothing external by default.
