# Architecture notes

## Data flow
`SketchData` (clean line art, stroke mask, ControlNet scribble) → `analysis` (silhouette, holes, symmetry, descriptors, prompt)
→ optional `generate` (N images) → `background` (RGBA) → ranking by `validate.mask_iou(candidate, sketch silhouette)`
→ backend (`Mesh` or GLB path) → `validate_mesh` → `write_glb/write_obj` → `silhouette_iou` → `run_report`.

## Key decisions
1. **Fallback over failure.** Every stage that can fail (model download, OOM, missing backend) degrades to a CPU alternative and
   records why in `notes`. The only hard errors are user-input errors (blank canvas).
2. **Sketch fidelity is measured, not assumed.** Orientation-agnostic silhouette IoU (12 axis-aligned orthographic views,
   normalised for scale/position) works for any backend's mesh and is reported in the UI.
3. **Geometry guard.** If best candidate IoU < `min_fidelity` (0.55) the sketch silhouette, not the generated matte, defines
   the shape; the generated image is only used as texture.
4. **Watertight CPU mesh.** Height-field front/back surfaces on a pixel grid; rim vertices (z = 0) are welded; remaining
   single-use edges get side-wall triangles; zero-thickness blocks are dropped and slivers (<3 px) removed so every edge is shared
   by exactly two faces. Euler characteristic = 2 for blobs, 0 for a ring (verified in tests). Taubin smoothing (λ|μ) removes
   pixel stair-casing without shrinking the shape.
5. **Dependency-free export.** GLB (glTF 2.0, embedded PNG, PBR material) and OBJ/MTL writers + GLB reader are numpy-only, so
   validation ("does the file open, is geometry non-empty, is texture preserved") runs everywhere; `trimesh` is used as a second
   opinion when installed.
6. **Holes.** `fill` (default) treats every enclosed region as solid; `nested` applies a contour-hierarchy parity rule
   (inside one outline = solid, inside two = hole, …) which matches how rings/frames are drawn.

## Extending
Add a backend: create `backends/<name>.py` with `available()` and `reconstruct(...)`, register it in `pipeline.run`'s backend
branch and `--backend` choices. It will automatically get validation, fidelity scoring, fallback and reporting.
