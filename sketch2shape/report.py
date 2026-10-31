"""Run report: timings, hardware, models used, validation, limitations (README requirement)."""
from __future__ import annotations
import json, os, time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter(); self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(time.perf_counter() - s, 3)

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 3)


def write_report(run_dir: str, data: dict) -> tuple[str, str]:
    jp, mp = os.path.join(run_dir, "run_report.json"), os.path.join(run_dir, "run_report.md")
    with open(jp, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    hw, v, a = data["hardware"], data.get("validation", {}), data.get("analysis", {})
    L = [f"# Sketch2Shape run report - {data['run_name']}", "",
         f"- **Total time:** {data['timings']['total']} s",
         f"- **Backend used:** `{data['backend_used']}` (requested: `{data['options']['backend']}`)",
         f"- **Image generator:** `{data['generator_used']}`  |  **Background removal:** `{data.get('bg_method', '-')}`",
         f"- **Hardware:** {hw.get('gpu') or 'CPU only'}"
         f"{' (' + str(hw['vram_gb']) + ' GB VRAM)' if hw.get('vram_gb') else ''}, {hw.get('cpu_count')} CPU threads, "
         f"{hw.get('ram_gb')} GB RAM, Python {hw.get('python')}, torch {hw.get('torch')}", "",
         "## Stage timings (s)", ""] + [f"- {k}: {t}" for k, t in data["timings"]["stages"].items()] + [
        "", "## Output", "", f"- GLB: `{os.path.basename(data['glb'])}`  |  OBJ: `{os.path.basename(data.get('obj') or '-')}`",
         f"- vertices / faces: {v.get('vertices')} / {v.get('faces')}  |  watertight: {v.get('watertight')}  |  "
         f"textured: {v.get('has_texture')}  |  file opens: {v.get('opens_with_reader')}",
         f"- sketch fidelity (silhouette IoU): **{data.get('fidelity', {}).get('silhouette_iou')}**",
         "", "## Sketch analysis", "",
         f"- aspect {a.get('aspect')}, symmetry {a.get('symmetry')}, through-holes {a.get('holes')}, "
         f"descriptors: {', '.join(a.get('descriptors', []))}", "", "## Notes / limitations", ""]
    L += [f"- {n}" for n in data.get("notes", [])] or ["- none"]
    open(mp, "w").write("\n".join(L) + "\n")
    return jp, mp
