"""Command-line tool:  python -m sketch2shape.cli sketch.png --prompt "red ceramic mug" """
from __future__ import annotations
import argparse, json, sys
from PIL import Image
from .pipeline import Options, run


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sketch2Shape AI - sketch image -> textured 3D (.glb/.obj)")
    ap.add_argument("sketch", help="path to a sketch image (png/jpg/webp)")
    ap.add_argument("--prompt", default="", help="optional text: type / colour / material / style")
    ap.add_argument("--backend", default="auto", choices=["auto", "inflate", "trellis", "instantmesh", "dreamfusion"])
    ap.add_argument("--generator", default="auto", choices=["auto", "sd_controlnet", "none"])
    ap.add_argument("--holes", default="fill", choices=["fill", "nested"],
                    help="'nested': inner outlines inside an outline become through-holes (rings, frames)")
    ap.add_argument("--candidates", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--name", default="")
    ap.add_argument("--use-claude", action="store_true", help="use Anthropic API to name the object (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args(argv)
    r = run(Image.open(a.sketch), Options(prompt=a.prompt, backend=a.backend, generator=a.generator, holes=a.holes,
                                          candidates=a.candidates, steps=a.steps, seed=a.seed, resolution=a.resolution,
                                          depth_scale=a.depth_scale, out_dir=a.out, run_name=a.name, use_claude=a.use_claude),
            progress=lambda f, m: print(f"[{int(f * 100):3d}%] {m}", file=sys.stderr))
    print(json.dumps({k: r[k] for k in ("glb", "obj", "backend_used", "generator_used", "report_md")}, indent=2))
    print(f"time {r['timings']['total']}s | faces {r['validation']['faces']} | watertight {r['validation']['watertight']} "
          f"| sketch IoU {r['fidelity']['silhouette_iou']}")
    for n in r["notes"]: print("note:", n)


if __name__ == "__main__":
    main()
