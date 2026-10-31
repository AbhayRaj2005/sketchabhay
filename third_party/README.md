# third_party

`stable-dreamfusion/` - unmodified copy of https://github.com/ashawkey/stable-dreamfusion (Apache-2.0, see its LICENSE),
trimmed of sample data (`data/`, `tets/`) to keep the repo small. It powers the optional, slow `dreamfusion` backend
(`backends/dreamfusion.py`): Zero-1-to-3 score-distillation image-to-3D.

To enable it (CUDA GPU required, 10-40 min per object):
```bash
cd third_party/stable-dreamfusion
pip install -r requirements.txt
bash scripts/install_ext.sh                    # builds raymarching / gridencoder / shencoder / freqencoder
mkdir -p pretrained/zero123 && cd pretrained/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
cd ../../../.. && python scripts/check_env.py  # should now show [OK] Stable-DreamFusion backend
python -m sketch2shape.cli samples/inputs/rocket.png --backend dreamfusion
```
Hofer's *Sketch-to-3d-Model* thesis code is NOT bundled (no license file in the repository); only its ideas
(flood-fill silhouette, Euler-number topology, IoU evaluation) were re-implemented from scratch in `sketch2shape/silhouette.py`.
