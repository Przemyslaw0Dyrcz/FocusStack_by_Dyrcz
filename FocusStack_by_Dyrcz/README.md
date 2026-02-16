# FocusStack by Dyrcz


## Contents
- `focusstack/` — Python package with modules (io, alignment, segmentation, filters, merging, composition, main)
- `main.py` — entrypoint CLI (wraps `focusstack.main`)
- `run_pipeline.py` — small wrapper for programmatic calls
- `requirements.txt`, `.gitignore`, `README.md`, `LICENSE`

## Quick start (Linux / WSL / macOS / Windows PowerShell)
1. clone or extract the ZIP
2. create virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   ```
3. install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. download FastSAM weights (e.g. `FastSAM-s.pt`) and place in project folder or give full path to `--fastsam_model`
5. run:
   ```bash
   python main.py --input /path/to/images --out_dir out --fastsam_model FastSAM-s.pt
   ```

## Notes / caveats
- You must have compatible versions of `torch` and `ultralytics` (and a CUDA-capable GPU + drivers if using `--device cuda`).
- If model inference fails, check `torch.cuda.is_available()` and the `fastsam` model path.
- The implementation expects images of similar resolution; if images vary, they will be resized to first image's size.


