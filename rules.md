# Project Rules

1. **Virtual Environment Execution:** No code will be run without first activating the virtual environment (`source .venv/bin/activate`). We will download all packages using `uv` under this virtual environment.
2. **Documentation & Planning Updates:** The `plan.md` and `docs.md` files must be updated after finishing each stage. They must be read before starting a new stage or for reference.
3. **Production-Grade Logging:** We will log everything, including every result, with production-grade quality.
4. **Third-Party Repositories:** Any cloned third-party repositories for the project must be placed into the `third_party` folder. We will access them from there.

./.venv/bin/python run_stage1.py --input_dir basic_pick_place --output_dir output --limit 25
./.venv/bin/pip install "numpy==1.23.5"
./.venv/bin/python run_stage1_objects.py --limit 2
./.venv/bin/python visualize.py --limit 2
./.venv/bin/python visualize_obj.py --file output/0_stage1.hdf5 --frame 0
./.venv/bin/python visualize_obj.py --file output/0_stage1.hdf5 --animate
./.venv/bin/python visualize_obj.py --batch --input_dir output --out_dir viz_output --animate --frame 0
