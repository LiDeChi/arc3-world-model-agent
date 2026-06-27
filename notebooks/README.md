# Learning ARC3 with Notebooks

This project is designed to be learned interactively through notebooks.
The `notebooks/` directory contains a progressive series that lets you explore the code, run real components, and inspect the implementation as you go.

**Perfect for "边看代码边学" (learn by reading + running the actual source).**

## Notebook Series (Recommended Order)

| Notebook | Focus | What you'll learn |
|----------|-------|-------------------|
| `01_intro_arc3.ipynb` | Big picture | The three core pieces: Env, WorldModelAgent (with imagination), Self-tuning platform |
| `02_environment_explorer.ipynb` | Environment | What observations look like, action space, MockArcEnv |
| `03_world_model_explained.ipynb` | Dreamer-lite | Encode → Predict latent → Decode + reward. The "imagination" core |
| `04_planning_and_imagination.ipynb` | Agent | How the agent uses imagined rollouts for planning |
| `05_self_tuning_platform.ipynb` | PBT | Population-Based Training controller and hyperparameter evolution |
| `06_results_dashboard.ipynb` | Visualization | Metrics, reports, and how to analyze runs |
| `07_submission_walkthrough.ipynb` | Deployment | The tiny offline submission agent + compliance check |

## How to Run (Your Choice)

### Option 1: With marimo (recommended since you just installed it)

```bash
# From repo root
marimo edit notebooks/01_intro_arc3.ipynb
```

Marimo gives you a beautiful reactive editing experience. Notebooks are still .ipynb for now, but you can convert:

```bash
marimo convert notebooks/01_intro_arc3.ipynb > notebooks/01_intro_arc3.py
marimo edit notebooks/01_intro_arc3.py   # native marimo .py format (great for git)
```

### Option 2: Traditional Jupyter

```bash
# Install notebook support
python -m pip install -e ".[notebook]"

# Launch
jupyter lab notebooks/
# or
jupyter notebook notebooks/
```

### Option 3: VS Code

Just open the `.ipynb` files — they have rich output and can execute against your Python environment.

## Setup Before Running

```bash
# From the arc3 root
source .venv/bin/activate          # or your preferred env
python -m pip install -e .         # at minimum

# For full viz + interactive features (optional)
python -m pip install -e ".[notebook,viz]"
```

All notebooks import from the installed `arcagent` package and the `submission` package, so editable install is important.

## Learning Tips

- Every notebook starts with high-level explanation, then executable code.
- Look for cells that do `import inspect; print(inspect.getsource(...))` — these let you read the real implementation right inside the notebook.
- Run cells sequentially. Many build on previous state.
- Feel free to experiment — change grid size, imagination horizon, etc. and re-run.
- The notebooks are safe "smoke" level; they won't run long training unless you explicitly ask.

## Want More?

- The source is small and readable — after the notebooks, go read:
  - `arcagent/models/dreamer_lite.py`
  - `arcagent/agent/world_model_agent.py`
  - `arcagent/platform/controller.py`
- Run a real training smoke and then open `06_results_dashboard.ipynb`.
- Use marimo's reactive features to live-edit and see the effect on imagined grids instantly.

Happy exploring! The whole point of this project is to make the "world model + imagination + self-tuning" ideas transparent and hackable.

If you'd like me to expand any specific notebook, convert the whole series to native marimo .py files, or add deeper source dives / exercises, just say the word.
