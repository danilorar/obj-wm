# obj-wm Makefile
# See README.md § Make targets for descriptions.

.PHONY: help setup anchors quickstart figures play drive

PYTHON    := python3.12
UV        := uv
VENV      := .venv
VENV_SIM  := .venv_sim
CK        ?= checkpoints/v2_d_rerun/v2_seed42_best.pth
DATA_MINI ?= data/nuscenes_mini
SCENE     ?= 0
PLANNER   ?= wm
LIVE      ?= 0

help:
	@echo "Targets:"
	@echo "  setup       create .venv (training) and .venv_sim (simulation)"
	@echo "  anchors     compute anchor trajectories  [needs nuScenes mini]"
	@echo "  quickstart  end-to-end pipeline on nuScenes mini  [needs nuScenes mini]"
	@echo "  figures     re-draw tables and figures from results/  [no data needed]"
	@echo "  play        run one scene and save a GIF"
	@echo "              SCENE=<idx> PLANNER={cv,wm} [LIVE=1 to open the GIF]"
	@echo "  drive       constant-action drive test (offscreen, saves GIF)"
	@echo "              SCENE=<idx>"
	@echo ""
	@echo "Variables: CK (checkpoint), DATA_MINI, SCENE, PLANNER, LIVE"

# ── environment ──────────────────────────────────────────────────────────────

setup:
	@echo "==> training venv ($(VENV), python 3.12)"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade pip
	$(VENV)/bin/pip install --quiet -r requirements.txt scikit-learn scipy
	@echo "==> simulation venv ($(VENV_SIM), python 3.10)"
	$(UV) venv --python 3.10 $(VENV_SIM)
	@if [ ! -d sim/metadrive ]; then \
	    git clone --depth 1 https://github.com/metadriverse/metadrive.git sim/metadrive; \
	fi
	@if [ ! -d sim/scenarionet ]; then \
	    git clone --depth 1 https://github.com/metadriverse/scenarionet.git sim/scenarionet; \
	fi
	VIRTUAL_ENV=$(VENV_SIM) $(UV) pip install --quiet \
	    -e sim/metadrive -e sim/scenarionet "nuscenes-devkit>=1.1.10" ijson
	VIRTUAL_ENV=$(VENV_SIM) $(UV) pip install --quiet torch \
	    --index-url https://download.pytorch.org/whl/cu132
	$(VENV_SIM)/bin/python -m metadrive.pull_asset
	@if [ ! -d "$(DATA_MINI)/v1.0-mini" ]; then \
	    echo ""; \
	    echo "==> nuScenes mini data not found in $(DATA_MINI)/"; \
	    echo "    Download from https://nuscenes.org/nuscenes#download (free, non-commercial):"; \
	    echo "      v1.0-mini         -> $(DATA_MINI)/v1.0-mini/"; \
	    echo "      map expansion v1.3 -> data/maps/"; \
	    echo "    Then: ln -s ../maps $(DATA_MINI)/maps"; \
	    echo ""; \
	fi
	@echo "==> setup done"

# ── data preparation ─────────────────────────────────────────────────────────

anchors: $(VENV)/bin/python
	$(VENV)/bin/python scripts/make_anchors.py

# ── end-to-end quickstart (nuScenes mini) ────────────────────────────────────

quickstart: $(VENV)/bin/python $(VENV_SIM)/bin/python
	$(VENV)/bin/python scripts/build_processed_dataset.py --mini --lane-tokens --workers 4
	$(VENV)/bin/python scripts/make_anchors.py
	$(VENV)/bin/python scripts/train_model.py \
	    --model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2 \
	    --epochs 30 --seed 42 --checkpoint-dir checkpoints/mini
	$(VENV)/bin/python scripts/train_model.py \
	    --model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2 \
	    --evaluate-only --checkpoint checkpoints/mini/v2_seed42_best.pth
	$(VENV_SIM)/bin/python scripts/sim_check.py convert --scenes 10
	$(VENV_SIM)/bin/python scripts/sim_check.py plan \
	    --predictor cv wm --scenes 8 \
	    --checkpoint checkpoints/mini/v2_seed42_best.pth \
	    --out outputs/sim/quickstart.json

# ── figures from results/ (no data needed) ───────────────────────────────────

figures: $(VENV)/bin/python
	$(VENV)/bin/python results/make_figures.py

# ── playground ───────────────────────────────────────────────────────────────

play: $(VENV_SIM)/bin/python
	@mkdir -p outputs/play
	$(VENV_SIM)/bin/python scripts/sim_check.py plan \
	    --predictor $(PLANNER) --scenes $(SCENE) --gif $(SCENE) \
	    --checkpoint $(CK) \
	    --out outputs/play/$(PLANNER)_scene$(SCENE).json
	@echo "==> GIF saved: outputs/play/plan_$(PLANNER)_scene$(SCENE).gif"
ifeq ($(LIVE),1)
	@xdg-open outputs/play/plan_$(PLANNER)_scene$(SCENE).gif 2>/dev/null \
	    || eog outputs/play/plan_$(PLANNER)_scene$(SCENE).gif 2>/dev/null \
	    || display outputs/play/plan_$(PLANNER)_scene$(SCENE).gif 2>/dev/null \
	    || echo "No viewer found — open the GIF manually."
endif

drive: $(VENV_SIM)/bin/python
	$(VENV_SIM)/bin/python scripts/sim_check.py drive --index $(SCENE)
