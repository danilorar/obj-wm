# obj-wm Makefile
# See README.md § Make targets for descriptions.

.PHONY: help setup data train-mini demo anchors quickstart figures play drive

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
	@echo "  data        process nuScenes mini into training data  [needs data download first]"
	@echo "  train-mini  train the world model on nuScenes mini (~30 epochs, any GPU)"
	@echo "  demo        run one scene closed-loop (CV vs WM), write a GIF  [needs train-mini]"
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

data: $(VENV)/bin/python
	@if [ ! -f "$(DATA_MINI)/v1.0-mini/scene.json" ]; then \
	    echo ""; \
	    echo "ERROR: nuScenes mini not found at $(DATA_MINI)/v1.0-mini/"; \
	    echo "  1. Create a free account at https://nuscenes.org/nuscenes#download"; \
	    echo "  2. Download: v1.0-mini.tgz  and  nuScenes-map-expansion-v1.3.zip"; \
	    echo "  3. Extract: mkdir -p $(DATA_MINI) && tar xf v1.0-mini.tgz -C $(DATA_MINI)/"; \
	    echo "             mkdir -p data/maps && unzip nuScenes-map-expansion-v1.3.zip -d data/maps/"; \
	    echo "             ln -s ../maps $(DATA_MINI)/maps"; \
	    echo "  4. Re-run: make data"; \
	    echo ""; \
	    exit 1; \
	fi
	$(VENV)/bin/python scripts/build_processed_dataset.py --mini --lane-tokens --workers 4
	$(VENV)/bin/python scripts/make_anchors.py
	@echo "==> data ready (mini_train / mini_val + anchors)"

anchors: $(VENV)/bin/python
	$(VENV)/bin/python scripts/make_anchors.py

# ── world model training (nuScenes mini) ─────────────────────────────────────

train-mini: $(VENV)/bin/python
	$(VENV)/bin/python scripts/train_model.py \
	    --model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2 \
	    --epochs 30 --seed 42 --checkpoint-dir checkpoints/mini
	$(VENV)/bin/python scripts/train_model.py \
	    --model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2 \
	    --evaluate-only --checkpoint checkpoints/mini/v2_seed42_best.pth
	@echo "==> checkpoint: checkpoints/mini/v2_seed42_best.pth"
	@echo "    (mini-only model is weak — val minADE6 ~3.5 m vs 1.24 m on full data)"

# ── closed-loop demo (CV vs WM, one mini scene) ──────────────────────────────

CK_DEMO ?= checkpoints/mini/v2_seed42_best.pth

demo: $(VENV_SIM)/bin/python
	@if [ ! -f "$(CK_DEMO)" ]; then \
	    echo "ERROR: checkpoint not found at $(CK_DEMO)"; \
	    echo "  Run 'make train-mini' first, or set CK_DEMO=<path>."; \
	    exit 1; \
	fi
	$(VENV_SIM)/bin/python scripts/sim_check.py convert --scenes 10
	@mkdir -p outputs/demo
	$(VENV_SIM)/bin/python scripts/demo_2x2.py clip_topdown \
	    --data sim/scenarios/nuscenes_mini \
	    --index 8 \
	    --out outputs/demo/scene1094_wm_vs_cv.gif \
	    --checkpoint $(CK_DEMO) \
	    --scene-label "mini scene-1094 (dev)" \
	    --label "rider enters from left" --label-t 4.0 --label-dur 2.5 \
	        --label-panel both --label-y 70 \
	    --wm-earlier-label \
	    --reactive idm
	@echo "==> demo GIF: outputs/demo/scene1094_wm_vs_cv.gif"

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
