# Transformer Interpretability Project

##  Overview
This project contains code for experimenting with transformer interpretability.

- Core logic is implemented in Python scripts
- Experiments and execution are done in a Jupyter notebook

---

##  Project Structure

.
├── logit.py        # core logic  
├── tuned.py        # core logic  
├── main_notebook.ipynb  # run experiments  
├── pyproject.toml  
├── uv.lock  

---

##  Setup Instructions

### Step 1: Clone the repository

git clone https://github.com/JesanOvi/Transformer-interpretability.git  
cd your-repo-name  

---

##  Option A: Using uv (Recommended)

### Install uv

pip install uv  

### Install dependencies

uv sync  

---

##  Running the project

### Run Python script

uv run python file1.py  

### Run notebook

1. Open notebook.ipynb  
2. Select kernel: Python (.venv)  
3. Run all cells  

---

##  Option B: Without uv

### Create virtual environment

python3 -m venv .venv  
source .venv/bin/activate  

### Install dependencies

pip install -r requirements.txt  

---

##  Notes

- This project uses uv for dependency management  


##  How to Use in `main_notebook.ipynb` (IMPORTANT)

In the notebook, you can directly create objects and run experiments like this:

### Logit Lens Example

```python
from logit import LogitLens

lens = LogitLens(model_name="EleutherAI/pythia-14m", top_k=5)
lens.run("The sky is blue and the grass is ")