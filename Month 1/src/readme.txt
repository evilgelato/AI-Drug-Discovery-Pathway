D) Quick exploratory notebook

Create notebooks/01_explore.ipynb (use VS Code’s Jupyter):

Cells to:

Read data/raw/dataset.csv

Compute descriptors for a few molecules

Plot MW vs LogP (matplotlib)

Save a processed snapshot to data/processed/…csv (optional)




) Data you can use immediately

For Month 1, your goal is binary classification of “drug-like vs not”. Easiest sources:

BBBP (blood-brain barrier penetration, binary): good starter for classification.

BACE (β-secretase inhibitors, active vs inactive).

ESOL (solubility) is regression—fine for practice, but aim for classification this month.

How to proceed today (no download automation yet):

Put your chosen CSV in data/raw/dataset.csv with columns:

smiles (string SMILES)

label (0/1)