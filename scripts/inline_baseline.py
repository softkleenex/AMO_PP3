import json

notebook_path = 'notebooks/kaggle_submission_template.ipynb'
baseline_path = 'src/kaggle_baseline.py'

# 1. Read the baseline code
with open(baseline_path, 'r', encoding='utf-8') as f:
    baseline_code = f.read()

# Split into lines and add newline character
baseline_lines = [line + '
' for line in baseline_code.split('
')]

# 2. Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 3. Create the inline code cell
inline_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["# INLINED KAGGLE BASELINE MODULE
"] + baseline_lines
}

# 4. Remove the old import cell
cells_to_keep = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "from kaggle_baseline import" in "".join(cell['source']):
        continue # Skip this cell
    cells_to_keep.append(cell)

nb['cells'] = cells_to_keep

# 5. Insert the inline cell right before the environment setup cell
setup_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and "# 2. Configure Environment" in "".join(cell['source']):
        setup_idx = i
        break

if setup_idx != -1:
    nb['cells'].insert(setup_idx, inline_cell)
else:
    nb['cells'].insert(2, inline_cell) # Fallback insertion

# 6. Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated to use inline kaggle_baseline code.")