import json
import os

notebook_path = 'notebooks/kaggle_submission_template.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "from kaggle_baseline import" in "".join(cell['source']):
        new_source = []
        for line in cell['source']:
            if line.strip().startswith("from kaggle_baseline import"):
                new_source.append("    from kaggle_baseline import CompetitionConfig, AIMSolver, MockLLM, VLLMEngine" + chr(10))
            else:
                new_source.append(line)
        cell['source'] = new_source

new_setup_source = [
    "# 2. Configure Environment & Model" + chr(10),
    "config = CompetitionConfig()" + chr(10),
    "config.n_repetitions = 16" + chr(10),
    chr(10),
    "print(f"Environment: {'Kaggle' if config.is_kaggle else 'Local'}")" + chr(10),
    chr(10),
    "if config.is_kaggle and os.path.exists(config.model_path):" + chr(10),
    "    print(f"Loading real VLLM Engine from {config.model_path}...")" + chr(10),
    "    llm = VLLMEngine(config)" + chr(10),
    "else:" + chr(10),
    "    print("Using MockLLM for local testing or if path not found.")" + chr(10),
    "    llm = MockLLM()" + chr(10),
    chr(10),
    "solver = AIMSolver(config, llm)" + chr(10)
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "# 2. Configure Environment" in "".join(cell['source']):
        cell['source'] = new_setup_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated to use VLLMEngine.")
