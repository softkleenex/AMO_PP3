import json

notebook_path = 'notebooks/kaggle_submission_template.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update imports in the first code cell to include VLLMEngine
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "from kaggle_baseline import" in "".join(cell['source']):
        new_source = []
        for line in cell['source']:
            if line.startswith("    from kaggle_baseline import"):
                new_source.append("    from kaggle_baseline import CompetitionConfig, AIMSolver, MockLLM, VLLMEngine
")
            else:
                new_source.append(line)
        cell['source'] = new_source

# Update the environment setup cell
new_setup_source = [
    "# 2. Configure Environment & Model
",
    "config = CompetitionConfig()
",
    "config.n_repetitions = 16  # 다수결 투표 횟수 (16회)
",
    "
",
    "print(f"Environment: {'Kaggle' if config.is_kaggle else 'Local'}")
",
    "
",
    "if config.is_kaggle and os.path.exists(config.model_path):
",
    "    print(f"Loading real VLLM Engine from {config.model_path}...")
",
    "    llm = VLLMEngine(config)
",
    "else:
",
    "    print("⚠️ Using MockLLM for local testing or if path not found.")
",
    "    llm = MockLLM()
",
    "
",
    "solver = AIMSolver(config, llm)
"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "# 2. Configure Environment" in "".join(cell['source']):
        cell['source'] = new_setup_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated to use VLLMEngine.")
