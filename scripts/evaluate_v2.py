import sys
import os
import pandas as pd
from tqdm import tqdm
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from solver import solve

def evaluate():
    print("Loading data...")
    data_path = os.path.join(project_root, 'data', 'reference.csv')
    df = pd.read_csv(data_path)
    
    print(f"Found {len(df)} problems.")
    
    correct_count = 0
    results = []
    
    start_time = time.time()
    
    # Iterate through problems
    for index, row in tqdm(df.iterrows(), total=len(df)):
        problem_id = row['id']
        problem_text = row['problem']
        true_answer = int(row['answer'])
        
        try:
            # Run solver
            predicted_answer = solve(problem_text)
        except Exception as e:
            print(f"Error on {problem_id}: {e}")
            predicted_answer = 0
            
        is_correct = (predicted_answer == true_answer)
        if is_correct:
            correct_count += 1
            
        results.append({
            'id': problem_id,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'correct': is_correct
        })
        
    end_time = time.time()
    accuracy = correct_count / len(df)
    
    print()
    print("="*30)
    print(f"Evaluation Complete.")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(df)})")
    print("="*30)
    
    # Save results
    output_path = os.path.join(project_root, 'submissions', 'evaluation_results.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    evaluate()
