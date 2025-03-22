import os
import pandas as pd

class MemoryHandler:
    def __init__(self):
        pass

    def commit_results_to_memory(self, project_id, metrics_path):
        results = os.path.join(metrics_path, 'results.csv')
        memory_path = f'./memory/project-{project_id}'
        os.makedirs(memory_path, exist_ok=True)
        result_save_path = os.path.join(memory_path, 'results.csv')
        df = pd.read_csv(results)  # Load existing results
        df.to_csv(result_save_path, index=False)
    
    def pull_latest_results_for(self, project_id)->list:
        if os.path.exists(f'./memory/project-{project_id}/results.csv'):
            df = pd.read_csv(f'./memory/project-{project_id}/results.csv')
            return df.to_dict(orient="records")
        else:
            return []
