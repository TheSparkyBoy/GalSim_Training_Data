# run_pipeline.py
import random
import numpy as np

# Import the config from the local directory
from config import DATASET_CONFIG

# Import the orchestrator from the core folder
from core.orchestrator import DatasetOrchestrator

if __name__ == '__main__':
    # 1. Seed the main thread globally to ensure the random sky coordinates are identical
    random.seed(DATASET_CONFIG['global_seed'])
    np.random.seed(DATASET_CONFIG['global_seed'])
    
    # 2. Ignite the pipeline!
    manager = DatasetOrchestrator(DATASET_CONFIG)
    manager.execute()