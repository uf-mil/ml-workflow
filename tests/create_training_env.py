import pytest
import os
from src.trainer import Trainer

def test_create_training_directories():
    base_path = f"./gym/project_{pytest.PROJECT_ID}"
    paths = [
        f"{base_path}/images/train",
        f"{base_path}/images/test",
        f"{base_path}/images/val",
        f"{base_path}/labels/train",
        f"{base_path}/labels/test",
        f"{base_path}/labels/val",
    ]
    
    # Create trainer and set up gym
    current_dir = os.getcwd()
    
    # Method being tested
    Trainer(pytest.PROJECT_ID,pytest.ls,pytest.ls_client)
    
    try:
        for path in paths:
            assert os.path.isdir(path), f"Directory {path} was not created"
    finally:
        os.chdir(current_dir)