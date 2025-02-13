import pytest
import os
import yaml

from src.trainer import Trainer

def test_organize_data():
    trainer = Trainer(pytest.PROJECT_ID,pytest.ls,pytest.ls_client)
    trainer.organize_data()
    # yaml_path = f"./gym/project_{pytest.PROJECT_ID}/data.yaml"
    # assert os.path.exists(f"./gym/project_{pytest.PROJECT_ID}/data.yaml")
    # with open(yaml_path, 'r') as file:
    #     prime_service = yaml.safe_load(file)
    #     assert prime_service["train"] == f"./gym/project_{pytest.PROJECT_ID}/images/train"
    #     assert prime_service["test"] == f"./gym/project_{pytest.PROJECT_ID}/images/test"
    #     assert prime_service["val"] == f"./gym/project_{pytest.PROJECT_ID}/images/val"
    #     assert prime_service["nc"] == len(prime_service["names"])