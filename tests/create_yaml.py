import pytest
import os
import yaml

from src.trainer import Trainer

def test_create_yaml():
    trainer = Trainer(pytest.PROJECT_ID,pytest.ls,pytest.ls_client)
    trainer.create_yaml()
    yaml_path = f"./gym/project_{pytest.PROJECT_ID}/data.yaml"
    assert os.path.exists(f"./gym/project_{pytest.PROJECT_ID}/data.yaml")
    with open(yaml_path, 'r') as file:
        prime_service = yaml.safe_load(file)
        assert prime_service["train"] == f"images/train"
        assert prime_service["test"] == f"images/test"
        assert prime_service["val"] == f"images/val"
        assert prime_service["nc"] == len(prime_service["names"])