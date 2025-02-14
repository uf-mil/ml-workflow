import pytest

from src.trainer import Trainer

def test_organize_data():
    trainer = Trainer(pytest.PROJECT_ID,pytest.ls,pytest.ls_client)
    trainer.organize_data()