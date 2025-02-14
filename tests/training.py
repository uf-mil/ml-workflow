import pytest
import asyncio

from src.trainer import Trainer

def test_training():
    trainer = Trainer(pytest.PROJECT_ID,pytest.ls,pytest.ls_client)
    result = asyncio.run(trainer.train())