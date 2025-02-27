import pytest
from unittest.mock import patch
import smbclient
import smbclient.path
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

FILE_SERVER_IP = os.getenv("FILE_SERVER_IP")
SHARED_FOLDER = os.getenv("SHARED_FOLDER")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def test_saving_file_remotely():
    model_dir = "dummy_model.txt"

    # Simulate remote saving
    print(FILE_SERVER_IP, USERNAME, PASSWORD)
    smbclient.register_session(FILE_SERVER_IP, username=USERNAME, password=PASSWORD)

    remote_path = f"//{FILE_SERVER_IP}/{SHARED_FOLDER}/ml-workflow/"
    smbclient.makedirs(remote_path, exist_ok=True)
    remote_path += model_dir

    # Assert connection exists
    assert smbclient.path.exists(f"//{FILE_SERVER_IP}/{SHARED_FOLDER}")

    # Create temp text file and save remotely
    temp_contents = b"Test file contents"
    with smbclient.open_file(remote_path, mode="wb") as remote_f:
        remote_f.write(temp_contents)

    # Assert file was saved and contents match
    assert smbclient.path.exists(remote_path)

    with smbclient.open_file(remote_path, mode="rb") as remote_f:
        assert remote_f.read() == temp_contents

    # Delete the file and check it no longer exists
    smbclient.remove(remote_path)
    assert not smbclient.path.exists(remote_path)
