import pytest
import os
import shutil
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client


load_dotenv()
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")
PROJECT_ID = 19

FILE_SERVER_IP = os.getenv("FILE_SERVER_IP")
SHARED_FOLDER = os.getenv("SHARED_FOLDER")    
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def pytest_configure():
    pytest.PROJECT_ID = PROJECT_ID
    pytest.ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    pytest.ls_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

@pytest.hookimpl()
def pytest_sessionfinish(session):
    print("Cleaning up testing session...")
    allow_delete = input("Delete created folders and files? [Y/n]")
    if allow_delete != 'n' and allow_delete != 'N':
        base_path = f"./gym/project_{PROJECT_ID}"
        local_save_path = "./local-saves"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)  # Delete the entire project directory
            print(f"\nDeleted project directories at {base_path}")
        if os.path.exists(local_save_path): # Delete local save
            shutil.rmtree(local_save_path)
            os.makedirs(local_save_path, exist_ok=True)
            print("Cleared local-saves folder!")
    else:
        print("Test files were not deleted!") 