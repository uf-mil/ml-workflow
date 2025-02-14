import requests
import shutil
from tqdm import tqdm
import json

from dotenv import load_dotenv
import os

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")

from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

# ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

from scheduler import Scheduler

if __name__ == "__main__":
    Scheduler().run()
