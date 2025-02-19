import requests
import shutil
from tqdm import tqdm
import json

from dotenv import load_dotenv
import os

from label_studio_sdk.client import LabelStudio

from scheduler import Scheduler

from flask import Flask, request
import json

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")

app = Flask(__name__)

scheduler = Scheduler()

@app.route("/")
def check_environment():
    # Check for valid Label studio connection

    # Look for USB

    # Check connection to file server

    # Check connection 

    # Go to interface for dynamic reconfiguration
    with open("src/static/index.html", 'r', encoding='utf-8') as file:
        html_string = file.read()
        return html_string

@app.route("/update", methods=['POST'])
def update_made_to_labelstudio():
    data = json(request.data)
    project_id = data['project']
    num_annotations = int(data['num_tasks_with_annotations'])
    
    scheduler.project_tasks_dif[project_id] = abs(num_annotations - scheduler.project_finished_tasks_dict[project_id])
    # Receive webhook and update tracking information
    return {"success": "Called"}, 201

if __name__ == "__main__":
    app.run(debug=True)
