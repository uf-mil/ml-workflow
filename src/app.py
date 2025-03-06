import os
import asyncio

from dotenv import load_dotenv

from scheduler import Scheduler

from flask import Flask, request

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")

# Scheduler settings
ASYNC_PROCESSES_ALLOWED = 2
BATCH_SIZE_THRESHOLD = 1
MINUTES_TO_WAIT_BEFORE_TRAINING = 0.1
MINIMUM_ANNOTATIONS_REQUIRED = 20

app = Flask(__name__)

scheduler = Scheduler(batch_size=BATCH_SIZE_THRESHOLD,
                      async_processes_allowed=ASYNC_PROCESSES_ALLOWED,
                      minutes_to_wait_before_training=MINUTES_TO_WAIT_BEFORE_TRAINING,
                      minimum_annotations_required=MINIMUM_ANNOTATIONS_REQUIRED)

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
    data = request.get_json()
    project_id = data['project']['id']
    num_annotations = int(data['project']['num_tasks_with_annotations'])
    
    # Receive webhook and update tracking information
    scheduler.project_tasks_dif[project_id] = abs(num_annotations - scheduler.project_finished_tasks_dict[project_id])
    
    print(project_id, num_annotations, scheduler.project_finished_tasks_dict[project_id], scheduler.project_tasks_dif[project_id])
    
    # Check to start training
    loop = asyncio.new_event_loop()
    tasks = [loop.create_task(scheduler.check_and_train())]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    
    return {"success": "Called"}, 201

if __name__ == "__main__":
    app.run(debug=True)
