import os
import asyncio

from dotenv import load_dotenv

from scheduler import Scheduler
from transporter import ModelTransporter

from flask import Flask, request, jsonify

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")

# Scheduler settings
ASYNC_PROCESSES_ALLOWED = 3
BATCH_SIZE_THRESHOLD = 32
MINUTES_TO_WAIT_BEFORE_TRAINING = 0.1
MINIMUM_ANNOTATIONS_REQUIRED = 10

app = Flask(__name__)

scheduler = Scheduler(batch_size=BATCH_SIZE_THRESHOLD,
                      async_processes_allowed=ASYNC_PROCESSES_ALLOWED,
                      minutes_to_wait_before_training=MINUTES_TO_WAIT_BEFORE_TRAINING,
                      minimum_annotations_required=MINIMUM_ANNOTATIONS_REQUIRED)

@app.route("/")
def check_environment():
    with open("src/static/index.html", 'r', encoding='utf-8') as file:
        html_string = file.read()
        return html_string

@app.route("/get-data", methods=['GET'])
def get_data():
    # Get device availability
    checker = ModelTransporter('')
    
    devices = [{
        'name': 'Local Storage',
        'available': True,
        'priority': 3
    }, {
        'name': f'File Server @ {os.getenv("FILE_SERVER_IP")}',
        'available': checker.is_file_server_available(),
        'priority': 2
    }, {
        'name': 'USB Device',
        'available': checker.scan_for_available_usb_device() != None,
        'priority': 1
    }]

    # Get monitored projects
    projects = [{
        'title': project.title,
        'id': project.id,
        'num_tasks_with_annotations': project.num_tasks_with_annotations,
        'task_number': project.task_number,
        'state':3 if project.id in scheduler.training_dict else 
                2 if project.id in scheduler.training_queue_set else 
                1 if project.id in scheduler.project_tasks_dif else 
                0
    } for project in scheduler.ls.projects.list()]

    # Get logs
    log_files = os.listdir(os.path.join(os.getcwd(),"logs"))
    log_files.sort()
    log_files.reverse()
    logs = [{'name': file} for file in log_files if file != 'example.txt']

    data = {
        'devices':devices,
        'projects': projects,
        'logs': logs
    }

    return data, 201

@app.route('/get-log-content', methods=['GET'])
def get_log_content():
    log_name = request.args.get('name')
    offset = int(request.args.get('offset', 0))
    chunk_size = 1024  # Size of each chunk in bytes

    log_path = os.path.join(os.getcwd(),"logs", log_name)

    if not os.path.exists(log_path):
        return jsonify({'error': 'Log file not found'}), 404

    try:
        with open(log_path, 'r') as log_file:
            log_file.seek(offset)
            chunk = log_file.read(chunk_size)
            new_offset = log_file.tell()

        return jsonify({'content': chunk, 'new_offset': new_offset})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/update", methods=['POST'])
async def update_made_to_labelstudio():
    data = request.get_json()
    project_id = data['project']['id']
    num_annotations = int(data['project']['num_tasks_with_annotations'])
    
    # Receive webhook and update tracking information
    scheduler.project_tasks_dif[project_id] = abs(num_annotations - scheduler.project_finished_tasks_dict[project_id])
    
    print(project_id, num_annotations, scheduler.project_finished_tasks_dict[project_id], scheduler.project_tasks_dif[project_id])
    
    # Check to start training
    await scheduler.check_and_train()

    print("NUMBER OF TRAIN CALLS MADE: ", scheduler.train_calls)
    
    return {"success": "Called"}, 201

if __name__ == "__main__":
    app.run(debug=True)
