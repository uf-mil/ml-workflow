import os
import asyncio
import traceback
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv, set_key

from scheduler import Scheduler
from service import Service
from transporter import ModelTransporter
from memoryHandler import MemoryHandler

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

SERVICE = Service()
SCHEDULER = Scheduler(service=SERVICE)

@app.route("/")
def check_environment():
    print(SERVICE.dark_mode)
    return render_template(
        'index.html',
        dark_mode=SERVICE.dark_mode
    )

@app.route("/settings")
def show_settings():
    return render_template(
        'settings.html',
        label_studio_url=SERVICE.label_studio_url,
        file_server_ip=SERVICE.file_server_ip,
        file_server_port=SERVICE.file_server_port,
        file_server_shared_folder=SERVICE.file_server_shared_folder,
        usb_key_file_name=SERVICE.usb_key_file_name,
        async_processes_allowed=SERVICE.async_processes_allowed,
        batch_size_threshold=SERVICE.batch_size_threshold,
        minutes_to_wait_for_next_annotation=SERVICE.minutes_to_wait_for_next_annotation,
        minimum_annotations_required=SERVICE.minimum_annotations_required,
        dark_mode = SERVICE.dark_mode
    )

@app.route("/project-<project_id>")
def project_overview(project_id):
    # Example data, replace with actual data fetching logic
    project_id = int(project_id)
    project = SCHEDULER.projects[project_id]

    project_data = {
        "project_name": project['title'],
        "project_id": project_id,
        "num_annotations": project['finished_tasks'],
        "total_tasks": project['total_tasks'],
        "project_status": 3 if project_id in SCHEDULER.training_dict else 
                            2 if project_id in SCHEDULER.training_queue_set else 
                            1 if project["tracked"] == True else 
                            0,
        # Project Metadata
        "training_duration": project["training_duration"],
        'date_time_last_trained': project['date_time_last_trained'],
        "epochs": project["epochs"],
        "locations_saved": project["locations_saved"],
        "latest_report": project["latest_report"],
        # Collected data
        "class_acc_string": project["class_acc_string"],
        "dark_mode":SERVICE.dark_mode
    }

    return render_template("project.html", **project_data)

@app.route("/get-latest-results-for-<project_id>")
def get_latest_results_for(project_id):
    results_data = MemoryHandler().pull_latest_results_for(project_id)
    return jsonify(results_data), 200

@app.route("/link-<project_id>")
def link_project(project_id):
    try:
        #TODO: Check if a webhook for this project with the link already exists
        
        SCHEDULER.ls.webhooks.create(
            url=f'{request.url_root}update',
            project=int(project_id),
            send_payload=True,
            send_for_all_actions=False,
            actions=[
                'ANNOTATION_CREATED',
                'ANNOTATIONS_DELETED',
                'ANNOTATIONS_CREATED',
                'ANNOTATIONS_DELETED',
                'PROJECT_UPDATED'
            ]
        )
        SCHEDULER.projects[int(project_id)]['tracked'] = True
        SCHEDULER.update_csv_memory()
        return jsonify({"message": "Link made successfully!"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": f"Error creating linking: {str(e)}"}), 500

@app.route("/unlink-<project_id>")
def unlink_project(project_id):
    try:
        webhooks = SCHEDULER.ls.webhooks.list()
        for wh in webhooks:
            print(wh.project, wh)
            if wh.project == int(project_id):
                print("deleted ", project_id)
                SCHEDULER.ls.webhooks.delete(wh.id)
        
        SCHEDULER.projects[int(project_id)]['tracked'] = False
        SCHEDULER.update_csv_memory()
        return jsonify({"message": "Link broken successfully!"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": f"Error breaking linking: {str(e)}"}), 500

@app.route('/train-<project_id>')
async def train_project(project_id):
    try:
        project_id = int(project_id)
        await SCHEDULER.check_and_train(overrided_project=int(project_id))
        location_of_metrics = SCHEDULER.projects[project_id]['location_of_metrics']
        json_data = {}
        if location_of_metrics != '':
            df = pd.read_csv(location_of_metrics + '/results.csv')
            json_data = df.to_dict(orient="records")  # Convert rows to a list of dictionaries
        return jsonify(json_data), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Error in manual train init: {str(e)}"}), 500

@app.route('/stop-<project_id>')
async def stop_project_from_training(project_id):
    # Check if project is in training queue
    project_id = int(project_id)
    if project_id in SCHEDULER.training_queue_set:
        SCHEDULER.training_queue_set.remove(project_id)
        SCHEDULER.training_queue.remove(project_id)
        return {'success': True}, 200
    elif project_id in SCHEDULER.training_dict:
        await SCHEDULER.stop_project_in_training(project_id)
        return {'success': True}, 200
    else:
        return {'warn': "Project not found"}, 404

@app.route('/listen-for-<project_id>')
def listen_for_id(project_id):
    project_id = int(project_id)
    project = SCHEDULER.projects[project_id]
    project['training_duration'] = str(datetime.now() - project['date_time_last_trained']) if type(project['date_time_last_trained']) is not str  else project['training_duration']
    project['in-queue'] = SCHEDULER.training_queue.index(int(project_id)) if int(project_id) in SCHEDULER.training_queue_set else -1
    project['training'] = int(project_id) in SCHEDULER.training_dict

    return project, 201

@app.route('/update-settings', methods=['POST'])
def update_settings():
    try:
        settings = request.json
        env_path = os.path.join(os.getcwd(), 'src', '.env')

        for key, value in settings.items():
            set_key(env_path, key, str(value))
        
        # Configure LabelStudio
        SERVICE.label_studio_url = settings['LABEL_STUDIO_URL']
        
        # Configure file server
        SERVICE.file_server_ip = settings['FILE_SERVER_IP']
        SERVICE.file_server_port = settings['FILE_SERVER_PORT']
        SERVICE.file_server_shared_folder = settings['SHARED_FOLDER']

        # Configure USB device detection
        SERVICE.usb_key_file_name = settings['USB_KEY_FILENAME']

        # Configure auto training logic
        SERVICE.async_processes_allowed = int(settings['ASYNC_PROCESSES_ALLOWED'])
        SERVICE.batch_size_threshold = int(settings['BATCH_SIZE_THRESHOLD'])
        SERVICE.minutes_to_wait_for_next_annotation = float(settings['MINUTES_TO_WAIT_FOR_NEXT_ANNOTATION']) 
        SERVICE.minimum_annotations_required = int(settings['MINIMUM_ANNOTATIONS_REQUIRED'])

        # Configure dark mode
        SERVICE.dark_mode = bool(settings["DARK_MODE"])

        return jsonify({"message": "Settings updated successfully!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error updating settings: {str(e)}"}), 500


@app.route("/get-data", methods=['GET'])
def get_data():
    # Get device availability
    checker = ModelTransporter('', service=SERVICE)
    
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
        'title': values["title"],
        'id': id,
        'num_tasks_with_annotations': values["finished_tasks"],
        'task_number': values["total_tasks"],
        'state':3 if id in SCHEDULER.training_dict else 
                2 if id in SCHEDULER.training_queue_set else 
                1 if values["tracked"] == True else 
                0
    } for id, values in SCHEDULER.projects.items()]

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

    log_path = os.path.join(os.getcwd(),"logs", log_name)

    if not os.path.exists(log_path):
        return jsonify({'error': 'Log file not found'}), 404

    try:
        with open(log_path, 'r') as log_file:
            content = log_file.readlines()

        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/update", methods=['POST'])
async def update_made_to_labelstudio():
    data = request.get_json()
    project_id = data['project']['id']
    num_annotations = int(data['project']['num_tasks_with_annotations'])
    
    # Receive webhook and update tracking information
    SCHEDULER.project_tasks_dif[project_id] = abs(num_annotations - SCHEDULER.project_finished_tasks_dict[project_id])
    
    print(project_id, num_annotations, SCHEDULER.project_finished_tasks_dict[project_id], SCHEDULER.project_tasks_dif[project_id])
    
    # Check to start training
    await SCHEDULER.check_and_train()

    print("NUMBER OF TRAIN CALLS MADE: ", SCHEDULER.train_calls)
    
    return {"success": "Called"}, 201

if __name__ == "__main__":

    app.run(debug=True)
