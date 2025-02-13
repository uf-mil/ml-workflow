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
    # while True:
    #     ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    #     projects = ls.projects.list()
    #     print(len(projects.items))
    # Scheduler(batch_size=1)
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    project = ls_client.get_project(24)

    tasks = project.get_tasks()
    labels = list(list(ls.projects.get(24).get_label_interface().labels)[0].keys())
    print(labels)

    def download_file(url, output_path):
        headers = {"Authorization": f"Token {API_KEY}"}  # Add authentication header
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
        else:
            print(f"Failed to download {url}: {response.status_code}, {response.text}")
    
    def convert_to_yolo(annotation, all_labels):
        """ Convert Label Studio bbox annotation to YOLO format """
        yolo_annotations = []
        for obj in annotation:
            obj = obj['value']
            if "rectanglelabels" in obj:
                label = labels.index(obj["rectanglelabels"][0])
                x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
                x_center, y_center = (x + w / 2) / 100, (y + h / 2) / 100
                w, h = w / 100, h / 100
                yolo_annotations.append(f"{label} {x_center} {y_center} {w} {h}\n")
        return yolo_annotations
    
    for i, task in enumerate(tqdm(tasks)):
        img_url = task['data']['image']
        annotations = task['annotations'][0]['result']
        
        if not annotations:
            continue
        
        img_filename = f"{i}.jpg"
        img_path = os.path.join("./", img_filename)
        label_path = os.path.join("./", f"{i}.txt")
        
        download_file(LABEL_STUDIO_URL+img_url, img_path)
        with open(label_path, "w") as f:
            yolo_data = convert_to_yolo(annotations, labels)
            print(yolo_data)
            f.writelines(yolo_data)

    # for p in projects:
    #     print(p.title, p.id, p.task_number, p.finished_task_number, p.parsed_label_config["label"]["labels"])
    #     # print(p)
