import yaml
import os

import requests
import shutil
from tqdm import tqdm
import random

from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")

class Trainer:
    def __init__(self, project_id:int, ls:LabelStudio, ls_client:Client):
        """
        Parameters:
            project_id: the id associated with the LabelStudio project.

        Handles creating the yaml file, loading in images and labels into train, validate, and test, and training on the best model for the project. 
        """
        print(f"[INFO]: Created trainer for project {project_id}: {ls.projects.get(project_id).title}")
        self.project_id = project_id
        self.project = ls.projects.get(id=project_id)
        self.project_client = ls_client.get_project(id=project_id)
        self.ls = ls
        self.ls_client = ls_client
        self.labels = list(list(ls.projects.get(24).get_label_interface().labels)[0].keys())

        try:
            os.makedirs(f"./gym/project_{project_id}/images/train",exist_ok=True)
            os.makedirs(f"./gym/project_{project_id}/images/test",exist_ok=True)
            os.makedirs(f"./gym/project_{project_id}/images/val",exist_ok=True)
            os.makedirs(f"./gym/project_{project_id}/labels/train",exist_ok=True)
            os.makedirs(f"./gym/project_{project_id}/labels/test",exist_ok=True)
            os.makedirs(f"./gym/project_{project_id}/labels/val",exist_ok=True)
        except Exception as e:
            print("[ERROR]: There was an issue creating the directories to train the model:")
            print(e)
            raise Exception("Could not initiate training!")

    def create_yaml(self):
        print("[INFO]: Creating yaml file for training...")
        yaml_data = {
            'train': f"./gym/project_{self.project_id}/images/train",
            'test': f"./gym/project_{self.project_id}/images/test",
            'val': f"./gym/project_{self.project_id}/images/val",
            'nc':len(self.project.parsed_label_config["label"]["labels"]),
            'names':self.labels
        }

        with open(f'./gym/project_{self.project_id}/data.yaml', 'w') as file:
            yaml.dump(yaml_data, file)
    
    def get_and_organize_data(self):
        tasks = self.project_client.get_tasks()

        def download_file(url, output_path):
            headers = {"Authorization": f"Token {API_KEY}"}  # Add authentication header
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            else:
                print(f"Failed to download {url}: {response.status_code}, {response.text}")
        
        def convert_to_yolo(annotation):
            """ Convert Label Studio bbox annotation to YOLO format """
            yolo_annotations = []
            for obj in annotation:
                obj = obj['value']
                if "rectanglelabels" in obj:
                    label = self.labels.index(obj["rectanglelabels"][0])
                    x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
                    x_center, y_center = (x + w / 2) / 100, (y + h / 2) / 100
                    w, h = w / 100, h / 100
                    yolo_annotations.append(f"{label} {x_center} {y_center} {w} {h}\n")
            return yolo_annotations
        
        def save_img_label_pair(i, task, group_type):
            img_url = task['data']['image']
            annotations = task['annotations'][0]['result']
            
            if not annotations:
                return
            
            img_filename = f"{i}.jpg"
            img_path = os.path.join(f"./gym/project_{self.project_id}/images/{group_type}", img_filename)
            label_path = os.path.join(f"./gym/project_{self.project_id}/labels/{group_type}", f"{i}.txt")
            
            download_file(LABEL_STUDIO_URL+img_url, img_path)
            with open(label_path, "w") as f:
                yolo_data = convert_to_yolo(annotations)
                print(yolo_data)
                f.writelines(yolo_data)

        def split_list(data, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=None):
            if not (0 <= train_ratio <= 1 and 0 <= test_ratio <= 1 and 0 <= val_ratio <= 1):
                raise ValueError("Ratios must be between 0 and 1")
            if abs((train_ratio + test_ratio + val_ratio) - 1.0) > 1e-6:
                raise ValueError("Ratios must sum to 1")
            
            if seed is not None:
                random.seed(seed)
            
            random.shuffle(data)
            
            train_end = int(len(data) * train_ratio)
            test_end = train_end + int(len(data) * test_ratio)
            
            train_set = data[:train_end]
            test_set = data[train_end:test_end]
            val_set = data[test_end:]
            
            return train_set, test_set, val_set
        # Place tasks into train/test/val

        train, test, val = split_list(tasks)
        
        for i, task in enumerate(tqdm(train)):
            save_img_label_pair(i, task, 'train')
        
        for i, task in enumerate(tqdm(test)):
            save_img_label_pair(i, task, 'test')
        
        for i, task in enumerate(tqdm(val)):
            save_img_label_pair(i, task, 'val')


    async def train(self):
        print("[INFO]: Creating importing data to folders...")
        self.create_yaml()

        print("[INFO]: Downloading and organizing data from LabelStudio...")
        self.get_and_organize_data()

        #TODO: When a training session finishes remove the project id from the training set
        print("[INFO]: Training model on tiny...")
        
        
        
