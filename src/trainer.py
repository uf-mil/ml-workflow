import yaml
import os

import requests
import shutil
import asyncio
from tqdm import tqdm
import random
from datetime import datetime
import pandas as pd

import torch
import traceback
from ultralytics import YOLO

from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

from transporter import ModelTransporter
from logger import Logger

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("API_KEY")
USB_KEY_FILENAME = os.getenv("USB_KEY_FILENAME")

class Trainer:
    def __init__(self, project_id:int, ls:LabelStudio, ls_client:Client):
        """
        Parameters:
            project_id: the id associated with the LabelStudio project.

        Handles creating the yaml file, loading in images and labels into train, validate, and test, and training on the best model for the project. 
        """
        print(f"[INFO]: Created trainer for project {project_id}: {ls.projects.get(project_id).title}")
        
        self.project_client = ls_client.get_project(id=project_id)
        self.project_id = project_id
        self.project = ls.projects.get(id=project_id)
        
        self.ls_client = ls_client
        self.ls = ls
        
        self.labels = list(list(ls.projects.get(project_id).get_label_interface().labels)[0].keys())
        self.data_count_map = {}

        self.model = YOLO("./models/yolo11n.pt")
        self.return_dict = {
            "epochs": None,
            "training_duration": None,
            "class_acc_string": None,
            "latest_report": None,
            "locations_saved": None,
            "location_of_metrics": None
        }
        self.save_folder = f"project_{project_id}"

        self.is_active = False

        self.will_cancel = False

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
        yaml_data = {
            'train': "images/train",
            'test': "images/test",
            'val': "images/val",
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
            annotations = task['annotations']
            
            if not annotations:
                return
            
            annotations = annotations[0]['result']
            
            img_filename = f"{i}.jpg"
            img_path = os.path.join(f"./gym/project_{self.project_id}/images/{group_type}", img_filename)
            label_path = os.path.join(f"./gym/project_{self.project_id}/labels/{group_type}", f"{i}.txt")
            
            download_file(LABEL_STUDIO_URL+img_url, img_path)
            with open(label_path, "w") as f:
                yolo_data = convert_to_yolo(annotations)
                f.writelines(yolo_data)
            
        def split_list(data, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=None):
            print("[INFO]: Splitting tasks into train/test/val sets.")
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

            self.data_count_map = {
                "total":len(tasks),
                "train":len(train_set),
                "test":len(test_set),
                "val":len(val_set)
            }
            
            return train_set, test_set, val_set

        train, test, val = split_list(tasks)
        
        for i, task in enumerate(tqdm(train)):
            save_img_label_pair(i, task, 'train')
        
        for i, task in enumerate(tqdm(test)):
            save_img_label_pair(i, task, 'test')
              
        for i, task in enumerate(tqdm(val)):
            save_img_label_pair(i, task, 'val')

    def __store_model(self, metrics_path)->str:
        log_msg, locations = ModelTransporter(self.save_folder).full_save(
            self.model, 
            f"project_{self.project_id}.pt", 
            metrics_path,
            self.project_id)
        self.return_dict['locations_saved'] = locations['model']
        self.return_dict['location_of_metrics'] = locations['metrics']
        return log_msg
    
    def __record_num_epochs(self, path):
        # Load in the results csv file and read the number of rows - 2
        df = pd.read_csv(path)
        num_epochs = len(df)
        self.return_dict["epochs"] = num_epochs


    async def begin_training(self):
        cwd = os.getcwd()
        print("Current working directory:", cwd)
        path = cwd + f'/gym/project_{self.project_id}/data.yaml'

        if not os.path.exists(path):
            print("Path does not exist...")
            return
        
        # Keep on training until no improvement is seen in ten epochs
        try:
            def check_for_cancellation(data):
                if self.will_cancel:
                    raise asyncio.CancelledError("Training Cancelled")
                
            start = datetime.now()
            self.model.add_callback("on_train_epoch_end", check_for_cancellation)
            results = self.model.train(
                data = path,
                epochs = 10,
                patience = 5,
                batch = -1,
                device = "cuda" if torch.cuda.is_available() else "cpu",
                project = cwd + f"/gym/project_{self.project_id}/runs"
            )
            duration =  datetime.now() - start
            self.return_dict["training_duration"] = str(duration)
            self.__record_num_epochs(results.save_dir / "results.csv")
            # Save the model to some location
            storing_output = self.__store_model(results.save_dir)

            # Log the training session
            Logger().log_training_success(results, self, storing_output)
        except Exception as e:
            print("LOGGING ERROR")
            Logger().log_training_error(e, self)
        
    def leave_gym(self):
        base_path = f"./gym/project_{self.project_id}"
        if os.path.exists(base_path):
            shutil.rmtree(base_path)  # Delete the entire project directory
            print(f"\nDeleted project directories at {base_path}")

    async def train(self, callback = None):
        self.is_active = True
        try:
            print("[INFO]: Creating yaml file for training")
            self.create_yaml()

            print("[INFO]: Downloading and organizing data from LabelStudio...")
            self.get_and_organize_data()

            #TODO: When a training session finishes remove the project id from the training set
            print("[INFO]: Training model on tiny...")
            await self.begin_training()

            print("[INFO]: Cleaning up project directory from the gym...")
            self.leave_gym()
        except Exception as e:
            print(f"[ERROR]: {e}")
            print(traceback.print_exc())
            return
        finally:
            if callback and callable(callback) and not self.will_cancel:
                await callback(self.project_id, self.return_dict)
            else:
                print("[ERROR] Did not call callback")
                raise asyncio.CancelledError
        
        
        
