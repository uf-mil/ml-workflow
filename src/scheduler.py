import os
import csv
import time
import asyncio
from threading import Thread
from datetime import datetime

from dotenv import load_dotenv

from typing import List

from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

from trainer import Trainer
from service import Service
from logger import Logger

class Scheduler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Scheduler, cls).__new__(cls)
        return cls._instance

    def __init__(self, service:Service):
        """
        Parameters:
            listen_every (millisecond): sleep period after every call to LabelStudio to get most recent training data.

            batch_size: number of finished tasks that need to be completed to trigger a new training cycle.

        The Scheduler handles calling the Trainer after a certain number of labeling tasks are completed per project. 
        """

        if hasattr(self, '_initialized') and self._initialized:
            return
        
        #DEBUG
        self.train_calls = 0

        self._initialized = True

        try:
            self.ls = LabelStudio(base_url=service.label_studio_url, api_key=service.label_studio_api_key)
            self.ls_client = Client(url=service.label_studio_url, api_key=service.label_studio_api_key)
        except Exception:
            self.ls = None
            self.ls_client = None

        self.service = Service()
        
        self.projects = {}
        self.project_finished_tasks_dict = {}
        self.training_dict = {}
        self.trainer_dict = {}
        self.training_queue_set = set()
        self.training_queue = []

        self.project_to_time_of_threshold_reached = {}

        # Create dict of projects and latest completed batch size
        # Check if project_tasks.csv exists and is populated
        webhooks_set = set([webhook.project for webhook in self.ls.webhooks.list()])

        if os.path.exists("project_tasks.csv"):
            project_ids = [int(project.id) for project in self.ls.projects.list()]
            with open("project_tasks.csv", 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.project_finished_tasks_dict[int(row["id"])] = int(row["finished_tasks"])  
                    self.projects[int(row["id"])] = {
                        'finished_tasks': row["finished_tasks"],
                        'total_tasks': row["total_tasks"],
                        'tracked': row["tracked"] == 'True',
                        'tracked_annotations': int(row['tracked_annotations']),
                        'title': row["title"],
                        'date_time_last_trained': row["date_time_last_trained"],
                        'training_duration': row["training_duration"],
                        'epochs': row["epochs"],
                        'locations_saved': row["locations_saved"],
                        'location_of_metrics': row["location_of_metrics"],
                        'class_acc_string': row["class_acc_string"],
                        'latest_report': row["latest_report"]
                    }
                    if int(row['id']) in project_ids:
                        project_ids.remove(int(row['id']))
            
            if len(project_ids) > 0:
                print(project_ids)
                for id in project_ids:
                    project = self.ls.projects.get(id)
                    self.__store_project(project, webhooks_set)


        else: # Load finished_task data from LabelStudio
            with open("project_tasks.csv", "w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["id","finished_tasks","total_tasks","tracked","tracked_annotations","title","date_time_last_trained","training_duration","epochs","locations_saved","location_of_metrics","class_acc_string","latest_report"])
                writer.writeheader()

                projects = self.ls.projects.list()
                
                for project in projects:
                    # Extracting available project data
                    project_data = {
                        'id': project.id,
                        'finished_tasks': project.num_tasks_with_annotations,
                        'total_tasks': project.task_number,
                        'tracked': project.id in webhooks_set,
                        'tracked_annotations': 0,
                        'title': project.title,
                        'date_time_last_trained': '',
                        'training_duration': '',
                        'epochs': '',
                        'locations_saved': '',
                        'location_of_metrics': '',
                        'class_acc_string': '',
                        'latest_report': ''
                    }

                    writer.writerow(project_data)
            
    def update_csv_memory(self):
        os.remove("project_tasks.csv")
        with open("project_tasks.csv", "w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["id","finished_tasks","total_tasks","tracked","tracked_annotations","title","date_time_last_trained","training_duration","epochs","locations_saved","location_of_metrics","class_acc_string","latest_report"])
                writer.writeheader()

                projects = self.ls.projects.list()
                webhooks_set = set([webhook.project for webhook in self.ls.webhooks.list()])
                
                for project in projects:
                    # Extracting available project data
                    local_project = self.projects[project.id]
                    project_data = {
                        'id': project.id,
                        'finished_tasks': project.num_tasks_with_annotations,
                        'total_tasks': project.task_number,
                        'tracked': project.id in webhooks_set,
                        'tracked_annotations': local_project['tracked_annotations'],
                        'title': project.title,
                        'date_time_last_trained': local_project['date_time_last_trained'],
                        'training_duration': local_project['training_duration'],
                        'epochs': local_project['epochs'],
                        'locations_saved': local_project['locations_saved'],
                        'location_of_metrics': local_project['location_of_metrics'],
                        'class_acc_string': local_project['class_acc_string'],
                        'latest_report': local_project['latest_report']
                    }

                    writer.writerow(project_data)
    
    async def stop_project_in_training(self, project_id):
        self.trainer_dict[project_id].will_cancel = True

    async def __listen_for_more_annotations_and_train(self, id, trainer:Trainer):
        try:
            # Store last amount of annotations made
            last_amount_annotated = self.project_finished_tasks_dict[id]

            # Wait 5-minutes before checking if the number of annotations has increased
            await asyncio.sleep(self.service.minutes_to_wait_for_next_annotation*60)

            # Check if there is the trainer has already began being trained
            if trainer.is_active:
                return

            # Start training if the number of annotations is the same
            if self.project_finished_tasks_dict[id] > last_amount_annotated:
                print("Listening for more annotations...")
                self.training_dict[id] = self.project_finished_tasks_dict[id]
                await self.__listen_for_more_annotations_and_train(id, trainer)
                return
            else:
                GREEN = '\033[32m'
                RESET = '\033[0m'
                print(f"{GREEN}TRAINER {id} BEGAN TRAINING{RESET}")
                async def callback(id, train_output):
                    self.project_finished_tasks_dict[id] = last_amount_annotated
                    # Store train output in dict
                    self.projects[id]['epochs'] = train_output['epochs']
                    self.projects[id]['training_duration'] = train_output['training_duration']
                    self.projects[id]['class_acc_string'] = train_output['class_acc_string']
                    self.projects[id]['latest_report'] = train_output['latest_report']
                    self.projects[id]['locations_saved'] = train_output['locations_saved']
                    self.projects[id]['location_of_metrics'] = train_output['location_of_metrics']
                    self.projects[id]['tracked_annotations'] = 0

                    self.training_dict.pop(id)
                    self.trainer_dict.pop(id)
                    self.update_csv_memory()
                    await self.check_and_train()

                self.train_calls += 1
                self.projects[id]['date_time_last_trained'] = datetime.now()
                await trainer.train(callback=callback)
        except asyncio.CancelledError:
            # Log the cancellation
            Logger().log_training_cancellation(trainer)
            self.projects[id]['latest_report'] = trainer.return_dict['latest_report']
            self.update_csv_memory()
            
            self.training_dict.pop(id)
            self.trainer_dict.pop(id)
            
            trainer.leave_gym()
            print(f"Training cancelled for {id}")
            await self.check_and_train()
            return


    async def check_and_train(self, overrided_project=None):        
        # Override
        if overrided_project is not None:
            id = overrided_project
            if id not in self.training_queue_set and id not in self.training_dict:
                    self.training_queue.append(id)
                    self.training_queue_set.add(id)

        for id, val in self.projects.items():
            val = int(val["tracked_annotations"])
            if val >= self.service.batch_size_threshold and self.project_finished_tasks_dict[id] > self.service.minimum_annotations_required: # Condition to set for training
                print('**',id, val, self.project_finished_tasks_dict[id])
                # Check if id is not already queued or if the id is training only add it back into the queue if a new batch of data one more batch was labeled while it was training
                if id not in self.training_queue_set and (id not in self.training_dict or val - self.service.batch_size_threshold > self.service.batch_size_threshold):
                    self.training_queue.append(id)
                    self.training_queue_set.add(id)
            else:
                print('-', id, val, self.project_finished_tasks_dict[id])
        
        print(self.training_queue)
        print("Training Q set size: ", len(self.training_queue_set))
        print("Training set size: ", len(self.training_dict.keys()))
        
        # Place next item in training set and begin training
        if len(self.training_dict.keys()) < self.service.async_processes_allowed and len(self.training_queue) > 0:
            training_tasks = []
            
            while len(self.training_dict.keys()) < self.service.async_processes_allowed and len(self.training_queue) > 0:
                id = self.training_queue[0]

                # If project already being trained don't create a trainer for it yet
                if id in self.training_dict:
                    continue
                
                self.training_dict[id] = self.project_finished_tasks_dict[id]
                id = self.training_queue.pop(0)
                self.training_queue_set.remove(id)
                
                self.project_to_time_of_threshold_reached[id] = datetime.now()
                trainer = Trainer(id, self.ls, self.ls_client)
                task = asyncio.create_task(self.__listen_for_more_annotations_and_train(id=trainer.project_id, trainer=trainer))
                training_tasks.append(task)
                self.trainer_dict[id] = trainer
            
            try:
                await asyncio.gather(*training_tasks)
            except asyncio.CancelledError:
                print("CANCELLED HERE")
                raise RuntimeError(f"Cancelled training")
    
    def get_data_spread(self, project_id):
        project = self.ls_client.get_project(project_id)
        tasks = project.get_labeled_tasks()
        classes = project.parsed_label_config["label"]["labels"]
        
        freq_dict = {}
        for cls in classes:
            freq_dict[cls] = 0
        
        for task in tasks:
            annotations = task['annotations'][0]['result']
            for obj in annotations:
                obj = obj['value']
                label = obj["rectanglelabels"][0]
                freq_dict[label] += 1

        return freq_dict
    
    
    def get_projects(self):
        external_projects = {p.id: p for p in self.ls_client.get_projects()}
        webhooks_set = set([webhook.project for webhook in self.ls.webhooks.list()])
        external_ids = set(external_projects.keys())
        internal_ids = set(self.projects.keys())
        
        external_only = external_ids - internal_ids
        local_only = internal_ids - external_ids

        for id in local_only:
            del self.projects[id]
        
        for id in external_only:
            self.__store_project(external_projects[id], webhooks_set)
        
        if len(local_only) > 0:
            self.update_csv_memory()
        
    def __store_project(self, project, webhooks_set):
        self.projects[project.id] = {
                'finished_tasks': project.num_tasks_with_annotations,
                'total_tasks': project.task_number,
                'tracked': project.id in webhooks_set,
                'tracked_annotations': 0,
                'title': project.title,
                'date_time_last_trained': '',
                'training_duration': '',
                'epochs': '',
                'locations_saved': '',
                'location_of_metrics': '',
                'class_acc_string': '',
                'latest_report': ''
            }
        with open("project_tasks.csv", "a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["id","finished_tasks","total_tasks","tracked","tracked_annotations","title","date_time_last_trained","training_duration","epochs","locations_saved","location_of_metrics","class_acc_string","latest_report"])
            project_data = {
                'id': project.id,
                'finished_tasks': project.num_tasks_with_annotations,
                'total_tasks': project.task_number,
                'tracked': project.id in webhooks_set,
                'tracked_annotations': 0,
                'title': project.title,
                'date_time_last_trained': '',
                'training_duration': '',
                'epochs': '',
                'locations_saved': '',
                'location_of_metrics': '',
                'class_acc_string': '',
                'latest_report': ''
            }

            writer.writerow(project_data)

        