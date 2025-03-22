import os
import csv
import time
import asyncio
from datetime import datetime

from dotenv import load_dotenv

from typing import List

from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

from trainer import Trainer
from service import Service

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

        load_dotenv()
        __LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        __API_KEY = os.getenv("API_KEY")

        self.ls = LabelStudio(base_url=__LABEL_STUDIO_URL, api_key=__API_KEY)
        self.ls_client = Client(url=__LABEL_STUDIO_URL, api_key=__API_KEY)

        self.service = Service()
        
        self.projects = {}
        self.project_finished_tasks_dict = {}
        self.project_tasks_dif = {}
        self.training_dict = {}
        self.training_queue_set = set()
        self.training_queue = []

        self.project_to_time_of_threshold_reached = {}

        # Create dict of projects and latest completed batch size
        # Check if project_tasks.csv exists and is populated
        if os.path.exists("project_tasks.csv"):
            with open("project_tasks.csv", 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.project_finished_tasks_dict[int(row["id"])] = int(row["finished_tasks"])  
                    self.projects[int(row["id"])] = {
                        'finished_tasks': row["finished_tasks"],
                        'total_tasks': row["total_tasks"],
                        'tracked': row["tracked"] == 'True',
                        'title': row["title"],
                        'date_time_last_trained': row["date_time_last_trained"],
                        'training_duration': row["training_duration"],
                        'epochs': row["epochs"],
                        'locations_saved': row["locations_saved"],
                        'class_acc_string': row["class_acc_string"],
                        'latest_report': row["latest_report"]
                    } 
        else: # Load finished_task data from LabelStudio
            with open("project_tasks.csv", "w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["id","finished_tasks","total_tasks","tracked","title","date_time_last_trained","training_duration","epochs","locations_saved","class_acc_string","latest_report"])
                writer.writeheader()

                projects = self.ls.projects.list()
                webhooks_set = set([webhook.project for webhook in self.ls.webhooks.list()])
                
                for project in projects:
                    # Extracting available project data
                    project_data = {
                        'id': project.id,
                        'finished_tasks': project.num_tasks_with_annotations,
                        'total_tasks': project.task_number,
                        'tracked': project.id in webhooks_set,
                        'title': project.title,
                        'date_time_last_trained': '',
                        'training_duration': '',
                        'epochs': '',
                        'locations_saved': '',
                        'class_acc_string': '',
                        'latest_report': ''
                    }

                    writer.writerow(project_data)
            
    def update_csv_memory(self):
        os.remove("project_tasks.csv")
        with open("project_tasks.csv", "w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["id","finished_tasks","total_tasks","tracked","title","date_time_last_trained","training_duration","epochs","locations_saved","class_acc_string","latest_report"])
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
                        'title': project.title,
                        'date_time_last_trained': local_project['date_time_last_trained'],
                        'training_duration': local_project['training_duration'],
                        'epochs': local_project['epochs'],
                        'locations_saved': local_project['locations_saved'],
                        'class_acc_string': local_project['class_acc_string'],
                        'latest_report': local_project['latest_report']
                    }

                    writer.writerow(project_data)
    
    async def __listen_for_more_annotations_and_train(self, id, trainer:Trainer):
        # Store last amount of annotations made
        last_amount_annotated = self.project_finished_tasks_dict[id]

        # Wait 5-minutes before checking if the number of annotations has increased
        await asyncio.sleep(self.service.minutes_to_wait_for_next_annotation*60)

        # Check if there is the trainer has already began being trained
        if trainer.is_active:
            return

        # Start training if the number of annotations is the same
        if self.project_finished_tasks_dict[id] > last_amount_annotated:
            self.training_dict[id] = self.project_finished_tasks_dict[id]
            self.__listen_for_more_annotations_and_train(id, trainer)
            return
        else:
            GREEN = '\033[32m'
            RESET = '\033[0m'
            print(f"{GREEN}TRAINER {id} BEGAN TRAINING{RESET}")
            async def callback(id, train_output):
                self.project_tasks_dif[id] = 0
                self.project_finished_tasks_dict[id] = last_amount_annotated
                # Store train output in dict
                self.projects[id]['epochs'] = train_output['epochs']
                self.projects[id]['training_duration'] = train_output['training_duration']
                self.projects[id]['class_acc_string'] = train_output['class_acc_string']
                self.projects[id]['latest_report'] = train_output['latest_report']
                self.projects[id]['locations_saved'] = train_output['locations_saved']

                self.training_dict.pop(id)
                self.update_csv_memory()
                await self.check_and_train()

            self.train_calls += 1
            self.projects[id]['date_time_last_trained'] = datetime.now()
            await trainer.train(callback=callback)


    async def check_and_train(self, overrided_project=None):        
        # Override
        if overrided_project is not None:
            id = overrided_project
            if id not in self.training_queue_set and (id not in self.training_dict or val - self.service.batch_size_threshold > self.service.batch_size_threshold):
                    self.training_queue.append(id)
                    self.training_queue_set.add(id)

        for id, val in self.project_tasks_dif.items():
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
                training_tasks.append(asyncio.create_task(self.__listen_for_more_annotations_and_train(id=trainer.project_id, trainer=trainer)))
            
            await asyncio.gather(*training_tasks)
        