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

class Scheduler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Scheduler, cls).__new__(cls)
        return cls._instance

    def __init__(self, async_processes_allowed:int=1, batch_size:int = 32, minutes_to_wait_before_training:float=5, minimum_annotations_required:int=20):
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

        self.batch_size = batch_size
        self.async_processes_allowed = async_processes_allowed
        self.minutes_to_wait_before_training = minutes_to_wait_before_training
        self.minimum_annotations_required = minimum_annotations_required
        
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
        else: # Load finished_task data from LabelStudio
            with open("project_tasks.csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id","finished_tasks"])

                projects = self.ls.projects
                for p in projects.list():
                    writer.writerow([p.id,p.finished_task_number])
                    self.project_finished_tasks_dict[p.id] = p.finished_task_number
            
    def __update_csv_memory(self):
        os.remove("project_tasks.csv")
        with open("project_tasks.csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id","finished_tasks"])

                projects = self.ls.projects
                for p in projects.list():
                    writer.writerow([p.id,p.finished_task_number])
                    self.project_finished_tasks_dict[p.id] = p.finished_task_number
    
    async def __listen_for_more_annotations_and_train(self, id, trainer:Trainer):
        # Store last amount of annotations made
        last_amount_annotated = self.project_finished_tasks_dict[id]

        # Wait 5-minutes before checking if the number of annotations has increased
        await asyncio.sleep(self.minutes_to_wait_before_training*60)

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
            async def callback(id):
                self.project_tasks_dif[id] = 0
                self.project_finished_tasks_dict[id] = last_amount_annotated
                self.training_dict.pop(id)
                self.__update_csv_memory()
                await self.check_and_train()

            self.train_calls += 1
            await trainer.train(callback=callback)


    async def check_and_train(self):
        for id, val in self.project_tasks_dif.items():
            if val >= self.batch_size and self.project_finished_tasks_dict[id] > self.minimum_annotations_required: # Condition to set for training
                print('**',id, val, self.project_finished_tasks_dict[id])
                # Check if id is not already queued or if the id is training only add it back into the queue if a new batch of data one more batch was labeled while it was training
                if id not in self.training_queue_set and (id not in self.training_dict or val - self.batch_size > self.batch_size):
                    self.training_queue.append(id)
                    self.training_queue_set.add(id)
            else:
                print('-', id, val, self.project_finished_tasks_dict[id])
        
        print(self.training_queue)
        print("Training Q set size: ", len(self.training_queue_set))
        print("Training set size: ", len(self.training_dict.keys()))
        
        # Place next item in training set and begin training
        if len(self.training_dict.keys()) < self.async_processes_allowed and len(self.training_queue) > 0:
            training_tasks = []
            
            while len(self.training_dict.keys()) < self.async_processes_allowed and len(self.training_queue) > 0:
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
        