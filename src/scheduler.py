import os
import csv
import time
from datetime import datetime

from dotenv import load_dotenv

from typing import List

from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

from trainer import Trainer

MINUTES_TO_WAIT_BEFORE_TRAINING = 5

class Scheduler:
    def __init__(self, async_processes_allowed:int=1, batch_size:int = 32):
        """
        Parameters:
            listen_every (millisecond): sleep period after every call to LabelStudio to get most recent training data.

            batch_size: number of finished tasks that need to be completed to trigger a new training cycle.

        The Scheduler handles calling the Trainer after a certain number of labeling tasks are completed per project. 
        """

        load_dotenv()
        __LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        __API_KEY = os.getenv("API_KEY")

        self.ls = LabelStudio(base_url=__LABEL_STUDIO_URL, api_key=__API_KEY)
        self.ls_client = Client(url=__LABEL_STUDIO_URL, api_key=__API_KEY)

        self.batch_size = batch_size
        self.async_processes_allowed = async_processes_allowed
        
        self.project_finished_tasks_dict = {}
        self.project_tasks_dif = {}
        self.training_set = set()
        self.training_queue_set = set()
        self.training_queue = []

        self.project_to_time_of_threshold_reached = {}

        # Create dict of projects and latest completed batch size
        # Check if project_tasks.csv exists and is populated
        if os.path.exists("project_tasks.csv"):
            with open("project_tasks.csv", 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.project_finished_tasks_dict[row["id"]] = row["finished_tasks"]   
        else: # Load finished_task data from LabelStudio
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
        time.sleep(MINUTES_TO_WAIT_BEFORE_TRAINING*60)

        # Start training if the number of 
        if self.project_finished_tasks_dict[id] > last_amount_annotated:
            self.__listen_for_more_annotations_and_train(id, trainer)
            return
        else:
            trainer.train(callback=lambda id: self.training_set.remove(id))
            self.project_tasks_dif[id] = 0
            self.project_finished_tasks_dict[id] = last_amount_annotated


    async def check_and_train(self):
        for id, val in self.project_tasks_dif:
            if val >= self.batch_size: # Condition to set for training
                if id not in self.training_queue_set:
                    self.training_queue.append(id)
                    self.training_queue_set.add(id)
        
        # Place next item in training set and begin training
        if len(self.training_set) < self.async_processes_allowed:
            id = self.training_queue.pop(0)
            self.project_to_time_of_threshold_reached[id] = datetime.now()
            trainer = Trainer(id, self.ls, self.ls_client)
            self.training_set.add(trainer)
            self.__listen_for_more_annotations_and_train(id=id, trainer=trainer)