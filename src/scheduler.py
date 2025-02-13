import os
import csv
import time

from dotenv import load_dotenv

from typing import List

from label_studio_sdk.client import LabelStudio

class Scheduler:
    def __init__(self, async_processes_allowed:int=1, listen_every:float = 500, batch_size:int = 32):
        """
        Parameters:
            listen_every (millisecond): sleep period after every call to LabelStudio to get most recent training data.

            batch_size: number of finished tasks that need to be completed to trigger a new training cycle.

        The Scheduler handles calling the Trainer after a certain number of labeling tasks are completed per project. 
        """

        load_dotenv()
        self.__LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        self.__API_KEY = os.getenv("API_KEY")

        self.batch_size = batch_size
        self.async_processes_allowed = async_processes_allowed
        self.listen_every = listen_every
        
        self.project_finished_tasks_dict = {}
        self.training_set = set()
        self.training_queue_set = set()
        self.training_queue = []

        # Create dict of projects and latest completed batch size
        # Check if memory.csv exists and is populated
        if os.path.exists("memory.csv"):
            with open("memory.csv", 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.project_finished_tasks_dict[reader["id"]] = reader["finished_tasks"]   
        else: # Load finished_task data from LabelStudio
            with open("memory.csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id","finished_tasks"])

                ls = LabelStudio(base_url=self.__LABEL_STUDIO_URL, api_key=self.__API_KEY)

                projects = ls.projects
                for p in projects.list():
                    writer.writerow([p.id,p.finished_task_number])
                    self.project_finished_tasks_dict[p.id] = p.finished_task_number

    def __listen(self)->List[int]:
        """
        Run a while loop that breaks when it detects that the difference between completed tasks and the new number of finished tasks is greater than the batch size.

        Returns list of project ids.
        """
        project_ids:List[int] = []

        while len(project_ids) == 0:
            time.sleep(self.listen_every/1000)
            ls = LabelStudio(base_url=self.__LABEL_STUDIO_URL, api_key=self.__API_KEY)
            projects = ls.projects
            for p in projects.list():
                if p.id in self.project_finished_tasks_dict:
                    task_count = self.project_finished_tasks_dict[p.id]
                    if p.finished_task_number - task_count >= self.batch_size:
                        project_ids.append(p.id)
                        self.project_finished_tasks_dict[p.id] = p.finished_task_number
                else:
                    self.project_finished_tasks_dict[p.id] = p.finished_task_number

        return project_ids

    def run(self):
        while True:
            print("[INFO]: Listening for new labeling tasks to be finished...")
            project_ids = self.__listen()

            # Check if there are locks available to train
            if self.training_set <= self.async_processes_allowed:

                next_sessions = self.training_queue[0:len(self.async_processes_allowed)-len(self.training_set)] if len(self.training_queue) > len(self.async_processes_allowed)-len(self.training_set) else self.training_queue.copy()

                for session in next_sessions:
                    # TODO: Create Trainer objects and train 
                    pass

                
            # TODO: pass the project ids to the trainer to retrieve 
