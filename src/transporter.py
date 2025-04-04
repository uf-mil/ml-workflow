import os
import glob
import smbclient.path
import torch
import shutil
import smbclient
import socket
import pandas as pd
from pathlib import Path

from typing import Tuple, Any
from memoryHandler import MemoryHandler
from service import Service

class ModelTransporter:
    def __init__(self, save_folder, service:Service):
        self.save_folder = save_folder
        self.service = service

    def scan_for_available_usb_device(self):
        usb_mount_points = glob.glob("/media/*/*")
        for mount in usb_mount_points:
            key_file_path = os.path.join(mount, self.service.usb_key_file_name)
            if os.path.exists(key_file_path):
                with open(key_file_path, "r") as f:
                    key_content = f.read().strip()
                if key_content == self.service.label_studio_api_key:
                    return mount  # Return the first valid USB mount point
        return None
        
    def is_file_server_available(self):
        try:
            socket.setdefaulttimeout(3)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.service.file_server_ip, self.service.file_server_port))
            return True
        except Exception as e:
            return False
    
    def save_model(self, model, weights_name)->Tuple[str, Any]:
        model_path = os.path.join(self.save_folder, "weights", weights_name)
        model_dir = os.path.join(self.save_folder, "weights")

        # Check for USB device
        usb_drive = self.scan_for_available_usb_device()
        if usb_drive:
            usb_path = os.path.join(usb_drive,"ml-workflow", model_dir)
            os.makedirs(usb_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(usb_path, weights_name))
            return f"✅ Model saved to USB: {usb_path}", usb_path
            

        # Check if file server is available
        try:
            if self.is_file_server_available():
                print("[INFO]: File server is available!")
                smbclient.register_session(self.service.file_server_ip, username=self.service.file_server_username, password=self.service.file_server_password)
                print("[SUCCESS]: File server is accessible!")

                remote_path = f"//{self.service.file_server_ip}/"+ os.path.join(self.service.file_server_shared_folder, "ml-workflow", model_dir)
                
                smbclient.makedirs(remote_path, exist_ok=True)

                remote_file_path = os.path.join(remote_path, weights_name)

                print(remote_file_path)
                with smbclient.open_file(remote_file_path, mode="wb") as remote_f:
                    torch.save(model.state_dict(), remote_f)
                
                return f"✅ Model saved to file server: {remote_path}", remote_path
        except Exception as e:
            print(f"[INFO]: Could not establish connection to file server because:\n{e}")

        # Save locally if all else fails
        model_dir = os.path.join("./local-saves",self.save_folder,"weights")
        model_path = "./local-saves/"+model_path
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return f"✅ Model saved locally: {model_path}", model_path
    
    def save_metrics_directory(self, metrics_path, project_id)->Tuple[str, Any]:
        metrics_path = Path(metrics_path)
        save_path = os.path.join(self.save_folder, "metrics")

        if not metrics_path.exists():
            return f"❌ Error: Training directory {metrics_path} does not exist.", None
            
        # Save to results.csv to service memory
        MemoryHandler().commit_results_to_memory(project_id, metrics_path)
        
        # Check for USB device
        usb_drive = self.scan_for_available_usb_device()
        if usb_drive:
            usb_path = os.path.join(usb_drive,"ml-workflow", save_path)
            os.makedirs(usb_path, exist_ok=True)
            allfiles = os.listdir(str(metrics_path))
            for f in allfiles:
                if os.path.isdir(os.path.join(str(metrics_path),f)):
                    continue
                shutil.move(os.path.join(str(metrics_path),f), os.path.join(str(usb_path),f))
            return f"✅ Metrics saved to USB: {usb_path}", usb_path
            

        # Check if file server is available
    
        try:
            if self.__is_file_server_available(self.service.file_server_ip):
                print("[INFO]: File server is available!")
                smbclient.register_session(self.service.file_server_ip, username=self.service.file_server_username, password=self.service.file_server_password)
                print("[SUCCESS]: File server is accessible!")

                remote_path = f"//{self.service.file_server_ip}/"+ os.path.join(self.service.file_server_shared_folder, "ml-workflow", save_path)
                
                smbclient.makedirs(remote_path, exist_ok=True)
                
                allfiles = os.listdir(str(metrics_path))
                for file in allfiles:
                    if os.path.isdir(os.path.join(str(metrics_path),file)):
                        continue
                    f = os.path.join(str(metrics_path),file)
                    with open(f, "rb") as src:
                        remote_file_path = os.path.join(remote_path, file)
                        with smbclient.open_file(remote_file_path, mode="wb") as dest:
                            shutil.copyfileobj(src, dest)
                                
                return f"✅ Metrics saved to file server: {remote_path}", remote_path
        except Exception as e:
            print(f"[INFO]: Could not establish connection to file server because:\n{e}")

        # Save locally if all else fails
        save_path = os.path.join("./local-saves/", save_path)
        os.makedirs(save_path, exist_ok=True)
        allfiles = os.listdir(str(metrics_path))
        for f in allfiles:
            if os.path.isdir(os.path.join(str(metrics_path),f)):
                    continue
            shutil.move(os.path.join(str(metrics_path),f), os.path.join(str(save_path),f))

        return f"✅ Metrics saved locally: {save_path}", save_path
    
    def full_save(self, model, weights_name, metrics_path, project_id)->Tuple[str, dict]:
        model_save_msg, model_path = self.save_model(model, weights_name)
        metrics_save_msg, data_path = self.save_metrics_directory(metrics_path, project_id)
        return model_save_msg + "\n" + metrics_save_msg, {'model': model_path, 'metrics': data_path}