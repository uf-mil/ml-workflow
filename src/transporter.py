import os
import glob
import smbclient.path
import torch
import shutil
import smbclient
import socket
from pathlib import Path

API_KEY = os.getenv("API_KEY")
USB_KEY_FILENAME = os.getenv("USB_KEY_FILENAME")

# SMB set up
FILE_SERVER_IP = os.getenv("FILE_SERVER_IP")
SHARED_FOLDER = os.getenv("SHARED_FOLDER")    
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")


class ModelTransporter:
    def __init__(self, save_folder):
        self.save_folder = save_folder

    def scan_for_available_usb_device(self):
        usb_mount_points = glob.glob("/media/*/*")
        for mount in usb_mount_points:
            key_file_path = os.path.join(mount, USB_KEY_FILENAME)
            if os.path.exists(key_file_path):
                with open(key_file_path, "r") as f:
                    key_content = f.read().strip()
                if key_content == API_KEY:
                    return mount  # Return the first valid USB mount point
        return None
        
    def is_file_server_available(self):
        try:
            socket.setdefaulttimeout(3)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((FILE_SERVER_IP, 2222))
            return True
        except Exception as e:
            return False
    
    def save_model(self, model, weights_name)->str:
        model_path = os.path.join(self.save_folder, "weights", weights_name)
        model_dir = os.path.join(self.save_folder, "weights")

        # Check for USB device
        usb_drive = self.scan_for_available_usb_device()
        if usb_drive:
            usb_path = os.path.join(usb_drive,"ml-workflow", model_dir)
            os.makedirs(usb_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(usb_path, weights_name))
            return f"✅ Model saved to USB: {usb_path}"
            

        # Check if file server is available
        try:
            if self.is_file_server_available():
                print("[INFO]: File server is available!")
                smbclient.register_session(FILE_SERVER_IP, username=USERNAME, password=PASSWORD)
                print("[SUCCESS]: File server is accessible!")

                remote_path = f"//{FILE_SERVER_IP}/"+ os.path.join(SHARED_FOLDER, "ml-workflow", model_dir)
                
                smbclient.makedirs(remote_path, exist_ok=True)

                remote_file_path = os.path.join(remote_path, weights_name)

                print(remote_file_path)
                with smbclient.open_file(remote_file_path, mode="wb") as remote_f:
                    torch.save(model.state_dict(), remote_f)
                
                return f"✅ Model saved to file server: {remote_path}"
        except Exception as e:
            print(f"[INFO]: Could not establish connection to file server because:\n{e}")

        # Save locally if all else fails
        model_dir = os.path.join("./local-saves",self.save_folder,"weights")
        model_path = "./local-saves/"+model_path
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return f"✅ Model saved locally: {model_path}"
    
    def save_metrics_directory(self, metrics_path)->str:
        metrics_path = Path(metrics_path)
        save_path = os.path.join(self.save_folder, "metrics")

        if not metrics_path.exists():
            return f"❌ Error: Training directory {metrics_path} does not exist."
            
        
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
            return f"✅ Metrics saved to USB: {usb_path}"
            

        # Check if file server is available
    
        try:
            if self.__is_file_server_available(FILE_SERVER_IP):
                print("[INFO]: File server is available!")
                smbclient.register_session(FILE_SERVER_IP, username=USERNAME, password=PASSWORD)
                print("[SUCCESS]: File server is accessible!")

                remote_path = f"//{FILE_SERVER_IP}/"+ os.path.join(SHARED_FOLDER, "ml-workflow", save_path)
                
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
                                
                return f"✅ Metrics saved to file server: {remote_path}"
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

        return f"✅ Metrics saved locally: {save_path}"
    
    def full_save(self, model, weights_name, metrics_path)->str:
        model_save = self.save_model(model, weights_name)
        metrics_save = self.save_metrics_directory(metrics_path)
        return model_save + "\n" + metrics_save