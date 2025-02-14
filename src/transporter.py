import os
import glob
import socket
import torch
import shutil
from pathlib import Path

API_KEY = os.getenv("API_KEY")
USB_KEY_FILENAME = os.getenv("USB_KEY_FILENAME")
FILE_SERVER_IP = os.getenv("FILE_SERVER_IP")

class ModelTransporter:
    def __init__(self, save_folder):
        self.save_folder = save_folder

    def __scan_for_available_usb_device(self):
        usb_mount_points = glob.glob("/media/*/*")
        for mount in usb_mount_points:
            key_file_path = os.path.join(mount, USB_KEY_FILENAME)
            if os.path.exists(key_file_path):
                with open(key_file_path, "r") as f:
                    key_content = f.read().strip()
                if key_content == API_KEY:
                    return mount  # Return the first valid USB mount point
        return None
        
    def __is_file_server_available(self, ip):
        try:
            socket.create_connection((ip, 22), timeout=2)  # Test SSH port
            return True
        except (socket.timeout, socket.error):
            return False
    
    def save_model(self, model, weights_name)->str:
        model_path = os.path.join(self.save_folder, "weights", weights_name)

        # Check for USB device
        usb_drive = self.__scan_for_available_usb_device()
        if usb_drive:
            usb_path = os.path.join(usb_drive, model_path)
            torch.save(model.state_dict(), usb_path)
            return f"✅ Model saved to USB: {usb_path}"
            

        # Check if file server is available
        if self.__is_file_server_available(FILE_SERVER_IP):
            print("File server is available but skipping for now")
            
            # remote_path = os.path.join(REMOTE_SAVE_PATH, MODEL_FILENAME)
            # torch.save(model.state_dict(), remote_path)
            # print(f"✅ Model saved to file server: {remote_path}")
            # return

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
        usb_drive = self.__scan_for_available_usb_device()
        if usb_drive:
            usb_path = os.path.join(usb_drive, save_path)
            shutil.move(str(metrics_path), str(usb_path))
            return f"✅ Metrics saved to USB: {usb_path}"
            

        # Check if file server is available
        if self.__is_file_server_available(FILE_SERVER_IP):
            print("File server is available but skipping for now")

        # Save locally if all else fails
        save_path = os.path.join("./local-saves/", save_path)
        os.makedirs(save_path, exist_ok=True)
        shutil.move(str(metrics_path), str(save_path))
        return f"✅ Metrics saved locally: {save_path}"
    
    def full_save(self, model, weights_name, metrics_path)->str:
        model_save = self.save_model(model, weights_name)
        metrics_save = self.save_metrics_directory(metrics_path)
        return model_save + "\n" + metrics_save