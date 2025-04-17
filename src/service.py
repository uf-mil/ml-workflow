import os
from dotenv import load_dotenv

class Service:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Service, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        load_dotenv()
        
        # Configure LabelStudio
        self.label_studio_url = os.getenv('LABEL_STUDIO_URL', 'None Found')
        self.label_studio_api_key = os.getenv('API_KEY', 'None Found')
        
        # Configure file server
        self.file_server_ip = os.getenv('FILE_SERVER_IP', '0.0.0.0')
        self.file_server_port = int(os.getenv('FILE_SERVER_PORT', '22'))
        self.file_server_shared_folder = os.getenv('SHARED_FOLDER', '--')
        self.file_server_username = os.getenv('USERNAME')
        self.file_server_password = os.getenv('PASSWORD')

        # Configure USB device detection
        self.usb_key_file_name = os.getenv('USB_KEY_FILENAME', 'None Found')

        # Configure auto training logic
        self.async_processes_allowed = int(os.getenv('ASYNC_PROCESSES_ALLOWED', 3))
        self.batch_size_threshold = int(os.getenv('BATCH_SIZE_THRESHOLD', 32))
        self.minutes_to_wait_for_next_annotation = float(os.getenv('MINUTES_TO_WAIT_FOR_NEXT_ANNOTATION', 5)) 
        self.minimum_annotations_required = int(os.getenv('MINIMUM_ANNOTATIONS_REQUIRED', 10))
        self.max_epochs_allowed = int(os.getenv('MAX_EPOCHS_ALLOWED', 10))
        self.patience = int(os.getenv("PATIENCE", 5))
        self.load_train_test_validation_split()

        # Dark mode
        self.dark_mode = os.getenv('DARK_MODE', 'False') == 'True'
    
    def load_train_test_validation_split(self):
        self.train = float(os.getenv('TRAIN', 0.8))
        self.test = float(os.getenv("TEST", 0.1))
        self.val = float(os.getenv("VAL", 0.1))

        if self.train + self.test + self.val != 1.0:
             self.train = 0.8
             self.test = 0.1
             self.val = 0.1


          

