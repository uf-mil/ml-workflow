import os
import traceback
import torch

from datetime import datetime

class Logger:
    def __init__(self):
        pass

    def __write_to_log_file(self, log_file, log_entry):
        log_file_exists = os.path.exists(log_file)
        with open(log_file, "r+" if log_file_exists else "w") as f:
            if log_file_exists:
                old_content = f.read()
                f.seek(0)                                                               
            f.write(log_entry + "\n")
            if log_file_exists:
                f.write(old_content)

    def log_training_error(self, error, trainer):

        def suggest_recovery(error_type):
            suggestions = {
                "CUDAOutOfMemoryError": "Try reducing batch size or image size.",
                "FileNotFoundError": "Check if the dataset path is correct.",
                "KeyError": "Ensure that dataset labels match the model's expected format.",
                "ValueError": "Verify that inputs to the model are in the correct shape and format.",
                "RuntimeError": "Check device compatibility and available GPU resources."
            }
            return suggestions.get(error_type, "Check the stack trace for more details.")

        log_file = os.path.join(os.getcwd(), "logs",f"{datetime.now().strftime('%Y-%m-%d')}.txt")
    
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        log_entry = f"""
=======================================
🚨 Training Error - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=======================================
🔹 **Project Name**: {trainer.project.title}
🔹 **Project ID**: {trainer.project_id}
🔹 **Total Data Points**: {trainer.data_count_map['total']}
🔹 **Training Samples**: {trainer.data_count_map['train']}
🔹 **Validation Samples**: {trainer.data_count_map['val']}
🔹 **Test Samples**: {trainer.data_count_map['test']}
🔹 **Number of Classes**: {len(trainer.labels)}
🔹 **Classes**: {trainer.labels}

📌 **Training Configuration**
- **Model**: {trainer.model.model_name}
- **Epochs Attempted**: {500}
- **Batch Size**: {-1}
- **Image Size**: {640}
- **Device**: {"cuda" if torch.cuda.is_available() else "cpu"}

❌ **Error Details**
- **Error Type**: {error_type}
- **Error Message**: {error_message}
- **Stack Trace**:
{stack_trace}

🔄 **Recovery Actions Taken**
- {suggest_recovery(error_type)}

---------------------------------------------------
    """

        # Write to log file
        self.__write_to_log_file(log_file, log_entry)
        
        trainer.return_dict["latest_report"] = log_entry
        print(f"Logged error to {log_file}")
    
    def log_training_success(self,results, trainer, footer=""):
        log_file = os.path.join(os.getcwd(),"logs",f"{datetime.now().strftime('%Y-%m-%d')}.txt")
    
        # Extract YOLO training results
        precision = results.results_dict.get("metrics/precision(B)", "N/A")
        recall = results.results_dict.get("metrics/recall(B)", "N/A")
        map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
        map50_95 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
        
        # Class-wise accuracy
        trainer.return_dict["class_acc_string"] = ",".join(
            [f"{class_name}:{results.maps[i]}" 
            for i, class_name in enumerate(trainer.labels)]
        )
        class_wise_metrics = "\n".join(
            [f"- {class_name}: {results.maps[i]}" 
            for i, class_name in enumerate(trainer.labels)]
        )

        # Create log entry
        log_entry = f"""
=======================================
Training Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=======================================
🔹 **Project Name**: {trainer.project.title}
🔹 **Project ID**: {trainer.project_id}
🔹 **Total Data Points**: {trainer.data_count_map['total']}
🔹 **Training Samples**: {trainer.data_count_map['train']}
🔹 **Validation Samples**: {trainer.data_count_map['val']}
🔹 **Test Samples**: {trainer.data_count_map['test']}
🔹 **Number of Classes**: {len(trainer.labels)}
🔹 **Classes**: {trainer.labels}

📌 **Training Configuration**
- **Model**: {trainer.model.model_name}
- **Epochs Attempted**: {trainer.return_dict["epochs"]}
- **Batch Size**: {-1}
- **Image Size**: {640}
- **Device**: {"cuda" if torch.cuda.is_available() else "cpu"}

📊 **Training Metrics**
- **Final Training Precision**: {precision}
- **Final Training Recall**: {recall}
- **Best mAP@50**: {map50}
- **Best mAP@50-95**: {map50_95}

📈 **Class-wise Performance**
{class_wise_metrics}

✅ **Training Completed Successfully**
{footer}
---------------------------------------------------
"""
        
        # Write to log file
        self.__write_to_log_file(log_file, log_entry)
        
        trainer.return_dict["latest_report"] = log_entry
        print(f"Logged training session to {log_file}")
    
    def log_training_cancellation(self, trainer):
        log_file = os.path.join(os.getcwd(), "logs",f"{datetime.now().strftime('%Y-%m-%d')}.txt")
        time_cancelled = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"""
=======================================
 ⚠️ Training Cancelled - {time_cancelled}
=======================================
🔹 **Project Name**: {trainer.project.title}
🔹 **Project ID**: {trainer.project_id}
🔹 **Total Data Points**: {trainer.data_count_map['total']}
🔹 **Training Samples**: {trainer.data_count_map['train']}
🔹 **Validation Samples**: {trainer.data_count_map['val']}
🔹 **Test Samples**: {trainer.data_count_map['test']}
🔹 **Number of Classes**: {len(trainer.labels)}
🔹 **Classes**: {trainer.labels}

📌 **Training Configuration**  
- **Model**: {trainer.model.model_name}
- **Epochs Attempted**: {500}
- **Batch Size**: {-1}
- **Image Size**: {640}
- **Device**: {"cuda" if torch.cuda.is_available() else "cpu"}

⚠️ **Cancellation Details**  
- **Cancelled By**: User Request  
- **Timestamp**: {time_cancelled}  
- **Reason**: Manual interruption
"""
        # Write to log file
        self.__write_to_log_file(log_file, log_entry)
        trainer.return_dict["latest_report"] = log_entry
        print(f"Logged cancellation to {log_file}")