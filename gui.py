import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import scrolledtext
import os
import subprocess
import threading
import sys

# Force UTF-8 encoding for print statements
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
my_env = os.environ.copy()
my_env["PYTHONIOENCODING"] = "utf-8"

# Modify this to pass the output_text from the GUI
# Function to handle file drop
def drop(event, output_text):
    file_path = event.data.strip()  # Strip any extra spaces or newlines
    file_path = file_path.strip('{}')  # Strip curly braces
    if os.path.isfile(file_path) and file_path.lower().endswith('.mp4'):
        process_file(file_path, output_text)
    else:
        output_text.insert(tk.END, "Please drop a valid .mp4 video file.\n")

# Modify this function to pass the output_text to vodlabeler.py
def process_file(file_path, output_text):
    def run_vodlabeler():
        try:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            vodlabeler_path = os.path.join(current_dir, 'vodlabeler.py')

            output_text.delete(1.0, tk.END)  # Clear previous output

            output_text.insert(tk.END, f"Processing video: {file_path}\n")
            output_text.see(tk.END)

            # Use sys.executable to ensure the same Python interpreter is used
            process = subprocess.Popen(
                [sys.executable, '-u', vodlabeler_path, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=my_env  # This ensures the environment uses UTF-8
            )

            # Stream the output from vodlabeler.py to the GUI
            for stdout_line in iter(process.stdout.readline, ''):
                output_text.insert(tk.END, stdout_line)
                output_text.see(tk.END)  # Auto scroll to latest output

            process.stdout.close()
            process.wait()  # Ensure process finishes

            stderr_output = process.stderr.read()
            if stderr_output:
                output_text.insert(tk.END, f"Errors:\n{stderr_output}\n")
                output_text.see(tk.END)
            process.stderr.close()

        except Exception as e:
            output_text.insert(tk.END, f"An error occurred: {e}\n")
            output_text.see(tk.END)

    # Use threading to keep the GUI responsive
    thread = threading.Thread(target=run_vodlabeler)
    thread.start()


# Function to handle file drop
def drop(event, output_text):
    file_path = event.data.strip()  # Strip any extra spaces or newlines
    file_path = file_path.strip('{}')  # Strip curly braces
    if os.path.isfile(file_path) and file_path.lower().endswith('.mp4'):
        process_file(file_path, output_text)
    else:
        output_text.insert(tk.END, "Please drop a valid .mp4 video file.\n")


# Create GUI window
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("VOD Labeler")
        self.geometry("600x400")

        # Add a label for instructions
        label = tk.Label(self, text="Drag and drop your .mp4 file here", font=("Arial", 14))
        label.pack(pady=20)

        # ScrolledText widget for displaying logs
        self.output_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=10, font=("Arial", 12))
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Enable drag and drop functionality
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', lambda event: drop(event, self.output_text))

# Run the application
if __name__ == "__main__":
    app = App()
    app.mainloop()