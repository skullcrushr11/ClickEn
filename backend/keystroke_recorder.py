import tkinter as tk
from tkinter import ttk
import json
import time
import uuid
import os
from datetime import datetime

class KeystrokeRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Keystroke Recorder")
        self.root.geometry("600x500")
        
        # Data storage
        self.keyboard_data = []
        self.session_id = str(uuid.uuid4())[:24]
        self.recording = False
        self.is_cheating = tk.BooleanVar()
        
        # Create UI elements
        self.create_widgets()
        
        # Bind events
        self.text_area.bind("<KeyPress>", self.on_key_press)
        self.text_area.bind("<KeyRelease>", self.on_key_release)
    
    def create_widgets(self):
        # Mode selection
        frame_mode = ttk.Frame(self.root, padding="10")
        frame_mode.pack(fill=tk.X)
        
        ttk.Label(frame_mode, text="Recording Mode:").pack(side=tk.LEFT)
        
        ttk.Radiobutton(frame_mode, text="Normal Typing", variable=self.is_cheating, 
                         value=False).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(frame_mode, text="Cheating", variable=self.is_cheating, 
                         value=True).pack(side=tk.LEFT, padx=10)
        
        # Start/Stop buttons
        frame_control = ttk.Frame(self.root, padding="10")
        frame_control.pack(fill=tk.X)
        
        self.btn_start = ttk.Button(frame_control, text="Start Recording", command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(frame_control, text="Stop Recording", command=self.stop_recording)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        self.btn_stop["state"] = "disabled"
        
        self.btn_save = ttk.Button(frame_control, text="Save Data", command=self.save_data)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        # Status display
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(frame_control, textvariable=self.status_var).pack(side=tk.LEFT, padx=20)
        
        # Text area for typing
        frame_text = ttk.Frame(self.root, padding="10")
        frame_text.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame_text, text="Type here:").pack(anchor=tk.W)
        
        self.text_area = tk.Text(frame_text, height=10, width=50, state="disabled")
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Stats display
        self.stats_frame = ttk.LabelFrame(self.root, text="Recording Statistics", padding="10")
        self.stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_var = tk.StringVar()
        self.stats_var.set("No data recorded")
        ttk.Label(self.stats_frame, textvariable=self.stats_var).pack()
    
    def start_recording(self):
        self.recording = True
        self.keyboard_data = []
        self.session_id = str(uuid.uuid4())[:24]
        self.text_area["state"] = "normal"
        self.text_area.delete(1.0, tk.END)
        self.text_area.focus_set()
        
        self.btn_start["state"] = "disabled"
        self.btn_stop["state"] = "normal"
        
        mode = "Cheating" if self.is_cheating.get() else "Normal Typing"
        self.status_var.set(f"Recording... Mode: {mode}")
        self.stats_var.set("Recording in progress...")
    
    def stop_recording(self):
        self.recording = False
        self.text_area["state"] = "disabled"
        
        self.btn_start["state"] = "normal"
        self.btn_stop["state"] = "disabled"
        
        self.status_var.set("Recording stopped")
        
        # Update stats
        if len(self.keyboard_data) > 0:
            stats = f"Keys recorded: {len(self.keyboard_data)}\n"
            stats += f"Session ID: {self.session_id}\n"
            stats += f"Mode: {'Cheating' if self.is_cheating.get() else 'Normal Typing'}"
            self.stats_var.set(stats)
        else:
            self.stats_var.set("No data recorded")
    
    def on_key_press(self, event):
        if not self.recording:
            return
        
        key = event.char if event.char else event.keysym
        timestamp = int(time.time() * 1000)  # Convert to milliseconds
        
        # Record key down event
        self.keyboard_data.append(["KD", key, timestamp])
    
    def on_key_release(self, event):
        if not self.recording:
            return
        
        key = event.char if event.char else event.keysym
        timestamp = int(time.time() * 1000)  # Convert to milliseconds
        
        # Record key up event
        self.keyboard_data.append(["KU", key, timestamp])
    
    def save_data(self):
        if not self.keyboard_data:
            self.status_var.set("No data to save")
            return
        
        # Create data directory if it doesn't exist
        os.makedirs("keystroke_data", exist_ok=True)
        
        # Create filename with mode and timestamp
        mode = "cheating" if self.is_cheating.get() else "normal"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"keystroke_data/{mode}_{timestamp}.json"
        
        # Create data in the format expected by the HMM
        data = {
            self.session_id: {
                "keyboard_data": self.keyboard_data
            }
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        self.status_var.set(f"Data saved to {filename}")

def main():
    root = tk.Tk()
    app = KeystrokeRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()