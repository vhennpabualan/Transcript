import whisper
import torch
import torch_directml
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Progressbar, Label, Button, OptionMenu, Checkbutton, Frame, Style
from tkinter import font
import threading
import queue
import os

# Create a queue to communicate with the main thread
gui_queue = queue.Queue()

# Create Tkinter root window
root = tk.Tk()
root.title("Whisper Transcriber")
root.geometry("620x700")

# Apply a theme
style = Style()
style.theme_use('clam')

# Configure the style for the progress bar
style.configure("green.Horizontal.TProgressbar", troughcolor='white', background='green')

# Create and pack widgets for model selection
frame = Frame(root, padding="10")
frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

bold_font = font.Font(weight="bold")

selected_file_label = Label(frame, text="No file selected", font=bold_font)
selected_file_label.grid(row=0, column=0, columnspan=2, pady=5)

def select_audio_file():
    global audio_file
    audio_file = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg *.aac *.m4a")])
    if audio_file:
        file_name = os.path.basename(audio_file)  # Get only the file name
        selected_file_label.config(text=f"Selected File: {file_name}")  # Update the label with the selected file name
        green_check_label.config(text="")  # Clear the green check mark

def start_transcribing():
    if audio_file:
        start_loading_animation()
        threading.Thread(target=convert_and_transcribe, args=(audio_file,)).start()
    else:
        messagebox.showwarning("Warning", "Please select an audio file first.")

def start_loading_animation():
    progress_bar.start()

def stop_loading_animation():
    progress_bar.stop()
    progress_label.config(text="")

def update_progress(current, total):
    percentage = int((current / total) * 100)
    progress_label.config(text=f"Progress: {percentage}%")
    root.update_idletasks()

def convert_and_transcribe(audio_file):
    try:
        selected_model = model_var.get()

        # Default to CPU
        device = "cpu"

        if gpu_var.get():
            try:
                available_devices = torch_directml.device_count()
                if available_devices > 0:
                    device = torch_directml.device(0)  # Assign DirectML GPU
                    print(f"Trying DirectML GPU: {device}")
            except Exception as e:
                print(f"Error detecting GPU: {e}")
                device = "cpu"

        print(f"Loading Whisper model '{selected_model}' on {device}...")

        # Load model
        model = whisper.load_model(selected_model, device="cpu")  # Always load on CPU first
        if device != "cpu":
            try:
                model.to(device)  # Move to GPU if available
            except Exception as gpu_error:
                print(f"DirectML error, falling back to CPU: {gpu_error}")
                device = "cpu"

        # Get selected language
        selected_language = language_var.get()

        # Transcribe audio file
        result = model.transcribe(audio_file, language=selected_language)

        # Ensure result contains segments
        if "segments" not in result:
            raise ValueError("Transcription result does not contain 'segments'")

        # Update GUI
        gui_queue.put(lambda: display_transcription(result))

    except Exception as e:
        print(f"An error occurred: {e}")
        gui_queue.put(lambda e=e: messagebox.showerror("Error", str(e)))  # ✅ Fixes NameError
    finally:
        gui_queue.put(stop_loading_animation)


def display_transcription(result):
    # Display transcription with timestamps in the text box
    transcription_text.delete(1.0, tk.END)
    total_segments = len(result["segments"])
    for i, segment in enumerate(result["segments"], start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        text = segment["text"]
        transcription_text.insert(tk.END, f"[{start:.2f}s - {end:.2f}s] {text}\n")
        update_progress(i, total_segments)
    
    # Display green check mark
    green_check_label.config(text="✔️ Transcription Complete")
    messagebox.showinfo("Transcription Complete", "The transcription process is complete.")

def save_transcription():
    try:
        text = transcription_text.get(1.0, tk.END)
        save_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if save_file:
            with open(save_file, "w") as file:
                file.write(text)
            messagebox.showinfo("Success", f"Transcription saved to {save_file}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

def process_gui_queue():
    while not gui_queue.empty():
        task = gui_queue.get()
        task()
    root.after(100, process_gui_queue)

Label(frame, text="Select Whisper Model:").grid(row=1, column=0, pady=5, sticky="w")
model_var = tk.StringVar(value="medium")
model_menu = OptionMenu(frame, model_var, "tiny", "base", "small", "medium", "large", "large-v3", "turbo")
model_menu.grid(row=1, column=1, pady=5, sticky="ew")

# Create and pack widgets for language selection
Label(frame, text="Select Transcription Language:").grid(row=2, column=0, pady=5, sticky="w")
language_var = tk.StringVar(value="en")
languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "tl"]  # Added "tl" for Tagalog
language_menu = OptionMenu(frame, language_var, *languages)
language_menu.grid(row=2, column=1, pady=5, sticky="ew")

# Checkbox for GPU option
gpu_var = tk.BooleanVar()
Checkbutton(frame, text="Use GPU (if available)", variable=gpu_var).grid(row=3, column=0, columnspan=2, pady=5, sticky="w")

Button(frame, text="Select Audio File", command=select_audio_file).grid(row=4, column=0, pady=10, sticky="ew")
Button(frame, text="Start Transcribing", command=start_transcribing).grid(row=4, column=1, pady=10, sticky="ew")

# Loading animation (progress bar)
progress_bar = Progressbar(frame, mode='indeterminate', style="green.Horizontal.TProgressbar")
progress_bar.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")

# Progress percentage label
progress_label = Label(frame, text="", foreground="green")
progress_label.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")

# Green check mark label
green_check_label = Label(frame, text="", foreground="green")
green_check_label.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")

transcription_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=70, height=15)
transcription_text.grid(row=8, column=0, columnspan=2, pady=10, sticky="nsew")
Button(frame, text="Save Transcription", command=save_transcription).grid(row=9, column=0, columnspan=2, pady=10, sticky="ew")

# Bind the on_closing function to the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start processing the GUI queue
root.after(100, process_gui_queue)

# Start the Tkinter event loop
root.mainloop()
