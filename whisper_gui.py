import whisper
import torch
import torch_directml
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Progressbar
import threading
import queue

# Create a queue to communicate with the main thread
gui_queue = queue.Queue()

def select_audio_file():
    audio_file = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg *.aac *.m4a")])
    if audio_file:
        start_loading_animation()
        green_check_label.config(text="")  # Clear the green check mark
        threading.Thread(target=convert_and_transcribe, args=(audio_file,)).start()

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

# Create Tkinter root window
root = tk.Tk()
root.title("Whisper Transcriber")
root.geometry("600x700")

# Create and pack widgets for model selection
tk.Label(root, text="Select Whisper Model:").pack(pady=5)
model_var = tk.StringVar(value="medium")
model_menu = tk.OptionMenu(root, model_var, "tiny", "base", "small", "medium", "large", "large-v3", "turbo")
model_menu.pack(pady=5)

# Create and pack widgets for language selection
tk.Label(root, text="Select Transcription Language:").pack(pady=5)
language_var = tk.StringVar(value="en")
languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "tl"]  # Added "tl" for Tagalog
language_menu = tk.OptionMenu(root, language_var, *languages)
language_menu.pack(pady=5)

# Checkbox for GPU option
gpu_var = tk.BooleanVar()
tk.Checkbutton(root, text="Use GPU (if available)", variable=gpu_var).pack(pady=5)

tk.Button(root, text="Select Audio File", command=select_audio_file).pack(pady=10)

# Loading animation (progress bar)
progress_bar = Progressbar(root, mode='indeterminate')
progress_bar.pack(pady=5)

# Progress percentage label
progress_label = tk.Label(root, text="")
progress_label.pack(pady=5)

# Green check mark label
green_check_label = tk.Label(root, text="", foreground="green")
green_check_label.pack(pady=5)

transcription_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15)
transcription_text.pack(pady=10)
tk.Button(root, text="Save Transcription", command=save_transcription).pack(pady=10)

# Bind the on_closing function to the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start processing the GUI queue
root.after(100, process_gui_queue)

# Start the Tkinter event loop
root.mainloop()
