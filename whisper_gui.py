import whisper
import torch
import torch_directml
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter.ttk import Progressbar, Label, Button, OptionMenu, Checkbutton, Frame, Style
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tkinter import font
import threading
import queue
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class ModernWhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.geometry("500x750")
        
        # Initialize dark mode and color schemes first
        # Initialize dark mode and color schemes first
        # Initialize dark mode and color schemes first
        self.is_dark_mode = tk.BooleanVar(value=False)
        
        # Define color schemes
        self.light_colors = {
            'bg': '#f0f2f5',
            'primary': '#2962ff',
            'secondary': '#f5f5f5',
            'text': '#1a1a1a',
            'success': '#43a047',
            'input_bg': 'white'
        }
        
        self.dark_colors = {
            'bg': '#1e1e1e',
            'primary': '#0d47a1',
            'secondary': '#2d2d2d',
            'text': '#ffffff',
            'success': '#2e7d32',
            'input_bg': '#3d3d3d'
        }
        
        # Set initial colors
        self.colors = self.light_colors
        
        # Initialize other variables
        self.gui_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_file = None
        self.converted_audio_file = None
        self.model_var = tk.StringVar(value="medium")
        self.language_var = tk.StringVar(value="en")
        self.gpu_var = tk.BooleanVar()

        # Now setup styles and create widgets
        self.setup_styles()
        self.create_widgets()
        self.apply_theme()
        
        # Setup window events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(50, self.process_gui_queue)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles with current theme colors
        self.style.configure(
            'Modern.TFrame',
            background=self.colors['bg']
        )
        
        # Configure normal button state
        self.style.configure(
            'Modern.TButton',
            background=self.colors['primary'],
            foreground='white',
            padding=10,
            font=('Segoe UI', 10)
        )
        
        # Configure button hover state
        self.style.map('Modern.TButton',
            foreground=[('active', 'black')],
            background=[('active', self.colors['secondary'])]
        )

        self.style.configure(
            'Modern.Horizontal.TProgressbar',
            troughcolor=self.colors['secondary'],
            background=self.colors['primary'],
            thickness=10
        )

        self.style.configure(
            'Modern.TLabel',
            background=self.colors['bg'],
            foreground=self.colors['text'],
            font=('Segoe UI', 10)
        )
        
    def toggle_dark_mode(self):
        """Toggle between light and dark mode."""
        self.is_dark_mode.set(not self.is_dark_mode.get())
        self.colors = self.dark_colors if self.is_dark_mode.get() else self.light_colors
        self.apply_theme()
        
    def apply_theme(self):
        """Apply the current theme colors to all widgets."""
        # Update root background
        self.root.configure(bg=self.colors['bg'])
        
        # Update styles
        self.style.configure('Modern.TFrame', background=self.colors['bg'])
        self.style.configure('Modern.TButton',
            background=self.colors['primary'],
            foreground='white'
        )
        self.style.configure('Modern.TLabel',
            background=self.colors['bg'],
            foreground=self.colors['text']
        )
        self.style.configure('Modern.TCheckbutton',
            background=self.colors['bg'],
            foreground=self.colors['text']
        )
        
        # Update text area
        self.transcription_text.configure(
            bg=self.colors['input_bg'],
            fg=self.colors['text']
        )
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding=20)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Whisper Audio Transcriber",
            style='Modern.TLabel',
            font=('Segoe UI', 24, 'bold')
        )
        # darkmode
        ttk.Button(
        main_frame,
        text="Toggle Dark Mode",
        command=self.toggle_dark_mode,
        style='Modern.TButton'
        ).grid(row=10, column=0, columnspan=2, pady=(10,0), sticky="ew")
        
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))

        # File selection section
        self.selected_file_label = ttk.Label(
            main_frame,
            text="No file selected",
            style='Modern.TLabel'
        )
        self.selected_file_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        # Controls section
        controls_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        controls_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))

        # Model selection
        ttk.Label(
            controls_frame,
            text="Model:",
            style='Modern.TLabel'
        ).grid(row=0, column=0, padx=(0, 10))

        model_menu = ttk.OptionMenu(
            controls_frame,
            self.model_var,
            "medium",
            "tiny", "base", "small", "medium", "large"
        )
        model_menu.grid(row=0, column=1, padx=10)

        # Language selection
        ttk.Label(
            controls_frame,
            text="Language:",
            style='Modern.TLabel'
        ).grid(row=0, column=2, padx=10)

        languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "tl"]
        language_menu = ttk.OptionMenu(
            controls_frame,
            self.language_var,
            "en",
            *languages
        )
        language_menu.grid(row=0, column=3, padx=10)

        # GPU checkbox
        gpu_check = ttk.Checkbutton(
            controls_frame,
            text="Use GPU",
            variable=self.gpu_var,
            style='Modern.TCheckbutton'
        )
        gpu_check.grid(row=0, column=4, padx=10)

        # Action buttons
        button_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)

        ttk.Button(
            button_frame,
            text="Select Audio File",
            command=self.select_audio_file,
            style='Modern.TButton'
        ).grid(row=0, column=0, padx=10)

        ttk.Button(
            button_frame,
            text="Start Transcribing",
            command=self.start_transcribing,
            style='Modern.TButton'
        ).grid(row=0, column=1, padx=10)

        # Progress section
        self.progress_bar = ttk.Progressbar(
            main_frame,
            style="Modern.Horizontal.TProgressbar",
            mode='determinate',
            length=400
        )
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=20, sticky="ew")

        self.progress_label = ttk.Label(
            main_frame,
            text="",
            style='Modern.TLabel'
        )
        self.progress_label.grid(row=5, column=0, columnspan=2)

        self.green_check_label = ttk.Label(
            main_frame,
            text="",
            style='Modern.TLabel'
        )
        self.green_check_label.grid(row=6, column=0, columnspan=2)

        # Transcription text area
        self.transcription_text = tk.Text(
            main_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            font=('Segoe UI', 11),
            bg='white',
            relief="flat"
        )
        self.transcription_text.grid(row=7, column=0, columnspan=2, pady=10, sticky="nsew")

        # Bottom buttons
        ttk.Button(
            main_frame,
            text="Save Transcription",
            command=self.save_transcription,
            style='Modern.TButton'
        ).grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="ew")

        ttk.Button(
            main_frame,
            text="Clear",
            command=self.refresh_transcription,
            style='Modern.TButton'
        ).grid(row=9, column=0, columnspan=2, sticky="ew")

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

    def select_audio_file(self):
        """
        Open a file dialog to select an audio file.
        """
        self.audio_file = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg *.aac *.m4a")])
        if self.audio_file:
            file_name = os.path.basename(self.audio_file)
            self.selected_file_label.config(text=f"Selected File: {file_name}")
            self.green_check_label.config(text="")
            supported_formats = (".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a")
            if not self.audio_file.lower().endswith(supported_formats):
                messagebox.showerror("Error", "Unsupported audio file format. Please select a supported file.")
                self.audio_file = None
                self.selected_file_label.config(text="No file selected")
                return
            if not self.audio_file.lower().endswith(".wav"):
                self.converted_audio_file = os.path.splitext(self.audio_file)[0] + ".wav"
                audio = AudioSegment.from_file(self.audio_file)
                audio.export(self.converted_audio_file, format="wav")
            else:
                self.converted_audio_file = self.audio_file
    def start_transcribing(self):
        if self.audio_file:
            self.toggle_loading_animation(start=True)  # Start animation
            self.executor.submit(self.convert_and_transcribe, self.converted_audio_file)
        else:
            messagebox.showwarning("Warning", "Please select an audio file first.")
    def toggle_loading_animation(self, start=True):
        """
        Start or stop the progress bar animation with real progress updates.
        """
        if start:
            self.progress_bar["mode"] = "determinate"  # Fill gradually
            self.progress_bar["value"] = 0  # Reset progress to 0%
        else:
            self.progress_bar["value"] = 100  # Ensure it reaches 100%
   
    def update_progress(self, current: int, total: int) -> None:
        if not isinstance(current, int) or not isinstance(total, int):
            raise TypeError("Current and total must be integers")
        if total <= 0:
            raise ValueError("Total must be positive")
        if current < 0 or current > total:
            raise ValueError("Current must be between 0 and total")
        
        percentage = int((current / total) * 100)
        
    def convert_and_transcribe(self, audio_file):
        """
        Convert the audio file to WAV format and transcribe it with real-time progress updates.
        """
        try:
            selected_model = self.model_var.get()
            device = "cpu"
            if self.gpu_var.get():
                try:
                    available_devices = torch_directml.device_count()
                    if available_devices > 0:
                        device = torch_directml.device(0)
                        logging.info(f"Trying DirectML GPU: {device}")
                except Exception as e:
                    logging.error(f"Error detecting GPU: {e}")
                    device = "cpu"
            logging.info(f"Loading Whisper model '{selected_model}' on {device}...")
            model = whisper.load_model(selected_model, device="cpu")
            if device != "cpu":
                try:
                    model.to(device)
                except Exception as gpu_error:
                    logging.error(f"DirectML error, falling back to CPU: {gpu_error}")
                    device = "cpu"
            selected_language = self.language_var.get()
            load_dotenv(dotenv_path="token.env")
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
            # Start transcription
            result = model.transcribe(audio_file, language=selected_language)
            if "segments" not in result:
                raise ValueError("Transcription result does not contain 'segments'")
            diarization = diarization_pipeline(audio_file)
            combined_result = self.combine_diarization_and_transcription(diarization, result)
            total_segments = len(combined_result["segments"])  # Get total for progress tracking
            for i, segment in enumerate(combined_result["segments"], start=1):
                # Send transcription to GUI queue
                self.gui_queue.put(lambda segment=segment, i=i: self.display_transcription(segment, i, total_segments))
                # **Update progress dynamically**
                progress_percentage = int((i / total_segments) * 100)
                self.gui_queue.put(lambda progress_percentage=progress_percentage: self.update_progress(progress_percentage, 100))
                # **Force GUI updates in real-time**
                self.root.update_idletasks()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            self.gui_queue.put(lambda e=e: messagebox.showerror("Error", str(e)))
        finally:
            # Ensure the progress bar reaches 100% at the end
            self.gui_queue.put(lambda: self.toggle_loading_animation(start=False))
            self.gui_queue.put(lambda: self.update_progress(100, 100))
    def display_transcription(self, segment, current, total):
        """
        Display the transcription segment in the text area with improved formatting.
        """
        try:
            # Validate segment data
            if not all(key in segment for key in ["text", "speaker"]):
                raise ValueError("Invalid segment data")

            # Add header separator on first segment
            if current == 1:
                self.transcription_text.delete(1.0, tk.END)  # Clear existing text
                self.transcription_text.insert(tk.END, "_"*80 + "\n")

            # Format and insert the segment text
            text = segment["text"].strip()
            speaker = segment["speaker"]
            formatted_text = f"[{speaker}]: {text}\n"
            self.transcription_text.insert(tk.END, formatted_text)
            
            # Autoscroll and update display
            self.transcription_text.see(tk.END)
            self.transcription_text.update_idletasks()
            
            # Update progress
            self.update_progress(current, total)
            
            # Show completion message
            if current == total:
                self.transcription_text.insert(tk.END, "_"*80 + "\n")
                self.green_check_label.config(text="✔️ Transcription Complete")
                messagebox.showinfo("Transcription Complete", 
                                "The transcription process has completed successfully.")
                
        except Exception as e:
            logging.error(f"Error displaying transcription: {e}")
            self.green_check_label.config(text="⚠️ Error occurred")
            messagebox.showerror("Error", f"Failed to display transcription: {str(e)}")
            
    def combine_diarization_and_transcription(self, diarization, transcription):
        """
        Combine the diarization and transcription results.
        """
        combined_segments = []
        current_segment = None
        for segment in transcription["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            speaker = "Unknown"
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= start and turn.end >= end:
                    speaker = speaker_label
                    break
            if current_segment and current_segment["speaker"] == speaker:
                current_segment["end"] = end
                current_segment["text"] += "" +text
            else:
                if current_segment:
                    combined_segments.append(current_segment)
                current_segment = {"start": start, "end": end, "text": text, "speaker": speaker}
        if current_segment:
            combined_segments.append(current_segment)
        return {"segments": combined_segments}
    def save_transcription(self):
        """
        Save the transcription to a text file.
        """
        text = self.transcription_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No transcription available to save.")
            return
        try:
            save_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if save_file:
                with open(save_file, "w") as file:
                    file.write(text)
                messagebox.showinfo("Success", f"Transcription saved to {save_file}")
        except Exception as e:
            logging.error(f"Error saving transcription: {e}")
            messagebox.showerror("Error", str(e))
    def refresh_transcription(self):
        """
        Refresh the transcription text area and reset the GUI.
        """
        self.transcription_text.delete(1.0, tk.END)
        self.selected_file_label.config(text="No file selected")
        self.green_check_label.config(text="")
        self.progress_label.config(text="")
        self.audio_file = None
        self.converted_audio_file = None
    def on_closing(self):
        """
        Handle the application closing event.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.executor.shutdown(wait=False)
            self.root.destroy()
    def process_gui_queue(self):
        """
        Process tasks in the GUI queue.
        """
        while not self.gui_queue.empty():
            task = self.gui_queue.get()
            task()
        self.root.after(50, self.process_gui_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernWhisperApp(root)
    root.mainloop()