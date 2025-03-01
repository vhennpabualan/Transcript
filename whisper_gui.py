import whisper
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter.ttk import Progressbar, Label, Button, OptionMenu, Checkbutton, Frame, Style
from pydub import AudioSegment
from pydub.silence import detect_silence
import queue
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import time
import torch
from multiprocessing import cpu_count

@lru_cache(maxsize=1)
def get_whisper_model(model_name, device="cpu"):
    """Optimized model loading with faster processing settings"""
    try:
        # Enable faster CPU operations
        torch.set_num_threads(cpu_count())
        torch.set_num_interop_threads(cpu_count())
        torch.backends.mkldnn.enabled = True
        
        # Use faster compute type
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision('high')
        
        model = whisper.load_model(model_name, device=device)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class ModernWhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.geometry("500x750") 
        # Initialize dark mode and color schemes first
        self.is_dark_mode = tk.BooleanVar(value=False)
        self.current_task = None  # Track current transcription task
        
        # Define color schemes
        self.light_colors = {
            'bg': '#f0f2f5','primary': '#2962ff','secondary': '#f5f5f5','text': '#1a1a1a','success': '#43a047','input_bg': 'white'
        }
        
        self.dark_colors = {
            'bg': '#1e1e1e', 'primary': '#0d47a1','secondary': '#2d2d2d','text': '#ffffff','success': '#2e7d32','input_bg': '#3d3d3d'
        }
        
        # Set initial colors
        self.colors = self.light_colors
        
        # Initialize other variables
        self.gui_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)  # Reduced to 1 worker for CPU optimizationd
        self.audio_file = None
        self.converted_audio_file = None
        self.model_var = tk.StringVar(value="base")  # Changed default to base for faster processing
        self.language_var = tk.StringVar(value="en")
        self.batch_size_var = tk.IntVar(value=3)  # New variable for batch updates
        self.transcription_running = False
        self.cancel_requested = False

        # Performance settings
        self.chunk_size = 30  # seconds
        self.max_workers = min(cpu_count(), 4)  # Limit max workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Control CPU usage
        self.max_cpu_percent = 80  # Target max CPU percentage
        self.cpu_threads = max(1, cpu_count() - 1)  # Leave one core free
        torch.set_num_threads(self.cpu_threads)

        # Add to __init__ method
        self.processing_settings = {
            'chunk_size': 25,  # Reduced chunk size
            'batch_size': 8,   # Increased batch size
            'cooling_delay': 0.05,  # Reduced cooling delay
            'frame_buffer': 4096,  # Optimized buffer size
        }

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
        self.style.configure('Modern.TFrame', background=self.colors['bg'])
        self.style.configure('Modern.TButton',
            background=self.colors['primary'],
            foreground='white',
            padding=10,
            font=('Segoe UI', 10)
        )
        self.style.map('Modern.TButton',
            foreground=[('active', 'white')],
            background=[('active', self.colors['primary'])]
        )
        self.style.configure('Modern.Horizontal.TProgressbar',
            troughcolor=self.colors['secondary'],
            background=self.colors['primary'],
            thickness=10
        )
        self.style.configure('Modern.TLabel',
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
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # File selection section
        self.selected_file_label = ttk.Label(
            main_frame,
            text="No file selected",
            style='Modern.TLabel'
        )
        self.selected_file_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Controls section
        controls_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        controls_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        # Model selection
        ttk.Label(controls_frame, text="Model:", style='Modern.TLabel').grid(row=0, column=0, padx=(0, 5))
        
        # Simplified model options
        model_menu = ttk.OptionMenu(
            controls_frame,
            self.model_var,
            "base",
            "tiny", "base", "small", "medium", "turbo"  # Removed larger models for CPU efficiency
        )
        model_menu.grid(row=0, column=1, padx=5)

        # Language selection
        ttk.Label(controls_frame, text="Language:", style='Modern.TLabel').grid(row=0, column=2, padx=5)
        
        languages = ["en", "tl", "fr", "de", "es", "it", "ja", "zh", "ar", "ru"]
        language_menu = ttk.OptionMenu(
            controls_frame,
            self.language_var,
            "en",
            *languages
        )
        language_menu.grid(row=0, column=3, padx=5)
        
        # Batch size selection
        ttk.Label(controls_frame, text="Batch size:", style='Modern.TLabel').grid(row=0, column=4, padx=5)
        batch_menu = ttk.OptionMenu(
            controls_frame,
            self.batch_size_var,
            3,
            1, 2, 3, 5, 10  # Batch size options
        )
        batch_menu.grid(row=0, column=5, padx=5)

        # Action buttons
        button_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        self.select_button = ttk.Button(
            button_frame,
            text="Select Audio File",
            command=self.select_audio_file,
            style='Modern.TButton'
        )
        self.select_button.grid(row=0, column=0, padx=5)

        self.transcribe_button = ttk.Button(
            button_frame,
            text="Start Transcribing",
            command=self.start_transcribing,
            style='Modern.TButton'
        )
        self.transcribe_button.grid(row=0, column=1, padx=5)
        
        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_transcription,
            style='Modern.TButton',
            state='disabled'
        )
        self.cancel_button.grid(row=0, column=2, padx=5)

        # Progress section
        self.progress_bar = ttk.Progressbar(
            main_frame,
            style="Modern.Horizontal.TProgressbar",
            mode='determinate',
            length=400
        )
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        self.progress_label = ttk.Label(
            main_frame,
            text="",
            style='Modern.TLabel'
        )
        self.progress_label.grid(row=5, column=0, columnspan=2)

        self.status_label = ttk.Label(
            main_frame,
            text="",
            style='Modern.TLabel'
        )
        self.status_label.grid(row=6, column=0, columnspan=2)

        # Transcription text area
        self.transcription_text = tk.Text(
            main_frame,
            wrap=tk.WORD,
            width=80,
            height=15,  # Increased height
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
        ).grid(row=8, column=0, columnspan=1, pady=(10, 5), sticky="ew")

        ttk.Button(
            main_frame,
            text="Clear",
            command=self.refresh_transcription,
            style='Modern.TButton'
        ).grid(row=8, column=1, columnspan=1, pady=(10, 5), sticky="ew")
        
        # Dark mode toggle button
        ttk.Button(
            main_frame,
            text="Toggle Dark Mode",
            command=self.toggle_dark_mode,
            style='Modern.TButton'
        ).grid(row=9, column=0, columnspan=2, pady=5, sticky="ew")

        self.create_performance_settings(main_frame)

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(7, weight=1)  # Make transcription box expandable
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

    def create_performance_settings(self, main_frame):
        """Add performance settings to UI"""
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Settings", style='Modern.TFrame')
        perf_frame.grid(row=10, column=0, columnspan=2, pady=5, sticky="ew")
        
        # CPU Thread control
        ttk.Label(perf_frame, text="CPU Threads:", style='Modern.TLabel').grid(row=0, column=0, padx=5)
        thread_var = tk.StringVar(value=str(self.cpu_threads))
        thread_menu = ttk.OptionMenu(
            perf_frame,
            thread_var,
            str(self.cpu_threads),
            *[str(i) for i in range(1, cpu_count() + 1)],
            command=lambda x: self._update_cpu_threads(int(x))
        )
        thread_menu.grid(row=0, column=1, padx=5)

    def _update_cpu_threads(self, num_threads):
        """Update the number of CPU threads used for processing."""
        try:
            # Ensure valid thread count
            num_threads = max(1, min(num_threads, cpu_count()))
            self.cpu_threads = num_threads
            
            # Update torch thread settings
            torch.set_num_threads(num_threads)
            
            # Update executor
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            
            logging.info(f"Updated CPU threads to: {num_threads}")
            
            # Update status label
            self.status_label.config(text=f"CPU threads set to: {num_threads}")
        except Exception as e:
            logging.error(f"Error updating CPU threads: {e}")
            messagebox.showerror("Error", f"Failed to update CPU threads: {e}")

    def select_audio_file(self):
        """Open a file dialog to select an audio file."""
        self.audio_file = filedialog.askopenfilename(
            title="Select Audio File", 
            filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg *.aac *.m4a")]
        )
        if self.audio_file:
            file_name = os.path.basename(self.audio_file)
            self.selected_file_label.config(text=f"Selected File: {file_name}")
            self.status_label.config(text="")
            
            # Check if format is supported
            supported_formats = (".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a")
            if not self.audio_file.lower().endswith(supported_formats):
                messagebox.showerror("Error", "Unsupported audio file format. Please select a supported file.")
                self.audio_file = None
                self.selected_file_label.config(text="No file selected")
                return
            
            # Only convert if needed - saves processing time
            if not self.audio_file.lower().endswith(".wav"):
                # Show conversion status
                self.progress_label.config(text="Converting audio to WAV format...")
                self.root.update_idletasks()
                
                # Convert in a separate thread to prevent GUI freezing
                threading.Thread(
                    target=self._convert_audio_to_wav,
                    daemon=True
                ).start()
            else:
                self.converted_audio_file = self.audio_file
                self.progress_label.config(text="Ready to transcribe")
                
    def _convert_audio_to_wav(self):
        """Convert audio to WAV in a separate thread."""
        try:
            self.converted_audio_file = os.path.splitext(self.audio_file)[0] + "_temp.wav"
            audio = AudioSegment.from_file(self.audio_file)
            audio.export(self.converted_audio_file, format="wav")
            self.gui_queue.put(lambda: self.progress_label.config(text="Audio conversion complete. Ready to transcribe."))
        except Exception as e:
            logging.error(f"Error converting audio: {e}")
            self.gui_queue.put(lambda e=e: messagebox.showerror("Error", f"Error converting audio: {e}"))
            self.gui_queue.put(lambda: self.progress_label.config(text="Error converting audio"))
                
    def start_transcribing(self):
        """Start the transcription process with proper UI updates."""
        if not self.audio_file or not os.path.exists(self.audio_file):
            messagebox.showwarning("Warning", "Please select a valid audio file first.")
            return
            
        if not self.converted_audio_file or not os.path.exists(self.converted_audio_file):
            messagebox.showwarning("Warning", "Audio conversion not complete. Please wait.")
            return
            
        # Update UI state
        self.transcription_running = True
        self.cancel_requested = False
        self.transcribe_button.config(state='disabled')
        self.select_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        
        # Clear existing text before starting new transcription
        self.transcription_text.delete(1.0, tk.END)
        
        # Reset progress indicators
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Starting transcription...")
        self.status_label.config(text="")
        
        # Start transcription in background
        self.current_task = self.executor.submit(
            self.transcribe_audio, 
            self.converted_audio_file
        )
    
    def cancel_transcription(self):
        """Cancel the ongoing transcription."""
        if self.transcription_running:
            self.cancel_requested = True
            self.status_label.config(text="Cancelling... (may take a moment)")
            # The transcribe_audio function checks for this flag periodically
            
    def transcribe_audio(self, audio_file):
        """Transcribe the audio file with batch updates."""
        try:
            selected_model = self.model_var.get()
            selected_language = self.language_var.get()
            batch_size = self.batch_size_var.get()
            
            start_time = time.time()
            
            # Update UI for model loading
            self.gui_queue.put(lambda: self.progress_label.config(text="Loading model..."))
            model = get_whisper_model(selected_model, "cpu")
            
            # Check if cancelled during model loading
            if self.cancel_requested:
                raise InterruptedError("Transcription cancelled by user")
            
            # Split audio into smaller chunks for better cancel response
            chunks = self.chunk_audio(audio_file, chunk_duration=15)  # Reduced chunk size
            
            total_segments = 0
            processed_segments = 0
            results = []
            
            # Process chunks with cancel checks
            for chunk in chunks:
                if self.cancel_requested:
                    raise InterruptedError("Transcription cancelled by user")
                    
                try:
                    result = model.transcribe(
                        chunk,
                        language=selected_language,
                        fp16=False,
                        beam_size=1,
                        best_of=1,
                        temperature=0.0
                    )
                    
                    if "segments" in result:
                        results.extend(result["segments"])
                        processed_segments += len(result["segments"])
                        self.update_progress(processed_segments, total_segments or len(chunks)*10)
                    
                    # Display partial results
                    if result["segments"]:
                        self.gui_queue.put(lambda segs=result["segments"]: self.display_batch(segs))
                    
                except Exception as e:
                    if self.cancel_requested:
                        raise InterruptedError("Transcription cancelled by user")
                    raise e
                    
                # Check cancel between chunks
                if self.cancel_requested:
                    raise InterruptedError("Transcription cancelled by user")
                    
                # Small delay for CPU cooling and cancel responsiveness
                time.sleep(0.05)
                
            # Only show completion if not cancelled
            if not self.cancel_requested:
                total_time = time.time() - start_time
                self.gui_queue.put(lambda t=total_time: self.status_label.config(
                    text=f"✓ Complete in {t:.1f}s ({processed_segments} segments)"
                ))

        except InterruptedError:
            self.gui_queue.put(lambda: self.status_label.config(text="Transcription cancelled"))
            # Clean up partial results
            self.gui_queue.put(lambda: self.progress_bar.configure(value=0))
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            self.gui_queue.put(lambda e=e: messagebox.showerror("Error", str(e)))
        finally:
            # Clean up temporary chunk files
            for chunk in chunks if 'chunks' in locals() else []:
                try:
                    if chunk != audio_file and os.path.exists(chunk):
                        os.remove(chunk)
                except Exception as e:
                    logging.error(f"Error cleaning up chunk file: {e}")
            
            # Reset UI state
            self.gui_queue.put(self.reset_ui_after_transcription)
            
    def reset_ui_after_transcription(self):
        """Reset UI elements after transcription completes."""
        self.transcription_running = False
        self.cancel_requested = False
        self.transcribe_button.config(state='normal')
        self.select_button.config(state='normal')
        self.cancel_button.config(state='disabled')
        self.progress_bar["value"] = 100 if not self.cancel_requested else 0
            
    def display_batch(self, segments):
        """Display a batch of transcription segments in the text area."""
        for segment in segments:
            text = segment["text"].strip()
            
            # Get current position to add timestamps as needed
            current_pos = self.transcription_text.index(tk.END)
            line_num = int(float(current_pos))
            
            # Format timestamp for this segment
            timestamp = f"[{self.format_time(segment['start'])} → {self.format_time(segment['end'])}] "
            
            # For first segment or if checkbox is checked, include timestamp
            if line_num == 1:  # First segment
                self.transcription_text.insert(tk.END, f"{timestamp}{text}\n")
            else:
                # Check if we need a space before the new text
                if not text.startswith((",", ".", "!", "?", ":", ";", ")", "]", "}")):
                    self.transcription_text.insert(tk.END, f" {text}")
                else:
                    self.transcription_text.insert(tk.END, text)
        
        # Auto-scroll to see latest text
        self.transcription_text.see(tk.END)
        
    def format_time(self, seconds):
        """Format seconds as mm:ss"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
            
    def save_transcription(self):
        """Save the transcription to a text file with proper Unicode encoding."""
        text = self.transcription_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No transcription available to save.")
            return
            
        try:
            save_file = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt")]
            )
            if save_file:
                with open(save_file, "w", encoding="utf-8") as file:
                    file.write(text)
                messagebox.showinfo("Success", f"Transcription saved to {save_file}")
        except Exception as e:
            logging.error(f"Error saving transcription: {e}")
            messagebox.showerror(
                "Error",
                "Failed to save file. Make sure you have write permissions and enough disk space."
            )
            
    def refresh_transcription(self):
        """Reset the transcription and UI."""
        # Cancel if running
        if self.transcription_running:
            self.cancel_transcription()
        
        # Clear text and reset labels
        self.transcription_text.delete(1.0, tk.END)
        self.status_label.config(text="")
        self.progress_label.config(text="")
        self.progress_bar["value"] = 0
                
    def _cleanup_temp_files(self):
        """Clean up any temporary WAV files."""
        if self.converted_audio_file and self.converted_audio_file != self.audio_file:
            try:
                if os.path.exists(self.converted_audio_file):
                    os.remove(self.converted_audio_file)
                    logging.info(f"Cleaned up temporary file: {self.converted_audio_file}")
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {e}")
    
    def on_closing(self):
        """Clean up resources when closing the application."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Set cancel flag for any running operation
            self.cancel_requested = True
            
            if self.transcription_running:
                # Give a moment for threads to respond to cancel
                try:
                    self.root.after(500)  # Brief pause
                except:
                    pass
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Clear cache and shutdown executor
            try:
                get_whisper_model.cache_clear()
                self.executor.shutdown(wait=False)
            except:
                pass
                
            self.root.destroy()
            
    def process_gui_queue(self):
        """Process tasks in the GUI queue."""
        try:
            # Process up to 10 items at once for smoother operation
            for _ in range(10):
                if self.gui_queue.empty():
                    break
                task = self.gui_queue.get_nowait()
                task()
        except Exception as e:
            logging.error(f"Error in GUI queue: {e}")
        finally:
            # Schedule the next check
            if not self.cancel_requested:
                self.root.after(50, self.process_gui_queue)

    # Add this new method to split long audio files
    def chunk_audio(self, audio_file, chunk_duration=30):
        """Split long audio files into chunks for faster processing."""
        try:
            audio = AudioSegment.from_file(audio_file)
            chunks = []
            
            # Split audio into 30-second chunks
            for i in range(0, len(audio), chunk_duration * 1000):
                chunk = audio[i:i + chunk_duration * 1000]
                chunk_path = f"{audio_file}_chunk_{i//1000}.wav"
                chunk.export(chunk_path, format="wav")
                chunks.append(chunk_path)
            
            return chunks
        except Exception as e:
            logging.error(f"Error chunking audio: {e}")
            return [audio_file]

    # Add this method for parallel processing
    def process_chunks(self, chunks, model):
        """Process audio chunks with CPU management"""
        chunk_size = max(1, len(chunks) // self.cpu_threads)
        results = []
        
        with ThreadPoolExecutor(max_workers=self.cpu_threads) as executor:
            futures = []
            
            for i in range(0, len(chunks), chunk_size):
                batch = chunks[i:i + chunk_size]
                future = executor.submit(
                    self._process_batch,
                    batch,
                    model
                )
                futures.append(future)
            
            for future in futures:
                if not self.cancel_requested:
                    results.extend(future.result())
        
        return results

    # Add memory cleanup method
    def cleanup_memory(self):
        """Clean up memory after transcription."""
        import gc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # Update optimize_audio method
    def optimize_audio(self, audio_file):
        """Optimize audio for faster processing"""
        try:
            audio = AudioSegment.from_file(audio_file)
            # Convert to mono
            audio = audio.set_channels(1)
            # Lower sample rate
            audio = audio.set_frame_rate(16000)
            # Normalize audio
            audio = audio.normalize()
            # Remove silence
            audio = detect_silence(audio, silence_thresh=-40)
            # Export optimized
            optimized_path = f"{audio_file}_optimized.wav"
            audio.export(
                optimized_path, 
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            return optimized_path
        except Exception as e:
            logging.error(f"Error optimizing audio: {e}")
            return audio_file

    def update_progress(self, current, total):
        """Update progress bar and label."""
        percentage = (current / total) * 100
        self.gui_queue.put(lambda: self.progress_bar.configure(value=percentage))
        self.gui_queue.put(lambda: self.progress_label.config(
            text=f"Processing: {percentage:.0f}% ({current}/{total} segments)"
        ))

    def _process_batch(self, batch, model):
        """Process a batch with cooling periods"""
        results = []
        for chunk in batch:
            if self.cancel_requested:
                break
                
            result = model.transcribe(
                chunk,
                language=self.language_var.get(),
                fp16=False,
                beam_size=1,
                best_of=1,
                temperature=0.0
            )
            results.append(result)
            
            # Add small delay to prevent CPU overheating
            time.sleep(0.1)
        
        return results

    def optimize_processing(self):
        """Configure optimal processing settings"""
        # Set environment variables for better performance
        os.environ['OMP_NUM_THREADS'] = str(self.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.cpu_threads)
        
        # Configure torch for faster processing
        torch.set_grad_enabled(False)  # Disable gradients
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernWhisperApp(root)
    root.mainloop()