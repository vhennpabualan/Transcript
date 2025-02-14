import pytest
import tkinter as tk
from unittest.mock import MagicMock
from whisper_gui import ModernWhisperApp

@pytest.fixture
def app():
    root = tk.Tk()  # Create a real root window
    app = ModernWhisperApp(root)

    # Mock necessary GUI components
    app.transcription_text = MagicMock()
    app.model_var = tk.StringVar(value="medium")
    app.language_var = tk.StringVar(value="en")
    app.gpu_var = tk.BooleanVar(value=False)
    app.is_dark_mode = tk.BooleanVar(value=False)
    app.colors = {"light": {}, "dark": {}}

    yield app  # Provide the app instance to the test function
    
    root.destroy()  # Destroy the root window after test execution
