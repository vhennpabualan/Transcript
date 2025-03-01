import pytest
import tkinter as tk
import queue
from unittest.mock import MagicMock, patch
from whisper_gui import ModernWhisperApp

@pytest.fixture
def app():
    root = tk.Tk()
    app = ModernWhisperApp(root)
    
    # Mock necessary GUI components
    app.transcription_text = MagicMock()
    app.model_var = tk.StringVar(value="medium")
    app.language_var = tk.StringVar(value="en")
    app.progress_bar = MagicMock()
    app.progress_label = MagicMock()
    app.green_check_label = MagicMock()
    app.gui_queue = queue.Queue()
    
    yield app
    
    root.destroy()

class TestWhisperGUI:
    """Test suite for WhisperGUI application"""

    def test_update_progress(self, app):
        """Test progress updates"""
        app.update_progress(50, 100)
        
        # Process GUI queue
        while not app.gui_queue.empty():
            task = app.gui_queue.get()
            task()
        
        app.progress_bar.configure.assert_called_with(value=50)
        app.progress_label.config.assert_called_with(
            text="Processing: 50% (50/100 segments)"
        )

    def test_save_transcription(self, app):
        """Test save functionality"""
        with patch('tkinter.filedialog.asksaveasfilename') as mock_save_dialog:
            mock_save_dialog.return_value = "test.txt"
            with patch('builtins.open', create=True) as mock_open:
                app.transcription_text.get.return_value = "Test transcription"
                app.save_transcription()
                mock_open.assert_called_once()

    def test_refresh_transcription(self, app):
        """Test refresh functionality"""
        app.refresh_transcription()
        # Additional tests for whisper_gui.py
        def test_model_selection(app):
            """Test model selection functionality"""
            app.model_var.set("small")
            assert app.model_var.get() == "small"
            
            app.model_var.set("medium") 
            assert app.model_var.get() == "medium"

        def test_language_selection(app):
            """Test language selection"""
            app.language_var.set("fr")
            assert app.language_var.get() == "fr"
            
            app.language_var.set("es")
            assert app.language_var.get() == "es"

        def test_batch_size_selection(app):
            """Test batch size selection"""
            app.batch_size_var.set(5)
            assert app.batch_size_var.get() == 5
            
            app.batch_size_var.set(10)
            assert app.batch_size_var.get() == 10

        @patch('whisper.load_model')
        def test_transcription_cancel_during_model_load(mock_model, app):
            """Test cancellation during model loading"""
            app.cancel_requested = True
            app.transcribe_audio("test.wav")
            assert app.status_label.config.called_with(text="Transcription cancelled")

        def test_error_handling(app):
            """Test error handling during transcription"""
            with patch('whisper.load_model', side_effect=Exception("Test error")):
                app.transcribe_audio("test.wav")
                assert "Error" in app.status_label.config.call_args[1]['text']

        def test_progress_updates(app):
            """Test progress updates during transcription"""
            updates = []
            app.progress_label.config = lambda **kwargs: updates.append(kwargs['text'])
            app.update_progress(25, 100)
            app.update_progress(50, 100)
            app.update_progress(75, 100)
            assert len(updates) == 3
            assert "25%" in updates[0]
            assert "50%" in updates[1]
            assert "75%" in updates[2]

        def test_file_cleanup(app, tmp_path):
            """Test cleanup of temporary files"""
            test_files = ["temp1.wav", "temp2.wav", "temp3.wav"]
            for f in test_files:
                (tmp_path / f).touch()
                app._cleanup_temp_files()
            for f in test_files:
                assert not (tmp_path / f).exists()

        def test_audio_optimization_error(app):
            """Test error handling in audio optimization"""
            with patch('pydub.AudioSegment.from_file', side_effect=Exception("Test error")):
                result = app.optimize_audio("bad_file.wav")
                assert result == "bad_file.wav"  # Should return original file on error

        def test_chunk_audio_error(app):
            """Test error handling in audio chunking"""
            with patch('pydub.AudioSegment.from_file', side_effect=Exception("Test error")):
                result = app.chunk_audio("bad_file.wav")
                assert result == ["bad_file.wav"]  # Should return original file on error

        @pytest.mark.parametrize("file_name", [
            "test.txt", "test.doc", "test.pdf", "test.jpg"
        ])
        def test_invalid_file_types(app, file_name):
            """Test handling of various invalid file types"""
            with patch('tkinter.filedialog.askopenfilename', return_value=file_name):
                app.select_audio_file()
                assert app.audio_file is None
