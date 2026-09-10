from whisper.process_video import extract_tutorial_number, is_processable_media


def test_extract_tutorial_number_from_episode_name():
    assert extract_tutorial_number("05 - #5 Python🐍 Program for Interview Preparation ｜ Find string is palindrome or not ｜ Python Programming.mp4") == "05"
    assert extract_tutorial_number("31 - Easiest Way to Plot 📈using Matplotlib in Python 🐍.mp4") == "31"


def test_only_processable_media_files_are_selected():
    assert is_processable_media("05 - #5 Python🐍 Program for Interview Preparation ｜ Find string is palindrome or not ｜ Python Programming.mp4") is True
    assert is_processable_media("05 - #5 Python🐍 Program for Interview Preparation ｜ Find string is palindrome or not ｜ Python Programming.en.vtt") is False
