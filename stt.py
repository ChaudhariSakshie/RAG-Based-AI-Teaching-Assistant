import whisper
from pathlib import Path
import json
import re


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(r"D:\AI_Projects(RAG)")

AUDIO_DIR = BASE_DIR / "audios"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"

TRANSCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# LOAD WHISPER
# ==========================================

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Whisper model loaded.")


# ==========================================
# GET AUDIO FILES
# ==========================================

audio_files = sorted(
    AUDIO_DIR.glob("*.mp3")
)

print(
    f"\nFound {len(audio_files)} audio files."
)


# ==========================================
# PROCESS EACH AUDIO FILE
# ==========================================

for audio_file in audio_files:

    # --------------------------------------
    # Extract lecture number
    # --------------------------------------

    match = re.search(
        r"(\d+)",
        audio_file.name
    )

    if not match:
        print(
            f"\nSkipping: {audio_file.name}"
        )
        continue

    lecture_number = int(
        match.group(1)
    )

    lecture_name = (
        f"lecture_{lecture_number}"
    )

    json_path = (
        TRANSCRIPT_DIR /
        f"{lecture_name}.json"
    )

    txt_path = (
        TRANSCRIPT_DIR /
        f"{lecture_name}.txt"
    )


    # --------------------------------------
    # Skip already processed lectures
    # --------------------------------------

    if json_path.exists():

        print(
            f"\nSkipping {lecture_name} "
            f"(already transcribed)"
        )

        continue


    # --------------------------------------
    # Transcribe
    # --------------------------------------

    print("\n================================")
    print(
        f"TRANSCRIBING: {audio_file.name}"
    )
    print("================================")

    result = model.transcribe(
        str(audio_file),
        fp16=False
    )


    # --------------------------------------
    # Save timestamped JSON
    # --------------------------------------

    segments = []

    for segment in result["segments"]:

        segments.append({

            "start": segment["start"],

            "end": segment["end"],

            "text": segment["text"].strip()

        })


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            segments,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------
    # Save TXT
    # --------------------------------------

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        for segment in segments:

            f.write(
                f"[{segment['start']:.2f} - "
                f"{segment['end']:.2f}] "
                f"{segment['text']}\n"
            )


    print(
        f"Saved: {json_path.name}"
    )

    print(
        f"Saved: {txt_path.name}"
    )


# ==========================================
# COMPLETE
# ==========================================

print("\n================================")
print("ALL TRANSCRIPTIONS COMPLETED")
print("================================")