# 🎙️ Voice Activity Detection (VAD) with Whisper Transcription

This Colab notebook performs voice activity detection (VAD) using `pyannote.audio`, splits the audio into chunks, and transcribes each segment using OpenAI's Whisper model (`faster-whisper`). The results are saved to both `.txt` and `.json` formats.

---

## 🛠️ Setup

### 🔧 Install dependencies
```bash
!pip install pyannote.audio
!pip install faster-whisper
!pip install torchaudio
!pip install ffmpeg-python pydub
```

---

## 🔐 Hugging Face Authentication

Some models like `pyannote/segmentation` are gated and require access approval.

### ✅ Access Gated Model
1. Visit [hf.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation) and **accept the user conditions**.
2. Go to [hf.co/settings/tokens](https://huggingface.co/settings/tokens) to **create a read access token**.
3. Use the token like this:
```python
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token="your_token_here")
```

### 🔐 Set the token for use
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"


Get your Hugging Face token from https://huggingface.co/settings/tokens and set it:

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```
---

## 🎵 Audio Upload and Conversion

Upload an audio file (`.mp3`) via Colab UI:

```python
from google.colab import files
uploaded = files.upload()
```

Convert MP3 to WAV with required format:

```python
!ffmpeg -i "input.mp3" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
```

---

## 🧠 Step 1: Voice Activity Detection (VAD)

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=os.getenv("HF_TOKEN"))
vad_result = pipeline("output.wav")
```

---

## ✂️ Step 2: Split Audio into Chunks

```python
from pydub import AudioSegment

audio = AudioSegment.from_file("output.wav")
chunks = []

for i, speech_turn in enumerate(vad_result.get_timeline()):
    start_ms = int(speech_turn.start * 1000)
    end_ms = int(speech_turn.end * 1000)
    chunk = audio[start_ms:end_ms]
    chunk_path = f"chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")
    chunks.append((chunk_path, speech_turn.start, speech_turn.end))
```

---

## 📝 Step 3: Transcribe with Whisper

```python
from faster_whisper import WhisperModel
import datetime

def convert_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

def transcribe_chunks_with_timestamps(chunks, whisper_model="base", compute_type="float16"):
    model = WhisperModel(whisper_model, compute_type=compute_type)
    transcripts = []

    for idx, (path, start, end) in enumerate(chunks):
        segments, _ = model.transcribe(path)
        text = " ".join(segment.text.strip() for segment in segments)

        transcripts.append({
            "segment_id": idx,
            "start": convert_time(start),
            "end": convert_time(end),
            "duration": str(datetime.timedelta(seconds=round(end - start))),
            "text": text
        })

    return transcripts

transcripts = transcribe_chunks_with_timestamps(chunks)
```

---

## 💾 Step 4: Save Outputs

### Save to `.txt`
```python
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(f"{'START':<10} {'END':<10} Text\n")
    f.write("=" * 80 + "\n")
    for segment in transcripts:
        f.write(f"{segment['start']:<10} {segment['end']:<10} {segment['text']}\n\n")

files.download("transcript.txt")
```

### Save to `.json`
```python
import json

with open("transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcripts, f, indent=4)

files.download("transcript.json")
```

---

## 📌 Notes

- This is built for use in Google Colab.
- All audio is processed at 16kHz mono WAV format (`pcm_s16le`).
- Whisper model used is `"base"` — change to `"small"` or `"medium"` for better quality.

---

## 📄 License

MIT or your custom license.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
