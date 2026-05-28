# Meeting Intelligence Assistant

## What it does

After every meeting you lose critical context — pricing decisions get buried in hour-long recordings, the exact slide being discussed when a key number was raised is impossible to recall, and nobody agrees on what was said. This system lets you upload your meeting recording, PDF slides, and video, then ask natural language questions and get answers grounded in exact timestamps, slide numbers, and speaker attribution. It processes everything in parallel on GPU and answers in seconds using Qwen2.5-Omni's cross-modal reasoning.

## Why I built this

Every time I reviewed a recorded meeting I found myself scrubbing through video looking for a specific moment, cross-referencing a slide PDF, and trying to remember who said what. The information was all there — just completely inaccessible. I wanted a system where I could just ask "what did Rahul say about the pricing change and which slide was he on?" and get a precise, cited answer in under 10 seconds. Deploying on JarvisLabs A100 makes this practical: Whisper Large v3 transcribes an hour of audio in under 3 minutes, and Qwen2.5-Omni can reason across text and images simultaneously.

## How to run it

### On JarvisLabs (Recommended)

```bash
git clone https://github.com/ashishSoni1234/meeting-intelligence-assistant
cd meeting-intelligence-assistant
chmod +x setup.sh
./setup.sh
# Edit .env to add your HUGGINGFACE_TOKEN and GROQ_API_KEY
nano .env
python app.py
```

Open the JarvisLabs port-forwarding URL for port 7860 in your browser.

To also run the FastAPI backend (optional, for API access):
```bash
# In a second terminal:
python main.py
# API docs at http://localhost:8000/docs
```

### Local (CPU only, slow)

```bash
git clone https://github.com/ashishSoni1234/meeting-intelligence-assistant
cd meeting-intelligence-assistant

# Install dependencies
pip install torch torchvision torchaudio  # CPU version
pip install -r requirements.txt

# Install system deps (macOS)
brew install poppler ffmpeg

# Install system deps (Ubuntu/Debian)
apt-get install -y poppler-utils ffmpeg

# Configure
cp .env.example .env
# Edit .env: set DEVICE=cpu, WHISPER_MODEL=base (faster on CPU)

python app.py
```

> **Note:** On CPU, Whisper transcription of a 1-hour meeting takes ~20 minutes. Qwen inference takes ~2-5 minutes per question. Use `WHISPER_MODEL=base` and `GROQ_API_KEY` for Groq fallback to make CPU mode practical.

### Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_TOKEN` | Yes (for Qwen) | HF token to download Qwen2.5-Omni-7B |
| `GROQ_API_KEY` | Optional | Fallback LLM if local model unavailable |
| `DEVICE` | Auto-detected | `cuda` or `cpu` |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `FRAME_INTERVAL` | `30` | Seconds between video frame extractions |

---

## Architecture decisions

### Decision 1: Qwen2.5-Omni-7B over Nemotron

Qwen2.5-Omni is a native multimodal model that accepts interleaved text and image tokens in a single forward pass. Nemotron is primarily a text model. For this use case — simultaneously reasoning about a transcript excerpt, a slide image, and a video frame — native multimodal attention is essential. Qwen2.5-Omni was trained on instruction-following with image inputs, meaning it understands "which slide is shown in this frame?" without any fine-tuning. The 7B size fits comfortably on A100 40GB in 8-bit quantization (~8GB VRAM), leaving room for Whisper and ChromaDB.

### Decision 2: ChromaDB for vector store

ChromaDB runs entirely in-process (no separate server required), persists to disk automatically, and has a simple Python API that doesn't require Kubernetes or Docker. For a single-node JarvisLabs deployment this is the right call — zero infrastructure overhead, sub-millisecond local queries, and built-in HNSW approximate nearest neighbor search. Pinecone or Weaviate would add latency, cost, and network dependencies without meaningful benefit at this scale.

### Decision 3: Parallel vs sequential processing

The three processing tasks (Whisper transcription, PDF rendering, video frame extraction) are completely independent — they read different files and write to different directories. Running them in parallel with `ThreadPoolExecutor(max_workers=3)` cuts total ingestion time by roughly 50-65% on a real meeting (e.g., 4 minutes instead of 10). The GIL isn't a bottleneck here because Whisper runs in C++ via CTranslate2, OpenCV is also C++, and PyMuPDF is Cython — all release the GIL during heavy computation.

---

## Cross-modal example questions

All three questions below are answered using the provided sample inputs (`Q1-2024-Earnings-Presentation.pdf` + `meeting.mp4`). None can be answered from a single input — each requires combining audio transcript, PDF slide content, and video frames.

---

**Q1: "Which slide was being shown when the recruitment program was discussed?"**

![Q&A screenshot showing recruitment program answer with Slide 5 image and video frame at timestamp 02:16](assets/screenshot_qa2.png)

```
Citations: 🕐 02:16 | 📊 Slide 5

[Timestamp: 02:16] [Slide: 5]
The recruitment program was discussed at 02:16 while Slide 5
("DEVELOPING OUR PEOPLE GLOBALLY") was being displayed.
The speaker asked: "Trudy, could you summarise the recent recruitment programme?"

[Slide 5 content]: DEVELOPING OUR PEOPLE GLOBALLY
```
> **Why cross-modal:** The audio transcript gives WHEN recruitment was mentioned [02:16]. The video frame proves WHICH slide was on screen at that exact moment. The PDF gives the full text of Slide 5. Audio alone cannot tell you which slide; PDF alone cannot tell you when; video alone has no spoken words.

---

**Q2: "Who were the presenters at this meeting and what roles do the slides assign them?"**

![Q&A screenshot showing cross-modal answer with citations, relevant slide image, and video frame](assets/screenshot_qa.png)

```
Citations: 🕐 00:24 | 📊 Slide 1 | 🗣 Stuart Bradie

[Timestamp: 00:24] [Slide: 1]
The presenters introduced at the start of the meeting were:
- Stuart Bradie — President and CEO (speaking at [00:24])
- Mark Sopp — Executive VP and CFO
- Jamie DuBray — VP of Investor Relations

Slide 1 lists all three with their exact titles and the date: April 30, 2024.
The video frame at [00:24] confirms Slide 1 was on screen during introductions.
```
> **Why cross-modal:** WHO was speaking at [00:24] comes from audio (speaker diarization identifies Stuart Bradie's voice). The TITLES (CEO, CFO) come from PDF Slide 1 — the audio never spells out the full titles. The video frame confirms Slide 1 was on screen at that moment, not a later slide. All three required.

---

**Q3: "What action items or next steps were mentioned, and at what point in the meeting?"**

![Q&A screenshot showing action items answer with Slide 1 and Slide 12 referenced alongside video frame](assets/screenshot_qa3.png)

```
Citations: 🕐 00:24 | 🕐 05:41 | 📊 Slide 1 | 📊 Slide 12

[Timestamp: 05:41] — "Let's summarise the actions and close the meeting."
The agenda contained seven items including any other business.

[00:24] [Slide 1] — Stuart Bradie (President and CEO) opened with the agenda.
April 30, 2024. Attendees: Stuart Bradie, Mark Sopp, Jamie DuBray.

[Quote: "The agenda contains seven items, including any other business."]
```
> **Why cross-modal:** WHEN the closing/action summary happened comes from the audio transcript [05:41]. WHO was responsible (Stuart Bradie CEO) requires matching the voice to the PDF title slide (Slide 1 lists his role). The video frame at [05:41] confirms the meeting was wrapping up. Audio alone gives the words but not the role; PDF alone gives the role but not the timing.

---

### Cross-modal evidence summary

| Question | Audio gives | PDF gives | Video gives |
|----------|------------|-----------|-------------|
| Which slide for recruitment? | ✅ Timestamp [02:16] when "recruitment" was said | ✅ Slide 5 full text "Developing Our People Globally" | ✅ Frame proves Slide 5 was on screen at [02:16] |
| Presenters and their roles? | ✅ [00:24] voice identifies Stuart Bradie speaking | ✅ Slide 1 lists CEO/CFO/VP titles | ✅ Frame confirms Slide 1 on screen at introduction |
| Action items and timing? | ✅ [05:41] "Let's summarise the actions" captured | ✅ Slide 1 and Slide 12 provide agenda context | ✅ Frame at [05:41] confirms meeting closing phase |

Each answer is **unanswerable from any single input** — it requires combining signals across audio, PDF, and video.

---

## What I used AI for

**Generated (with AI assistance):**
- Initial boilerplate for FastAPI endpoint signatures and Pydantic models
- ChromaDB collection setup and upsert patterns
- Gradio layout structure and component wiring
- setup.sh bash structure and color formatting
- Docstrings for all functions

**Written by hand (human judgment required):**
- The fallback hierarchy logic in `omni_model.py` (8-bit → 4-bit → text-only → Groq)
- The slide transition detection algorithm in `video_processor.py`
- The temporal slide estimator in `ingestion.py` (mapping video transitions to slide numbers)
- The grounding extraction regex patterns in `grounding.py`
- All architectural decisions and their rationale
- Error handling strategy (which errors to surface to UI vs. log only)

---

## What I would change with 4 more weeks

1. **Real-time streaming transcription**: Use Whisper's streaming API to start indexing while the meeting is still running, so questions can be answered mid-meeting.

2. **Speaker identity resolution**: Replace generic "Speaker 1/2" labels with real names by prompting the user to identify speakers after transcription, then backfilling all segments.

3. **Slide-to-transcript alignment via OCR**: Use TrOCR to extract text from video frames and compare it to PDF slide text, giving a more accurate frame→slide mapping than the transition-counting heuristic.

4. **Multi-meeting search**: Allow querying across a corpus of indexed meetings ("has Q3 budget been discussed before?") with a per-meeting namespace in ChromaDB.

5. **Highlight reel generation**: Use the grounding engine to automatically clip the 3 most important moments from the video and export them as a short summary video.

---

## Screenshots

Deployed and tested on **JarvisLabs A100-PCIE-40GB** at `7a4d913d1b75039de8.gradio.live`.

**Upload & Process** — KBR Q1-2024 earnings PDF (21 slides) + meeting video processed in 23.8s, 198 chunks indexed on NVIDIA A100:

![Upload & Process tab showing PDF + Video processing complete in 23.8s with 21 slides, 12 frames, 198 chunks indexed on NVIDIA A100](assets/screenshot_upload.png)

**Ask Questions** — Cross-modal answer with citations (timestamps + slide numbers), relevant slide image, and video frame shown side by side:

![Q&A tab showing grounded answer with citations, relevant slide image from PDF, and video frame from the recording](assets/screenshot_qa.png)

**Meeting Summary** — Auto-generated structured summary with KEY DECISIONS, ACTION ITEMS, and TOPIC TIMELINE all with slide references:

![Meeting Summary tab showing structured output with key decisions, action items, and topic timeline referencing slide numbers and timestamps](assets/screenshot_summary.png)

---

## Sample inputs

Two sample files are provided in `sample_inputs/` — a real Q1-2024 earnings presentation with its meeting recording:

| File | Type | Description |
|------|------|-------------|
| `Q1-2024-Earnings-Presentation.pdf` | PDF | 21-page KBR Q1-2024 earnings presentation (the same meeting in the screenshots) |
| `meeting.mp4` | Video | ~5 min real meeting recording — audio is extracted automatically by the pipeline |

> **Note:** Upload both files in the UI. A separate audio file is optional — the pipeline automatically extracts the audio track from the video using ffmpeg if no MP3 is provided. You can also upload a separate audio file alongside the video for speaker diarization.

After processing these two files you should see approximately:
- **Slides:** 21 pages extracted to `extracted_slides/`
- **Video frames:** ~10 frames at 30s interval
- **ChromaDB:** ~150–200 chunks indexed
- **Processing time:** ~20–25s on A100, ~5–8 min on CPU

---

## JarvisLabs GPU usage

| Component | Runs on GPU | Why |
|-----------|-------------|-----|
| Whisper Large v3 | ✅ CUDA | CTranslate2 CUDA kernel — 10x faster than CPU for float16 |
| Qwen2.5-Omni-7B | ✅ CUDA (8-bit) | 7B model requires GPU for sub-10s inference per question |
| PyMuPDF PDF rendering | ❌ CPU | Fast enough on CPU; no GPU PDF rasterizer needed |
| OpenCV frame extraction | ❌ CPU | Sequential I/O — GPU JPEG decode adds complexity without speedup |
| Sentence Transformers | ✅ CUDA (if available) | Embedding 500-2000 chunks is 5x faster on GPU |
| ChromaDB HNSW search | ❌ CPU | In-memory index; queries are <5ms regardless |

**VRAM budget on A100 40GB:**
- Whisper Large v3: ~3 GB
- Qwen2.5-Omni-7B (8-bit): ~8 GB
- Sentence Transformers (all-MiniLM): ~0.4 GB
- PyTorch overhead + activations: ~3 GB
- **Total: ~14.4 GB — well within 40 GB**

Models are lazy-loaded and unloaded between heavy operations to keep peak VRAM usage under 20 GB.
