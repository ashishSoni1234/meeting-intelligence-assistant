# Sample Inputs

Real meeting files provided for reviewers to test the system immediately — no setup needed.

## Files in this directory

| File | Size | Description |
|------|------|-------------|
| `Q1-2024-Earnings-Presentation.pdf` | ~2.3 MB | KBR Q1-2024 earnings presentation, 21 pages — same as the screenshots in the main README |
| `meeting.mp4` | ~19 MB | ~5 min real meeting recording matching the presentation |

## How to run

1. Start the app: `python app.py`
2. Open the Gradio UI (port 7860)
3. In **Upload & Process** tab:
   - Upload `Q1-2024-Earnings-Presentation.pdf` in the PDF slot
   - Upload `meeting.mp4` in the Video slot
   - Leave the Audio slot empty (pipeline extracts audio from the video automatically)
4. Click **Process Files**
5. Switch to **Ask Questions** tab

## Expected output after processing

```
✅ Processing complete in ~23s | 21 slides | ~10 frames | ~198 chunks indexed
```

Verified on JarvisLabs NVIDIA A100-PCIE-40GB.

| Component | Expected value |
|-----------|----------------|
| Slides extracted | 21 pages |
| Video frames (30s interval) | ~10 |
| Slide transitions detected | ~3–5 |
| ChromaDB chunks indexed | ~150–200 |
| Processing time (A100) | ~20–25s |
| Processing time (CPU) | ~5–8 min |

## The three demo questions

These questions match the sample files and are the same ones shown in the README screenshots:

1. **"Which slide was being shown when the recruitment program was discussed?"**
   - Expected: Slide 5 cited, timestamp ~02:16, video frame proves slide on screen

2. **"Who were the presenters at this meeting and what roles do the slides assign them?"**
   - Expected: Stuart Bradie CEO, Mark Sopp CFO — roles from Slide 1, voice from audio

3. **"What action items or next steps were mentioned, and at what point in the meeting?"**
   - Expected: Timestamp ~05:41 cited, Slide 1 and Slide 12 referenced

## Bring your own files

The system works with any meeting content:
- **PDF:** Any presentation, 5–30 slides
- **Video:** MP4/MOV screen recording or camera recording (audio extracted automatically)
- **Audio (optional):** Upload a separate MP3/WAV if you want speaker diarization without a video
