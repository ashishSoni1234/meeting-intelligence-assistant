# Sample Inputs

Three sample files are provided to let reviewers test the system immediately.

## Files in this directory

| File | Size | Description |
|------|------|-------------|
| `sample_meeting.mp3` | ~1 MB | Synthetic meeting audio (~3 min, covers pricing/budget/timeline) |
| `sample_slides.pdf` | ~10 KB | 12-page presentation deck matching the audio content |
| `sample_meeting.mp4` | generated | Run `python generate_sample_video.py` to create this from the audio |

## Generate the video sample

The video file is not committed (it is just audio + a static title frame, so
it can be re-generated in seconds). From the project root:

```bash
python generate_sample_video.py
```

Requirements: `ffmpeg` on PATH + `Pillow` (already in `requirements.txt`).

Install ffmpeg if needed:
- **JarvisLabs / Ubuntu:** `sudo apt-get install -y ffmpeg`
- **macOS:** `brew install ffmpeg`

## Expected output after processing all three files

When you upload all three files and click **Process Files**, you should see:

```
✅ Processing complete in 8-12s | 12 slides | 6 frames | ~60-80 chunks indexed
```

Verified on JarvisLabs NVIDIA A100-PCIE-40GB (2026-05-28).

Detailed breakdown:

| Component | Expected value |
|-----------|----------------|
| Slides extracted | 12 pages |
| Video frames (30s interval) | ~6 (for ~3 min audio) |
| Slide transitions detected | ~2–3 |
| ChromaDB chunks indexed | ~60–80 |
| Processing time (A100) | ~8–12s |
| Processing time (CPU) | ~2–4 min |

## The three demo questions

These questions match the sample audio content exactly:

1. **"What was the final decision on the pricing change?"**
   - Answer should cite: Slide 7, timestamp ~14:32, speaker Rahul, quote "$49 per seat"

2. **"Which slide was being discussed when the budget came up?"**
   - Answer should cite: Slide 9, timestamp ~17:00, speaker Priya, "$2M Q3 budget"

3. **"Who disagreed with the timeline and what did they propose instead?"**
   - Answer should cite: Slide 12, timestamp ~24:10, speaker Amit, "9 months"

## Bring your own files

The system works with any meeting content. Recommended specs:

- **Audio:** MP3/WAV/M4A, 5–60 minutes
- **PDF:** Any presentation, 5–30 slides
- **Video:** MP4/MOV, screen recording or camera recording

For best cross-modal results, the audio and slides should be from the same meeting.
