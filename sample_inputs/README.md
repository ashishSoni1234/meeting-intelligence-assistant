# Sample Inputs

Three sample files are provided to let reviewers test the system immediately.

## Files in this directory

| File | Size | Description |
|------|------|-------------|
| `sample_meeting.mp3` | ~2 MB | Synthetic meeting audio (~3 min, covers pricing/budget/timeline) |
| `sample_slides.pdf` | ~500 KB | 15-slide presentation deck matching the audio |
| `sample_meeting.mp4` | generated | Run `generate_sample_video.sh` to create this from the audio |

## Generate the video sample

The video file is not committed (it is just audio + a static title frame, so
it can be re-generated in seconds). From the project root:

```bash
chmod +x generate_sample_video.sh
./generate_sample_video.sh
```

Requirements: `ffmpeg` and `Pillow` (already in `requirements.txt`).

## Expected output after processing all three files

When you upload all three files and click **Process Files**, you should see:

```
✅ Processing complete in ~45s | 15 slides | 6 frames | 72 chunks indexed
```

Detailed breakdown:

| Component | Expected value |
|-----------|----------------|
| Slides extracted | 15 pages |
| Audio duration | ~3 min |
| Transcript segments | ~30 |
| Video frames (30s interval) | ~6 |
| Slide transitions detected | ~3 |
| ChromaDB chunks indexed | 60–80 |
| Processing time (A100) | ~40–60s |
| Processing time (CPU) | ~5–10 min |

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
