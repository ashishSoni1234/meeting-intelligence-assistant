#!/bin/bash
# ================================================================
# Generate sample_meeting.mp4 from existing audio + a title slide
# Run this on JarvisLabs or any machine with ffmpeg + Python
#
# Usage:
#   chmod +x generate_sample_video.sh
#   ./generate_sample_video.sh
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO="$SCRIPT_DIR/sample_inputs/sample_meeting.mp3"
OUTPUT="$SCRIPT_DIR/sample_inputs/sample_meeting.mp4"
COVER_IMG="/tmp/meeting_cover.png"

echo "=== Sample Video Generator ==="

# Check audio exists
if [ ! -f "$AUDIO" ]; then
    echo "ERROR: $AUDIO not found."
    echo "Make sure sample_inputs/sample_meeting.mp3 exists first."
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found."
    echo "Install it: apt-get install -y ffmpeg   (Ubuntu/JarvisLabs)"
    exit 1
fi

echo "Step 1: Creating title slide image..."

# Try Python PIL first (always available if requirements.txt installed)
python3 - << 'PYEOF'
from PIL import Image, ImageDraw
import sys

img = Image.new("RGB", (1280, 720), color=(15, 23, 42))
draw = ImageDraw.Draw(img)

# Title
draw.rectangle([80, 240, 1200, 480], fill=(30, 41, 59), outline=(59, 130, 246), width=2)
draw.text((640, 320), "Q3 Planning Meeting", fill=(255, 255, 255), anchor="mm")
draw.text((640, 390), "Agenda: Pricing · Budget · Timeline", fill=(148, 163, 184), anchor="mm")
draw.text((640, 640), "Recorded on JarvisLabs A100", fill=(71, 85, 105), anchor="mm")

img.save("/tmp/meeting_cover.png")
print("  Title slide image created: /tmp/meeting_cover.png")
PYEOF

echo "Step 2: Encoding video with ffmpeg..."

ffmpeg -y \
    -loop 1 -i "$COVER_IMG" \
    -i "$AUDIO" \
    -c:v libx264 -tune stillimage -crf 23 -preset fast \
    -c:a aac -b:a 128k \
    -shortest \
    -pix_fmt yuv420p \
    -vf "scale=1280:720" \
    "$OUTPUT" 2>&1 | tail -5

if [ -f "$OUTPUT" ]; then
    SIZE=$(du -sh "$OUTPUT" | cut -f1)
    DURATION=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$OUTPUT" 2>/dev/null | awk '{printf "%.0f", $1}')
    echo ""
    echo "Done!"
    echo "  Output : $OUTPUT"
    echo "  Size   : $SIZE"
    echo "  Duration: ${DURATION}s"
    echo ""
    echo "You now have all 3 sample inputs:"
    echo "  sample_inputs/sample_meeting.mp3  (audio)"
    echo "  sample_inputs/sample_slides.pdf   (slides)"
    echo "  sample_inputs/sample_meeting.mp4  (video) <- just created"
else
    echo "ERROR: Output file not created. Check ffmpeg output above."
    exit 1
fi
