"""
generate_sample_video.py

Generates sample_inputs/sample_meeting.mp4 from sample_meeting.mp3
using only Python (Pillow + subprocess ffmpeg).

Works on Windows, macOS, Linux, and JarvisLabs.

Usage:
    python generate_sample_video.py

Requirements:
    - ffmpeg on PATH  (Ubuntu/JarvisLabs: apt-get install -y ffmpeg)
    - Pillow          (already in requirements.txt)
    - sample_inputs/sample_meeting.mp3 must exist
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def create_cover_image(output_path: str) -> None:
    """Create a 1280x720 title slide PNG using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1280, 720), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Background card
    draw.rectangle([80, 200, 1200, 520], fill=(30, 41, 59), outline=(59, 130, 246), width=3)

    # Try to use a system font; fall back to default if not found
    try:
        title_font = ImageFont.truetype("arial.ttf", 56)
        sub_font = ImageFont.truetype("arial.ttf", 32)
        footer_font = ImageFont.truetype("arial.ttf", 24)
    except (IOError, OSError):
        try:
            # Linux / JarvisLabs path
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
            sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
            footer_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except (IOError, OSError):
            title_font = ImageFont.load_default()
            sub_font = title_font
            footer_font = title_font

    # Title
    draw.text((640, 300), "Q3 Business Review", fill=(255, 255, 255), anchor="mm", font=title_font)
    # Subtitle
    draw.text(
        (640, 390),
        "Agenda: Pricing  ·  Budget  ·  Timeline",
        fill=(148, 163, 184),
        anchor="mm",
        font=sub_font,
    )
    # Divider
    draw.line([(200, 440), (1080, 440)], fill=(59, 130, 246), width=2)
    # Footer
    draw.text((640, 480), "Sample Meeting — Meeting Intelligence Assistant", fill=(71, 85, 105), anchor="mm", font=footer_font)
    draw.text((640, 660), "Powered by Qwen2.5-Omni + Whisper on JarvisLabs A100", fill=(51, 65, 85), anchor="mm", font=footer_font)

    img.save(output_path)
    print(f"  Title slide created: {output_path}")


def get_audio_duration(audio_path: str) -> float:
    """Use ffprobe to get audio duration in seconds."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def generate_video(audio_path: str, cover_path: str, output_path: str) -> None:
    """Combine still image + audio into an MP4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", cover_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1280:720",
        output_path,
    ]
    print("  Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:")
        print(result.stderr[-1000:])
        raise RuntimeError("ffmpeg failed — see output above")


def main() -> None:
    print("=" * 50)
    print("  Sample Video Generator")
    print("=" * 50)

    # Paths
    script_dir = Path(__file__).parent
    audio_path = script_dir / "sample_inputs" / "sample_meeting.mp3"
    output_path = script_dir / "sample_inputs" / "sample_meeting.mp4"

    # Validate audio
    if not audio_path.exists():
        print(f"ERROR: Audio not found: {audio_path}")
        print("Make sure sample_inputs/sample_meeting.mp3 exists.")
        sys.exit(1)

    # Check ffmpeg
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found on PATH.")
        print()
        print("Install it:")
        print("  Ubuntu / JarvisLabs : sudo apt-get install -y ffmpeg")
        print("  macOS               : brew install ffmpeg")
        print("  Windows             : https://ffmpeg.org/download.html")
        sys.exit(1)

    print(f"\nAudio : {audio_path}")
    print(f"Output: {output_path}")

    # Step 1: Create cover image in a temp file
    print("\nStep 1: Creating title slide image...")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cover_path = tmp.name

    create_cover_image(cover_path)

    # Step 2: Encode video
    print("\nStep 2: Encoding MP4 with ffmpeg...")
    generate_video(str(audio_path), cover_path, str(output_path))

    # Cleanup temp image
    Path(cover_path).unlink(missing_ok=True)

    # Report
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_000_000
        duration = get_audio_duration(str(output_path))
        print()
        print("Done!")
        print(f"  Output   : {output_path}")
        print(f"  Size     : {size_mb:.1f} MB")
        print(f"  Duration : {duration:.0f}s")
        print()
        print("You now have all 3 sample inputs:")
        print(f"  {script_dir / 'sample_inputs' / 'sample_meeting.mp3'}  (audio)")
        print(f"  {script_dir / 'sample_inputs' / 'sample_slides.pdf'}   (slides)")
        print(f"  {output_path}  (video)")
    else:
        print("ERROR: Output file was not created.")
        sys.exit(1)


if __name__ == "__main__":
    main()
