#!/bin/bash
# ================================================================
# Meeting Intelligence Assistant — One-Command Setup Script
# Target: JarvisLabs GPU Instance (A100 40GB)
# Usage: chmod +x setup.sh && ./setup.sh
# ================================================================

set -e  # Exit on any error

# ─────────────────────────────────────────────────────────────────
# COLORS & FORMATTING
# ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  Meeting Intelligence Assistant — Setup${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}${BOLD}[STEP]${NC} $1"
}

print_ok() {
    echo -e "${GREEN}${BOLD}[ OK ]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}${BOLD}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}${BOLD}[ERR ]${NC} $1"
}

# ─────────────────────────────────────────────────────────────────
# STEP 0: Print header + timestamp
# ─────────────────────────────────────────────────────────────────
print_header
echo -e "  Started at: $(date)"
echo -e "  Running as: $(whoami)"
echo -e "  Directory:  $(pwd)"
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 1: Check Python version
# ─────────────────────────────────────────────────────────────────
print_step "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
print_ok "Python $PYTHON_VERSION found"

# ─────────────────────────────────────────────────────────────────
# STEP 2: Check GPU availability
# ─────────────────────────────────────────────────────────────────
print_step "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
    GPU_MEM=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
    GPU_MEM_GB=$(echo "scale=1; $GPU_MEM/1024" | bc 2>/dev/null || echo "unknown")
    print_ok "GPU detected: ${GPU_NAME} (${GPU_MEM_GB} GB VRAM)"
    DEVICE="cuda"

    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null || echo "unknown")
    print_ok "CUDA Version: $CUDA_VERSION"
else
    print_warn "No GPU detected — will run on CPU (slow but functional)"
    DEVICE="cpu"
fi

# ─────────────────────────────────────────────────────────────────
# STEP 3: Create .env from .env.example if not exists
# ─────────────────────────────────────────────────────────────────
print_step "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        # Update DEVICE in .env
        sed -i "s/DEVICE=cuda/DEVICE=${DEVICE}/" .env
        print_ok ".env created from .env.example (DEVICE=${DEVICE})"
        print_warn "Edit .env to add your HUGGINGFACE_TOKEN and GROQ_API_KEY"
    else
        print_warn ".env.example not found — creating minimal .env"
        cat > .env << EOF
DEVICE=${DEVICE}
MODEL_NAME=Qwen/Qwen2.5-Omni-7B
WHISPER_MODEL=large-v3
CHROMA_PATH=./chroma_db
FRAMES_DIR=./extracted_frames
SLIDES_DIR=./extracted_slides
FRAME_INTERVAL=30
MAX_FRAMES=50
API_HOST=0.0.0.0
API_PORT=8000
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
API_BASE_URL=http://localhost:8000
LOG_LEVEL=INFO
EOF
    fi
else
    print_ok ".env already exists — skipping"
fi

# ─────────────────────────────────────────────────────────────────
# STEP 4: Create required directories
# ─────────────────────────────────────────────────────────────────
print_step "Creating required directories..."
mkdir -p chroma_db extracted_frames extracted_slides logs sample_inputs
print_ok "Directories: chroma_db/, extracted_frames/, extracted_slides/, logs/"

# ─────────────────────────────────────────────────────────────────
# STEP 5: Upgrade pip
# ─────────────────────────────────────────────────────────────────
print_step "Upgrading pip..."
pip install --upgrade pip --quiet
print_ok "pip upgraded"

# ─────────────────────────────────────────────────────────────────
# STEP 6: Install PyTorch with CUDA support
# ─────────────────────────────────────────────────────────────────
print_step "Installing PyTorch (CUDA 12.1)..."
if [ "$DEVICE" = "cuda" ]; then
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121 --quiet
    print_ok "PyTorch (CUDA) installed"
else
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==0.17.2 --quiet
    print_ok "PyTorch (CPU) installed"
fi

# ─────────────────────────────────────────────────────────────────
# STEP 7: Install all Python requirements
# ─────────────────────────────────────────────────────────────────
print_step "Installing Python requirements (this may take 5-10 minutes)..."
pip install -r requirements.txt --quiet 2>&1 | tail -5
print_ok "All Python requirements installed"

# ─────────────────────────────────────────────────────────────────
# STEP 8: Install system packages for PDF/video support
# ─────────────────────────────────────────────────────────────────
print_step "Installing system packages (poppler, ffmpeg)..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq poppler-utils ffmpeg libsm6 libxext6
    print_ok "System packages installed via apt"
elif command -v yum &> /dev/null; then
    yum install -y poppler-utils ffmpeg
    print_ok "System packages installed via yum"
elif command -v brew &> /dev/null; then
    brew install poppler ffmpeg
    print_ok "System packages installed via brew"
else
    print_warn "Could not install system packages automatically."
    print_warn "Please install manually: poppler-utils ffmpeg"
fi

# ─────────────────────────────────────────────────────────────────
# STEP 9: Pre-download Whisper Large v3 model
# ─────────────────────────────────────────────────────────────────
print_step "Pre-downloading Whisper Large v3 model (~3GB)..."
python3 - << 'PYEOF'
import sys
try:
    from faster_whisper import WhisperModel
    print("  Downloading Whisper large-v3...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", download_root="./models/whisper")
    del model
    print("  Whisper large-v3 downloaded successfully")
except Exception as e:
    print(f"  Warning: Could not pre-download Whisper: {e}")
    print("  It will be downloaded on first use.")
PYEOF
print_ok "Whisper model ready"

# ─────────────────────────────────────────────────────────────────
# STEP 10: Pre-download Qwen2.5-Omni-7B weights
# ─────────────────────────────────────────────────────────────────
print_step "Pre-downloading Qwen2.5-Omni-7B weights (~15GB — this takes time)..."
# Load HF token if available
HF_TOKEN=""
if [ -f ".env" ]; then
    HF_TOKEN=$(grep "^HUGGINGFACE_TOKEN=" .env | cut -d'=' -f2 | xargs)
fi

python3 - << PYEOF
import sys, os
os.environ.setdefault("HF_HOME", "./models/hf_cache")

hf_token = "${HF_TOKEN}"
if not hf_token or hf_token == "hf_your_token_here":
    print("  No HuggingFace token found — skipping Qwen model pre-download.")
    print("  Model will be downloaded on first use.")
    print("  Add HUGGINGFACE_TOKEN to .env to pre-download.")
    sys.exit(0)

try:
    from huggingface_hub import snapshot_download
    print("  Downloading Qwen2.5-Omni-7B (~15GB)...")
    path = snapshot_download(
        repo_id="Qwen/Qwen2.5-Omni-7B",
        token=hf_token,
        cache_dir="./models/hf_cache",
        ignore_patterns=["*.pt", "flax_model*", "tf_model*"],
    )
    print(f"  Model downloaded to: {path}")
except Exception as e:
    print(f"  Warning: Could not pre-download Qwen model: {e}")
    print("  It will be downloaded on first use.")
PYEOF
print_ok "Qwen model step complete"

# ─────────────────────────────────────────────────────────────────
# STEP 11: Initialize ChromaDB
# ─────────────────────────────────────────────────────────────────
print_step "Initializing ChromaDB vector store..."
python3 - << 'PYEOF'
try:
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    # Create default collection to verify DB works
    col = client.get_or_create_collection("meeting_transcripts")
    print(f"  ChromaDB initialized at ./chroma_db")
    print(f"  Default collection 'meeting_transcripts' ready")
except Exception as e:
    print(f"  Warning: ChromaDB init issue: {e}")
PYEOF
print_ok "ChromaDB initialized"

# ─────────────────────────────────────────────────────────────────
# STEP 12: Validate installation
# ─────────────────────────────────────────────────────────────────
print_step "Validating installation..."
python3 - << 'PYEOF'
import sys
errors = []

modules = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("faster_whisper", "Faster Whisper"),
    ("gradio", "Gradio"),
    ("fastapi", "FastAPI"),
    ("chromadb", "ChromaDB"),
    ("cv2", "OpenCV"),
    ("fitz", "PyMuPDF"),
    ("sentence_transformers", "Sentence Transformers"),
    ("groq", "Groq"),
]

for module, name in modules:
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ✗ {name}: {e}")
        errors.append(name)

if errors:
    print(f"\n  WARNING: {len(errors)} package(s) missing: {', '.join(errors)}")
    print("  Run: pip install -r requirements.txt")
else:
    print("\n  All packages validated successfully!")

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA not available — CPU mode")
except Exception as e:
    print(f"  ⚠ CUDA check failed: {e}")
PYEOF
print_ok "Validation complete"

# ─────────────────────────────────────────────────────────────────
# FINAL: Print ready message
# ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Setup Complete!${NC}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}To start the application:${NC}"
echo ""
echo -e "  ${BOLD}Option 1: Gradio UI only (recommended)${NC}"
echo -e "    ${YELLOW}python app.py${NC}"
echo -e "    Open: http://localhost:7860"
echo ""
echo -e "  ${BOLD}Option 2: FastAPI backend only${NC}"
echo -e "    ${YELLOW}python main.py${NC}"
echo -e "    API Docs: http://localhost:8000/docs"
echo ""
echo -e "  ${BOLD}Option 3: Both simultaneously${NC}"
echo -e "    ${YELLOW}python main.py &${NC}"
echo -e "    ${YELLOW}python app.py${NC}"
echo ""
echo -e "  ${CYAN}If running on JarvisLabs:${NC}"
echo -e "    Forward port 7860 in the JarvisLabs dashboard"
echo -e "    Then open the provided URL in your browser"
echo ""
echo -e "  ${BOLD}${RED}Don't forget to add API keys to .env!${NC}"
echo ""
