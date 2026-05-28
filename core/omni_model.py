"""
core/omni_model.py

Qwen2.5-Omni-7B multimodal inference engine.
Accepts text + images together for cross-modal reasoning.
Falls back to text-only Qwen2.5-7B or Groq if OOM / GPU unavailable.
"""

import gc
import os
import time
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GPU_MEM_THRESHOLD_GB = float(os.getenv("GPU_MEMORY_THRESHOLD", "20"))
HF_CACHE_DIR = os.getenv("HF_HOME", "./models/hf_cache")

# Groq fallback model
GROQ_MODEL = "llama-3.1-70b-versatile"

# System prompt for meeting Q&A
SYSTEM_PROMPT = """You are a Meeting Intelligence Assistant with access to:
1. A meeting transcript with timestamps
2. Presentation slides (as images)
3. Video frames from the meeting

Your job is to answer questions about the meeting with precise citations.
ALWAYS include:
- [Timestamp: MM:SS] when referencing spoken content
- [Slide: N] when referencing slide content
- [Speaker: Name] when attributing statements
- [Quote: "..."] for direct quotations from the transcript

Be concise, factual, and ground every claim in the provided evidence."""


# ─────────────────────────────────────────────────────────────────
# OmniModelResponse
# ─────────────────────────────────────────────────────────────────
class OmniModelResponse:
    """Container for model inference response."""

    def __init__(
        self,
        text: str,
        model_used: str,
        inference_time_s: float,
        used_images: bool = False,
        error: Optional[str] = None,
    ):
        """
        Initialize response container.

        Args:
            text: Generated answer text.
            model_used: Name of model that produced the answer.
            inference_time_s: Wall-clock inference time in seconds.
            used_images: Whether image inputs were processed.
            error: Error message if inference failed.
        """
        self.text = text
        self.model_used = model_used
        self.inference_time_s = inference_time_s
        self.used_images = used_images
        self.error = error

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "text": self.text,
            "model_used": self.model_used,
            "inference_time_s": round(self.inference_time_s, 2),
            "used_images": self.used_images,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────
# OmniModel
# ─────────────────────────────────────────────────────────────────
class OmniModel:
    """
    Qwen2.5-Omni-7B multimodal inference wrapper.

    Loads model in 8-bit quantization on GPU for memory efficiency.
    Automatic fallback hierarchy:
      1. Qwen2.5-Omni-7B (8-bit, GPU)
      2. Qwen2.5-Omni-7B (4-bit, GPU — if 8-bit OOM)
      3. Qwen2.5-7B-Instruct text-only (if multimodal fails)
      4. Groq API (if local model unavailable)
    """

    def __init__(self):
        """Initialize OmniModel — weights loaded lazily on first call."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_loaded_name: Optional[str] = None
        self.mode: str = "unloaded"  # unloaded | omni | text_only | groq
        self._groq_client = None
        logger.info("OmniModel init | lazy loading enabled")

    def _get_available_vram_gb(self) -> float:
        """Return available GPU VRAM in GB, or 0 if no GPU."""
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return free / 1e9

    def _check_transformers_version(self) -> bool:
        """Return True if transformers version supports Qwen2.5-Omni."""
        try:
            import transformers
            from packaging.version import Version
            if Version(transformers.__version__) < Version("4.51.0"):
                logger.error(
                    f"transformers {transformers.__version__} detected. "
                    "Qwen2.5-Omni requires transformers>=4.51.0. "
                    "Run: pip install 'transformers>=4.51.0'"
                )
                return False
            return True
        except Exception:
            return True  # packaging not available, try anyway

    def _load_omni_8bit(self) -> bool:
        """
        Attempt to load Qwen2.5-Omni in 8-bit quantization.

        Returns True if successful, False otherwise.
        """
        if not self._check_transformers_version():
            return False
        try:
            from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

            logger.info(f"Loading {MODEL_NAME} in 8-bit on GPU...")
            kwargs = dict(
                cache_dir=HF_CACHE_DIR,
                device_map="cuda",
                load_in_8bit=True,
                trust_remote_code=True,
            )
            if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
                kwargs["token"] = HF_TOKEN

            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_NAME, **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
                token=kwargs.get("token"),
            )
            self.model_loaded_name = MODEL_NAME
            self.mode = "omni"
            logger.info("Qwen2.5-Omni-7B loaded in 8-bit")
            return True

        except torch.cuda.OutOfMemoryError:
            logger.warning("8-bit OOM — trying 4-bit...")
            return False
        except Exception as e:
            logger.warning(f"Failed to load Omni 8-bit: {e}")
            return False

    def _load_omni_4bit(self) -> bool:
        """
        Attempt to load Qwen2.5-Omni in 4-bit quantization (NF4).

        Returns True if successful, False otherwise.
        """
        try:
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5OmniForConditionalGeneration,
            )

            logger.info(f"Loading {MODEL_NAME} in 4-bit NF4 on GPU...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs = dict(
                cache_dir=HF_CACHE_DIR,
                device_map="cuda",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
                kwargs["token"] = HF_TOKEN

            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_NAME, **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
                token=kwargs.get("token"),
            )
            self.model_loaded_name = MODEL_NAME
            self.mode = "omni"
            logger.info("Qwen2.5-Omni-7B loaded in 4-bit NF4")
            return True

        except torch.cuda.OutOfMemoryError:
            logger.warning("4-bit Omni also OOM — falling back to text-only")
            return False
        except Exception as e:
            logger.warning(f"Failed to load Omni 4-bit: {e}")
            return False

    def _load_text_only(self) -> bool:
        """
        Load Qwen2.5-7B-Instruct as text-only fallback.

        Returns True if successful, False otherwise.
        """
        text_model = "Qwen/Qwen2.5-7B-Instruct"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading text-only fallback: {text_model}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            kwargs = dict(
                cache_dir=HF_CACHE_DIR,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
                kwargs["token"] = HF_TOKEN

            self.tokenizer = AutoTokenizer.from_pretrained(text_model, **kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(text_model, **kwargs)
            self.model_loaded_name = text_model
            self.mode = "text_only"
            logger.info(f"{text_model} loaded in text-only mode")
            return True

        except Exception as e:
            logger.warning(f"Failed to load text-only fallback: {e}")
            return False

    def _load_groq(self) -> bool:
        """
        Initialize Groq API client as final fallback.

        Returns True if API key available, False otherwise.
        """
        if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_groq_key_here":
            logger.warning("No Groq API key — cannot use Groq fallback")
            return False
        try:
            from groq import Groq

            self._groq_client = Groq(api_key=GROQ_API_KEY)
            self.mode = "groq"
            self.model_loaded_name = GROQ_MODEL
            logger.info("Groq API client initialized")
            return True
        except Exception as e:
            logger.warning(f"Failed to init Groq client: {e}")
            return False

    def _ensure_loaded(self) -> None:
        """Load model through fallback hierarchy if not already loaded."""
        if self.mode != "unloaded":
            return

        available_vram = self._get_available_vram_gb()
        logger.info(f"Available VRAM: {available_vram:.1f} GB")

        if available_vram >= GPU_MEM_THRESHOLD_GB:
            if self._load_omni_8bit():
                return
            if self._load_omni_4bit():
                return
        elif available_vram > 0:
            logger.info("Low VRAM — trying 4-bit directly")
            if self._load_omni_4bit():
                return

        if self._load_text_only():
            return

        if self._load_groq():
            return

        logger.error("All model loading attempts failed!")
        self.mode = "unavailable"

    def generate(
        self,
        question: str,
        context_text: str,
        image_paths: Optional[list[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> OmniModelResponse:
        """
        Generate an answer to a meeting question with optional images.

        Args:
            question: User's question about the meeting.
            context_text: Relevant transcript/slide text retrieved from ChromaDB.
            image_paths: Optional list of slide/frame image paths to include.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            OmniModelResponse with generated answer and metadata.
        """
        self._ensure_loaded()

        if self.mode == "unavailable":
            return OmniModelResponse(
                text="Error: No model available. Please check GPU/API configuration.",
                model_used="none",
                inference_time_s=0.0,
                error="No model available",
            )

        # Filter image_paths to existing files
        valid_images: list[str] = []
        if image_paths:
            for p in image_paths:
                if p and Path(p).exists():
                    valid_images.append(p)

        t_start = time.time()

        if self.mode == "omni" and valid_images:
            response = self._generate_omni(
                question, context_text, valid_images, max_new_tokens, temperature
            )
        elif self.mode in ("omni", "text_only"):
            response = self._generate_text_only(
                question, context_text, max_new_tokens, temperature
            )
        elif self.mode == "groq":
            response = self._generate_groq(
                question, context_text, max_new_tokens
            )
        else:
            response = OmniModelResponse(
                text="Error: Model not loaded.",
                model_used="none",
                inference_time_s=0.0,
                error="Model not loaded",
            )

        elapsed = time.time() - t_start
        response.inference_time_s = elapsed
        response.used_images = bool(valid_images)
        return response

    def _generate_omni(
        self,
        question: str,
        context_text: str,
        image_paths: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> OmniModelResponse:
        """Run multimodal inference with Qwen2.5-Omni (text + images)."""
        try:
            # Load and resize images (max 1024px on longer side)
            images = []
            for path in image_paths[:3]:  # Limit to 3 images per query
                img = Image.open(path).convert("RGB")
                img = self._resize_image(img, max_size=1024)
                images.append(img)

            # Build message with images + context
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({
                "type": "text",
                "text": (
                    f"Meeting Context:\n{context_text}\n\n"
                    f"Question: {question}"
                ),
            })

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

            # Use processor to prepare inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            input_len = inputs.input_ids.shape[1]
            new_tokens = output_ids[0][input_len:]
            answer = self.processor.decode(new_tokens, skip_special_tokens=True)

            return OmniModelResponse(
                text=answer.strip(),
                model_used=self.model_loaded_name,
                inference_time_s=0.0,
            )

        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during Omni inference — falling back to text-only")
            self._clear_gpu_memory()
            return self._generate_text_only(
                question, context_text, max_new_tokens, temperature
            )
        except Exception as e:
            logger.error(f"Omni generation failed: {e}")
            return OmniModelResponse(
                text=f"Generation error: {str(e)[:200]}",
                model_used=self.model_loaded_name or "unknown",
                inference_time_s=0.0,
                error=str(e),
            )

    def _generate_text_only(
        self,
        question: str,
        context_text: str,
        max_new_tokens: int,
        temperature: float,
    ) -> OmniModelResponse:
        """Run text-only inference (Qwen or text fallback mode)."""
        try:
            if self.mode == "omni":
                # Use processor/tokenizer from Omni model in text-only mode
                tok = self.processor.tokenizer
                gen_model = self.model
                device = next(gen_model.parameters()).device
            else:
                tok = self.tokenizer
                gen_model = self.model
                device = next(gen_model.parameters()).device

            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
                f"Meeting Context:\n{context_text}\n\n"
                f"Question: {question}\n<|assistant|>\n"
            )

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3000)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = gen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tok.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_len:]
            answer = tok.decode(new_tokens, skip_special_tokens=True)

            return OmniModelResponse(
                text=answer.strip(),
                model_used=self.model_loaded_name,
                inference_time_s=0.0,
            )

        except torch.cuda.OutOfMemoryError:
            logger.warning("Text-only also OOM — switching to Groq")
            self._clear_gpu_memory()
            if self._load_groq():
                return self._generate_groq(question, context_text, max_new_tokens)
            return OmniModelResponse(
                text="GPU out of memory and no Groq key available.",
                model_used="none",
                inference_time_s=0.0,
                error="OOM",
            )
        except Exception as e:
            logger.error(f"Text-only generation failed: {e}")
            return OmniModelResponse(
                text=f"Generation error: {str(e)[:200]}",
                model_used=self.model_loaded_name or "unknown",
                inference_time_s=0.0,
                error=str(e),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _generate_groq(
        self, question: str, context_text: str, max_new_tokens: int
    ) -> OmniModelResponse:
        """Run inference via Groq API (text-only fallback)."""
        try:
            completion = self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Meeting Context:\n{context_text}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                max_tokens=max_new_tokens,
                temperature=0.1,
            )
            answer = completion.choices[0].message.content
            return OmniModelResponse(
                text=answer.strip(),
                model_used=f"groq/{GROQ_MODEL}",
                inference_time_s=0.0,
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return OmniModelResponse(
                text=f"Groq API error: {str(e)[:200]}",
                model_used=f"groq/{GROQ_MODEL}",
                inference_time_s=0.0,
                error=str(e),
            )

    @staticmethod
    def _resize_image(img: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image so longest side is at most max_size pixels."""
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        if w >= h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _clear_gpu_memory(self) -> None:
        """Free GPU memory after OOM event."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.mode = "unloaded"

    def unload(self) -> None:
        """Explicitly unload model and free memory."""
        self._clear_gpu_memory()
        logger.info("OmniModel unloaded")

    def get_status(self) -> dict:
        """Return current model status for health checks."""
        vram_free = self._get_available_vram_gb()
        return {
            "mode": self.mode,
            "model_loaded": self.model_loaded_name,
            "device": DEVICE,
            "vram_free_gb": round(vram_free, 2),
            "cuda_available": torch.cuda.is_available(),
        }


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_omni_instance: Optional[OmniModel] = None


def get_omni_model() -> OmniModel:
    """Return module-level singleton OmniModel."""
    global _omni_instance
    if _omni_instance is None:
        _omni_instance = OmniModel()
    return _omni_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = OmniModel()
    status = model.get_status()
    print(f"Status: {status}")

    context = (
        "[00:14:32] [Speaker 1] We've decided to increase pricing by 15% "
        "starting Q3. The slides show our revenue projections on slide 7."
    )
    question = "What was the final decision on pricing?"

    print(f"\nQuestion: {question}")
    print(f"Context:  {context[:100]}...")
    print("\nGenerating answer...")
    response = model.generate(question=question, context_text=context)
    print(f"\nAnswer [{response.model_used}] ({response.inference_time_s:.2f}s):")
    print(response.text)
