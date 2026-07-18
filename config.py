"""Configuration constants for the Local OCR application."""

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Suggestions only. These tags are never assumed to exist on the server and
# are never pulled automatically.
EXAMPLE_MODELS = ["gemma4:12b", "qwen3.6:27b"]

DPI_OPTIONS = [100, 150, 200, 300]
DEFAULT_DPI = 150

PDF_EXTENSIONS = frozenset({".pdf"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

# Seconds. Model listing should fail fast; OCR of a large page can be slow.
MODEL_LIST_TIMEOUT = 10
OCR_TIMEOUT = 600

# Milliseconds between main-thread drains of the worker event queue.
UI_POLL_INTERVAL_MS = 50

SYSTEM_PROMPT = (
    "Convert this image into Markdown text format. Your task is to perform "
    "high-accuracy Optical Character Recognition (OCR). Preserve the "
    "document's structure as accurately as possible: headers, lists, and "
    "tables. Do not add any greetings, explanations, or "
    "introductory/concluding remarks. Output only the raw recognized text."
)

USER_PROMPT = "Recognize this document page."
