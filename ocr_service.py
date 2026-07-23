"""Service layer for Local OCR: validation, PDF rendering, Ollama, saving.

This module must stay free of Tk imports so every function can be tested
headlessly and no worker can accidentally touch the GUI.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import ollama
import pymupdf

import config

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int], None]  # phase, current, total


class OCRServiceError(Exception):
    """A service operation (Ollama, PDF rendering, saving) failed."""


@dataclass(frozen=True)
class OCRRequest:
    """Immutable snapshot of everything an OCR worker needs."""

    input_path: Path
    output_path: Path
    ollama_url: str
    model: str
    dpi: int


def normalize_ollama_url(value: str) -> str:
    """Validate a user-entered Ollama base URL and return it normalized.

    Keeps any path prefix so reverse-proxy URLs work; never appends /api
    because the official client handles API paths itself.
    """
    url = value.strip().rstrip("/")
    if not url:
        raise ValueError("Ollama server URL is empty.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "Ollama server URL must start with http:// or https:// "
            f"(got: {value.strip()!r})."
        )
    if not parsed.netloc:
        raise ValueError(f"Ollama server URL has no host: {value.strip()!r}.")
    return url


def validate_input_path(path: Path) -> None:
    """Raise ValueError unless path is a readable, supported document."""
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. Supported: {supported}"
        )
    if not os.access(path, os.R_OK):
        raise ValueError(f"File is not readable: {path}")


def build_output_path(input_path: Path) -> Path:
    """Return /dir/document_extracted.md for /dir/document.<ext>."""
    return input_path.with_name(f"{input_path.stem}_extracted.md")


def list_models(url: str) -> list[str]:
    """Fetch model tags from an Ollama server, deduplicated and sorted."""
    try:
        client = ollama.Client(host=url, timeout=config.MODEL_LIST_TIMEOUT)
        response = client.list()
        tags = {
            (getattr(item, "model", None) or "").strip()
            for item in response.models
        }
    except Exception as exc:
        raise OCRServiceError(f"Could not fetch models from {url}: {exc}") from exc
    tags.discard("")
    return sorted(tags, key=str.lower)


def render_pdf(
    pdf_path: Path,
    dpi: int,
    temp_dir: Path,
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Render every PDF page to a PNG in temp_dir, in original page order."""
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise OCRServiceError(f"Could not open PDF: {exc}") from exc
    with document:
        if document.needs_pass:
            raise OCRServiceError(
                "PDF is password-protected; encrypted documents are not supported."
            )
        page_count = document.page_count
        if page_count == 0:
            raise OCRServiceError("PDF contains no pages.")
        image_paths: list[Path] = []
        for index in range(page_count):
            page_number = index + 1
            if progress_callback is not None:
                progress_callback("render", page_number, page_count)
            log_callback(f"Rendering page {page_number}/{page_count}...")
            try:
                page = document.load_page(index)
                pixmap = page.get_pixmap(
                    dpi=dpi, colorspace=pymupdf.csRGB, alpha=False
                )
                image_path = temp_dir / f"page_{page_number:04d}.png"
                pixmap.save(str(image_path))
            except Exception as exc:
                raise OCRServiceError(
                    f"Failed to render page {page_number}/{page_count}: {exc}"
                ) from exc
            image_paths.append(image_path)
    return image_paths


def recognize_images(
    client: "ollama.Client",
    model: str,
    image_paths: list[Path],
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    """Send one independent chat request per image; return texts in order."""
    total = len(image_paths)
    results: list[str] = []
    for number, image_path in enumerate(image_paths, start=1):
        if progress_callback is not None:
            progress_callback("ocr", number, total)
        log_callback(f"Sending page {number}/{total} to Ollama...")
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": config.USER_PROMPT,
                        "images": [str(image_path)],
                    },
                ],
            )
        except Exception as exc:
            raise OCRServiceError(
                f"Ollama request failed on page {number}/{total} "
                f"(model {model!r}): {exc}"
            ) from exc
        content = (response.message.content or "").strip()
        if not content:
            raise OCRServiceError(
                f"Ollama returned no text for page {number}/{total} "
                f"(model {model!r})."
            )
        results.append(content)
    return results


def save_markdown_atomic(output_path: Path, content: str) -> None:
    """Write content as UTF-8 with \\n newlines, then publish atomically."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(normalized)
            tmp_file.flush()
        os.replace(tmp_path, output_path)
    except Exception as exc:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise OCRServiceError(f"Could not save output file: {exc}") from exc


def process_ocr(request: OCRRequest, event_queue) -> Path:
    """Run the full OCR pipeline; emit ('log', message) events; return output.

    Raises on any failure. The temporary render directory is always removed
    in the one outer finally, on success and on every failure path. The
    caller (worker wrapper) enqueues the single terminal success/error event
    after this function has returned or raised, so cleanup always precedes
    the terminal event.
    """

    def log(message: str) -> None:
        event_queue.put(("log", message))

    def progress(phase: str, current: int, total: int) -> None:
        event_queue.put(("progress", {"phase": phase, "current": current, "total": total}))

    temp_dir: Path | None = None
    try:
        if request.input_path.suffix.lower() in config.PDF_EXTENSIONS:
            log("[1/3] Preparing document...")
            try:
                temp_dir = Path(tempfile.mkdtemp(prefix="local_ocr_"))
            except Exception as exc:
                raise OCRServiceError(
                    f"Could not create temporary render directory: {exc}"
                ) from exc
            image_paths = render_pdf(
                request.input_path,
                request.dpi,
                temp_dir,
                lambda message: log(f"[1/3] {message}"),
                progress,
            )
        else:
            log("[1/3] Preparing image...")
            image_paths = [request.input_path]

        try:
            client = ollama.Client(host=request.ollama_url, timeout=config.OCR_TIMEOUT)
        except Exception as exc:
            raise OCRServiceError(
                f"Could not create Ollama client for {request.ollama_url}: {exc}"
            ) from exc

        page_texts = recognize_images(
            client,
            request.model,
            image_paths,
            lambda message: log(f"[2/3] {message}"),
            progress,
        )

        log("[3/3] Saving Markdown...")
        save_markdown_atomic(request.output_path, "\n\n".join(page_texts))
        return request.output_path
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
