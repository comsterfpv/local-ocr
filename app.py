"""LocalOCRApp: all widgets, state, and main-thread UI behavior.

Threading contract: workers never touch Tk. They only put plain-value
events on self.event_queue; drain_ui_events() runs on the Tk main thread
via .after() and performs every GUI update.
"""

from __future__ import annotations

import queue
import threading
from enum import Enum, auto
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

import config
import ocr_service
from ocr_service import OCRRequest


class OperationState(Enum):
    IDLE = auto()
    REFRESHING_MODELS = auto()
    PROCESSING_OCR = auto()


FILE_DIALOG_FILTERS = [
    ("Supported documents", "*.pdf *.png *.jpg *.jpeg *.webp"),
    ("PDF files", "*.pdf"),
    ("Images", "*.png *.jpg *.jpeg *.webp"),
    ("All files", "*.*"),
]

PADX = 12
PADY = 8


class LocalOCRApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local OCR")
        self.geometry("780x680")
        self.minsize(620, 520)

        self.operation_state = OperationState.IDLE
        self.closing = False
        self.selected_path: Path | None = None
        self.event_queue: queue.Queue = queue.Queue()
        self._render_phase_seen = False

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(config.UI_POLL_INTERVAL_MS, self.drain_ui_events)

    # ------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)  # the log absorbs resize space

        # File section
        file_frame = ctk.CTkFrame(self)
        file_frame.grid(row=0, column=0, sticky="ew", padx=PADX, pady=(PADY, 4))
        file_frame.grid_columnconfigure(1, weight=1)
        self.select_button = ctk.CTkButton(
            file_frame, text="Select File", command=self.select_file
        )
        self.select_button.grid(row=0, column=0, padx=PADX, pady=PADY)
        self.file_label = ctk.CTkLabel(file_frame, text="No file selected", anchor="w")
        self.file_label.grid(row=0, column=1, sticky="ew", padx=(0, PADX), pady=PADY)

        # Settings section
        settings = ctk.CTkFrame(self)
        settings.grid(row=1, column=0, sticky="ew", padx=PADX, pady=4)
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings, text="Ollama server URL").grid(
            row=0, column=0, sticky="w", padx=PADX, pady=(PADY, 4)
        )
        self.url_entry = ctk.CTkEntry(settings)
        self.url_entry.insert(0, config.DEFAULT_OLLAMA_URL)
        self.url_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(PADY, 4))
        self.refresh_button = ctk.CTkButton(
            settings, text="Refresh Models", width=130, command=self.refresh_models
        )
        self.refresh_button.grid(row=0, column=2, padx=(0, PADX), pady=(PADY, 4))

        ctk.CTkLabel(settings, text="Model").grid(
            row=1, column=0, sticky="w", padx=PADX, pady=4
        )
        self.model_combobox = ctk.CTkComboBox(
            settings, values=list(config.EXAMPLE_MODELS)
        )
        self.model_combobox.set("")  # suggestions are not installed models
        self.model_combobox.grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(0, PADX), pady=4
        )

        ctk.CTkLabel(settings, text="PDF DPI").grid(
            row=2, column=0, sticky="w", padx=PADX, pady=(4, PADY)
        )
        self.dpi_combobox = ctk.CTkComboBox(
            settings,
            values=[str(dpi) for dpi in config.DPI_OPTIONS],
            state="readonly",
            width=120,
        )
        self.dpi_combobox.set(str(config.DEFAULT_DPI))
        self.dpi_combobox.grid(row=2, column=1, sticky="w", pady=(4, PADY))

        # Action + feedback section
        self.start_button = ctk.CTkButton(
            self,
            text="Start OCR",
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_ocr,
        )
        self.start_button.grid(row=2, column=0, sticky="ew", padx=PADX, pady=4)

        # Status section: progress bar + page counter label
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=PADX, pady=4)
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(self.status_frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(PADX, 8), pady=PADY)
        self.status_label = ctk.CTkLabel(self.status_frame, text="", width=160)
        self.status_label.grid(row=0, column=1, padx=(0, PADX), pady=PADY)
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Courier New", size=12),
            state="disabled",
            wrap="word",
        )
        self.log_box.grid(row=4, column=0, sticky="nsew", padx=PADX, pady=(4, PADY))

    # ---------------------------------------------------- event plumbing

    def drain_ui_events(self) -> None:
        if self.closing:
            return  # discard queued UI work during shutdown
        try:
            while True:
                try:
                    kind, payload = self.event_queue.get_nowait()
                except queue.Empty:
                    break
                self.handle_event(kind, payload)
        finally:
            # Reschedule even if a handler raised; a broken .after() chain
            # would silently stop all event processing and leave the UI
            # stuck in its busy state.
            self.after(config.UI_POLL_INTERVAL_MS, self.drain_ui_events)

    def handle_event(self, kind: str, payload) -> None:
        if kind == "log":
            self.append_log(payload)
        elif kind == "models_loaded":
            self.on_models_loaded(payload)
        elif kind == "refresh_error":
            self.on_refresh_error(payload)
        elif kind == "ocr_success":
            self.on_ocr_success(payload)
        elif kind == "progress":
            self.on_progress(payload)
        elif kind == "ocr_error":
            self.on_ocr_error(payload)
        else:
            # Surface protocol mismatches (e.g. a new worker event kind
            # without a handler) instead of dropping them silently.
            self.append_log(f"[Warn] Unhandled event kind: {kind!r}")

    def append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ---------------------------------------------------- control states

    def _apply_refresh_busy_state(self) -> None:
        self.url_entry.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.start_button.configure(state="disabled")

    def _apply_ocr_busy_state(self) -> None:
        self.select_button.configure(state="disabled")
        self.url_entry.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.model_combobox.configure(state="disabled")
        self.dpi_combobox.configure(state="disabled")
        self.start_button.configure(
            state="disabled", text="Processing, please wait..."
        )
        self._render_phase_seen = False
        self.status_label.configure(text="")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

    def _restore_idle(self) -> None:
        self.select_button.configure(state="normal")
        self.url_entry.configure(state="normal")
        self.refresh_button.configure(state="normal")
        self.model_combobox.configure(state="normal")
        self.dpi_combobox.configure(state="readonly")
        self.start_button.configure(state="normal", text="Start OCR")
        self.progress.stop()
        self.progress.configure(mode="indeterminate")
        self.progress.set(0)
        self.status_label.configure(text="")
        self._render_phase_seen = False
        self.operation_state = OperationState.IDLE

    # ----------------------------------------------------- file selection

    def select_file(self) -> None:
        if self.operation_state is not OperationState.IDLE:
            return
        filename = filedialog.askopenfilename(
            title="Select a PDF or image",
            filetypes=FILE_DIALOG_FILTERS,
            parent=self,
        )
        if not filename:
            return
        path = Path(filename)
        try:
            ocr_service.validate_input_path(path)
        except ValueError as exc:
            # Keep the previous valid selection.
            messagebox.showwarning("Unsupported file", str(exc), parent=self)
            return
        self.selected_path = path
        self.file_label.configure(text=path.name)

    # ------------------------------------------------------ model refresh

    def refresh_models(self) -> None:
        if self.operation_state is not OperationState.IDLE:
            return
        try:
            url = ocr_service.normalize_ollama_url(self.url_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid URL", str(exc), parent=self)
            return
        self.operation_state = OperationState.REFRESHING_MODELS
        self._apply_refresh_busy_state()
        self.append_log(f"Refreshing model list from {url}...")
        threading.Thread(
            target=self._refresh_worker, args=(url,), daemon=True
        ).start()

    def _refresh_worker(self, url: str) -> None:
        try:
            models = ocr_service.list_models(url)
        except Exception as exc:
            self.event_queue.put(("refresh_error", str(exc)))
        else:
            self.event_queue.put(("models_loaded", models))

    def on_progress(self, payload: dict) -> None:
        phase = payload["phase"]
        current = payload["current"]
        total = payload["total"]

        if phase == "render":
            self._render_phase_seen = True
            fraction = 0.2 * current / total
        else:  # "ocr"
            if self._render_phase_seen:
                fraction = 0.2 + 0.8 * current / total
            else:
                fraction = current / total

        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(fraction)

        phase_label = "OCR" if phase == "ocr" else "Render"
        self.status_label.configure(text=f"Page {current} / {total} ({phase_label})")

    def on_models_loaded(self, models: list[str]) -> None:
        self._restore_idle()
        if not models:
            self.append_log(
                "No models found on the server; enter a model tag manually."
            )
            return
        typed = self.model_combobox.get().strip()
        self.model_combobox.configure(values=models)
        self.model_combobox.set(typed if typed else models[0])
        self.append_log(f"Found {len(models)} model(s).")

    def on_refresh_error(self, message: str) -> None:
        self._restore_idle()
        self.append_log(f"[Error] {message}")
        messagebox.showerror("Model refresh failed", message, parent=self)

    # -------------------------------------------------------------- OCR

    def start_ocr(self) -> None:
        if self.operation_state is not OperationState.IDLE:
            return
        # Snapshot and validate every input on the main thread.
        if self.selected_path is None:
            messagebox.showerror(
                "No file", "Select a PDF or image file first.", parent=self
            )
            return
        input_path = self.selected_path
        try:
            ocr_service.validate_input_path(input_path)
        except ValueError as exc:
            messagebox.showerror("Invalid file", str(exc), parent=self)
            return
        try:
            url = ocr_service.normalize_ollama_url(self.url_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid URL", str(exc), parent=self)
            return
        model = self.model_combobox.get().strip()
        if not model:
            messagebox.showerror(
                "No model", "Enter or select an Ollama model tag.", parent=self
            )
            return
        try:
            dpi = int(self.dpi_combobox.get())
        except ValueError:
            dpi = -1
        if dpi not in config.DPI_OPTIONS:
            options = ", ".join(str(d) for d in config.DPI_OPTIONS)
            messagebox.showerror(
                "Invalid DPI", f"DPI must be one of: {options}", parent=self
            )
            return

        output_path = ocr_service.build_output_path(input_path)
        if output_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite existing file?",
                f"{output_path.name} already exists in the same folder.\n"
                "Overwrite it?",
                parent=self,
            )
            if not overwrite:
                return

        request = OCRRequest(
            input_path=input_path,
            output_path=output_path,
            ollama_url=url,
            model=model,
            dpi=dpi,
        )
        self.operation_state = OperationState.PROCESSING_OCR
        self._apply_ocr_busy_state()
        self.append_log(f"[Start] Input: {input_path}")
        self.append_log(f"[Start] Ollama: {url} | Model: {model}")
        threading.Thread(
            target=self._ocr_worker, args=(request,), daemon=True
        ).start()

    def _ocr_worker(self, request: OCRRequest) -> None:
        saved_path = None
        error: Exception | None = None
        try:
            saved_path = ocr_service.process_ocr(request, self.event_queue)
        except Exception as exc:
            error = exc
        # process_ocr's finally has already cleaned up temporary files;
        # now enqueue exactly one terminal event.
        if error is not None:
            self.event_queue.put(("ocr_error", str(error)))
        else:
            self.event_queue.put(("ocr_success", str(saved_path)))

    def on_ocr_success(self, saved_path: str) -> None:
        self._restore_idle()
        self.append_log(f"[Success] File saved: {saved_path}")
        messagebox.showinfo(
            "OCR complete", f"Markdown saved to:\n{saved_path}", parent=self
        )

    def on_ocr_error(self, message: str) -> None:
        self._restore_idle()
        self.append_log(f"[Error] {message}")
        messagebox.showerror("OCR failed", message, parent=self)

    # ---------------------------------------------------------- shutdown

    def on_close(self) -> None:
        if self.closing or self.operation_state is OperationState.IDLE:
            self.closing = True
            self.destroy()
            return
        if messagebox.askyesno(
            "Quit", "An operation is still running. Close anyway?", parent=self
        ):
            self.closing = True
            self.destroy()
