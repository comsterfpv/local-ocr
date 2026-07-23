"""GUI smoke tests for the Result panel and streaming.

These tests create a real ``LocalOCRApp`` window (withdrawn) so that
widget state can be inspected. They are skipped automatically when no
display is available (CI, headless containers).
"""

import unittest
from unittest import mock

import app as app_module


class TestResultPanelGui(unittest.TestCase):
    def setUp(self):
        # on_ocr_success / on_ocr_error pop modal message boxes that would
        # block the test run forever (no user to dismiss them). Patch the
        # whole messagebox module so these smoke tests can inspect widget
        # state without a real dialog.
        self._messagebox_patcher = mock.patch.object(app_module, "messagebox")
        self._messagebox_patcher.start()
        try:
            self.app = app_module.LocalOCRApp()
            self.app.withdraw()
        except Exception:
            self._messagebox_patcher.stop()
            self.skipTest("No display available — skipping GUI smoke tests.")

    def tearDown(self):
        try:
            self.app.destroy()
        except Exception:
            pass
        self._messagebox_patcher.stop()

    # -- helpers ----------------------------------------------------------

    def _page_text(self, page, total, text):
        self.app.on_page_text(
            {"page": page, "total": total, "text": text}
        )

    def _result_text(self):
        return self.app.result_box.get("1.0", "end-1c")

    # -- tests ------------------------------------------------------------

    def test_two_page_texts_produce_p1_separated_p2(self):
        """on_page_text × 2 → panel content 'p1\\n\\np2'."""
        self._page_text(1, 2, "p1")
        self._page_text(2, 2, "p2")
        self.assertEqual(self._result_text(), "p1\n\np2")

    def test_single_page_text(self):
        self._page_text(1, 1, "only page")
        self.assertEqual(self._result_text(), "only page")

    def test_ocr_success_switches_to_result_tab(self):
        self._page_text(1, 1, "hello")
        self.app.tabview.set("Log")
        with mock.patch.object(self.app, "_show_completion_dialog"):
            self.app.on_ocr_success("/tmp/fake.md")
        self.assertEqual(self.app.tabview.get(), "Result")

    def test_ocr_error_switches_to_log_tab(self):
        self.app.tabview.set("Result")
        self.app.on_ocr_error("something went wrong")
        self.assertEqual(self.app.tabview.get(), "Log")

    def test_apply_ocr_busy_state_clears_result_panel(self):
        """A new run clears the Result panel and resets to Log tab."""
        self._page_text(1, 1, "previous run text")
        self.assertEqual(self._result_text(), "previous run text")
        self.app.tabview.set("Result")
        self.app._apply_ocr_busy_state()
        self.assertEqual(self._result_text(), "")
        self.assertEqual(self.app.tabview.get(), "Log")

    def test_stream_chunk_appends_to_result(self):
        """on_stream_chunk buffers and appends text to the Result panel."""
        self.app.on_stream_chunk({"page": 1, "text": "Hel"})
        self.app.on_stream_chunk({"page": 1, "text": "lo"})
        # Force flush since .after() may not fire in a withdrawn window.
        self.app._flush_stream_buffer()
        self.assertEqual(self._result_text(), "Hello")

    def test_copy_result_puts_text_in_clipboard(self):
        """copy_result copies the Result panel content to the clipboard."""
        self._page_text(1, 1, "clipboard test")
        self.app.clipboard_clear()
        self.app.copy_result()
        self.assertEqual(self.app.clipboard_get(), "clipboard test")

    def test_page_text_after_streaming_does_not_duplicate(self):
        """The service streams the deltas, then emits page_text with the full
        assembled text. The panel must show that text once, not twice."""
        self.app.on_stream_chunk({"page": 1, "text": "Hello "})
        self.app.on_stream_chunk({"page": 1, "text": "world"})
        self.app.on_page_text({"page": 1, "total": 1, "text": "Hello world"})
        self.assertEqual(self._result_text(), "Hello world")

    def test_multi_page_streaming_inserts_single_separator(self):
        """Two streamed pages read as 'p1\\n\\np2' with the separator in the
        right place (between the pages, not after)."""
        self.app.on_stream_chunk({"page": 1, "text": "A"})
        self.app.on_page_text({"page": 1, "total": 2, "text": "A"})
        self.app.on_stream_chunk({"page": 2, "text": "B"})
        self.app.on_page_text({"page": 2, "total": 2, "text": "B"})
        self.app._flush_stream_buffer()
        self.assertEqual(self._result_text(), "A\n\nB")

    def test_page_text_without_stream_is_appended(self):
        """Fallback: a page that produced no stream chunks is still written
        (drives the panel exactly like the pre-streaming design)."""
        self.app.on_page_text({"page": 1, "total": 2, "text": "p1"})
        self.app.on_page_text({"page": 2, "total": 2, "text": "p2"})
        self.assertEqual(self._result_text(), "p1\n\np2")


if __name__ == "__main__":
    unittest.main()