"""Entry point for the Local OCR application."""

import customtkinter as ctk

from app import LocalOCRApp


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = LocalOCRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
