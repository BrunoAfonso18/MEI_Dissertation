"""
Interactive ABSA sentiment smiley - desktop app.

Real-time visual demo of this project's own aspect-based sentiment pipeline
(app/backend/models/extractor.py + app/backend/models/fuzzy_sentiment.py),
running on the fine-tuned polarity model in
app/absa_module/absa_model_final_polarity.

Type a restaurant review and the smiley reacts to the aggregated, fuzzy-logic
sentiment score, while the panel below lists the aspects that were detected.

Run with:
    python app/frontend/app.py
"""

import queue
import threading
import time

import customtkinter
from PIL import Image

from absa_pipeline import AbsaSentimentPipeline, SentimentResult
from smiley import render_smiley

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

DEBOUNCE_SECONDS = 0.3
PLACEHOLDER_TEXT = "A comida estava deliciosa mas o serviço foi muito lento."


class SmileyDisplay(customtkinter.CTkLabel):
    """CTkLabel that renders BGRA numpy frames from render_smiley() as an image."""

    def __init__(self, master, size: tuple[int, int] = (380, 380)) -> None:
        super().__init__(master, text="", image=None, width=size[0], height=size[1])
        self.size = size
        self._image_ref: customtkinter.CTkImage | None = None

    def show(self, frame) -> None:
        rgba = frame[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
        pil_image = Image.fromarray(rgba, mode="RGBA")
        ctk_image = customtkinter.CTkImage(light_image=pil_image, dark_image=pil_image, size=self.size)
        self._image_ref = ctk_image  # keep a reference so tkinter doesn't garbage-collect it
        self.configure(image=ctk_image)


class App(customtkinter.CTk):
    def __init__(self, pipeline: AbsaSentimentPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline

        self.title("ABSA Sentiment Smiley")
        self.geometry("640x840")
        self.minsize(520, 700)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        customtkinter.CTkLabel(
            self,
            text="Escreve uma review de restaurante:",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        self.text_input = customtkinter.CTkTextbox(self, height=100, font=("Segoe UI", 14), wrap="word")
        self.text_input.insert("1.0", PLACEHOLDER_TEXT)
        self.text_input.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.text_input.bind("<KeyRelease>", lambda _event: self._on_text_changed())

        self.smiley_display = SmileyDisplay(self)
        self.smiley_display.grid(row=2, column=0, padx=20, pady=10)

        self.status_var = customtkinter.StringVar(value="A carregar o modelo...")
        customtkinter.CTkLabel(
            self,
            textvariable=self.status_var,
            font=("Segoe UI", 15, "bold"),
        ).grid(row=3, column=0, pady=(0, 5))

        self.aspects_box = customtkinter.CTkTextbox(self, font=("Consolas", 12))
        self.aspects_box.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.aspects_box.configure(state="disabled")

        self.smiley_display.show(render_smiley(0.0, self.smiley_display.size))

        self._pending_text: str | None = None
        self._lock = threading.Lock()
        self._result_queue: "queue.Queue[SentimentResult]" = queue.Queue()
        self._stop_event = threading.Event()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_results)
        self._on_text_changed()

    def _on_text_changed(self) -> None:
        text = self.text_input.get("1.0", "end").strip()
        with self._lock:
            self._pending_text = text

    def _worker_loop(self) -> None:
        last_processed = None
        while not self._stop_event.is_set():
            with self._lock:
                text = self._pending_text
            if text is not None and text != last_processed:
                last_processed = text
                result = self.pipeline.run(text)
                self._result_queue.put(result)
            time.sleep(DEBOUNCE_SECONDS)

    def _poll_results(self) -> None:
        try:
            while True:
                result = self._result_queue.get_nowait()
                self._apply_result(result)
        except queue.Empty:
            pass
        self.after(50, self._poll_results)

    def _apply_result(self, result: SentimentResult) -> None:
        frame = render_smiley(result.positivity, self.smiley_display.size)
        self.smiley_display.show(frame)

        if result.aspects:
            self.status_var.set(f"Sentimento geral: {result.positivity:+.2f}  ({result.label})")
        else:
            self.status_var.set("Nenhum aspeto detetado - smiley neutro")

        self.aspects_box.configure(state="normal")
        self.aspects_box.delete("1.0", "end")
        for aspect in result.aspects:
            line = f"[{aspect.polarity:<8}] {aspect.term:<25} confiança={aspect.confidence:.2f}\n"
            self.aspects_box.insert("end", line)
        self.aspects_box.configure(state="disabled")

    def _on_close(self) -> None:
        self._stop_event.set()
        self.destroy()


def main() -> None:
    pipeline = AbsaSentimentPipeline()
    app = App(pipeline)
    app.mainloop()


if __name__ == "__main__":
    main()
