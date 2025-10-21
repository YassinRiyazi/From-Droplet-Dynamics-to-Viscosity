"""
Author:
    - ChatGPT
Date:
    14-08-2025

Description:
    Real-time plotting utility (OpenGL preferred) ---------------------------
    Tries vispy (OpenGL) -> matplotlib (interactive) -> ASCII fallback

TODO:
    - [V] Change background color to black
    - Annotate the 3 lowest points in loss
    - clear plot on new epoch by pressing C
    - [V] make Y scale logarithmic
    - Plot GPT validation loss over time on top to compare
        https://arxiv.org/pdf/2210.11399
        https://arxiv.org/pdf/2206.14486
"""

from typing import Optional
from vispy import app, scene
import matplotlib.pyplot as plt

class RealTimePlotter:
    def __init__(self, title: str = "Training Loss (Real-Time)", prefer_opengl: bool = True):
        self.title = title
        self.prefer_opengl = prefer_opengl
        self.backend = None
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

        # Try vispy (OpenGL)
        self._vispy = None
        if self.prefer_opengl:
            try:
                self._vispy = {"app": app, "scene": scene}
                self._init_vispy()
                self.backend = "vispy"
                return
            except Exception:
                self._vispy = None

        # Try matplotlib
        self._mpl = None
        try:
            self._mpl = {"plt": plt}
            self._init_mpl()
            self.backend = "mpl"
            return
        except Exception:
            self._mpl = None

        # ASCII fallback
        self.backend = "ascii"
        self._last_ascii = ""

    # -------------------- vispy (OpenGL) --------------------
    def _init_vispy(self):
        scene = self._vispy["scene"]
        app = self._vispy["app"]

        self._canvas = scene.SceneCanvas(title=self.title, keys="interactive", show=True, size=(800, 400), bgcolor="white")
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = "panzoom"

        self._line_train = scene.Line(pos=[[0, 0], [1, 0]], color="blue", parent=self._view.scene, width=2)
        self._line_val   = scene.Line(pos=[[0, 0], [1, 0]], color="green", parent=self._view.scene, width=2)
        self._x_range = [0, 1]
        self._y_range = [0, 1]
        self._view.camera.set_range(x=self._x_range, y=self._y_range)

        # Keep GUI responsive
        app.use_app()  # ensure event loop exists

    def _vispy_update(self):
        import numpy as np
        if not self.epochs:
            return
        x = np.array(self.epochs, dtype=float)
        # normalize x to start at 1 for nicer axis
        # keep original values for axis range
        yt = np.array(self.train_losses, dtype=float)
        yv = np.array(self.val_losses, dtype=float) if self.val_losses else None

        # Update lines
        self._line_train.set_data(pos=np.column_stack([x, yt]))
        if yv is not None:
            self._line_val.set_data(pos=np.column_stack([x, yv]))

        # Update ranges with margins
        xmin, xmax = float(min(x)), float(max(x))
        ymin = float(min(yt.min(), yv.min() if yv is not None else yt.min()))
        ymax = float(max(yt.max(), yv.max() if yv is not None else yt.max()))
        if ymin == ymax:  # avoid zero height
            ymin -= 1e-6
            ymax += 1e-6
        self._x_range = [xmin, xmax]
        self._y_range = [ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)]
        self._view.camera.set_range(x=self._x_range, y=self._y_range)

        # Process one iteration of the vispy event loop
        self._vispy["app"].process_events()

    # -------------------- matplotlib --------------------
    def _init_mpl(self):
        plt = self._mpl["plt"]
        plt.style.use('dark_background')
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 4))
        self._fig.canvas.manager.set_window_title(self.title)
        self._line_train_mpl, = self._ax.plot([], [], label="Train", linewidth=2)
        self._line_val_mpl,   = self._ax.plot([], [], label="Val", linewidth=2)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")


        self._ax.set_yscale('log')


        self._ax.set_title(self.title)
        self._ax.legend(loc="best")
        self._fig.tight_layout()
        self._fig.show()

    def _mpl_update(self):
        plt = self._mpl["plt"]
        self._line_train_mpl.set_data(self.epochs, self.train_losses)
        if self.val_losses:
            self._line_val_mpl.set_data(self.epochs, self.val_losses)
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # -------------------- ASCII fallback --------------------
    def _ascii_update(self):
        # Simple last-line print: epoch and current values
        msg = f"[Plot] Epoch {self.epochs[-1]} | Train: {self.train_losses[-1]:.6f}"
        if self.val_losses:
            msg += f" | Val: {self.val_losses[-1]:.6f}"
        # Avoid spamming identical lines
        if msg != self._last_ascii:
            print(msg)
            self._last_ascii = msg

    # -------------------- Public API --------------------
    def update(self, epoch: int, train_loss: Optional[float] = None, val_loss: Optional[float] = None):
        if train_loss is not None:
            self.epochs.append(epoch)
            self.train_losses.append(float(train_loss))
            # pad val if missing to keep arrays aligned in some backends
            if val_loss is None and len(self.val_losses) < len(self.train_losses) - 1:
                self.val_losses.append(None)

        if val_loss is not None:
            # ensure epochs vector grows only once per epoch
            if len(self.epochs) < len(self.val_losses) + 1:
                self.epochs.append(epoch)
            self.val_losses.append(float(val_loss))

        # Update chosen backend
        if self.backend == "vispy":
            self._vispy_update()
        elif self.backend == "mpl":
            self._mpl_update()
        else:
            self._ascii_update()

    def close(self):
        if self.backend == "mpl" and self._mpl:
            try:
                self._mpl["plt"].ioff()
                self._mpl["plt"].close(self._fig)
            except Exception:
                pass
        # vispy window closes with process; no explicit close needed
