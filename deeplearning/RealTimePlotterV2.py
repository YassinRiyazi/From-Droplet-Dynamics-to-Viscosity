from __future__ import annotations

"""
Author:
    - Yassin Riyazi
Date:
    14-08-2025

Description:
    Real-time plotting utility (OpenGL preferred) ---------------------------
    Tries vispy (OpenGL) -> matplotlib (interactive) -> ASCII fallback

TODO:
    - [V] Change background color to black
    - [V] make Y scale logarithmic
    - [V] Annotate the 3 lowest points in loss and update it accordingly
    - [V] clear plot on new epoch by pressing C
    - [V] Add option to save the plot as an image file
    - [V] add grid lines
    - Plot GPT validation loss over time on top to compare
        https://arxiv.org/pdf/2210.11399
        https://arxiv.org/pdf/2206.14486
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

# import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from vispy import app, scene
from PIL import Image
import cv2
# from skimage.io import imsave


class RealTimePlotter:
    """Utility for real-time visualization of training and validation losses."""

    def __init__(self, title: str = "Training Loss (Real-Time)", prefer_opengl: bool = True) -> None:
        self.title: str = title
        self.prefer_opengl: bool = prefer_opengl
        self.backend: Optional[str] = None

        self.train_epochs: List[float] = []
        self.train_losses: List[float] = []
        self.val_epochs: List[float] = []
        self.val_losses: List[float] = []

        self._train_annotations: List[Any] = []
        self._val_annotations: List[Any] = []
        self._last_ascii: str = ""
        self._grid: Any = None

        self._vispy: Optional[Dict[str, Any]] = None
        self._canvas: Any = None
        self._view: Any = None
        self._line_train: Any = None
        self._line_val: Any = None

        self._mpl: Optional[Dict[str, Any]] = None
        self._fig: Any = None
        self._ax: Any = None
        self._line_train_mpl: Any = None
        self._line_val_mpl: Any = None

        if self.prefer_opengl:
            try:
                self._vispy = {"app": app, "scene": scene}
                self._init_vispy()
                self.backend = "vispy"
                return
            except Exception:
                self._vispy = None

        try:
            self._mpl = {"plt": plt}
            self._init_mpl()
            self.backend = "mpl"
            return
        except Exception:
            self._mpl = None

        self.backend = "ascii"

    # -------------------- vispy (OpenGL) --------------------
    def _init_vispy(self) -> None:
        assert self._vispy is not None
        scene_mod = self._vispy["scene"]
        app_mod = self._vispy["app"]

        self._canvas = scene_mod.SceneCanvas(
            title=self.title,
            keys="interactive",
            show=True,
            size=(800, 400),
            bgcolor="black",
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = "panzoom"

        self._line_train = scene_mod.Line(
            pos=[[0, 0], [1, 0]], color="#1f77b4", parent=self._view.scene, width=2
        )
        self._line_val = scene_mod.Line(
            pos=[[0, 0], [1, 0]], color="#2ca02c", parent=self._view.scene, width=2
        )
        self._grid = scene_mod.GridLines(color=(0.4, 0.4, 0.4, 0.3), parent=self._view.scene)

        self._canvas.events.key_press.connect(self._vispy_on_key_press)
        app_mod.use_app()

    def _vispy_update(self) -> None:
        if self._vispy is None:
            return

        if not (self.train_epochs or self.val_epochs):
            return

        if self.train_epochs:
            train_pos = np.column_stack(
                [self.train_epochs, self._log_scale(self.train_losses)]
            )
            if train_pos.shape[0] == 1:
                train_pos = np.vstack([train_pos, train_pos])
            self._line_train.set_data(pos=train_pos)
        if self.val_epochs:
            val_pos = np.column_stack(
                [self.val_epochs, self._log_scale(self.val_losses)]
            )
            if val_pos.shape[0] == 1:
                val_pos = np.vstack([val_pos, val_pos])
            self._line_val.set_data(pos=val_pos)

        x_values = np.array(self.train_epochs + self.val_epochs, dtype=float)
        y_values = np.array(
            self._log_scale(self.train_losses + self.val_losses), dtype=float
        )

        if not len(x_values):
            x_values = np.array([0.0, 1.0])
        if not len(y_values):
            y_values = np.array([0.0, 1.0])

        xmin, xmax = float(x_values.min()), float(x_values.max())
        ymin, ymax = float(y_values.min()), float(y_values.max())
        if xmin == xmax:
            pad = 1.0 if xmin == 0 else abs(xmin) * 0.1 or 1.0
            xmin -= pad
            xmax += pad
        if ymin == ymax:
            ymin -= 1e-6
            ymax += 1e-6

        self._view.camera.set_range(
            x=(xmin, xmax),
            y=(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)),
        )

        self._vispy_update_annotations()
        self._vispy["app"].process_events()

    def _vispy_on_key_press(self, event: Any) -> None:
        if event.text and event.text.lower() == "c":
            self.clear()

    def _vispy_update_annotations(self) -> None:
        if self._vispy is None or self._view is None:
            return

        for item in self._train_annotations:
            item.parent = None
        for item in self._val_annotations:
            item.parent = None
        self._train_annotations = []
        self._val_annotations = []

        for epoch, loss in self._lowest_points(self.train_epochs, self.train_losses):
            log_loss = self._log_scale([loss])[0]
            text = scene.Text(
                f"Train {loss:.4f}",
                color="#ff7f0e",
                parent=self._view.scene,
                font_size=10,
                pos=(epoch, log_loss),
                anchor_x="left",
                anchor_y="bottom",
            )
            self._train_annotations.append(text)

        for epoch, loss in self._lowest_points(self.val_epochs, self.val_losses):
            log_loss = self._log_scale([loss])[0]
            text = scene.Text(
                f"Val {loss:.4f}",
                color="red",
                parent=self._view.scene,
                font_size=10,
                pos=(epoch, log_loss),
                anchor_x="left",
                anchor_y="top",
            )
            self._val_annotations.append(text)

    # -------------------- matplotlib --------------------
    def _init_mpl(self) -> None:
        assert self._mpl is not None
        plt_mod = self._mpl["plt"]
        plt_mod.style.use("dark_background")
        plt_mod.ion()

        self._fig, self._ax = plt_mod.subplots(figsize=(8, 4))
        manager = getattr(self._fig.canvas, "manager", None)
        if manager is not None:
            set_title = getattr(manager, "set_window_title", None)
            if callable(set_title):
                set_title(self.title)
        self._line_train_mpl, = self._ax.plot([], [], label="Train", linewidth=2, color="#1f77b4")
        self._line_val_mpl, = self._ax.plot([], [], label="Val", linewidth=2, color="#2ca02c")
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")
        self._ax.set_yscale("log")
        self._ax.set_title(self.title)
        self._ax.legend(loc="best")
        self._ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        self._fig.tight_layout()
        self._fig.show()
        self._fig.canvas.mpl_connect("key_press_event", self._mpl_on_key_press)

    def _mpl_on_key_press(self, event: Any) -> None:
        if event.key and event.key.lower() == "c":
            self.clear()

    def _mpl_update(self) -> None:
        if self._mpl is None or self._fig is None or self._ax is None:
            return
        plt_mod = self._mpl["plt"]
        self._line_train_mpl.set_data(self.train_epochs, self.train_losses)
        self._line_val_mpl.set_data(self.val_epochs, self.val_losses)
        self._ax.relim()
        self._ax.autoscale_view()
        self._update_mpl_annotations()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt_mod.pause(0.001)

    def _update_mpl_annotations(self) -> None:
        for ann in self._train_annotations:
            if hasattr(ann, "remove"):
                ann.remove()
        for ann in self._val_annotations:
            if hasattr(ann, "remove"):
                ann.remove()
        self._train_annotations = []
        self._val_annotations = []

        for epoch, loss in self._lowest_points(self.train_epochs, self.train_losses):
            ann = self._ax.annotate(
                f"Train {loss:.4f}",
                xy=(epoch, loss),
                xytext=(0, 8),
                textcoords="offset points",
                color="#ff7f0e",
                fontsize=8,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
            )
            self._train_annotations.append(ann)

        for epoch, loss in self._lowest_points(self.val_epochs, self.val_losses):
            ann = self._ax.annotate(
                f"Val {loss:.4f}",
                xy=(epoch, loss),
                xytext=(0, -12),
                textcoords="offset points",
                color="#17becf",
                fontsize=8,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#17becf"),
            )
            self._val_annotations.append(ann)

    # -------------------- ASCII fallback --------------------
    def _ascii_update(self) -> None:
        message_parts = []
        if self.train_epochs:
            message_parts.append(f"Epoch {self.train_epochs[-1]} | Train: {self.train_losses[-1]:.6f}")
        if self.val_epochs:
            message_parts.append(f"Val: {self.val_losses[-1]:.6f}")
        msg = "[Plot] " + " | ".join(message_parts) if message_parts else "[Plot]"
        if msg != self._last_ascii:
            print(msg)
            self._last_ascii = msg

    # -------------------- Public API --------------------
    def update(
        self,
        epoch: float,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        if train_loss is not None:
            self.train_epochs.append(float(epoch))
            self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_epochs.append(float(epoch))
            self.val_losses.append(float(val_loss))

        try:
            if self.backend == "vispy":
                self._vispy_update()
            elif self.backend == "mpl":
                self._mpl_update()
            else:
                self._ascii_update()
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._handle_backend_failure(exc)
            self._ascii_update()

    def clear(self) -> None:
        self.train_epochs = []
        self.train_losses = []
        self.val_epochs = []
        self.val_losses = []
        self._train_annotations = []
        self._val_annotations = []
        self._last_ascii = ""

        if self.backend == "vispy" and self._line_train is not None and self._vispy is not None:
            zero_line = np.array([[0.0, 0.0]])
            self._line_train.set_data(pos=zero_line)
            self._line_val.set_data(pos=zero_line)
            self._vispy["app"].process_events()
        elif self.backend == "mpl" and self._line_train_mpl is not None:
            self._line_train_mpl.set_data([], [])
            self._line_val_mpl.set_data([], [])
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def save(self, file_path: str) -> None:
        if self.backend == "vispy" and self._canvas is not None:
            try:
                image = self._canvas.render()

                Image.fromarray(image).save(file_path)
            except Exception as exc:
                print(f"[WARNING] Failed to save vispy canvas: {exc}")
        elif self.backend == "mpl" and self._fig is not None:
            try:
                self._fig.savefig(file_path, dpi=300, bbox_inches="tight")
            except Exception as exc:
                print(f"[WARNING] Failed to save matplotlib figure: {exc}")
        else:
            try:
                with open(file_path, "w", encoding="utf-8") as handle:
                    for epoch, loss in zip(self.train_epochs, self.train_losses):
                        handle.write(f"train,{epoch},{loss}\n")
                    for epoch, loss in zip(self.val_epochs, self.val_losses):
                        handle.write(f"val,{epoch},{loss}\n")
            except OSError as exc:
                print(f"[WARNING] Failed to save ASCII plot data: {exc}")

    def close(self) -> None:
        if self.backend == "mpl" and self._mpl:
            try:
                plt_mod = self._mpl["plt"]
                plt_mod.ioff()
                plt_mod.close(self._fig)
            except Exception:
                pass

    @staticmethod
    def _lowest_points(
        epochs: Sequence[float], losses: Sequence[float], count: int = 3
    ) -> List[Tuple[float, float]]:
        pairs = [(epoch, loss) for epoch, loss in zip(epochs, losses)]
        pairs.sort(key=lambda item: item[1])
        return pairs[:count]

    @staticmethod
    def _log_scale(values: Sequence[float]) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        arr = np.clip(arr, 1e-12, None)
        return np.log10(arr)

    def _handle_backend_failure(self, exc: Exception) -> None:
        backend_name = self.backend or "unknown"
        print(
            f"[WARNING] RealTimePlotter backend '{backend_name}' failed: {exc}. "
            "Falling back to ASCII output."
        )
        if self.backend == "vispy" and self._canvas is not None:
            try:
                self._canvas.close()
            except Exception:
                pass
        if self.backend == "mpl" and self._mpl is not None:
            try:
                plt_mod = self._mpl.get("plt")
                if plt_mod is not None:
                    plt_mod.close(self._fig)
            except Exception:
                pass
        self.backend = "ascii"
