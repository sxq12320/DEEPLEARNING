"""Generate an Escher-style recursive logarithmic-spiral image.

The effect is created by geometric remapping, not alpha compositing.  In the
default ``spiral`` mode, a change of distance by ``scale_factor`` is one
logarithmic period and the image rotates continuously through that period.
The optional ``conformal`` mode uses a complex-power approximation.
The ``logtile`` mode follows the 3Blue1Brown construction most directly and is
the one that actually produces a seamless *rotating* loop from an ordinary photo:
the photo is first **rectified** into a log-polar tile (as in 3b1b's
``CreatePiHouseLog``), which makes the angular direction periodic, and only then
is ``log(z)`` mapped through that tile and returned to the image plane with ``exp``.

The ``nested`` mode is a local Droste transform. It keeps the outer part of
an ordinary photo unchanged and applies the recursive mapping only inside a
central window. This is the appropriate starting point for a normal photo;
the 3Blue1Brown/Escher result is based on a specially rectified source image,
not an arbitrary square photo used as a global periodic texture.

An arbitrary photo can produce a recursive spiral texture.  An exact
self-contained version of Escher's "Print Gallery" requires a source image
whose inner and outer regions were designed to match.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image


def _square_image(image: Image.Image, size: int) -> np.ndarray:
    """Center-crop an image to a square and resize it to ``size``."""
    image = image.convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample an RGB image at floating-point pixel coordinates."""
    height, width, _ = image.shape
    x = np.clip(x, 0.0, width - 1.001)
    y = np.clip(y, 0.0, height - 1.001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    top = image[y0, x0] * (1.0 - dx) + image[y0, x1] * dx
    bottom = image[y1, x0] * (1.0 - dx) + image[y1, x1] * dx
    return top * (1.0 - dy) + bottom * dy


def _sample_complex_image(source: np.ndarray, z_source: np.ndarray) -> np.ndarray:
    """Sample a square RGB image using normalized complex coordinates."""
    source_size = source.shape[0]
    pixel_x = (z_source.real + 1.0) * 0.5 * (source_size - 1)
    pixel_y = (-z_source.imag + 1.0) * 0.5 * (source_size - 1)
    return _bilinear_sample(source, pixel_x, pixel_y)


def _log_spiral_map(
    z: np.ndarray,
    scale_factor: float,
    rotation_sign: int,
) -> np.ndarray:
    """Map square image coordinates into a recursive logarithmic spiral.

    A square radius is used because the source image is rectangular.  The
    radius is folded into one scale period, while the angle changes by one
    full turn across that period.  This creates nested, rotating copies with
    no transparent overlays.
    """
    log_period = math.log(scale_factor)
    square_radius = np.maximum(np.maximum(np.abs(z.real), np.abs(z.imag)), 1e-5)
    period_index = np.ceil(np.log(square_radius) / log_period)
    folded = z * np.exp(-period_index * log_period)
    phase = rotation_sign * 2.0 * math.pi * np.log(square_radius) / log_period
    return folded * np.exp(1.0j * phase)


def _complex_power_map(
    z: np.ndarray,
    scale_factor: float,
    rotation_sign: int,
) -> tuple[np.ndarray, complex]:
    """Apply the conformal complex-power version of the Droste map."""
    log_period = math.log(scale_factor)
    rotation_rate = rotation_sign * log_period / (2.0 * math.pi)
    inverse_beta = 1.0 + 1.0j * rotation_rate
    beta = 1.0 / inverse_beta

    radius = np.maximum(np.abs(z), 1e-5)
    z_safe = radius * np.exp(1.0j * np.angle(z))
    z_source = np.exp(inverse_beta * np.log(z_safe))

    source_radius = np.abs(z_source)
    source_angle = np.angle(z_source)
    period_index = np.ceil(np.log(np.maximum(source_radius, 1e-8)) / log_period)
    folded_radius = source_radius / np.exp(period_index * log_period)
    folded_radius = np.clip(folded_radius, 1.0 / scale_factor, 1.0)
    return folded_radius * np.exp(1.0j * source_angle), beta


def _logtile_map(
    z: np.ndarray,
    scale_factor: float,
    rotation_sign: int,
    fixed_point: complex,
) -> tuple[np.ndarray, complex]:
    """Map image-plane coordinates back to a periodic logarithmic texture.

    The 3Blue1Brown construction uses a texture tile of width ``ln(c)`` and
    height ``2*pi`` in the log plane.  The affine map below makes the diagonal
    period ``ln(c) + 2*pi*i`` become one full turn after ``exp``.
    """
    log_period = math.log(scale_factor)
    angular_period = 2.0 * math.pi
    const = 1.0j * angular_period / complex(
        log_period,
        rotation_sign * angular_period,
    )

    radius = np.maximum(np.abs(z), 1e-6)
    log_z = np.log(radius) + 1.0j * np.angle(z)
    source_log = (log_z - fixed_point) / const + fixed_point

    source_x = np.mod(source_log.real, log_period) / log_period
    source_y = np.mod(source_log.imag + math.pi, angular_period) / angular_period
    source_y = 1.0 - source_y
    z_source = (source_x * 2.0 - 1.0) + 1.0j * (source_y * 2.0 - 1.0)
    return z_source, const


def _nested_droste_map(
    z: np.ndarray,
    scale_factor: float,
    rotation_sign: int,
    twist_degrees: float,
    recursion_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map only a central window into recursively smaller copies.

    The outer image is left unchanged. Inside a circular window with radius
    ``1 / scale_factor``, the target is mapped to the previous recursion
    level. This backward formulation avoids alpha compositing and gives an
    ordinary photograph a visible complete center copy.
    """
    if recursion_depth < 1:
        raise ValueError("recursion_depth must be at least 1.")

    window_radius = 1.0 / scale_factor
    radius = np.maximum(np.abs(z), 1e-8)
    inside = radius < window_radius
    levels = np.zeros_like(radius, dtype=np.int32)

    log_radius = np.log(radius[inside])
    log_window = math.log(window_radius)
    levels[inside] = np.floor(log_radius / log_window).astype(np.int32)
    levels[inside] = np.clip(levels[inside], 1, recursion_depth)

    z_source = z.copy()
    inner_levels = levels[inside].astype(np.float32)
    scale = np.power(window_radius, inner_levels)
    twist = rotation_sign * math.radians(twist_degrees) * inner_levels
    z_source[inside] = (z[inside] / scale) * np.exp(1.0j * twist)
    return z_source, levels


def _rectify_to_log_tile(
    source: np.ndarray,
    scale_factor: float,
    blend: float = 0.25,
) -> np.ndarray:
    """Resample a square photo into a log-polar tile (3b1b's rectification step).

    The seamless rotating Droste loop requires the source to be *periodic in the
    log plane* (see ``CreatePiHouseLog`` in the 3Blue1Brown source). This builds
    that tile from an arbitrary photo: the vertical axis is angle over
    ``[0, 2*pi)`` (which wraps seamlessly, giving a true rotation loop), the
    horizontal axis is log-radius over one scale period ``ln(scale_factor)``. A
    normal photo is not self-similar, so the radial (zoom) direction still has a
    seam; ``blend`` cross-fades a band at both radial edges toward their shared
    average so the tile also repeats continuously under scaling (approximate, but
    with no visible crack).
    """
    source_size = source.shape[0]
    log_period = math.log(scale_factor)
    tile = source_size

    cols = (np.arange(tile, dtype=np.float32) + 0.5) / tile  # 0..1 across log-radius
    rows = (np.arange(tile, dtype=np.float32) + 0.5) / tile  # 0..1 across angle
    jj, ii = np.meshgrid(cols, rows)
    log_r = (jj - 1.0) * log_period          # (-ln c, 0]  ->  radius in (1/c, 1]
    theta = ii * 2.0 * math.pi               # [0, 2*pi)
    z = np.exp(log_r + 1.0j * theta)
    rect = _sample_complex_image(source, z)  # angularly periodic by construction

    band = int(np.clip(blend, 0.0, 0.49) * tile)
    if band >= 1:
        edge = 0.5 * (rect[:, :1] + rect[:, -1:])                    # shared seam colour
        ramp = (np.arange(band, dtype=np.float32) / max(band - 1, 1))[None, :, None]
        rect[:, :band] = edge * (1.0 - ramp) + rect[:, :band] * ramp  # left edge -> edge
        r2 = ramp[:, ::-1]
        rect[:, -band:] = edge * (1.0 - r2) + rect[:, -band:] * r2    # right edge -> edge
    return rect


def render_escher_image(
    input_image: Image.Image,
    scale_factor: float = 3.0,
    output_size: int = 1000,
    rotation_sign: int = -1,
    source_size: int | None = None,
    mode: str = "spiral",
    fixed_point: complex = 3.8j,
    blend: float = 0.25,
    twist_degrees: float = 0.0,
    recursion_depth: int = 6,
) -> Image.Image:
    """Render the transformed image in memory."""
    if scale_factor <= 1.0:
        raise ValueError("scale_factor must be greater than 1.")
    if rotation_sign not in (-1, 1):
        raise ValueError("rotation_sign must be -1 or 1.")
    if mode not in ("nested", "spiral", "conformal", "logtile"):
        raise ValueError(
            "mode must be 'nested', 'spiral', 'conformal', or 'logtile'."
        )

    output_size = int(output_size)
    source_size = int(source_size or output_size)
    if output_size < 16 or source_size < 16:
        raise ValueError("output_size and source_size must be at least 16.")

    source = _square_image(input_image, source_size)

    axis = (np.arange(output_size, dtype=np.float32) + 0.5) / output_size
    axis = axis * 2.0 - 1.0
    xx, yy = np.meshgrid(axis, axis)
    z = xx + 1.0j * (-yy)

    sampling_source = source
    if mode == "nested":
        z_source, _ = _nested_droste_map(
            z,
            scale_factor=scale_factor,
            rotation_sign=rotation_sign,
            twist_degrees=twist_degrees,
            recursion_depth=recursion_depth,
        )
        map_parameter = None
    elif mode == "spiral":
        z_source = _log_spiral_map(z, scale_factor, rotation_sign)
        map_parameter = None
    elif mode == "conformal":
        z_source, map_parameter = _complex_power_map(z, scale_factor, rotation_sign)
    else:  # logtile — rectify the photo into a log-polar tile first (seamless in angle)
        sampling_source = _rectify_to_log_tile(source, scale_factor, blend=blend)
        z_source, map_parameter = _logtile_map(
            z,
            scale_factor=scale_factor,
            rotation_sign=rotation_sign,
            fixed_point=fixed_point,
        )

    result = _sample_complex_image(sampling_source, z_source)

    center = output_size // 2
    center_color = source[source_size // 2, source_size // 2]
    result[center - 1 : center + 1, center - 1 : center + 1] = center_color

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_escher_image(
    input_path: str | Path,
    output_path: str | Path,
    scale_factor: float = 3.0,
    output_size: int = 1000,
    rotation_sign: int = -1,
    source_size: int | None = None,
    mode: str = "spiral",
    fixed_point: complex = 3.8j,
    blend: float = 0.25,
    twist_degrees: float = 0.0,
    recursion_depth: int = 6,
) -> None:
    """Generate a logarithmic-spiral recursive image and save it."""
    with Image.open(input_path) as input_image:
        result_image = render_escher_image(
            input_image=input_image,
            scale_factor=scale_factor,
            output_size=output_size,
            rotation_sign=rotation_sign,
            source_size=source_size,
            mode=mode,
            fixed_point=fixed_point,
            blend=blend,
            twist_degrees=twist_degrees,
            recursion_depth=recursion_depth,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(output_path)

    print(f"Saved: {output_path.resolve()}")
    print(f"mode={mode}, scale_factor={scale_factor:.3f}, direction={rotation_sign}")
    if mode == "logtile":
        print(
            "fixed_point="
            f"{fixed_point.real:.6f}{fixed_point.imag:+.6f}j "
            "(log-plane fixed point)"
        )
    if mode == "nested":
        print(
            f"twist_degrees={twist_degrees:.3f}, "
            f"recursion_depth={recursion_depth}"
        )


class EscherApp:
    """Desktop interface for interactive image transformation."""

    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("Escher Logarithmic Image Lab")
        self.root.geometry("1280x820")
        self.root.minsize(980, 680)

        self.input_image: Image.Image | None = None
        self.result_image: Image.Image | None = None
        self.input_path = tk.StringVar()
        self.status = tk.StringVar(value="请选择一张图片开始")
        self.mode = tk.StringVar(value="nested")
        self.scale = tk.DoubleVar(value=3.0)
        self.direction = tk.IntVar(value=-1)
        self.direction_text = tk.StringVar(value="逆时针")
        self.twist = tk.DoubleVar(value=0.0)
        self.recursion_depth = tk.IntVar(value=6)
        self.fixed_real = tk.DoubleVar(value=0.0)
        self.fixed_imag = tk.DoubleVar(value=3.8)
        self.render_size = tk.IntVar(value=800)
        self._busy = False
        self._preview_refs = []

        self._build_layout()
        default_input = Path(__file__).resolve().parent / "test.jpg"
        if default_input.exists():
            self._load_image(default_input)

    def _build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk

        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.columnconfigure(1, weight=0)
        root_frame.rowconfigure(1, weight=1)

        title = ttk.Label(
            root_frame,
            text="Escher Logarithmic Image Lab",
            font=("Segoe UI", 18, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        preview_frame = ttk.Frame(root_frame)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, text="原图", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, pady=(0, 6)
        )
        ttk.Label(preview_frame, text="变换结果", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=1, pady=(0, 6)
        )

        self.original_view = ttk.Label(
            preview_frame,
            text="尚未选择图片",
            anchor="center",
            relief="solid",
        )
        self.original_view.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.result_view = ttk.Label(
            preview_frame,
            text="点击“生成预览”",
            anchor="center",
            relief="solid",
        )
        self.result_view.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        control_frame = ttk.LabelFrame(root_frame, text="参数控制", padding=12)
        control_frame.grid(row=1, column=1, sticky="ns")
        control_frame.columnconfigure(1, weight=1)

        ttk.Button(
            control_frame,
            text="选择图片",
            command=self.choose_image,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        path_entry = ttk.Entry(control_frame, textvariable=self.input_path, width=34)
        path_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        path_entry.configure(state="readonly")

        ttk.Label(control_frame, text="算法模式").grid(row=2, column=0, sticky="w", pady=4)
        mode_box = ttk.Combobox(
            control_frame,
            textvariable=self.mode,
            values=("nested", "spiral", "conformal", "logtile"),
            state="readonly",
            width=18,
        )
        mode_box.grid(row=2, column=1, sticky="ew", pady=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._update_hint())

        ttk.Label(control_frame, text="递归尺度").grid(row=3, column=0, sticky="w", pady=4)
        tk.Scale(
            control_frame,
            variable=self.scale,
            from_=1.2,
            to=32.0,
            resolution=0.1,
            orient="horizontal",
            showvalue=True,
            length=210,
        ).grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(control_frame, text="每层旋转角度").grid(
            row=4, column=0, sticky="w", pady=4
        )
        tk.Scale(
            control_frame,
            variable=self.twist,
            from_=-180.0,
            to=180.0,
            resolution=1.0,
            orient="horizontal",
            showvalue=True,
            length=210,
        ).grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Label(control_frame, text="递归层数").grid(
            row=5, column=0, sticky="w", pady=4
        )
        ttk.Spinbox(
            control_frame,
            from_=1,
            to=12,
            increment=1,
            textvariable=self.recursion_depth,
            width=10,
        ).grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(control_frame, text="旋转方向").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Button(
            control_frame,
            textvariable=self.direction_text,
            command=self.toggle_direction,
        ).grid(row=6, column=1, sticky="ew", pady=4)

        ttk.Label(control_frame, text="输出尺寸").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Spinbox(
            control_frame,
            from_=256,
            to=2048,
            increment=64,
            textvariable=self.render_size,
            width=10,
        ).grid(row=7, column=1, sticky="w", pady=4)

        ttk.Separator(control_frame).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=12
        )
        ttk.Label(control_frame, text="对数平面不动点").grid(
            row=9, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(control_frame, text="实部").grid(row=10, column=0, sticky="w", pady=4)
        ttk.Entry(control_frame, textvariable=self.fixed_real, width=12).grid(
            row=10, column=1, sticky="w", pady=4
        )
        ttk.Label(control_frame, text="虚部").grid(row=11, column=0, sticky="w", pady=4)
        ttk.Entry(control_frame, textvariable=self.fixed_imag, width=12).grid(
            row=11, column=1, sticky="w", pady=4
        )

        self.hint_label = ttk.Label(
            control_frame,
            text="nested：外层保留原图，只在中心形成递归副本；建议先用此模式",
            wraplength=260,
            foreground="#555555",
        )
        self.hint_label.grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 12))

        self.generate_button = ttk.Button(
            control_frame,
            text="生成预览",
            command=self.generate_preview,
        )
        self.generate_button.grid(row=13, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(
            control_frame,
            text="保存结果",
            command=self.save_result,
        ).grid(row=14, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(
            control_frame,
            text="恢复默认",
            command=self.reset,
        ).grid(row=15, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Label(root_frame, textvariable=self.status, foreground="#444444").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

    def _update_hint(self) -> None:
        hints = {
            "nested": "nested：外层保留原图，只在中心形成递归副本；建议先用此模式",
            "spiral": "spiral：整幅照片进行连续螺旋映射，容易出现强烈拉伸",
            "conformal": "conformal：复数幂近似，局部角度变化更规则",
            "logtile": "logtile：先把照片整流成 log-polar 贴图，再做 log→exp；旋转方向无缝循环（推荐）",
        }
        self.hint_label.configure(text=hints[self.mode.get()])

    def toggle_direction(self) -> None:
        self.direction.set(-self.direction.get())
        self.direction_text.set("逆时针" if self.direction.get() < 0 else "顺时针")

    def choose_image(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="选择输入图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if selected:
            self._load_image(Path(selected))

    def _load_image(self, path: Path) -> None:
        try:
            with Image.open(path) as image:
                self.input_image = image.convert("RGB")
            self.input_path.set(str(path))
            self.result_image = None
            self._show_image(self.original_view, self.input_image)
            self.result_view.configure(image="", text="点击“生成预览”")
            self.status.set(f"已载入：{path.name}")
        except Exception as exc:
            self._show_error(f"无法打开图片：{exc}")

    def _show_image(self, widget, image: Image.Image) -> None:
        from PIL import ImageTk

        display = image.copy()
        display.thumbnail((520, 650), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display)
        widget.configure(image=photo, text="")
        widget.image = photo
        self._preview_refs.append(photo)
        self._preview_refs = self._preview_refs[-4:]

    def _read_parameters(self) -> dict:
        return {
            "scale_factor": float(self.scale.get()),
            "output_size": int(self.render_size.get()),
            "rotation_sign": int(self.direction.get()),
            "mode": self.mode.get(),
            "fixed_point": complex(float(self.fixed_real.get()), float(self.fixed_imag.get())),
            "twist_degrees": float(self.twist.get()),
            "recursion_depth": int(self.recursion_depth.get()),
        }

    def generate_preview(self) -> None:
        if self.input_image is None:
            self._show_error("请先选择一张输入图片。")
            return
        if self._busy:
            return
        try:
            parameters = self._read_parameters()
        except ValueError:
            self._show_error("参数格式不正确，请检查尺度、尺寸和不动点。")
            return

        self._busy = True
        self.generate_button.configure(state="disabled")
        self.status.set("正在计算，请稍候...")
        input_copy = self.input_image.copy()

        def worker() -> None:
            try:
                result = render_escher_image(input_copy, **parameters)
                self.root.after(0, lambda: self._generation_done(result))
            except Exception as exc:
                self.root.after(0, lambda: self._generation_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _generation_done(self, result: Image.Image) -> None:
        self.result_image = result
        self._show_image(self.result_view, result)
        self.status.set("预览生成完成")
        self._busy = False
        self.generate_button.configure(state="normal")

    def _generation_failed(self, exc: Exception) -> None:
        self._busy = False
        self.generate_button.configure(state="normal")
        self._show_error(f"生成失败：{exc}")

    def save_result(self) -> None:
        from tkinter import filedialog

        if self.result_image is None:
            self._show_error("请先生成预览。")
            return
        selected = filedialog.asksaveasfilename(
            title="保存变换结果",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg")],
        )
        if selected:
            self.result_image.save(selected)
            self.status.set(f"已保存：{Path(selected).name}")

    def reset(self) -> None:
        self.mode.set("nested")
        self.scale.set(3.0)
        self.direction.set(-1)
        self.direction_text.set("逆时针")
        self.twist.set(0.0)
        self.recursion_depth.set(6)
        self.fixed_real.set(0.0)
        self.fixed_imag.set(3.8)
        self.render_size.set(800)
        self._update_hint()
        self.status.set("已恢复默认参数")

    def _show_error(self, message: str) -> None:
        from tkinter import messagebox

        self.status.set(message)
        messagebox.showerror("提示", message)


def launch_gui() -> None:
    """Launch the interactive desktop application."""
    import tkinter as tk

    root = tk.Tk()
    EscherApp(root)
    root.mainloop()


def main() -> None:
    if len(sys.argv) == 1:
        launch_gui()
        return

    parser = argparse.ArgumentParser(
        description="Generate an Escher-style Droste / log-spiral image."
    )
    parser.add_argument("--gui", action="store_true", help="Open the interactive GUI.")
    parser.add_argument("--input", default="test.jpg", help="Input image path.")
    parser.add_argument("--output", default="escher_output.jpg", help="Output image path.")
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Scale period; 2.5-5 shows several copies, 8-32 shows fewer larger copies.",
    )
    parser.add_argument("--size", type=int, default=1000, help="Output width and height.")
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    parser.add_argument(
        "--mode",
        choices=("nested", "spiral", "conformal", "logtile"),
        default="nested",
        help=(
            "nested=local center recursion, spiral=global spiral, "
            "conformal=complex-power, logtile=3b1b log-exp mapping."
        ),
    )
    parser.add_argument(
        "--fixed-point",
        nargs=2,
        type=float,
        metavar=("REAL", "IMAG"),
        default=(0.0, 3.8),
        help="Log-plane fixed point for logtile mode, for example: --fixed-point 0 3.8.",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.25,
        help="logtile radial seam blend in [0, 0.49]; softens the zoom seam of non-self-similar photos.",
    )
    parser.add_argument(
        "--twist-degrees",
        type=float,
        default=0.0,
        help="Rotation applied at each nested level in nested mode.",
    )
    parser.add_argument(
        "--recursion-depth",
        type=int,
        default=6,
        help="Maximum number of nested levels in nested mode.",
    )
    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    generate_escher_image(
        input_path=args.input,
        output_path=args.output,
        scale_factor=args.scale,
        output_size=args.size,
        rotation_sign=args.direction,
        mode=args.mode,
        fixed_point=complex(*args.fixed_point),
        blend=args.blend,
        twist_degrees=args.twist_degrees,
        recursion_depth=args.recursion_depth,
    )


if __name__ == "__main__":
    main()
