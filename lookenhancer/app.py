import os
import threading
import time
import tkinter as tk
from tkinter import colorchooser, ttk

import cv2
import numpy as np

from .enhance import Enhancer
from .settings import PRESETS, RESOLUTIONS, Settings
from .softbox import Softbox

try:
    import pyvirtualcam
except Exception:
    pyvirtualcam = None

PREVIEW_W = 480


class CameraWorker(threading.Thread):
    def __init__(self, settings, camera=0, backend=None, device=None):
        super().__init__(daemon=True)
        self.s = settings
        self.camera, self.backend, self.device = camera, backend, device
        self.preview = None

    def _open(self, res):
        w, h = RESOLUTIONS[res]
        api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera, api)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self.s.fps)
        cam, err = None, ""
        if pyvirtualcam is None:
            err = "pyvirtualcam not installed"
        else:
            try:
                kw = {k: v for k, v in (("backend", self.backend), ("device", self.device)) if v}
                cam = pyvirtualcam.Camera(w, h, self.s.fps, fmt=pyvirtualcam.PixelFormat.BGR, **kw)
            except Exception as e:  # no virtual camera driver: keep running as preview only
                err = str(e).splitlines()[0] if str(e) else type(e).__name__
        if not cap.isOpened():
            self.s.status = "camera not available (permission or in use)"
        elif cam is not None:
            self.s.status = f"virtual camera: {cam.device} {w}x{h}@{self.s.fps}"
        else:
            self.s.status = f"preview only ({err})"
        return cap, cam, w, h

    def run(self):
        s = self.s
        enh = Enhancer()
        cap = cam = None
        cur = None
        w = h = 0
        while s.running:
            key = (s.resolution, s.fps)
            if key != cur:
                if cap is not None:
                    cap.release()
                if cam is not None:
                    cam.close()
                cap, cam, w, h = self._open(s.resolution)
                cur = key
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            out = frame if (s.compare or not s.enabled) else enh.process(frame, s)
            s.face_luma, s.face_found = enh.luma, enh.face_found
            if s.softbox_on and s.softbox_auto:
                lvl = s.softbox_level + (s.softbox_target - enh.luma) * 0.03
                s.softbox_level = min(1.0, max(0.05, lvl))
            pv = cv2.resize(out, (PREVIEW_W, int(PREVIEW_W * h / w)), interpolation=cv2.INTER_AREA)
            self.preview = cv2.cvtColor(pv, cv2.COLOR_BGR2RGB)
            if cam is not None:
                cam.send(out)
                cam.sleep_until_next_frame()
        if cap is not None:
            cap.release()
        if cam is not None:
            cam.close()


def run_ui(camera=0, backend=None, device=None, resolution="720p", fps=30):
    s = Settings(resolution=resolution, fps=fps)
    root = tk.Tk()
    root.title("Look Enhancer")
    worker = CameraWorker(s, camera, backend, device)
    box = Softbox(root, s)

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    img_label = ttk.Label(frm)
    img_label.grid(row=0, column=0, rowspan=40, padx=(0, 12), sticky="n")
    status = ttk.Label(frm, text="", wraplength=PREVIEW_W)
    status.grid(row=40, column=0, sticky="w")

    vars_ = {}
    row = [0]

    def title(text):
        ttk.Label(frm, text=text, font=("", 10, "bold")).grid(row=row[0], column=1, columnspan=2, sticky="w", pady=(8, 0))
        row[0] += 1

    def slider(label, name, lo=0.0, hi=1.0):
        ttk.Label(frm, text=label).grid(row=row[0], column=1, sticky="w")
        v = tk.DoubleVar(value=getattr(s, name))
        vars_[name] = v
        ttk.Scale(frm, from_=lo, to=hi, variable=v, length=200,
                  command=lambda _x, n=name, v=v: setattr(s, n, float(v.get()))).grid(row=row[0], column=2)
        row[0] += 1

    def check(label, name):
        v = tk.BooleanVar(value=getattr(s, name))
        vars_[name] = v
        ttk.Checkbutton(frm, text=label, variable=v,
                        command=lambda n=name, v=v: setattr(s, n, bool(v.get()))).grid(row=row[0], column=1, columnspan=2, sticky="w")
        row[0] += 1

    def combo(label, values, current, cb):
        ttk.Label(frm, text=label).grid(row=row[0], column=1, sticky="w")
        c = ttk.Combobox(frm, values=values, state="readonly", width=10)
        c.set(current)
        c.bind("<<ComboboxSelected>>", lambda _e: cb(c.get()))
        c.grid(row=row[0], column=2, sticky="w")
        row[0] += 1

    title("Output")
    combo("Resolution", list(RESOLUTIONS), s.resolution, lambda v: setattr(s, "resolution", v))
    combo("FPS", ["15", "24", "30"], str(s.fps), lambda v: setattr(s, "fps", int(v)))
    check("Enhance", "enabled")
    check("Show original (compare)", "compare")

    title("Look")
    pf = ttk.Frame(frm)
    pf.grid(row=row[0], column=1, columnspan=2, sticky="w")
    row[0] += 1

    def preset(name):
        for k, val in PRESETS[name].items():
            setattr(s, k, val)
            vars_[k].set(val)

    slider("Skin smoothing", "smooth")
    slider("Even skin tone", "tone")
    slider("Eye brightening", "eyes")
    slider("Auto exposure", "exposure")
    slider("Warmth", "warmth", -1.0, 1.0)
    for n in PRESETS:
        ttk.Button(pf, text=n, command=lambda n=n: preset(n)).pack(side="left", padx=2)

    title("Softbox (screen light bars)")
    check("Light on", "softbox_on")
    check("Auto level (lights face until it is bright enough)", "softbox_auto")
    slider("Level", "softbox_level", 0.0, 1.0)
    slider("Bar width", "softbox_width", 0.02, 0.2)
    ef = ttk.Frame(frm)
    ef.grid(row=row[0], column=1, columnspan=2, sticky="w")
    row[0] += 1
    edge_vars = {}

    def set_edges():
        s.softbox_edges = "".join(k for k, v in edge_vars.items() if v.get())

    for k, name in (("t", "Top"), ("l", "Left"), ("r", "Right"), ("b", "Bottom")):
        edge_vars[k] = tk.BooleanVar(value=k in s.softbox_edges)
        ttk.Checkbutton(ef, text=name, variable=edge_vars[k], command=set_edges).pack(side="left")

    def pick_color():
        c = colorchooser.askcolor(color="#%02x%02x%02x" % s.softbox_color)[0]
        if c:
            s.softbox_color = tuple(int(x) for x in c)

    ttk.Button(frm, text="Light colour...", command=pick_color).grid(row=row[0], column=1, sticky="w", pady=4)

    def refresh():
        pv = worker.preview
        if pv is not None:
            h, w = pv.shape[:2]
            ppm = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(pv).tobytes()
            img = tk.PhotoImage(data=ppm, format="PPM")
            img_label.configure(image=img)
            img_label.image = img
        if s.softbox_auto:
            vars_["softbox_level"].set(s.softbox_level)
        status.configure(text=f"{s.status}\nface: {'found' if s.face_found else 'not found'}  light: {s.face_luma:.2f}")
        root.after(80, refresh)

    def close():
        s.running = False
        root.after(200, root.destroy)

    root.protocol("WM_DELETE_WINDOW", close)
    worker.start()
    refresh()
    root.mainloop()
    s.running = False
    worker.join(timeout=2)
