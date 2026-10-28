import tkinter as tk

GEOMETRY = {
    "t": lambda sw, sh, t: (sw, t, 0, 0),
    "b": lambda sw, sh, t: (sw, t, 0, sh - t),
    "l": lambda sw, sh, t: (t, sh, 0, 0),
    "r": lambda sw, sh, t: (t, sh, sw - t, 0),
}


class Softbox:
    """Screen-edge light bars (borderless, always-on-top) that light your face from the display."""

    def __init__(self, root, settings):
        self.root = root
        self.s = settings
        self.wins = {}
        self._sig = None
        self._tick()

    def _tick(self):
        s = self.s
        sig = (s.softbox_on, s.softbox_color, round(s.softbox_level, 2), round(s.softbox_width, 3), s.softbox_edges)
        if sig != self._sig:
            self._sig = sig
            self._apply()
        if s.running:
            self.root.after(100, self._tick)
        else:
            self.close()

    def _apply(self):
        s = self.s
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        t = max(20, int(sh * s.softbox_width))
        r, g, b = (int(c * s.softbox_level) for c in s.softbox_color)
        color = f"#{r:02x}{g:02x}{b:02x}"
        for edge, geo in GEOMETRY.items():
            want = s.softbox_on and edge in s.softbox_edges and s.softbox_level > 0.01
            win = self.wins.get(edge)
            if not want:
                if win is not None:
                    win.destroy()
                    del self.wins[edge]
                continue
            if win is None:
                win = tk.Toplevel(self.root)
                win.overrideredirect(True)
                win.attributes("-topmost", True)
                self.wins[edge] = win
            w, h, x, y = geo(sw, sh, t)
            win.geometry(f"{w}x{h}+{x}+{y}")
            win.configure(bg=color)

    def close(self):
        for win in self.wins.values():
            win.destroy()
        self.wins.clear()
