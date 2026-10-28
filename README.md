# Look Enhancer

Webcam beauty filter + on-screen softbox, exposed as a virtual camera you pick in Google Meet, Zoom, Teams, or OBS.

- **Enhancement** (real time, CPU): MediaPipe face mesh builds a skin mask; edge-preserving smoothing, tone evening, eye brightening, auto exposure, warmth. Falls back to a skin-colour mask if mediapipe is unavailable.
- **Softbox**: borderless always-on-top light bars on the screen edges (top/left/right/bottom, any colour, width, level). *Auto level* raises the light until your face reaches the target brightness, so the filter works on a well-lit image.
- **Resolution**: 1080p / 720p / 540p / 480p / 360p, switchable live. Lower = less CPU (the frame is downscaled before processing).
- No face found → only exposure/warmth are applied.

## Model and performance

Face model: MediaPipe **Face Landmarker** (478-point mesh, ~4 MB, float16). It is downloaded once to `~/.cache/lookenhancer/` on first run; older mediapipe versions use the bundled legacy mesh instead. No heavy restoration network (GFPGAN/CodeFormer) is used, since those can't hold 30 fps on a laptop CPU.

Speed tricks: mesh runs on a 256 px image every 3rd frame; all skin work happens on the face crop at <=256 px with the original texture added back; exposure and warmth are one LUT pass; capture uses MJPG with a 1-frame buffer; preview updates every 3rd frame.

Measured here (shared cloud CPU, enhancer only, one face): 360p 7 ms, 720p 14 ms, 1080p 28 ms per frame. The status line in the app shows live ms/frame.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m lookenhancer            # --res 360p --fps 24 --camera 1 ...
```

Linux headless/minimal images may need `libegl1` (`sudo apt install libegl1`) for mediapipe.

Try without a camera: `python -m lookenhancer --still in.jpg out.jpg`

## Getting it into Google Meet

Browsers only list real camera devices, so a **virtual camera driver** is required (nothing in Python can create one alone). Install one once; OBS only needs to be installed, **not opened**:

| OS | Driver (pick one) | Run |
|---|---|---|
| Windows | OBS Studio (installs "OBS Virtual Camera"), or the lighter [UnityCapture](https://github.com/schellingb/UnityCapture) | `python -m lookenhancer` (`--backend unitycapture` for UnityCapture) |
| macOS | OBS Studio 26+ (start/stop the Virtual Camera once to approve the extension, then close OBS) | `python -m lookenhancer` |
| Linux | `sudo modprobe v4l2loopback exclusive_caps=1 card_label="Look Enhancer"` | `python -m lookenhancer --backend v4l2loopback` |

Then in Meet: Settings → Video → camera → **OBS Virtual Camera** (or *Unity Video Capture* / *Look Enhancer*). macOS: allow the terminal/Python under System Settings → Privacy → Camera.

To use it **inside OBS** instead: add a *Video Capture Device* source. With the OBS backend, use UnityCapture/v4l2loopback so OBS's own virtual camera isn't the same device.

Without a driver the app still runs as a live preview (status line says "preview only").

## Tips

- The top light bar covers app toolbars; shrink *Bar width* or untick edges you don't need.
- Low-power laptop: 540p/360p and 24 fps.
