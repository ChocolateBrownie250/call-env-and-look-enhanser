import cv2
import numpy as np

try:
    import mediapipe as mp

    _fm = mp.solutions.face_mesh
except Exception:  # mediapipe missing or without the legacy solutions API
    _fm = None

DETECT_EVERY = 3  # run the face mesh every Nth frame, reuse masks in between
SMALL_W = 256  # width of the working image for smoothing / masks


def _idx(conns):
    return sorted({i for e in conns for i in e})


class Enhancer:
    """Real-time skin smoothing, tone evening, eye brightening and auto exposure.

    Cost control: the face mesh runs on a 256 px image every 3rd frame, and all
    skin work happens on the face crop at <=256 px, with the original texture
    added back at full resolution.
    """

    def __init__(self):
        self.mesh = None
        if _fm is not None:
            self.mesh = _fm.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._oval = _idx(_fm.FACEMESH_FACE_OVAL)
            self._holes = [
                _idx(c)
                for c in (
                    _fm.FACEMESH_LEFT_EYE,
                    _fm.FACEMESH_RIGHT_EYE,
                    _fm.FACEMESH_LIPS,
                    _fm.FACEMESH_LEFT_EYEBROW,
                    _fm.FACEMESH_RIGHT_EYEBROW,
                )
            ]
            self._eye_idx = [_idx(_fm.FACEMESH_LEFT_EYE), _idx(_fm.FACEMESH_RIGHT_EYE)]
        self._n = 0
        self._shape = None
        self._roi = None
        self._small = None
        self._skin_s = None
        self._skin = None
        self._eyes = None
        self._gain = 1.0
        self._lut_key = None
        self._lut = None
        self.luma = 0.5
        self.face_found = False

    def _detect(self, bgr):
        h, w = bgr.shape[:2]
        self._shape = (h, w)
        self._roi = self._skin = self._eyes = None
        self.face_found = False

        dw = min(SMALL_W, w)
        det = cv2.resize(bgr, (dw, max(1, int(h * dw / w))), interpolation=cv2.INTER_AREA)
        pts = None
        if self.mesh is not None:
            res = self.mesh.process(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts = np.array([(p.x * w, p.y * h) for p in lm], np.float32)

        if pts is None:
            self._set_luma(float(cv2.cvtColor(det, cv2.COLOR_BGR2GRAY).mean()) / 255.0)
            if self.mesh is not None:
                return
            x0, y0, x1, y1 = 0, 0, w, h  # no mesh: skin-colour mask over the whole frame
        else:
            self.face_found = True
            ox = pts[self._oval]
            x0, y0 = ox.min(0)
            x1, y1 = ox.max(0)
            pad = 0.12 * max(x1 - x0, y1 - y0)
            x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            x1, y1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
            if x1 - x0 < 16 or y1 - y0 < 16:
                self.face_found = False
                return

        rw, rh = x1 - x0, y1 - y0
        sw = min(SMALL_W, rw)
        sh = max(1, int(rh * sw / rw))
        small = cv2.resize(bgr[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_AREA)

        gate = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), (0, 130, 70), (255, 182, 138))
        gate = cv2.GaussianBlur(gate, (5, 5), 0).astype(np.float32) / 255.0

        eyes_s = None
        if pts is not None:
            local = (pts - (x0, y0)) * (sw / rw, sh / rh)

            def hull(ix):
                return cv2.convexHull(local[ix].astype(np.int32))

            skin = np.zeros((sh, sw), np.uint8)
            cv2.fillConvexPoly(skin, hull(self._oval), 255)
            holes = np.zeros((sh, sw), np.uint8)
            for ix in self._holes:
                cv2.fillConvexPoly(holes, hull(ix), 255)
            skin[cv2.dilate(holes, np.ones((3, 3), np.uint8)) > 0] = 0
            skin = cv2.GaussianBlur(skin, (9, 9), 0).astype(np.float32) / 255.0
            skin_s = skin * (0.35 + 0.65 * gate)

            eyes = np.zeros((sh, sw), np.uint8)
            for ix in self._eye_idx:
                cv2.fillConvexPoly(eyes, hull(ix), 255)
            eyes = cv2.dilate(eyes, np.ones((3, 3), np.uint8))
            eyes_s = cv2.GaussianBlur(eyes, (7, 7), 0).astype(np.float32) / 255.0
        else:
            skin_s = gate

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        wsum = float(skin_s.sum())
        if pts is not None and wsum > 1:
            self._set_luma(float((gray * skin_s).sum() / wsum) / 255.0)

        self._roi = (x0, y0, x1, y1)
        self._small = (sw, sh)
        self._skin_s = skin_s
        self._skin = cv2.resize(skin_s, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if eyes_s is not None:
            self._eyes = cv2.resize(eyes_s, (rw, rh), interpolation=cv2.INTER_LINEAR)

    def _set_luma(self, v):
        self.luma += 0.3 * (v - self.luma)

    def _tone_lut(self, exposure, warmth):
        if exposure > 0:
            g = np.log(0.5) / np.log(min(max(self.luma, 0.05), 0.95))
            g = float(np.clip(g, 0.55, 1.2)) ** exposure
            self._gain += 0.1 * (g - self._gain)
        else:
            self._gain = 1.0
        key = (round(self._gain, 3), round(warmth, 2))
        if key != self._lut_key:
            base = np.power(np.arange(256) / 255.0, self._gain) * 255.0
            lut = np.empty((256, 1, 3), np.uint8)
            for c, k in enumerate((1 - 0.06 * warmth, 1.0, 1 + 0.06 * warmth)):  # B, G, R
                lut[:, 0, c] = np.clip(base * k, 0, 255)
            self._lut_key, self._lut = key, lut
        return self._lut

    def process(self, bgr, s):
        h, w = bgr.shape[:2]
        if self._n % DETECT_EVERY == 0 or self._shape != (h, w):
            self._detect(bgr)
        self._n += 1

        out = bgr
        if s.exposure > 0 or s.warmth:
            out = cv2.LUT(out, self._tone_lut(s.exposure, s.warmth))

        if self._roi is None:
            return out
        x0, y0, x1, y1 = self._roi
        rw, rh = x1 - x0, y1 - y0
        roi = out[y0:y1, x0:x1]
        skin = self._skin[..., None]
        res = None

        if s.smooth > 0 or s.tone > 0:
            sw, sh = self._small
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            s2 = small
            if s.smooth > 0:
                sm = cv2.bilateralFilter(small, 5, 20 + 60 * s.smooth, 3)
                s2 = cv2.addWeighted(small, 1 - min(1.0, s.smooth * 1.4), sm, min(1.0, s.smooth * 1.4), 0)
            if s.tone > 0:
                lab = cv2.cvtColor(s2, cv2.COLOR_BGR2LAB).astype(np.float32)
                ab = lab[..., 1:]
                blur = cv2.GaussianBlur(ab, (0, 0), max(1.5, sw * 0.03))
                lab[..., 1:] = ab + (blur - ab) * s.tone
                s2 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            up = cv2.resize(s2, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            detail = roi.astype(np.float32) - cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
            target = up + detail * (1.0 - 0.7 * s.smooth)
            cur = roi.astype(np.float32)
            res = cur + (target - cur) * skin

        if self._eyes is not None and s.eyes > 0:
            if res is None:
                res = roi.astype(np.float32)
            res += self._eyes[..., None] * (s.eyes * 35.0)

        if res is not None:
            roi[:] = np.clip(res, 0, 255).astype(np.uint8)
        return out
