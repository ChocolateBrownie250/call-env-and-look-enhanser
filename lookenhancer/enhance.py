import cv2
import numpy as np

try:
    import mediapipe as mp

    _fm = mp.solutions.face_mesh
except Exception:  # mediapipe missing or without the legacy solutions API
    _fm = None


def _idx(conns):
    return sorted({i for e in conns for i in e})


def _odd(n):
    n = max(3, int(n))
    return n if n % 2 else n + 1


class Enhancer:
    """Real-time skin smoothing, tone evening, eye brightening and auto exposure."""

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
            self._eyes = [_idx(_fm.FACEMESH_LEFT_EYE), _idx(_fm.FACEMESH_RIGHT_EYE)]
        self._n = 0
        self._skin = None
        self._eye_mask = None
        self._shape = None
        self._gain = 1.0
        self.luma = 0.5
        self.face_found = False

    @staticmethod
    def _gate(bgr):
        """Soft skin-colour mask (YCrCb); keeps hair, beard and backdrop out."""
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        g = cv2.inRange(ycc, (0, 130, 70), (255, 182, 138))
        g = cv2.GaussianBlur(g, (_odd(bgr.shape[1] * 0.01),) * 2, 0)
        return g.astype(np.float32) / 255.0

    def _detect(self, bgr):
        h, w = bgr.shape[:2]
        gate = self._gate(bgr)
        self._shape = (h, w)
        if self.mesh is None:
            self._skin = gate
            self._eye_mask = None
            self.face_found = False
            return
        sw = 320 if w > 320 else w
        small = cv2.resize(bgr, (sw, int(h * sw / w)), interpolation=cv2.INTER_AREA)
        res = self.mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            self._skin = self._eye_mask = None
            self.face_found = False
            return
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], np.float32)

        def hull(ix):
            return cv2.convexHull(pts[ix].astype(np.int32))

        skin = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(skin, hull(self._oval), 255)
        holes = np.zeros((h, w), np.uint8)
        for ix in self._holes:
            cv2.fillConvexPoly(holes, hull(ix), 255)
        holes = cv2.dilate(holes, np.ones((_odd(w * 0.006),) * 2, np.uint8))
        skin[holes > 0] = 0
        k = _odd(w * 0.02)
        skin = cv2.GaussianBlur(skin, (k, k), 0).astype(np.float32) / 255.0
        self._skin = skin * (0.35 + 0.65 * gate)

        eyes = np.zeros((h, w), np.uint8)
        for ix in self._eyes:
            cv2.fillConvexPoly(eyes, hull(ix), 255)
        eyes = cv2.dilate(eyes, np.ones((_odd(w * 0.008),) * 2, np.uint8))
        self._eye_mask = cv2.GaussianBlur(eyes, (k, k), 0).astype(np.float32) / 255.0
        self.face_found = True

    def process(self, bgr, s):
        h, w = bgr.shape[:2]
        self._n += 1
        if self._n % 2 == 1 or self._shape != (h, w):
            self._detect(bgr)
        skin, eyes = self._skin, self._eye_mask

        # measure physical light on the face (before any correction)
        gray = cv2.cvtColor(cv2.resize(bgr, (160, max(1, int(160 * h / w)))), cv2.COLOR_BGR2GRAY)
        if skin is not None and self.face_found:
            m = cv2.resize(skin, (gray.shape[1], gray.shape[0]))
            wsum = float(m.sum())
            luma = float((gray * m).sum() / wsum) / 255.0 if wsum > 1 else float(gray.mean()) / 255.0
        else:
            luma = float(gray.mean()) / 255.0
        self.luma += 0.1 * (luma - self.luma)

        out = bgr

        # auto exposure (gamma toward a mid-tone face)
        if s.exposure > 0:
            g = np.log(0.5) / np.log(min(max(self.luma, 0.05), 0.95))
            g = float(np.clip(g, 0.55, 1.2)) ** s.exposure
            self._gain += 0.1 * (g - self._gain)
            lut = (np.power(np.arange(256) / 255.0, self._gain) * 255).astype(np.uint8)
            out = cv2.LUT(out, lut)

        if skin is not None and (s.smooth > 0 or s.tone > 0):
            base = out.astype(np.float32)
            if s.smooth > 0:
                small = cv2.resize(out, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                sm = cv2.bilateralFilter(small, 7, 20 + 60 * s.smooth, 5)
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                detail = base - cv2.GaussianBlur(base, (0, 0), 2)
                sm += detail * 0.25
                a = (skin * min(1.0, s.smooth * 1.4))[..., None]
                base = base * (1 - a) + sm * a
            if s.tone > 0:
                lab = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                ab = lab[..., 1:3]
                blur = cv2.GaussianBlur(ab, (0, 0), max(2.0, w * 0.02))
                a = (skin * s.tone)[..., None]
                lab[..., 1:3] = ab * (1 - a) + blur * a
                base = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            out = np.clip(base, 0, 255).astype(np.uint8)

        if eyes is not None and s.eyes > 0:
            lift = (eyes * s.eyes * 35.0)[..., None]
            out = np.clip(out.astype(np.float32) + lift, 0, 255).astype(np.uint8)

        if s.warmth:
            f = out.astype(np.float32)
            f[..., 2] *= 1 + 0.06 * s.warmth
            f[..., 0] *= 1 - 0.06 * s.warmth
            out = np.clip(f, 0, 255).astype(np.uint8)
        return out
