import argparse

from .settings import RESOLUTIONS


def main():
    p = argparse.ArgumentParser(prog="lookenhancer", description="Webcam beauty filter + screen softbox -> virtual camera")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--res", default="720p", choices=list(RESOLUTIONS))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--backend", help="pyvirtualcam backend: obs, unitycapture, v4l2loopback (default: auto)")
    p.add_argument("--device", help="virtual camera device name/path")
    p.add_argument("--still", nargs=2, metavar=("IN", "OUT"), help="enhance one image and exit (no camera needed)")
    a = p.parse_args()

    if a.still:
        import cv2

        from .enhance import Enhancer
        from .settings import Settings

        img = cv2.imread(a.still[0])
        if img is None:
            raise SystemExit(f"cannot read {a.still[0]}")
        cv2.imwrite(a.still[1], Enhancer().process(img, Settings()))
        return

    from .app import run_ui

    run_ui(a.camera, a.backend, a.device, a.res, a.fps)


if __name__ == "__main__":
    main()
