"""Camera intrinsic calibration scaffold.

Why this exists
---------------
Depth Anything V2 assumes an unknown pinhole camera, so its metric output
carries a per-camera bias that a single scale factor cannot fully remove.
Knowing the intrinsics lets us:

  * undistort the frame, which removes the systematic error at the image edges,
    where depth is worst;
  * reason in metric 3D rather than in pixels (e.g. estimate lateral offset and
    the true along-road range of a wheel contact point);
  * correct the optical-flow scale in ego-motion estimation, which currently
    assumes a typical 70 degree horizontal FOV.

Status
------
Deliberately NOT part of the MVP path. Calibration needs a chessboard capture
rig and a dozen or so known-position images, which the prototype does not
require. The architecture supports adding it without touching the pipeline:
run `calibrate_from_images`, save to config/calibration.yaml, and the rest of
the code reads `intrinsics()` when it is present.

Usage
-----
    python main_fcw.py --calibrate-cam 0            # live: grab frames, then solve
    python main_fcw.py --source calib/left01.jpg --show-depth   # preview a map
"""

import os

import numpy as np

import cv2

CALIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config")
CALIB_PATH = os.path.join(CALIB_DIR, "calibration.yaml")


class CameraCalibration:
    """Holds and persists pinhole intrinsics plus lens distortion."""

    def __init__(self, camera_matrix=None, dist_coeffs=None, image_size=None,
                 reprojection_error=None):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64) if camera_matrix is not None else None
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel() if dist_coeffs is not None else None
        self.image_size = tuple(image_size) if image_size else None  # (width, height)
        self.reprojection_error = reprojection_error

    @property
    def is_valid(self):
        return (self.camera_matrix is not None
                and self.camera_matrix.shape == (3, 3)
                and np.isfinite(self.camera_matrix).all()
                and self.camera_matrix[0, 0] > 0)

    @property
    def has_distortion(self):
        return (self.dist_coeffs is not None
                and self.dist_coeffs.size >= 4
                and np.abs(self.dist_coeffs[:4]).max() > 1e-6)

    def focal_length_px(self):
        if not self.is_valid:
            return None
        return float(self.camera_matrix[0, 0])

    def horizontal_fov_deg(self):
        """Field of view, used to replace the 70 degree assumption in ego-motion."""
        f = self.focal_length_px()
        if not f or not self.image_size:
            return None
        width = float(self.image_size[0])
        return float(np.degrees(2.0 * np.arctan((width * 0.5) / f)))

    def intrinsics(self):
        """Plain dict, or None when uncalibrated."""
        if not self.is_valid:
            return None
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "image_size": list(self.image_size) if self.image_size else None,
            "focal_length_px": round(self.focal_length_px(), 2),
            "horizontal_fov_deg": (round(self.horizontal_fov_deg(), 2)
                                   if self.horizontal_fov_deg() else None),
            "reprojection_error": (round(self.reprojection_error, 4)
                                   if self.reprojection_error is not None else None),
        }

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path=CALIB_PATH):
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"intrinsics": self.intrinsics()}
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def load(cls, path=CALIB_PATH):
        """Load calibration if present and valid. Never raises."""
        if not os.path.exists(path):
            return cls()
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
            node = payload.get("intrinsics") or {}
            if not isinstance(node, dict):
                return cls()
            matrix = node.get("camera_matrix")
            if not matrix:
                return cls()
            calib = cls(
                camera_matrix=np.array(matrix, dtype=np.float64),
                dist_coeffs=node.get("dist_coeffs"),
                image_size=node.get("image_size"),
                reprojection_error=node.get("reprojection_error"),
            )
            if not calib.is_valid:
                print("[FCW] calibration.yaml present but invalid; ignoring it.")
                return cls()
            print(f"[FCW] Loaded camera calibration from {path} "
                  f"(f={calib.focal_length_px():.1f}px, "
                  f"FOV={calib.horizontal_fov_deg():.1f} deg)")
            return calib
        except Exception as exc:
            print(f"[FCW] Could not read {path}: {exc}")
            return cls()

    # ── Solving ──────────────────────────────────────────────────────
    @classmethod
    def calibrate_from_images(cls, image_paths, pattern_size=(9, 6),
                              square_size=1.0, show=False):
        """Solve intrinsics from chessboard images.

        `pattern_size` is the number of INNER corners, (columns, rows).
        `square_size` is the real edge length in the same unit as `square_size`;
        only the ratio matters for intrinsics, not the absolute unit.
        """
        if not image_paths:
            raise ValueError("No calibration images supplied")

        obj_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        obj_points *= float(square_size)

        obj_list, img_list = [], []
        corners = (pattern_size[0], pattern_size[1])
        image_size = None

        for path in image_paths:
            image = cv2.imread(path)
            if image is None:
                print(f"[FCW] Skipping unreadable image: {path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])

            found, corners_found = cv2.findChessboardCorners(gray, corners, None)
            if not found:
                print(f"[FCW] No chessboard found in {os.path.basename(path)}")
                continue
            corners_found = cv2.cornerSubPix(
                gray, corners_found, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_list.append(obj_points)
            img_list.append(corners_found)
            if show:
                preview = image.copy()
                cv2.drawChessboardCorners(preview, corners, corners_found, found)
                cv2.imshow("calibration", preview)
                cv2.waitKey(200)

        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if len(obj_list) < 4:
            raise RuntimeError(
                f"Only {len(obj_list)} usable images; need at least 4 for a "
                f"stable solve. Capture more angles, including the edges of the view.")

        rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, image_size, None, None)

        calib = cls(camera_matrix=matrix, dist_coeffs=dist,
                    image_size=image_size, reprojection_error=rms)
        print(f"[FCW] Calibration solved: {image_size[0]}x{image_size[1]}, "
              f"fx={matrix[0, 0]:.2f} fy={matrix[1, 1]:.2f}, "
              f"FOV={calib.horizontal_fov_deg():.1f} deg, RMS={rms:.3f}px")
        return calib

    # ── Image operations ─────────────────────────────────────────────
    def undistort(self, frame, target_size=None):
        """Remove lens distortion when coefficients are known.

        Falls back to the original frame when there is nothing to correct, so
        callers can apply it unconditionally.
        """
        if not self.is_valid or not self.has_distortion:
            return frame
        h, w = frame.shape[:2]
        size = target_size or (w, h)
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        return cv2.undistort(frame, self.dist_coeffs, new_matrix,
                             new_camera_matrix=new_matrix)

    def project_point(self, point_3d, rvec=None, tvec=None):
        """Project a metric 3D point to pixels. Requires a calibration."""
        if not self.is_valid:
            return None
        pts = np.array([point_3d], dtype=np.float64).reshape(1, 1, 3)
        if rvec is None:
            rvec = np.zeros(3)
        if tvec is None:
            tvec = np.zeros(3)
        projected, _ = cv2.projectPoints(pts, rvec, tvec, self.camera_matrix,
                                         self.dist_coeffs)
        return tuple(float(v) for v in projected.reshape(2))


def estimate_fov_from_aspect(width, height, horizontal_fov_deg=None):
    """Fallback FOV estimate when calibration is unavailable.

    Used only to document the ego-motion assumption. Without intrinsics the
    system assumes a typical dashcam FOV, which is an assumption, not a
    measurement.
    """
    if horizontal_fov_deg:
        return float(horizontal_fov_deg)
    return 70.0