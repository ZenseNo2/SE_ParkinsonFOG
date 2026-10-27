#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pose.py - Tahap 2: deteksi orang (YOLOX) + estimasi pose multi-person
(RTMPose, Halpe-26) pada setiap frame ke-`stride` (default: setiap frame kedua).

Keluaran: satu file <video_id>.npz per video di --out, berisi
  det_frame       (K,)       int32    indeks frame tiap deteksi
  det_boxes       (K, 4)     float32  x1, y1, x2, y2 (piksel asli)
  det_scores      (K,)       float32  skor deteksi YOLOX
  det_kpts        (K, 26, 2) float32  keypoint Halpe-26 (x, y); NaN jika pose tidak dihitung
  det_kpt_scores  (K, 26)    float32  skor keypoint
  frames_processed(F,)       int32    SEMUA frame yang diproses (termasuk yang tanpa deteksi)
  meta            ()         str      JSON: fps, ukuran frame, stride, model, ambang, dll.
Urutan deteksi di dalam satu frame = urutan skor menurun. Format ini dibaca
langsung oleh label_patient.py (--det-cache) dan oleh pelacak di Tahap 3.

Contoh:
  python run_pose.py --videos videos --out cache/det --device cuda
  python run_pose.py --videos videos --out cache/det --labels labels/patient_labels.csv \
      --preview-dir cache/preview
  python run_pose.py --out cache/det --labels labels/patient_labels.csv --check-only

Kebutuhan:
  pip install rtmlib onnxruntime opencv-python numpy        (CPU)
  pip install rtmlib onnxruntime-gpu opencv-python numpy    (GPU NVIDIA, lalu --device cuda)
Model diunduh otomatis oleh rtmlib saat pertama kali dijalankan (~100-400 MB).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from rtmlib import YOLOX as _YOLOX, RTMPose as _RTMPose, BodyWithFeet as _BodyWithFeet
except ImportError:          # memungkinkan --check-only tanpa rtmlib
    _YOLOX, _RTMPose, _BodyWithFeet = object, None, None

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
N_KPTS = 26
HALPE26_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]
HALPE26_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (18, 19),
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 19), (12, 19), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 20), (15, 22), (15, 24), (16, 21), (16, 23), (16, 25),
]

# Model Halpe-26 resmi rtmlib (sama dengan BodyWithFeet). Dipakai jika
# BodyWithFeet.MODE tidak tersedia di versi rtmlib yang terpasang.
_OMM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
FALLBACK_MODES = {
    "performance": dict(det=_OMM + "yolox_x_8xb8-300e_humanart-a39d44ed.zip", det_input_size=(640, 640),
                        pose=_OMM + "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
                        pose_input_size=(288, 384)),
    "balanced": dict(det=_OMM + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip", det_input_size=(640, 640),
                     pose=_OMM + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
                     pose_input_size=(192, 256)),
    "lightweight": dict(det=_OMM + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", det_input_size=(416, 416),
                        pose=_OMM + "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
                        pose_input_size=(192, 256)),
}


# ----------------------------------------------------------------------------
# Deteksi: YOLOX yang juga mengembalikan skor
# ----------------------------------------------------------------------------

def nms(boxes, scores, thr):
    """NMS satu kelas (NumPy). Kembalikan indeks yang dipertahankan."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return np.asarray(keep, int)


def decode_yolox(outputs, ratio, input_size, score_thr, nms_thr):
    """Ubah keluaran ONNX YOLOX menjadi (boxes Nx4, scores N) untuk kelas orang.

    Menangani dua jenis model:
      * ONNX dengan NMS di dalamnya (mmdeploy end2end, dipakai rtmlib):
        outputs[0] = dets (1, N, 5) [x1, y1, x2, y2, skor], outputs[1] = label (opsional)
      * ONNX tanpa NMS: outputs[0] = (1, A, 5 + C) [cx, cy, w, h, obj, cls...]
    """
    out = np.asarray(outputs[0])
    if out.shape[-1] == 5:
        dets = out.reshape(-1, 5).astype(np.float32)
        boxes, scores = dets[:, :4] / ratio, dets[:, 4]
        if len(outputs) > 1:
            labels = np.asarray(outputs[1]).reshape(-1)
            if labels.shape[0] == boxes.shape[0]:
                boxes, scores = boxes[labels == 0], scores[labels == 0]
        m = scores >= score_thr
        boxes, scores = boxes[m], scores[m]
    else:
        pred = out.reshape(-1, out.shape[-1]).astype(np.float32).copy()
        grids, strides = [], []
        for s in (8, 16, 32):
            hs, ws = input_size[0] // s, input_size[1] // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), s))
        grids, strides = np.concatenate(grids), np.concatenate(strides)
        pred[:, :2] = (pred[:, :2] + grids) * strides
        pred[:, 2:4] = np.exp(pred[:, 2:4]) * strides
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / ratio
        scores = pred[:, 4] * pred[:, 5]          # obj * P(person), kelas 0
        m = scores >= score_thr
        boxes, scores = boxes[m], scores[m]
        keep = nms(boxes, scores, nms_thr)
        boxes, scores = boxes[keep], scores[keep]
    return boxes.astype(np.float32), scores.astype(np.float32)


class YOLOXWithScores(_YOLOX):
    """YOLOX rtmlib yang mengembalikan (boxes, scores) dengan ambang yang bisa diatur.
    YOLOX bawaan rtmlib membuang skor dan memakai ambang tetap 0,3, padahal
    ByteTrack membutuhkan kotak berskor rendah beserta skornya."""

    def __call__(self, image):
        img, ratio = self.preprocess(image)
        outputs = self.inference(img)
        return decode_yolox(outputs, ratio, self.model_input_size, self.score_thr, self.nms_thr)


def clean_dets(boxes, scores, w, h, max_dets, min_size=4):
    """Potong ke batas frame, buang kotak terlalu kecil, urutkan skor menurun."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    ok = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    boxes, scores = boxes[ok], scores[ok]
    order = np.argsort(-scores, kind="stable")[:max_dets]
    return boxes[order], scores[order]


def build_models(args):
    if _RTMPose is None:
        sys.exit("rtmlib belum terpasang:  pip install rtmlib onnxruntime")
    device = args.device
    if args.backend == "onnxruntime" and device.startswith("cuda"):
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print("[peringatan] onnxruntime-gpu/CUDA tidak tersedia; memakai CPU.")
            device = "cpu"
    modes = getattr(_BodyWithFeet, "MODE", None) or FALLBACK_MODES
    cfg = modes[args.mode]
    det = YOLOXWithScores(cfg["det"], model_input_size=cfg["det_input_size"],
                          nms_thr=args.nms_thr, score_thr=args.det_thr,
                          backend=args.backend, device=device)
    pose = _RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                    to_openpose=False, backend=args.backend, device=device)
    return det, pose, cfg, device


# ----------------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------------

def probe_video(path):
    """(fps, n_frames) - sama dengan label_patient.py: ffprobe dulu, cadangan OpenCV."""
    fps = nb = None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=avg_frame_rate,nb_frames", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=60)
        s = json.loads(r.stdout)["streams"][0]
        num, den = (float(x) for x in s.get("avg_frame_rate", "0/1").split("/"))
        if num > 0 and den > 0:
            fps = num / den
        nb_s = str(s.get("nb_frames", ""))
        if nb_s.isdigit() and int(nb_s) > 0:
            nb = int(nb_s)
    except (FileNotFoundError, subprocess.SubprocessError, KeyError, IndexError,
            ValueError, json.JSONDecodeError):
        pass
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"tidak bisa membuka video {path}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    nb = nb or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0 or fps > 1000:
        print(f"  [peringatan] FPS tidak valid untuk {Path(path).name}, memakai 30.")
        fps = 30.0
    return float(fps), int(nb)


def collect_videos(items):
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*")) if q.suffix.lower() in VIDEO_EXTS]
        elif p.is_file():
            out.append(p)
        else:
            print(f"[peringatan] tidak ditemukan: {p}")
    seen = {}
    for q in out:
        if q.stem in seen:
            sys.exit(f"Nama video ganda: {q} dan {seen[q.stem]}. video_id harus unik.")
        seen[q.stem] = q
    return out


def fmt_time(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ----------------------------------------------------------------------------
# Gambar pratinjau
# ----------------------------------------------------------------------------

def draw_pose(img, boxes, scores, kpts, kscores, gt=None, kpt_thr=0.3):
    out = img.copy()
    for b, s, kp, ks in zip(boxes, scores, kpts, kscores):
        b = b.astype(int)
        col = (0, 200, 255) if s >= 0.5 else (0, 120, 160)
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), col, 2)
        cv2.putText(out, f"{s:.2f}", (b[0] + 3, b[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if np.isnan(kp).any():
            continue
        for a, c in HALPE26_LINKS:
            if ks[a] >= kpt_thr and ks[c] >= kpt_thr:
                cv2.line(out, tuple(kp[a].astype(int)), tuple(kp[c].astype(int)), (255, 255, 0), 2)
        for j, (p, q) in enumerate(zip(kp, ks)):
            if q >= kpt_thr:
                r = 5 if j in (15, 16) else 3          # pergelangan kaki lebih besar
                cv2.circle(out, tuple(p.astype(int)), r, (0, 0, 255) if j in (15, 16) else (0, 255, 0), -1)
    if gt is not None:
        g = np.asarray(gt).astype(int)
        cv2.rectangle(out, (g[0], g[1]), (g[2], g[3]), (255, 0, 255), 2)
        cv2.putText(out, "LABEL", (g[0], max(18, g[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return out


# ----------------------------------------------------------------------------
# Pemrosesan satu video
# ----------------------------------------------------------------------------

def save_npz_atomic(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:          # file handle: numpy tidak menambah ".npz"
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def process_video(path, out_path, det, pose, cfg, args, preview=None):
    """preview: dict frame_idx -> kotak label (atau None) untuk disimpan sebagai JPG."""
    fps, nb = probe_video(path)
    stride = args.stride
    n_target = (nb + stride - 1) // stride if nb > 0 else 0
    preview = preview or {}

    fr_l, bx_l, sc_l, kp_l, ks_l, done = [], [], [], [], [], []
    cap = cv2.VideoCapture(str(path))
    i, W, H = 0, None, None
    t0 = last_print = time.time()
    while True:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                if H is None:
                    H, W = frame.shape[:2]
                boxes, scores = det(frame)
                boxes, scores = clean_dets(boxes, scores, W, H, args.max_dets)
                n = len(boxes)
                kp = np.full((n, N_KPTS, 2), np.nan, np.float32)
                ks = np.zeros((n, N_KPTS), np.float32)
                need = scores >= args.pose_min_score
                if need.any():
                    k, s = pose(frame, bboxes=[b for b in boxes[need]])
                    k, s = np.asarray(k), np.asarray(s)
                    if k.shape[1] != N_KPTS:
                        raise RuntimeError(f"model pose menghasilkan {k.shape[1]} keypoint, bukan 26 (Halpe-26)")
                    kp[need], ks[need] = k, s
                done.append(i)
                if n:
                    fr_l.append(np.full(n, i, np.int32))
                    bx_l.append(boxes)
                    sc_l.append(scores)
                    kp_l.append(kp)
                    ks_l.append(ks)
                if i in preview and args.preview_dir:
                    img = draw_pose(frame, boxes, scores, kp, ks, gt=preview[i])
                    pdir = Path(args.preview_dir)
                    pdir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(pdir / f"{path.stem}_f{i:06d}.jpg"), img)
                now = time.time()
                if now - last_print > 2 or len(done) == n_target:
                    rate = len(done) / max(1e-6, now - t0)
                    eta = (n_target - len(done)) / rate if n_target and rate > 0 else 0
                    print(f"\r  {len(done)}/{n_target or '?'} frame  {rate:5.1f} frame/s  sisa {fmt_time(eta)}   ",
                          end="", flush=True)
                    last_print = now
        i += 1
    cap.release()
    print()
    if not done:
        raise IOError("tidak ada frame yang terbaca")

    cat = lambda lst, shape, dt: np.concatenate(lst).astype(dt) if lst else np.zeros(shape, dt)
    meta = {
        "video": path.name, "video_id": path.stem, "fps": fps, "n_frames_read": i,
        "n_frames_metadata": nb, "width": W, "height": H, "stride": stride,
        "mode": args.mode, "det_model": cfg["det"], "pose_model": cfg["pose"],
        "det_thr": args.det_thr, "nms_thr": args.nms_thr, "max_dets": args.max_dets,
        "pose_min_score": args.pose_min_score, "keypoint_format": "halpe26",
        "keypoint_names": HALPE26_NAMES, "created": datetime.now().isoformat(timespec="seconds"),
    }
    save_npz_atomic(
        out_path,
        det_frame=cat(fr_l, (0,), np.int32),
        det_boxes=cat(bx_l, (0, 4), np.float32),
        det_scores=cat(sc_l, (0,), np.float32),
        det_kpts=cat(kp_l, (0, N_KPTS, 2), np.float32),
        det_kpt_scores=cat(ks_l, (0, N_KPTS), np.float32),
        frames_processed=np.asarray(done, np.int32),
        meta=np.array(json.dumps(meta)),
    )
    n_det = sum(len(x) for x in fr_l)
    return len(done), n_det, time.time() - t0


# ----------------------------------------------------------------------------
# Pemeriksaan terhadap label manual
# ----------------------------------------------------------------------------

def iou_one_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def read_labels(path):
    by_vid = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_vid[r["video_id"]].append(r)
    return by_vid


def label_box(r):
    if str(r.get("x1", "")).strip() == "":
        return None
    return np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])


def check_labels(labels_csv, cache_dir, out_csv, iou_thr=0.5):
    """Recall deteksi pada frame berlabel: apakah pasien terdeteksi sama sekali?
    Juga memastikan indeks frame label cocok dengan frame yang diproses."""
    labels = read_labels(labels_csv)
    rows, missing_cache = [], []
    for vid, lab_rows in sorted(labels.items()):
        p = Path(cache_dir) / f"{vid}.npz"
        if not p.exists():
            missing_cache.append(vid)
            continue
        d = np.load(p)
        fr, bx, sc = d["det_frame"], d["det_boxes"], d["det_scores"]
        processed = set(d["frames_processed"].tolist())
        for r in lab_rows:
            f = int(r["frame_idx"])
            gt = label_box(r)
            m = fr == f
            out = {"video_id": vid, "frame_idx": f, "status": r["status"], "n_det": int(m.sum()),
                   "frame_diproses": int(f in processed), "best_iou": "", "best_score": "", "terdeteksi": ""}
            if gt is not None:
                ious = iou_one_many(gt, bx[m])
                if len(ious):
                    j = int(ious.argmax())
                    out.update(best_iou=f"{ious[j]:.3f}", best_score=f"{sc[m][j]:.3f}")
                    out["terdeteksi"] = int(ious[j] >= iou_thr)
                else:
                    out.update(best_iou="0.000", terdeteksi=0)
            rows.append(out)

    if rows:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    vis = [r for r in rows if r["terdeteksi"] != ""]
    not_proc = [r for r in rows if not r["frame_diproses"]]
    print("\n=== Pemeriksaan deteksi vs label manual ===")
    if missing_cache:
        print(f"Video berlabel tanpa cache  : {len(missing_cache)} ({', '.join(missing_cache[:5])}"
              f"{', ...' if len(missing_cache) > 5 else ''})")
    if not vis:
        print("Belum ada frame berlabel yang bisa diperiksa.")
        return
    rec = np.mean([r["terdeteksi"] for r in vis])
    miou = np.mean([float(r["best_iou"]) for r in vis])
    print(f"Frame berlabel dengan kotak : {len(vis)}")
    print(f"Recall deteksi (IoU>={iou_thr}) : {rec:.1%}")
    print(f"Rata-rata IoU terbaik       : {miou:.3f}")
    for st in ("terlihat", "terhalang_sebagian"):
        sub = [r["terdeteksi"] for r in vis if r["status"] == st]
        if sub:
            print(f"  status {st:<20}: {np.mean(sub):.1%} dari {len(sub)}")
    if not_proc:
        print(f"[PERINGATAN] {len(not_proc)} frame label TIDAK ada di frames_processed. "
              f"Periksa --stride atau perbedaan cara baca video.")
    if miou < 0.3:
        print("[PERINGATAN] IoU sangat rendah secara umum: kemungkinan indeks frame atau orientasi "
              "video berbeda antara alat label dan skrip ini.")
    per_vid = defaultdict(list)
    for r in vis:
        per_vid[r["video_id"]].append(r["terdeteksi"])
    worst = sorted(per_vid.items(), key=lambda kv: np.mean(kv[1]))[:5]
    print("Video dengan recall terendah: " +
          ", ".join(f"{v} ({np.mean(x):.0%})" for v, x in worst))
    print(f"Detail per frame: {out_csv}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Tahap 2: YOLOX + RTMPose (Halpe-26) pada setiap frame ke-stride.")
    ap.add_argument("--videos", nargs="+", help="folder video dan/atau file video")
    ap.add_argument("--out", default="cache/det", help="folder keluaran .npz")
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                    help="lightweight=YOLOX-tiny+RTMPose-s, balanced=YOLOX-m+RTMPose-m, "
                         "performance=YOLOX-x+RTMPose-x")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | mps")
    ap.add_argument("--backend", default="onnxruntime", choices=["onnxruntime", "openvino", "opencv"])
    ap.add_argument("--stride", type=int, default=2, help="proses setiap frame ke-N (default 2)")
    ap.add_argument("--det-thr", type=float, default=0.1,
                    help="ambang skor deteksi yang DISIMPAN (rendah agar ByteTrack bisa memakai kotak lemah)")
    ap.add_argument("--nms-thr", type=float, default=0.45)
    ap.add_argument("--max-dets", type=int, default=30, help="maksimum kotak per frame")
    ap.add_argument("--pose-min-score", type=float, default=None,
                    help="hitung pose hanya untuk kotak dengan skor >= nilai ini (default = --det-thr, semua kotak)")
    ap.add_argument("--labels", default=None, help="CSV label manual (untuk pratinjau dan pemeriksaan recall)")
    ap.add_argument("--preview-dir", default=None,
                    help="simpan JPG pratinjau pose di frame berlabel (atau frame tengah jika tanpa label)")
    ap.add_argument("--only", default=None, help="hanya video yang namanya mengandung teks ini")
    ap.add_argument("--overwrite", action="store_true", help="proses ulang video yang cache-nya sudah ada")
    ap.add_argument("--check-only", action="store_true", help="hanya jalankan pemeriksaan label vs cache")
    args = ap.parse_args()
    if args.pose_min_score is None:
        args.pose_min_score = args.det_thr
    out_dir = Path(args.out)

    if args.check_only:
        if not args.labels:
            sys.exit("--check-only membutuhkan --labels")
        check_labels(args.labels, out_dir, out_dir / "det_recall.csv")
        return

    if not args.videos:
        sys.exit("--videos wajib diisi (kecuali dengan --check-only)")
    videos = collect_videos(args.videos)
    if args.only:
        videos = [v for v in videos if args.only in v.stem]
    if not videos:
        sys.exit("Tidak ada video.")
    labels = read_labels(args.labels) if args.labels else {}

    todo = [v for v in videos if args.overwrite or not (out_dir / f"{v.stem}.npz").exists()]
    print(f"{len(videos)} video, {len(videos) - len(todo)} sudah punya cache, {len(todo)} akan diproses.")
    if todo:
        det, pose, cfg, device = build_models(args)
        print(f"Mode {args.mode} | device {device} | stride {args.stride} | det_thr {args.det_thr}")

    t_all = time.time()
    try:
        for n, path in enumerate(todo, 1):
            print(f"[{n}/{len(todo)}] {path.name}")
            preview = {}
            if args.preview_dir:
                rows = labels.get(path.stem, [])
                if rows:
                    preview = {int(r["frame_idx"]): label_box(r) for r in rows}
                else:
                    _, nb = probe_video(path)
                    preview = {(nb // 2 // args.stride) * args.stride: None}
            try:
                nf, nd, dt = process_video(path, out_dir / f"{path.stem}.npz", det, pose, cfg, args, preview)
                print(f"  selesai: {nf} frame, {nd} deteksi ({nd / max(nf, 1):.1f}/frame), {fmt_time(dt)}")
            except Exception as e:  # noqa: BLE001
                print(f"\n  [GAGAL] {e.__class__.__name__}: {e}")
    except KeyboardInterrupt:
        print("\nDihentikan. Video yang sedang diproses tidak disimpan; jalankan ulang untuk melanjutkan.")
    print(f"Total waktu: {fmt_time(time.time() - t_all)}")

    if args.labels:
        check_labels(args.labels, out_dir, out_dir / "det_recall.csv")


if __name__ == "__main__":
    main()
