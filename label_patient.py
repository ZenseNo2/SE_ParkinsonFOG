#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_patient.py - Alat bantu pelabelan manual identitas pasien (Tahap 1, FOG Parkinson)

Untuk setiap video, alat ini mengambil N cuplikan frame pada waktu tetap
    t_k = k * T / (N + 1),  k = 1..N
dibulatkan ke kelipatan --stride (default 2, supaya sama dengan frame yang
diproses deteksi di Tahap 2). Labeler lalu menandai kotak mana yang pasien.

Dua mode (otomatis):
  * Dengan cache deteksi (--det-cache): kotak hasil deteksi ditampilkan dan
    labeler cukup mengklik kotak pasien (atau menekan angka 1-9).
  * Tanpa cache: labeler menggambar kotak pasien dengan klik-tarik.
  Klik-tarik selalu bisa dipakai untuk kotak manual, misalnya saat pasien
  tidak terdeteksi. Tepi/sudut kotak pasien bisa diseret untuk merapikan.

Konfirmasi kotak:
  Setiap label punya kolom `verifikasi` (belum / sesuai). Kotak dianggap
  sesuai bila memuat seluruh tubuh pasien (kepala sampai kaki) dan tidak
  ikut memuat observer. Konfirmasi bisa per cuplikan (y, atau z untuk cek
  perbesaran) atau sekaligus di layar tinjauan (g, lalu y). Mengubah kotak
  mengembalikan verifikasi ke "belum".
  Status video: belum -> sebagian -> lengkap (semua tersimpan) -> terverifikasi.

Format cache deteksi (satu file per video: <video_id>.npz):
  det_frame  (K,)    int    indeks frame tiap deteksi
  det_boxes  (K, 4)  float  x1, y1, x2, y2 dalam piksel asli
  det_scores (K,)    float  skor deteksi (opsional)
  det_kpts / det_kpt_scores (opsional, dari Tahap 2) dipakai di layar cek.
  Kolom det_idx di CSV = urutan deteksi DI DALAM frame tersebut (0-based),
  mengikuti urutan kemunculannya di file .npz.

Keluaran: CSV, satu baris per (video, frame). File disimpan ulang setiap kali
satu label diterima, jadi aman dihentikan kapan saja lalu dilanjutkan.

Tombol - cuplikan:
  klik kiri          pilih kotak deteksi di bawah kursor
  klik-tarik kiri    gambar kotak manual
  seret tepi kotak   rapikan tepi/sudut kotak pasien
  1-9                pilih kotak deteksi bernomor
  Enter / Spasi      simpan label, lanjut ke cuplikan berikutnya
  y                  simpan + konfirmasi kotak SESUAI, lanjut
  z                  cek perbesaran kotak (y = sesuai, t = tidak sesuai)
  Backspace / b      kembali ke cuplikan sebelumnya
  n                  tandai "tidak_ada" (pasien tidak terlihat di frame)
  o                  ubah status terlihat <-> terhalang_sebagian
  x                  hapus pilihan di cuplikan ini
  v                  putar klip +/- --clip-s detik di sekitar frame
  c                  tulis catatan (diketik di terminal)
  g                  tinjau & konfirmasi semua potongan pasien video ini
Tombol - video:
  ] / PageDown       video berikutnya
  [ / PageUp         video sebelumnya
  u                  video berikutnya yang belum terverifikasi
  k                  lompat ke video (nomor atau nama, diketik di terminal)
  i                  tampilkan/sembunyikan daftar video & statusnya
  h                  tampilkan/sembunyikan bantuan
  q / Esc            keluar (label yang sudah disimpan tetap aman)

Contoh:
  python label_patient.py --videos data/videos --out data/labels/patient_labels.csv
  python label_patient.py --videos data/videos --out data/labels/patient_labels.csv \
      --det-cache outputs/02_pose/balanced_s2_det0.1/npz --min-score 0.3
  python label_patient.py ... --start 12          # mulai di video ke-12
  python label_patient.py ... --start pasien05    # mulai di video bernama pasien05

Kebutuhan: pip install opencv-python numpy   (ffprobe opsional, untuk FPS akurat)
"""

import argparse
import csv
import getpass
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
STATUSES = ("terlihat", "terhalang_sebagian", "tidak_ada")
FIELDS = ["video_id", "frame_idx", "t_sec", "fps", "width", "height",
          "x1", "y1", "x2", "y2", "status", "sumber", "det_idx",
          "labeler", "waktu_label", "catatan", "verifikasi"]
WIN = "Label pasien"

C_DET = (255, 200, 0)      # kotak deteksi (BGR)
C_SEL = (0, 220, 0)        # kotak pasien terpilih
C_MAN = (255, 0, 255)      # kotak manual
C_WARN = (0, 140, 255)
C_SAVED = (0, 200, 255)    # tersimpan, belum dikonfirmasi
C_BAD = (60, 60, 255)

KPT_THR = 0.3              # skor keypoint minimum di layar cek
HANDLE_PX = 9              # jarak (piksel layar) untuk menyeret tepi kotak

STATE_COL = {"belum": (150, 150, 150), "sebagian": C_WARN,
             "lengkap": C_SAVED, "terverifikasi": C_SEL}


# ----------------------------------------------------------------------------
# Tombol (waitKeyEx berbeda antar OS/backend)
# ----------------------------------------------------------------------------

K_PGUP = {65365, 2162688, 16777238, 63276}
K_PGDN = {65366, 2228224, 16777239, 63277}
K_BACK = {8, 127, 65288, 16777219}
K_ENTER = {10, 13, 65293, 65421, 16777220, 16777221}
K_ESC = {27, 65307, 16777216}


def get_key(delay):
    """Kembalikan None, 'pgup', 'pgdn', 'back', 'enter', 'esc', atau karakter."""
    k = cv2.waitKeyEx(delay)
    if k == -1:
        return None
    if k in K_PGUP:
        return "pgup"
    if k in K_PGDN:
        return "pgdn"
    if k in K_BACK:
        return "back"
    if k in K_ENTER:
        return "enter"
    if k in K_ESC:
        return "esc"
    if 32 <= k < 127:
        return chr(k)
    return "?"


def window_closed():
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# ----------------------------------------------------------------------------
# Utilitas video
# ----------------------------------------------------------------------------

def probe_video(path):
    """Kembalikan (fps, n_frames). Utamakan ffprobe, cadangan OpenCV."""
    fps = nb = None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,nb_frames",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        s = json.loads(r.stdout)["streams"][0]
        num, den = (float(x) for x in s.get("avg_frame_rate", "0/1").split("/"))
        if num > 0 and den > 0:
            fps = num / den
        nb_s = str(s.get("nb_frames", ""))
        if nb_s.isdigit() and int(nb_s) > 0:
            nb = int(nb_s)
    except (FileNotFoundError, subprocess.SubprocessError, KeyError,
            IndexError, ValueError, json.JSONDecodeError):
        pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"tidak bisa membuka video {path}")
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if nb is None:
        nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0 or fps > 1000:
        print(f"  [peringatan] FPS tidak valid untuk {Path(path).name}, memakai 30.")
        fps = 30.0
    return float(fps), int(nb)


def select_frames(n_total, n_samples, stride):
    """Indeks frame di t_k = k*T/(N+1), dibulatkan ke kelipatan stride."""
    if n_total <= 0:
        return []
    last = ((n_total - 1) // stride) * stride
    out = []
    for k in range(1, n_samples + 1):
        f = int(round(k * n_total / (n_samples + 1) / stride)) * stride
        f = min(max(f, 0), last)
        if f not in out:
            out.append(f)
    return out


def read_frames(path, targets):
    """Baca frame target secara berurutan (akurat, tidak bergantung seek).
    Kembalikan (dict frame->gambar, jumlah frame yang terbaca)."""
    want = sorted(set(targets))
    got = {}
    cap = cv2.VideoCapture(str(path))
    i = j = 0
    while j < len(want):
        if not cap.grab():
            break
        if i == want[j]:
            ok, fr = cap.retrieve()
            if ok:
                got[i] = fr
            j += 1
        i += 1
    cap.release()
    return got, i


def load_video_frames(path, n_samples, stride):
    fps, nb = probe_video(path)
    targets = select_frames(nb, n_samples, stride)
    got, n_read = read_frames(path, targets)
    if len(got) < len(targets) and 0 < n_read < nb:
        # Metadata melebihkan jumlah frame; pilih ulang dengan jumlah sebenarnya.
        print(f"  [info] jumlah frame sebenarnya {n_read} (metadata {nb}); memilih ulang cuplikan.")
        targets = select_frames(n_read, n_samples, stride)
        got, _ = read_frames(path, targets)
    return fps, [(f, got[f]) for f in targets if f in got]


def load_detections(cache_dir, video_id, frame_ids):
    """Kembalikan dict frame -> (boxes Kx4, scores K, kpts Kx26x2|None, kscores Kx26|None),
    atau None jika tidak ada cache."""
    if cache_dir is None:
        return None
    p = Path(cache_dir) / f"{video_id}.npz"
    if not p.exists():
        print(f"  [info] cache deteksi {p} tidak ada; mode manual.")
        return None
    d = np.load(p)
    fr = d["det_frame"].astype(int)
    bx = d["det_boxes"].astype(float).reshape(-1, 4)
    sc = d["det_scores"].astype(float) if "det_scores" in d.files else np.ones(len(fr))
    kp = d["det_kpts"].astype(float) if "det_kpts" in d.files else None
    ks = d["det_kpt_scores"].astype(float) if "det_kpt_scores" in d.files else None
    out = {}
    for f in frame_ids:
        m = fr == f
        out[f] = (bx[m], sc[m],
                  kp[m] if kp is not None else None,
                  ks[m] if ks is not None else None)
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------
# Penyimpanan label
# ----------------------------------------------------------------------------

class LabelStore:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows[(r["video_id"], int(r["frame_idx"]))] = r

    def get(self, vid, f):
        return self.rows.get((vid, f))

    def rows_for(self, vid):
        return [r for (v, _), r in self.rows.items() if v == vid]

    def count(self, vid):
        return sum(1 for (v, _) in self.rows if v == vid)

    def put(self, row):
        self.rows[(row["video_id"], int(row["frame_idx"]))] = row
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for key in sorted(self.rows):
                w.writerow(self.rows[key])
        os.replace(tmp, self.path)   # atomik: CSV tidak pernah setengah tertulis


def video_state(store, vid, n_frames):
    rows = store.rows_for(vid)
    if not rows:
        return "belum"
    if len(rows) < n_frames:
        return "sebagian"
    if all(r.get("verifikasi") == "sesuai" for r in rows):
        return "terverifikasi"
    return "lengkap"


def empty_label():
    return {"box": None, "status": "terlihat", "sumber": "", "det_idx": None,
            "catatan": "", "verif": "belum"}


def label_from_row(r):
    box = None
    if str(r.get("x1", "")).strip() != "":
        box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    di = str(r.get("det_idx", "")).strip()
    sumber = r.get("sumber", "")
    return {"box": box,
            "status": r.get("status") or "terlihat",
            "sumber": "" if sumber == "none" else sumber,
            "det_idx": int(di) if di.isdigit() else None,
            "catatan": r.get("catatan", ""),
            "verif": r.get("verifikasi") or "belum"}


# ----------------------------------------------------------------------------
# Gambar
# ----------------------------------------------------------------------------

def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def shade(img, x0, y0, x1, y1, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


HELP_LINES = [
    "klik: pilih kotak | klik-tarik: kotak manual | seret tepi: rapikan | 1-9: kotak bernomor",
    "Enter/Spasi: simpan & lanjut | y: simpan + SESUAI | z: cek perbesaran | Backspace/b: kembali",
    "n: tidak_ada | o: terhalang | x: hapus | v: putar klip | c: catatan | g: tinjau & konfirmasi",
    "] / PgDn: video berikut | [ / PgUp: sebelumnya | u: berikut belum verif | k: lompat | i: daftar",
    "q/Esc: keluar | h: tutup bantuan",
]

SHORT_STATUS = {"terlihat": "ok", "terhalang_sebagian": "terhalang", "tidak_ada": "tidak ada"}


# ----------------------------------------------------------------------------
# Sesi pelabelan satu video
# ----------------------------------------------------------------------------

class VideoSession:
    def __init__(self, path, store, args, vi, videos):
        self.path = Path(path)
        self.vid = self.path.stem
        self.store, self.args, self.vi, self.videos = store, args, vi, videos
        self.fps, self.frames = load_video_frames(self.path, args.n_frames, args.stride)
        if not self.frames:
            raise IOError("tidak ada frame yang terbaca")
        self.h, self.w = self.frames[0][1].shape[:2]
        self.s = min(args.max_w / self.w, args.max_h / self.h, 1.0)
        self.det = load_detections(args.det_cache, self.vid, [f for f, _ in self.frames])

        self.labels, self.saved = [], []
        for f, _ in self.frames:
            r = store.get(self.vid, f)
            self.labels.append(label_from_row(r) if r else empty_label())
            self.saved.append(r is not None)
        self.k = self.first_open()

        self.drag0 = self.drag1 = None
        self.resize = None          # (sisi yang diseret, kotak awal) saat merapikan tepi
        self.show_help = False
        self.show_list = False
        self.msg, self.msg_until = "", 0.0

    # --- status -------------------------------------------------------------
    @property
    def complete(self):
        return all(self.saved)

    @property
    def verified(self):
        return self.complete and all(l["verif"] == "sesuai" for l in self.labels)

    @property
    def lab(self):
        return self.labels[self.k]

    def first_open(self):
        for i, ok in enumerate(self.saved):
            if not ok:
                return i
        for i, l in enumerate(self.labels):
            if l["verif"] != "sesuai":
                return i
        return 0

    def flash(self, text, sec=2.5):
        self.msg, self.msg_until = text, time.time() + sec

    def edited(self, reset_verif=True):
        self.saved[self.k] = False
        if reset_verif:
            self.lab["verif"] = "belum"

    def dets_here(self, k=None):
        """Daftar (box, skor, det_idx asli) di atas --min-score untuk cuplikan k."""
        if self.det is None:
            return []
        k = self.k if k is None else k
        boxes, scores, _, _ = self.det.get(self.frames[k][0], (np.zeros((0, 4)), np.zeros(0), None, None))
        return [(boxes[j], float(scores[j]), j) for j in range(len(boxes))
                if scores[j] >= self.args.min_score]

    def kpts_for(self, k):
        """Keypoint (26x2, skor 26) milik deteksi terpilih di cuplikan k, atau None."""
        lab = self.labels[k]
        if self.det is None or lab["det_idx"] is None or not lab["sumber"].startswith("deteksi"):
            return None
        _, _, kp, ks = self.det.get(self.frames[k][0], (None, None, None, None))
        j = lab["det_idx"]
        if kp is None or ks is None or j >= len(kp) or np.isnan(kp[j]).all():
            return None
        return kp[j], ks[j]

    def checks(self, k=None):
        """Peringatan otomatis untuk kotak pasien di cuplikan k (daftar teks)."""
        k = self.k if k is None else k
        lab, out = self.labels[k], []
        if lab["box"] is None:
            return out
        b = lab["box"]
        hs = [l["box"][3] - l["box"][1] for i, l in enumerate(self.labels)
              if i != k and l["box"] is not None]
        if len(hs) >= 3:
            r = (b[3] - b[1]) / max(1.0, float(np.median(hs)))
            if r > 1.4 or r < 0.7:
                out.append(f"tinggi kotak {r:.1f}x cuplikan lain")
        for bb, _, j in self.dets_here(k):
            if lab["sumber"].startswith("deteksi") and j == lab["det_idx"]:
                continue
            if iou(b, bb) > 0.3:
                out.append("tumpang tindih dengan orang lain")
                break
        kk = self.kpts_for(k)
        if kk is not None:
            kp, ks = kk
            ok = ks >= KPT_THR
            outside = ok & ((kp[:, 0] < b[0] - 2) | (kp[:, 0] > b[2] + 2) |
                            (kp[:, 1] < b[1] - 2) | (kp[:, 1] > b[3] + 2))
            if outside.sum():
                out.append(f"{int(outside.sum())} keypoint di luar kotak")
        return out

    # --- interaksi ----------------------------------------------------------
    def set_det(self, box, j):
        self.lab.update(box=tuple(float(v) for v in box), sumber="deteksi", det_idx=int(j))
        if self.lab["status"] == "tidak_ada":
            self.lab["status"] = "terlihat"
        self.edited()

    def click_select(self, x, y):
        hits = [(b, j) for b, _, j in self.dets_here()
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if not hits:
            self.flash("Tidak ada kotak di titik itu. Klik-tarik untuk kotak manual.")
            return
        # Kotak bertumpuk (pasien + pendamping): ambil yang terkecil.
        b, j = min(hits, key=lambda h: (h[0][2] - h[0][0]) * (h[0][3] - h[0][1]))
        self.set_det(b, j)

    def select_number(self, n):
        d = self.dets_here()
        if 1 <= n <= len(d):
            self.set_det(d[n - 1][0], d[n - 1][2])
        else:
            self.flash(f"Tidak ada kotak nomor {n}.")

    def edges_at(self, x, y):
        """Sisi kotak pasien (layar) yang dekat titik (x, y): subset 'lrtb'."""
        if self.lab["box"] is None:
            return ""
        p = [v * self.s for v in self.lab["box"]]
        t = HANDLE_PX
        inside_x = p[0] - t <= x <= p[2] + t
        inside_y = p[1] - t <= y <= p[3] + t
        e = ""
        if inside_y and abs(x - p[0]) <= t:
            e += "l"
        elif inside_y and abs(x - p[2]) <= t:
            e += "r"
        if inside_x and abs(y - p[1]) <= t:
            e += "t"
        elif inside_x and abs(y - p[3]) <= t:
            e += "b"
        return e

    def on_mouse(self, ev, x, y, flags, _param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            e = self.edges_at(x, y)
            if e:
                self.resize = (e, self.lab["box"], (x, y))
            else:
                self.drag0 = self.drag1 = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.resize is not None:
            e, b0, _ = self.resize
            x1, y1, x2, y2 = b0
            X = min(max(x / self.s, 0.0), float(self.w - 1))
            Y = min(max(y / self.s, 0.0), float(self.h - 1))
            if "l" in e:
                x1 = min(X, x2 - 5)
            if "r" in e:
                x2 = max(X, x1 + 5)
            if "t" in e:
                y1 = min(Y, y2 - 5)
            if "b" in e:
                y2 = max(Y, y1 + 5)
            self.lab["box"] = (x1, y1, x2, y2)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag0 is not None:
            self.drag1 = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.resize is not None:
            e, b0, (x0, y0) = self.resize
            self.resize = None
            if abs(x - x0) < 3 and abs(y - y0) < 3:      # hanya klik, bukan seret
                self.lab["box"] = b0
                self.click_select(x / self.s, y / self.s)
                return
            if self.lab["sumber"] == "deteksi":
                self.lab["sumber"] = "deteksi_disesuaikan"
            self.edited()
        elif ev == cv2.EVENT_LBUTTONUP and self.drag0 is not None:
            x0, y0 = self.drag0
            self.drag0 = self.drag1 = None
            if abs(x - x0) < 6 and abs(y - y0) < 6:
                self.click_select(x / self.s, y / self.s)
                return
            x1, x2 = sorted((x0 / self.s, x / self.s))
            y1, y2 = sorted((y0 / self.s, y / self.s))
            x1, x2 = max(0.0, x1), min(float(self.w - 1), x2)
            y1, y2 = max(0.0, y1), min(float(self.h - 1), y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                return
            self.lab.update(box=(x1, y1, x2, y2), sumber="manual", det_idx=None)
            if self.lab["status"] == "tidak_ada":
                self.lab["status"] = "terlihat"
            self.edited()

    def toggle_absent(self):
        if self.lab["status"] == "tidak_ada":
            self.lab["status"] = "terlihat"
        else:
            self.lab.update(status="tidak_ada", box=None, sumber="", det_idx=None)
        self.edited()

    def toggle_occluded(self):
        if self.lab["box"] is None:
            self.flash("Pilih kotak dulu.")
            return
        self.lab["status"] = ("terhalang_sebagian" if self.lab["status"] == "terlihat"
                              else "terlihat")
        self.edited()

    def clear(self):
        note = self.lab["catatan"]
        self.labels[self.k] = empty_label()
        self.lab["catatan"] = note
        self.edited()

    def add_note(self):
        f = self.frames[self.k][0]
        print(f"\nCatatan untuk {self.vid} frame {f} (Enter kosong = hapus catatan):")
        try:
            txt = input("> ").strip().replace("\n", " ")
        except EOFError:
            txt = ""
        self.lab["catatan"] = txt
        self.edited(reset_verif=False)
        self.flash("Catatan diperbarui. Tekan Enter untuk menyimpan.")

    def accept(self, k=None, quiet=False):
        k = self.k if k is None else k
        lab = self.labels[k]
        if lab["box"] is None and lab["status"] != "tidak_ada":
            if not quiet:
                self.flash("Pilih/gambar kotak pasien dulu, atau tekan n jika tidak terlihat.")
            return False
        f = self.frames[k][0]
        box = [f"{v:.1f}" for v in lab["box"]] if lab["box"] else ["", "", "", ""]
        old = self.store.get(self.vid, f)
        unchanged = self.saved[k] and old is not None
        row = {
            "video_id": self.vid, "frame_idx": f, "t_sec": f"{f / self.fps:.3f}",
            "fps": f"{self.fps:.4f}", "width": self.w, "height": self.h,
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "status": lab["status"],
            "sumber": lab["sumber"] or "none",
            "det_idx": "" if lab["det_idx"] is None else lab["det_idx"],
            "labeler": (old or {}).get("labeler", self.args.labeler) if unchanged else self.args.labeler,
            "waktu_label": (old or {}).get("waktu_label") if unchanged
            else datetime.now().isoformat(timespec="seconds"),
            "catatan": lab["catatan"],
            "verifikasi": lab["verif"],
        }
        try:
            self.store.put(row)
        except OSError as e:
            self.flash(f"GAGAL menyimpan ({e.__class__.__name__}). CSV sedang dibuka program lain?", 4)
            return False
        self.saved[k] = True
        return True

    def confirm_all(self):
        """Tandai semua cuplikan tersimpan sebagai SESUAI. Kembalikan True jika semua terverifikasi."""
        for k in range(len(self.frames)):
            if self.saved[k] and self.labels[k]["verif"] != "sesuai":
                self.labels[k]["verif"] = "sesuai"
                self.saved[k] = False
                self.accept(k, quiet=True)
        return self.verified

    def advance(self):
        """Lanjut ke cuplikan berikutnya; di akhir buka layar tinjauan."""
        if self.k < len(self.frames) - 1:
            self.k += 1
            return None
        return self.review()

    # --- tampilan -----------------------------------------------------------
    def scaled(self, img):
        if self.s >= 1.0:
            return img.copy()
        return cv2.resize(img, (int(self.w * self.s), int(self.h * self.s)),
                          interpolation=cv2.INTER_AREA)

    def draw_list(self, disp):
        H, W = disp.shape[:2]
        n = len(self.videos)
        rows = max(5, min(n, (H - 140) // 22))
        lo = max(0, min(self.vi - rows // 2, n - rows))
        x0 = max(0, W - 430)
        disp = shade(disp, x0, 84, W, 84 + 22 * min(rows, n) + 40, 0.8)
        put_text(disp, "Daftar video (k: lompat, i: tutup)", (x0 + 10, 104), 0.5, (0, 255, 255))
        for r, i in enumerate(range(lo, min(n, lo + rows))):
            st = video_state(self.store, self.videos[i].stem, self.args.n_frames)
            cnt = self.store.count(self.videos[i].stem)
            mark = ">" if i == self.vi else " "
            put_text(disp, f"{mark}{i + 1:3d} {self.videos[i].stem[:22]:<22} {cnt}/{self.args.n_frames} {st}",
                     (x0 + 10, 128 + 22 * r), 0.48, STATE_COL[st], 2 if i == self.vi else 1)
        return disp

    def render(self):
        f, img = self.frames[self.k]
        disp = self.scaled(img)
        s, lab = self.s, self.lab
        dets = self.dets_here()

        for n, (b, sc, j) in enumerate(dets, start=1):
            p = (np.asarray(b) * s).astype(int)
            chosen = lab["sumber"].startswith("deteksi") and lab["det_idx"] == j
            col = C_SEL if chosen else C_DET
            cv2.rectangle(disp, (p[0], p[1]), (p[2], p[3]), col, 2 if chosen else 1)
            put_text(disp, f"{n} ({sc:.2f})", (p[0] + 3, p[1] + 18), 0.5, col)

        if lab["box"] is not None:
            p = (np.asarray(lab["box"]) * s).astype(int)
            col = C_MAN if lab["sumber"] == "manual" else C_SEL
            cv2.rectangle(disp, (p[0], p[1]), (p[2], p[3]), col, 3)
            for cx, cy in ((p[0], p[1]), (p[2], p[1]), (p[0], p[3]), (p[2], p[3]),
                           ((p[0] + p[2]) // 2, p[1]), ((p[0] + p[2]) // 2, p[3]),
                           (p[0], (p[1] + p[3]) // 2), (p[2], (p[1] + p[3]) // 2)):
                cv2.rectangle(disp, (cx - 4, cy - 4), (cx + 4, cy + 4), col, -1)
            tag = {"manual": "PASIEN (manual)", "deteksi_disesuaikan": "PASIEN (dirapikan)"}.get(
                lab["sumber"], "PASIEN")
            if lab["verif"] == "sesuai":
                tag += " - SESUAI"
            put_text(disp, tag, (p[0], max(16, p[1] - 8)), 0.6, col, 2)

        if self.drag0 is not None and self.drag1 is not None:
            cv2.rectangle(disp, self.drag0, self.drag1, C_MAN, 1)

        # Header
        H, W = disp.shape[:2]
        disp = shade(disp, 0, 0, W, 84, 0.6)
        vstate = video_state(self.store, self.vid, self.args.n_frames)
        put_text(disp, f"{self.vid}  [video {self.vi + 1}/{len(self.videos)}: {vstate}]   cuplikan "
                       f"{self.k + 1}/{len(self.frames)}  frame {f}  t={f / self.fps:.2f}s",
                 (8, 22), 0.55, STATE_COL[vstate] if vstate == "terverifikasi" else (255, 255, 255))
        mode = "mode manual (klik-tarik)" if self.det is None else f"{len(dets)} deteksi"
        if not self.saved[self.k]:
            state, scol = "belum disimpan", C_WARN
        elif lab["verif"] == "sesuai":
            state, scol = "TERSIMPAN, kotak SESUAI", C_SEL
        else:
            state, scol = "tersimpan, belum dikonfirmasi (y / z)", C_SAVED
        put_text(disp, f"status: {lab['status']}  |  {mode}  |  {state}", (8, 46), 0.5, scol)
        for i, ok in enumerate(self.saved):           # titik progres
            c = (14 + 18 * i, 68)
            col = (90, 90, 90) if not ok else (C_SEL if self.labels[i]["verif"] == "sesuai" else C_SAVED)
            cv2.circle(disp, c, 6, col, -1)
            if i == self.k:
                cv2.circle(disp, c, 8, (255, 255, 255), 1)
        warn = self.checks()
        if warn:
            put_text(disp, "periksa: " + "; ".join(warn), (30 + 18 * len(self.saved), 73), 0.5, C_BAD, 1)

        if lab["status"] == "tidak_ada":
            put_text(disp, "PASIEN TIDAK ADA DI FRAME", (W // 2 - 170, H // 2), 0.8, (0, 0, 255), 2)
        if lab["catatan"]:
            put_text(disp, f"catatan: {lab['catatan'][:80]}", (8, 104), 0.5, (200, 255, 255))

        if self.show_list:
            disp = self.draw_list(disp)

        # Bantuan / pesan
        if self.show_help:
            y0 = H - 22 * len(HELP_LINES) - 10
            disp = shade(disp, 0, y0 - 18, W, H, 0.7)
            for i, line in enumerate(HELP_LINES):
                put_text(disp, line, (8, y0 + 22 * i), 0.5)
        else:
            put_text(disp, "h: bantuan   y: kotak sesuai   z: cek   ] [: ganti video   i: daftar",
                     (8, H - 10), 0.45, (220, 220, 220))
        if time.time() < self.msg_until:
            put_text(disp, self.msg, (8, H - 40), 0.6, (0, 255, 255), 2)
        return disp

    def zoom_check(self):
        """Perbesar kotak pasien untuk memastikan kotak pas. Kembalikan 'y', 't', atau None."""
        lab = self.lab
        if lab["box"] is None:
            self.flash("Belum ada kotak untuk dicek.")
            return None
        f, img = self.frames[self.k]
        x1, y1, x2, y2 = lab["box"]
        bw, bh = x2 - x1, y2 - y1
        ex1, ey1, ex2, ey2 = x1 - 0.35 * bw, y1 - 0.12 * bh, x2 + 0.35 * bw, y2 + 0.12 * bh
        kk = self.kpts_for(self.k)
        if kk is not None:                      # perluas agar keypoint di luar kotak ikut terlihat
            ok = (kk[1] >= KPT_THR) & ~np.isnan(kk[0][:, 0])
            if ok.any():
                ex1, ey1 = min(ex1, kk[0][ok, 0].min() - 20), min(ey1, kk[0][ok, 1].min() - 20)
                ex2, ey2 = max(ex2, kk[0][ok, 0].max() + 20), max(ey2, kk[0][ok, 1].max() + 20)
        a, b = max(0, int(ex1)), max(0, int(ey1))
        c, d = min(self.w, int(ex2)), min(self.h, int(ey2))
        crop = img[b:d, a:c]
        avail_h, avail_w = self.args.max_h - 110, self.args.max_w
        sc = min(avail_h / crop.shape[0], avail_w / crop.shape[1], 4.0)
        view = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))),
                          interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
        T = lambda X, Y: (int((X - a) * sc), int((Y - b) * sc))   # noqa: E731

        # Orang lain di sekitar: kotak tipis oranye, untuk melihat apakah ikut termuat.
        for bb, _, j in self.dets_here():
            if lab["sumber"].startswith("deteksi") and j == lab["det_idx"]:
                continue
            if iou(lab["box"], bb) > 0.01:
                cv2.rectangle(view, T(bb[0], bb[1]), T(bb[2], bb[3]), C_WARN, 1)
        cv2.rectangle(view, T(x1, y1), T(x2, y2), C_SEL, 2)
        n_out = 0
        if kk is not None:
            kp, ks = kk
            for (X, Y), q in zip(kp, ks):
                if q < KPT_THR or np.isnan(X):
                    continue
                inside = x1 - 2 <= X <= x2 + 2 and y1 - 2 <= Y <= y2 + 2
                n_out += not inside
                cv2.circle(view, T(X, Y), 4 if inside else 6, (0, 255, 255) if inside else C_BAD, -1)
                cv2.circle(view, T(X, Y), 5 if inside else 8, (0, 0, 0) if inside else (255, 255, 255), 1)

        top = np.full((100, max(view.shape[1], 760), 3), 25, np.uint8)
        put_text(top, f"CEK KOTAK  {self.vid}  cuplikan {self.k + 1}/{len(self.frames)}  (perbesaran {sc:.1f}x)",
                 (10, 24), 0.6, (0, 255, 255), 2)
        put_text(top, "Sesuai = memuat seluruh tubuh pasien (kepala s.d. kaki) dan TIDAK memuat observer.",
                 (10, 50), 0.5)
        info = []
        if kk is not None:
            info.append(f"titik kuning = keypoint di dalam, merah = di luar kotak ({n_out})")
        info += self.checks()
        put_text(top, " | ".join(info) if info else "tidak ada peringatan otomatis", (10, 72), 0.48,
                 C_BAD if self.checks() else (200, 200, 200))
        put_text(top, "y: SESUAI   t: tidak sesuai (perbaiki)   z / Esc: kembali", (10, 94), 0.5, C_SEL)
        canvas = np.full((top.shape[0] + view.shape[0], top.shape[1], 3), 25, np.uint8)
        canvas[:top.shape[0]] = top
        canvas[top.shape[0]:, :view.shape[1]] = view

        while True:
            cv2.imshow(WIN, canvas)
            key = get_key(50)
            if key is None:
                if window_closed():
                    return None
                continue
            if key == "y":
                return "y"
            if key == "t":
                return "t"
            if key in ("z", "esc", "back", "q", "enter"):
                return None

    def play_clip(self):
        """Putar klip di sekitar cuplikan untuk konteks (strap IMU, tugas berputar).
        Seek OpenCV bisa meleset beberapa frame; klip ini hanya untuk konteks."""
        f0 = self.frames[self.k][0]
        half = int(round(self.args.clip_s * self.fps))
        start, end = max(0, f0 - half), f0 + half
        delay = max(1, int(1000 / self.fps))
        cap = cv2.VideoCapture(str(self.path))
        stop = False
        while not stop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            f, n_read = start, 0
            while f <= end:
                ok, fr = cap.read()
                if not ok:
                    break
                n_read += 1
                disp = self.scaled(fr)
                if self.lab["box"] is not None:
                    p = (np.asarray(self.lab["box"]) * self.s).astype(int)
                    cv2.rectangle(disp, (p[0], p[1]), (p[2], p[3]), C_SEL, 3 if f == f0 else 1)
                put_text(disp, f"KLIP  t={(f - f0) / self.fps:+.2f}s  (tombol apa saja = berhenti)",
                         (8, 24), 0.6, (0, 255, 255) if f == f0 else (255, 255, 255), 2)
                cv2.imshow(WIN, disp)
                if cv2.waitKey(delay) != -1:
                    stop = True
                    break
                f += 1
            if n_read == 0:
                break
        cap.release()

    def build_montage(self):
        tw, th, cols = 200, 300, 4
        n = len(self.frames)
        rows = (n + cols - 1) // cols
        canvas = np.full((40 + rows * (th + 48) + 56, cols * tw, 3), 30, np.uint8)
        put_text(canvas, f"{self.vid}: apakah semua kotak memuat orang yang SAMA dan pas?", (8, 26), 0.55)
        for k, (f, img) in enumerate(self.frames):
            r, c = divmod(k, cols)
            x0, y0 = c * tw, 40 + r * (th + 48)
            lab = self.labels[k]
            tile = np.full((th, tw, 3), 50, np.uint8)
            if lab["box"] is not None:
                x1, y1, x2, y2 = lab["box"]
                px, py = 0.15 * (x2 - x1), 0.10 * (y2 - y1)
                a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
                cc, d = min(self.w, int(x2 + px)), min(self.h, int(y2 + py))
                crop = img[b:d, a:cc]
                if crop.size:
                    sc = min(tw / crop.shape[1], th / crop.shape[0])
                    cr = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))))
                    oy, ox = (th - cr.shape[0]) // 2, (tw - cr.shape[1]) // 2
                    tile[oy:oy + cr.shape[0], ox:ox + cr.shape[1]] = cr
                    cv2.rectangle(tile, (ox + int((x1 - a) * sc), oy + int((y1 - b) * sc)),
                                  (ox + int((x2 - a) * sc), oy + int((y2 - b) * sc)), C_SEL, 1)
            else:
                put_text(tile, "tidak ada" if lab["status"] == "tidak_ada" else "belum dilabel",
                         (10, th // 2), 0.55)
            if not self.saved[k]:
                col, txt = C_WARN, "belum disimpan"
            elif lab["verif"] == "sesuai":
                col, txt = C_SEL, "SESUAI"
            else:
                col, txt = C_SAVED, "belum dikonfirmasi"
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), col, 3)
            canvas[y0:y0 + th, x0:x0 + tw] = tile
            put_text(canvas, f"{k + 1}: t={f / self.fps:.1f}s {SHORT_STATUS.get(lab['status'], '')}",
                     (x0 + 6, y0 + th + 18), 0.48, col)
            warn = self.checks(k)
            put_text(canvas, txt + (" | cek!" if warn else ""), (x0 + 6, y0 + th + 38), 0.45,
                     C_BAD if warn else col)
        put_text(canvas, "y / Enter: semua kotak SESUAI -> video berikutnya | 1-9: perbaiki cuplikan",
                 (8, canvas.shape[0] - 32), 0.5, (220, 220, 220))
        put_text(canvas, "b: kembali | ] [: video berikut/sebelum | q: keluar",
                 (8, canvas.shape[0] - 10), 0.5, (220, 220, 220))
        if canvas.shape[1] > self.args.max_w or canvas.shape[0] > self.args.max_h:
            sc = min(self.args.max_w / canvas.shape[1], self.args.max_h / canvas.shape[0])
            canvas = cv2.resize(canvas, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        return canvas

    def review(self):
        """Layar tinjauan. Kembalikan None (kembali menyunting) atau aksi navigasi."""
        while True:
            cv2.imshow(WIN, self.build_montage())
            key = get_key(50)
            if key is None:
                if window_closed():
                    return "quit"
                continue
            if key in ("enter", " ", "y"):
                if not self.complete:
                    self.k = next(i for i, ok in enumerate(self.saved) if not ok)
                    self.flash("Masih ada cuplikan yang belum disimpan.")
                    return None
                self.confirm_all()
                return "done"
            if key.isdigit() and key != "0" and int(key) <= len(self.frames):
                self.k = int(key) - 1
                return None
            if key in ("back", "b", "g", "esc"):
                return None
            if key in ("]", "pgdn", "s"):
                return "next"
            if key in ("[", "pgup"):
                return "prev"
            if key == "q":
                return "quit"

    # --- loop utama ---------------------------------------------------------
    def run(self):
        """Kembalikan 'quit', 'next', 'prev', 'done', 'todo' atau 'goto'."""
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            key = get_key(20)
            if key is None:
                if window_closed():
                    return "quit"
                continue
            res = None
            if key in ("enter", " "):
                if self.accept():
                    res = self.advance()
            elif key == "y":
                if self.lab["box"] is None and self.lab["status"] != "tidak_ada":
                    self.flash("Pilih kotak dulu, lalu tekan y.")
                else:
                    self.lab["verif"] = "sesuai"
                    if self.accept():
                        res = self.advance()
            elif key == "z":
                r = self.zoom_check()
                if r == "y":
                    self.lab["verif"] = "sesuai"
                    if self.accept():
                        res = self.advance()
                elif r == "t":
                    self.lab["verif"] = "belum"
                    self.saved[self.k] = False
                    self.flash("Perbaiki: seret tepi kotak, klik kotak lain, atau klik-tarik kotak baru.", 4)
            elif key in ("back", "b"):
                self.k = max(0, self.k - 1)
            elif key == "n":
                self.toggle_absent()
            elif key == "o":
                self.toggle_occluded()
            elif key == "x":
                self.clear()
            elif key.isdigit() and key != "0":
                self.select_number(int(key))
            elif key == "v":
                self.play_clip()
            elif key == "c":
                self.add_note()
            elif key == "g":
                res = self.review()
            elif key in ("]", "pgdn", "s"):
                res = "next"
            elif key in ("[", "pgup"):
                res = "prev"
            elif key == "u":
                res = "todo"
            elif key == "k":
                res = "goto"
            elif key == "i":
                self.show_list = not self.show_list
            elif key == "h":
                self.show_help = not self.show_help
            elif key in ("q", "esc"):
                res = "quit"
            if res is not None:
                return res


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

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
            sys.exit(f"Nama video ganda: {q} dan {seen[q.stem]}. video_id (nama file tanpa ekstensi) harus unik.")
        seen[q.stem] = q
    return out


def find_video(videos, text, current=0):
    """Cari indeks video dari nomor (1-based) atau sebagian nama. None jika tidak ketemu."""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        i = int(text) - 1
        return i if 0 <= i < len(videos) else None
    exact = [i for i, v in enumerate(videos) if v.stem == text]
    if exact:
        return exact[0]
    hits = [i for i, v in enumerate(videos) if text.lower() in v.stem.lower()]
    if not hits:
        return None
    after = [i for i in hits if i > current]
    return after[0] if after else hits[0]


def next_todo(store, videos, n_frames, idx, broken):
    """Video berikutnya (melingkar) yang belum terverifikasi."""
    n = len(videos)
    for d in range(1, n + 1):
        i = (idx + d) % n
        if i not in broken and video_state(store, videos[i].stem, n_frames) != "terverifikasi":
            return i
    return None


def print_summary(store, videos, n_frames):
    st = Counter(r["status"] for r in store.rows.values())
    src = Counter(r["sumber"] for r in store.rows.values())
    ver = Counter(r.get("verifikasi") or "belum" for r in store.rows.values())
    vs = Counter(video_state(store, v.stem, n_frames) for v in videos)
    print("\n=== Ringkasan ===")
    print("Status video    : " + ", ".join(f"{k}={vs.get(k, 0)}" for k in STATE_COL))
    print(f"Total label     : {len(store.rows)}")
    print("Status pasien   : " + ", ".join(f"{k}={v}" for k, v in st.items()))
    print("Sumber kotak    : " + ", ".join(f"{k}={v}" for k, v in src.items()))
    print("Verifikasi kotak: " + ", ".join(f"{k}={v}" for k, v in ver.items()))
    print(f"File            : {store.path}")


def show_loading(text, w=900, h=200):
    img = np.full((h, w, 3), 30, np.uint8)
    put_text(img, text, (20, h // 2), 0.7, (0, 255, 255), 2)
    cv2.imshow(WIN, img)
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description="Pelabelan manual identitas pasien per cuplikan frame.")
    ap.add_argument("--videos", nargs="+", required=True, help="folder video dan/atau file video")
    ap.add_argument("--out", default="patient_labels.csv", help="file CSV keluaran")
    ap.add_argument("--det-cache", default=None, help="folder cache deteksi <video_id>.npz (opsional)")
    ap.add_argument("--n-frames", type=int, default=8, help="jumlah cuplikan per video")
    ap.add_argument("--stride", type=int, default=2, help="cuplikan dibulatkan ke kelipatan ini")
    ap.add_argument("--min-score", type=float, default=0.0, help="skor minimum kotak yang ditampilkan")
    ap.add_argument("--max-w", type=int, default=1400, help="lebar tampilan maksimum (piksel)")
    ap.add_argument("--max-h", type=int, default=850, help="tinggi tampilan maksimum (piksel)")
    ap.add_argument("--clip-s", type=float, default=2.0, help="setengah durasi klip konteks (detik)")
    ap.add_argument("--labeler", default=getpass.getuser(), help="nama labeler (disimpan di CSV)")
    ap.add_argument("--start", default=None, help="mulai di video ini (nomor 1-based atau nama)")
    ap.add_argument("--redo", action="store_true", help="mulai dari video pertama, walau sudah terverifikasi")
    ap.add_argument("--only", default=None, help="hanya video yang namanya mengandung teks ini")
    args = ap.parse_args()

    videos = collect_videos(args.videos)
    if args.only:
        videos = [v for v in videos if args.only in v.stem]
    if not videos:
        sys.exit("Tidak ada video.")
    store = LabelStore(args.out)
    print(f"{len(videos)} video, {len(store.rows)} label sudah ada di {args.out}")

    broken = set()
    if args.start:
        idx = find_video(videos, args.start)
        if idx is None:
            sys.exit(f"Video '{args.start}' tidak ditemukan.")
    elif args.redo:
        idx = 0
    else:
        idx = next_todo(store, videos, args.n_frames, len(videos) - 1, broken)
        if idx is None:
            print("Semua video sudah terverifikasi; membuka video pertama.")
            idx = 0

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cache = {}            # indeks -> VideoSession (beberapa terakhir, agar bolak-balik cepat)
    step, note = 1, None
    try:
        while True:
            path = videos[idx]
            sess = cache.get(idx)
            if sess is None:
                show_loading(f"Memuat {path.name}  [{idx + 1}/{len(videos)}] ...")
                print(f"[{idx + 1}/{len(videos)}] {path.name}")
                try:
                    sess = VideoSession(path, store, args, idx, videos)
                except Exception as e:  # noqa: BLE001
                    print(f"  [lewati] {e}")
                    broken.add(idx)
                    nxt = idx + step
                    if len(broken) >= len(videos) or not 0 <= nxt < len(videos):
                        nxt = next(i for i in range(len(videos)) if i not in broken) \
                            if len(broken) < len(videos) else None
                    if nxt is None:
                        break
                    idx = nxt
                    continue
                cache[idx] = sess
                while len(cache) > 4:
                    cache.pop(next(i for i in cache if i != idx))
            if note:
                sess.flash(note, 3)
                note = None

            res = sess.run()
            if res == "quit":
                break
            if res == "next":
                step = 1
                if idx + 1 < len(videos):
                    idx += 1
                else:
                    note = "Ini video terakhir."
            elif res == "prev":
                step = -1
                if idx > 0:
                    idx -= 1
                else:
                    note = "Ini video pertama."
            elif res in ("done", "todo"):
                step = 1
                nxt = next_todo(store, videos, args.n_frames, idx, broken)
                if res == "done":
                    print(f"  {path.stem}: {video_state(store, path.stem, args.n_frames)}")
                if nxt is None or nxt == idx:
                    note = "Semua video sudah terverifikasi!"
                    if res == "done" and nxt is None and idx + 1 < len(videos):
                        idx += 1
                else:
                    idx = nxt
            elif res == "goto":
                print(f"\nLompat ke video (nomor 1-{len(videos)} atau sebagian nama, Enter = batal):")
                try:
                    txt = input("> ")
                except EOFError:
                    txt = ""
                j = find_video(videos, txt, idx)
                if j is None:
                    note = f"Video '{txt.strip()}' tidak ditemukan." if txt.strip() else None
                else:
                    step, idx = (1 if j >= idx else -1), j
    finally:
        cv2.destroyAllWindows()
        print_summary(store, videos, args.n_frames)


if __name__ == "__main__":
    main()
