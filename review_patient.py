#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_patient.py - Pemutar video untuk menandai ANOMALI identitas pasien, per video.

Tonton setiap video seperti di video player. Kotak HIJAU-TOSKA tebal = PASIEN (menurut tracker + 8 label Anda),
ORANYE = observer, MAGENTA = kotak label manual Anda (hanya di 8 frame label).
Saat melihat anomali, tekan tombol jenisnya SAAT ANOMALI MULAI dan tekan LAGI SAAT SELESAI (video tetap berjalan):

  1  label_pindah_ke_observer        kotak berlabel PASIEN ada di observer (label pindah ke orang lain)
                                      -> klik kotak orang yang sebenarnya pasien (opsional)
  2  observer_mengisi_kotak_pasien   kotak PASIEN berisi observer: kotak menyatu / kebesaran sampai mencakup
                                      observer, atau observer menutupi pasien -> klik kotak observernya (opsional)
  3  pasien_hilang                   pasien terlihat tapi tidak ada kotak PASIEN
  4  lainnya                         anomali lain (tulis catatan dengan c)
  Esc  batalkan tanda yang sedang direkam      x  hapus tanda di posisi sekarang (atau terdekat <= 1 s)
  c    catatan (ketik di terminal) untuk tanda di posisi sekarang, atau untuk video jika tidak ada tanda
  [ ]  lompat ke tanda Anda sebelumnya / berikutnya
Penilaian video (setelah ditonton):
  y    VIDEO OK: tidak ada anomali (hanya bisa jika video tidak punya tanda)
  o    VIDEO SELESAI: otomatis OK jika tanpa tanda, ADA_ANOMALI jika ada tanda
  u    batalkan penilaian (kembali ke belum_selesai)
  Setelah y / o, program otomatis pindah ke video berikutnya yang belum dinilai (matikan dengan --no-auto-next).

Pemutar:
  Spasi  putar / jeda          panah atas/bawah atau + -   kecepatan 0.25x .. 4x
  a d (panah kiri/kanan)  -/+ 2 detik      j l  -/+ 10 detik      , .  -/+ 1 frame      0  ke awal
  klik / geser di bilah waktu = lompat ke waktu itu
  n p  lompat ke dugaan otomatis berikutnya / sebelumnya (segitiga di bilah waktu)
Panel DAFTAR ANOMALI (di kanan video; e = tampil/sembunyi):
  berisi semua tanda anomali dari data, urut waktu:
    SAYA      tanda yang Anda buat (1-4)
    OTOMATIS  dugaan otomatis dari notebook (events.csv: anchor_conflict, jump, handover, ...)
    POLA      pola kotak dari data track: kotak PASIEN tumpang tindih observer, kotak melebar,
              kotak memendek, atau pasien tanpa kotak (segmen berurutan)
  klik baris      lompat ke awal anomali itu, video tetap DIJEDA (Spasi = putar)
  klik kanan / z  tandai baris OTOMATIS / POLA sebagai "bukan anomali" (klik lagi = batal)
  t / r           baris berikutnya / sebelumnya      f  filter: semua / saya / otomatis / pola / belum dicek
  roda mouse      gulir daftar
  Baris yang sudah diklik = DICEK; yang tumpang tindih dengan tanda Anda = DITANDAI.
  Disimpan di identity/suspect_review.csv.
Video:
  PageDown / m  video berikutnya     PageUp / b / Backspace  video sebelumnya
  g  video berikutnya yang BELUM dinilai          k  pilih video (ketik nomor / nama di terminal)
  i  daftar video + status: panah atas/bawah pilih, Enter / klik buka, i / Esc tutup
  h  bantuan          q  keluar (semua sudah tersimpan)

SEMUA TANDA LANGSUNG TERSIMPAN. Cadangan dibuat setiap kali program dibuka (identity/review_backups/).
Hasil (di outputs/02b_analysis/<RUN>/identity/):
  anomaly_review.csv  satu baris per tanda: video, jenis, mulai, selesai, track terkait, catatan
  suspect_review.csv  satu baris per baris panel yang sudah dicek / ditandai bukan anomali
  video_review.csv    satu baris per video: status (belum_selesai / ok / ada_anomali), jumlah tanda per jenis,
                      berapa persen video sudah ditonton, catatan
Tinjauan lama (switch_review.csv dari versi sebelumnya) diimpor otomatis sekali.

Contoh (di folder proyek, tempat notebook berada):
  python review_patient.py                       # mulai dari video pertama yang belum selesai
  python review_patient.py --video pasien07      # hanya video yang namanya mengandung teks ini
  python review_patient.py --show done           # hanya video yang sudah selesai (untuk merevisi)
  python review_patient.py --from-start          # mulai dari video 1
  python review_patient.py --start pasien12       # mulai dari video bernama / bernomor ini

Kebutuhan: pip install opencv-python numpy pandas
"""

import argparse
import csv
import getpass
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
WIN = "Tinjau anomali pasien"

# anomaly types: key -> (csv value, short label, BGR colour)
TYPES = {
    ord("1"): ("label_pindah_ke_observer", "PINDAH", (60, 60, 235)),
    ord("2"): ("observer_mengisi_kotak_pasien", "OBS>PASIEN", (0, 200, 255)),
    ord("3"): ("pasien_hilang", "HILANG", (200, 200, 200)),
    ord("4"): ("lainnya", "LAIN", (220, 110, 220)),
}
TYPE_BY_NAME = {v[0]: (k, v[1], v[2]) for k, v in TYPES.items()}
ANOM_FIELDS = ["anomaly_id", "video_id", "type", "start_frame", "end_frame", "start_s", "end_s", "duration_s",
               "patient_track_at_start", "true_patient_track", "observer_track", "note", "reviewer",
               "created_at", "updated_at"]
VIDEO_FIELDS = ["video_id", "status", "anomalies", "label_pindah_ke_observer", "observer_mengisi_kotak_pasien",
                "pasien_hilang", "lainnya", "watched_pct", "duration_s", "note", "reviewer", "updated_at"]

C_PAT, C_OBS, C_LABEL = (122, 175, 27), (52, 104, 235), (255, 0, 255)
C_BAR, C_NONE, C_HEAD, C_WARN = (214, 120, 42), (80, 80, 80), (0, 230, 255), (0, 200, 255)
KEY_LEFT, KEY_RIGHT = {2424832, 65361, 63234}, {2555904, 65363, 63235}
KEY_UP, KEY_DOWN = {2490368, 65362, 63232}, {2621440, 65364, 63233}
KEY_PGUP, KEY_PGDN = {2162688, 65365, 63276, 16777238}, {2228224, 65366, 63277, 16777239}
KEY_ENTER = {10, 13, 65293, 65421, 16777220, 16777221}
KEY_HOME = {2359296, 65360, 63273}
KEY_BACKSPACE = {8, 127, 65288, 16777219}
STATUS_COL = {"ok": (0, 200, 0), "ada_anomali": (60, 60, 235), "belum_selesai": (170, 170, 170)}
STATUS_TXT = {"ok": "VIDEO OK", "ada_anomali": "ADA ANOMALI", "belum_selesai": "BELUM DINILAI"}
SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
HEADER_H, BAR_H, MIN_W = 70, 96, 900
PANEL_W, ROW_H, LIST_Y0 = 360, 22, 78
CUE_GAP_S, CUE_MIN_S = 0.5, 0.2  # pola kotak: gabung celah <= 0.5 s, buang segmen < 0.2 s
CUE_TXT = {"overlap": "TUMPANG TINDIH", "wide": "KOTAK MELEBAR", "short": "KOTAK MEMENDEK",
           "missing": "TANPA KOTAK PASIEN"}
KIND_TXT = {"saya": "SAYA", "otomatis": "OTOMATIS", "pola": "POLA"}
KIND_COL = {"saya": (255, 255, 255), "otomatis": (60, 60, 255), "pola": (0, 200, 255)}
FILTERS = [("semua", None), ("tanda saya", "saya"), ("otomatis", "otomatis"), ("pola kotak", "pola"),
           ("belum dicek", "todo")]
SUSPECT_FIELDS = ["item_id", "video_id", "kind", "type", "start_frame", "end_frame", "start_s", "end_s",
                  "seen", "decision", "reviewer", "updated_at"]
ENC_IOU = 0.15        # cue: patient box overlaps another box at least this much
WIDE_RATIO = 1.4      # cue: patient box this much wider than its usual width in the video
SHORT_RATIO = 0.7     # cue: patient box this much shorter than usual

HELP = [
    "TANDAI (tekan saat mulai, tekan LAGI saat selesai): 1 label pindah ke observer | 2 observer mengisi kotak pasien",
    "   3 pasien hilang | 4 lainnya | Esc batal rekam | x hapus tanda di sini | c catatan | [ ] tanda sebelumnya/berikut",
    "klik kotak saat ada tanda: jenis 1 = pasien yang benar, jenis 2 = observernya",
    "NILAI VIDEO: y = VIDEO OK (tanpa anomali) | o = selesai (OK / ada anomali otomatis) | u = batalkan penilaian",
    "Spasi putar/jeda | atas/bawah kecepatan | a/d -+2 s | j/l -+10 s | , . frame | klik bilah waktu = lompat | n/p dugaan",
    "PANEL ANOMALI: klik = lompat (video diam) | klik kanan / z = bukan anomali | t / r baris berikut/sebelum | f filter | e panel",
    "PageDown/m video berikut | PageUp/b/Backspace sebelumnya | g berikut yg belum dinilai | k pilih | i daftar | q keluar",
]


# ----------------------------------------------------------------------------------------------- helpers
def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    x, y = org
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)):   # outline, no drift
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def num(v, default=-1):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def fmt_t(s):
    s = max(0.0, float(s))
    return f"{int(s // 60):d}:{s % 60:05.2f}"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def find_identity_dir(arg):
    if arg:
        p = Path(arg)
        if not (p / "identity_summary.csv").exists():
            sys.exit(f"{p} tidak berisi identity_summary.csv. Jalankan dulu sel 12 notebook 02b_analysis.ipynb.")
        return p
    cands = sorted(Path("outputs").glob("02b_analysis/*/identity/identity_summary.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        sys.exit("Tidak menemukan outputs/02b_analysis/*/identity/.\n"
                 "Jalankan program ini di folder proyek (tempat 02b_analysis.ipynb) setelah sel 12 selesai,\n"
                 "atau beri --identity <folder identity>.")
    if len(cands) > 1:
        print("Beberapa hasil ditemukan, memakai yang terbaru:", *[c.parent for c in cands], sep="\n  ")
    return cands[-1].parent


def find_videos(data_dir, ids):
    found = {}
    for p in sorted(Path(data_dir).rglob("*")) if Path(data_dir).exists() else []:
        if p.suffix.lower() in VIDEO_EXTS and p.stem in ids:
            if p.stem not in found or len(p.as_posix()) < len(found[p.stem].as_posix()):
                found[p.stem] = p
    return found


class CsvStore:
    """A CSV keyed by one column, rewritten atomically after every change."""

    def __init__(self, path, fields, key):
        self.path, self.fields, self.key = Path(path), fields, key
        self.rows = {}
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows[r[key]] = r

    def put(self, row):
        self.rows[row[self.key]] = row
        self.save()

    def delete(self, k):
        if self.rows.pop(k, None) is not None:
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore")
            w.writeheader()
            for k in sorted(self.rows):
                w.writerow(self.rows[k])
        os.replace(tmp, self.path)

    def backup(self, folder):
        if self.path.exists() and self.rows:
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / f"{self.path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            dst.write_bytes(self.path.read_bytes())
            return dst
        return None


class FrameCache(threading.Thread):
    """Decodes the processed frames of one video in the background (sequential read = same frame numbers as
    Stage 1/2) and keeps them as small JPEGs, so playing, pausing and seeking back are instant."""

    def __init__(self, path, frames, scale, quality=80, size=None):
        super().__init__(daemon=True)
        self.path, self.wanted, self.scale, self.quality = path, set(int(f) for f in frames), scale, quality
        self.size = size                                   # (w, h) of the display, so every frame has the same shape
        self.last = max(self.wanted) if self.wanted else -1
        self.jpg, self.bytes, self.read_upto, self.done, self.error = {}, 0, -1, False, None
        self.halt = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(str(self.path))
        i = 0
        try:
            while not self.halt.is_set() and i <= self.last:
                if not cap.grab():
                    break
                if i in self.wanted:
                    ok, fr = cap.retrieve()
                    if ok:
                        if self.size is not None and (fr.shape[1], fr.shape[0]) != tuple(self.size):
                            fr = cv2.resize(fr, tuple(self.size), interpolation=cv2.INTER_AREA)
                        buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1]
                        self.jpg[i] = buf
                        self.bytes += buf.nbytes
                self.read_upto = i
                i += 1
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            cap.release()
            self.done = True

    def stop(self):
        self.halt.set()
        self.join(5)


class VideoData:
    def __init__(self, ident, vid, srow, anchors, events):
        T = pd.read_csv(ident / "tracks" / f"{vid}.csv")
        P = pd.read_csv(ident / "patient" / f"{vid}.csv")
        self.vid = vid
        self.frames = P["frame_idx"].astype(int).to_numpy()
        self.patient = P["patient_track"].astype(int).to_numpy()
        self.pos_of = {int(f): k for k, f in enumerate(self.frames)}
        self.boxes = {}
        for f, t, x1, y1, x2, y2 in T[["frame_idx", "track_id", "x1", "y1", "x2", "y2"]].itertuples(index=False):
            self.boxes.setdefault(int(f), []).append((int(t), (float(x1), float(y1), float(x2), float(y2))))
        self.fps, self.stride = float(srow["fps"]), int(srow["stride"])
        self.fps_eff = self.fps / max(1, self.stride)
        self.W, self.H = int(srow["width"]), int(srow["height"])
        self.duration = (self.frames[-1] + self.stride) / self.fps if len(self.frames) else 0.0
        self.anchors = {}
        for r in anchors.itertuples(index=False):
            ok = pd.notna(r.x1) and str(r.x1).strip() != ""
            self.anchors[int(r.frame_idx)] = (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) if ok else None
        self.events = [(int(r.frame_idx), float(r.priority), str(r.types)) for r in events.itertuples(index=False)]
        pw, ph = [], []                                   # usual size of the patient box in this video
        for f, t in zip(self.frames, self.patient):
            b = next((bb for tt, bb in self.boxes.get(int(f), []) if tt == t), None) if t >= 0 else None
            if b is not None:
                pw.append(b[2] - b[0]); ph.append(b[3] - b[1])
        self.med_w = float(np.median(pw)) if pw else None
        self.med_h = float(np.median(ph)) if ph else None
        valid = np.where(self.patient >= 0)[0]          # positions where the patient track id changes
        self.id_changes = [int(valid[k]) for k in range(1, len(valid)) if self.patient[valid[k]] != self.patient[valid[k - 1]]]
        self.cues = self.find_cues()

    def find_cues(self):
        """Segmen berurutan di mana kotak PASIEN berpola aneh (dari data track, tanpa model tambahan)."""
        n = len(self.frames)
        flag = {k: np.zeros(n, bool) for k in CUE_TXT}
        val = {k: np.zeros(n) for k in CUE_TXT}
        for k, (f, t) in enumerate(zip(self.frames, self.patient)):
            boxes = self.boxes.get(int(f), [])
            pb = next((b for tt, b in boxes if tt == t), None) if t >= 0 else None
            if pb is None:
                flag["missing"][k] = bool(boxes)
                continue
            ov = max((iou(pb, b) for tt, b in boxes if tt != t), default=0.0)
            if ov >= ENC_IOU:
                flag["overlap"][k], val["overlap"][k] = True, ov
            if self.med_w and (pb[2] - pb[0]) / self.med_w >= WIDE_RATIO:
                flag["wide"][k], val["wide"][k] = True, (pb[2] - pb[0]) / self.med_w
            if self.med_h and (pb[3] - pb[1]) / self.med_h <= SHORT_RATIO:
                flag["short"][k], val["short"][k] = True, (pb[3] - pb[1]) / self.med_h
        gap = max(1, int(round(CUE_GAP_S * self.fps_eff)))
        min_n = max(2, int(round(CUE_MIN_S * self.fps_eff)))
        segs = []
        for sub, arr in flag.items():
            idx = np.where(arr)[0]
            if not len(idx):
                continue
            for run in np.split(idx, np.where(np.diff(idx) > gap)[0] + 1):
                if len(run) < min_n:
                    continue
                v = val[sub][run]
                v = v[v > 0]
                if sub == "overlap":
                    det = f"IoU maks {v.max():.2f}"
                elif sub == "wide":
                    det = f"lebar {v.max():.1f}x biasa"
                elif sub == "short":
                    det = f"tinggi {v.min():.1f}x biasa"
                else:
                    det = "pasien tidak punya kotak"
                segs.append(dict(sub=sub, a=int(self.frames[run[0]]), b=int(self.frames[run[-1]]), detail=det))
        return segs

    def pos_at(self, f):
        return self.pos_of.get(int(f), int(np.clip(np.searchsorted(self.frames, f), 0, max(0, len(self.frames) - 1))))


# ----------------------------------------------------------------------------------------------- player
class Player:
    def __init__(self, args, ident, summary, events, anchors, videos):
        self.args, self.ident = args, ident
        self.summary = summary.set_index("video_id")
        self.events_all, self.anchors_all, self.paths = events, anchors, videos
        self.anoms = CsvStore(ident / "anomaly_review.csv", ANOM_FIELDS, "anomaly_id")
        self.vrev = CsvStore(ident / "video_review.csv", VIDEO_FIELDS, "video_id")
        self.susp = CsvStore(ident / "suspect_review.csv", SUSPECT_FIELDS, "item_id")
        bdir = ident / "review_backups"
        self.backups = [b for b in (self.anoms.backup(bdir), self.vrev.backup(bdir), self.susp.backup(bdir)) if b]
        self.show_panel = not getattr(args, "no_panel", False)
        self.filter_i, self.panel_scroll, self.panel_scroll_t, self.sel_item = 0, 0, 0.0, None
        self.vd, self.cache, self.watched = None, None, None
        self.imported = self.import_old_review()
        self.videos = self.build_list()
        self.vidx = self.start_index()
        self.pos, self.acc, self.last_t = 0, 0.0, time.time()
        self.playing, self.speed_i = False, SPEEDS.index(1.0)
        self.rec = None                                     # anomaly_id being recorded (open segment)
        self.show_help = self.show_list = False
        self.list_sel = self.vidx
        self.msg, self.msg_until = "", 0.0
        self.dragging = False
        if self.videos:
            self.open_video()

    # ---------------------------------------------------------------- setup
    def import_old_review(self):
        """Convert decisions from the previous (event-by-event) version once, so earlier work is kept."""
        old = self.ident / "switch_review.csv"
        if self.anoms.rows or not old.exists():
            return 0
        n = 0
        with open(old, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            typ = {"pindah_ke_observer": "label_pindah_ke_observer", "pasien_hilang": "pasien_hilang"}.get(r["decision"])
            if not typ or str(r.get("video_id", "")) not in set(self.summary.index):
                continue
            f = num(r["switch_frame"], num(r["event_frame"]))
            fps = float(self.summary.loc[r["video_id"], "fps"])
            aid = f"{r['video_id']}_a{f:06d}"
            self.anoms.rows[aid] = dict(anomaly_id=aid, video_id=r["video_id"], type=typ, start_frame=f, end_frame=f,
                                        start_s=f"{f / fps:.2f}", end_s=f"{f / fps:.2f}", duration_s="0.00",
                                        patient_track_at_start=r.get("wrong_track", ""),
                                        true_patient_track=r.get("true_track", ""), observer_track="",
                                        note=("[impor] " + r.get("note", "")).strip(), reviewer=r.get("reviewer", ""),
                                        created_at=r.get("reviewed_at", ""), updated_at=now_iso())
            n += 1
        if n:
            self.anoms.save()
            for v in {r["video_id"] for r in self.anoms.rows.values()}:
                self.update_video_row(v)
        return n

    def build_list(self):
        a = self.args
        vids = [v for v in self.summary.index if not a.video or a.video in v]
        missing = [v for v in vids if v not in self.paths]
        if missing:
            print(f"[peringatan] video asli tidak ditemukan untuk {len(missing)} video (mis. {missing[:3]}); dilewati.")
        vids = [v for v in vids if v in self.paths]
        show = getattr(a, "show", "all")
        if show == "done":
            vids = [v for v in vids if self.is_done(v)]
        elif show == "todo":
            vids = [v for v in vids if not self.is_done(v)]
        return vids

    def is_done(self, vid):
        return self.vrev.rows.get(vid, {}).get("status") in ("ok", "ada_anomali")

    def start_index(self):
        a = self.args
        if not self.videos:
            return 0
        if getattr(a, "start", None):
            st = str(a.start).strip()
            if st.isdigit():
                return int(np.clip(int(st) - 1, 0, len(self.videos) - 1))
            i = next((k for k, v in enumerate(self.videos) if v == st), None)
            if i is None:
                i = next((k for k, v in enumerate(self.videos) if st.lower() in v.lower()), None)
            if i is None:
                print(f"[peringatan] --start {st} tidak ditemukan; mulai dari video pertama yang belum dinilai.")
            else:
                return i
        if getattr(a, "from_start", False):
            return 0
        return next((i for i, v in enumerate(self.videos) if not self.is_done(v)), 0)

    @property
    def vid(self):
        return self.videos[self.vidx]

    def open_video(self):
        if self.cache is not None:
            self.cache.stop()
        if self.rec:
            self.close_rec()
        vid = self.vid
        anc = self.anchors_all[self.anchors_all["video_id"] == vid] if len(self.anchors_all) else self.anchors_all
        ev = self.events_all[self.events_all["video_id"] == vid] if len(self.events_all) else self.events_all
        self.vd = VideoData(self.ident, vid, self.summary.loc[vid], anc, ev)
        avail_w = self.args.max_w - (PANEL_W if not getattr(self.args, "no_panel", False) else 0)
        self.scale = min(max(320, avail_w) / self.vd.W, self.args.max_h / self.vd.H, 1.0)
        self.panel_scroll, self.sel_item = 0, None
        self.dw, self.dh = int(self.vd.W * self.scale), int(self.vd.H * self.scale)
        self.cache = FrameCache(self.paths[vid], self.vd.frames, self.scale, self.args.jpeg_quality,
                                size=(self.dw, self.dh))
        self.cache.start()
        self.pos, self.acc, self.playing = 0, 0.0, False
        self.watched = np.zeros(len(self.vd.frames), bool)
        row = self.vrev.rows.get(vid)
        if row and num(row.get("watched_pct"), 0) > 0:          # earlier sessions (approximate, from the start)
            self.watched[:int(len(self.watched) * num(row["watched_pct"], 0) / 100)] = True
        st = self.vrev.rows.get(vid, {}).get("status", "belum_selesai")
        self.flash(f"{vid} [{STATUS_TXT.get(st, st)}]: Spasi = putar. Tandai anomali 1-4; "
                   "selesai: y = VIDEO OK, o = selesai.", 4)

    def flash(self, text, sec=2.5):
        self.msg, self.msg_until = text, time.time() + sec

    @property
    def frame(self):
        return int(self.vd.frames[self.pos]) if len(self.vd.frames) else 0

    @property
    def t(self):
        return self.frame / self.vd.fps

    # ---------------------------------------------------------------- anomalies
    def video_anoms(self, vid=None):
        vid = vid or self.vid
        return sorted([r for r in self.anoms.rows.values() if r["video_id"] == vid], key=lambda r: num(r["start_frame"]))

    def anom_here(self, tol_s=1.0):
        """The anomaly being recorded, else the one containing the current frame, else the nearest within tol_s."""
        if self.rec and self.rec in self.anoms.rows:
            return self.anoms.rows[self.rec]
        f, tol = self.frame, tol_s * self.vd.fps
        inside = [r for r in self.video_anoms() if num(r["start_frame"]) <= f <= num(r["end_frame"])]
        if inside:
            return inside[-1]
        dist = lambda r: min(abs(num(r["start_frame"]) - f), abs(num(r["end_frame"]) - f))
        near = [r for r in self.video_anoms() if dist(r) <= tol]
        return min(near, key=dist) if near else None

    def write_anom(self, r):
        s, e = num(r["start_frame"]), num(r["end_frame"])
        if e < s:
            s, e = e, s
        r.update(start_frame=s, end_frame=e, start_s=f"{s / self.vd.fps:.2f}", end_s=f"{e / self.vd.fps:.2f}",
                 duration_s=f"{(e - s) / self.vd.fps:.2f}", updated_at=now_iso(), reviewer=self.args.reviewer)
        self.anoms.put(r)
        self.update_video_row(r["video_id"])

    def start_rec(self, typ):
        f = self.frame
        aid = f"{self.vid}_a{f:06d}"
        while aid in self.anoms.rows:
            aid += "b"
        pt = int(self.vd.patient[self.pos])
        self.write_anom(dict(anomaly_id=aid, video_id=self.vid, type=typ, start_frame=f, end_frame=f,
                             patient_track_at_start="" if pt < 0 else pt, true_patient_track="", observer_track="",
                             note="", created_at=now_iso()))
        self.rec = aid
        k, lab, _ = TYPE_BY_NAME[typ]
        self.flash(f"MEREKAM {lab} sejak {fmt_t(self.t)}. Tekan {chr(k)} lagi saat anomali selesai.", 4)

    def close_rec(self):
        r = self.anoms.rows.get(self.rec)
        self.rec = None
        if r is None:
            return
        r["end_frame"] = self.frame
        self.write_anom(r)
        s, e = num(r["start_frame"]) / self.vd.fps, num(r["end_frame"]) / self.vd.fps
        self.flash(f"Tersimpan: {TYPE_BY_NAME[r['type']][1]} {fmt_t(s)} - {fmt_t(e)} ({e - s:.1f} s)")

    def on_type_key(self, key):
        typ = TYPES[key][0]
        if self.rec:
            cur = self.anoms.rows.get(self.rec, {}).get("type")
            self.close_rec()
            if cur == typ:
                return
        self.start_rec(typ)

    def delete_here(self):
        if self.rec:
            self.anoms.delete(self.rec)
            self.rec = None
            self.update_video_row(self.vid)
            self.flash("Rekaman tanda dibatalkan.")
            return
        r = self.anom_here()
        if r is None:
            self.flash("Tidak ada tanda di posisi ini (<= 1 s). Pakai [ ] untuk lompat ke tanda.")
            return
        self.anoms.delete(r["anomaly_id"])
        self.update_video_row(self.vid)
        self.flash(f"Tanda {TYPE_BY_NAME[r['type']][1]} di {fmt_t(num(r['start_frame']) / self.vd.fps)} dihapus.")

    def click_track(self, t):
        r = self.anom_here()
        if r is None:
            self.flash(f"Track {t}. Tekan 1 atau 2 dulu untuk menandai anomali di sini.")
            return
        if r["type"] == "label_pindah_ke_observer":
            r["true_patient_track"] = t
            self.write_anom(r)
            self.flash(f"Tersimpan: pasien yang benar = track {t}")
        elif r["type"] == "observer_mengisi_kotak_pasien":
            r["observer_track"] = t
            self.write_anom(r)
            self.flash(f"Tersimpan: observer yang mengisi kotak pasien = track {t}")
        else:
            self.flash("Klik kotak hanya dipakai untuk tanda jenis 1 dan 2.")

    def ask(self, prompt):
        self.playing = False
        print(prompt)
        try:
            return input("> ").strip().replace("\n", " ")
        except EOFError:
            return ""

    def note(self):
        r = self.anom_here()
        if r is not None:
            r["note"] = self.ask(f"\nCatatan untuk tanda {TYPE_BY_NAME[r['type']][1]} di {self.vid} "
                                 f"{fmt_t(num(r['start_frame']) / self.vd.fps)} (Enter kosong = hapus):")
            self.write_anom(r)
        else:
            row = self.video_row(self.vid)
            row["note"] = self.ask(f"\nCatatan untuk VIDEO {self.vid} (Enter kosong = hapus):")
            self.vrev.put(row)
        self.flash("Catatan tersimpan.")

    # ---------------------------------------------------------------- per-video status
    def video_row(self, vid):
        return dict(self.vrev.rows.get(vid) or dict(video_id=vid, status="belum_selesai", note=""))

    def update_video_row(self, vid, done=None):
        row = self.video_row(vid)
        an = self.video_anoms(vid)
        c = Counter(r["type"] for r in an)
        is_done = (row.get("status") in ("ok", "ada_anomali")) if done is None else done
        row.update(status=("ada_anomali" if an else "ok") if is_done else "belum_selesai", anomalies=len(an),
                   label_pindah_ke_observer=c["label_pindah_ke_observer"],
                   observer_mengisi_kotak_pasien=c["observer_mengisi_kotak_pasien"],
                   pasien_hilang=c["pasien_hilang"], lainnya=c["lainnya"], reviewer=self.args.reviewer,
                   updated_at=now_iso())
        if self.vd is not None and vid == self.vd.vid and self.watched is not None:
            row.update(watched_pct=int(round(100 * self.watched.mean())), duration_s=f"{self.vd.duration:.1f}")
        self.vrev.put(row)
        return row

    def set_verdict(self, kind):
        """kind: 'ok' (video OK, tanpa tanda), 'done' (otomatis OK / ada_anomali), 'undo'."""
        if self.rec:
            self.close_rec()
        n = len(self.video_anoms())
        if kind == "undo":
            self.update_video_row(self.vid, done=False)
            self.flash("Penilaian dibatalkan: video kembali BELUM DINILAI.")
            return
        if kind == "ok" and n:
            self.flash(f"Video ini punya {n} tanda anomali. Jika sebenarnya OK, hapus tandanya dulu ([ ] lalu x), "
                       "atau tekan o untuk menyimpan sebagai ADA ANOMALI.", 6)
            return
        row = self.update_video_row(self.vid, done=True)
        pct = row.get("watched_pct", 0)
        msg = (f"{self.vid} dinilai {STATUS_TXT[row['status']]} ({row['anomalies']} tanda, ditonton {pct}%).")
        if num(pct, 0) < 90:
            msg += " Catatan: video belum ditonton penuh."
        print("  " + msg)
        if self.args.no_auto_next:
            self.flash(msg + " PageDown = video berikutnya.", 5)
            return
        nxt = self.next_todo()
        if nxt is None:
            self.flash(msg + " SEMUA VIDEO SUDAH DINILAI.", 6)
            return
        self.switch_to(nxt)
        self.flash(msg + f"  ->  sekarang: {self.vid}", 5)

    def next_todo(self):
        """Video berikutnya (melingkar) yang belum dinilai, atau None."""
        n = len(self.videos)
        for d in range(1, n + 1):
            i = (self.vidx + d) % n
            if not self.is_done(self.videos[i]):
                return i
        return None

    # ---------------------------------------------------------------- navigation
    def seek_pos(self, p):
        if len(self.vd.frames):
            self.pos = int(np.clip(p, 0, len(self.vd.frames) - 1))
            self.acc = 0.0

    def seek_s(self, ds):
        self.seek_pos(self.pos + int(round(ds * self.vd.fps_eff)))

    def seek_frame(self, f):
        self.seek_pos(self.vd.pos_at(f))

    def jump_anom(self, d):
        f = self.frame
        starts = [num(r["start_frame"]) for r in self.video_anoms()]
        tgt = [s for s in starts if s < f] if d < 0 else [s for s in starts if s > f]
        if not tgt:
            self.flash("Tidak ada tanda " + ("sebelumnya." if d < 0 else "berikutnya."))
            return
        self.seek_frame(tgt[-1] if d < 0 else tgt[0])

    def jump_event(self, d):
        lead = int(1.5 * self.vd.fps)                      # land 1.5 s before the suspect
        f = self.frame + lead
        ev = sorted(e[0] for e in self.vd.events)
        tgt = [e for e in ev if e > f] if d > 0 else [e for e in ev if e < f - 1]
        if not tgt:
            self.flash("Tidak ada dugaan otomatis " + ("berikutnya." if d > 0 else "sebelumnya."))
            return
        e = tgt[0] if d > 0 else tgt[-1]
        self.seek_frame(e - lead)
        self.flash(f"Dugaan otomatis di {fmt_t(e / self.vd.fps)}: " + next(x[2] for x in self.vd.events if x[0] == e))

    def switch_to(self, i):
        if i == self.vidx:
            return
        if self.rec:
            self.close_rec()
        self.update_video_row(self.vid)          # save watched %
        self.vidx = self.list_sel = i
        self.open_video()

    def go_video(self, d):
        new = self.vidx + d
        if not 0 <= new < len(self.videos):
            self.flash("Ini video " + ("terakhir." if d > 0 else "pertama."))
            return
        self.switch_to(new)

    def go_todo(self):
        nxt = self.next_todo()
        if nxt is None:
            self.flash("Semua video sudah dinilai.")
        elif nxt == self.vidx:
            self.flash("Hanya video ini yang belum dinilai.")
        else:
            self.switch_to(nxt)

    def goto_video(self, text):
        text = (text or "").strip()
        if not text:
            return False
        if text.isdigit() and 1 <= int(text) <= len(self.videos):
            i = int(text) - 1
        else:
            i = next((k for k, v in enumerate(self.videos) if v == text), None)
            if i is None:
                i = next((k for k, v in enumerate(self.videos) if text.lower() in v.lower()), None)
        if i is None:
            self.flash(f"Video '{text}' tidak ditemukan.")
            return False
        self.switch_to(i)
        return True

    # ---------------------------------------------------------------- playback
    def tick(self):
        """Advance the playhead according to real time and speed, only over frames already decoded."""
        t_now = time.time()
        dt, self.last_t = t_now - self.last_t, t_now
        if not self.playing or not len(self.vd.frames):
            return
        self.acc += dt * self.vd.fps_eff * SPEEDS[self.speed_i]
        steps = int(self.acc)
        moved = 0
        while moved < steps and self.pos < len(self.vd.frames) - 1 and int(self.vd.frames[self.pos + 1]) in self.cache.jpg:
            self.pos += 1
            moved += 1
            self.watched[self.pos] = True
        self.acc = min(self.acc - moved, 3.0)                # while buffering, do not build up a big jump
        if self.pos >= len(self.vd.frames) - 1:
            self.playing = False
            if self.video_anoms():
                self.flash("Akhir video. o = selesai (ADA ANOMALI), PageDown = video berikutnya.", 6)
            else:
                self.flash("Akhir video. Tidak ada tanda: tekan y = VIDEO OK (atau tandai anomali dulu).", 6)

    # ---------------------------------------------------------------- anomaly panel
    def panel_items(self):
        """Semua tanda anomali video ini, urut waktu, sesudah filter."""
        vd, vid, out = self.vd, self.vid, []
        marks = self.video_anoms()
        for r in marks:
            k, lab, col = TYPE_BY_NAME[r["type"]]
            out.append(dict(id=r["anomaly_id"], kind="saya", type=r["type"], a=num(r["start_frame"]),
                            b=num(r["end_frame"]), label=lab, detail=(r.get("note") or "")[:28], col=col))
        for f, pr, types in vd.events:
            out.append(dict(id=f"{vid}_otomatis_{f:06d}", kind="otomatis", type=types, a=f, b=f,
                            label=types.split("+")[0][:16], detail=f"prioritas {pr:g}" +
                            (f" +{types.count('+')}" if "+" in types else ""), col=KIND_COL["otomatis"]))
        for c in vd.cues:
            out.append(dict(id=f"{vid}_pola_{c['sub']}_{c['a']:06d}", kind="pola", type=c["sub"], a=c["a"], b=c["b"],
                            label=CUE_TXT[c["sub"]], detail=c["detail"], col=KIND_COL["pola"]))
        tol = self.vd.fps * 1.0
        for it in out:
            row = self.susp.rows.get(it["id"], {})
            it["seen"] = row.get("seen") == "1"
            it["dismissed"] = row.get("decision") == "bukan_anomali"
            it["marked"] = it["kind"] == "saya" or any(
                num(r["start_frame"]) - tol <= it["b"] and it["a"] <= num(r["end_frame"]) + tol for r in marks)
        out.sort(key=lambda it: (it["a"], it["kind"]))
        flt = FILTERS[self.filter_i][1]
        if flt == "todo":
            out = [it for it in out if not (it["seen"] or it["dismissed"] or it["marked"])]
        elif flt:
            out = [it for it in out if it["kind"] == flt]
        return out

    def save_item(self, it, **kw):
        row = dict(self.susp.rows.get(it["id"]) or dict(item_id=it["id"], video_id=self.vid, kind=it["kind"],
                                                       type=it["type"], seen="", decision=""))
        row.update(start_frame=it["a"], end_frame=it["b"], start_s=f"{it['a'] / self.vd.fps:.2f}",
                   end_s=f"{it['b'] / self.vd.fps:.2f}", reviewer=self.args.reviewer, updated_at=now_iso(), **kw)
        self.susp.put(row)

    def open_item(self, it):
        self.sel_item = it["id"]
        self.playing = False                               # tetap diam; putar sendiri dengan Spasi
        self.seek_frame(it["a"])
        if it["kind"] != "saya":
            self.save_item(it, seen="1")
        dur = (it["b"] - it["a"]) / self.vd.fps
        self.flash(f"{KIND_TXT[it['kind']]} {it['label']} di {fmt_t(it['a'] / self.vd.fps)}"
                   + (f" ({dur:.1f} s)" if dur > 0 else "") + f": {it['detail']}.  Spasi = putar, a = mundur 2 s. "
                   "Anomali? 1-4. Bukan? klik kanan / z.", 6)

    def dismiss_item(self, it):
        if it["kind"] == "saya":
            self.flash("Ini tanda Anda sendiri; hapus dengan x di posisinya.")
            return
        new = "" if it["dismissed"] else "bukan_anomali"
        self.save_item(it, seen="1", decision=new)
        self.flash(f"{it['label']} {fmt_t(it['a'] / self.vd.fps)}: " + ("BUKAN ANOMALI" if new else "dibatalkan"))

    def current_item(self, items):
        """Baris yang dipilih terakhir, atau yang memuat posisi sekarang."""
        f = self.frame
        sel = next((it for it in items if it["id"] == self.sel_item), None)
        if sel is not None:
            return sel
        inside = [it for it in items if it["a"] - self.vd.fps <= f <= it["b"]]
        return inside[-1] if inside else None

    def step_item(self, d):
        items = self.panel_items()
        if not items:
            self.flash("Daftar anomali kosong (cek filter dengan f).")
            return
        f = self.frame
        cur = next((k for k, it in enumerate(items) if it["id"] == self.sel_item), None)
        if cur is not None:
            k = cur + d
        elif d > 0:
            k = next((k for k, it in enumerate(items) if it["a"] > f), len(items))
        else:
            k = max((k for k, it in enumerate(items) if it["a"] < f - 1), default=-1)
        if not 0 <= k < len(items):
            self.flash("Tidak ada baris " + ("berikutnya." if d > 0 else "sebelumnya."))
            return
        self.panel_scroll_t = 0.0
        self.open_item(items[k])

    def panel_rows(self, H):
        return max(3, (H - LIST_Y0 - 46) // ROW_H)

    def draw_panel(self, H):
        items = self.panel_items()
        P = np.full((H, PANEL_W, 3), 32, np.uint8)
        cv2.line(P, (0, 0), (0, H), (90, 90, 90), 1)
        put_text(P, "ppy ANOMALI", (10, 22), 0.6, C_HEAD, 2)
        put_text(P, f"filter (f): {FILTERS[self.filter_i][0]}   {len(items)} baris", (10, 44), 0.45, (220, 220, 220))
        put_text(P, "klik = lihat | klik kanan / z = bukan anomali", (10, 64), 0.4, (170, 170, 170))
        n_rows = self.panel_rows(H)
        cur = self.current_item(items)
        if cur is not None and time.time() - self.panel_scroll_t > 3.0:      # ikuti posisi kecuali sedang digulir
            k = items.index(cur)
            if not self.panel_scroll <= k < self.panel_scroll + n_rows:
                self.panel_scroll = k - n_rows // 3
        self.panel_scroll = int(np.clip(self.panel_scroll, 0, max(0, len(items) - n_rows)))
        if not items:
            put_text(P, "(tidak ada baris untuk filter ini)", (10, LIST_Y0 + 20), 0.45, (170, 170, 170))
        for r, it in enumerate(items[self.panel_scroll:self.panel_scroll + n_rows]):
            y = LIST_Y0 + r * ROW_H
            if cur is not None and it["id"] == cur["id"]:
                cv2.rectangle(P, (2, y), (PANEL_W - 2, y + ROW_H - 2), (90, 70, 20), -1)
            cv2.rectangle(P, (4, y + 3), (9, y + ROW_H - 5), it["col"], -1)
            dim = it["dismissed"]
            col = (110, 110, 110) if dim else ((255, 255, 255) if it["kind"] == "saya" else (225, 225, 225))
            dur = (it["b"] - it["a"]) / self.vd.fps
            tt = fmt_t(it["a"] / self.vd.fps) + (f" +{dur:.1f}s" if dur >= 0.1 else "")
            put_text(P, tt, (14, y + 15), 0.4, col)
            put_text(P, f"{KIND_TXT[it['kind']][0]} {it['label'][:18]}", (118, y + 15), 0.4,
                     (110, 110, 110) if dim else it["col"])
            tag, tcol = "", (150, 150, 150)
            if dim:
                tag, tcol = "BUKAN", (140, 140, 140)
            elif it["kind"] != "saya" and it["marked"]:
                tag, tcol = "DITANDAI", (0, 200, 0)
            elif it["seen"]:
                tag, tcol = "dicek", (170, 170, 170)
            if tag:
                put_text(P, tag, (PANEL_W - 70, y + 15), 0.38, tcol)
        if len(items) > n_rows:                                             # scrollbar
            y0, y1 = LIST_Y0, LIST_Y0 + n_rows * ROW_H
            a = y0 + (y1 - y0) * self.panel_scroll // len(items)
            b = y0 + (y1 - y0) * min(len(items), self.panel_scroll + n_rows) // len(items)
            cv2.rectangle(P, (PANEL_W - 5, a), (PANEL_W - 2, b), (150, 150, 150), -1)
        alls = self.panel_items_count()
        put_text(P, f"S saya {alls['saya']} | O otomatis {alls['otomatis']} | P pola {alls['pola']}",
                 (10, H - 28), 0.4, (200, 200, 200))
        put_text(P, f"sudah dievaluasi {alls['done']}/{alls['auto']} (dicek / ditandai / bukan)", (10, H - 10), 0.4,
                 (200, 200, 200))
        if cur is not None:
            put_text(P, cur["detail"][:44], (10, H - 46), 0.4, cur["col"])
        return P

    def panel_items_count(self):
        keep, self.filter_i = self.filter_i, 0
        items = self.panel_items()
        self.filter_i = keep
        c = Counter(it["kind"] for it in items)
        auto = [it for it in items if it["kind"] != "saya"]
        c["auto"] = len(auto)
        c["done"] = sum(1 for it in auto if it["seen"] or it["dismissed"] or it["marked"])
        return c

    def panel_click(self, x, y, right):
        H = HEADER_H + self.dh + BAR_H
        if y < LIST_Y0:
            if y < 50 and not right:
                self.filter_i = (self.filter_i + 1) % len(FILTERS)
                self.panel_scroll = 0
            return
        r = (y - LIST_Y0) // ROW_H
        items = self.panel_items()
        if r >= self.panel_rows(H) or self.panel_scroll + r >= len(items):
            return
        it = items[self.panel_scroll + r]
        if right:
            self.dismiss_item(it)
        else:
            self.panel_scroll_t = time.time()                 # jangan geser daftar tepat saat diklik
            self.open_item(it)

    # ---------------------------------------------------------------- drawing
    def compose(self):
        vd, f = self.vd, self.frame
        W = max(self.dw, MIN_W)
        canvas = np.zeros((HEADER_H + self.dh + BAR_H, W, 3), np.uint8)
        jpg = self.cache.jpg.get(f)
        if jpg is not None:
            img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        else:
            img = np.zeros((self.dh, self.dw, 3), np.uint8)
            pct = 100 * max(0, self.cache.read_upto) / max(1, self.cache.last)
            put_text(img, f"memuat video... {pct:.0f}%", (20, self.dh // 2), 0.8, C_WARN, 2)
        self.watched[self.pos] = True
        s = self.scale
        pat = int(vd.patient[self.pos])
        boxes = vd.boxes.get(f, [])
        pbox = next((b for t, b in boxes if t == pat), None)
        for t, b in boxes:
            p = (np.asarray(b) * s).astype(int)
            is_p = t == pat
            col = C_PAT if is_p else C_OBS
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), col, 4 if is_p else 2)
            ty = max(16, p[1] - 6) if is_p else min(self.dh - 6, p[3] + 20)       # patient above, observers below
            put_text(img, f"PASIEN #{t}" if is_p else f"OBS #{t}", (p[0], ty), 0.6, col, 2)
        if f in vd.anchors and vd.anchors[f] is not None:
            p = (np.asarray(vd.anchors[f]) * s).astype(int)
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), C_LABEL, 2)
            put_text(img, "LABEL", (min(self.dw - 60, p[2] + 4), p[1] + 18), 0.55, C_LABEL, 2)
        cues = []                                                   # quick visual cues while watching
        if pbox is not None:
            ov = max((iou(pbox, b) for t, b in boxes if t != pat), default=0.0)
            if ov >= ENC_IOU:
                cues.append(f"PASIEN & OBS TUMPANG TINDIH (IoU {ov:.2f})")
            if vd.med_w and (pbox[2] - pbox[0]) / vd.med_w >= WIDE_RATIO:
                cues.append(f"KOTAK PASIEN {(pbox[2] - pbox[0]) / vd.med_w:.1f}x LEBIH LEBAR DARI BIASA (menyatu dengan observer?)")
            if vd.med_h and (pbox[3] - pbox[1]) / vd.med_h <= SHORT_RATIO:
                cues.append(f"KOTAK PASIEN {(pbox[3] - pbox[1]) / vd.med_h:.1f}x TINGGI BIASA (tertutup?)")
        elif boxes:
            cues.append("TIDAK ADA KOTAK PASIEN")
        for k, txt in enumerate(cues):
            put_text(img, txt, (10, 26 + 26 * k), 0.65, (60, 60, 255), 2)
        st = self.vrev.rows.get(self.vid, {}).get("status", "belum_selesai")   # verdict badge, top-right
        badge = STATUS_TXT.get(st, st) + (f" ({len(self.video_anoms())})" if st == "ada_anomali" else "")
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx = self.dw - tw - 24
        if st == "belum_selesai":
            cv2.rectangle(img, (bx - 6, 8), (self.dw - 10, 16 + th + 8), STATUS_COL[st], 1)
            put_text(img, badge, (bx, 14 + th), 0.7, STATUS_COL[st], 2)
        else:
            cv2.rectangle(img, (bx - 6, 8), (self.dw - 10, 16 + th + 8), STATUS_COL.get(st, (128, 128, 128)), -1)
            cv2.putText(img, badge, (bx, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if img.shape[:2] != (self.dh, self.dw):          # safety: video size differs from the metadata
            img = cv2.resize(img, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        canvas[HEADER_H:HEADER_H + self.dh, :self.dw] = img
        self.draw_header(canvas)
        self.draw_bar(canvas)
        if self.show_list:
            canvas = self.draw_list(canvas)
        if self.show_help:
            y0 = HEADER_H + self.dh - 24 * len(HELP) - 8
            ov = canvas.copy()
            cv2.rectangle(ov, (0, y0 - 18), (W, HEADER_H + self.dh), (0, 0, 0), -1)
            canvas = cv2.addWeighted(ov, 0.82, canvas, 0.18, 0)
            for i, line in enumerate(HELP):
                put_text(canvas, line, (8, y0 + 24 * i), 0.47)
        if time.time() < self.msg_until:
            put_text(canvas, self.msg, (8, HEADER_H + self.dh - 14), 0.6, (0, 255, 255), 2)
        self.main_w = canvas.shape[1]
        if self.show_panel:
            canvas = np.hstack([canvas, self.draw_panel(canvas.shape[0])])
        return canvas

    def draw_header(self, c):
        vd = self.vd
        status = self.vrev.rows.get(self.vid, {}).get("status", "belum_selesai")
        an = self.video_anoms()
        cnt = Counter(TYPE_BY_NAME[r["type"]][1] for r in an)
        st_col = STATUS_COL.get(status, (200, 200, 200))
        put_text(c, f"{self.vid}  [{self.vidx + 1}/{len(self.videos)}]   {fmt_t(self.t)} / {fmt_t(vd.duration)}   "
                    f"{SPEEDS[self.speed_i]:g}x  {'PUTAR' if self.playing else 'JEDA'}", (8, 22), 0.6)
        put_text(c, f"penilaian: {STATUS_TXT.get(status, status)}", (8, 44), 0.55, st_col, 2)
        put_text(c, f"tanda: {len(an)}" + (" (" + ", ".join(f"{k} {v}" for k, v in cnt.items()) + ")" if an else ""),
                 (290, 44), 0.5)
        if self.rec and self.rec in self.anoms.rows:
            r = self.anoms.rows[self.rec]
            k, lab, col = TYPE_BY_NAME[r["type"]]
            put_text(c, f"* MEREKAM {lab} sejak {fmt_t(num(r['start_frame']) / vd.fps)} - tekan {chr(k)} lagi untuk "
                        "selesai, Esc batal", (8, 64), 0.5, col, 2)
        else:
            here = self.anom_here(tol_s=0.0)
            if here:
                k, lab, col = TYPE_BY_NAME[here["type"]]
                extra = ""
                if here["type"] == "label_pindah_ke_observer":
                    extra = f"  pasien benar = {here['true_patient_track'] or '(klik kotaknya)'}"
                elif here["type"] == "observer_mengisi_kotak_pasien":
                    extra = f"  observer = {here['observer_track'] or '(klik kotaknya)'}"
                put_text(c, f"di dalam tanda {lab} ({here['start_s']}-{here['end_s']} s){extra}   x hapus, c catatan",
                         (8, 64), 0.5, col)
            else:
                put_text(c, "1 pindah | 2 observer mengisi kotak pasien | 3 hilang | 4 lain || y VIDEO OK | o selesai | "
                            "u batal || PgUp/PgDn video | g belum dinilai | i daftar | h bantuan", (8, 64), 0.45,
                         (190, 190, 190))
        put_text(c, f"cache {self.cache.bytes / 1e6:.0f} MB" + ("" if self.cache.done else " (memuat)"),
                 (c.shape[1] - 170, 22), 0.45, (150, 150, 150))

    def bar_x(self, W):
        return 10, W - 10

    def x_of_pos(self, p, x0, x1):
        return int(x0 + (x1 - x0) * p / max(1, len(self.vd.frames) - 1))

    def draw_bar(self, c):
        vd = self.vd
        y0, W = HEADER_H + self.dh, c.shape[1]
        c[y0:, :] = (25, 25, 25)
        x0, x1 = self.bar_x(W)
        n = len(vd.frames)
        if n == 0:
            return
        if not self.video_anoms():
            put_text(c, "tanda Anda (belum ada)", (x0, y0 + 15), 0.38, (170, 170, 170))
        for r in self.video_anoms():                                    # row A: your marks
            a_, b_ = vd.pos_at(num(r["start_frame"])), vd.pos_at(num(r["end_frame"]))
            if r["anomaly_id"] == self.rec:
                b_ = max(a_, self.pos)
            xa = self.x_of_pos(a_, x0, x1)
            xb = max(self.x_of_pos(b_, x0, x1), xa + 5)
            col = TYPE_BY_NAME[r["type"]][2]
            cv2.rectangle(c, (xa, y0 + 20), (xb, y0 + 32), col, -1)
            put_text(c, TYPE_BY_NAME[r["type"]][1], (xa, y0 + 15), 0.38, col)
        has = vd.patient >= 0                                           # row B: patient present / absent
        xs = np.linspace(x0, x1, n + 1).astype(int)
        start = 0
        for k in range(1, n + 1):
            if k == n or has[k] != has[start]:
                cv2.rectangle(c, (xs[start], y0 + 38), (max(xs[start] + 1, xs[k] - 1), y0 + 54),
                              C_BAR if has[start] else C_NONE, -1)
                start = k
        for p in vd.id_changes:
            xx = self.x_of_pos(p, x0, x1)
            cv2.line(c, (xx, y0 + 35), (xx, y0 + 57), (255, 255, 255), 2)
        dec_pos = int(np.searchsorted(vd.frames, self.cache.read_upto, "right")) - 1     # decoded so far
        cv2.line(c, (x0, y0 + 57), (max(x0, self.x_of_pos(max(0, dec_pos), x0, x1)), y0 + 57), (110, 110, 110), 1)
        w, start = self.watched, None                                    # watched: green
        for k in range(n + 1):
            on = k < n and w[k]
            if on and start is None:
                start = k
            if not on and start is not None:
                cv2.line(c, (self.x_of_pos(start, x0, x1), y0 + 59), (self.x_of_pos(k - 1, x0, x1), y0 + 59), (0, 180, 0), 2)
                start = None
        for f, pr, _ in vd.events:                                       # row C: automatic suspects
            xx = self.x_of_pos(vd.pos_at(f), x0, x1)
            cv2.drawMarker(c, (xx, y0 + 69), (60, 60, 255) if pr >= 2 else (150, 150, 150), cv2.MARKER_TRIANGLE_UP,
                           11 if pr >= 2 else 8, 2)
        for f in vd.anchors:                                             # your 8 labelled frames
            if f in vd.pos_of:
                xx = self.x_of_pos(vd.pos_of[f], x0, x1)
                cv2.line(c, (xx, y0 + 63), (xx, y0 + 75), C_LABEL, 2)
        xp = self.x_of_pos(self.pos, x0, x1)
        cv2.line(c, (xp, y0 + 15), (xp, y0 + 77), C_HEAD, 2)
        put_text(c, "biru = ada PASIEN | abu = tidak ada | putih = ID pasien berganti | hijau = sudah ditonton | "
                    "segitiga = dugaan otomatis | magenta = frame label | klik = lompat",
                 (x0, y0 + BAR_H - 6), 0.36, (170, 170, 170))

    def list_rows(self):
        n_rows = max(5, min(len(self.videos), (self.dh - 60) // 20))
        lo = int(np.clip(self.list_sel - n_rows // 2, 0, max(0, len(self.videos) - n_rows)))
        return list(range(lo, min(len(self.videos), lo + n_rows)))

    def draw_list(self, canvas):
        rows = self.list_rows()
        st = Counter(self.vrev.rows.get(v, {}).get("status", "belum_selesai") for v in self.videos)
        h = 34 + 20 * len(rows) + 8
        ov = canvas.copy()
        cv2.rectangle(ov, (0, HEADER_H), (canvas.shape[1], HEADER_H + h), (0, 0, 0), -1)
        canvas = cv2.addWeighted(ov, 0.87, canvas, 0.13, 0)
        put_text(canvas, f"Daftar video: OK {st['ok']} | ADA ANOMALI {st['ada_anomali']} | BELUM {st['belum_selesai']}"
                         "    (atas/bawah pilih, Enter / klik buka, i tutup)", (8, HEADER_H + 20), 0.5, C_WARN)
        for k, i in enumerate(rows):
            v = self.videos[i]
            r = self.vrev.rows.get(v, {})
            s_ = r.get("status", "belum_selesai")
            detail = ", ".join(f"{TYPE_BY_NAME[t][1]} {r.get(t)}" for t in TYPE_BY_NAME if num(r.get(t), 0) > 0)
            y = HEADER_H + 42 + 20 * k
            if i == self.list_sel:
                cv2.rectangle(canvas, (4, y - 15), (canvas.shape[1] - 4, y + 5), (90, 70, 20), -1)
            mark = ">" if i == self.vidx else " "
            put_text(canvas, f"{mark}{i + 1:4d}  {v:<20} {STATUS_TXT.get(s_, s_):<14} ditonton "
                             f"{r.get('watched_pct') or 0:>3}%   {detail}", (8, y), 0.45,
                     STATUS_COL.get(s_, (230, 230, 230)), 2 if i == self.vidx else 1)
        return canvas

    # ---------------------------------------------------------------- input
    def on_mouse(self, ev, x, y, flags, _p):
        y_bar = HEADER_H + self.dh
        main_w = getattr(self, "main_w", max(self.dw, MIN_W))
        if self.show_panel and x >= main_w and not self.dragging:
            if ev == cv2.EVENT_MOUSEWHEEL:
                self.panel_scroll += -3 if flags > 0 else 3
                self.panel_scroll_t = time.time()
            elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                self.panel_click(x - main_w, y, ev == cv2.EVENT_RBUTTONUP)
            return
        if self.show_list and ev == cv2.EVENT_LBUTTONUP and HEADER_H + 26 <= y < y_bar:
            rows = self.list_rows()
            k = (y - (HEADER_H + 42 - 15)) // 20
            if 0 <= k < len(rows):
                self.show_list = False
                self.switch_to(rows[k])
            return
        x0, x1 = self.bar_x(max(self.dw, MIN_W))
        if y >= y_bar and ev == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
        if self.dragging and ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            frac = float(np.clip((x - x0) / max(1, x1 - x0), 0, 1))
            self.seek_pos(int(round(frac * (len(self.vd.frames) - 1))))
            return
        if ev == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                return
            if HEADER_H <= y < y_bar and x < self.dw:
                fx, fy = x / self.scale, (y - HEADER_H) / self.scale
                hits = [(t, b) for t, b in self.vd.boxes.get(self.frame, []) if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]]
                if hits:
                    self.click_track(min(hits, key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]))[0])

    def handle_key(self, key):
        """Returns 'quit' to stop, otherwise None."""
        if key == ord("q"):
            return "quit"
        if key == 27:
            if self.rec:
                self.delete_here()
            elif self.show_help or self.show_list:
                self.show_help = self.show_list = False
            else:
                return "quit"
            return None
        if self.show_list and (key in KEY_UP or key in KEY_DOWN or key in KEY_ENTER):
            if key in KEY_ENTER:
                self.show_list = False
                self.switch_to(self.list_sel)
            else:
                self.list_sel = int(np.clip(self.list_sel + (1 if key in KEY_DOWN else -1), 0, len(self.videos) - 1))
            return None
        if key == ord(" "):
            self.playing = not self.playing
            self.last_t, self.acc = time.time(), 0.0
            if self.playing and self.pos >= len(self.vd.frames) - 1:
                self.seek_pos(0)
        elif key in TYPES:
            self.on_type_key(key)
        elif key in KEY_UP or key in (ord("+"), ord("=")):
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
            self.flash(f"Kecepatan {SPEEDS[self.speed_i]:g}x")
        elif key in KEY_DOWN or key == ord("-"):
            self.speed_i = max(0, self.speed_i - 1)
            self.flash(f"Kecepatan {SPEEDS[self.speed_i]:g}x")
        elif key in KEY_LEFT or key == ord("a"):
            self.seek_s(-2)
        elif key in KEY_RIGHT or key == ord("d"):
            self.seek_s(2)
        elif key == ord("j"):
            self.seek_s(-10)
        elif key == ord("l"):
            self.seek_s(10)
        elif key == ord(","):
            self.playing = False
            self.seek_pos(self.pos - 1)
        elif key == ord("."):
            self.playing = False
            self.seek_pos(self.pos + 1)
        elif key == ord("0") or key in KEY_HOME:
            self.seek_pos(0)
        elif key == ord("x"):
            self.delete_here()
        elif key == ord("c"):
            self.note()
        elif key == ord("["):
            self.jump_anom(-1)
        elif key == ord("]"):
            self.jump_anom(1)
        elif key == ord("n"):
            self.jump_event(1)
        elif key == ord("p"):
            self.jump_event(-1)
        elif key == ord("y"):
            self.set_verdict("ok")
        elif key == ord("o"):
            self.set_verdict("done")
        elif key == ord("u"):
            self.set_verdict("undo")
        elif key == ord("g"):
            self.go_todo()
        elif key == ord("e"):
            self.show_panel = not self.show_panel
        elif key == ord("f"):
            self.filter_i = (self.filter_i + 1) % len(FILTERS)
            self.panel_scroll = 0
            self.flash(f"Filter daftar anomali: {FILTERS[self.filter_i][0]}")
        elif key == ord("t"):
            self.step_item(1)
        elif key == ord("r"):
            self.step_item(-1)
        elif key == ord("z"):
            it = self.current_item(self.panel_items())
            if it is None:
                self.flash("Pilih baris di daftar anomali dulu (klik, atau t / r).")
            else:
                self.dismiss_item(it)
        elif key in KEY_PGDN or key == ord("m"):
            self.go_video(1)
        elif key in KEY_PGUP or key in KEY_BACKSPACE or key == ord("b"):
            self.go_video(-1)
        elif key == ord("k"):
            self.goto_video(self.ask(f"\nPilih video: nomor 1-{len(self.videos)} atau sebagian nama:"))
        elif key == ord("i"):
            self.show_list = not self.show_list
            self.list_sel = self.vidx
        elif key == ord("h"):
            self.show_help = not self.show_help
        return None

    # ---------------------------------------------------------------- main loop
    def run(self):
        if not self.videos:
            print("Tidak ada video untuk ditinjau (cek --video / --show, atau video asli tidak ditemukan di --data).")
            return
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.last_t = time.time()
        try:
            while True:
                self.tick()
                cv2.imshow(WIN, self.compose())
                key = cv2.waitKeyEx(15)
                if key != -1 and self.handle_key(key) == "quit":
                    break
        finally:
            self.close()

    def close(self):
        if self.vd is not None:
            if self.rec:
                self.close_rec()
            self.update_video_row(self.vid)
        if self.cache is not None:
            self.cache.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.print_summary()

    def print_summary(self):
        st = Counter(self.vrev.rows.get(v, {}).get("status", "belum_selesai") for v in self.videos)
        ty = Counter(r["type"] for r in self.anoms.rows.values())
        print("\n=== Ringkasan ===")
        print(f"Video      : {len(self.videos)}  ->  " + ", ".join(f"{k}={v}" for k, v in st.items()))
        print(f"Tanda      : {len(self.anoms.rows)}  ->  " + ", ".join(f"{k}={v}" for k, v in ty.items()))
        print(f"File       : {self.anoms.path}\n             {self.vrev.path}")
        for b in self.backups:
            print(f"Cadangan   : {b}")
        sv = [r for r in self.susp.rows.values() if r["video_id"] in set(self.videos)]
        print(f"Panel      : {sum(r.get('seen') == '1' for r in sv)} baris dicek, "
              f"{sum(r.get('decision') == 'bukan_anomali' for r in sv)} ditandai bukan anomali  ({self.susp.path})")
        print("Lanjutkan analisis: jalankan sel 14 di 02b_analysis.ipynb")


def main():
    ap = argparse.ArgumentParser(description="Pemutar video untuk menandai anomali identitas pasien per video.")
    ap.add_argument("--identity", default=None, help="folder outputs/02b_analysis/<RUN>/identity (default: otomatis)")
    ap.add_argument("--data", default="data", help="folder yang berisi video asli (dicari di semua subfolder)")
    ap.add_argument("--video", default=None, help="hanya video yang namanya mengandung teks ini")
    ap.add_argument("--show", default="all", choices=["all", "todo", "done"],
                    help="semua video, hanya yang belum selesai (todo), atau hanya yang sudah selesai (done)")
    ap.add_argument("--start", default=None, help="mulai dari video ini (nomor 1-based atau nama)")
    ap.add_argument("--no-auto-next", action="store_true", help="jangan pindah video otomatis setelah y / o")
    ap.add_argument("--from-start", action="store_true", help="mulai dari video 1 (default: video pertama yang belum selesai)")
    ap.add_argument("--max-w", type=int, default=1300, help="lebar jendela maksimum, termasuk panel (piksel)")
    ap.add_argument("--no-panel", action="store_true", help="sembunyikan panel daftar anomali (lebih lebar untuk video)")
    ap.add_argument("--max-h", type=int, default=680, help="tinggi tampilan video maksimum (piksel)")
    ap.add_argument("--jpeg-quality", type=int, default=80, help="kualitas cache frame (lebih kecil = hemat memori)")
    ap.add_argument("--reviewer", default=getpass.getuser(), help="nama peninjau (disimpan di CSV)")
    args = ap.parse_args()

    ident = find_identity_dir(args.identity)
    summary = pd.read_csv(ident / "identity_summary.csv", dtype={"video_id": str})
    ev_path, an_path = ident / "events.csv", ident / "anchors.csv"
    events = pd.read_csv(ev_path, dtype={"video_id": str}) if ev_path.exists() and ev_path.stat().st_size > 1 \
        else pd.DataFrame(columns=["video_id", "frame_idx", "priority", "types"])
    anchors = pd.read_csv(an_path, dtype={"video_id": str}) if an_path.exists() and an_path.stat().st_size > 1 \
        else pd.DataFrame(columns=["video_id", "frame_idx", "x1", "y1", "x2", "y2"])
    videos = find_videos(args.data, set(summary["video_id"]))
    print(f"Hasil identitas : {ident}")
    print(f"Video asli      : {len(videos)}/{len(summary)} ditemukan di {Path(args.data).resolve()}")
    pl = Player(args, ident, summary, events, anchors, videos)
    if pl.imported:
        print(f"Tinjauan lama   : {pl.imported} tanda diimpor dari switch_review.csv")
    print(f"{len(pl.videos)} video. Tekan h di jendela untuk bantuan.")
    pl.run()


if __name__ == "__main__":
    main()
