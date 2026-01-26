#!/usr/bin/env python3
"""
Flow-Lenia classic tracker (no neural nets):
- Per-frame CC detection on HSV value/hue
- Track-by-detection association
- Grouping into organisms
- Eating rule
- Overlays and JSON tracks
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
from PIL import Image

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# --------------------------- Config ------------------------------------- #
@dataclass
class Config:
    # detection
    det_v_thr_hi: int = 55
    h_bins: int = 24
    min_area: int = 20
    min_mass: float = 500.0
    use_hue_bins: bool = True
    marker_split: bool = True
    seed_radius: int = 3

    # tracker
    max_dist: float = 60.0
    min_iou: float = 0.05
    max_missed: int = 10
    w_dist: float = 1.0
    w_iou: float = 80.0
    w_area: float = 10.0
    w_col: float = 30.0
    strict_color: bool = False
    iou_zero_dist_gate: Optional[float] = None  # if None -> 0.7*max_dist
    iou_dilate_r: int = 2
    hue_dist_max: float = 0.5  # normalized (0..1) distance in hue-bin space for strict_color

    # grouping
    group_window: int = 50
    tau_sigma: float = 3.0
    tau_dist: float = 40.0

    # eating
    close_r: int = 4
    eta_eat: float = 0.6
    eat_confirm_frames: int = 3

    # outputs
    save_parts_debug: bool = True
    draw_ids_largest_k: int = 0


# --------------------------- Video I/O ----------------------------------- #
class VideoLoader:
    @staticmethod
    def get_video_info(path: str) -> Tuple[float, int, Tuple[int, int]]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, frame_count, (width, height)

    @staticmethod
    def load_mp4_frames(
        path: str,
        stride: int = 1,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> List[Image.Image]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {path}")
        frames: List[Image.Image] = []
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                if resize is not None:
                    frame_bgr = cv2.resize(frame_bgr, resize, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
        cap.release()
        return frames


# --------------------------- Detection ----------------------------------- #
@dataclass
class Detection:
    mask_u8: np.ndarray
    hue_bin: int
    area: int
    mass: float
    cx: float
    cy: float
    bbox: Tuple[int, int, int, int]
    seed_track_id: Optional[int] = None


class MaskGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate_detections(
        self,
        rgb_u8: np.ndarray,
        v_thr_hi: int,
        seeds: Optional[List[Tuple[int, float, float]]] = None,
        use_marker_split: bool = False,
        seed_radius: int = 3,
    ) -> Tuple[List[Detection], Dict[int, Detection]]:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        v = hsv[:, :, 2]
        strong = v > v_thr_hi
        if not np.any(strong):
            return [], {}
        if use_marker_split and seeds:
            return self._marker_split(h, v, strong, seeds, seed_radius)

        detections: List[Detection] = []
        use_bins = self.cfg.use_hue_bins
        if use_bins:
            h_bin = (h.astype(np.int32) * self.cfg.h_bins) // 180
            for b in range(self.cfg.h_bins):
                mask_b = strong & (h_bin == b)
                detections.extend(self._components(mask_b, v, h, b))
        else:
            detections.extend(self._components(strong, v, h, -1))
        return detections, {}

    def _mask_to_detection(
        self,
        mask_bin: np.ndarray,
        v: np.ndarray,
        h: np.ndarray,
        hue_bin_override: Optional[int] = None,
    ) -> Optional[Detection]:
        if not np.any(mask_bin):
            return None
        area = int(mask_bin.sum())
        if area < self.cfg.min_area:
            return None
        mass = float(v[mask_bin].sum())
        if mass < self.cfg.min_mass:
            return None
        ys, xs = np.where(mask_bin)
        cx = float(xs.mean())
        cy = float(ys.mean())
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        hue_bin = -1
        if self.cfg.use_hue_bins:
            if hue_bin_override is not None and hue_bin_override >= 0:
                hue_bin = hue_bin_override
            else:
                bins = (h.astype(np.int32) * self.cfg.h_bins) // 180
                vals = bins[mask_bin]
                if vals.size > 0:
                    hue_bin = int(np.bincount(vals.flatten(), minlength=self.cfg.h_bins).argmax())
        return Detection(
            mask_u8=mask_bin.astype(np.uint8),
            hue_bin=hue_bin,
            area=area,
            mass=mass,
            cx=cx,
            cy=cy,
            bbox=(x0, y0, x1, y1),
        )

    def _components(
        self, mask_bin: np.ndarray, v: np.ndarray, h: np.ndarray, hue_bin: int
    ) -> List[Detection]:
        if not np.any(mask_bin):
            return []
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            mask_bin.astype(np.uint8), connectivity=8
        )
        outs: List[Detection] = []
        for comp_id in range(1, num_labels):
            comp_mask = labels == comp_id
            det = self._mask_to_detection(comp_mask, v, h, hue_bin_override=hue_bin if hue_bin >= 0 else None)
            if det is not None:
                outs.append(det)
        return outs

    def _marker_split(
        self,
        h: np.ndarray,
        v: np.ndarray,
        strong: np.ndarray,
        seeds: List[Tuple[int, float, float]],
        seed_radius: int,
    ) -> Tuple[List[Detection], Dict[int, Detection]]:
        strong_u8 = strong.astype(np.uint8)
        hgt, wdt = strong.shape
        markers = np.zeros_like(strong_u8, dtype=np.int32)
        seed_ids: List[int] = []
        for tid, cx, cy in seeds:
            xi = int(round(cx))
            yi = int(round(cy))
            if xi < 0 or xi >= wdt or yi < 0 or yi >= hgt:
                continue
            cv2.circle(markers, (xi, yi), max(1, seed_radius), int(tid), -1)
            seed_ids.append(int(tid))

        num_labels, comp_labels = cv2.connectedComponents(strong_u8)
        next_label = (max(seed_ids) + 1) if seed_ids else 1
        for comp_id in range(1, num_labels):
            comp_mask = comp_labels == comp_id
            if np.any(markers[comp_mask] > 0):
                continue
            markers[comp_mask] = next_label
            next_label += 1

        dist = cv2.distanceTransform(strong_u8, cv2.DIST_L2, 5)
        elevation = cv2.normalize(-dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elevation_color = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
        markers_ws = markers.copy()
        cv2.watershed(elevation_color, markers_ws)
        markers_ws[~strong] = 0

        detections: List[Detection] = []
        seeded: Dict[int, Detection] = {}
        seed_set = set(seed_ids)
        labels = np.unique(markers_ws)
        for lbl in labels:
            if lbl <= 0:
                continue
            region = markers_ws == lbl
            det = self._mask_to_detection(region, v, h, None)
            if det is None:
                continue
            if lbl in seed_set:
                det.seed_track_id = int(lbl)
                seeded[int(lbl)] = det
            else:
                detections.append(det)
        return detections, seeded


# --------------------------- Metrics ------------------------------------- #
def iou_u8(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    if union == 0:
        return 0.0
    return inter / union


def area_ratio(a: float, b: float) -> float:
    eps = 1e-6
    return abs(math.log((a + eps) / (b + eps)))


def dilate_mask(mask_u8: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask_u8
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_u8.astype(np.uint8), kernel)


def bbox_from_mask(mask_u8: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def roi_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])
    if x1 < x0 or y1 < y0:
        return 0.0
    a_roi = mask_a[y0 : y1 + 1, x0 : x1 + 1]
    b_roi = mask_b[y0 : y1 + 1, x0 : x1 + 1]
    inter = float((a_roi.astype(np.uint8) & b_roi.astype(np.uint8)).sum())
    union = float((a_roi.astype(np.uint8) | b_roi.astype(np.uint8)).sum())
    if union == 0:
        return 0.0
    return inter / union


# --------------------------- TrackerClassic ------------------------------ #
@dataclass
class Track:
    track_id: int
    mask_u8: np.ndarray
    cx: float
    cy: float
    area: float
    hue_bin: int
    vx: float
    vy: float
    last_frame: int
    missed: int
    bbox: Tuple[int, int, int, int]


class TrackerClassic:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def predict_seeds(self, frame_idx: int) -> List[Tuple[int, float, float]]:
        seeds: List[Tuple[int, float, float]] = []
        for tid, tr in self.tracks.items():
            if tr.missed > self.cfg.max_missed:
                continue
            dt_pred = max(1, frame_idx - tr.last_frame)
            cx_pred = tr.cx + tr.vx * dt_pred
            cy_pred = tr.cy + tr.vy * dt_pred
            seeds.append((tid, cx_pred, cy_pred))
        return seeds

    def _update_track_from_det(self, tid: int, det: Detection, frame_idx: int) -> None:
        old = self.tracks[tid]
        dt_obs = max(1, frame_idx - old.last_frame)
        vx = (det.cx - old.cx) / dt_obs
        vy = (det.cy - old.cy) / dt_obs
        self.tracks[tid] = Track(
            track_id=tid,
            mask_u8=det.mask_u8,
            cx=det.cx,
            cy=det.cy,
            area=det.area,
            hue_bin=det.hue_bin,
            vx=vx,
            vy=vy,
            last_frame=frame_idx,
            missed=0,
            bbox=det.bbox,
        )

    def update(
        self,
        frame_idx: int,
        detections: List[Detection],
        seeded: Optional[Dict[int, Detection]] = None,
    ) -> Dict[int, np.ndarray]:
        seeded = seeded or {}
        track_ids_all = list(self.tracks.keys())
        if not track_ids_all and not detections and not seeded:
            return {}

        unmatched_tracks: Set[int] = set(track_ids_all)
        unmatched_dets: Set[int] = set(range(len(detections)))

        # apply seeded matches (marker-based watershed regions)
        for tid, det in seeded.items():
            if tid not in self.tracks:
                continue
            self._update_track_from_det(tid, det, frame_idx)
            unmatched_tracks.discard(tid)

        track_list = list(unmatched_tracks)
        T = len(track_list)
        D = len(detections)
        matches: List[Tuple[int, int]] = []
        if T > 0 and D > 0:
            # spatial grid on detections
            cell = max(1.0, self.cfg.max_dist)
            grid: Dict[Tuple[int, int], List[int]] = {}
            for idx, det in enumerate(detections):
                gx = int(det.cx // cell)
                gy = int(det.cy // cell)
                grid.setdefault((gx, gy), []).append(idx)

            dilated_cache: Dict[int, np.ndarray] = {}
            dilated_bbox_cache: Dict[int, Tuple[int, int, int, int]] = {}
            for tid in track_list:
                dil = dilate_mask(self.tracks[tid].mask_u8, self.cfg.iou_dilate_r)
                dilated_cache[tid] = dil
                dilated_bbox_cache[tid] = bbox_from_mask(dil)

            cost = np.full((T, D), np.inf, dtype=np.float32)
            for row, tid in enumerate(track_list):
                tr = self.tracks[tid]
                dt_pred = max(1, frame_idx - tr.last_frame)
                cx_pred = tr.cx + tr.vx * dt_pred
                cy_pred = tr.cy + tr.vy * dt_pred
                cell_x = int(cx_pred // cell)
                cell_y = int(cy_pred // cell)
                candidates: List[int] = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        candidates.extend(grid.get((cell_x + dx, cell_y + dy), []))
                seen: Set[int] = set()
                for det_idx in candidates:
                    if det_idx in seen:
                        continue
                    seen.add(det_idx)
                    det = detections[det_idx]
                    dist = math.hypot(cx_pred - det.cx, cy_pred - det.cy)
                    if dist > self.cfg.max_dist:
                        continue
                    col_pen = 0.0
                    if self.cfg.use_hue_bins and tr.hue_bin >= 0 and det.hue_bin >= 0:
                        bins = self.cfg.h_bins
                        d_bin = abs(tr.hue_bin - det.hue_bin)
                        d_bin = min(d_bin, bins - d_bin)
                        dist_norm = d_bin / (bins / 2.0)
                        if self.cfg.strict_color and dist_norm > self.cfg.hue_dist_max:
                            continue
                        col_pen = self.cfg.w_col * dist_norm
                    ar = area_ratio(tr.area, det.area)
                    c_val = self.cfg.w_dist * dist + self.cfg.w_area * ar + col_pen
                    if self.cfg.w_iou != 0:
                        iou = roi_iou(dilated_cache[tid], det.mask_u8, dilated_bbox_cache[tid], det.bbox)
                        c_val += self.cfg.w_iou * (1.0 - iou)
                    cost[row, det_idx] = c_val

            if SCIPY_AVAILABLE:
                finite_any = np.isfinite(cost).any()
                if finite_any:
                    try:
                        cost_f = cost.copy()
                        cost_f[~np.isfinite(cost_f)] = 1e9
                        row_ind, col_ind = linear_sum_assignment(cost_f)
                        for r, c in zip(row_ind, col_ind):
                            if np.isfinite(cost[r, c]):
                                matches.append((r, c))
                    except ValueError:
                        finite_any = False
                if not finite_any:
                    pairs = [
                        (i, j, cost[i, j]) for i in range(T) for j in range(D) if np.isfinite(cost[i, j])
                    ]
                    pairs.sort(key=lambda x: x[2])
                    used_t: Set[int] = set()
                    used_d: Set[int] = set()
                    for i, j, _ in pairs:
                        if i in used_t or j in used_d:
                            continue
                        matches.append((i, j))
                        used_t.add(i)
                        used_d.add(j)
            else:
                pairs = [(i, j, cost[i, j]) for i in range(T) for j in range(D) if np.isfinite(cost[i, j])]
                pairs.sort(key=lambda x: x[2])
                used_t: Set[int] = set()
                used_d: Set[int] = set()
                for i, j, _ in pairs:
                    if i in used_t or j in used_d:
                        continue
                    matches.append((i, j))
                    used_t.add(i)
                    used_d.add(j)

        for row, col in matches:
            tid = track_list[row]
            det = detections[col]
            self._update_track_from_det(tid, det, frame_idx)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(col)

        for j in unmatched_dets:
            det = detections[j]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                mask_u8=det.mask_u8,
                cx=det.cx,
                cy=det.cy,
                area=det.area,
                hue_bin=det.hue_bin,
                vx=0.0,
                vy=0.0,
                last_frame=frame_idx,
                missed=0,
                bbox=det.bbox,
            )

        to_del: List[int] = []
        for tid in unmatched_tracks:
            tr = self.tracks[tid]
            tr.missed += 1
            if tr.missed > self.cfg.max_missed:
                to_del.append(tid)
            else:
                self.tracks[tid] = tr
        for tid in to_del:
            del self.tracks[tid]

        masks_out: Dict[int, np.ndarray] = {}
        for tid, tr in self.tracks.items():
            if tr.missed == 0 and tr.last_frame == frame_idx:
                masks_out[tid] = tr.mask_u8.astype(np.uint8)
            elif tr.missed > 0 and tr.missed <= self.cfg.max_missed:
                dt_shift = min(tr.missed, 3)
                dx = int(round(tr.vx * dt_shift))
                dy = int(round(tr.vy * dt_shift))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(
                    tr.mask_u8.astype(np.uint8),
                    M,
                    (tr.mask_u8.shape[1], tr.mask_u8.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
                masks_out[tid] = shifted
        return masks_out


# --------------------------- Feature Extraction -------------------------- #
class FeatureExtractor:
    def __init__(self):
        self.tracks: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
        self._xs = None
        self._ys = None

    def update(self, frame_idx: int, v_u8: np.ndarray, masks_t: Dict[int, np.ndarray]) -> None:
        if self._xs is None or self._ys is None:
            self._ys, self._xs = np.indices(v_u8.shape)
        xs = self._xs
        ys = self._ys
        for obj_id, mask_u8 in masks_t.items():
            mask = mask_u8.astype(bool)
            if not np.any(mask):
                continue
            weights = v_u8[mask].astype(np.float32)
            total = float(weights.sum())
            if total <= 0:
                continue
            cx = float((xs[mask] * weights).sum() / total)
            cy = float((ys[mask] * weights).sum() / total)
            area = float(mask.sum())
            rg2 = float((((xs[mask] - cx) ** 2 + (ys[mask] - cy) ** 2) * weights).sum() / total)
            rg = math.sqrt(rg2)
            self.tracks.setdefault(obj_id, []).append((frame_idx, cx, cy, total, area, rg))


def compute_tracks_from_segments(
    video_segments: Dict[int, Dict[int, np.ndarray]],
    v_per_frame: Dict[int, np.ndarray],
) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    feat = FeatureExtractor()
    for t, masks in video_segments.items():
        v = v_per_frame[t]
        feat.update(t, v, masks)
    return feat.tracks


# --------------------------- Grouping ------------------------------------ #
def _smooth_positions(track: List[Tuple[int, float, float, float, float, float]], window: int):
    if not track:
        return {}
    alpha = 2.0 / (window + 1)
    smoothed = {}
    cx_ema = track[0][1]
    cy_ema = track[0][2]
    for t, cx, cy, _, _, _ in track:
        cx_ema = alpha * cx + (1 - alpha) * cx_ema
        cy_ema = alpha * cy + (1 - alpha) * cy_ema
        smoothed[int(t)] = (float(cx_ema), float(cy_ema))
    return smoothed


def group_parts_into_organisms(
    tracks: Dict[int, List[Tuple[int, float, float, float, float, float]]],
    window: int,
    tau_sigma: float,
    tau_dist: float,
) -> Tuple[List[Set[int]], Dict[int, int]]:
    part_ids = sorted(tracks.keys())
    if not part_ids:
        return [], {}
    smoothed = {pid: _smooth_positions(trk, window) for pid, trk in tracks.items()}

    def overlap_times(a: Dict[int, Tuple[float, float]], b: Dict[int, Tuple[float, float]]):
        times = sorted(set(a.keys()) & set(b.keys()))
        if not times:
            return []
        return times[-window:] if len(times) > window else times

    edges = []
    for i, pid_i in enumerate(part_ids):
        for pid_j in part_ids[i + 1 :]:
            times = overlap_times(smoothed[pid_i], smoothed[pid_j])
            if len(times) < max(3, window // 5):
                continue
            vecs = []
            dists = []
            for t in times:
                cx_i, cy_i = smoothed[pid_i][t]
                cx_j, cy_j = smoothed[pid_j][t]
                dx = cx_j - cx_i
                dy = cy_j - cy_i
                vecs.append((dx, dy))
                dists.append(math.hypot(dx, dy))
            vecs_arr = np.asarray(vecs, dtype=np.float32)
            median_dx = float(np.median(vecs_arr[:, 0]))
            median_dy = float(np.median(vecs_arr[:, 1]))
            dev = np.sqrt(((vecs_arr[:, 0] - median_dx) ** 2 + (vecs_arr[:, 1] - median_dy) ** 2))
            sigma = float(np.std(dev))
            mean_dist = float(np.mean(dists))
            if sigma < tau_sigma and mean_dist < tau_dist:
                edges.append((pid_i, pid_j))

    parent = {pid: pid for pid in part_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    root_to_org: Dict[int, int] = {}
    groups: List[Set[int]] = []
    part_to_org: Dict[int, int] = {}
    next_org = 1
    for pid in part_ids:
        root = find(pid)
        if root not in root_to_org:
            root_to_org[root] = next_org
            next_org += 1
            groups.append(set())
        org_id = root_to_org[root]
        part_to_org[pid] = org_id
        groups[org_id - 1].add(pid)
    return groups, part_to_org


# --------------------------- Organism masks ------------------------------ #
def build_organism_segments(
    part_segments: Dict[int, Dict[int, np.ndarray]],
    part_to_org: Dict[int, int],
) -> Dict[int, Dict[int, np.ndarray]]:
    organism_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for t, masks in part_segments.items():
        org_map: Dict[int, np.ndarray] = {}
        for part_id, m in masks.items():
            org_id = part_to_org.get(part_id)
            if org_id is None:
                continue
            if org_id not in org_map:
                org_map[org_id] = m.astype(np.uint8)
            else:
                org_map[org_id] = np.clip(org_map[org_id] + m.astype(np.uint8), 0, 1)
        organism_segments[t] = org_map
    return organism_segments


# --------------------------- Eating policy ------------------------------- #
class EaterPolicy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * cfg.close_r + 1, 2 * cfg.close_r + 1)
        )

    def apply(
        self,
        organism_segments: Dict[int, Dict[int, np.ndarray]],
        org_groups: List[Set[int]],
    ) -> Tuple[List[Set[int]], Dict[int, int]]:
        if not organism_segments:
            return org_groups, {}
        counts: Dict[Tuple[int, int], int] = {}
        for _, org_masks in organism_segments.items():
            body_cache = {}
            filled_cache = {}
            for org_id, m in org_masks.items():
                body = m.astype(np.uint8)
                body_cache[org_id] = body
                closed = cv2.morphologyEx(body, cv2.MORPH_CLOSE, self.kernel)
                filled = self._fill_holes(closed)
                filled_cache[org_id] = filled
            for j_id, body_j in body_cache.items():
                area_j = float(body_j.sum())
                if area_j <= 0:
                    continue
                best_g = None
                best_r = 0.0
                for g_id, filled_g in filled_cache.items():
                    if g_id == j_id:
                        continue
                    inner_g = (filled_g.astype(bool) & (~body_cache[g_id].astype(bool))).astype(np.uint8)
                    inter = float((body_j.astype(np.uint8) & inner_g).sum())
                    r = inter / area_j
                    if r > best_r:
                        best_r = r
                        best_g = g_id
                if best_r >= self.cfg.eta_eat and best_g is not None:
                    key = (j_id, best_g)
                    counts[key] = counts.get(key, 0) + 1

        eaten_pairs = {k: v for k, v in counts.items() if v >= self.cfg.eat_confirm_frames}
        if not eaten_pairs:
            part_to_org = {p: idx + 1 for idx, g in enumerate(org_groups) for p in g}
            return org_groups, part_to_org

        org_ids = [idx + 1 for idx, _ in enumerate(org_groups)]
        parent = {oid: oid for oid in org_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (j, g), _ in eaten_pairs.items():
            union(g, j)

        new_groups: Dict[int, Set[int]] = {}
        for idx, parts in enumerate(org_groups):
            org_id = idx + 1
            root = find(org_id)
            new_groups.setdefault(root, set()).update(parts)

        merged_groups = list(new_groups.values())
        part_to_org: Dict[int, int] = {}
        for new_org_id, parts in enumerate(merged_groups, start=1):
            for p in parts:
                part_to_org[p] = new_org_id
        return merged_groups, part_to_org

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        h, w = mask_u8.shape
        inv = (mask_u8 == 0).astype(np.uint8)
        ff = inv.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(ff, flood_mask, (0, 0), 2)
        holes = (ff == 1).astype(np.uint8)
        filled = (mask_u8 | holes).astype(np.uint8)
        return filled


# --------------------------- Visualization ------------------------------- #
class Visualizer:
    @staticmethod
    def _color_from_id(obj_id: int) -> Tuple[int, int, int]:
        base = np.array(
            [(obj_id * 37) % 256, (obj_id * 17) % 256, (obj_id * 29) % 256],
            dtype=np.float32,
        )
        base = (base * 0.6 + 80).clip(0, 255)
        return int(base[0]), int(base[1]), int(base[2])

    @staticmethod
    def write_overlay_video(
        frames: List[Image.Image],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        out_path: str,
        fps: float,
        alpha: float = 0.45,
        draw_contours: bool = True,
        draw_ids: bool = True,
        frame_indices: Optional[List[int]] = None,
        draw_ids_largest_k: int = 0,
    ) -> None:
        if not frames:
            return
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        first = np.array(frames[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for i, pil_frame in enumerate(frames):
            frame_rgb = np.array(pil_frame).copy()
            frame_key = frame_indices[i]
            masks_t = video_segments.get(frame_key, {})
            if masks_t:
                overlay = frame_rgb.copy()
                for obj_id, mask_u8 in masks_t.items():
                    if not np.any(mask_u8):
                        continue
                    color = np.array(Visualizer._color_from_id(obj_id), dtype=np.uint8)
                    mask = mask_u8.astype(bool)
                    overlay[mask] = (
                        (1.0 - alpha) * overlay[mask].astype(np.float32)
                        + alpha * color.astype(np.float32)
                    ).astype(np.uint8)
                frame_rgb = overlay
                if draw_contours:
                    for obj_id, mask_u8 in masks_t.items():
                        if not np.any(mask_u8):
                            continue
                        contours, _ = cv2.findContours(
                            mask_u8.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        color = Visualizer._color_from_id(obj_id)
                        cv2.drawContours(frame_rgb, contours, -1, color, 1)
                if draw_ids:
                    ids_to_draw = list(masks_t.keys())
                    if draw_ids_largest_k and draw_ids_largest_k > 0:
                        areas = [(obj_id, int(mask_u8.sum())) for obj_id, mask_u8 in masks_t.items()]
                        areas.sort(key=lambda x: x[1], reverse=True)
                        ids_to_draw = [obj for obj, _ in areas[:draw_ids_largest_k]]
                    for obj_id in ids_to_draw:
                        mask_u8 = masks_t[obj_id]
                        if not np.any(mask_u8):
                            continue
                        ys, xs = np.where(mask_u8 > 0)
                        if ys.size == 0:
                            continue
                        cx = int(xs.mean())
                        cy = int(ys.mean())
                        color = Visualizer._color_from_id(obj_id)
                        cv2.putText(
                            frame_rgb,
                            str(obj_id),
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()


# --------------------------- Pipeline ------------------------------------ #
def _parse_resize(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    if "x" in value:
        parts = value.lower().split("x")
    elif "," in value:
        parts = value.split(",")
    else:
        return None
    if len(parts) != 2:
        return None
    return int(parts[0]), int(parts[1])


def run_pipeline(
    video_path: str,
    out_dir: str,
    cfg: Config,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    draw_ids_largest_k: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    fps, _, _ = VideoLoader.get_video_info(video_path)
    frames = VideoLoader.load_mp4_frames(video_path, stride=stride, max_frames=max_frames, resize=resize)
    if not frames:
        raise RuntimeError("No frames loaded.")
    frame_indices = list(range(0, len(frames) * stride, stride))
    out_fps = fps / float(stride)

    v_per_frame: Dict[int, np.ndarray] = {}
    for idx, fr in enumerate(frames):
        v_per_frame[frame_indices[idx]] = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2HSV)[:, :, 2]

    mask_gen = MaskGenerator(cfg)
    tracker = TrackerClassic(cfg)
    feature_parts = FeatureExtractor()

    part_segments: Dict[int, Dict[int, np.ndarray]] = {}

    for t, pil_frame in enumerate(frames):
        rgb = np.array(pil_frame)
        seeds = tracker.predict_seeds(frame_indices[t]) if cfg.marker_split else []
        detections, seeded = mask_gen.generate_detections(
            rgb,
            cfg.det_v_thr_hi,
            seeds=seeds,
            use_marker_split=cfg.marker_split,
            seed_radius=cfg.seed_radius,
        )
        masks_by_track = tracker.update(frame_indices[t], detections, seeded)
        part_segments[frame_indices[t]] = masks_by_track
        feature_parts.update(frame_indices[t], v_per_frame[frame_indices[t]], masks_by_track)

    part_tracks = feature_parts.tracks
    org_groups_initial, part_to_org_initial = group_parts_into_organisms(
        part_tracks,
        window=cfg.group_window,
        tau_sigma=cfg.tau_sigma,
        tau_dist=cfg.tau_dist,
    )
    organism_segments_initial = build_organism_segments(part_segments, part_to_org_initial)

    eater = EaterPolicy(cfg)
    org_groups_final, part_to_org_final = eater.apply(organism_segments_initial, org_groups_initial)

    organism_segments = build_organism_segments(part_segments, part_to_org_final)

    organism_tracks = compute_tracks_from_segments(organism_segments, v_per_frame)

    overlay_org = os.path.join(out_dir, "overlay_organisms.mp4")
    Visualizer.write_overlay_video(
        frames,
        organism_segments,
        overlay_org,
        out_fps,
        alpha=0.45,
        draw_contours=True,
        draw_ids=True,
        frame_indices=frame_indices,
        draw_ids_largest_k=draw_ids_largest_k,
    )
    overlay_parts = os.path.join(out_dir, "overlay_parts.mp4")
    Visualizer.write_overlay_video(
        frames,
        part_segments,
        overlay_parts,
        out_fps,
        alpha=0.45,
        draw_contours=True,
        draw_ids=True,
        frame_indices=frame_indices,
        draw_ids_largest_k=draw_ids_largest_k,
    )

    tracks_parts_json = {
        "tracks": {
            str(pid): [
                {
                    "t": int(t),
                    "cx": float(cx),
                    "cy": float(cy),
                    "mass": float(mass),
                    "area": float(area),
                    "rg": float(rg),
                }
                for t, cx, cy, mass, area, rg in seq
            ]
            for pid, seq in part_tracks.items()
        }
    }
    with open(os.path.join(out_dir, "tracks_parts.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_parts_json, f, indent=2)

    tracks_org_json = {
        "tracks": {
            str(oid): [
                {
                    "t": int(t),
                    "cx": float(cx),
                    "cy": float(cy),
                    "mass": float(mass),
                    "area": float(area),
                    "rg": float(rg),
                }
                for t, cx, cy, mass, area, rg in seq
            ]
            for oid, seq in organism_tracks.items()
        },
        "initial_part_to_org": {str(k): int(v) for k, v in part_to_org_initial.items()},
        "final_part_to_org": {str(k): int(v) for k, v in part_to_org_final.items()},
        "org_groups_final": [list(map(int, g)) for g in org_groups_final],
        "debug": {
            "det_v_thr_hi": cfg.det_v_thr_hi,
            "h_bins": cfg.h_bins,
            "use_hue_bins": cfg.use_hue_bins,
            "marker_split": cfg.marker_split,
            "seed_radius": cfg.seed_radius,
            "tracker": {
                "max_dist": cfg.max_dist,
                "min_iou": cfg.min_iou,
                "max_missed": cfg.max_missed,
                "w_dist": cfg.w_dist,
                "w_iou": cfg.w_iou,
                "w_area": cfg.w_area,
                "w_col": cfg.w_col,
                "strict_color": cfg.strict_color,
                "iou_dilate_r": cfg.iou_dilate_r,
            },
            "draw_ids_largest_k": cfg.draw_ids_largest_k,
        },
    }
    with open(os.path.join(out_dir, "tracks_organisms.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_org_json, f, indent=2)

    return {
        "part_segments": part_segments,
        "organism_segments": organism_segments,
        "part_tracks": part_tracks,
        "organism_tracks": organism_tracks,
        "part_to_org_initial": part_to_org_initial,
        "part_to_org_final": part_to_org_final,
        "overlay_org": overlay_org,
        "overlay_parts": overlay_parts,
    }


# --------------------------- CLI ---------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Classic CC tracker for Flow-Lenia (no NN)")
    parser.add_argument("--video", required=True, help="Path to input mp4")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to load")
    parser.add_argument("--resize", type=str, default=None, help="Resize WxH, e.g. 640x360")
    parser.add_argument("--det_v_thr_hi", type=int, default=Config.det_v_thr_hi)
    parser.add_argument("--h_bins", type=int, default=Config.h_bins)
    parser.add_argument("--min_area", type=int, default=Config.min_area)
    parser.add_argument("--min_mass", type=float, default=Config.min_mass)
    parser.add_argument("--use_hue_bins", dest="use_hue_bins", action="store_true", default=True)
    parser.add_argument("--no_use_hue_bins", dest="use_hue_bins", action="store_false")
    parser.add_argument("--marker_split", dest="marker_split", action="store_true", default=Config.marker_split)
    parser.add_argument("--no_marker_split", dest="marker_split", action="store_false")
    parser.add_argument("--seed_radius", type=int, default=Config.seed_radius)
    parser.add_argument("--max_dist", type=float, default=Config.max_dist)
    parser.add_argument("--min_iou", type=float, default=Config.min_iou)  # kept for compatibility, not used as gate
    parser.add_argument("--max_missed", type=int, default=Config.max_missed)
    parser.add_argument("--w_dist", type=float, default=Config.w_dist)
    parser.add_argument("--w_iou", type=float, default=Config.w_iou)
    parser.add_argument("--w_area", type=float, default=Config.w_area)
    parser.add_argument("--w_col", type=float, default=Config.w_col)
    parser.add_argument("--strict_color", action="store_true", default=Config.strict_color)
    parser.add_argument("--iou_zero_dist_gate", type=float, default=None)
    parser.add_argument("--iou_dilate_r", type=int, default=Config.iou_dilate_r)
    parser.add_argument("--hue_dist_max", type=float, default=Config.hue_dist_max)
    parser.add_argument("--group_window", type=int, default=Config.group_window)
    parser.add_argument("--tau_sigma", type=float, default=Config.tau_sigma)
    parser.add_argument("--tau_dist", type=float, default=Config.tau_dist)
    parser.add_argument("--eta_eat", type=float, default=Config.eta_eat)
    parser.add_argument("--close_r", type=int, default=Config.close_r)
    parser.add_argument("--eat_confirm_frames", type=int, default=Config.eat_confirm_frames)
    parser.add_argument("--save_parts_debug", dest="save_parts_debug", action="store_true", default=True)
    parser.add_argument("--no_save_parts_debug", dest="save_parts_debug", action="store_false")
    parser.add_argument("--draw_ids_largest_k", type=int, default=Config.draw_ids_largest_k)
    args = parser.parse_args()

    cfg = Config(
        det_v_thr_hi=args.det_v_thr_hi,
        h_bins=args.h_bins,
        min_area=args.min_area,
        min_mass=args.min_mass,
        use_hue_bins=args.use_hue_bins,
        marker_split=args.marker_split,
        seed_radius=args.seed_radius,
        max_dist=args.max_dist,
        min_iou=args.min_iou,
        max_missed=args.max_missed,
        w_dist=args.w_dist,
        w_iou=args.w_iou,
        w_area=args.w_area,
        w_col=args.w_col,
        strict_color=args.strict_color,
        iou_zero_dist_gate=args.iou_zero_dist_gate if args.iou_zero_dist_gate is not None else 0.7 * args.max_dist,
        iou_dilate_r=args.iou_dilate_r,
        hue_dist_max=args.hue_dist_max,
        group_window=args.group_window,
        tau_sigma=args.tau_sigma,
        tau_dist=args.tau_dist,
        eta_eat=args.eta_eat,
        close_r=args.close_r,
        eat_confirm_frames=args.eat_confirm_frames,
        save_parts_debug=args.save_parts_debug,
        draw_ids_largest_k=args.draw_ids_largest_k,
    )
    resize = _parse_resize(args.resize)
    run_pipeline(
        video_path=args.video,
        out_dir=args.out_dir,
        cfg=cfg,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        resize=resize,
        draw_ids_largest_k=args.draw_ids_largest_k,
    )


if __name__ == "__main__":
    main()
