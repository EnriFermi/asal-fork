#!/usr/bin/env python3
"""
Flow‑Lenia pipeline: external CC masks -> SAM2 video propagation -> grouping -> eating.
SAM2 is used only as a tracker; object masks come from our own CC detector.
"""
import argparse
import json
import math
import os
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


# --------------------------- Config ------------------------------------- #
@dataclass
class Config:
    # mask generation
    init_v_thr_hi: int = 45
    birth_v_thr_hi: int = 55
    h_bins: int = 24
    min_area: int = 20
    min_mass: float = 500.0
    use_hue_bins: bool = True
    model_id: str = "facebook/sam2.1-hiera-large"

    # births
    birth_confirm_frames: int = 2
    births_per_frame: int = 20

    # grouping (parts -> organisms)
    group_window: int = 50
    tau_sigma: float = 3.0
    tau_dist: float = 40.0

    # eating rule
    close_r: int = 4
    eta_eat: float = 0.6
    eat_confirm_frames: int = 3

    # debug / outputs
    save_parts_debug: bool = True  # overlay_parts always saved; keep flag for compatibility


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


# --------------------------- Mask Generator ----------------------------- #
class MaskGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate_masks(
        self,
        rgb_u8: np.ndarray,
        uncovered_mask: Optional[np.ndarray],
        v_thr_hi: int,
        use_hue_bins: Optional[bool] = None,
    ) -> List[np.ndarray]:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        v = hsv[:, :, 2]
        strong_visible = v > v_thr_hi
        if uncovered_mask is not None:
            strong_visible &= uncovered_mask
        if not np.any(strong_visible):
            return []
        use_bins = self.cfg.use_hue_bins if use_hue_bins is None else use_hue_bins
        masks: List[np.ndarray] = []
        if use_bins:
            h_bin = (h.astype(np.int32) * self.cfg.h_bins) // 180
            bins = range(self.cfg.h_bins)
            for b in bins:
                mask_b = strong_visible & (h_bin == b)
                masks.extend(self._components(mask_b, v))
        else:
            masks.extend(self._components(strong_visible, v))
        return masks

    def _components(self, mask_bin: np.ndarray, v: np.ndarray) -> List[np.ndarray]:
        if not np.any(mask_bin):
            return []
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin.astype(np.uint8), connectivity=8
        )
        out = []
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < self.cfg.min_area:
                continue
            comp_mask = labels == comp_id
            mass = float(v[comp_mask].sum())
            if mass < self.cfg.min_mass:
                continue
            out.append(comp_mask.astype(np.uint8))
        return out


# --------------------------- Birth policy (masks) ------------------------ #
class BirthPolicy:
    def __init__(self, cfg: Config, mask_gen: MaskGenerator):
        self.cfg = cfg
        self.mask_gen = mask_gen
        self.cache: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        self.cell = 8

    def propose_masks(
        self,
        frame_idx: int,
        rgb_u8: np.ndarray,
        masks_t: Dict[int, np.ndarray],
    ) -> List[np.ndarray]:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        covered = np.zeros(v.shape, dtype=bool)
        for mask in masks_t.values():
            covered |= mask.astype(bool)
        uncovered_visible = (v > self.cfg.birth_v_thr_hi) & (~covered)
        candidates = self.mask_gen.generate_masks(
            rgb_u8, uncovered_mask=uncovered_visible, v_thr_hi=self.cfg.birth_v_thr_hi
        )
        if not candidates:
            self._prune(frame_idx)
            return []

        confirmed: List[np.ndarray] = []
        for m in candidates:
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            cx = int(xs.mean())
            cy = int(ys.mean())
            area = int(m.sum())
            area_bucket = area // 50
            key = (cx // self.cell, cy // self.cell, area_bucket)
            entry = self.cache.get(key)
            if entry is not None and entry["last_frame"] == frame_idx - 1:
                entry["count"] += 1
            else:
                entry = {"count": 1}
            entry["last_frame"] = frame_idx
            entry["mask"] = m
            self.cache[key] = entry
            if entry["count"] >= self.cfg.birth_confirm_frames:
                confirmed.append(m)
        self._prune(frame_idx)
        if confirmed:
            confirmed = confirmed[: self.cfg.births_per_frame]
        return confirmed

    def _prune(self, frame_idx: int) -> None:
        to_drop = []
        for key, entry in self.cache.items():
            if frame_idx - int(entry["last_frame"]) > self.cfg.birth_confirm_frames:
                to_drop.append(key)
        for key in to_drop:
            del self.cache[key]


# --------------------------- Tracker (SAM2 HF) --------------------------- #
class TrackerSAM2HF:
    def __init__(self, model_id: str, device: torch.device, dtype: Optional[torch.dtype] = None):
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id)
        self.device = device
        self.dtype = dtype if dtype is not None else self._default_dtype(device)
        self.model.to(device, dtype=self.dtype)
        self.model.eval()
        self.session = None
        self.obj_ids: List[int] = []
        self.next_obj_id = 1
        self._add_inputs_sig = inspect.signature(self.processor.add_inputs_to_inference_session)
        self._postprocess_sig = inspect.signature(self.processor.post_process_masks)
        self._init_session_sig = inspect.signature(self.processor.init_video_session)
        self._propagate_sig = inspect.signature(self.model.propagate_in_video_iterator)
        self._forward_sig = inspect.signature(self.model.forward)
        self._has_initial_conditioning = False

    @staticmethod
    def _default_dtype(device: torch.device) -> torch.dtype:
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def init_session(self, frames: List[Image.Image]) -> None:
        kwargs = {}
        if "video" in self._init_session_sig.parameters:
            kwargs["video"] = frames
        elif "frames" in self._init_session_sig.parameters:
            kwargs["frames"] = frames
        else:
            kwargs["video"] = frames

        if "inference_device" in self._init_session_sig.parameters:
            kwargs["inference_device"] = self.device
        elif "device" in self._init_session_sig.parameters:
            kwargs["device"] = self.device

        if "dtype" in self._init_session_sig.parameters:
            kwargs["dtype"] = self.dtype

        self.session = self.processor.init_video_session(**kwargs)

    def add_masks(self, frame_idx: int, masks: List[np.ndarray]) -> List[int]:
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        if not masks:
            return []
        arr = np.stack([m.astype(np.float32) for m in masks], axis=0)  # (N,H,W)
        arr = arr[None, ...]  # (1,N,H,W) as expected by HF
        num = arr.shape[1]
        obj_ids = list(range(self.next_obj_id, self.next_obj_id + num))
        self.next_obj_id += num
        self.obj_ids.extend(obj_ids)

        kwargs = {}
        if "frame_idx" in self._add_inputs_sig.parameters:
            kwargs["frame_idx"] = frame_idx
        elif "frame_index" in self._add_inputs_sig.parameters:
            kwargs["frame_index"] = frame_idx

        if "obj_ids" in self._add_inputs_sig.parameters:
            kwargs["obj_ids"] = obj_ids
        elif "input_obj_ids" in self._add_inputs_sig.parameters:
            kwargs["input_obj_ids"] = obj_ids

        if "input_masks" in self._add_inputs_sig.parameters:
            kwargs["input_masks"] = arr
        elif "mask_inputs" in self._add_inputs_sig.parameters:
            kwargs["mask_inputs"] = arr
        elif "masks" in self._add_inputs_sig.parameters:
            kwargs["masks"] = arr

        self.processor.add_inputs_to_inference_session(self.session, **kwargs)
        return obj_ids

    def propagate_iter(self, start_frame_idx: Optional[int] = None):
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        kwargs = {}
        if start_frame_idx is not None:
            if "start_frame_idx" in self._propagate_sig.parameters:
                kwargs["start_frame_idx"] = start_frame_idx
            elif "start_frame_index" in self._propagate_sig.parameters:
                kwargs["start_frame_index"] = start_frame_idx
        return self.model.propagate_in_video_iterator(self.session, **kwargs)

    def post_process_masks(self, pred_masks, original_size: Tuple[int, int]) -> np.ndarray:
        kwargs = {"binarize": True}
        if "original_sizes" in self._postprocess_sig.parameters:
            kwargs["original_sizes"] = [original_size]
        elif "target_sizes" in self._postprocess_sig.parameters:
            kwargs["target_sizes"] = [original_size]
        masks_in = pred_masks
        if not isinstance(masks_in, (list, tuple)):
            masks_in = [masks_in]
        masks = self.processor.post_process_masks(masks_in, **kwargs)
        if isinstance(masks, list):
            masks = masks[0]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        if masks.ndim == 2:
            masks = masks[None, :, :]
        return masks

    def run_frame_inference(
        self,
        frame_idx: int,
        is_initial: Optional[bool] = None,
        is_conditioning: Optional[bool] = None,
    ):
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        if is_initial is None:
            is_initial = (frame_idx == 0) and (not self._has_initial_conditioning)
        if is_conditioning is None:
            is_conditioning = True

        kwargs = {}
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in self._forward_sig.parameters.values()
        )

        if "inference_session" in self._forward_sig.parameters or accepts_kwargs:
            kwargs["inference_session"] = self.session
        if "frame_idx" in self._forward_sig.parameters or accepts_kwargs:
            kwargs["frame_idx"] = frame_idx
        elif "frame_index" in self._forward_sig.parameters:
            kwargs["frame_index"] = frame_idx

        if "is_initial_conditioning_frame" in self._forward_sig.parameters or accepts_kwargs:
            kwargs["is_initial_conditioning_frame"] = bool(is_initial)
        elif "is_initial_frame" in self._forward_sig.parameters:
            kwargs["is_initial_frame"] = bool(is_initial)
        elif "initial_conditioning" in self._forward_sig.parameters:
            kwargs["initial_conditioning"] = bool(is_initial)

        if "is_conditioning_frame" in self._forward_sig.parameters or accepts_kwargs:
            kwargs["is_conditioning_frame"] = bool(is_conditioning)
        elif "conditioning_frame" in self._forward_sig.parameters:
            kwargs["conditioning_frame"] = bool(is_conditioning)

        out = self.model(**kwargs) if kwargs else self.model(self.session, frame_idx=frame_idx)
        if is_initial:
            self._has_initial_conditioning = True
        return out


# --------------------------- Post-process overlaps ----------------------- #
class PostProcess:
    @staticmethod
    def keep_single_component(
        mask_u8: np.ndarray,
        prev_mask_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if mask_u8.ndim != 2:
            mask_u8 = mask_u8.squeeze()
        mask = (mask_u8 > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 2:
            return mask
        choose_id = None
        if prev_mask_u8 is not None:
            prev = (prev_mask_u8 > 0).astype(np.uint8)
            best_inter = 0
            for comp_id in range(1, num_labels):
                inter = int(((labels == comp_id) & (prev > 0)).sum())
                if inter > best_inter:
                    best_inter = inter
                    choose_id = comp_id
        if choose_id is None:
            best_area = 0
            for comp_id in range(1, num_labels):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area > best_area:
                    best_area = area
                    choose_id = comp_id
        out = (labels == choose_id).astype(np.uint8)
        return out

    @staticmethod
    def resolve_overlaps_none(masks_by_obj: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # Do nothing; return as-is
        return {k: v.copy() for k, v in masks_by_obj.items()}


# --------------------------- Metrics / Tracks ---------------------------- #
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


# --------------------------- Grouping parts -> organisms ----------------- #
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
        h, w = mask.shape
        filled = mask.copy().astype(np.uint8)
        border = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(filled, border, (0, 0), 255)
        holes = (filled == 0).astype(np.uint8)
        return mask | holes


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
                    for obj_id, mask_u8 in masks_t.items():
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = TrackerSAM2HF(cfg.model_id, device)
    tracker.init_session(frames)

    mask_gen = MaskGenerator(cfg)
    birth_policy = BirthPolicy(cfg, mask_gen)
    feature_parts = FeatureExtractor()

    rgb0 = np.array(frames[0])
    masks0 = mask_gen.generate_masks(rgb0, uncovered_mask=None, v_thr_hi=cfg.init_v_thr_hi)
    if not masks0:
        raise RuntimeError("No initial masks found on frame 0.")
    tracker.add_masks(0, masks0)
    tracker.run_frame_inference(0, is_initial=True, is_conditioning=True)

    part_segments: Dict[int, Dict[int, np.ndarray]] = {}
    prev_mask_by_obj: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        for t, out in enumerate(tracker.propagate_iter(start_frame_idx=0)):
            if t >= len(frames):
                break
            rgb = np.array(frames[t])
            h, w = rgb.shape[:2]
            v = v_per_frame[frame_indices[t]]

            pred_masks = out.pred_masks if hasattr(out, "pred_masks") else out["pred_masks"]
            obj_ids = out.obj_ids if hasattr(out, "obj_ids") else out.get("obj_ids", tracker.obj_ids)
            if obj_ids is None:
                obj_ids = tracker.obj_ids

            masks = tracker.post_process_masks(pred_masks, (h, w))
            masks_by_obj: Dict[int, np.ndarray] = {}
            for idx_m, obj_id in enumerate(obj_ids):
                if idx_m >= masks.shape[0]:
                    break
                masks_by_obj[int(obj_id)] = (masks[idx_m] > 0).astype(np.uint8)

            cleaned_masks: Dict[int, np.ndarray] = {}
            for oid, m in masks_by_obj.items():
                prev = prev_mask_by_obj.get(oid)
                cleaned_masks[oid] = PostProcess.keep_single_component(m, prev_mask_u8=prev)

            prev_mask_by_obj = {oid: m.copy() for oid, m in cleaned_masks.items()}

            frame_id = frame_indices[t]
            part_segments[frame_id] = PostProcess.resolve_overlaps_none(cleaned_masks)
            feature_parts.update(frame_id, v, part_segments[frame_id])

            birth_masks = birth_policy.propose_masks(t, rgb, part_segments[frame_id])
            if birth_masks:
                tracker.add_masks(t, birth_masks)
                tracker.run_frame_inference(t, is_initial=False, is_conditioning=True)

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
            "overlap_mode_parts": "none",
            "mask_gen": {
                "h_bins": cfg.h_bins,
                "init_v_thr_hi": cfg.init_v_thr_hi,
                "birth_v_thr_hi": cfg.birth_v_thr_hi,
                "use_hue_bins": cfg.use_hue_bins,
            },
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
    parser = argparse.ArgumentParser(description="SAM2 video propagation with external CC masks")
    parser.add_argument("--video", required=True, help="Path to input mp4")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--model", default=Config.model_id, help="HF model id")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to load")
    parser.add_argument("--resize", type=str, default=None, help="Resize WxH, e.g. 640x360")
    parser.add_argument("--init_v_thr_hi", type=int, default=Config.init_v_thr_hi)
    parser.add_argument("--birth_v_thr_hi", type=int, default=Config.birth_v_thr_hi)
    parser.add_argument("--use_hue_bins", action="store_true", default=Config.use_hue_bins)
    parser.add_argument("--group_window", type=int, default=Config.group_window)
    parser.add_argument("--eta_eat", type=float, default=Config.eta_eat)
    parser.add_argument("--close_r", type=int, default=Config.close_r)
    parser.add_argument("--eat_confirm_frames", type=int, default=Config.eat_confirm_frames)
    parser.add_argument("--tau_sigma", type=float, default=Config.tau_sigma)
    parser.add_argument("--tau_dist", type=float, default=Config.tau_dist)
    parser.add_argument("--save_parts_debug", action="store_true", help="(kept for compatibility)", default=True)
    args = parser.parse_args()

    cfg = Config(
        model_id=args.model,
        init_v_thr_hi=args.init_v_thr_hi,
        birth_v_thr_hi=args.birth_v_thr_hi,
        use_hue_bins=args.use_hue_bins,
        group_window=args.group_window,
        eta_eat=args.eta_eat,
        close_r=args.close_r,
        eat_confirm_frames=args.eat_confirm_frames,
        tau_sigma=args.tau_sigma,
        tau_dist=args.tau_dist,
        save_parts_debug=args.save_parts_debug,
    )
    resize = _parse_resize(args.resize)
    run_pipeline(
        video_path=args.video,
        out_dir=args.out_dir,
        cfg=cfg,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        resize=resize,
    )


if __name__ == "__main__":
    main()
