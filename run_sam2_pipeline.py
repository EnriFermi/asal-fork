#!/usr/bin/env python3
import argparse
import json
import math
import os
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


@dataclass
class Config:
    v_thr: int = 25
    h_bins: int = 12
    min_area: int = 20
    min_mass: float = 500.0
    birth_v_thr: int = 35
    birth_confirm_frames: int = 2
    births_per_frame: int = 20
    refine_every: int = 5
    refine_radius: int = 6
    refine_v_thr: int = 40
    refine_topk: int = 3
    iou_dedup_thr: float = 0.3
    model_id: str = "facebook/sam2.1-hiera-large"


@dataclass
class Seed:
    x: int
    y: int
    score: float
    color_bin: int


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
        grabbed = True
        while grabbed:
            grabbed, frame_bgr = cap.read()
            if not grabbed:
                break
            if idx % stride != 0:
                idx += 1
                continue
            if resize is not None:
                frame_bgr = cv2.resize(frame_bgr, resize, interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            idx += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
        cap.release()
        return frames


class SeedGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(
        self,
        rgb_u8: np.ndarray,
        uncovered_mask: Optional[np.ndarray],
        v_thr_override: Optional[int] = None,
    ) -> List[Seed]:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        v = hsv[:, :, 2]
        thr = self.cfg.v_thr if v_thr_override is None else v_thr_override
        mask_visible = v > thr
        if uncovered_mask is not None:
            mask_visible &= uncovered_mask
        if not np.any(mask_visible):
            return []
        h_bin = (h.astype(np.int32) * self.cfg.h_bins) // 180
        seeds: List[Seed] = []
        for b in range(self.cfg.h_bins):
            mask_b = mask_visible & (h_bin == b)
            if not np.any(mask_b):
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_b.astype(np.uint8), connectivity=8
            )
            for comp_id in range(1, num_labels):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < self.cfg.min_area:
                    continue
                comp_mask = labels == comp_id
                mass = float(v[comp_mask].sum())
                if mass < self.cfg.min_mass:
                    continue
                comp_v = v.copy()
                comp_v[~comp_mask] = 0
                flat_idx = int(comp_v.argmax())
                if comp_v.flat[flat_idx] <= 0:
                    continue
                y, x = np.unravel_index(flat_idx, v.shape)
                seeds.append(Seed(x=int(x), y=int(y), score=float(v[y, x]), color_bin=b))
        return seeds


class TrackerSAM2HF:
    def __init__(self, model_id: str, device: torch.device):
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.session = None
        self.obj_ids: List[int] = []
        self.next_obj_id = 1
        self._add_inputs_sig = inspect.signature(self.processor.add_inputs_to_inference_session)
        self._postprocess_sig = inspect.signature(self.processor.post_process_masks)

    def init_session(self, frames: List[Image.Image]) -> None:
        self.session = self.processor.init_video_session(frames)

    def add_seeds(self, frame_idx: int, seeds: List[Seed], mode: str = "new_obj") -> List[int]:
        new_ids: List[int] = []
        if mode != "new_obj":
            return new_ids
        for seed in seeds:
            obj_id = self.next_obj_id
            self.next_obj_id += 1
            self.obj_ids.append(obj_id)
            self._add_points(frame_idx, obj_id, [(seed.x, seed.y)], [1])
            new_ids.append(obj_id)
        return new_ids

    def add_points(
        self,
        frame_idx: int,
        obj_id: int,
        points_xy: List[Tuple[int, int]],
        labels: List[int],
    ) -> None:
        self._add_points(frame_idx, obj_id, points_xy, labels)

    def _add_points(
        self,
        frame_idx: int,
        obj_id: int,
        points_xy: List[Tuple[int, int]],
        labels: List[int],
    ) -> None:
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[None, :, :]
        lbs = np.asarray(labels, dtype=np.int64)
        if lbs.ndim == 1:
            lbs = lbs[None, :]

        kwargs = {}
        if "frame_idx" in self._add_inputs_sig.parameters:
            kwargs["frame_idx"] = frame_idx
        elif "frame_index" in self._add_inputs_sig.parameters:
            kwargs["frame_index"] = frame_idx

        if "obj_id" in self._add_inputs_sig.parameters:
            kwargs["obj_id"] = obj_id

        if "input_points" in self._add_inputs_sig.parameters:
            kwargs["input_points"] = pts
        elif "point_coords" in self._add_inputs_sig.parameters:
            kwargs["point_coords"] = pts

        if "input_labels" in self._add_inputs_sig.parameters:
            kwargs["input_labels"] = lbs
        elif "point_labels" in self._add_inputs_sig.parameters:
            kwargs["point_labels"] = lbs

        self.processor.add_inputs_to_inference_session(self.session, **kwargs)

    def propagate_iter(self):
        if self.session is None:
            raise RuntimeError("Session not initialized.")
        return self.model.propagate_in_video_iterator(self.session)

    def post_process_masks(self, pred_masks, original_size: Tuple[int, int]) -> np.ndarray:
        kwargs = {"binarize": True}
        if "original_sizes" in self._postprocess_sig.parameters:
            kwargs["original_sizes"] = [original_size]
        elif "target_sizes" in self._postprocess_sig.parameters:
            kwargs["target_sizes"] = [original_size]
        masks = self.processor.post_process_masks(pred_masks, **kwargs)
        if isinstance(masks, list):
            masks = masks[0]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[None, :, :]
        return masks


class PostProcess:
    @staticmethod
    def resolve_overlaps(
        masks_by_obj: Dict[int, np.ndarray],
        v_u8: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        if not masks_by_obj:
            return np.zeros((0, 0), dtype=np.int32), {}
        any_mask = next(iter(masks_by_obj.values()))
        h, w = any_mask.shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        confidences: Dict[int, float] = {}
        for obj_id, mask in masks_by_obj.items():
            mask_bool = mask.astype(bool)
            if v_u8 is None:
                confidences[obj_id] = float(mask_bool.sum())
            else:
                if np.any(mask_bool):
                    confidences[obj_id] = float(v_u8[mask_bool].mean())
                else:
                    confidences[obj_id] = 0.0
        sorted_ids = sorted(confidences.keys(), key=lambda k: confidences[k], reverse=True)
        for obj_id in sorted_ids:
            mask_bool = masks_by_obj[obj_id].astype(bool)
            label_map[mask_bool & (label_map == 0)] = obj_id
        resolved = {obj_id: (label_map == obj_id).astype(np.uint8) for obj_id in masks_by_obj}
        return label_map, resolved


class BirthPolicy:
    def __init__(self, cfg: Config, seed_gen: SeedGenerator):
        self.cfg = cfg
        self.seed_gen = seed_gen
        self.cache: Dict[Tuple[int, int], Dict[str, object]] = {}
        self.cell = 8

    def propose(
        self,
        frame_idx: int,
        rgb_u8: np.ndarray,
        masks_t: Dict[int, np.ndarray],
    ) -> List[Seed]:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        covered = np.zeros(v.shape, dtype=bool)
        for mask in masks_t.values():
            covered |= mask.astype(bool)
        uncovered_visible = (v > self.cfg.birth_v_thr) & (~covered)
        seeds = self.seed_gen.generate(
            rgb_u8, uncovered_visible, v_thr_override=self.cfg.birth_v_thr
        )
        if not seeds:
            self._prune(frame_idx)
            return []
        seeds = sorted(seeds, key=lambda s: s.score, reverse=True)

        confirmed: List[Seed] = []
        for seed in seeds:
            if covered[seed.y, seed.x]:
                continue
            key = (seed.x // self.cell, seed.y // self.cell)
            entry = self.cache.get(key)
            if entry is not None and entry["last_frame"] == frame_idx - 1:
                entry["count"] += 1
            else:
                entry = {"count": 1}
            entry["last_frame"] = frame_idx
            entry["seed"] = seed
            self.cache[key] = entry
            if entry["count"] >= self.cfg.birth_confirm_frames:
                confirmed.append(seed)
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


class Refiner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * cfg.refine_radius + 1, 2 * cfg.refine_radius + 1)
        )

    def propose(
        self,
        frame_idx: int,
        rgb_u8: np.ndarray,
        masks_resolved: Dict[int, np.ndarray],
        masks_raw: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, Tuple[List[Tuple[int, int]], List[int]]]:
        if frame_idx % self.cfg.refine_every != 0:
            return {}
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        out: Dict[int, Tuple[List[Tuple[int, int]], List[int]]] = {}

        union_raw = None
        if masks_raw:
            union_raw = np.zeros(v.shape, dtype=bool)
            for mask in masks_raw.values():
                union_raw |= mask.astype(bool)

        for obj_id, mask_u8 in masks_resolved.items():
            mask_bool = mask_u8.astype(bool)
            if not np.any(mask_bool):
                continue
            dilated = cv2.dilate(mask_u8, self.kernel).astype(bool)
            candidate = dilated & (v > self.cfg.refine_v_thr) & (~mask_bool)
            pos_points = self._top_component_seeds(v, candidate, self.cfg.refine_topk)
            points: List[Tuple[int, int]] = []
            labels: List[int] = []
            for x, y in pos_points:
                points.append((x, y))
                labels.append(1)

            if masks_raw and union_raw is not None and obj_id in masks_raw:
                raw_mask = masks_raw[obj_id].astype(bool)
                overlap = raw_mask & (union_raw & (~raw_mask))
                if np.any(overlap):
                    neg_y, neg_x = self._max_v_point(v, overlap)
                    if neg_x is not None:
                        points.append((neg_x, neg_y))
                        labels.append(0)

            if points:
                out[obj_id] = (points, labels)
        return out

    @staticmethod
    def _max_v_point(v: np.ndarray, mask: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        if not np.any(mask):
            return None, None
        temp = v.copy()
        temp[~mask] = 0
        flat_idx = int(temp.argmax())
        if temp.flat[flat_idx] <= 0:
            return None, None
        y, x = np.unravel_index(flat_idx, v.shape)
        return int(y), int(x)

    def _top_component_seeds(
        self, v: np.ndarray, mask: np.ndarray, topk: int
    ) -> List[Tuple[int, int]]:
        if not np.any(mask):
            return []
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        comps: List[Tuple[float, int, np.ndarray]] = []
        for comp_id in range(1, num_labels):
            comp_mask = labels == comp_id
            mass = float(v[comp_mask].sum())
            comps.append((mass, comp_id, comp_mask))
        comps.sort(key=lambda x: x[0], reverse=True)
        seeds: List[Tuple[int, int]] = []
        for mass, comp_id, comp_mask in comps[:topk]:
            if mass <= 0:
                continue
            temp = v.copy()
            temp[~comp_mask] = 0
            flat_idx = int(temp.argmax())
            if temp.flat[flat_idx] <= 0:
                continue
            y, x = np.unravel_index(flat_idx, v.shape)
            seeds.append((int(x), int(y)))
        return seeds


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


def group_objects(tracks: Dict[int, List[Tuple[int, float, float, float, float, float]]], window: int = 50) -> Dict[int, int]:
    obj_ids = sorted(tracks.keys())
    if not obj_ids:
        return {}
    pos = {}
    for obj_id, seq in tracks.items():
        pos[obj_id] = {int(t): (float(cx), float(cy)) for t, cx, cy, _, _, _ in seq}

    parent = {obj_id: obj_id for obj_id in obj_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    var_thr = 4.0
    for i, obj_i in enumerate(obj_ids):
        for obj_j in obj_ids[i + 1 :]:
            overlap = sorted(set(pos[obj_i].keys()) & set(pos[obj_j].keys()))
            if len(overlap) < window:
                continue
            dists = np.array(
                [
                    math.hypot(
                        pos[obj_i][t][0] - pos[obj_j][t][0],
                        pos[obj_i][t][1] - pos[obj_j][t][1],
                    )
                    for t in overlap
                ],
                dtype=np.float32,
            )
            if dists.size >= window:
                var = float(np.var(dists[-window:]))
                if var <= var_thr:
                    union(obj_i, obj_j)

    root_to_id: Dict[int, int] = {}
    mapping: Dict[int, int] = {}
    next_id = 1
    for obj_id in obj_ids:
        root = find(obj_id)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        mapping[obj_id] = root_to_id[root]
    return mapping


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = TrackerSAM2HF(cfg.model_id, device)
    tracker.init_session(frames)

    seed_gen = SeedGenerator(cfg)
    birth_policy = BirthPolicy(cfg, seed_gen)
    refiner = Refiner(cfg)
    feature = FeatureExtractor()

    rgb0 = np.array(frames[0])
    seeds0 = seed_gen.generate(rgb0, None)
    if not seeds0:
        raise RuntimeError("No initial seeds found on frame 0.")
    tracker.add_seeds(0, seeds0, mode="new_obj")

    video_segments: Dict[int, Dict[int, np.ndarray]] = {}

    with torch.no_grad():
        for t, out in enumerate(tracker.propagate_iter()):
            if t >= len(frames):
                break
            rgb = np.array(frames[t])
            h, w = rgb.shape[:2]
            v = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]

            pred_masks = out.pred_masks if hasattr(out, "pred_masks") else out["pred_masks"]
            obj_ids = out.obj_ids if hasattr(out, "obj_ids") else out.get("obj_ids", tracker.obj_ids)
            if obj_ids is None:
                obj_ids = tracker.obj_ids

            masks = tracker.post_process_masks(pred_masks, (h, w))
            masks_by_obj: Dict[int, np.ndarray] = {}
            for idx, obj_id in enumerate(obj_ids):
                if idx >= masks.shape[0]:
                    break
                masks_by_obj[int(obj_id)] = (masks[idx] > 0).astype(np.uint8)

            _, masks_resolved = PostProcess.resolve_overlaps(masks_by_obj, v_u8=v)
            frame_id = frame_indices[t]
            video_segments[frame_id] = masks_resolved
            feature.update(frame_id, v, masks_resolved)

            new_seeds = birth_policy.propose(t, rgb, masks_resolved)
            if new_seeds:
                tracker.add_seeds(t, new_seeds, mode="new_obj")

            refine_points = refiner.propose(t, rgb, masks_resolved, masks_raw=masks_by_obj)
            for obj_id, (pts, labels) in refine_points.items():
                tracker.add_points(t, obj_id, pts, labels)

    overlay_path = os.path.join(out_dir, "overlay.mp4")
    Visualizer.write_overlay_video(
        frames,
        video_segments,
        overlay_path,
        out_fps,
        alpha=0.45,
        draw_contours=True,
        draw_ids=True,
        frame_indices=frame_indices,
    )

    tracks = feature.tracks
    grouping = group_objects(tracks, window=50)
    json_out = {
        "tracks": {
            str(obj_id): [
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
            for obj_id, seq in tracks.items()
        },
        "grouping": {str(obj_id): int(org_id) for obj_id, org_id in grouping.items()},
    }
    json_path = os.path.join(out_dir, "tracks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    return video_segments, tracks, grouping


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


def main():
    parser = argparse.ArgumentParser(description="SAM2 video segmentation + tracking pipeline")
    parser.add_argument("--video", required=True, help="Path to input mp4")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--model", default=Config.model_id, help="HF model id")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to load")
    parser.add_argument("--resize", type=str, default=None, help="Resize as WxH, e.g. 640x360")
    args = parser.parse_args()

    cfg = Config(model_id=args.model)
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
