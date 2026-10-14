"""Semantic labels for detected changes from a vision-language model.

For each change, the best before/after views are cropped (yellow outline around the
change) and sent to the VLM in one call that must return
    {"object": str, "action": "added|removed|moved", "confidence": 0-1, "spatial_summary": str}

The VLM only names what it sees. Action, sizes, positions and displacement come from the
geometry; the description is composed in code, and a summary containing digits is dropped.
Replies are validated with pydantic; an invalid reply is retried with the validation error
appended, and after `max_retries` the change is labelled "label_unavailable".

View score = fraction of the object's surface points that are not occluded in that
recording's render x projected box area (capped, with a penalty for boxes cut by the image
border). Removed: best A view + B view of the same place; added: the reverse; moved: best
A view of the old box + best B view of the new one.

Optional label check (`label_check.enabled`): SigLIP scores the 80 COCO names plus the VLM
label on a tight crop of the view WITH the object and of the view WITHOUT it; the difference
flags labels that fit the empty view better and changes where nothing recognisable appeared or
disappeared. It adds `label_check` to labels.json and never changes the label or the geometry.

Reads  cluster/changes.json, diff/evidence.npz, align/poses.json, align/alignment.json,
       fusion/grid_{A,B}.npz, sfm/undistorted/images/
Writes semantics/crops/*.jpg, semantics/labels.json
"""

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from src.utils.geometry import project_points


class ChangeLabel(BaseModel):
    """Schema the VLM must return."""

    object: str = Field(min_length=1, max_length=80)
    action: Literal["added", "removed", "moved"]
    confidence: float = Field(ge=0.0, le=1.0)
    spatial_summary: str = Field(min_length=1, max_length=300)


def parse_label(text: str) -> ChangeLabel | None:
    """Extract the first JSON object from model text and validate it. None if invalid."""
    return parse_label_with_error(text)[0]


def parse_label_with_error(text: str) -> tuple[ChangeLabel | None, str]:
    """(label, "") if valid, else (None, short human-readable reason) for the retry prompt."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None, "the answer contained no JSON object"
    try:
        return ChangeLabel.model_validate_json(m.group(0)), ""
    except ValidationError as e:
        reasons = [
            f"field '{'.'.join(str(x) for x in err['loc']) or 'json'}': {err['msg']}" for err in e.errors()
        ]
        return None, "; ".join(reasons)[:300]


def box_corners(change: dict, key: str) -> np.ndarray:
    """8 box corners for a change ('box', 'box_before' or 'box_after')."""
    return np.array(change[key]["corners"])


def surface_in_box(cloud_pts: np.ndarray, box: dict, max_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Fused-cloud points inside a (yaw-only) box: the object's actual observed surface."""
    c, s = np.cos(box["yaw_rad"]), np.sin(box["yaw_rad"])
    R2 = np.array([[c, -s], [s, c]])
    local = (cloud_pts[:, :2] - np.array(box["center"][:2])) @ R2
    half = np.array(box["extents"]) / 2
    m = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (np.abs(cloud_pts[:, 2] - box["center"][2]) <= half[2])
    )
    pts = cloud_pts[m]
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    return pts


def projected_box(corners: np.ndarray, K: np.ndarray, T_cw: np.ndarray, W: int, H: int):
    """(clipped pixel box, fraction of the projected box inside the image), or None if behind."""
    uv, z = project_points(corners, K, T_cw)
    if np.any(z <= 0):
        return None
    (x0, y0), (x1, y1) = uv.min(axis=0), uv.max(axis=0)
    full = (x1 - x0) * (y1 - y0)
    cx0, cy0 = np.clip([x0, y0], 0, [W, H])
    cx1, cy1 = np.clip([x1, y1], 0, [W, H])
    clip = (cx1 - cx0) * (cy1 - cy0)
    if full <= 0 or clip <= 0:
        return None
    return (int(cx0), int(cy0), int(np.ceil(cx1)), int(np.ceil(cy1))), float(clip / full)


def visible_fraction(
    surf: np.ndarray, K: np.ndarray, T_cw: np.ndarray, rendered: np.ndarray, tol: float
) -> float:
    """Fraction of in-image surface points not hidden behind this epoch's rendered surface.

    For the epoch WITHOUT the object, the same test asks whether the empty region is in
    view (the rendered surface is at or behind the object's former surface).
    """
    Hs, Ws = rendered.shape
    uv, z = project_points(surf, K, T_cw)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < Ws) & (uv[:, 1] >= 0) & (uv[:, 1] < Hs)
    if ok.sum() == 0:
        return 0.0
    d = rendered[np.floor(uv[ok, 1]).astype(int), np.floor(uv[ok, 0]).astype(int)]
    return float(np.mean((d == 0) | (z[ok] <= d + tol)))


def best_view(
    corners: np.ndarray,
    surf: np.ndarray,
    epoch: str,
    poses: dict[str, dict],
    renderer,
    cfg: dict,
) -> tuple[str, tuple[int, int, int, int], float] | None:
    """Best frame of `epoch` for a change region: (frame name, pixel box, score) or None.

    score = min(clipped box area, max_area_frac * image area) * inside_fraction^2 * visible
    fraction: large but not cut off by the image border, and not occluded.
    """
    best = None
    s = cfg["occlusion_render_scale"]
    for name, p in poses.items():
        if p["epoch"] != epoch:
            continue
        K, T, W, H = np.array(p["K"]), np.array(p["T_cw"]), p["width"], p["height"]
        pb = projected_box(corners, K, T, W, H)
        if pb is None:
            continue
        box, inside = pb
        area = min((box[2] - box[0]) * (box[3] - box[1]), cfg["max_area_frac"] * W * H)
        upper = area * inside**2
        if best is not None and upper <= best[2]:
            continue  # visibility <= 1, cannot beat the current best
        vis = 1.0
        if len(surf):
            Ks = K.copy()
            Ks[:2] *= s  # exact for COLMAP corner-origin coordinates
            rd = renderer.render(Ks, T, int(round(W * s)), int(round(H * s)))
            vis = visible_fraction(surf, Ks, T, rd, cfg["occlusion_tol"])
        score = upper * vis
        if best is None or score > best[2]:
            best = (name, box, score)
    return best


def crop_with_outline(
    img: np.ndarray, box: tuple[int, int, int, int], pad_rel: float, min_size: int
) -> np.ndarray:
    """Draw the change region's outline, then crop around it with padding (clipped)."""
    import cv2

    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    img = img.copy()
    cv2.rectangle(img, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 230, 0), 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0), (y1 - y0), min_size) * (1 + 2 * pad_rel) / 2
    xa, xb = int(max(0, cx - half)), int(min(W, cx + half))
    ya, yb = int(max(0, cy - half)), int(min(H, cy + half))
    return img[ya:yb, xa:xb]


def geometry_facts(change: dict, units: str) -> str:
    """Plain-text geometry facts for the prompt and the final description (numbers from code)."""

    def box_txt(b):
        c, e = b["center"], b["extents"]
        return (
            f"centre at x={c[0]:.2f}, y={c[1]:.2f} {units}; size about "
            f"{e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f} {units} (top {c[2] + e[2] / 2:.2f} {units} above the z=0 plane)"
        )

    if change["action"] == "moved":
        d = change["displacement"]
        return (
            f"An object moved. Before: {box_txt(change['box_before'])}. After: "
            f"{box_txt(change['box_after'])}. Displacement {change['distance']:.2f} {units} "
            f"(dx={d[0]:+.2f}, dy={d[1]:+.2f}, dz={d[2]:+.2f})."
        )
    verb = (
        "added (present only in the after recording)"
        if change["action"] == "added"
        else "removed (present only in the before recording)"
    )
    return f"An object was {verb}. Region: {box_txt(change['box'])}."


SITUATION = {
    "removed": "Inside the yellow outline, image 1 (BEFORE) shows an object that is GONE in image 2 (AFTER).",
    "added": "Inside the yellow outline, image 1 (BEFORE) shows the place EMPTY; image 2 (AFTER) shows a NEW object there.",
    "moved": "Image 1 (BEFORE) shows an object inside the yellow outline at its OLD place; image 2 (AFTER) "
    "shows the SAME object inside the yellow outline at its NEW place.",
}

# No numbers are given to the model, so it has none to copy; all numbers come from geometry.
PROMPT = """You are labeling a change between two photos of the same room.
{situation}
Name that object with a short common noun phrase (e.g. "office chair", "cardboard box", "trash bin").
Describe only what you can see. If unsure, use a generic name and a low confidence.
Answer with ONLY this JSON, no other text:
{{"object": "<noun phrase>", "action": "{action}", "confidence": <0..1>, "spatial_summary": "<one short sentence, no numbers, about where it is relative to other visible furniture>"}}"""


# Appended on a retry. Decoding is greedy, so resending the same prompt returns the same
# invalid answer; the model has to be told what was wrong.
RETRY_SUFFIX = """
Your previous answer was invalid: {error}.
Reply again with ONLY the JSON. The "action" value must be exactly "{action}"."""


class LocalQwenVL:
    """Qwen2-VL-Instruct via transformers, fp16 on CUDA with CPU fallback."""

    def __init__(self, model_id: str, device_pref: str, max_pixels: int, max_new_tokens: int):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.device = "cpu"
        if device_pref in ("auto", "cuda") and torch.cuda.is_available():
            try:
                self.model = (
                    Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16)
                    .to("cuda")
                    .eval()
                )
                self.device = "cuda"
            except torch.cuda.OutOfMemoryError:
                print("  VLM: CUDA out of memory while loading; falling back to CPU")
                torch.cuda.empty_cache()
        if self.device == "cpu":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).eval()
        print(f"  VLM {model_id} on {self.device}")

    def generate(self, images: list[np.ndarray], prompt: str) -> str:
        from PIL import Image

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        inputs = self.processor(
            images=[Image.fromarray(i) for i in images], text=[text], return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0]


# Candidate names for the label check: the 80 COCO classes, a fixed public list (not chosen
# for the recordings in this repo), plus whatever the VLM said.
COCO_CLASSES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,"
    "parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,"
    "handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,"
    "broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,"
    "mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,"
    "teddy bear,hair drier,toothbrush"
).split(",")


class SigLIPScorer:
    """SigLIP image-text scores (independent sigmoid per pair), fp16 on CUDA if available."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        print(f"  label check: {model_id} on {self.device}")

    def scores(self, images: list[np.ndarray], names: list[str]) -> np.ndarray:
        """(n_images, n_names) probabilities that the image shows each name."""
        from PIL import Image

        inputs = self.processor(
            text=[f"This is a photo of a {n}." for n in names],
            images=[Image.fromarray(i) for i in images],
            padding="max_length",  # SigLIP was trained with max-length padding
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits_per_image.float()
        return self.torch.sigmoid(logits).cpu().numpy()


def tight_crop(img: np.ndarray, box: tuple[int, int, int, int], pad_rel: float) -> np.ndarray:
    """Square crop around a pixel box, without the outline (the image encoder would see it)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    half = max(x1 - x0, y1 - y0) * (1 + 2 * pad_rel) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return img[
        int(max(0, cy - half)) : int(min(H, cy + half)), int(max(0, cx - half)) : int(min(W, cx + half))
    ]


def label_verdict(diff: np.ndarray, names: list[str], vlm_label: str | None, min_evidence: float) -> dict:
    """Judge the VLM label from per-name score differences (object view minus empty view).

    The difference cancels whatever both views share (wall, bed, floor), so it measures what
    appeared or disappeared. `names[0]` is the VLM label when there is one.
      no_visible_object   no name gains more than `min_evidence`: nothing recognisable changed
      label_contradicted  the VLM label fits the EMPTY view better than the object view
      label_supported     otherwise (`vlm_rank` says how it compares with the COCO names)
    """
    order = np.argsort(-diff)
    out = {
        "object_evidence": round(float(diff.max()), 4),
        "top": [[names[i], round(float(diff[i]), 4)] for i in order[:3]],
    }
    if vlm_label is not None:
        out["vlm_label_score"] = round(float(diff[0]), 4)
        out["vlm_rank"] = int((diff > diff[0]).sum()) + 1
    if diff.max() < min_evidence:
        out["verdict"] = "no_visible_object"
    elif vlm_label is not None and diff[0] <= 0:
        out["verdict"] = "label_contradicted"
    else:
        out["verdict"] = "label_supported"
    return out


def check_labels(results: list[dict], view_images: dict[int, dict[str, np.ndarray]], cfg: dict) -> None:
    """Add a `label_check` entry to each added/removed result (moved changes are not checked:
    no view shows the old place empty from the after recording)."""
    scorer = SigLIPScorer(cfg["model"])
    for res in results:
        views = view_images.get(res["id"], {})
        if res["action"] == "moved" or len(views) != 2:
            res["label_check"] = {"verdict": "not_checked"}
            continue
        present, absent = ("B", "A") if res["action"] == "added" else ("A", "B")
        vlm = None if res["object"] == "label_unavailable" else res["object"].strip().lower()
        names = ([vlm] if vlm else []) + [c for c in COCO_CLASSES if c != vlm]
        s = scorer.scores([views[present], views[absent]], names)
        res["label_check"] = {
            "model": cfg["model"],
            **label_verdict(s[0] - s[1], names, vlm, cfg["min_object_evidence"]),
        }
        print(
            f"  change {res['id']}: label check -> {res['label_check']['verdict']} {res['label_check']['top'][0]}"
        )


def make_backend(cfg: dict):
    if cfg["backend"] == "local":
        return LocalQwenVL(cfg["local_model"], cfg["device"], cfg["max_pixels"], cfg["max_new_tokens"])
    if cfg["backend"] == "none":
        return None
    raise NotImplementedError(
        f"semantics.backend={cfg['backend']!r} is not implemented (only 'local' and 'none'); "
        "see README Known limitations."
    )


def label_one(backend, images: list[np.ndarray], change: dict, units: str, max_retries: int) -> dict:
    """Ask the VLM, validate, retry; geometry fields are always overwritten by code."""
    facts = geometry_facts(change, units)
    raw_outputs = []
    label = None
    if backend is not None:
        prompt = PROMPT.format(situation=SITUATION[change["action"]], action=change["action"])
        error = ""
        for _ in range(1 + max_retries):
            p = prompt + (RETRY_SUFFIX.format(error=error, action=change["action"]) if error else "")
            raw = backend.generate(images, p)
            raw_outputs.append(raw)
            label, error = parse_label_with_error(raw)
            if label is not None:
                break
    if label is None:
        return {
            "object": "label_unavailable",
            "confidence": 0.0,
            "spatial_summary": "",
            "vlm_raw": raw_outputs,
            "vlm_action_agreed": None,
            "description": f"{facts} (no valid semantic label)",
        }
    agreed = label.action == change["action"]
    return {
        "object": label.object,
        "confidence": round(float(label.confidence), 3),
        # numbers must come from geometry: a summary containing any digit is discarded
        "spatial_summary": "" if re.search(r"\d", label.spatial_summary) else label.spatial_summary.strip(),
        "vlm_raw": raw_outputs,
        "vlm_action_agreed": agreed,
        "description": f"{label.object.capitalize()} {change['action']}: {facts}",
    }


def label_changes(out_dir: Path, cfg: dict, fusion_cfg: dict) -> list[dict]:
    """Stage entry."""
    import cv2
    import open3d as o3d

    from src.fusion import DepthRenderer

    changes = json.load(open(out_dir / "cluster" / "changes.json"))
    poses = json.load(open(out_dir / "align" / "poses.json"))
    align = json.load(open(out_dir / "align" / "alignment.json"))
    units = "m (approx.)" if align["scale_is_metric"] else "scene units"
    img_dir = out_dir / "sfm" / "undistorted" / "images"
    sdir = out_dir / "semantics"
    (sdir / "crops").mkdir(parents=True, exist_ok=True)

    renderers = {
        e: DepthRenderer(
            o3d.t.geometry.VoxelBlockGrid.load(str(out_dir / "fusion" / f"grid_{e}.npz")), fusion_cfg
        )
        for e in ("A", "B")
    }
    clouds = {e: np.load(out_dir / "fusion" / f"cloud_{e}.npz")["points"] for e in ("A", "B")}
    rng = np.random.default_rng(0)

    def view_pair(ch):
        """[(epoch, box corners, object surface points)] for the before and after views.

        The surface points always come from the epoch in which the object exists."""
        n = cfg["max_surface_points"]
        if ch["action"] == "moved":
            return [
                ("A", box_corners(ch, "box_before"), surface_in_box(clouds["A"], ch["box_before"], n, rng)),
                ("B", box_corners(ch, "box_after"), surface_in_box(clouds["B"], ch["box_after"], n, rng)),
            ]
        present = "A" if ch["action"] == "removed" else "B"
        surf = surface_in_box(clouds[present], ch["box"], n, rng)
        return [("A", box_corners(ch, "box"), surf), ("B", box_corners(ch, "box"), surf)]

    plans = []
    for ch in changes:
        plans.append(
            [
                (epoch, best_view(corners, surf, epoch, poses, renderers[epoch], cfg))
                for epoch, corners, surf in view_pair(ch)
            ]
        )

    backend = make_backend(cfg) if changes else None
    results = []
    view_images: dict[int, dict[str, np.ndarray]] = {}  # tight crops for the label check
    for ch, views in zip(changes, plans):
        imgs, used = [], []
        view_images[ch["id"]] = {}
        for epoch, bv in views:
            if bv is None:
                continue
            name, box, score = bv
            rgb = cv2.cvtColor(cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            c = crop_with_outline(rgb, box, cfg["crop_pad_rel"], cfg["crop_min_px"])
            view_images[ch["id"]][epoch] = tight_crop(rgb, box, cfg["label_check"]["crop_pad_rel"])
            path = sdir / "crops" / f"change{ch['id']:02d}_{epoch}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            imgs.append(c)
            used.append(
                {
                    "epoch": epoch,
                    "frame": name,
                    "box_px": box,
                    "score": round(score, 1),
                    "crop": str(path.relative_to(out_dir)),
                }
            )
        res = label_one(backend if len(imgs) == 2 else None, imgs, ch, units, cfg["max_retries"])
        res.update({"id": ch["id"], "action": ch["action"], "views": used})
        results.append(res)
        print(f"  change {ch['id']}: {ch['action']:8s} -> {res['object']} (conf {res['confidence']})")
    if results and cfg["label_check"]["enabled"]:
        if backend is not None:  # free the VLM's GPU memory first (6 GB card)
            import gc

            import torch

            del backend
            gc.collect()
            torch.cuda.empty_cache()
        check_labels(results, view_images, cfg["label_check"])
    with open(sdir / "labels.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
