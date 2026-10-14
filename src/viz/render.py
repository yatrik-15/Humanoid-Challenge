"""scene_3d.png / scene_3d.gif: both reconstructed rooms rendered from one virtual camera."""

import json
from pathlib import Path

import numpy as np

from src.utils.geometry import camera_center, in_box, project_points
from src.viz.images import draw_box
from src.viz.report import action_color, change_boxes, shown_in


def load_meshes(out_dir: Path, weight_threshold: float) -> dict[str, dict]:
    """Zero-level surface of each recording's TSDF grid: vertices, triangles, vertex colours
    (0-1) and vertex normals. A closed, coloured surface reads as a room, where the 2 cm
    surface points read as a scatter of dots."""
    import open3d as o3d

    meshes = {}
    for e in ("A", "B"):
        grid = o3d.t.geometry.VoxelBlockGrid.load(str(out_dir / "fusion" / f"grid_{e}.npz"))
        m = grid.extract_triangle_mesh(weight_threshold).to_legacy()
        m.compute_vertex_normals()
        meshes[e] = {
            "vertices": np.asarray(m.vertices, np.float64),
            "triangles": np.asarray(m.triangles, np.int64),
            "colors": np.clip(np.asarray(m.vertex_colors, np.float64), 0, 1),
            "normals": np.asarray(m.vertex_normals, np.float64),
        }
    return meshes


def simplify_mesh(mesh: dict, max_triangles: int) -> dict:
    """Quadric decimation to at most `max_triangles` (vertex colours are kept)."""
    import open3d as o3d

    if len(mesh["triangles"]) <= max_triangles:
        return mesh
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh["vertices"]),
        o3d.utility.Vector3iVector(mesh["triangles"].astype(np.int32)),
    )
    m.vertex_colors = o3d.utility.Vector3dVector(mesh["colors"])
    m = m.simplify_quadric_decimation(max_triangles)
    m.compute_vertex_normals()
    return {
        "vertices": np.asarray(m.vertices),
        "triangles": np.asarray(m.triangles, np.int64),
        "colors": np.clip(np.asarray(m.vertex_colors), 0, 1),
        "normals": np.asarray(m.vertex_normals),
    }


class MeshView:
    """Ray-cast renderer for one mesh: interpolated vertex colours, soft shading, and the
    3-D hit point of every pixel (used to tint the surface inside change boxes)."""

    BACKGROUND = np.array([32, 32, 32], np.float64)

    def __init__(self, mesh: dict):
        import open3d as o3d
        import open3d.core as o3c

        self.mesh = mesh
        self.scene = o3d.t.geometry.RaycastingScene()
        if len(mesh["triangles"]):
            self.scene.add_triangles(
                o3c.Tensor(mesh["vertices"].astype(np.float32)),
                o3c.Tensor(mesh["triangles"].astype(np.uint32)),
            )

    def render(
        self, K: np.ndarray, T_cw: np.ndarray, W: int, H: int, tints: list[tuple[dict, list[int]]] = ()
    ) -> np.ndarray:
        """(H, W, 3) uint8 RGB. `tints`: (box, rgb) pairs; surface inside a box is blended
        towards that colour, so the changed object itself lights up."""
        import open3d.core as o3c

        from src.utils.geometry import pixel_rays

        img = np.tile(self.BACKGROUND, (H * W, 1))
        if not len(self.mesh["triangles"]):
            return img.reshape(H, W, 3).astype(np.uint8)
        o, d = pixel_rays(K, T_cw, H, W)
        ans = self.scene.cast_rays(o3c.Tensor(np.concatenate([o, d], axis=1).astype(np.float32)))
        t = ans["t_hit"].numpy()
        hit = np.isfinite(t)
        tri = self.mesh["triangles"][ans["primitive_ids"].numpy()[hit]]
        uv = ans["primitive_uvs"].numpy()[hit]
        w = np.c_[1 - uv.sum(axis=1), uv]  # barycentric weights of the three vertices
        col = np.einsum("nk,nkc->nc", w, self.mesh["colors"][tri])
        n = np.einsum("nk,nkc->nc", w, self.mesh["normals"][tri])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        view = d[hit] / np.linalg.norm(d[hit], axis=1, keepdims=True)
        # headlight plus a little light from above: shapes read without hiding the colours
        shade = 0.55 + 0.35 * np.abs(np.sum(n * view, axis=1)) + 0.10 * np.clip(n[:, 2], 0, 1)
        rgb = np.clip(col * shade[:, None], 0, 1) * 255
        X = o[hit] + t[hit, None] * d[hit]
        for box, color in tints:
            m = in_box(box, X, margin=0.02)
            rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(color, np.float64)
        img[hit] = rgb
        return img.reshape(H, W, 3).astype(np.uint8)


def overview_pose(out_dir: Path, report: dict, angle_deg: float):
    """(eye, target) of an elevated virtual camera behind the recorded paths, looking at the
    changes (or the scene centre), rotated by `angle_deg` about the vertical axis."""
    poses = json.load(open(out_dir / "align" / "poses.json"))
    C = np.array([camera_center(np.array(p["T_cw"])) for p in poses.values()])
    fwd = np.array([np.array(p["T_cw"])[2, :3] for p in poses.values()]).mean(axis=0)
    boxes = [b for b, *_ in change_boxes(report)]
    if boxes:
        target = np.mean([b["center"] for b in boxes], axis=0)
    else:  # scene centre: about 1.5 m in front of the cameras
        target = C.mean(axis=0) + fwd / np.linalg.norm(fwd) * 1.5
    back = C.mean(axis=0) - target
    back[2] = 0
    dist = max(np.linalg.norm(back), 1.0)
    back /= dist
    a = np.deg2rad(angle_deg)
    back = np.array([back[0] * np.cos(a) - back[1] * np.sin(a), back[0] * np.sin(a) + back[1] * np.cos(a), 0])
    eye = target + back * dist * 1.2 + np.array([0, 0, 1.0]) * dist * 0.8
    return eye, target


def overview_camera(out_dir: Path, report: dict, angle_deg: float, W: int, H: int, pts: np.ndarray):
    """(K, T_cw) for `overview_pose`; focal length and principal point are fitted so that
    98 % of `pts` fill the image."""
    from src.utils.geometry import look_at_T_cw

    eye, target = overview_pose(out_dir, report, angle_deg)
    T = look_at_T_cw(eye, target, np.array([0.0, 0.0, 1.0]))
    Xc = pts @ T[:3, :3].T + T[:3, 3]
    Xc = Xc[Xc[:, 2] > 0.05]
    n = Xc[:, :2] / Xc[:, 2:]
    lo, hi = np.percentile(n, 1, axis=0), np.percentile(n, 99, axis=0)
    f = 0.92 * min(W / (hi[0] - lo[0]), H / (hi[1] - lo[1]))
    K = np.array(
        [[f, 0, W / 2 - f * (lo[0] + hi[0]) / 2], [0, f, H / 2 - f * (lo[1] + hi[1]) / 2], [0, 0, 1]]
    )
    return K, T


def render_overview(
    out_dir: Path, report: dict, cfg: dict, views: dict[str, MeshView], angle_deg: float, W: int, H: int
) -> np.ndarray:
    """BEFORE | AFTER from one virtual camera: shaded meshes rendered at twice the size and
    averaged down, changed surfaces tinted, then camera paths, boxes and labels."""
    import cv2

    ss = 2
    pts = views["B"].mesh["vertices"]
    K, T = overview_camera(out_dir, report, angle_deg, W, H, pts)
    Ks = K.copy()
    Ks[:2] *= ss  # exact for COLMAP corner-origin pixel coordinates
    poses = json.load(open(out_dir / "align" / "poses.json"))
    boxes = list(change_boxes(report))
    tiles = []
    for e in ("A", "B"):
        tints = [(b, action_color(cfg, a)) for b, a, _, w in boxes if shown_in(a, w, e)]
        big = views[e].render(Ks, T, W * ss, H * ss, tints)
        img = cv2.cvtColor(cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        C = np.array(
            [camera_center(np.array(p["T_cw"])) for n, p in sorted(poses.items()) if p["epoch"] == e]
        )
        uv, z = project_points(C, K, T)
        path = np.round(uv[z > 0]).astype(np.int32)
        col = (255, 140, 80) if e == "A" else (60, 170, 255)
        if len(path) > 1:
            cv2.polylines(img, [path], False, col, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(path[-1]), 4, col, -1, cv2.LINE_AA)
        for box, action, text, which in boxes:
            if shown_in(action, which, e):
                draw_box(img, box, K, T, action_color(cfg, action), text)
        title = "BEFORE (A)" if e == "A" else "AFTER (B)"
        cv2.putText(img, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)


def write_overview(out_dir: Path, report: dict, cfg: dict, meshes: dict[str, dict]) -> None:
    """scene_3d.png (one view) and scene_3d.gif (a slow orbit) of the fused recordings."""
    import imageio.v3 as iio

    views = {e: MeshView(m) for e, m in meshes.items()}
    W, H = cfg["overview_size"]
    iio.imwrite(str(out_dir / "scene_3d.png"), render_overview(out_dir, report, cfg, views, 0.0, W, H))
    span, n = cfg["overview_orbit_deg"], 18
    angles = np.concatenate([np.linspace(-span, span, n), np.linspace(span, -span, n)[1:-1]])
    w, h = W // 2, H // 2
    frames = [render_overview(out_dir, report, cfg, views, a, w, h) for a in angles]
    write_gif(frames, out_dir / "scene_3d.gif", duration_ms=110)


def write_gif(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    """Animated GIF with ONE palette for all frames and no dithering: per-frame palettes
    make colours flicker, and dithering adds speckle noise to smooth surfaces."""
    from PIL import Image

    sample = np.vstack([f[::4, ::4] for f in frames])
    palette = Image.fromarray(sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
