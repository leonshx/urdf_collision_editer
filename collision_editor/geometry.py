from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .models import CollisionShape, axis_rpy, pose_matrix


def primitive_mesh(shape: CollisionShape) -> trimesh.Trimesh:
    if shape.kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=shape.radius)
    if shape.kind == "box":
        return trimesh.creation.box(extents=shape.size)
    if shape.kind == "cylinder":
        return trimesh.creation.cylinder(radius=shape.radius, height=shape.length, sections=48)
    if shape.kind == "capsule":
        if shape.length == 0:
            return trimesh.creation.icosphere(subdivisions=3, radius=shape.radius)
        # Construct a centered surface explicitly; trimesh versions differ in capsule origin.
        angles = np.r_[np.linspace(-np.pi / 2, 0, 13), np.linspace(0, np.pi / 2, 13)]
        z = (
            np.sin(angles) * shape.radius
            + np.r_[np.full(13, -shape.length / 2), np.full(13, shape.length / 2)]
        )
        profile = np.c_[np.cos(angles) * shape.radius, z]
        return trimesh.creation.revolve(profile, sections=48)
    raise ValueError(f"不是基本体：{shape.kind}")


def fit(
    mesh: trimesh.Trimesh,
    link: str,
    kind: str,
    alignment="free",
    axis="Z",
    asset_dir: Path | None = None,
) -> CollisionShape:
    vertices = np.asarray(mesh.vertices)
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("源网格顶点不足或包含非有限数值")
    span = np.ptp(vertices, axis=0)
    if np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=1e-10) < 3:
        raise ValueError("源网格退化（共面、共线或零体积），请手动添加基本体")
    eps = max(float(np.max(span)) * 1e-7, 1e-8)
    values = dict(
        link=link,
        kind=kind,
        alignment=alignment,
        axis=axis,
        name={
            "sphere": "球体",
            "box": "长方体",
            "cylinder": "圆柱",
            "capsule": "胶囊",
            "convex": "凸包",
        }[kind],
    )
    if kind == "convex":
        if asset_dir is None:
            raise ValueError("凸包需要资源目录")
        hull = mesh.convex_hull
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / f"convex-{uuid4().hex}.stl"
        hull.export(path)
        return CollisionShape(**{**values, "alignment": "free", "mesh_path": str(path.resolve())})
    if kind == "sphere":
        center, radius = trimesh.nsphere.minimum_nsphere(vertices)
        radius = max(float(radius), float(np.linalg.norm(vertices - center, axis=1).max())) + eps
        return CollisionShape(**values, xyz=tuple(center), radius=radius)
    if alignment == "link":
        rotation = pose_matrix(rpy=axis_rpy(axis))[:3, :3] if kind != "box" else np.eye(3)
    else:
        if kind == "box":
            transform, _ = trimesh.bounds.oriented_bounds(mesh)
            rotation = np.linalg.inv(transform)[:3, :3]
        else:
            # The longest principal direction becomes local Z.
            _, _, vt = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
            z = vt[0]
            x = vt[1]
            rotation = np.column_stack([x, np.cross(z, x), z])
    projected = vertices @ rotation
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = (low + high) / 2
    local = projected - center
    xyz = rotation @ center
    if alignment == "link":
        rpy = axis_rpy(axis) if kind in ("cylinder", "capsule") else (0.0, 0.0, 0.0)
    else:
        rpy = tuple(Rotation.from_matrix(rotation).as_euler("xyz"))
    if kind == "box":
        return CollisionShape(**values, xyz=tuple(xyz), rpy=rpy, size=tuple(high - low + 2 * eps))
    radial = np.linalg.norm(local[:, :2], axis=1)
    radius = float(radial.max()) + eps
    length = float(high[2] - low[2])
    if kind == "capsule":
        # Each vertex must satisfy radial² + distance_to_segment² <= radius².
        half = np.max(np.abs(local[:, 2]) - np.sqrt(np.maximum(radius**2 - radial**2, 0)))
        length = 2 * max(0.0, float(half)) + 2 * eps
    return CollisionShape(**values, xyz=tuple(xyz), rpy=rpy, radius=radius, length=length + 2 * eps)


def contains(shape: CollisionShape, vertices: np.ndarray, tolerance=1e-7) -> np.ndarray:
    local = trimesh.transform_points(vertices, np.linalg.inv(shape.transform))
    if shape.kind == "sphere":
        return np.linalg.norm(local, axis=1) <= shape.radius + tolerance
    if shape.kind == "box":
        return (np.abs(local) <= np.asarray(shape.size) / 2 + tolerance).all(axis=1)
    radial = np.linalg.norm(local[:, :2], axis=1)
    if shape.kind == "cylinder":
        return (radial <= shape.radius + tolerance) & (
            np.abs(local[:, 2]) <= shape.length / 2 + tolerance
        )
    if shape.kind == "capsule":
        beyond = np.maximum(np.abs(local[:, 2]) - shape.length / 2, 0)
        return np.hypot(radial, beyond) <= shape.radius + tolerance
    raise ValueError("覆盖检查仅支持基本体")
