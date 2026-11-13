from __future__ import annotations

from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation


def pose_matrix(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def axis_rpy(axis: str) -> tuple[float, float, float]:
    return {"X": (0.0, np.pi / 2, 0.0), "Y": (-np.pi / 2, 0.0, 0.0), "Z": (0.0, 0.0, 0.0)}[axis]


class CollisionShape(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(default_factory=lambda: uuid4().hex)
    link: str
    name: str = "碰撞体"
    kind: Literal["sphere", "box", "cylinder", "capsule", "convex", "mesh"]
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = Field(default=0.05, gt=0)
    length: float = Field(default=0.1, ge=0)
    size: tuple[float, float, float] = (0.1, 0.1, 0.1)
    alignment: Literal["free", "link"] = "free"
    axis: Literal["X", "Y", "Z"] = "Z"
    mesh_path: str | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Original element index preserves extensions on unchanged collisions.
    original_index: int | None = None

    @model_validator(mode="after")
    def validate_geometry(self):
        if any(v <= 0 for v in self.size):
            raise ValueError("长方体各边长必须大于零")
        if self.kind == "cylinder" and self.length <= 0:
            raise ValueError("圆柱长度必须大于零")
        if self.kind in ("mesh", "convex") and not self.mesh_path:
            raise ValueError("网格路径不能为空")
        if self.kind in ("mesh", "convex") and self.alignment == "link":
            raise ValueError("网格和凸包不支持部件轴对齐")
        if self.alignment == "link":
            expected = axis_rpy(self.axis) if self.kind in ("cylinder", "capsule") else (0, 0, 0)
            if not np.allclose(pose_matrix(rpy=self.rpy), pose_matrix(rpy=expected), atol=1e-8):
                raise ValueError("轴对齐模式下不能自由旋转")
        return self

    @property
    def transform(self) -> np.ndarray:
        return pose_matrix(self.xyz, self.rpy)

    def changed(self, **values) -> CollisionShape:
        return CollisionShape.model_validate({**self.model_dump(), **values})


class EditorState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shapes: list[CollisionShape] = Field(default_factory=list)
    edited_links: set[str] = Field(default_factory=set)


class ParameterFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    robot: str
    source_sha256: str
    output_urdf_sha256: str | None = None
    units: Literal["m,rad"] = "m,rad"
    links: dict[str, list[CollisionShape]]
    edited_links: list[str]

    @model_validator(mode="after")
    def validate_links(self):
        ids = []
        for link, shapes in self.links.items():
            for shape in shapes:
                if shape.link != link:
                    raise ValueError("参数部件分组与碰撞体所属部件不一致")
                ids.append(shape.id)
        if len(ids) != len(set(ids)):
            raise ValueError("碰撞体 ID 重复")
        if not set(self.edited_links).issubset(self.links):
            raise ValueError("编辑部件缺少参数分组")
        return self
