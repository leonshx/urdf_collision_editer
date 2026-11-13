from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from lxml import etree as ET

from .models import CollisionShape, EditorState


def validate_urdf_source(text: str) -> bytes:
    """Validate editable URDF XML and return a safe serialized document."""
    if not text.strip():
        raise ValueError("URDF 源码不能为空")
    try:
        root = ET.fromstring(
            text.encode("utf-8"), ET.XMLParser(resolve_entities=False, no_network=True)
        )
    except ET.XMLSyntaxError as error:
        raise ValueError(f"URDF XML 语法错误：{error}") from error
    if root.tag != "robot":
        raise ValueError("根节点必须是 robot")
    links = [node.get("name") for node in root.findall("link")]
    if not links or None in links or len(links) != len(set(links)):
        raise ValueError("URDF 必须包含名称唯一的 link")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def numbers(text: str | None, default) -> tuple:
    return tuple(float(v) for v in text.split()) if text else tuple(default)


def origin(element) -> tuple[tuple, tuple]:
    node = element.find("origin")
    if node is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = numbers(node.get("xyz"), (0, 0, 0))
    if node.get("quat_xyzw"):
        from scipy.spatial.transform import Rotation

        rpy = tuple(
            Rotation.from_quat(numbers(node.get("quat_xyzw"), (0, 0, 0, 1))).as_euler("xyz")
        )
    else:
        rpy = numbers(node.get("rpy"), (0, 0, 0))
    return xyz, rpy


class RobotDocument:
    def __init__(self, source: Path | str, package_dirs=(), *, source_base: Path | None = None):
        self.source = Path(source).expanduser().resolve()
        self.base = source_base or self.source.parent
        self.raw = self.source.read_bytes()
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self.tree = ET.fromstring(
            self.raw, ET.XMLParser(remove_blank_text=False, resolve_entities=False, no_network=True)
        )
        if self.tree.tag != "robot":
            raise ValueError("文件不是 URDF robot 文档")
        if any("xacro" in str(node.tag) for node in self.tree.iter()):
            raise ValueError("请先将 Xacro 展开为 URDF")
        self.name = self.tree.get("name", "robot")
        self.links = {el.get("name"): el for el in self.tree.findall("link")}
        if not self.links or None in self.links:
            raise ValueError("URDF 没有有效的 link")
        self.packages: dict[str, Path] = {}
        candidates = [self.base, *self.base.parents[:3]]
        candidates += [Path(p).expanduser().resolve() for p in package_dirs]
        for root in candidates:
            paths = [root / "package.xml", *root.glob("*/package.xml")]
            # Explicit workspace roots may contain ROS packages under src/.
            paths += list(root.glob("src/*/package.xml"))
            for package in paths:
                if package.is_file():
                    try:
                        name = ET.parse(str(package)).getroot().findtext("name")
                        if name:
                            self.packages.setdefault(name, package.parent)
                    except ET.XMLSyntaxError:
                        continue
        self.warnings: list[str] = []
        self.mesh_cache: dict[tuple, trimesh.Trimesh] = {}
        self.original = EditorState()
        for link, element in self.links.items():
            for index, collision in enumerate(element.findall("collision")):
                try:
                    self.original.shapes.append(self.parse_shape(link, collision, index))
                except ValueError as error:
                    # Keep extensions and future URDF geometry untouched until the user explicitly
                    # replaces this link's collisions. One unknown geometry must not prevent
                    # inspecting the rest of a robot.
                    self.warnings.append(f"{link}: 已保留未支持的原始碰撞体：{error}")
        # Motion graph is independent of unsupported geometry and missing resources.
        motion = copy.deepcopy(self.tree)
        motion.attrib.pop("version", None)
        for joint in motion.findall("joint"):
            node = joint.find("origin")
            if node is not None and node.get("quat_xyzw"):
                _, rpy = origin(joint)
                node.attrib.pop("quat_xyzw")
                node.set("rpy", " ".join(str(v) for v in rpy))
        for node in motion.findall("link"):
            for child in list(node):
                if child.tag in ("visual", "collision"):
                    node.remove(child)
        self.robot = yourdfpy.URDF.load(
            io.BytesIO(ET.tostring(motion)), load_meshes=False, build_scene_graph=True
        )

    def resolve(self, filename: str) -> Path:
        if filename.startswith("package://"):
            package, _, relative = filename[10:].partition("/")
            if package not in self.packages:
                raise FileNotFoundError(f"找不到 ROS 包 {package}；请补充包目录")
            path = self.packages[package] / relative
        elif filename.startswith("file://"):
            path = Path(filename[7:])
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = self.base / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到网格：{path}")
        return path

    def protect_output(self, path: Path):
        """Reject writing over any input URDF, geometry or texture, including symlinks."""
        protected = {self.source.resolve()}
        for node in self.tree.findall(".//mesh") + self.tree.findall(".//texture"):
            if node.get("filename"):
                try:
                    protected.add(self.resolve(node.get("filename")))
                except FileNotFoundError:
                    pass
        if path.resolve() in protected:
            raise ValueError("不能覆盖源 URDF 或源资源，请选择其他输出路径")

    def parse_shape(self, link, element, index=None) -> CollisionShape:
        geometry = element.find("geometry")
        if geometry is None or len(geometry) == 0:
            raise ValueError(f"部件 {link} 的 collision 缺少 geometry")
        node = geometry[0]
        xyz, rpy = origin(element)
        values = dict(
            link=link,
            name=element.get("name", f"原始碰撞体 {index + 1}" if index is not None else "外观"),
            kind=node.tag,
            xyz=xyz,
            rpy=rpy,
            original_index=index,
        )
        if node.tag == "box":
            values["size"] = numbers(node.get("size"), (0.1,) * 3)
        elif node.tag in ("sphere", "cylinder", "capsule"):
            values["radius"] = float(node.get("radius", "0.05"))
            if node.tag != "sphere":
                values["length"] = float(node.get("length", "0.1"))
        elif node.tag == "mesh":
            filename = node.get("filename", "")
            try:
                values["mesh_path"] = str(self.resolve(filename))
            except FileNotFoundError as error:
                self.warnings.append(f"{link}: {error}")
                values["mesh_path"] = filename
            values["mesh_scale"] = numbers(node.get("scale"), (1, 1, 1))
        else:
            raise ValueError(f"部件 {link} 包含不支持的几何类型：{node.tag}")
        return CollisionShape(**values)

    def mesh(self, shape: CollisionShape) -> trimesh.Trimesh:
        from .geometry import primitive_mesh

        if shape.kind not in ("mesh", "convex"):
            return primitive_mesh(shape)
        key = (shape.mesh_path, shape.mesh_scale)
        if key not in self.mesh_cache:
            path = self.resolve(shape.mesh_path)
            scene = trimesh.load_scene(path, process=False)
            mesh = scene.to_mesh()
            if not len(mesh.vertices):
                raise ValueError(f"空网格：{path}")
            mesh.apply_scale(shape.mesh_scale)
            self.mesh_cache[key] = mesh
        return self.mesh_cache[key].copy()

    def source_mesh(self, link: str, source="auto") -> trimesh.Trimesh:
        element = self.links[link]
        tag = "collision" if source == "auto" and element.findall("collision") else source
        if tag == "auto":
            tag = "visual"
        meshes = []
        for node in element.findall(tag):
            shape = self.parse_shape(link, node)
            mesh = self.mesh(shape)
            mesh.apply_transform(shape.transform)
            meshes.append(mesh)
        if not meshes:
            raise ValueError(f"部件 {link} 没有可用于拟合的 {tag} 几何")
        return trimesh.util.concatenate(meshes)

    def world_transform(self, link: str) -> np.ndarray:
        return self.robot.get_transform(link)
