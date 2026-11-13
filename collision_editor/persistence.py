from __future__ import annotations

import copy
import hashlib
import os
import shlex
import tempfile
from pathlib import Path

import numpy as np
import yaml
from lxml import etree as ET

from .document import RobotDocument
from .models import CollisionShape, EditorState, ParameterFile


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def floats(values) -> str:
    return " ".join(format(float(value), ".12g") for value in values)


class AssetBundler:
    """Content-addressed resources with material/texture references relocated."""

    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[Path, str] = {}

    def copy(self, source: Path) -> str:
        source = source.resolve()
        if source in self.cache:
            return self.cache[source]
        content = source.read_bytes()
        digest = hashlib.sha256(str(source).encode() + content).hexdigest()[:16]
        relative = f"assets/{digest}/{source.name}"
        self.cache[source] = relative
        destination = self.root / relative
        if destination.resolve() == source:
            return relative

        def reference(filename):
            dep = (source.parent / filename.removeprefix("file://")).resolve()
            if not dep.is_file():
                raise FileNotFoundError(f"网格关联资源缺失：{dep}")
            return os.path.relpath(self.root / self.copy(dep), destination.parent)

        if source.suffix.lower() == ".dae":
            tree = ET.fromstring(content, ET.XMLParser(resolve_entities=False, no_network=True))
            for node in tree.xpath("//*[local-name()='image']//*[local-name()='init_from']"):
                if node.text:
                    node.text = reference(node.text.strip())
            content = ET.tostring(tree, xml_declaration=True, encoding="utf-8")
        elif source.suffix.lower() in (".obj", ".mtl"):
            lines = []
            for line in content.decode("utf-8").splitlines():
                parts = shlex.split(line, comments=True)
                if parts and parts[0] == "mtllib":
                    line = "mtllib " + " ".join(reference(p) for p in parts[1:])
                elif parts and (
                    parts[0].startswith("map_") or parts[0] in ("bump", "disp", "decal")
                ):
                    line = " ".join(parts[:-1] + [reference(parts[-1])])
                lines.append(line)
            content = ("\n".join(lines) + "\n").encode()
        if not destination.exists() or destination.read_bytes() != content:
            atomic_write(destination, content)
        return relative


class ParameterIO:
    @staticmethod
    def encode(
        document: RobotDocument,
        state: EditorState,
        bundler: AssetBundler | None = None,
        *,
        output_urdf_sha256: str | None = None,
    ):
        links = {link: [] for link in document.links}
        for shape in state.shapes:
            values = shape.model_dump()
            if shape.mesh_path and bundler:
                values["mesh_path"] = bundler.copy(document.resolve(shape.mesh_path))
            links[shape.link].append(CollisionShape.model_validate(values))
        model = ParameterFile(
            robot=document.name,
            source_sha256=document.sha256,
            output_urdf_sha256=output_urdf_sha256,
            links=links,
            edited_links=sorted(state.edited_links),
        )
        return yaml.safe_dump(
            model.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        ).encode("utf-8")

    @staticmethod
    def read(path: Path, document: RobotDocument) -> tuple[EditorState, str]:
        params = ParameterFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        unknown = set(params.links) - set(document.links)
        missing = set(document.links) - set(params.links)
        if unknown or missing:
            raise ValueError(f"部件不匹配；未知：{sorted(unknown)}；缺少：{sorted(missing)}")
        shapes = []
        for link_shapes in params.links.values():
            for shape in link_shapes:
                if params.source_sha256 != document.sha256:
                    # XML indices belong to the original document layout.
                    shape = shape.changed(original_index=None)
                if shape.mesh_path:
                    resource = Path(shape.mesh_path)
                    if not resource.is_absolute():
                        resource = path.parent / resource
                    resource = resource.resolve()
                    if not resource.is_file():
                        raise FileNotFoundError(f"参数网格缺失：{resource}")
                    shape = shape.changed(mesh_path=str(resource))
                shapes.append(shape)
        # All groups are authoritative, including unchanged original groups in another file.
        state = EditorState(shapes=shapes, edited_links=set(params.links))
        note = f"匹配 {len(params.links)} 个部件、{len(shapes)} 个碰撞体。"
        if params.source_sha256 != document.sha256:
            note += " 源文件指纹不同，请检查模型是否一致。"
        return state, note


class UrdfWriter:
    def __init__(self, document: RobotDocument):
        self.document = document

    def element(self, shape: CollisionShape, mode: str) -> list:
        if shape.kind == "capsule" and (mode == "compatible" or shape.length == 0):
            if shape.length == 0:
                return self.element(shape.changed(kind="sphere", alignment="free"), mode)
            result = self.element(shape.changed(kind="cylinder", name=shape.name + "_直段"), mode)
            direction = shape.transform[:3, 2]
            for side in (-1, 1):
                xyz = np.asarray(shape.xyz) + side * shape.length / 2 * direction
                sphere = shape.changed(
                    kind="sphere",
                    alignment="free",
                    xyz=tuple(xyz),
                    name=shape.name + ("_端球1" if side == -1 else "_端球2"),
                )
                result += self.element(sphere, mode)
            return result
        node = ET.Element("collision", name=shape.name)
        originals = self.document.links[shape.link].findall("collision")
        if shape.original_index is not None and 0 <= shape.original_index < len(originals):
            node = copy.deepcopy(originals[shape.original_index])
            node.set("name", shape.name)
            for child in list(node):
                if child.tag in ("geometry", "origin"):
                    node.remove(child)
        ET.SubElement(node, "origin", xyz=floats(shape.xyz), rpy=floats(shape.rpy))
        geometry = ET.SubElement(node, "geometry")
        if shape.kind == "box":
            ET.SubElement(geometry, "box", size=floats(shape.size))
        elif shape.kind in ("sphere", "cylinder", "capsule"):
            geom = ET.SubElement(geometry, shape.kind, radius=floats([shape.radius]))
            if shape.kind != "sphere":
                geom.set("length", floats([shape.length]))
        else:
            ET.SubElement(
                geometry, "mesh", filename=shape.mesh_path, scale=floats(shape.mesh_scale)
            )
        return [node]

    def xml(self, state: EditorState, mode="compatible", bundler: AssetBundler | None = None):
        if mode not in ("compatible", "native"):
            raise ValueError("未知胶囊导出模式")
        root = copy.deepcopy(self.document.tree)
        for link in root.findall("link"):
            name = link.get("name")
            shapes = [shape for shape in state.shapes if shape.link == name]
            if name in state.edited_links:
                for node in link.findall("collision"):
                    link.remove(node)
                for shape in shapes:
                    link.extend(self.element(shape, mode))
        if mode == "native" and root.findall(".//capsule"):
            if tuple(map(int, root.get("version", "1.0").split("."))) < (1, 1):
                root.set("version", "1.1")
        for node in root.findall(".//mesh") + root.findall(".//texture"):
            filename = node.get("filename")
            if not filename:
                continue
            try:
                resolved = self.document.resolve(filename)
                node.set("filename", bundler.copy(resolved) if bundler else str(resolved))
            except FileNotFoundError:
                if bundler:
                    raise
                # Keep incomplete models editable; a portable export must resolve all resources.
        return ET.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=False)

    def save(self, state: EditorState, output: Path, mode="compatible") -> Path:
        output = output.expanduser().resolve()
        target = output / "robot.urdf"
        self.document.protect_output(target)
        self.document.protect_output(output / "collisions.yaml")
        bundler = AssetBundler(output)
        xml = self.xml(state, mode, bundler)
        parameters = ParameterIO.encode(
            self.document, state, bundler, output_urdf_sha256=hashlib.sha256(xml).hexdigest()
        )
        # Versioned payloads are committed before the public URDF entry point.
        atomic_write(output / "collisions.yaml", parameters)
        atomic_write(target, xml)
        return target

    def export_parameters(self, state: EditorState, path: Path):
        path = path.expanduser().resolve()
        self.document.protect_output(path)
        atomic_write(path, ParameterIO.encode(self.document, state, AssetBundler(path.parent)))
