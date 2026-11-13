import hashlib
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from lxml import etree as ET

from collision_editor.document import RobotDocument, validate_urdf_source
from collision_editor.geometry import fit
from collision_editor.models import CollisionShape, EditorState
from collision_editor.persistence import ParameterIO, UrdfWriter


def without_collisions(xml):
    root = ET.fromstring(xml)
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
    return ET.tostring(root, method="c14n")


def test_editable_urdf_source_requires_robot_and_unique_links():
    valid = validate_urdf_source('<robot name="test"><link name="base"/></robot>')
    assert ET.fromstring(valid).tag == "robot"

    with pytest.raises(ValueError, match="根节点"):
        validate_urdf_source("<link name='base'/>")
    with pytest.raises(ValueError, match="名称唯一"):
        validate_urdf_source("<robot><link name='base'/><link name='base'/></robot>")
    with pytest.raises(ValueError, match="XML 语法"):
        validate_urdf_source("<robot>")


def test_noncollision_xml_semantics_and_source_hash(tiny_document, tmp_path):
    original = tiny_document.raw
    state = tiny_document.original.model_copy(deep=True)
    state.shapes = [CollisionShape(link="base", kind="sphere", radius=0.15)]
    state.edited_links.add("base")
    output = UrdfWriter(tiny_document).save(state, tmp_path / "export")
    assert tiny_document.source.read_bytes() == original
    assert tiny_document.sha256 == hashlib.sha256(original).hexdigest()
    assert without_collisions(output.read_bytes()) == without_collisions(original)
    tree = ET.parse(str(output))
    assert tree.xpath("count(/robot/link[@name='base']/collision)") == 1
    assert tree.xpath("count(/robot/link[@name='child']/visual)") == 1


def test_unknown_collision_is_preserved_until_its_link_is_edited(tmp_path):
    source = tmp_path / "extensions.urdf"
    source.write_text("""<robot name="extensions">
      <link name="base"><collision name="custom"><geometry><ellipsoid size="1 2 3"/></geometry></collision>
      <collision name="known"><geometry><sphere radius="0.1"/></geometry></collision></link>
    </robot>""")
    document = RobotDocument(source)
    assert len(document.original.shapes) == 1
    assert document.warnings
    untouched = UrdfWriter(document).xml(document.original)
    assert b"ellipsoid" in untouched
    edited = EditorState(shapes=[CollisionShape(link="base", kind="box")], edited_links={"base"})
    replaced = UrdfWriter(document).xml(edited)
    assert b"ellipsoid" not in replaced


def test_capsule_compatible_endpoints_follow_rotation(tiny_document):
    shape = CollisionShape(
        link="base", kind="capsule", radius=0.1, length=0.8, xyz=(1, 2, 3), rpy=(0.31, -0.42, 0.58)
    )
    nodes = UrdfWriter(tiny_document).element(shape, "compatible")
    assert [node.find("geometry")[0].tag for node in nodes] == ["cylinder", "sphere", "sphere"]
    expected = np.asarray(shape.xyz)
    direction = shape.transform[:3, 2]
    for node, side in zip(nodes[1:], (-1, 1)):
        actual = np.fromstring(node.find("origin").get("xyz"), sep=" ")
        np.testing.assert_allclose(
            actual, expected + side * shape.length / 2 * direction, atol=1e-10
        )


def test_native_capsule_sets_urdf_version(tiny_document):
    shape = CollisionShape(link="base", kind="capsule", radius=0.1, length=0.6)
    state = EditorState(shapes=[shape], edited_links={"base"})
    root = ET.fromstring(UrdfWriter(tiny_document).xml(state, "native"))
    assert root.get("version") == "1.1"
    assert len(root.findall(".//capsule")) == 1


@pytest.mark.parametrize("axis,rpy", [("X", (0, np.pi / 2, 0)), ("Y", (-np.pi / 2, 0, 0))])
@pytest.mark.parametrize("mode", ["compatible", "native"])
def test_zero_length_aligned_capsule_exports_as_sphere(tiny_document, axis, rpy, mode):
    shape = CollisionShape(
        link="base", kind="capsule", alignment="link", axis=axis, rpy=rpy, radius=0.2, length=0
    )
    nodes = UrdfWriter(tiny_document).element(shape, mode)
    assert len(nodes) == 1
    assert nodes[0].find("geometry")[0].tag == "sphere"
    np.testing.assert_allclose(np.fromstring(nodes[0].find("origin").get("rpy"), sep=" "), rpy)


def test_save_reload_yaml_alignment_and_convex_asset(tiny_document, tmp_path):
    convex = fit(
        trimesh.creation.box(extents=(0.2, 0.3, 0.4)),
        "base",
        "convex",
        asset_dir=tmp_path / "generated",
    )
    aligned = CollisionShape(
        link="base",
        kind="cylinder",
        alignment="link",
        axis="X",
        rpy=(0, np.pi / 2, 0),
        radius=0.2,
        length=0.7,
    )
    state = EditorState(shapes=[convex, aligned], edited_links={"base"})
    output = tmp_path / "export"
    UrdfWriter(tiny_document).save(state, output)
    params = yaml.safe_load((output / "collisions.yaml").read_text())
    assert params["source_sha256"] == tiny_document.sha256
    assert params["units"] == "m,rad"
    assert params["links"]["base"][0]["mesh_path"].startswith("assets/")
    assert (output / params["links"]["base"][0]["mesh_path"]).is_file()
    loaded, note = ParameterIO.read(output / "collisions.yaml", tiny_document)
    assert len(loaded.shapes) == 2 and "匹配" in note
    restored = next(shape for shape in loaded.shapes if shape.kind == "cylinder")
    assert (restored.alignment, restored.axis) == ("link", "X")
    assert Path(
        next(shape for shape in loaded.shapes if shape.kind == "convex").mesh_path
    ).is_file()
    exported = RobotDocument(output / "robot.urdf")
    assert not exported.warnings
    assert len(
        exported.mesh(next(s for s in exported.original.shapes if s.kind == "mesh")).vertices
    )


def test_parameter_import_uses_all_groups_and_reports_hash_mismatch(tiny_document, tmp_path):
    state = tiny_document.original.model_copy(deep=True)
    parameters = tmp_path / "params.yaml"
    UrdfWriter(tiny_document).export_parameters(state, parameters)
    data = yaml.safe_load(parameters.read_text())
    data["source_sha256"] = "0" * 64
    data["edited_links"] = []
    data["links"]["child"] = [CollisionShape(link="child", kind="sphere").model_dump(mode="json")]
    parameters.write_text(yaml.safe_dump(data))
    imported, note = ParameterIO.read(parameters, tiny_document)
    assert imported.edited_links == set(tiny_document.links)
    assert {shape.link for shape in imported.shapes} == set(tiny_document.links)
    assert all(shape.original_index is None for shape in imported.shapes)
    assert "指纹不同" in note


def test_export_protects_source_urdf_and_mesh(tmp_path):
    mesh = tmp_path / "part.stl"
    trimesh.creation.box().export(mesh)
    source = tmp_path / "robot.urdf"
    source.write_text("""<robot name="protected"><link name="base"><visual>
      <geometry><mesh filename="part.stl"/></geometry></visual></link></robot>""")
    document = RobotDocument(source)
    original_source, original_mesh = source.read_bytes(), mesh.read_bytes()
    writer = UrdfWriter(document)
    with pytest.raises(ValueError, match="不能覆盖"):
        writer.save(document.original, tmp_path)
    with pytest.raises(ValueError, match="不能覆盖"):
        writer.export_parameters(document.original, mesh)
    assert source.read_bytes() == original_source
    assert mesh.read_bytes() == original_mesh


def test_missing_mesh_allows_partial_load_but_blocks_portable_export(tmp_path):
    source = tmp_path / "missing.urdf"
    source.write_text("""<robot name="missing"><link name="base">
      <visual><geometry><mesh filename="absent.stl"/></geometry></visual>
      <collision><geometry><box size="1 1 1"/></geometry></collision>
    </link></robot>""")
    document = RobotDocument(source)
    assert document.original.shapes and not document.warnings
    assert document.source_mesh("base").vertices.size
    with pytest.raises(FileNotFoundError, match="找不到网格"):
        UrdfWriter(document).save(document.original, tmp_path / "export")
    assert not (tmp_path / "export" / "robot.urdf").exists()


def test_missing_collision_mesh_is_reported_without_blocking_load(tmp_path):
    source = tmp_path / "missing_collision.urdf"
    source.write_text("""<robot name="missing"><link name="base">
      <collision><geometry><mesh filename="absent.stl"/></geometry></collision>
    </link></robot>""")
    document = RobotDocument(source)
    assert len(document.original.shapes) == 1
    assert "找不到网格" in document.warnings[0]
    with pytest.raises(FileNotFoundError):
        document.source_mesh("base")
    with pytest.raises(FileNotFoundError):
        UrdfWriter(document).save(document.original, tmp_path / "export")


def test_obj_material_and_texture_are_copied(tmp_path):
    mesh_root = tmp_path / "source"
    mesh_root.mkdir()
    (mesh_root / "tex.png").write_bytes(b"texture-data")
    (mesh_root / "mat.mtl").write_text("newmtl m\nmap_Kd tex.png\n")
    (mesh_root / "part.obj").write_text("mtllib mat.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    source = mesh_root / "robot.urdf"
    source.write_text("""<robot name="textured"><link name="base"><visual>
      <geometry><mesh filename="part.obj"/></geometry></visual></link></robot>""")
    document = RobotDocument(source)
    out = tmp_path / "export"
    UrdfWriter(document).save(document.original, out)
    exported = ET.parse(str(out / "robot.urdf"))
    obj = out / exported.find(".//mesh").get("filename")
    assert obj.is_file()
    mtl = (obj.parent / obj.read_text().split("mtllib ")[1].splitlines()[0]).resolve()
    assert mtl.is_file()
    texture = (mtl.parent / mtl.read_text().split("map_Kd ")[1].splitlines()[0]).resolve()
    assert texture.read_bytes() == b"texture-data"
