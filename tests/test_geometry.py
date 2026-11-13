import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from collision_editor.app import EditorApp
from collision_editor.geometry import contains, fit, primitive_mesh
from collision_editor.models import CollisionShape, axis_rpy


def test_quick_slider_step_keeps_a_usable_adjustment_resolution():
    assert EditorApp._slider_step(500) == 0.5
    assert EditorApp._slider_step(1) == 0.001


def test_sample_package_resolution_and_nine_links(sample_document):
    document = sample_document
    assert len(document.links) == 9
    assert len(document.original.shapes) == 9
    assert not document.warnings
    assert all(document.mesh(shape).vertices.size for shape in document.original.shapes)
    assert any(shape.mesh_path.endswith(".STL") for shape in document.original.shapes)


@pytest.mark.parametrize("kind", ["sphere", "box", "cylinder", "capsule"])
@pytest.mark.parametrize(
    "alignment,axis", [("free", "Z"), ("link", "X"), ("link", "Y"), ("link", "Z")]
)
def test_fits_cover_transformed_source_vertices(kind, alignment, axis):
    mesh = trimesh.creation.box(extents=(0.23, 0.47, 0.81))
    transform = trimesh.transformations.euler_matrix(0.31, -0.22, 0.57)
    transform[:3, 3] = (1.2, -0.8, 0.5)
    mesh.apply_transform(transform)
    shape = fit(mesh, "base", kind, alignment=alignment, axis=axis)
    assert contains(shape, mesh.vertices).all()
    if alignment == "link":
        expected = (0, 0, 0) if kind in ("box", "sphere") else axis_rpy(axis)
        np.testing.assert_allclose(
            shape.transform[:3, :3],
            CollisionShape(link="base", kind=kind, rpy=expected).transform[:3, :3],
            atol=1e-8,
        )


def test_capsule_mesh_centered_and_has_expected_extent():
    shape = CollisionShape(link="base", kind="capsule", radius=0.2, length=0.7)
    low, high = primitive_mesh(shape).bounds
    np.testing.assert_allclose((low + high) / 2, (0, 0, 0), atol=1e-8)
    np.testing.assert_allclose(high - low, (0.4, 0.4, 1.1), atol=1e-8)


def test_degenerate_hull_and_invalid_dimensions(tmp_path):
    planar = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        faces=[[0, 1, 2], [1, 2, 3]],
        process=False,
    )
    with pytest.raises(ValueError, match="退化"):
        fit(planar, "base", "convex", asset_dir=tmp_path)
    assert not list(tmp_path.iterdir())
    for values in (
        {"kind": "box", "size": (1, 0, 1)},
        {"kind": "sphere", "radius": 0},
        {"kind": "cylinder", "length": 0},
    ):
        with pytest.raises(ValidationError):
            CollisionShape(link="base", **values)


def test_source_mesh_composes_multiple_visual_origins_and_scales(tmp_path):
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.export(tmp_path / "part.stl")
    source = tmp_path / "two.urdf"
    source.write_text("""<robot name="two"><link name="base">
      <visual><origin xyz="1 0 0"/><geometry><mesh filename="part.stl" scale="2 1 1"/></geometry></visual>
      <visual><origin xyz="0 3 0"/><geometry><mesh filename="part.stl" scale="1 2 1"/></geometry></visual>
    </link></robot>""")
    from collision_editor.document import RobotDocument

    document = RobotDocument(source)
    actual = document.source_mesh("base", "visual")
    np.testing.assert_allclose(actual.bounds, [[-0.5, -0.5, -0.5], [2, 4, 0.5]])
