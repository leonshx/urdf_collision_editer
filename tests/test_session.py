import numpy as np
import trimesh
import yaml
from lxml import etree as ET

from collision_editor.document import RobotDocument
from collision_editor.geometry import fit
from collision_editor.models import CollisionShape
from collision_editor.session import EditorSession


def test_multi_shape_gesture_is_one_undo_step(tiny_document, tmp_path):
    session = EditorSession(tiny_document, tmp_path / "work")
    first = CollisionShape(link="base", kind="sphere", xyz=(0, 0, 0))
    second = CollisionShape(link="base", kind="box", xyz=(1, 0, 0))
    session.add(first)
    session.add(second)
    baseline = session.state.model_copy(deep=True)
    count = len(session.history.undo_stack)
    session.begin_gesture()
    session.update(first.changed(xyz=(0.2, 0, 0)))
    session.update(second.changed(xyz=(1.3, 0, 0)))
    session.update(first.changed(xyz=(0.4, 0, 0)))
    session.end_gesture()
    assert len(session.history.undo_stack) == count + 1
    moved = session.state.model_copy(deep=True)
    session.undo()
    assert session.state == baseline
    session.redo()
    assert session.state == moved
    session.close()


def test_nonzero_fk_and_link_local_edit(tiny_document, tmp_path):
    session = EditorSession(tiny_document, tmp_path / "work")
    world = tiny_document.world_transform("child")
    np.testing.assert_allclose(world[:3, 3], (1, 2, 3))
    local = np.array([0.2, -0.1, 0.3, 1.0])
    world_point = world @ local
    round_trip = np.linalg.inv(world) @ world_point
    np.testing.assert_allclose(round_trip, local)
    shape = CollisionShape(link="child", kind="sphere", xyz=tuple(round_trip[:3]))
    session.add(shape)
    assert session.state.shapes[-1].xyz == shape.xyz
    session.close()


def test_restore_original_and_recovery(tiny_document, tmp_path):
    workspace = tmp_path / "work"
    session = EditorSession(tiny_document, workspace)
    original = session.state.model_copy(deep=True)
    session.add(CollisionShape(link="base", kind="sphere", radius=0.3), replace=True)
    session.flush()
    session.close()
    recovered = EditorSession.recover(workspace)
    assert recovered.dirty
    assert len(recovered.state.shapes) == 1 and recovered.state.shapes[0].kind == "sphere"
    recovered.restore_link("base")
    assert [shape.model_dump(exclude={"id"}) for shape in recovered.state.shapes] == [
        shape.model_dump(exclude={"id"}) for shape in original.shapes
    ]
    assert "base" not in recovered.state.edited_links
    recovered.close()


def test_delete_original_collisions_keeps_added_shapes_and_can_undo(tiny_document, tmp_path):
    session = EditorSession(tiny_document, tmp_path / "work")
    original = session.state.shapes[0]
    added = CollisionShape(link="base", kind="sphere", radius=0.25)
    session.add(added)

    session.delete_original_collisions("base")
    assert session.state.shapes == [added]
    assert "base" in session.state.edited_links
    xml = session.writer.xml(session.state)
    assert b'<sphere radius="0.25"' in xml
    root = ET.fromstring(xml)
    collisions = root.find("link[@name='base']").findall("collision")
    assert [node.find("geometry")[0].tag for node in collisions] == ["sphere"]

    session.undo()
    assert {shape.id for shape in session.state.shapes} == {original.id, added.id}
    session.close()


def test_write_failure_keeps_unsynced_status(tiny_document, tmp_path, monkeypatch):
    session = EditorSession(tiny_document, tmp_path / "work")
    session.add(CollisionShape(link="base", kind="sphere", radius=0.3))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("collision_editor.session.atomic_write", fail)
    session.flush()
    assert session.sync_status == "同步失败"
    assert "disk full" in session.error
    monkeypatch.undo()
    session.flush()
    assert session.sync_status == "已同步"
    session.close()


def test_drag_keeps_first_autosave_timer(tiny_document, tmp_path, monkeypatch):
    session = EditorSession(tiny_document, tmp_path / "work")
    shape = CollisionShape(link="base", kind="sphere")
    session.add(shape)
    session.flush()
    timers = []

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay, self.callback, self.cancelled = delay, callback, False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr("collision_editor.session.threading.Timer", FakeTimer)
    session.begin_gesture()
    for x in (0.1, 0.2, 0.3):
        session.update(shape.changed(xyz=(x, 0, 0)))
    assert len(timers) == 1
    assert timers[0].delay == 0.3
    assert not timers[0].cancelled
    timers[0].callback()
    assert session.sync_status == "已同步"
    session.end_gesture()
    session.close()


def test_reopen_export_restores_logical_shapes_only_with_matching_sidecar(tiny_document, tmp_path):
    session = EditorSession(tiny_document, tmp_path / "work")
    capsule = CollisionShape(
        link="base",
        kind="capsule",
        radius=0.13,
        length=0.48,
        alignment="link",
        axis="Y",
        rpy=(-np.pi / 2, 0, 0),
    )
    convex = fit(
        trimesh.creation.box(extents=(0.2, 0.3, 0.4)), "base", "convex", asset_dir=session.assets
    )
    session.add(capsule, replace=True)
    session.add(convex)
    exported = session.save(tmp_path / "export")
    session.close()

    exported_document = RobotDocument(exported)
    assert len(exported_document.original.shapes) == 4  # compatible capsule is 3 parts
    reopened = EditorSession(exported_document, tmp_path / "reopened")
    assert len(reopened.state.shapes) == 2
    logical = next(shape for shape in reopened.state.shapes if shape.kind == "capsule")
    assert (logical.alignment, logical.axis, logical.length) == ("link", "Y", 0.48)
    assert any(shape.kind == "convex" for shape in reopened.state.shapes)
    assert not reopened.dirty
    reopened.close()

    sidecar = exported.parent / "collisions.yaml"
    data = yaml.safe_load(sidecar.read_text())
    data["output_urdf_sha256"] = "0" * 64
    sidecar.write_text(yaml.safe_dump(data))
    unrelated = EditorSession(RobotDocument(exported), tmp_path / "unrelated")
    assert len(unrelated.state.shapes) == 4
    assert all(shape.kind != "capsule" for shape in unrelated.state.shapes)
    unrelated.close()


def test_export_mode_participates_in_dirty_tracking(tiny_document, tmp_path):
    session = EditorSession(tiny_document, tmp_path / "work")
    assert not session.dirty
    session.mode = "native"
    assert session.dirty
    session.save(tmp_path / "export")
    assert not session.dirty
    session.mode = "compatible"
    assert session.dirty
    session.flush()
    session.close()
    recovered = EditorSession.recover(tmp_path / "work")
    assert recovered.mode == "compatible"
    assert recovered.saved_mode == "native"
    assert recovered.dirty
    recovered.close()
