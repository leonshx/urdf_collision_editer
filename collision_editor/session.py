from __future__ import annotations

import json
import threading
from functools import wraps
from pathlib import Path
from uuid import uuid4

import yaml

from .document import RobotDocument
from .models import CollisionShape, EditorState, ParameterFile
from .persistence import ParameterIO, UrdfWriter, atomic_write


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapped


class CommandHistory:
    def __init__(self):
        self.undo_stack: list[EditorState] = []
        self.redo_stack: list[EditorState] = []

    def record(self, state):
        self.undo_stack.append(state.model_copy(deep=True))
        self.redo_stack.clear()

    def undo(self, current):
        if not self.undo_stack:
            return current
        self.redo_stack.append(current.model_copy(deep=True))
        return self.undo_stack.pop()

    def redo(self, current):
        if not self.redo_stack:
            return current
        self.undo_stack.append(current.model_copy(deep=True))
        return self.redo_stack.pop()


class EditorSession:
    def __init__(self, document: RobotDocument, workspace: Path | None = None, *, recover=False):
        self.document = document
        self.workspace = (workspace or Path(".collision-editor") / uuid4().hex).resolve()
        if not recover and (self.workspace / "session.json").exists():
            raise ValueError("该工作区已有会话，请选择新目录或使用恢复工作区")
        for filename in (
            "source.urdf",
            "state.json",
            "working.urdf",
            "parameters.yaml",
            "session.json",
        ):
            document.protect_output(self.workspace / filename)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.assets = self.workspace / "generated"
        self.assets.mkdir(exist_ok=True)
        self.writer = UrdfWriter(document)
        self.history = CommandHistory()
        self.lock = threading.RLock()
        self.state = document.original.model_copy(deep=True)
        # A matching sidecar restores logical capsules, convex types and alignment.
        # An unrelated file must not be applied just because its name matches.
        sidecar = document.source.parent / "collisions.yaml"
        if not recover and sidecar.is_file():
            try:
                parameters = ParameterFile.model_validate(yaml.safe_load(sidecar.read_text()))
                if parameters.output_urdf_sha256 == document.sha256:
                    self.state, _ = ParameterIO.read(sidecar, document)
            except (ValueError, OSError, yaml.YAMLError) as error:
                document.warnings.append(f"未加载配套参数：{error}")
        self.saved_state = self.state.model_dump_json()
        self.mode = "compatible"
        self.saved_mode = self.mode
        self.output: Path | None = None
        self.error = ""
        self.sync_status = "已同步"
        self.on_status = lambda: None
        self.timer: threading.Timer | None = None
        self.closed = False
        self.gesture_before = None
        if recover:
            self.state = EditorState.model_validate_json(
                (self.workspace / "state.json").read_bytes()
            )
            if not {s.link for s in self.state.shapes}.issubset(document.links):
                raise ValueError("恢复文件部件不匹配")
            metadata = json.loads((self.workspace / "session.json").read_text())
            self.saved_state = metadata.get("saved_state", self.saved_state)
            self.mode = metadata.get("mode", "compatible")
            self.saved_mode = metadata.get("saved_mode", "compatible")
            self.output = Path(metadata["output"]) if metadata.get("output") else None
        else:
            atomic_write(self.workspace / "source.urdf", document.raw)
            self._metadata()
        self.flush()

    @classmethod
    def recover(cls, workspace: Path):
        metadata = json.loads((workspace / "session.json").read_text())
        doc = RobotDocument(
            workspace / "source.urdf",
            metadata["package_dirs"],
            source_base=Path(metadata["source_base"]),
        )
        # Keep original path protection, even when recovering from the snapshot.
        doc.source = Path(metadata["source"])
        return cls(doc, workspace, recover=True)

    def _metadata(self):
        metadata = dict(
            source=str(self.document.source),
            source_base=str(self.document.base),
            package_dirs=[str(p) for p in self.document.packages.values()],
            saved_state=self.saved_state,
            mode=self.mode,
            saved_mode=self.saved_mode,
            output=str(self.output) if self.output else None,
        )
        atomic_write(self.workspace / "session.json", json.dumps(metadata).encode())

    @property
    def dirty(self):
        saved = EditorState.model_validate_json(self.saved_state)
        return self.state != saved or self.mode != self.saved_mode

    def change(self, new_state: EditorState):
        with self.lock:
            if not ({s.link for s in new_state.shapes} | new_state.edited_links).issubset(
                self.document.links
            ):
                raise ValueError("编辑状态包含不存在的部件")
            if len({s.id for s in new_state.shapes}) != len(new_state.shapes):
                raise ValueError("碰撞体 ID 重复")
            if new_state == self.state:
                return
            if self.gesture_before is None:
                self.history.record(self.state)
            self.state = new_state.model_copy(deep=True)
            self.schedule()

    @synchronized
    def add(self, shape: CollisionShape, replace=False):
        state = self.state.model_copy(deep=True)
        if replace:
            state.shapes = [s for s in state.shapes if s.link != shape.link]
        state.shapes.append(shape)
        state.edited_links.add(shape.link)
        self.change(state)

    @synchronized
    def update(self, shape: CollisionShape):
        if not any(s.id == shape.id for s in self.state.shapes):
            raise ValueError("选中的碰撞体已不存在")
        state = self.state.model_copy(deep=True)
        state.shapes = [shape if s.id == shape.id else s for s in state.shapes]
        state.edited_links.add(shape.link)
        self.change(state)

    @synchronized
    def delete(self, shape: CollisionShape):
        state = self.state.model_copy(deep=True)
        state.shapes = [s for s in state.shapes if s.id != shape.id]
        state.edited_links.add(shape.link)
        self.change(state)

    @synchronized
    def delete_original_collisions(self, link: str):
        """Remove only source-URDF collisions for a link, retaining newly added shapes."""
        state = self.state.model_copy(deep=True)
        state.shapes = [
            shape
            for shape in state.shapes
            if shape.link != link or shape.original_index is None
        ]
        state.edited_links.add(link)
        self.change(state)

    @synchronized
    def restore_link(self, link):
        state = self.state.model_copy(deep=True)
        state.shapes = [s for s in state.shapes if s.link != link]
        state.shapes += [
            s.model_copy(deep=True) for s in self.document.original.shapes if s.link == link
        ]
        state.edited_links.discard(link)
        self.change(state)

    def begin_gesture(self):
        with self.lock:
            self.gesture_before = self.state.model_copy(deep=True)

    def end_gesture(self):
        with self.lock:
            if self.gesture_before is not None and self.gesture_before != self.state:
                self.history.record(self.gesture_before)
            self.gesture_before = None
            self.flush()

    def undo(self):
        with self.lock:
            self.end_gesture()
            self.state = self.history.undo(self.state)
            self.flush()

    def redo(self):
        with self.lock:
            self.end_gesture()
            self.state = self.history.redo(self.state)
            self.flush()

    def schedule(self):
        if self.closed:
            return
        self.sync_status = "等待同步"
        if self.timer and self.gesture_before is not None:
            self.on_status()
            return
        if self.timer:
            self.timer.cancel()
        # During a drag, retain the timer instead of indefinitely postponing writes.
        self.timer = threading.Timer(0.3, self.flush)
        self.timer.daemon = True
        self.timer.start()
        self.on_status()

    def flush(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            try:
                state_bytes = self.state.model_dump_json().encode()
                xml = self.writer.xml(self.state, self.mode)
                atomic_write(self.workspace / "state.json", state_bytes)
                atomic_write(self.workspace / "working.urdf", xml)
                atomic_write(
                    self.workspace / "parameters.yaml",
                    ParameterIO.encode(self.document, self.state),
                )
                self._metadata()
                self.error = ""
                self.sync_status = "已同步"
            except Exception as error:
                self.error = str(error)
                self.sync_status = "同步失败"
            self.on_status()

    def save(self, output: Path):
        with self.lock:
            path = self.writer.save(self.state, output, self.mode)
            self.saved_state = self.state.model_dump_json()
            self.saved_mode = self.mode
            self.output = output.resolve()
            self.flush()
            return path

    def close(self):
        with self.lock:
            self.closed = True
            self.flush()
