"""Browser based, Chinese language URDF collision editor."""

from __future__ import annotations

import argparse
import html
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np
import viser
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import XmlLexer
from scipy.spatial.transform import Rotation

from .document import RobotDocument, validate_urdf_source
from .geometry import fit
from .models import CollisionShape, axis_rpy
from .persistence import ParameterIO, atomic_write
from .session import EditorSession

LOG = logging.getLogger(__name__)
KINDS = {"sphere": "球体", "box": "长方体", "cylinder": "圆柱", "capsule": "胶囊", "convex": "凸包"}
REVERSE_KINDS = {v: k for k, v in KINDS.items()}
SOURCES = {"自动": "auto", "原始碰撞体": "collision", "可视外观": "visual"}

ROUNDED_UI_STYLE = """
<style>
  :root { --mantine-radius-default: 14px; --mantine-radius-md: 14px; --mantine-radius-lg: 20px; }
  .mantine-Button-root, .mantine-Input-input, .mantine-Select-input,
  .mantine-NumberInput-input, .mantine-Textarea-input { border-radius: 12px !important; }
  .mantine-Paper-root, .mantine-Accordion-item { border-radius: 18px !important; }
  .mantine-Accordion-control, .mantine-Accordion-label { overflow: visible !important; line-height: 1.35; }
</style>
<div style="padding:17px 18px; margin:2px 0 12px; border-radius:20px; overflow:visible;
background:linear-gradient(135deg,#0d6efd 0%,#63b3ff 100%); color:#fff;
box-shadow:0 10px 28px rgba(13,110,253,.24)">
  <div style="font-size:11px; line-height:1.35; letter-spacing:.12em; opacity:.8">URDF COLLISION WORKBENCH</div>
  <div style="font-size:21px; line-height:1.35; font-weight:800; margin-top:4px">碰撞体简化编辑</div>
  <div style="font-size:13px; line-height:1.45; margin-top:4px; opacity:.9">选择部件 · 拟合图形 · 精确调整 · 安全导出</div>
</div>
"""

SOFT_DARK_STYLE = """
<style>
html, body, #root { background:#202936 !important; }
[class*="mantine-Paper-root"], [class*="mantine-Modal-content"],
[class*="mantine-Input-input"], [class*="mantine-Textarea-input"] {
  background-color:#293545 !important; border-color:#415166 !important;
}
</style>
"""


def _pose(matrix):
    return tuple(np.asarray(matrix[:3, 3], dtype=float)), tuple(
        Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    )


class EditorApp:
    def __init__(
        self,
        *,
        urdf: Path | None = None,
        package_dirs=(),
        recover: Path | None = None,
        host="127.0.0.1",
        port=8080,
        workspace: Path | None = None,
    ):
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="collision-fit")
        self.fit_generation = 0
        self.dark_mode = False
        self.started_with_urdf = urdf is not None
        self.server = viser.ViserServer(host=host, port=port, label="URDF 碰撞体编辑器")
        self.document: RobotDocument | None = None
        self.session: EditorSession | None = None
        self.package_dirs = [Path(p) for p in package_dirs]
        self.workspace_choice = workspace
        self.selected_link = ""
        self.selected_shape_id = ""
        self.isolated = False
        self.hidden_links: set[str] = set()
        self.scene_nodes: list = []
        self.link_frames: dict[str, object] = {}
        self.shape_nodes: dict[str, object] = {}
        self.shape_signatures: dict[str, tuple] = {}
        self.gizmo = None
        self.axis_widget = None
        self.source_panel = None
        self.source_code = None
        self.source_highlight = None
        self.rotation_panel = None
        self.rotation_vector = None
        self.rotation_sliders = []
        self.joint_panel = None
        self.joint_controls: list[tuple[str, object, object]] = []
        self.joint_angles: dict[str, float] = {}
        self._syncing = False
        self.notice = ""
        self._scene_revision = 0
        self._build_gui()
        self.server.scene.add_grid(
            "/地面", width=10, height=10, cell_size=0.1, section_size=1, plane="xy"
        )
        if recover:
            self._load_recovery(Path(recover))
        elif urdf:
            self._load_urdf(Path(urdf))
        self.server.on_client_connect(lambda client: self._focus_all(client))

    def _build_gui(self):
        gui = self.server.gui
        gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            dark_mode=self.dark_mode,
            brand_color=(13, 110, 253),
            show_logo=False,
            show_share_button=False,
        )
        gui.add_html(ROUNDED_UI_STYLE)
        self.theme_style = gui.add_html("")
        self.theme_button = gui.add_button("🌙 深色模式")
        self.source_open_button = gui.add_button("打开 URDF 源码编辑器", color="blue")
        self.joint_open_button = gui.add_button("打开关节角度调整")
        self.theme_button.on_click(lambda e: self._toggle_theme())
        self.source_open_button.on_click(lambda e: self._run(self._open_source_editor))
        self.joint_open_button.on_click(lambda e: self._run(self._open_joint_editor))
        with gui.add_folder("打开模型", expand_by_default=True):
            self.urdf_path = gui.add_text("URDF 文件路径", "")
            self.package_path = gui.add_text(
                "ROS 包目录（分号分隔）", ";".join(map(str, self.package_dirs))
            )
            self.open_button = gui.add_button("打开 URDF")
            self.recover_path = gui.add_text("恢复工作区路径", "")
            self.recover_button = gui.add_button("恢复工作区")
        self.open_button.on_click(
            lambda e: self._run(lambda: self._load_urdf(Path(self.urdf_path.value)))
        )
        self.recover_button.on_click(
            lambda e: self._run(lambda: self._load_recovery(Path(self.recover_path.value)))
        )
        self.status = gui.add_markdown("尚未打开模型")
        with gui.add_folder("场景与选择"):
            self.link_search = gui.add_text("搜索部件", "")
            self.link_select = gui.add_dropdown("当前部件", ["（未加载）"])
            self.shape_select = gui.add_dropdown("当前碰撞体", ["（无）"])
            self.show_visual = gui.add_checkbox("显示外观", True)
            self.show_original = gui.add_checkbox("显示原始碰撞体", False)
            self.show_edited = gui.add_checkbox("显示编辑碰撞体", True)
            self.visual_opacity = gui.add_slider("外观透明度", 0.05, 1.0, 0.05, 0.35)
            self.original_opacity = gui.add_slider("原始透明度", 0.05, 1.0, 0.05, 0.25)
            self.edited_opacity = gui.add_slider("编辑透明度", 0.05, 1.0, 0.05, 0.75)
            self.isolate_button = gui.add_button("仅看当前部件")
            self.hide_button = gui.add_button("隐藏当前部件")
            self.restore_visibility_button = gui.add_button("显示全部部件")
            self.focus_button = gui.add_button("聚焦当前部件")
            self.focus_all_button = gui.add_button("聚焦整机")
        self.link_search.on_update(lambda e: self._filter_links() if e.client else None)
        self.link_select.on_update(
            lambda e: self._choose_link(self.link_select.value) if e.client else None
        )
        self.shape_select.on_update(
            lambda e: self._choose_shape(self.shape_select.value) if e.client else None
        )
        for control in (self.show_visual, self.show_original, self.show_edited):
            control.on_update(lambda e: self._apply_visibility() if e.client else None)
        for control in (self.visual_opacity, self.original_opacity, self.edited_opacity):
            control.on_update(lambda e: self._run(self._update_opacity) if e.client else None)
        self.isolate_button.on_click(lambda e: self._toggle_isolate())
        self.hide_button.on_click(lambda e: self._hide_link())
        self.restore_visibility_button.on_click(lambda e: self._restore_visibility())
        self.focus_button.on_click(lambda e: self._focus())
        self.focus_all_button.on_click(lambda e: self._focus_all())
        with gui.add_folder("添加与拟合"):
            self.kind_select = gui.add_dropdown(
                "形状类型", list(REVERSE_KINDS), initial_value="长方体"
            )
            self.source_select = gui.add_dropdown("拟合来源", list(SOURCES), initial_value="自动")
            self.new_alignment = gui.add_dropdown(
                "新形状对齐", ["自由", "部件轴"], initial_value="自由"
            )
            self.new_axis = gui.add_dropdown("新形状轴向", ["X", "Y", "Z"], initial_value="Z")
            self.add_button = gui.add_button("手动添加")
            self.fit_button = gui.add_button("拟合并添加")
            self.replace_button = gui.add_button("拟合并替换当前部件全部碰撞体", color="red")
            self.duplicate_button = gui.add_button("复制选中")
            self.delete_button = gui.add_button("删除选中", color="red")
            self.rename_field = gui.add_text("名称", "")
            self.delete_original_button = gui.add_button("删除当前部件原始碰撞体", color="red")
            self.restore_button = gui.add_button("恢复当前部件原始碰撞体")
        self.add_button.on_click(lambda e: self._run(self._manual_add))
        self.fit_button.on_click(lambda e: self._fit_async(False))
        self.replace_button.on_click(lambda e: self._fit_async(True))
        self.duplicate_button.on_click(lambda e: self._run(self._duplicate))
        self.delete_button.on_click(lambda e: self._run(self._delete))
        self.delete_original_button.on_click(lambda e: self._run(self._delete_original_collisions))
        self.restore_button.on_click(lambda e: self._run(self._restore_link))
        self.rename_field.on_update(lambda e: self._edit_name() if e.client else None)
        with gui.add_folder("精确编辑"):
            self.length_unit = gui.add_dropdown("距离单位", ["mm", "m"], initial_value="mm")
            self.xyz = gui.add_vector3("中心位置 XYZ", (0.0, 0.0, 0.0), step=1.0)
            self.rpy_summary = gui.add_markdown("**旋转 RPY（°）：** X 0 · Y 0 · Z 0")
            self.rotation_open_button = gui.add_button("打开碰撞体旋转调整")
            self.size = gui.add_vector3("长方体尺寸 XYZ", (100.0, 100.0, 100.0), step=1.0)
            self.radius = gui.add_number("半径", 50.0, min=0.001, step=0.1)
            self.length = gui.add_number("圆柱/胶囊直段长度", 100.0, min=0, step=0.1)
            self.quick_range_folder = gui.add_folder("半径与长度滑条范围", expand_by_default=False)
            with self.quick_range_folder:
                self.radius_max = gui.add_number("半径滑条最大值 (mm)", 500.0, min=0.001, step=1.0)
                self.length_max = gui.add_number("长度滑条最大值 (mm)", 1000.0, min=0.001, step=1.0)
                self.radius_slider = None
                self.length_slider = None
            self.alignment = gui.add_dropdown("对齐方式", ["自由", "部件轴"], initial_value="自由")
            self.axis = gui.add_dropdown("轴向", ["X", "Y", "Z"], initial_value="Z")
        self._build_quick_sliders()
        self.length_unit.on_update(lambda e: self._refresh_fields() if e.client else None)
        for field in (self.xyz, self.size, self.radius, self.length):
            field.on_update(lambda e: self._run(self._edit_numeric) if e.client else None)
        self.rotation_open_button.on_click(lambda e: self._run(self._open_rotation_editor))
        self.radius_max.on_update(lambda e: self._rebuild_quick_sliders() if e.client else None)
        self.length_max.on_update(lambda e: self._rebuild_quick_sliders() if e.client else None)
        self.alignment.on_update(lambda e: self._run(self._edit_alignment) if e.client else None)
        self.axis.on_update(lambda e: self._run(self._edit_alignment) if e.client else None)
        with gui.add_folder("撤销与文件", expand_by_default=False):
            self.undo_button = gui.add_button("撤销")
            self.redo_button = gui.add_button("重做")
            self.mode = gui.add_dropdown(
                "胶囊导出", ["兼容模式（三个基本体）", "原生胶囊（URDF 1.1）"]
            )
            self.output_dir = gui.add_text("输出目录", "")
            self.save_button = gui.add_button("保存")
            self.save_as_button = gui.add_button("另存为")
            self.yaml_path = gui.add_text("参数 YAML 路径", "")
            self.import_button = gui.add_button("导入 YAML（预览）")
            self.export_button = gui.add_button("导出 YAML")
        self.undo_button.on_click(lambda e: self._run(self._undo))
        self.redo_button.on_click(lambda e: self._run(self._redo))
        self.mode.on_update(lambda e: self._run(self._change_mode) if e.client else None)
        self.save_button.on_click(lambda e: self._run(lambda: self._save(False)))
        self.save_as_button.on_click(lambda e: self._run(lambda: self._save(True)))
        self.import_button.on_click(lambda e: self._run(self._import_preview))
        self.export_button.on_click(lambda e: self._run(self._export_yaml))
        self._apply_theme_style()

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            dark_mode=self.dark_mode,
            brand_color=(13, 110, 253),
            show_logo=False,
            show_share_button=False,
        )
        self.theme_button.label = "☀️ 浅色模式" if self.dark_mode else "🌙 深色模式"
        self._apply_theme_style()
        self._refresh_source_highlight()

    def _apply_theme_style(self):
        self.theme_style.content = SOFT_DARK_STYLE if self.dark_mode else ""

    @staticmethod
    def _slider_step(maximum: float) -> float:
        """Keep about one thousand steps while preserving sub-millimetre precision."""
        return max(maximum / 1000, 0.001)

    def _build_quick_sliders(self):
        """Recreate range-controlled sliders because Viser slider bounds are immutable."""
        old_radius, old_length = self.radius_slider, self.length_slider
        for slider in (old_radius, old_length):
            if slider is not None:
                slider.remove()
        radius_max = float(self.radius_max.value)
        length_max = float(self.length_max.value)
        with self.quick_range_folder:
            self.radius_slider = self.server.gui.add_slider(
                "半径快速调整 (mm)", 0.001, radius_max, self._slider_step(radius_max), min(50.0, radius_max)
            )
            self.length_slider = self.server.gui.add_slider(
                "长度快速调整 (mm)", 0.0, length_max, self._slider_step(length_max), min(100.0, length_max)
            )
        self.radius_slider.on_update(
            lambda e: self._run(lambda: self._edit_slider("radius")) if e.client else None
        )
        self.length_slider.on_update(
            lambda e: self._run(lambda: self._edit_slider("length")) if e.client else None
        )
        if self.session:
            self._refresh_quick_sliders(self._shape())

    def _run(self, action):
        try:
            with self.lock:
                self.notice = ""
                action()
                self._status()
        except Exception as exc:
            LOG.exception("编辑操作失败")
            self._status(str(exc))

    def _status(self, message=""):
        if message:
            self.notice = message
        session = self.session
        if session is None:
            self.status.content = f"**状态：** {message or '尚未打开模型'}"
            return
        dirty = "有未保存修改" if session.dirty else "已保存"
        details = f"**{self.document.name}** · {dirty} · {session.sync_status}"
        details += f"\n\n工作副本：`{session.workspace / 'working.urdf'}`"
        if session.error or self.notice:
            details += f"\n\n**提示：** {session.error or self.notice}"
        if self.document.warnings:
            details += "\n\n**资源提示：** " + "；".join(self.document.warnings[:3])
        self.status.content = details

    def _attach_session(self, session):
        if self.session:
            self.session.close()
        self.fit_generation += 1
        self.session = session
        self.document = session.document
        self.session.on_status = self._status
        self.selected_link = next(iter(self.document.links))
        self.selected_shape_id = ""
        self.hidden_links.clear()
        self.isolated = False
        self.urdf_path.value = str(self.document.source)
        self.recover_path.value = str(session.workspace)
        self.output_dir.value = str(
            session.output or (self.document.source.parent / "collision_output")
        )
        self.yaml_path.value = str(session.workspace / "parameters.yaml")
        self.mode.value = (
            "原生胶囊（URDF 1.1）" if session.mode == "native" else "兼容模式（三个基本体）"
        )
        self._filter_links()
        self._render_scene()
        self._build_joint_controls()
        self._focus()
        self._status()

    def _load_urdf(self, path):
        if not str(path).strip() or not path.is_file():
            raise ValueError("请输入存在的本地 URDF 文件路径")
        packages = [Path(p.strip()) for p in self.package_path.value.split(";") if p.strip()]
        doc = RobotDocument(path, packages)
        self._attach_session(EditorSession(doc, self.workspace_choice))
        self.workspace_choice = None

    def _load_recovery(self, path):
        if not (path / "session.json").is_file():
            raise ValueError("工作区中找不到 session.json")
        self._attach_session(EditorSession.recover(path))

    def _shape(self):
        if not self.session:
            return None
        return next((s for s in self.session.state.shapes if s.id == self.selected_shape_id), None)

    def _filter_links(self):
        if not self.document:
            return
        query = self.link_search.value.casefold().strip()
        links = [link for link in self.document.links if query in link.casefold()]
        if self.selected_link not in links:
            links.insert(0, self.selected_link)
        self.link_select.options = links
        self.link_select.value = self.selected_link
        self._refresh_shape_list()

    def _refresh_shape_list(self):
        if not self.session:
            return
        shapes = [s for s in self.session.state.shapes if s.link == self.selected_link]
        labels = [f"{s.name} [{s.id[:8]}]" for s in shapes]
        self._shape_ids = dict(zip(labels, [s.id for s in shapes]))
        if self.selected_shape_id not in self._shape_ids.values():
            self.selected_shape_id = shapes[0].id if shapes else ""
        self.shape_select.options = labels or ["（无）"]
        self.shape_select.value = next(
            (label for label, id_ in self._shape_ids.items() if id_ == self.selected_shape_id),
            "（无）",
        )
        self._refresh_fields()
        self._set_gizmo()

    def _choose_link(self, value):
        if not self.document or value not in self.document.links:
            return
        self.selected_link = value
        self.selected_shape_id = ""
        self._filter_links()
        self._render_shapes()

    def _choose_shape(self, label):
        self.selected_shape_id = self._shape_ids.get(label, "")
        self._refresh_fields()
        self._render_shapes()

    def _refresh_fields(self):
        shape = self._shape()
        self._syncing = True
        try:
            scale = 1000 if self.length_unit.value == "mm" else 1
            self.rename_field.value = shape.name if shape else ""
            self.xyz.value = tuple(v * scale for v in shape.xyz) if shape else (0.0, 0.0, 0.0)
            degrees = np.degrees(shape.rpy) if shape else np.zeros(3)
            self.rpy_summary.content = (
                "**旋转 RPY（°）：** "
                f"X {degrees[0]:.2f} · Y {degrees[1]:.2f} · Z {degrees[2]:.2f}"
            )
            self.size.value = tuple(v * scale for v in shape.size) if shape else (0.1 * scale,) * 3
            self.radius.value = shape.radius * scale if shape else 0.05 * scale
            self.length.value = shape.length * scale if shape else 0.1 * scale
            self._refresh_quick_sliders(shape)
            self.alignment.value = "部件轴" if shape and shape.alignment == "link" else "自由"
            self.axis.value = shape.axis if shape else "Z"
            self.rotation_open_button.disabled = (
                shape is None or shape.alignment == "link" or shape.kind == "sphere"
            )
            self.alignment.disabled = shape is None or shape.kind in ("mesh", "convex")
            self.axis.disabled = (
                shape is None
                or shape.kind not in ("cylinder", "capsule")
                or shape.alignment != "link"
            )
            self.size.disabled = shape is None or shape.kind != "box"
            self.radius.disabled = shape is None or shape.kind not in (
                "sphere",
                "cylinder",
                "capsule",
            )
            self.length.disabled = shape is None or shape.kind not in ("cylinder", "capsule")
            self.radius_slider.disabled = self.radius.disabled
            self.length_slider.disabled = self.length.disabled
            self._refresh_rotation_editor(shape)
        finally:
            self._syncing = False

    def _commit_shape(self, updated):
        self.session.update(updated)
        self._render_shapes()
        self._refresh_shape_list()
        self._status()

    def _edit_name(self):
        if self._syncing or not self._shape():
            return
        name = self.rename_field.value.strip()
        if name:
            self._run(lambda: self._commit_shape(self._shape().changed(name=name)))

    def _edit_numeric(self):
        if self._syncing or not (shape := self._shape()):
            return
        scale = 0.001 if self.length_unit.value == "mm" else 1
        values = dict(
            xyz=tuple(float(x) * scale for x in self.xyz.value),
            size=tuple(float(x) * scale for x in self.size.value),
            radius=float(self.radius.value) * scale,
            length=float(self.length.value) * scale,
        )
        self._commit_shape(shape.changed(**values))

    def _refresh_quick_sliders(self, shape):
        """Show the selected absolute size, growing the user-configured range if required."""
        if self.radius_slider is None or self.length_slider is None:
            return
        radius = (shape.radius if shape else 0.05) * 1000
        length = (shape.length if shape else 0.1) * 1000
        # Preserve visibility of imported/large values; after this automatic expansion,
        # the user may deliberately set a smaller or larger maximum in the range folder.
        if radius > self.radius_max.value:
            self.radius_max.value = radius
            self._build_quick_sliders()
            return
        if length > self.length_max.value:
            self.length_max.value = length
            self._build_quick_sliders()
            return
        self.radius_slider.value = max(0.001, radius)
        self.length_slider.value = max(0.0, length)

    def _rebuild_quick_sliders(self):
        if self._syncing:
            return
        self._run(self._build_quick_sliders)

    def _edit_slider(self, dimension):
        if self._syncing or not (shape := self._shape()):
            return
        value_mm = self.radius_slider.value if dimension == "radius" else self.length_slider.value
        self._commit_shape(shape.changed(**{dimension: float(value_mm) / 1000}))

    def _edit_alignment(self):
        if self._syncing or not (shape := self._shape()):
            return
        if shape.kind in ("mesh", "convex"):
            return
        alignment = "link" if self.alignment.value == "部件轴" else "free"
        axis = self.axis.value
        if alignment == "link":
            self._align_fit_async(shape, axis)
            return
        else:
            shape = shape.changed(alignment="free", axis=axis)
        self._commit_shape(shape)

    def _align_fit_async(self, shape, axis):
        self.fit_generation += 1
        generation = self.fit_generation
        session, doc = self.session, self.document
        source = SOURCES[self.source_select.value]
        snapshot = session.state.model_dump_json()
        self._status(f"正在沿 {axis} 轴重新拟合 {shape.name}…")

        def job():
            with self.lock:
                mesh = doc.source_mesh(shape.link, source)
            return fit(
                mesh, shape.link, shape.kind, alignment="link", axis=axis, asset_dir=session.assets
            )

        future = self.executor.submit(job)

        def complete(f):
            with self.lock:
                if generation != self.fit_generation or session is not self.session:
                    return
                if snapshot != session.state.model_dump_json():
                    self._status("拟合期间模型已更改；请重新选择轴向")
                    return
                try:
                    fitted = f.result()
                    current = next((s for s in session.state.shapes if s.id == shape.id), None)
                    if current is None:
                        return
                    self._commit_shape(
                        current.changed(
                            xyz=fitted.xyz,
                            rpy=fitted.rpy,
                            size=fitted.size,
                            radius=fitted.radius,
                            length=fitted.length,
                            alignment="link",
                            axis=axis,
                        )
                    )
                except Exception as exc:
                    LOG.exception("轴向拟合失败")
                    self._status(f"轴向拟合失败：{exc}")
                    self._refresh_fields()

        future.add_done_callback(complete)

    def _manual_add(self):
        if not self.session:
            raise ValueError("请先打开 URDF")
        kind = REVERSE_KINDS[self.kind_select.value]
        if kind == "convex":
            raise ValueError("凸包由源网格生成，请使用“拟合并添加”")
        alignment = "link" if self.new_alignment.value == "部件轴" else "free"
        axis = self.new_axis.value
        shape = CollisionShape(
            link=self.selected_link,
            kind=kind,
            name=KINDS[kind],
            alignment=alignment,
            axis=axis,
            rpy=axis_rpy(axis)
            if alignment == "link" and kind in ("cylinder", "capsule")
            else (0.0, 0.0, 0.0),
        )
        self.session.add(shape)
        self.selected_shape_id = shape.id
        self._render_shapes()
        self._refresh_shape_list()

    def _fit_async(self, replace):
        if not self.session:
            self._status("请先打开 URDF")
            return
        with self.lock:
            self.fit_generation += 1
            generation = self.fit_generation
            doc, session = self.document, self.session
            link = self.selected_link
            kind = REVERSE_KINDS[self.kind_select.value]
            source = SOURCES[self.source_select.value]
            alignment = (
                "free"
                if kind == "convex"
                else ("link" if self.new_alignment.value == "部件轴" else "free")
            )
            axis = self.new_axis.value
            state_snapshot = session.state.model_dump_json()
            self._status(f"正在后台拟合 {link}…")

        def job():
            with self.lock:
                mesh = doc.source_mesh(link, source)
            return fit(mesh, link, kind, alignment=alignment, axis=axis, asset_dir=session.assets)

        future = self.executor.submit(job)

        def complete(f):
            with self.lock:
                if generation != self.fit_generation or session is not self.session:
                    return
                if state_snapshot != session.state.model_dump_json():
                    self._status("拟合期间模型已更改；请重新拟合")
                    return
                try:
                    shape = f.result()
                    self.session.add(shape, replace=replace)
                    self.selected_link = link
                    self.selected_shape_id = shape.id
                    self._render_shapes()
                    self._refresh_shape_list()
                    self._status(f"已拟合 {link}：{shape.name}")
                except Exception as exc:
                    LOG.exception("拟合失败")
                    self._status(f"拟合失败：{exc}")

        future.add_done_callback(complete)

    def _duplicate(self):
        if not (shape := self._shape()):
            return
        copy = shape.changed(id=uuid4().hex, name=shape.name + " 副本", original_index=None)
        self.session.add(copy)
        self.selected_shape_id = copy.id
        self._render_shapes()
        self._refresh_shape_list()

    def _delete(self):
        if shape := self._shape():
            self.session.delete(shape)
            self.selected_shape_id = ""
            self._render_shapes()
            self._refresh_shape_list()

    def _delete_original_collisions(self):
        if not self.session or not self.selected_link:
            return
        self.session.delete_original_collisions(self.selected_link)
        self.selected_shape_id = ""
        self._render_shapes()
        self._refresh_shape_list()
        self._status("已从导出副本删除当前部件的原始碰撞体；可撤销或恢复")

    def _restore_link(self):
        if self.session:
            self.session.restore_link(self.selected_link)
            self.selected_shape_id = ""
            self._render_shapes()
            self._refresh_shape_list()

    def _undo(self):
        if self.session:
            self.session.undo()
            self._render_shapes()
            self._refresh_shape_list()

    def _redo(self):
        if self.session:
            self.session.redo()
            self._render_shapes()
            self._refresh_shape_list()

    def _change_mode(self):
        if self.session:
            self.session.mode = "native" if self.mode.value.startswith("原生") else "compatible"
            self.session.flush()

    def _save(self, as_new):
        if not self.session:
            raise ValueError("请先打开 URDF")
        raw = self.output_dir.value.strip()
        if not raw:
            raise ValueError("请输入输出目录")
        path = Path(raw).expanduser()
        if not as_new and self.session.output:
            path = self.session.output
        saved = self.session.save(path)
        self.output_dir.value = str(path.resolve())
        self._status(f"已保存到 {saved}")

    def _export_yaml(self):
        if not self.session:
            raise ValueError("请先打开 URDF")
        path = Path(self.yaml_path.value.strip()).expanduser()
        self.session.writer.export_parameters(self.session.state, path)
        self._status(f"已导出 {path.resolve()}")

    def _open_source_editor(self):
        if not self.session:
            raise ValueError("请先打开 URDF")
        if self.source_panel is not None:
            return
        panel = self.server.gui.add_panel()
        self.source_panel = panel
        with panel.add_tab("URDF 源码编辑器"):
            self.server.gui.add_markdown(
                "编辑的是工作副本；应用后会创建新的可恢复工作区，源文件不会被覆盖。编辑区本身支持 XML 语法高亮。"
            )
            self.source_code = self.server.gui.add_text("工作副本 URDF 源码", "", multiline=True)
            self.source_highlight = self.server.gui.add_html("")
            refresh = self.server.gui.add_button("刷新为当前工作副本")
            apply = self.server.gui.add_button("校验并应用源码", color="blue")
            close = self.server.gui.add_button("关闭源码编辑器")
        panel.float(x=-20, y=72, width=700, height=680)
        refresh.on_click(lambda e: self._run(self._refresh_source_code))
        apply.on_click(lambda e: self._run(self._apply_source_code))
        close.on_click(lambda e: self._close_source_editor())
        self._refresh_source_code()

    def _close_source_editor(self):
        if self.source_panel is not None:
            self.source_panel.remove()
            self.source_panel = None
        self.source_code = None
        self.source_highlight = None

    def _refresh_source_code(self):
        if not self.session or self.source_code is None:
            raise ValueError("请先打开 URDF")
        self.session.flush()
        self.source_code.value = (self.session.workspace / "working.urdf").read_text(encoding="utf-8")
        self._refresh_source_highlight()
        self._status("已载入当前工作副本源码")

    def _refresh_source_highlight(self):
        if self.source_code is None or self.source_highlight is None:
            return
        style = "native" if self.dark_mode else "friendly"
        rendered = highlight(
            self.source_code.value,
            XmlLexer(),
            HtmlFormatter(noclasses=True, style=style, nowrap=False),
        )
        background = "#293545" if self.dark_mode else "#f7f9fc"
        border = "#52657a" if self.dark_mode else "#dce3ec"
        # The highlighted contenteditable surface is the actual editing area.
        # Its input event updates the hidden Viser textarea, which keeps all
        # validation and apply logic on the Python side.
        sync_script = (
            "const target=[...document.querySelectorAll('textarea')]"
            ".find((node)=>!node.dataset.xmlHighlightEditor);"
            "if(!target)return;"
            "const setter=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;"
            "setter.call(target,this.innerText);"
            "target.dispatchEvent(new Event('input',{bubbles:true}));"
        )
        self.source_highlight.content = (
            '<style>textarea { position:fixed !important; left:-10000px !important; '
            'top:-10000px !important; width:1px !important; height:1px !important; }</style>'
            '<div style="margin-top:8px"><div style="font-weight:700; margin-bottom:6px">'
            "XML 源码（可编辑，语法高亮）</div>"
            f'<div data-xml-highlight-editor="1" contenteditable="true" spellcheck="false" '
            f'oninput="{html.escape(sync_script, quote=True)}" '
            f'style="max-height:430px; overflow:auto; border:1px solid {border}; '
            f'border-radius:12px; background:{background}; padding:10px; white-space:pre; '
            'font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; outline:none">'
            f"{rendered}</div></div>"
        )

    def _apply_source_code(self):
        if not self.session or self.source_code is None:
            raise ValueError("请先打开 URDF")
        source = validate_urdf_source(self.source_code.value)

        # Treat source edits as a new working input. It never replaces the original path:
        # this makes a malformed or undesired edit recoverable through the old workspace.
        old_session = self.session
        old_session.flush()
        edited_dir = old_session.workspace.parent / f"{old_session.workspace.name}-source-edit-{uuid4().hex[:8]}"
        edited_dir.mkdir(parents=True, exist_ok=False)
        edited_file = edited_dir / "edited.urdf"
        atomic_write(edited_file, source)
        document = RobotDocument(edited_file, self.package_dirs, source_base=self.document.base)
        session = EditorSession(document, edited_dir / "workspace")
        self._attach_session(session)
        self._refresh_source_code()
        self._status("URDF 源码已校验并应用到新的工作副本；原会话仍保留，可单独恢复")

    def _open_rotation_editor(self):
        shape = self._shape()
        if shape is None:
            raise ValueError("请先选择碰撞体")
        if shape.alignment == "link" or shape.kind == "sphere":
            raise ValueError("当前碰撞体的旋转已锁定")
        if self.rotation_panel is not None:
            return
        panel = self.server.gui.add_panel()
        self.rotation_panel = panel
        degrees = tuple(map(float, np.degrees(shape.rpy)))
        with panel.add_tab("旋转角度调整"):
            self.server.gui.add_markdown("按部件局部坐标系调整 RPY 欧拉角，单位为度（°）。")
            self.rotation_vector = self.server.gui.add_vector3("X / Y / Z（°）", degrees, step=1.0)
            self.rotation_sliders = [
                self.server.gui.add_slider(label, -180.0, 180.0, 1.0, value)
                for label, value in zip(("X / Roll（°）", "Y / Pitch（°）", "Z / Yaw（°）"), degrees)
            ]
            reset = self.server.gui.add_button("归零旋转")
            close = self.server.gui.add_button("关闭旋转调整")
        panel.float(x=-20, y=-24, width=380, height=420)
        self.rotation_vector.on_update(
            lambda e: self._run(lambda: self._set_rotation_degrees(self.rotation_vector.value))
            if e.client
            else None
        )
        for index, slider in enumerate(self.rotation_sliders):
            slider.on_update(
                lambda e, i=index: self._run(lambda: self._set_rotation_axis(i, e.target.value))
                if e.client
                else None
            )
        reset.on_click(lambda e: self._run(lambda: self._set_rotation_degrees((0.0, 0.0, 0.0))))
        close.on_click(lambda e: self._close_rotation_editor())

    def _close_rotation_editor(self):
        if self.rotation_panel is not None:
            self.rotation_panel.remove()
            self.rotation_panel = None
        self.rotation_vector = None
        self.rotation_sliders = []

    def _set_rotation_axis(self, index, value):
        if self.rotation_vector is None:
            return
        degrees = list(self.rotation_vector.value)
        degrees[index] = float(value)
        self._set_rotation_degrees(degrees)

    def _set_rotation_degrees(self, degrees):
        if self._syncing or not (shape := self._shape()):
            return
        if shape.alignment == "link" or shape.kind == "sphere":
            return
        self._commit_shape(shape.changed(rpy=tuple(np.radians(degrees))))

    def _refresh_rotation_editor(self, shape):
        if self.rotation_panel is None or self.rotation_vector is None:
            return
        degrees = tuple(map(float, np.degrees(shape.rpy))) if shape else (0.0, 0.0, 0.0)
        locked = shape is None or shape.alignment == "link" or shape.kind == "sphere"
        self.rotation_vector.value = degrees
        self.rotation_vector.disabled = locked
        for slider, value in zip(self.rotation_sliders, degrees):
            slider.value = float(np.clip(value, -180.0, 180.0))
            slider.disabled = locked

    def _import_preview(self):
        if not self.session:
            raise ValueError("请先打开 URDF")
        path = Path(self.yaml_path.value.strip()).expanduser()
        state, note = ParameterIO.read(path, self.document)
        import_session = self.session
        current = len(self.session.state.shapes)
        with self.server.gui.add_modal("导入参数预览") as modal:
            self.server.gui.add_markdown(
                f"将从 `{path}` 导入。\n\n{note}\n\n当前 {current} 个碰撞体；导入后 {len(state.shapes)} 个。"
                "\n\n确认后可用“撤销”恢复。"
            )
            apply_button = self.server.gui.add_button("应用导入")
            cancel_button = self.server.gui.add_button("取消")

        def apply(event):
            def commit():
                if self.session is not import_session:
                    modal.close()
                    raise ValueError("模型已切换，请重新预览参数导入")
                if self.session:
                    self.session.change(state)
                    self._render_shapes()
                    self._refresh_shape_list()
                modal.close()

            self._run(commit)

        apply_button.on_click(apply)
        cancel_button.on_click(lambda e: modal.close())

    def _clear_scene(self):
        if self.gizmo:
            self.gizmo.remove()
            self.gizmo = None
        if self.axis_widget:
            self.axis_widget.remove()
            self.axis_widget = None
        for node in reversed(self.scene_nodes):
            try:
                node.remove()
            except Exception:
                pass
        self.scene_nodes.clear()
        self.link_frames.clear()
        self.shape_nodes.clear()
        self.shape_signatures.clear()

    def _add_mesh(self, name, mesh, color, opacity, position=(0, 0, 0), wxyz=(1, 0, 0, 0)):
        node = self.server.scene.add_mesh_simple(
            name,
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.uint32),
            color=color,
            opacity=float(opacity),
            side="double",
            position=position,
            wxyz=wxyz,
        )
        self.scene_nodes.append(node)
        return node

    def _render_scene(self):
        if not self.document:
            return
        self._clear_scene()
        for i, link in enumerate(self.document.links):
            position, wxyz = _pose(self.document.world_transform(link))
            base = f"/robot/link_{i}"
            frame = self.server.scene.add_frame(base, show_axes=False, position=position, wxyz=wxyz)
            self.scene_nodes.append(frame)
            self.link_frames[link] = frame
            element = self.document.links[link]
            for kind, tag, color, opacity in (
                ("visual", "visual", (110, 160, 210), self.visual_opacity.value),
                ("original", "collision", (255, 135, 70), self.original_opacity.value),
            ):
                for j, xml in enumerate(element.findall(tag)):
                    try:
                        shape = self.document.parse_shape(link, xml)
                        mesh = self.document.mesh(shape)
                        pose, quat = _pose(shape.transform)
                        node = self._add_mesh(
                            f"{base}/{kind}_{j}", mesh, color, opacity, pose, quat
                        )
                        node.on_click(
                            lambda e, selected_link=link: self._run(
                                lambda: self._choose_link(selected_link)
                            )
                        )
                    except Exception as exc:
                        LOG.warning("无法显示 %s %s: %s", link, tag, exc)
                        warning = f"{link} {tag} 显示失败：{exc}"
                        if warning not in self.document.warnings:
                            self.document.warnings.append(warning)
        self._render_shapes()
        self._apply_visibility()

    def _render_shapes(self):
        if not self.session:
            return
        present = {shape.id for shape in self.session.state.shapes}
        for id_, node in list(self.shape_nodes.items()):
            if id_ not in present:
                node.remove()
                self.scene_nodes = [item for item in self.scene_nodes if item is not node]
                del self.shape_nodes[id_]
                self.shape_signatures.pop(id_, None)
        links = list(self.document.links)
        for shape in self.session.state.shapes:
            try:
                pose, quat = _pose(shape.transform)
                signature = (
                    shape.kind,
                    shape.size,
                    shape.radius,
                    shape.length,
                    shape.mesh_path,
                    shape.mesh_scale,
                    shape.link,
                    shape.id == self.selected_shape_id,
                    self.edited_opacity.value,
                )
                existing = self.shape_nodes.get(shape.id)
                if existing and self.shape_signatures.get(shape.id) == signature:
                    existing.position = pose
                    existing.wxyz = quat
                    continue
                if existing:
                    existing.remove()
                    self.scene_nodes = [item for item in self.scene_nodes if item is not existing]
                mesh = self.document.mesh(shape)
                base = f"/robot/link_{links.index(shape.link)}"
                node = self._add_mesh(
                    f"{base}/edited_{shape.id}",
                    mesh,
                    (240, 205, 55) if shape.id != self.selected_shape_id else (80, 255, 125),
                    self.edited_opacity.value,
                    pose,
                    quat,
                )
                node.on_click(lambda e, s=shape: self._run(lambda: self._select_clicked(s)))
                self.shape_nodes[shape.id] = node
                self.shape_signatures[shape.id] = signature
            except Exception as exc:
                LOG.warning("无法显示碰撞体 %s: %s", shape.name, exc)
                warning = f"{shape.link} / {shape.name} 显示失败：{exc}"
                if warning not in self.document.warnings:
                    self.document.warnings.append(warning)
        self._apply_visibility()
        self._set_gizmo()

    def _select_clicked(self, shape):
        self.selected_link = shape.link
        self.selected_shape_id = shape.id
        self._filter_links()
        self._render_shapes()

    def _apply_visibility(self):
        if not self.document:
            return
        for i, link in enumerate(self.document.links):
            visible = link not in self.hidden_links and (
                not self.isolated or link == self.selected_link
            )
            base = f"/robot/link_{i}/"
            for node in self.scene_nodes:
                if not node.name.startswith(base):
                    continue
                category = node.name[len(base) :].split("_")[0]
                enabled = {
                    "visual": self.show_visual.value,
                    "original": self.show_original.value,
                    "edited": self.show_edited.value,
                }.get(category, True)
                node.visible = visible and enabled
        if self.gizmo:
            self.gizmo.visible = (
                self.show_edited.value and self.selected_link not in self.hidden_links
            )

    def _update_opacity(self):
        for node in self.scene_nodes:
            if not isinstance(node, viser.MeshHandle):
                continue
            category = node.name.rsplit("/", 1)[-1].split("_")[0]
            value = {
                "visual": self.visual_opacity.value,
                "original": self.original_opacity.value,
                "edited": self.edited_opacity.value,
            }.get(category)
            if value is not None:
                node.opacity = value
        for key, signature in self.shape_signatures.items():
            self.shape_signatures[key] = (*signature[:-1], self.edited_opacity.value)

    def _toggle_isolate(self):
        self.isolated = not self.isolated
        self._apply_visibility()

    def _hide_link(self):
        if self.selected_link:
            self.hidden_links.add(self.selected_link)
            self._apply_visibility()

    def _restore_visibility(self):
        self.hidden_links.clear()
        self.isolated = False
        self._apply_visibility()

    def _link_bounds_world(self, link):
        meshes = []
        for shape in self.session.state.shapes:
            if shape.link == link:
                try:
                    mesh = self.document.mesh(shape)
                    mesh.apply_transform(shape.transform)
                    mesh.apply_transform(self.document.world_transform(link))
                    meshes.append(mesh)
                except Exception:
                    pass
        if not meshes:
            try:
                mesh = self.document.source_mesh(link, "visual")
                mesh.apply_transform(self.document.world_transform(link))
                meshes.append(mesh)
            except Exception:
                pass
        if meshes:
            bounds = np.vstack([m.bounds for m in meshes])
            return (bounds.min(axis=0), bounds.max(axis=0))
        center = self.document.world_transform(link)[:3, 3]
        return center - 0.1, center + 0.1

    def _focus(self):
        if not self.session or not self.selected_link:
            return
        low, high = self._link_bounds_world(self.selected_link)
        center = (low + high) / 2
        distance = max(float(np.linalg.norm(high - low)) * 2.5, 0.5)
        for client in self.server.get_clients().values():
            client.camera.look_at = tuple(center)
            client.camera.position = tuple(
                center + np.array([distance, -distance, distance]) / np.sqrt(3)
            )

    def _focus_all(self, client=None):
        if not self.session:
            return
        bounds = np.vstack([self._link_bounds_world(link) for link in self.document.links])
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        center = (low + high) / 2
        distance = max(float(np.linalg.norm(high - low)) * 1.7, 0.3)
        clients = [client] if client else self.server.get_clients().values()
        for target in clients:
            target.camera.look_at = tuple(center)
            target.camera.position = tuple(center + distance * np.array([1, -1, 0.8]) / 1.6)
            target.camera.up_direction = (0, 0, 1)

    def _view(self, name):
        if not self.session:
            return
        low, high = self._link_bounds_world(self.selected_link)
        center = (low + high) / 2
        distance = max(float(np.linalg.norm(high - low)) * 2.5, 0.5)
        direction = {
            "正视图": (0, -1, 0),
            "后视图": (0, 1, 0),
            "左视图": (-1, 0, 0),
            "右视图": (1, 0, 0),
            "俯视图": (0, 0, 1),
            "仰视图": (0, 0, -1),
            "等轴测": (1, -1, 1),
        }[name]
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        for client in self.server.get_clients().values():
            client.camera.look_at = tuple(center)
            client.camera.position = tuple(center + distance * d)
            client.camera.up_direction = (0, 1, 0) if name in ("俯视图", "仰视图") else (0, 0, 1)

    def _set_gizmo(self):
        if self.gizmo:
            self.gizmo.remove()
            self.gizmo = None
        if self.axis_widget:
            self.axis_widget.remove()
            self.axis_widget = None
        shape = self._shape()
        if not shape or shape.link not in self.link_frames:
            return
        i = list(self.document.links).index(shape.link)
        pose, quat = _pose(shape.transform)
        self.gizmo = self.server.scene.add_transform_controls(
            f"/robot/link_{i}/gizmo",
            position=pose,
            wxyz=quat,
            disable_rotations=shape.alignment == "link" or shape.kind == "sphere",
            scale=0.15,
        )
        # Viser's transform control uses its own theme. Add an explicit local
        # frame so the selected shape always communicates the requested pure
        # X/Y/Z convention independently of that theme.
        self.axis_widget = self.server.scene.add_arrows(
            f"/robot/link_{i}/selected_axes",
            points=np.asarray(
                [((0, 0, 0), (0.18, 0, 0)), ((0, 0, 0), (0, 0.18, 0)), ((0, 0, 0), (0, 0, 0.18))],
                dtype=np.float32,
            ),
            colors=np.asarray(
                [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                dtype=np.uint8,
            ),
            shaft_radius=0.008,
            head_radius=0.018,
            head_length=0.035,
            position=pose,
            wxyz=quat,
        )

        async def update(event):
            if event.client is None:
                return
            with self.lock:
                current = self._shape()
                if not current or current.id != shape.id:
                    return
                if event.phase == "start":
                    self.session.begin_gesture()
                if event.phase in ("start", "update", "end"):
                    rpy = (
                        current.rpy
                        if current.alignment == "link"
                        else tuple(
                            Rotation.from_quat(
                                np.asarray(event.target.wxyz)[[1, 2, 3, 0]]
                            ).as_euler("xyz")
                        )
                    )
                    self.session.update(
                        current.changed(xyz=tuple(map(float, event.target.position)), rpy=rpy)
                    )
                    node = self.shape_nodes.get(current.id)
                    if node:
                        node.position = event.target.position
                        node.wxyz = event.target.wxyz
                    if self.axis_widget:
                        self.axis_widget.position = event.target.position
                        self.axis_widget.wxyz = event.target.wxyz
                    self._refresh_fields()
                if event.phase == "end":
                    self.session.end_gesture()
                self._status()

        self.gizmo.on_update(update)
        self._apply_visibility()

    def _joint_specs(self):
        if not self.document:
            return []
        specs = []
        for name in self.document.robot.actuated_joint_names:
            joint = next((j for j in self.document.robot.joint_map.values() if j.name == name), None)
            if joint is None:
                continue
            limits = getattr(joint, "limit", None)
            low = float(limits.lower) if limits and limits.lower is not None else -np.pi
            high = float(limits.upper) if limits and limits.upper is not None else np.pi
            if high <= low:
                high = low + 1
            specs.append((name, low, high))
        return specs

    def _build_joint_controls(self):
        """Reset joint-preview state after loading a different URDF."""
        self._close_joint_editor()
        self.joint_angles = {
            name: min(max(0.0, low), high) for name, low, high in self._joint_specs()
        }

    def _open_joint_editor(self):
        if not self.document:
            raise ValueError("请先打开 URDF")
        if self.joint_panel is not None:
            return
        specs = self._joint_specs()
        if not specs:
            raise ValueError("当前 URDF 没有可动关节")
        panel = self.server.gui.add_panel()
        self.joint_panel = panel
        with panel.add_tab("关节角度调整"):
            self.server.gui.add_markdown("单位为度（°）。调整仅用于三维预览，不写入 URDF。")
            for name, low, high in specs:
                low_deg, high_deg = np.degrees((low, high))
                current_deg = float(np.degrees(self.joint_angles.get(name, 0.0)))
                current_deg = float(np.clip(current_deg, low_deg, high_deg))
                slider = self.server.gui.add_slider(
                    f"{name}（°）", low_deg, high_deg, max((high_deg - low_deg) / 360, 0.1), current_deg
                )
                number = self.server.gui.add_number(f"{name} 精确角度（°）", current_deg, step=0.1)
                self.joint_controls.append((name, slider, number))
                slider.on_update(
                    lambda e, s=slider, n=number: self._run(lambda: self._set_joint_angle(s, n, True))
                    if e.client
                    else None
                )
                number.on_update(
                    lambda e, s=slider, n=number: self._run(lambda: self._set_joint_angle(s, n, False))
                    if e.client
                    else None
                )
            reset = self.server.gui.add_button("关节角度归零")
            close = self.server.gui.add_button("关闭关节角度调整")
        panel.float(x=-20, y=72, width=430, height=680)
        reset.on_click(lambda e: self._run(self._reset_joint_angles))
        close.on_click(lambda e: self._close_joint_editor())

    def _close_joint_editor(self):
        if self.joint_panel is not None:
            self.joint_panel.remove()
            self.joint_panel = None
        self.joint_controls.clear()

    def _set_joint_angle(self, slider, number, from_slider):
        value = float(slider.value if from_slider else number.value)
        if from_slider:
            number.value = value
        else:
            slider.value = value
        self._update_joints()

    def _reset_joint_angles(self):
        for name, slider, number in self.joint_controls:
            value = float(np.clip(0.0, slider.min, slider.max))
            slider.value = value
            number.value = value
        self._update_joints()

    def _update_joints(self):
        if not self.document:
            return
        with self.lock:
            cfg = {
                name: float(np.radians(slider.value)) for name, slider, _ in self.joint_controls
            }
            self.joint_angles = cfg.copy()
            self.document.robot.update_cfg(cfg)
            for link, frame in self.link_frames.items():
                frame.position, frame.wxyz = _pose(self.document.world_transform(link))

    def close(self):
        with self.lock:
            self.fit_generation += 1
            if self.session:
                self.session.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="中文浏览器版 URDF 碰撞体编辑器")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--urdf", type=Path, help="本地 URDF 文件")
    source.add_argument("--recover", type=Path, help="恢复已有 .collision-editor 工作区")
    parser.add_argument(
        "--package-dir",
        type=Path,
        action="append",
        default=[],
        help="ROS 包或工作区目录；可多次指定",
    )
    parser.add_argument("--workspace", type=Path, help="新会话工作区目录")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8080, help="端口（默认 8080）")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    app = EditorApp(
        urdf=args.urdf,
        package_dirs=args.package_dir,
        recover=args.recover,
        workspace=args.workspace,
        host=args.host,
        port=args.port,
    )
    print(f"浏览器访问 http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        app.close()


if __name__ == "__main__":
    main()
