# 第三方项目说明

本项目的实现参考了相关工具的工作流程与交互方式。截至本说明编写时，未从下列参考项目复制代码、图像或其他素材。列出其许可仅为准确标注原项目，并不表示这些许可自动适用于本仓库全部内容，也不替代本仓库自己的授权声明。

| 项目 | 参考内容 | 原项目声明 |
| --- | --- | --- |
| [Bubblify](https://pypi.org/project/bubblify/) | 在浏览器中检查 URDF 碰撞几何、调整近似形状并导出 URDF 与 YAML 的工作流程 | PyPI 项目页声明 Apache-2.0；作者 Bas van der Heijden |
| [URDF_kitchen](https://github.com/Ninagawa123/URDF_kitchen) | 可视化编辑碰撞几何的交互思路 | 其 [LICENSE](https://github.com/Ninagawa123/URDF_kitchen/blob/main/LICENSE) 声明 GPL-3.0 |
| [Robot Viewer](https://github.com/zixingjiang/robot-viewer) | 基于浏览器的机器人模型查看与关节交互方式 | 其仓库声明 MIT |

本项目使用的 Python 包由 `pyproject.toml` 列出，包括 Viser、yourdfpy、Trimesh、NumPy、SciPy、lxml、Pydantic、PyYAML 和 pycollada。各依赖继续遵循其自身的许可；分发时请根据实际打包的版本保留适用的许可与声明。

仓库中的 `RXL-6F-C3/` 是随附的示例机器人资源。使用或再分发这些模型文件时，请核对其来源与适用授权；此处的参考项目声明不涵盖该模型。
