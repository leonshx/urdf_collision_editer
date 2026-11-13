from pathlib import Path

import pytest

from collision_editor.document import RobotDocument

SAMPLE = Path(__file__).resolve().parents[1] / "RXL-6F-C3/urdf/RXL-6F-C3.urdf"


@pytest.fixture(scope="module")
def sample_document():
    return RobotDocument(SAMPLE)


@pytest.fixture
def tiny_document(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_text("""<?xml version="1.0"?>
<robot name="tiny" version="1.0">
  <material name="blue"><color rgba="0 0 1 1"/></material>
  <link name="base">
    <visual name="appearance"><origin xyz="0.1 0 0"/>
      <geometry><box size="0.2 0.3 0.4"/></geometry><material name="blue"/></visual>
    <collision name="original"><origin xyz="0 0 0"/>
      <geometry><box size="0.2 0.3 0.4"/></geometry></collision>
  </link>
  <link name="child"><visual><geometry><sphere radius="0.1"/></geometry></visual></link>
  <joint name="fixed" type="fixed"><parent link="base"/><child link="child"/>
    <origin xyz="1 2 3" rpy="0 0 0"/></joint>
</robot>""")
    return RobotDocument(source)
