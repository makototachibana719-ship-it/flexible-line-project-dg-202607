"""Username-independent project staging for the delivered Peg runtime."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET


def discover_installed_assets() -> Path:
    """Resolve assets from the imported package, never from a username literal."""
    from mujoco_env.tasks import base_task

    assets = Path(base_task.__file__).resolve().parent.parent / "assets"
    if not assets.is_dir():
        raise FileNotFoundError(f"installed mujoco_env assets not found: {assets}")
    return assets


def runtime_paths(project_root: Path, assets: Path | None = None) -> dict[str, Path]:
    root = Path(project_root).resolve()
    runtime = root / "runtime/peg_insertion"
    return {
        "project_root": root,
        "runtime_root": runtime,
        "generated": runtime / "generated",
        "asset_link": runtime / "assets",
        "assets": Path(assets).resolve() if assets else discover_installed_assets(),
    }


def _ensure_asset_link(link: Path, assets: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != assets:
            raise RuntimeError(f"asset link points to unexpected target: {link}")
        return
    if link.exists():
        raise FileExistsError(f"refusing to replace runtime asset path: {link}")
    link.symlink_to(assets, target_is_directory=True)


def _portable_asset_path(value: str, generated_xml: Path, assets: Path) -> str:
    candidate = (generated_xml.parent / value).resolve()
    try:
        relative = candidate.relative_to(assets)
    except ValueError:
        return value
    return (Path("../assets") / relative).as_posix()


def rewrite_generated_asset_paths(xml_path: Path, assets: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        # Every generated mesh already has an explicit path. Keeping meshdir
        # would prefix those rewritten paths a second time in MuJoCo.
        compiler.attrib.pop("meshdir")
    for element in root.iter():
        if element.get("file"):
            element.set("file", _portable_asset_path(element.get("file"), xml_path, assets))
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


@contextmanager
def portable_peg_runtime(project_root: Path) -> Iterator[dict[str, Path]]:
    """Scope generated XML to project runtime and assets through a local symlink."""
    paths = runtime_paths(project_root)
    paths["generated"].mkdir(parents=True, exist_ok=True)
    _ensure_asset_link(paths["asset_link"], paths["assets"])

    from mujoco_env.tasks import base_task

    cls = base_task.RobotAssembler
    original_build = cls.build_robot_scene

    def build(self, scene_name, robot_config, _output_dir):
        xml_path = Path(original_build(self, scene_name, robot_config, paths["generated"]))
        rewrite_generated_asset_paths(xml_path, paths["assets"])
        return xml_path

    cls.build_robot_scene = build
    try:
        yield paths
    finally:
        cls.build_robot_scene = original_build
