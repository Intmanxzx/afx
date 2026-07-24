from __future__ import annotations

from pathlib import Path

from autofusedebugger.camodel import camodel_env, camodel_info, resolve_soc_dir


def test_camodel_discovery(tmp_path: Path) -> None:
    sim = tmp_path / "x86_64-linux" / "simulator"
    (sim / "dav_3510" / "lib").mkdir(parents=True)
    (sim / "dav_3510" / "camodel").mkdir(parents=True)
    (sim / "dav_3510" / "camodel" / "sim").write_text("", encoding="utf-8")

    assert resolve_soc_dir(tmp_path, "dav_3510").name == "dav_3510"
    env = camodel_env(tmp_path, "dav_3510")
    assert env["AUTOFUSE_CAMODEL_SOC_DIR"].endswith("dav_3510")
    assert "dav_3510/lib" in env["LD_LIBRARY_PATH"]
    info = camodel_info(tmp_path)
    assert info["socs"][0]["has_camodel"] is True
