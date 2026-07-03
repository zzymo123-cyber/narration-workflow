from pathlib import Path

from api import color_palettes


def test_ensure_all_color_palette_reference_images_generates_static_pngs(tmp_path, monkeypatch):
    palette_dir = tmp_path / "static" / "palettes"
    monkeypatch.setattr(color_palettes, "STATIC_PALETTE_DIR", palette_dir)

    paths = color_palettes.ensure_all_color_palette_reference_images()

    assert len(paths) == 4
    assert [Path(path).name for path in paths] == [
        "suspense_cold_blue_palette.png",
        "family_warm_gray_palette.png",
        "hospital_cold_white_palette.png",
        "domestic_brown_gray_palette.png",
    ]
    for path in paths:
        palette_path = Path(path)
        assert palette_path.parent == palette_dir
        assert palette_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
