import json
import re
import struct
import zlib
from functools import lru_cache
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config" / "color_palettes.json"
STATIC_PALETTE_DIR = APP_ROOT / "static" / "palettes"
PALETTE_REFERENCE_NOTE = (
    "最后一张参考图为色板参考图，仅用于控制整体色调、冷暖关系、明暗层次和氛围，"
    "不参考其具体构图、色块形状或文字内容。"
)


def _text(value) -> str:
    return str(value or "").strip()


@lru_cache(maxsize=1)
def load_color_palettes() -> dict[str, dict]:
    if not CONFIG_PATH.exists():
        return {}
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    palettes = {}
    for item in data if isinstance(data, list) else []:
        palette_id = _text(item.get("id")) if isinstance(item, dict) else ""
        if palette_id:
            palettes[palette_id] = item
    return palettes


def get_color_palette(palette_id: str | None) -> dict | None:
    palette_id = _text(palette_id)
    if not palette_id:
        return None
    return load_color_palettes().get(palette_id)


def board_color_palette(board: dict) -> dict | None:
    return get_color_palette(board.get("palette_id"))


def color_palette_video_note(board: dict) -> str:
    palette = board_color_palette(board)
    if not palette:
        return ""
    suffix = _text(palette.get("prompt_suffix"))
    if suffix:
        return f"{suffix}\n{PALETTE_REFERENCE_NOTE}"
    return PALETTE_REFERENCE_NOTE


def ensure_all_color_palette_reference_images() -> list[str]:
    paths = []
    for palette in load_color_palettes().values():
        path = ensure_palette_reference_image(palette)
        if path:
            paths.append(path)
    return paths


def ensure_palette_reference_image(palette: dict) -> str | None:
    colors = [_parse_hex_color(color) for color in palette.get("hex", [])]
    colors = [color for color in colors if color is not None]
    if not colors:
        return None

    image_name = _safe_image_name(palette.get("reference_image_name") or f"{palette.get('id')}_palette.png")
    output = STATIC_PALETTE_DIR / image_name
    if output.exists():
        return str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_palette_png(output, colors)
    return str(output)


def ensure_color_palette_reference_image(project_dir: Path, board: dict) -> str | None:
    palette = board_color_palette(board)
    if not palette:
        return None
    return ensure_palette_reference_image(palette)


def _safe_image_name(name: str) -> str:
    name = Path(_text(name)).name
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', "_", name)
    if not cleaned.lower().endswith(".png"):
        cleaned += ".png"
    return cleaned or "palette.png"


def _parse_hex_color(value: str) -> tuple[int, int, int] | None:
    value = _text(value)
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return None
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def _write_palette_png(path: Path, colors: list[tuple[int, int, int]], width: int = 720, height: int = 1280) -> None:
    rows = []
    stripe_height = max(1, height // len(colors))
    for y in range(height):
        color = colors[min(len(colors) - 1, y // stripe_height)]
        rows.append(b"\x00" + bytes(color) * width)
    raw = b"".join(rows)

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    png = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw, level=9)),
        chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(png))
