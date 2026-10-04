"""Genera una imagen PNG ficticia de cámara ANPR por cada transacción del ledger.

Para cada transacción de data/ledger.json dibuja una placa colombiana (amarilla,
texto negro) con la placa que realmente aparece en la foto (placa_foto_anpr),
dentro de un marco gris de cámara, con la estación y la hora abajo.
Salida: data/plates/<tx_id>.png

Uso: python scripts/make_plates.py
"""

import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "ledger.json"
OUT_DIR = ROOT / "data" / "plates"

WIDTH, HEIGHT = 640, 400
FRAME_GRAY = (70, 72, 76)
SCENE_GRAY = (118, 121, 126)
PLATE_YELLOW = (255, 204, 0)
BLACK = (15, 15, 15)
WHITE = (240, 240, 240)
REC_RED = (220, 40, 40)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Intenta fuentes comunes de Windows/Linux; si no hay, usa la de Pillow."""
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def format_plate(plate: str) -> str:
    """'ABC123' -> 'ABC 123', como se ve en las placas colombianas."""
    return f"{plate[:3]} {plate[3:]}" if len(plate) == 6 else plate


def draw_centered(draw: ImageDraw.ImageDraw, cx: int, cy: int, text: str, font, fill) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top), text, font=font, fill=fill)


def make_plate_image(tx: dict) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), FRAME_GRAY)
    draw = ImageDraw.Draw(img)

    # Escena de la cámara dentro del marco gris.
    scene = (20, 44, WIDTH - 20, HEIGHT - 76)
    draw.rectangle(scene, fill=SCENE_GRAY)

    # Encabezado del marco: indicador REC y nombre de la cámara.
    small = load_font(16, bold=True)
    draw.ellipse((24, 16, 36, 28), fill=REC_RED)
    draw.text((44, 13), f"REC  CÁMARA ANPR · SURTIDOR {tx['surtidor']}", font=small, fill=WHITE)
    draw.text((WIDTH - 110, 13), tx["tx_id"], font=small, fill=WHITE)

    # Placa colombiana: fondo amarillo, borde y texto negros, ciudad abajo.
    pw, ph = 340, 150
    cx, cy = WIDTH // 2, (scene[1] + scene[3]) // 2
    plate_box = (cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2)
    draw.rounded_rectangle(plate_box, radius=14, fill=PLATE_YELLOW, outline=BLACK, width=6)
    draw_centered(draw, cx, cy - 12, format_plate(tx["placa_foto_anpr"]), load_font(74, bold=True), BLACK)
    draw_centered(draw, cx, plate_box[3] - 20, "BOGOTÁ D.C.", load_font(18, bold=True), BLACK)

    # Pie: estación y hora.
    ts = datetime.fromisoformat(tx["fecha_hora"]).strftime("%Y-%m-%d %H:%M:%S")
    info = load_font(18)
    draw.text((24, HEIGHT - 64), tx["estacion"], font=info, fill=WHITE)
    draw.text((24, HEIGHT - 38), ts, font=info, fill=WHITE)
    return img


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tx in ledger["transacciones"]:
        path = OUT_DIR / f"{tx['tx_id']}.png"
        make_plate_image(tx).save(path)
        mismatch = tx["placa_foto_anpr"] != tx["placa_cuenta"]
        note = f"  <- foto {tx['placa_foto_anpr']} != cuenta {tx['placa_cuenta']}" if mismatch else ""
        print(f"{path.relative_to(ROOT)}{note}")


if __name__ == "__main__":
    main()
