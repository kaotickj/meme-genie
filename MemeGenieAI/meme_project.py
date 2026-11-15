"""Portable editable meme projects. No scripts or external background paths."""
from dataclasses import dataclass, asdict
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
import zlib
from PIL import Image, ImageDraw, ImageFont

MAX_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 25_000_000

@dataclass
class TextLayer:
    text: str = ""
    x: float = .05
    y: float = .4
    width: float = .9
    size: int = 60
    font: str = "DejaVuSans-Bold.ttf"
    align: str = "center"
    color: tuple = (255, 255, 255, 255)
    shadow: bool = True

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid text-layer fields")
        if not isinstance(data['text'], str) or len(data['text']) > 20000:
            raise ValueError("Invalid caption text")
        for key in ('x', 'y', 'width'):
            value = data[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid text position")
        if data['width'] < .05 or data['x'] + data['width'] > 1.000001:
            raise ValueError("Text box extends beyond the canvas")
        if type(data['size']) is not int or not 10 <= data['size'] <= 250:
            raise ValueError("Invalid font size")
        if not isinstance(data['font'], str) or len(data['font']) > 255 or '/' in data['font'] or '\\' in data['font']:
            raise ValueError("Invalid font name")
        if data['align'] not in ('left', 'center', 'right') or type(data['shadow']) is not bool:
            raise ValueError("Invalid text style")
        color = data['color']
        if not isinstance(color, (list, tuple)) or len(color) != 4 or any(type(v) is not int or not 0 <= v <= 255 for v in color):
            raise ValueError("Invalid text color")
        return cls(**{**data, 'color': tuple(color)})


def validate_image(image):
    if max(image.size) > 16384 or image.width * image.height > MAX_PIXELS:
        raise ValueError("Project background exceeds the dimension or 25-million-pixel limit")


def save_project(path, background, layers, attribution=False):
    validate_image(background)
    if not 1 <= len(layers) <= 20:
        raise ValueError("Projects support 1 to 20 text layers")
    records = [asdict(TextLayer.parse(asdict(layer))) for layer in layers]
    buf = BytesIO(); background.save(buf, format='PNG')
    if len(buf.getvalue()) > MAX_BYTES:
        raise ValueError("Project background is too large")
    manifest = {'version': 1, 'layers': records, 'attribution': True}
    target = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.meme-project-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('project.json', json.dumps(manifest))
                archive.writestr('background.png', buf.getvalue())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_project(path):
    try:
        if Path(path).stat().st_size > MAX_BYTES + 1024 * 1024:
            raise ValueError("Project file is too large")
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ['background.png', 'project.json']:
                raise ValueError("Invalid project contents")
            if archive.getinfo('project.json').file_size > 1024 * 1024 or archive.getinfo('background.png').file_size > MAX_BYTES:
                raise ValueError("Project contents are too large")
            if any(info.flag_bits & 1 for info in archive.infolist()):
                raise ValueError('Encrypted project files are not supported')
            manifest = json.loads(archive.read('project.json'))
            if not isinstance(manifest, dict) or manifest.get('version') != 1:
                raise ValueError("Unsupported project version")
            records = manifest.get('layers')
            if not isinstance(records, list) or not 1 <= len(records) <= 20:
                raise ValueError("Invalid text-layer count")
            layers = [TextLayer.parse(record) for record in records]
            if type(manifest.get('attribution')) is not bool:
                raise ValueError("Invalid attribution setting")
            with Image.open(BytesIO(archive.read('background.png'))) as im:
                validate_image(im); im.load(); background = im.convert('RGBA')
        return background, layers, manifest['attribution']
    except (zipfile.BadZipFile, KeyError, TypeError, RuntimeError, NotImplementedError, zlib.error, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid meme project") from exc


def wrap_text_lines(draw, text, font, width):
    """Shared wrapping for the renderer and AI's measured caption fitting."""
    lines=[]
    for paragraph in text.split('\n'):
        line=''
        for word in paragraph.split():
            candidate=(line+' '+word).strip()
            if line and draw.textlength(candidate,font=font)>width:
                lines.append(line);line=''
            if draw.textlength(word,font=font)>width:
                for char in word:
                    if line and draw.textlength(line+char,font=font)>width:
                        lines.append(line);line=''
                    line+=char
            else:
                line=(line+' '+word).strip()
        lines.append(line)
    return lines


def render_layers(background, layers, font_map=None):
    result = background.convert('RGBA').copy()
    draw = ImageDraw.Draw(result)
    font_map = font_map or {}
    for layer in layers:
        layer = TextLayer.parse(asdict(layer))
        if not layer.text.strip(): continue
        left = round(layer.x * result.width)
        top = round(layer.y * result.height)
        width = max(1, round(layer.width * result.width))
        size = max(10, round(layer.size * result.width / 512))
        def get_font(px):
            try: return ImageFont.truetype(font_map.get(layer.font, layer.font), px)
            except OSError:
                try: return ImageFont.truetype('DejaVuSans-Bold.ttf', px)
                except OSError: return ImageFont.load_default(size=px)
        # Fit the caption into its remaining canvas area. Break long tokens safely.
        while True:
            font = get_font(size)
            lines = wrap_text_lines(draw,layer.text,font,width)
            line_height = max(1, int(size * 1.25))
            if size <= 10 or line_height * len(lines) <= result.height - top: break
            size -= 1
        # Draw on a bounded layer so captions cannot bleed into adjacent boxes.
        overlay = Image.new('RGBA', (width, max(1, result.height-top)))
        painter = ImageDraw.Draw(overlay)
        for row, line in enumerate(lines):
            span = painter.textlength(line, font=font)
            x = 0 if layer.align == 'left' else (width-span if layer.align == 'right' else (width-span)/2)
            y = row * line_height
            shadow = (0,0,0,255) if sum(layer.color[:3]) >= 384 else (255,255,255,255)
            if layer.shadow: painter.text((x+2,y+2),line,font=font,fill=shadow)
            painter.text((x,y),line,font=font,fill=layer.color)
        result.alpha_composite(overlay, (left, top))
    return result
