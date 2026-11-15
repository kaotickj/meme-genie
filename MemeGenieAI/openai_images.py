"""Opt-in, text-only OpenAI background generation; never uploads editor images."""
from __future__ import annotations

import base64
import binascii
import io
import json
import time

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from openai_layout import api_key
from meme_project import validate_image

MODEL = 'gpt-image-2.6-flare'
ENDPOINT = 'https://api.openai.com/v1/images/generations'
MAX_RESPONSE = 32 * 1024 * 1024


def native_size(dimensions):
    if (not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2 or
            any(type(n) is not int or not 1 <= n <= 4096 for n in dimensions)):
        raise ValueError('Choose a valid image resolution')
    w, h = dimensions
    return '1024x1024' if w == h else ('1536x1024' if w > h else '1024x1536')


def generation_body(prompt, negative_prompt, dimensions):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Enter an image prompt or choose a preset')
    if not isinstance(negative_prompt, str):
        raise ValueError('Negative prompt must be text')
    combined = prompt.strip()
    if negative_prompt.strip():
        combined += '\n\nAvoid these elements where possible: ' + negative_prompt.strip()
    if len(combined) > 32000:
        raise ValueError('Keep the combined image prompt below 32,000 characters')
    return {'model': MODEL, 'prompt': combined, 'n': 1, 'size': native_size(dimensions),
            'quality': 'medium', 'complexity': 'medium', 'output_format': 'png',
            'background': 'opaque'}


def generate_image(prompt, negative_prompt, dimensions, *, canceled=lambda: False):
    body = generation_body(prompt, negative_prompt, dimensions)
    key = api_key()
    if canceled():
        raise ValueError('OpenAI generation canceled before sending')
    deadline = time.monotonic() + 180
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.post(ENDPOINT, headers={'Authorization': 'Bearer ' + key}, json=body,
                              timeout=(5, 180), allow_redirects=False, stream=True) as response:
                if response.status_code != 200:
                    reason = {400: 'request was rejected (check prompt and supported options)',
                              401: 'key was rejected',
                              403: 'access was denied (check model access or organization verification)',
                              404: 'image model is unavailable for this account',
                              429: 'rate limit or API quota was reached'}.get(response.status_code, 'image request failed')
                    raise ValueError(f'OpenAI {reason} (HTTP {response.status_code}); no automatic retry was made')
                chunks = bytearray()
                for chunk in response.iter_content(65536):
                    if canceled():
                        raise ValueError('OpenAI result discarded; the request may still be charged')
                    if time.monotonic() > deadline:
                        raise requests.Timeout()
                    chunks.extend(chunk)
                    if len(chunks) > MAX_RESPONSE:
                        raise ValueError('OpenAI image response exceeded the size limit')
                payload = json.loads(chunks)
    except requests.Timeout:
        raise ValueError('OpenAI image request timed out; it may still be charged. No automatic retry was made') from None
    except requests.RequestException:
        raise ValueError('OpenAI connection failed; the request may still be charged. No automatic retry was made') from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError('OpenAI returned invalid image response JSON') from None
    if canceled():
        raise ValueError('OpenAI result discarded; the request may still be charged')
    try:
        if not isinstance(payload, dict) or payload.get('error'):
            raise ValueError('OpenAI returned an image generation error')
        records = payload['data']
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError('OpenAI did not return exactly one image')
        encoded = records[0]['b64_json']
        if not isinstance(encoded, str) or not encoded:
            raise ValueError('OpenAI returned no image data')
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'PNG':
                raise ValueError('OpenAI returned an unexpected image format')
            validate_image(image)
            if image.size != tuple(map(int, body['size'].split('x'))):
                raise ValueError('OpenAI returned unexpected image dimensions')
            image.load()
            # Preserve the existing requested canvas sizes without stretching.
            return ImageOps.fit(image.convert('RGBA'), tuple(dimensions), Image.Resampling.LANCZOS)
    except (KeyError, TypeError, binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError('OpenAI returned invalid image data') from None
