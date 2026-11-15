"""Opt-in Luna caption/layout generation. No credentials enter saved app state."""
from __future__ import annotations

import base64
import copy
import io
import json
import os
from pathlib import Path
import stat
import time

import requests

MODEL = 'gpt-6-luna'
ENDPOINT = 'https://api.openai.com/v1/responses'


def api_key():
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        path = Path(os.environ.get('MEMEGENIE_OPENAI_KEY_FILE') or
                    Path.home() / '.memegenie-ai' / 'openai.key').expanduser()
        try:
            with path.open('r', encoding='utf-8') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or (os.name == 'posix' and
                        (info.st_mode & 0o077 or info.st_uid != os.getuid())):
                    raise ValueError('OpenAI key file must be owned by you and private (chmod 600)')
                key = handle.read(4097).strip()
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError):
            raise ValueError('Cannot read the private OpenAI key file') from None
    if not key:
        raise ValueError('Set OPENAI_API_KEY or create ~/.memegenie-ai/openai.key (private, chmod 600) to use OpenAI')
    if len(key) > 4096 or any(c.isspace() for c in key) or not key.isascii():
        raise ValueError('OpenAI key configuration is invalid')
    return key


def request_layout(image, prompt, schema, timeout):
    key = api_key()  # Fail before image encoding or network access.
    image = image.convert('RGB').copy()
    image.thumbnail((768, 768))
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=85)
    body = {
        'model': MODEL, 'store': False, 'max_output_tokens': 4000,
        'reasoning': {'effort': 'low'},
        'input': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': prompt},
            {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' +
             base64.b64encode(buf.getvalue()).decode('ascii')},
        ]}],
        'text': {'format': {'type': 'json_schema', 'name': 'meme_layout',
                            'strict': True, 'schema': schema}},
    }
    deadline = time.monotonic() + min(timeout, 120)
    try:
        with requests.Session() as session:
            # Fixed HTTPS endpoint; no environment proxies, netrc, redirects or retries.
            session.trust_env = False
            with session.post(ENDPOINT, headers={'Authorization': 'Bearer ' + key},
                              json=body, timeout=(5, min(timeout, 120)),
                              allow_redirects=False, stream=True) as response:
                status = response.status_code
                if status != 200:
                    reason = {401: 'key was rejected', 403: 'access was denied',
                              404: 'model is unavailable for this account',
                              429: 'rate limit or API quota was reached'}.get(status, 'request failed')
                    raise ValueError(f'OpenAI {reason} (HTTP {status}); no automatic retry was made')
                chunks = bytearray()
                for chunk in response.iter_content(16384):
                    if time.monotonic() > deadline:
                        raise requests.Timeout()
                    chunks.extend(chunk)
                    if len(chunks) > 1_000_000:
                        raise ValueError('OpenAI response exceeded the size limit')
                payload = json.loads(chunks)
    except requests.Timeout:
        raise ValueError('OpenAI request timed out; no automatic retry was made') from None
    except requests.RequestException:
        raise ValueError('Cannot connect to OpenAI; check your connection. No automatic retry was made') from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError('OpenAI returned invalid JSON') from None
    try:
        if payload.get('status') != 'completed' or payload.get('error'):
            raise ValueError('OpenAI did not complete the suggestion; try again explicitly')
        texts = []
        for item in payload['output']:
            if item.get('type') != 'message':
                continue
            for part in item['content']:
                if part.get('type') == 'refusal':
                    raise ValueError('OpenAI declined this suggestion request')
                if part.get('type') == 'output_text':
                    texts.append(part['text'])
        if len(texts) != 1:
            raise ValueError('OpenAI returned no usable layout')
        return json.loads(texts[0])
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
        raise ValueError('OpenAI returned an invalid layout response') from None


def suggest_layout(background, instruction, timeout=300, *, alternatives=False):
    from ai_meme_layout import (
        SCHEMA, SCENE_OPTIONS_SCHEMA, build_caption_prompt, requested_exact_captions,
        preserve_exact_captions, validate_options, validate_suggestion, validate_regions,
        place_around_subjects, NoCaptionSpace,
    )
    if not isinstance(instruction, str) or len(instruction) > 2000:
        raise ValueError('Keep the caption request below 2,000 characters')
    exact = requested_exact_captions(instruction)
    prompt = build_caption_prompt(instruction)
    if alternatives:
        prompt += (' Return two different finished joke and layout alternatives. Identify up to four '
                   'protected_regions as tight boxes around heads (including hats) and key objects, '
                   'not whole people or the whole scene. Use fractional top-left x,y,width,height. '
                   'Vary font and composition, and avoid these regions. Keep captions concise and complete.')
    data = request_layout(background, prompt, copy.deepcopy(SCENE_OPTIONS_SCHEMA if alternatives else SCHEMA), timeout)
    if not alternatives:
        description, layers = validate_suggestion(data)
        return description, preserve_exact_captions([{'layers': layers}], exact)[0]['layers']
    if not isinstance(data, dict) or 'protected_regions' not in data:
        raise ValueError('AI did not identify protected subject regions')
    regions = validate_regions(data.pop('protected_regions'))
    description, options = validate_options(data)
    options = preserve_exact_captions(options, exact)
    usable = []
    for option in options:
        try:
            layers = place_around_subjects(background, option['layers'], regions)
        except NoCaptionSpace:
            continue
        usable.append({**option, 'layers': layers, 'subject_checked': True,
                       'protected_regions': copy.deepcopy(regions)})
    if not usable:
        usable = [{**option, 'protected_regions': copy.deepcopy(regions),
                   'placement_error': 'Review the subject areas: no readable caption space remains.'}
                  for option in options]
    return description, usable
