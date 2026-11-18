import re
from html import unescape


_DISPLAY_PART = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});|\s+|.", re.DOTALL)


def _display_text(text):
    chars, offsets = [], []
    for part in _DISPLAY_PART.finditer(text):
        raw = part.group()
        value = unescape(raw) if raw.startswith("&") else raw
        if len(value) != 1 and not value.isspace():
            for index in range(part.start(), part.end()):
                chars.append(text[index])
                offsets.append((index, index + 1))
            continue
        if value.isspace():
            if chars and chars[-1] == " ":
                offsets[-1] = (offsets[-1][0], part.end())
                continue
            value = " "
        chars.append(value)
        offsets.append(part.span())
    return "".join(chars), offsets


def verified_quote(quote, source):
    if quote in source:
        return quote
    # Map a display-only match back to the exact stored substring. Never change
    # the submitted source reference or accept edits to the wording.
    text, offsets = _display_text(source)
    needle, _ = _display_text(quote)
    needle = needle.strip()
    start = text.find(needle) if needle else -1
    if start < 0 or text.find(needle, start + 1) >= 0:
        return None
    excerpt = source[offsets[start][0]:offsets[start + len(needle) - 1][1]].strip()
    return excerpt if len(excerpt) <= 500 else None


def verified_evidence(evidence, lookup):
    checked = []
    for item in evidence:
        source = lookup.get((item.type, item.ref))
        quote = verified_quote(item.quote, source["text"]) if source else None
        if quote is None:
            return None
        checked.append(item.model_copy(update={"quote": quote}))
    return checked
