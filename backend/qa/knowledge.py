"""Convert Markdown instructions to the existing versioned policy fragments."""
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from .contracts import KnowledgeInput


ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|$)")
SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(\s*(?:<([^>\n]+)>|([^\s)]+))(?:\s+\"[^\"]*\")?\s*\)")
SEARCH_STOP_WORDS = {
    "the", "and", "for", "with", "after", "before", "from", "this", "that", "your",
    "please", "can", "will", "support", "what", "how", "you", "are", "was", "about",
}


def markdown_fragments(markdown, *, include_code=True):
    lines = markdown.removeprefix("\ufeff").splitlines()
    headings = []
    body = []
    fragments = {}

    def flush():
        content = "\n".join(body).strip()
        if not content:
            return
        section = " › ".join(title for _, title in headings)
        fragment = f"{section}\n\n{content}" if section else content
        if len(fragment) > 5000:
            raise ValueError(f"Section ‘{section or 'Introduction'}’ exceeds 5,000 characters. Add subheadings.")
        if len(fragments) == 30:
            raise ValueError("An instruction version can contain up to 30 sections with text.")
        fragments[f"section-{len(fragments) + 1:02}"] = fragment

    fence = None
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        delimiter = FENCE.match(line)
        if fence:
            if include_code:
                body.append(line)
            if (delimiter and delimiter[1][0] == fence[0] and len(delimiter[1]) >= len(fence)
                    and not delimiter[2].strip()):
                fence = None
            continue
        if delimiter and (delimiter[1][0] == "~" or "`" not in delimiter[2]):
            fence = delimiter[1]
            if include_code:
                body.append(line)
            continue
        heading = ATX.fullmatch(line)
        underline = SETEXT.fullmatch(lines[index]) if index < len(lines) else None
        if heading:
            level = len(heading[1])
            title = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", heading[2] or "").strip()
        elif underline and line.strip() and not line.startswith(("    ", "\t")):
            level = 1 if underline[1][0] == "=" else 2
            title = line.strip()
            index += 1
        else:
            body.append(line)
            continue
        flush()
        body.clear()
        while headings and headings[-1][0] >= level:
            headings.pop()
        if title:
            headings.append((level, title))

    flush()
    if not fragments:
        raise ValueError("Add rule text below a heading or in the document body.")
    return fragments


def bundle_path(value):
    url = urlsplit(value)
    path = PurePosixPath(unquote(url.path))
    if (url.scheme or url.netloc or url.query or url.fragment or path.is_absolute()
            or ".." in path.parts or "\\" in value or "\\" in str(path)
            or path.suffix.lower() != ".md" or path.as_posix().lower() == "index.md"):
        raise ValueError(f"Use a relative path to a Markdown instruction: {value}")
    return path.as_posix()


def parse_bundle(bundle):
    documents = {}
    seen_versions = set()
    for entry in bundle.documents:
        path = bundle_path(entry.path)
        identity = (entry.document, entry.version)
        if path in documents or identity in seen_versions:
            raise ValueError(f"Duplicate path or instruction version: {entry.path}")
        seen_versions.add(identity)
        try:
            documents[path] = KnowledgeInput(**entry.model_dump(exclude={"markdown", "path"}),
                                             fragments=markdown_fragments(entry.markdown))
        except ValueError as exc:
            raise ValueError(f"{entry.path}: {exc}") from exc

    sections = []
    linked = set()
    for fragment in markdown_fragments(bundle.index, include_code=False).values():
        links = list(LINK.finditer(fragment))
        if not links:
            continue
        paths = {bundle_path(link[2] or link[3]) for link in links}
        missing = paths - documents.keys()
        if missing:
            raise ValueError("The index links to missing instructions: " + ", ".join(sorted(missing)))
        linked.update(paths)
        sections.append({"text": LINK.sub(lambda link: link[1], fragment),
                         "documents": sorted({documents[path].document for path in paths}),
                         "paths": sorted(paths)})
    unlinked = documents.keys() - linked
    if unlinked:
        raise ValueError("Link all bundle instructions from index.md: " + ", ".join(sorted(unlinked)))
    return documents, sections


def policy_terms(text):
    return set(re.findall(r"\w{3,}", text.casefold())) - SEARCH_STOP_WORDS
