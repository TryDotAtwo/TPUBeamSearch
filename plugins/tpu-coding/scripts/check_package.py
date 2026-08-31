#!/usr/bin/env python3
"""Narrow, offline integrity checks for the TPU Coding plugin package."""
import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


REQUIRED = (
    ".codex-plugin/plugin.json",
    "skills/code-tpu/SKILL.md",
    "skills/code-tpu/references/evidence.json",
)
STATUSES = {"documented", "measured", "hypothesis", "superseded"}
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]*)\)")
INVALID_JSON = object()


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def error(errors, text):
    errors.append(text)


def load_json(path, errors, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        error(errors, f"{label}: invalid JSON ({exc})")
        return INVALID_JSON


def check_manifest(root, errors):
    path = root / ".codex-plugin" / "plugin.json"
    data = load_json(path, errors, "plugin manifest")
    if data is INVALID_JSON:
        return
    if not isinstance(data, dict):
        error(errors, "plugin manifest: JSON root must be an object")
        return
    for field in ("name", "version", "description"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            error(errors, f"plugin manifest: {field} must be a nonempty string")
    skills = data.get("skills")
    skills = [skills] if isinstance(skills, str) else skills
    if not isinstance(skills, list) or not skills:
        error(errors, "plugin manifest: skills must be a nonempty list of paths")
        return
    for item in skills:
        if not isinstance(item, str) or not item.strip():
            error(errors, "plugin manifest: skills contains an invalid path")
            continue
        target = (root / item).resolve()
        if not inside(target, root):
            error(errors, f"plugin manifest: skills path escapes package: {item}")
        elif not target.is_dir():
            error(errors, f"plugin manifest: skills path is missing: {item}")


def link_target(raw):
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1:raw.index(">")]
    return raw.split(None, 1)[0] if raw else ""


def is_https_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


def check_links(root, errors):
    for markdown in root.rglob("*.md"):
        try:
            text = markdown.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            error(errors, f"{markdown.relative_to(root)}: cannot read Markdown ({exc})")
            continue
        for match in LINK.finditer(text):
            target = link_target(match.group(1))
            if not target or target.startswith("#") or target.startswith(("https://", "http://", "mailto:", "codex:")):
                continue
            target = target.split("#", 1)[0]
            candidate = (markdown.parent / target).resolve()
            shown = markdown.relative_to(root)
            if not inside(candidate, root):
                error(errors, f"{shown}: local link escapes package: {target}")
            elif not candidate.exists():
                error(errors, f"{shown}: missing local link: {target}")


def check_evidence(root, errors):
    data = load_json(root / "skills/code-tpu/references/evidence.json", errors, "evidence")
    if data is INVALID_JSON:
        return
    if not isinstance(data, dict):
        error(errors, "evidence: JSON root must be an object")
        return
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        error(errors, "evidence: schema_version must be 1")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        error(errors, "evidence: entries must be a nonempty list")
        return
    ids = set()
    fields = ("claim", "scope", "recheck_when", "regression_case")
    for number, entry in enumerate(entries, 1):
        label = f"evidence entry {number}"
        if not isinstance(entry, dict):
            error(errors, f"{label}: must be an object")
            continue
        ident = entry.get("id")
        if not isinstance(ident, str) or not ident.strip():
            error(errors, f"{label}: id must be a nonempty string")
        elif ident in ids:
            error(errors, f"{label}: duplicate evidence id: {ident}")
        else:
            ids.add(ident)
        if not isinstance(entry.get("status"), str) or entry["status"] not in STATUSES:
            error(errors, f"{label}: status must be documented, measured, hypothesis, or superseded")
        for field in fields:
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                error(errors, f"{label}: {field} must be a nonempty string")
        try:
            date.fromisoformat(entry.get("checked_on", ""))
        except (TypeError, ValueError):
            error(errors, f"{label}: checked_on must be an ISO date")
        urls = entry.get("source_urls")
        if not isinstance(urls, list) or not urls or any(not is_https_url(url) for url in urls):
            error(errors, f"{label}: source_urls must be a nonempty list of https URLs")


def check_package(root: Path) -> list[str]:
    """Return integrity errors only; this does not assess scientific claims."""
    root = Path(root).resolve()
    errors = []
    if not root.is_dir():
        return [f"package root is not a directory: {root}"]
    missing = [item for item in REQUIRED if not (root / item).is_file()]
    for item in missing:
        error(errors, f"missing required file: {item}")
    if not (root / REQUIRED[0]).is_file():
        return errors
    check_manifest(root, errors)
    check_links(root, errors)
    if (root / REQUIRED[2]).is_file():
        check_evidence(root, errors)
    return errors


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python scripts/check_package.py <plugin-root>", file=sys.stderr)
        return 1
    errors = check_package(Path(args[0]))
    for item in errors:
        print(f"ERROR: {item}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
