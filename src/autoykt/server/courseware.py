"""Export account-visible slide JSON, original images and prepared answers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from autoykt.server.cache import ClassroomCache
from autoykt.server.solver import AssetStore


def write_json(path: Path, data):
    """Keep exported classroom material private and replace files atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        if os.name != "nt":
            os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def export_courseware(
    cache: ClassroomCache, assets: AssetStore, course: str, destination: Path
) -> dict:
    """Export cached pages without connecting or implying future pages exist."""
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "schema_version": 1,
        "domain": cache.domain,
        "course_id": course,
        "scope": "all_returned_slides_not_editable_pptx",
        "presentations": [],
    }
    extensions = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    for presentation in cache.presentation_ids(course):
        folder = hashlib.sha256(presentation.encode()).hexdigest()[:20]
        deck_dir = destination / folder
        deck_dir.mkdir(exist_ok=True, mode=0o700)
        document = cache.presentation(course, presentation)
        write_json(deck_dir / "presentation.json", document)
        questions = []
        for question in cache.questions(course, presentation):
            cached = cache.answer(question)
            questions.append(
                {
                    "question_id": question.question_id,
                    "slide_id": question.slide_id,
                    "index": question.index,
                    "kind": question.kind,
                    "text": question.text,
                    "options": question.options,
                    "image_url": question.image_url,
                    "signature": question.signature,
                    "prepared_answer": cached[0] if cached else None,
                    "answer_source": cached[1] if cached else None,
                }
            )
        write_json(deck_dir / "questions.json", questions)
        write_json(
            deck_dir / "answers.json",
            cache.export_answers(course, presentation),
        )
        images = []
        for position, slide in enumerate(document.get("slides") or [], 1):
            if not isinstance(slide, dict):
                continue
            url = slide.get("cover_url") or ""
            saved = assets.cached_image(url)
            relative = None
            if saved:
                data, mime = saved
                relative = f"slide-{position:04d}.{extensions[mime]}"
                image = deck_dir / relative
                image.write_bytes(data)
                if os.name != "nt":
                    image.chmod(0o600)
            images.append({"position": position, "url": url, "file": relative})
        manifest["presentations"].append(
            {
                "id": presentation,
                "title": document.get("title") or document.get("Title") or "",
                "directory": folder,
                "slide_count": len(document.get("slides") or []),
                "question_count": len(questions),
                "images": images,
            }
        )
    write_json(destination / "manifest.json", manifest)
    return manifest
