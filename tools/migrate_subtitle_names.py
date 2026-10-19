"""Migrate old-layout subtitles to the PotPlayer naming standard.

Old layout (the translation overwrote the video's default subtitle)::

    movie.srt            ← translated text (the default track was hijacked)
    movie.bilingual.srt  ← bilingual
    movie.jpn.srt        ← copy of the source language

New layout::

    movie.srt                  ← source language, the player's default track
    movie.chs.srt              ← translated only
    movie.bilingual.chs.srt    ← bilingual

Usage::

    python -m tools.migrate_subtitle_names ./output          # preview
    python -m tools.migrate_subtitle_names ./output --apply  # rename

Re-running is safe: files already using the new names are left alone.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.translator.types import (
    PLAYER_LANG_SUFFIX,
    bilingual_player_suffix,
    iso_to_player_suffix,
)

# Player-readable language tags the old layout used for the source copy,
# e.g. {"chs", "jpn", "eng", ...}. ISO 639-1 keys are excluded: a two-letter
# tag belongs to whichever language it names and is resolved per target below.
_LANG_TAGS = frozenset(PLAYER_LANG_SUFFIX.values())
_OLD_BILINGUAL = "bilingual"


@dataclass
class _Group:
    """One subtitle family (``movie.srt`` and everything derived from it)."""

    stem: str
    directory: Path
    default: Path | None = None       # movie.srt
    old_bilingual: Path | None = None  # movie.bilingual.srt
    source_copy: tuple[str, Path] | None = None  # (lang tag, movie.jpn.srt)
    migrated: list[Path] = field(default_factory=list)  # already new-style

    @property
    def is_migrated(self) -> bool:
        return bool(self.migrated)


def _classify(
    srt_files: list[Path], target_lang: str
) -> dict[tuple[Path, str], _Group]:
    """Group subtitle files by the media they belong to, tagging each role."""
    lang = iso_to_player_suffix(target_lang)
    groups: dict[tuple[Path, str], _Group] = {}
    for path in srt_files:
        parts = path.name[: -len(".srt")].split(".")
        if len(parts) == 1:
            # movie.srt — the plain default track, whatever it currently holds.
            stem, role = parts[0], "default"
        else:
            last = parts[-1]
            # movie.bilingual.chs.srt carries two tags; both belong to the media.
            if len(parts) >= 3 and last == lang and parts[-2] == _OLD_BILINGUAL:
                stem = ".".join(parts[:-2])
                role = "migrated"
            elif last == lang:
                stem = ".".join(parts[:-1])
                role = "migrated"
            else:
                stem = ".".join(parts[:-1])
                if last == _OLD_BILINGUAL:
                    role = "old_bilingual"
                elif last in _LANG_TAGS and last != lang:
                    role = "source_copy"
                else:
                    role = "default"
            if role == "default":
                stem = ".".join(parts)

        group = groups.setdefault(
            (path.parent, stem), _Group(stem=stem, directory=path.parent)
        )
        if role == "migrated":
            group.migrated.append(path)
        elif role == "old_bilingual":
            group.old_bilingual = path
        elif role == "source_copy":
            group.source_copy = (last, path)
        else:
            group.default = path
    return groups


def plan_migration(
    root: Path, target_lang: str = "zh"
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return ``(renames, warnings)`` for every subtitle family under *root*."""
    lang_suffix = iso_to_player_suffix(target_lang)
    bilingual_suffix = bilingual_player_suffix(target_lang)
    renames: list[tuple[Path, Path]] = []
    warnings: list[str] = []

    srt_files = sorted(p for p in root.rglob("*.srt") if p.is_file())
    for group in _classify(srt_files, target_lang).values():
        if group.is_migrated:
            continue  # already on the new naming; leave it alone
        if group.default is None:
            continue  # no plain .srt: nothing was hijacked, nothing to fix
        if group.old_bilingual is None and group.source_copy is None:
            # A lone .srt carries no evidence that it is a translation — it is
            # far more likely an untranscribed original. Guessing here would
            # rename every untouched subtitle in the library.
            continue

        translated = group.directory / f"{group.stem}.{lang_suffix}.srt"
        renames.append((group.default, translated))

        if group.old_bilingual is not None:
            renames.append((
                group.old_bilingual,
                group.directory / f"{group.stem}.{bilingual_suffix}.srt",
            ))

        if group.source_copy is not None:
            # The saved copy is the real source subtitle: restore it as default.
            _tag, copy_path = group.source_copy
            renames.append((copy_path, group.directory / f"{group.stem}.srt"))
        else:
            warnings.append(
                f"{group.default.name}: 没有源语言副本，原始字幕无法恢复 —— "
                f"译文已归位为 {translated.name}，但视频暂时没有默认字幕，"
                f"需要重新转写才能补回。"
            )

    # Never overwrite: skip a rename whose destination is already occupied.
    # A destination that is itself being renamed away this run is free.
    freed = {source for source, _dest in renames}
    blocked = {
        dest for _src, dest in renames
        if dest.exists() and dest not in freed
    }
    if blocked:
        renames = [pair for pair in renames if pair[1] not in blocked]
        warnings.extend(
            f"skipped: {dest.name} already exists" for dest in sorted(blocked)
        )
    return renames, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=["./output"],
        help="directories to migrate (searched recursively)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="rename for real; without it the plan is only printed",
    )
    parser.add_argument("--target", default="zh", help="target language (default zh)")
    args = parser.parse_args(argv)

    renames: list[tuple[Path, Path]] = []
    warnings: list[str] = []
    for raw in args.paths:
        root = Path(raw)
        if not root.is_dir():
            print(f"跳过（不是目录）: {root}")
            continue
        found, notes = plan_migration(root, args.target)
        renames.extend(found)
        warnings.extend(notes)

    if not renames:
        print("没有需要迁移的字幕（已是新命名，或目录里没有旧布局的三件套）。")
        for note in warnings:
            print(f"  ! {note}")
        return 0

    verb = "将重命名" if not args.apply else "已重命名"
    print(f"{verb} {len(renames)} 个文件：")
    for source, dest in renames:
        print(f"  {source.relative_to(Path.cwd())} → {dest.name}")

    if not args.apply:
        print("\n这是预览。加 --apply 才会真正改名。")
        return 0

    for source, dest in renames:
        source.replace(dest)
    for note in warnings:
        print(f"  ! {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
