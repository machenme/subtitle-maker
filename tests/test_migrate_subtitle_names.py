from pathlib import Path

from tools.migrate_subtitle_names import main, plan_migration


def _srt(path: Path, text: str = "1\n00:00:00,000 --> 00:00:01,000\ntext\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_old_layout_is_renamed_to_potplayer_names(tmp_path: Path):
    _srt(tmp_path / "movie.srt", "译文")
    _srt(tmp_path / "movie.bilingual.srt", "原文\n译文")
    _srt(tmp_path / "movie.jpn.srt", "原文")

    renames, warnings = plan_migration(tmp_path)

    assert warnings == []
    assert set(renames) == {
        (tmp_path / "movie.srt", tmp_path / "movie.chs.srt"),
        (tmp_path / "movie.bilingual.srt", tmp_path / "movie.bilingual.chs.srt"),
        (tmp_path / "movie.jpn.srt", tmp_path / "movie.srt"),
    }


def test_apply_restores_the_source_as_the_default_subtitle(tmp_path: Path):
    _srt(tmp_path / "movie.srt", "译文")
    _srt(tmp_path / "movie.bilingual.srt", "原文\n译文")
    _srt(tmp_path / "movie.jpn.srt", "原文")

    assert main([str(tmp_path), "--apply"]) == 0

    assert not (tmp_path / "movie.bilingual.srt").exists()
    assert (tmp_path / "movie.srt").read_text(encoding="utf-8") == "原文"
    assert (tmp_path / "movie.chs.srt").read_text(encoding="utf-8") == "译文"
    assert (tmp_path / "movie.bilingual.chs.srt").read_text(encoding="utf-8") == (
        "原文\n译文"
    )


def test_rerunning_after_migration_changes_nothing(tmp_path: Path):
    _srt(tmp_path / "movie.srt", "原文")
    _srt(tmp_path / "movie.chs.srt", "译文")
    _srt(tmp_path / "movie.bilingual.chs.srt", "原文\n译文")

    renames, _warnings = plan_migration(tmp_path)

    assert renames == []


def test_missing_source_copy_warns_instead_of_inventing_one(tmp_path: Path):
    """No .jpn.srt: the original is gone, so say so instead of guessing."""
    _srt(tmp_path / "movie.srt", "译文")
    _srt(tmp_path / "movie.bilingual.srt", "原文\n译文")

    renames, warnings = plan_migration(tmp_path)

    assert set(renames) == {
        (tmp_path / "movie.srt", tmp_path / "movie.chs.srt"),
        (tmp_path / "movie.bilingual.srt", tmp_path / "movie.bilingual.chs.srt"),
    }
    assert len(warnings) == 1 and "movie.srt" in warnings[0]


def test_untranslated_subtitles_are_left_alone(tmp_path: Path):
    """A lone .srt is usually an untranscribed original — never rename it."""
    _srt(tmp_path / "movie.srt", "原文")

    renames, warnings = plan_migration(tmp_path)

    assert renames == []
    assert warnings == []


def test_preview_does_not_touch_the_disk(tmp_path: Path):
    _srt(tmp_path / "movie.srt", "译文")
    _srt(tmp_path / "movie.jpn.srt", "原文")

    assert main([str(tmp_path)]) == 0

    assert (tmp_path / "movie.srt").read_text(encoding="utf-8") == "译文"
    assert (tmp_path / "movie.jpn.srt").exists()
