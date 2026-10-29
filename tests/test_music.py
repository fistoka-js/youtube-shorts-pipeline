"""Tests for verticals/music.py — track selection by mood tags, duck filter generation.

No test needs real mp3 contents: tracks are empty files in a temp folder, and
the speech-region step (Whisper) is replaced with a fixed value.
"""

import random
from pathlib import Path

import pytest
import yaml

from verticals import music
from verticals.music import build_duck_filter, select_and_prepare_music
from verticals.niche import get_music_config, load_niche

REAL_MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"


class TestBuildDuckFilter:
    def test_empty_regions(self):
        result = build_duck_filter([])
        assert result == "volume=0.25"

    def test_single_region(self):
        result = build_duck_filter([(1.0, 3.0)])
        assert "volume=" in result
        assert "between(t," in result
        assert "0.12" in result
        assert "0.25" in result

    def test_multiple_regions(self, sample_speech_regions):
        result = build_duck_filter(sample_speech_regions)
        assert result.count("between(t,") == 2
        assert "0.12" in result
        assert "0.25" in result
        assert "eval=frame" in result

    def test_buffer_applied(self):
        result = build_duck_filter([(1.0, 2.0)], buffer=0.5)
        # Start should be max(0, 1.0-0.5) = 0.5
        # End should be 2.0+0.5 = 2.5
        assert "0.50" in result
        assert "2.50" in result

    def test_buffer_no_negative_start(self):
        result = build_duck_filter([(0.1, 1.0)], buffer=0.3)
        # Start should be max(0, 0.1-0.3) = 0.0
        assert "0.00" in result

    def test_returns_ffmpeg_filter_format(self, sample_speech_regions):
        result = build_duck_filter(sample_speech_regions)
        # Should be a valid ffmpeg volume filter
        assert result.startswith("volume=")
        assert ":eval=frame" in result

    def test_if_expression_structure(self):
        result = build_duck_filter([(5.0, 10.0)])
        assert "if(" in result
        assert ", 0.12, 0.25)" in result


@pytest.fixture
def fake_music(tmp_path, monkeypatch):
    """Point MUSIC_DIR at a temp folder and skip the Whisper step.

    Call make(tags_yaml, files) to create the tracks. files is a list of
    file names to create as empty mp3s; tags_yaml is the text of tracks.yaml
    (or None to leave the file out).
    """
    folder = tmp_path / "music"
    folder.mkdir()
    monkeypatch.setattr(music, "MUSIC_DIR", folder)
    monkeypatch.setattr(music, "_get_speech_regions", lambda *a, **k: [(0.0, 1.0)])

    def make(tags_yaml, files):
        for name in files:
            (folder / name).write_bytes(b"")
        if tags_yaml is not None:
            (folder / "tracks.yaml").write_text(tags_yaml, encoding="utf-8")
        return folder

    return make


def _pick(keywords, tmp_path):
    result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=keywords)
    return Path(result["track_path"]).name if result else None


def _pick_mood(keywords, video_mood, tmp_path):
    result = select_and_prepare_music(
        tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=keywords, video_mood=video_mood,
    )
    return Path(result["track_path"]).name if result else None


class TestSelectByTags:
    def test_top_score_wins(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [cosmic, space]\n"
            "b.mp3: [cosmic]\n"
            "c.mp3: [dark]\n",
            ["a.mp3", "b.mp3", "c.mp3"],
        )
        assert _pick(["cosmic", "space"], tmp_path) == "a.mp3"

    def test_tie_is_random_only_among_top_scorers(self, fake_music, tmp_path, monkeypatch):
        fake_music(
            "a.mp3: [cosmic, space]\n"
            "b.mp3: [space, cosmic]\n"
            "c.mp3: [cosmic]\n"
            "d.mp3: [ambient]\n",
            ["a.mp3", "b.mp3", "c.mp3", "d.mp3"],
        )
        picks = set()
        for seed in range(60):
            monkeypatch.setattr(music, "random", random.Random(seed))
            picks.add(_pick(["cosmic", "space"], tmp_path))
        # Both top scorers (score 2) are picked at some point, and nothing else is
        assert picks == {"a.mp3", "b.mp3"}

    def test_no_overlap_falls_back_to_ambient(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [ambient, calm]\n"
            "c.mp3: [quirky]\n",
            ["a.mp3", "b.mp3", "c.mp3"],
        )
        assert _pick(["cosmic"], tmp_path) == "b.mp3"

    def test_ambient_fallback_is_random_only_among_ambient(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [ambient]\n"
            "c.mp3: [ambient]\n",
            ["a.mp3", "b.mp3", "c.mp3"],
        )
        picks = {_pick(["cosmic"], tmp_path) for _ in range(40)}
        assert picks <= {"b.mp3", "c.mp3"}
        assert "a.mp3" not in picks

    def test_no_keywords_uses_ambient(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [ambient]\n",
            ["a.mp3", "b.mp3"],
        )
        assert _pick(None, tmp_path) == "b.mp3"
        assert _pick([], tmp_path) == "b.mp3"

    def test_no_ambient_track_returns_no_music(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [quirky]\n",
            ["a.mp3", "b.mp3"],
        )
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"]) == {}

    def test_missing_tracks_yaml_returns_no_music(self, fake_music, tmp_path):
        fake_music(None, ["a.mp3", "ambient.mp3"])
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"]) == {}

    def test_malformed_tracks_yaml_returns_no_music(self, fake_music, tmp_path):
        fake_music("- just\n- a list\n", ["a.mp3"])
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"]) == {}

    def test_listed_file_not_on_disk_is_ignored(self, fake_music, tmp_path):
        fake_music(
            "ghost.mp3: [cosmic]\n"
            "b.mp3: [ambient]\n",
            ["b.mp3"],
        )
        # ghost.mp3 matches the keyword but is not on disk, so the ambient track is picked
        assert _pick(["cosmic"], tmp_path) == "b.mp3"

    def test_mp3_not_in_yaml_is_never_picked(self, fake_music, tmp_path):
        fake_music(
            "b.mp3: [ambient]\n",
            ["a.mp3", "b.mp3"],
        )
        picks = {_pick(["cosmic"], tmp_path) for _ in range(30)}
        assert picks == {"b.mp3"}

    def test_keyword_match_is_case_insensitive(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [Cosmic]\n"
            "b.mp3: [ambient]\n",
            ["a.mp3", "b.mp3"],
        )
        assert _pick(["COSMIC"], tmp_path) == "a.mp3"

    def test_returns_track_path_and_duck_filter(self, fake_music, tmp_path):
        folder = fake_music("a.mp3: [ambient]\n", ["a.mp3"])
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"])
        assert set(result) == {"track_path", "duck_filter"}
        assert Path(result["track_path"]) == folder / "a.mp3"
        assert result["duck_filter"].startswith("volume=")


class TestVideoMoodFake:
    def test_mood_tag_outranks_one_keyword_overlap(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [cosmic, space]\n"
            "b.mp3: [hopeful]\n",
            ["a.mp3", "b.mp3"],
        )
        # a.mp3 has 2 keyword overlaps (2 points), b.mp3 has the mood tag (10 points)
        assert _pick_mood(["cosmic", "space"], "hopeful", tmp_path) == "b.mp3"

    def test_mood_without_keywords_still_picks_mood_track(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [ambient]\n"
            "c.mp3: [hopeful]\n",
            ["a.mp3", "b.mp3", "c.mp3"],
        )
        assert _pick_mood(None, "hopeful", tmp_path) == "c.mp3"

    def test_mood_with_no_tagged_track_falls_back_to_ambient(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [ambient]\n",
            ["a.mp3", "b.mp3"],
        )
        assert _pick_mood(["cosmic"], "quirky", tmp_path) == "b.mp3"

    def test_invalid_mood_gives_same_pick_as_none(self, fake_music, tmp_path, monkeypatch):
        fake_music(
            "a.mp3: [cosmic, space]\n"
            "b.mp3: [cosmic]\n"
            "c.mp3: [hopeful]\n"
            "d.mp3: [ambient]\n",
            ["a.mp3", "b.mp3", "c.mp3", "d.mp3"],
        )
        for seed in range(30):
            monkeypatch.setattr(music, "random", random.Random(seed))
            without = _pick_mood(["cosmic", "space"], None, tmp_path)
            monkeypatch.setattr(music, "random", random.Random(seed))
            invalid = _pick_mood(["cosmic", "space"], "banana", tmp_path)
            assert invalid == without

    def test_mood_is_normalised(self, fake_music, tmp_path):
        fake_music(
            "a.mp3: [dark]\n"
            "b.mp3: [ambient]\n"
            "c.mp3: [hopeful]\n",
            ["a.mp3", "b.mp3", "c.mp3"],
        )
        assert _pick_mood(None, "  Hopeful ", tmp_path) == "c.mp3"

    def test_chosen_mood_is_logged(self, fake_music, tmp_path, monkeypatch):
        fake_music("a.mp3: [hopeful]\n", ["a.mp3"])
        messages = []
        monkeypatch.setattr(music, "log", messages.append)
        _pick_mood(["cosmic"], "hopeful", tmp_path)
        assert any("hopeful" in m and "Video mood" in m for m in messages)
        assert any("video mood hopeful" in m for m in messages)

    def test_invalid_mood_is_logged_as_ignored(self, fake_music, tmp_path, monkeypatch):
        fake_music("a.mp3: [ambient]\n", ["a.mp3"])
        messages = []
        monkeypatch.setattr(music, "log", messages.append)
        _pick_mood(["cosmic"], "banana", tmp_path)
        assert any("banana" in m and "ignoring" in m for m in messages)


class TestVideoMoodRealTracks:
    """Uses the real music/ folder and tracks.yaml, with the science niche's keywords."""

    COSMIC = {"pulsar.mp3", "event-horizon.mp3", "singularity.mp3", "laniakea.mp3", "nebula.mp3"}
    MYSTERIOUS = {"rapid-unscheduled-disassembly.mp3", "false-vacuum-decay.mp3", "red-shift.mp3"}
    HOPEFUL = {"spaghettification.mp3", "frame-dragging.mp3", "turn-in-the-sun.mp3", "way-back-home.mp3"}
    QUIRKY = {"caribbean-hide-and-seek.mp3", "fun-house-trap.mp3", "critter-cruise.mp3"}

    @pytest.fixture
    def real_music(self, tmp_path, monkeypatch):
        monkeypatch.setattr(music, "MUSIC_DIR", REAL_MUSIC_DIR)
        monkeypatch.setattr(music, "_get_speech_regions", lambda *a, **k: [(0.0, 1.0)])
        return REAL_MUSIC_DIR

    @staticmethod
    def _science_keywords():
        # Same keyword list cmd_produce builds from the niche profile
        mc = get_music_config(load_niche("science"))
        kw = [w.strip() for w in mc.get("mood", "").replace(",", " ").split() if w.strip()]
        kw += [str(t).strip() for t in mc.get("tags", [])]
        return kw

    def _real_pick(self, tmp_path, video_mood):
        result = select_and_prepare_music(
            tmp_path / "vo.mp3", tmp_path / "work",
            mood_keywords=self._science_keywords(), video_mood=video_mood,
        )
        return Path(result["track_path"]).name

    def test_mysterious_picks_mysterious_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "mysterious") for _ in range(60)}
        assert picks <= self.MYSTERIOUS
        assert picks  # at least one pick was made

    def test_hopeful_picks_hopeful_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "hopeful") for _ in range(60)}
        assert picks <= self.HOPEFUL

    def test_quirky_picks_quirky_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "quirky") for _ in range(60)}
        assert picks <= self.QUIRKY

    def test_cosmic_picks_cosmic_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "cosmic") for _ in range(60)}
        assert picks <= self.COSMIC

    def test_none_picks_cosmic_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, None) for _ in range(60)}
        assert picks <= self.COSMIC

    def test_invalid_mood_picks_same_as_none(self, real_music, tmp_path, monkeypatch):
        for seed in range(30):
            monkeypatch.setattr(music, "random", random.Random(seed))
            without = self._real_pick(tmp_path, None)
            monkeypatch.setattr(music, "random", random.Random(seed))
            invalid = self._real_pick(tmp_path, "banana")
            assert invalid == without


class TestRealMusicFolder:
    def test_every_mp3_is_listed_in_tracks_yaml(self):
        data = yaml.safe_load((REAL_MUSIC_DIR / "tracks.yaml").read_text(encoding="utf-8"))
        listed = set(data)
        on_disk = {p.name for p in REAL_MUSIC_DIR.glob("*.mp3")}
        assert on_disk - listed == set(), "mp3 files in music/ with no entry in tracks.yaml"

    def test_every_tracks_yaml_entry_exists_on_disk(self):
        data = yaml.safe_load((REAL_MUSIC_DIR / "tracks.yaml").read_text(encoding="utf-8"))
        on_disk = {p.name for p in REAL_MUSIC_DIR.glob("*.mp3")}
        assert set(data) - on_disk == set(), "tracks.yaml entries with no file in music/"
