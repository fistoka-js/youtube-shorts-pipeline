"""Tests for verticals/music.py — track selection by mood tags, duck filter generation,
start point, gain and the music level, plus the music filter in assemble.py.

Most tests need no real mp3 contents: tracks are empty files in a temp folder, and
the speech-region step (Whisper) is replaced with a fixed value. The render tests
at the end make tiny synthetic WAV files with ffmpeg and are skipped without ffmpeg.
"""

import math
import random
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml

from verticals import music
from verticals.assemble import build_audio_filter
from verticals.music import build_duck_filter, select_and_prepare_music
from verticals.niche import get_music_config, load_niche

REAL_MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"

HAS_FFMPEG = shutil.which("ffmpeg") is not None

# Measured values are stored to 0.1 dB, so a check on them can be off by up to 0.1 dB
STORED_ROUNDING_DB = 0.1


class TestBuildDuckFilter:
    SPEECH, GAP = music.duck_levels()

    def test_empty_regions(self):
        result = build_duck_filter([])
        assert result == f"volume={self.GAP:.4f}"

    def test_single_region(self):
        result = build_duck_filter([(1.0, 3.0)])
        assert "volume=" in result
        assert "between(t," in result
        assert f"{self.SPEECH:.4f}" in result
        assert f"{self.GAP:.4f}" in result

    def test_multiple_regions(self, sample_speech_regions):
        result = build_duck_filter(sample_speech_regions)
        assert result.count("between(t,") == 2
        assert f"{self.SPEECH:.4f}" in result
        assert f"{self.GAP:.4f}" in result
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
        assert f", {self.SPEECH:.4f}, {self.GAP:.4f})" in result


class TestDuckLevels:
    """The music level relative to the voice, from the constants in music.py."""

    @staticmethod
    def _rel_db(volume):
        # Level of the music relative to the voice, in dB, once gained to TARGET_LUFS
        return 20 * math.log10(volume) + music.TARGET_LUFS - music.VOICE_LUFS

    def test_speech_level_is_music_below_voice_db(self):
        speech, _ = music.duck_levels()
        assert self._rel_db(speech) == pytest.approx(-music.MUSIC_BELOW_VOICE_DB, abs=1e-9)

    def test_production_music_level_is_19_db_below_voice(self):
        # Jacob chose 19 dB after listening to the audition. A silent change must fail here.
        assert music.MUSIC_BELOW_VOICE_DB == 19.0

    def test_19_db_below_voice_gives_these_literal_volumes(self):
        # Written out by hand: a voice at -20.3 LUFS with music 19 dB under it, gained to
        # -16 LUFS, is a speech volume of 10^(-23.3/20) = 0.0684. The gap is 6.375 dB higher
        # (the old duck ratio 0.25/0.12), which is -12.6 dB, so the 6 dB cap does not engage.
        # That gives 0.0684 * 2.0833 = 0.1425.
        speech, gap = music.duck_levels(19)
        assert speech == pytest.approx(0.0684, abs=0.0001)
        assert gap == pytest.approx(0.1425, abs=0.0001)

    def test_16_db_below_voice_gives_these_literal_volumes(self):
        # 16 dB is no longer the production level. It is kept as an arbitrary input.
        # Written out by hand: a voice at -20.3 LUFS with music 16 dB under it, gained to
        # -16 LUFS, is a speech volume of 0.0966. The gap is 6.375 dB higher, which is
        # -9.6 dB, so the 6 dB cap does not engage here either. That gives 0.2013.
        speech, gap = music.duck_levels(16)
        assert speech == pytest.approx(0.0966, abs=0.0001)
        assert gap == pytest.approx(0.2013, abs=0.0001)

    def test_gap_keeps_the_old_duck_ratio_when_not_capped(self):
        speech, gap = music.duck_levels()
        assert gap / speech == pytest.approx(0.25 / 0.12, rel=1e-9)

    def test_gap_is_never_closer_than_the_cap_to_the_voice(self):
        for level in (6, 8, 12, 16, 20, 30):
            _, gap = music.duck_levels(level)
            assert self._rel_db(gap) <= -music.GAP_MUSIC_MIN_BELOW_VOICE_DB + 1e-9, level

    def test_cap_engages_for_a_high_speech_level(self):
        # 8 dB below voice in speech would put the gap at -1.6 dB; the cap keeps it at -6
        _, gap = music.duck_levels(8)
        assert self._rel_db(gap) == pytest.approx(-music.GAP_MUSIC_MIN_BELOW_VOICE_DB, abs=1e-9)

    def test_gap_is_never_quieter_than_speech(self):
        # Guards against an inverted duck (gaps quieter than speech)
        for level in (6, 8, 12, 16, 20, 30):
            speech, gap = music.duck_levels(level)
            assert self._rel_db(gap) >= self._rel_db(speech) - 1e-9, level

    def test_speech_level_closer_than_the_cap_is_refused(self):
        with pytest.raises(ValueError):
            music.duck_levels(2)
        with pytest.raises(ValueError):
            music.duck_levels(music.GAP_MUSIC_MIN_BELOW_VOICE_DB - 0.5)

    def test_music_is_never_louder_than_voice_in_gaps_or_speech(self):
        for level in (6, 12, 16, 20):
            speech, gap = music.duck_levels(level)
            assert self._rel_db(speech) < 0
            assert self._rel_db(gap) < 0



@pytest.fixture
def fake_music(tmp_path, monkeypatch):
    """Point MUSIC_DIR at a temp folder and skip the Whisper step.

    Call make(tags_yaml, files) to create the tracks. files is a list of
    file names to create as empty mp3s; tags_yaml is the text of tracks.yaml
    (or None to leave the file out). No niche limit is passed, so these tests
    exercise the mood logic on every file. The per-niche limit has its own tests.
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


def _pick_allowed(keywords, allowed, tmp_path):
    result = select_and_prepare_music(
        tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=keywords, allowed_tracks=allowed,
    )
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
        assert set(result) == {"track_path", "duck_filter", "start_at", "gain_db"}
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


class TestStartAndGain:
    """start_at and gain_db from tracks.yaml reach the caller, and unusable tracks are never picked."""

    def test_plain_tag_list_starts_at_zero_with_no_gain(self, fake_music, tmp_path):
        fake_music("a.mp3: [ambient]\n", ["a.mp3"])
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"])
        assert result["start_at"] == 0.0
        assert result["gain_db"] == 0.0

    def test_mapping_entry_returns_its_start_and_gain(self, fake_music, tmp_path):
        fake_music(
            "a.mp3:\n"
            "  tags: [ambient]\n"
            "  start_at: 12.5\n"
            "  gain_db: -3.25\n",
            ["a.mp3"],
        )
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"])
        assert result["start_at"] == pytest.approx(12.5)
        assert result["gain_db"] == pytest.approx(-3.25)

    def test_mapping_entry_without_start_or_gain_defaults_to_zero(self, fake_music, tmp_path):
        fake_music("a.mp3:\n  tags: [ambient]\n", ["a.mp3"])
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"])
        assert result["start_at"] == 0.0
        assert result["gain_db"] == 0.0

    def test_unusable_track_is_never_picked_even_by_fallback(self, fake_music, tmp_path):
        fake_music(
            "a.mp3:\n  tags: [ambient]\n  unusable: true\n"
            "b.mp3: [dark]\n",
            ["a.mp3", "b.mp3"],
        )
        # a.mp3 is the only ambient track but is unusable, so there is no music
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"]) == {}

    def test_unusable_track_is_skipped_when_a_usable_one_exists(self, fake_music, tmp_path):
        fake_music(
            "a.mp3:\n  tags: [cosmic]\n  unusable: true\n"
            "b.mp3: [cosmic]\n",
            ["a.mp3", "b.mp3"],
        )
        picks = {_pick(["cosmic"], tmp_path) for _ in range(40)}
        assert picks == {"b.mp3"}

    def test_non_numeric_start_at_makes_track_unusable(self, fake_music, tmp_path):
        fake_music(
            "a.mp3:\n  tags: [ambient]\n  start_at: soon\n",
            ["a.mp3"],
        )
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"]) == {}

    def test_track_not_in_niche_list_is_never_picked_even_by_fallback(self, fake_music, tmp_path):
        fake_music("a.mp3: [ambient]\nb.mp3: [ambient]\n", ["a.mp3", "b.mp3"])
        picks = {_pick_allowed(["cosmic"], ["b.mp3"], tmp_path) for _ in range(40)}
        assert picks == {"b.mp3"}

    def test_no_track_in_niche_list_means_no_music(self, fake_music, tmp_path):
        fake_music("a.mp3: [ambient]\n", ["a.mp3"])
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work",
                                        mood_keywords=["cosmic"], allowed_tracks=["zzz.mp3"]) == {}

    def test_no_niche_list_means_no_limit(self, fake_music, tmp_path):
        fake_music("a.mp3: [ambient]\nb.mp3: [ambient]\n", ["a.mp3", "b.mp3"])
        picks = {_pick_allowed(["cosmic"], None, tmp_path) for _ in range(60)}
        assert picks == {"a.mp3", "b.mp3"}


class TestNicheAllowedTracks:
    """The per-niche limit. Only the science niche has one."""

    COSMIC = {"pulsar.mp3", "event-horizon.mp3", "singularity.mp3", "laniakea.mp3", "nebula.mp3"}

    def test_science_niche_allows_only_the_five_cosmic_tracks(self):
        allowed = get_music_config(load_niche("science"))["allowed_tracks"]
        assert set(allowed) == self.COSMIC

    def test_every_science_allowed_track_exists_on_disk_and_is_in_tracks_yaml(self):
        data = yaml.safe_load((REAL_MUSIC_DIR / "tracks.yaml").read_text(encoding="utf-8"))
        on_disk = {p.name for p in REAL_MUSIC_DIR.glob("*.mp3")}
        for name in get_music_config(load_niche("science"))["allowed_tracks"]:
            assert name in on_disk, name
            assert name in data, name

    def test_tech_niche_has_no_track_limit(self):
        assert get_music_config(load_niche("tech"))["allowed_tracks"] is None

    def test_only_science_limits_its_tracks(self):
        niches_dir = Path(__file__).resolve().parent.parent / "niches"
        limited = []
        for path in sorted(niches_dir.glob("*.yaml")):
            if get_music_config(load_niche(path.stem))["allowed_tracks"] is not None:
                limited.append(path.stem)
        assert limited == ["science"]


class TestVideoMoodRealTracks:
    """Uses the real music/ folder and tracks.yaml, with the science niche's keywords.

    The science niche allows only the five cosmic tracks (allowed_tracks in
    niches/science.yaml), so the video's mood never brings in a mysterious, hopeful
    or quirky track. Of the cosmic tracks, the ones marked unusable in tracks.yaml
    are skipped.
    """

    NON_COSMIC = {
        "rapid-unscheduled-disassembly.mp3", "false-vacuum-decay.mp3", "red-shift.mp3",
        "spaghettification.mp3", "frame-dragging.mp3", "turn-in-the-sun.mp3", "way-back-home.mp3",
        "caribbean-hide-and-seek.mp3", "fun-house-trap.mp3", "critter-cruise.mp3",
    }

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

    @staticmethod
    def _science_allowed():
        return get_music_config(load_niche("science"))["allowed_tracks"]

    def _usable_cosmic(self):
        entries = music._load_track_entries()
        return {n for n in self._science_allowed() if entries.get(n, {}).get("usable")}

    def _real_pick(self, tmp_path, video_mood):
        result = select_and_prepare_music(
            tmp_path / "vo.mp3", tmp_path / "work",
            mood_keywords=self._science_keywords(), video_mood=video_mood,
            allowed_tracks=self._science_allowed(),
        )
        return Path(result["track_path"]).name

    def test_mysterious_does_not_pick_a_mysterious_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "mysterious") for _ in range(200)}
        assert picks.isdisjoint(self.NON_COSMIC)
        assert picks <= self._usable_cosmic()

    def test_hopeful_does_not_pick_a_hopeful_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "hopeful") for _ in range(200)}
        assert picks.isdisjoint(self.NON_COSMIC)
        assert picks <= self._usable_cosmic()

    def test_quirky_does_not_pick_a_quirky_track(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "quirky") for _ in range(200)}
        assert picks.isdisjoint(self.NON_COSMIC)
        assert picks <= self._usable_cosmic()

    def test_cosmic_picks_only_usable_cosmic_tracks(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "cosmic") for _ in range(200)}
        assert picks == self._usable_cosmic()

    def test_none_picks_only_usable_cosmic_tracks(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, None) for _ in range(200)}
        assert picks == self._usable_cosmic()

    def test_unusable_laniakea_is_never_picked(self, real_music, tmp_path):
        picks = {self._real_pick(tmp_path, "cosmic") for _ in range(200)}
        assert "laniakea.mp3" not in picks

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

    def test_every_entry_is_measured(self):
        data = yaml.safe_load((REAL_MUSIC_DIR / "tracks.yaml").read_text(encoding="utf-8"))
        for name, value in data.items():
            assert isinstance(value, dict), f"{name} has no measured data - run scripts/measure_music_tracks.py"
            assert isinstance(value["steady_lufs"], (int, float)), name
            if value.get("unusable") is True:
                assert "start_at" not in value, f"unusable {name} must have no start_at"
            else:
                assert isinstance(value["start_at"], (int, float)), name
                assert isinstance(value["gain_db"], (int, float)), name
                assert isinstance(value["start_lufs"], (int, float)), name

    def test_gain_brings_each_track_to_the_target(self):
        data = yaml.safe_load((REAL_MUSIC_DIR / "tracks.yaml").read_text(encoding="utf-8"))
        for name, value in data.items():
            if value.get("unusable") is True:
                continue
            assert value["steady_lufs"] + value["gain_db"] == pytest.approx(music.TARGET_LUFS, abs=0.11), name

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
    def test_track_never_starts_at_a_quiet_section_measured_by_ffmpeg(self):
        # For every usable track: cut the 3 s that playback hears from start_at with
        # ffmpeg and measure it. This does not use the measuring script's functions,
        # so it checks the audio itself, not the script's own numbers.
        data = yaml.safe_load((REAL_MUSIC_DIR / "tracks.yaml").read_text(encoding="utf-8"))
        checked = 0
        for name, value in data.items():
            if value.get("unusable") is True:
                assert "start_at" not in value, name
                continue
            assert isinstance(value["start_at"], (int, float)) and value["start_at"] >= 0, name
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(value["start_at"]), "-t", "3",
                 "-i", str(REAL_MUSIC_DIR / name), "-af", "ebur128=peak=none", "-f", "null", "-"],
                capture_output=True, text=True, check=True)
            heard = float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", r.stderr)[-1])
            # steady_lufs is stored to 0.1 dB, so allow that rounding on top of the 3 dB rule
            assert abs(heard - value["steady_lufs"]) <= 3.0 + STORED_ROUNDING_DB, (
                f"{name}: the 3 s from start_at measure {heard:.1f} LUFS, more than 3 dB from "
                f"steady {value['steady_lufs']:.1f}")
            checked += 1
        assert checked > 0


class TestBuildAudioFilter:
    """The filter string that assemble_video (and the audition) use for the music."""

    def test_music_starts_at_start_at_and_loops_from_there(self):
        f = build_audio_filter(40.0, None, start_at=12.5, gain_db=0.0)
        assert "[2:a]atrim=start=12.500,asetpts=PTS-STARTPTS,aloop=loop=-1" in f

    def test_loop_comes_after_the_trim_to_start_at(self):
        f = build_audio_filter(40.0, None, start_at=12.5, gain_db=0.0)
        assert f.index("atrim=start=") < f.index("aloop=")

    def test_gain_is_applied_before_the_duck(self):
        f = build_audio_filter(40.0, "volume='if(between(t,0,1), 0.1, 0.2)':eval=frame", gain_db=-3.25)
        assert "volume=-3.25dB" in f
        assert f.index("volume=-3.25dB") < f.index("if(between")

    def test_one_second_fade_in_and_two_second_fade_out(self):
        f = build_audio_filter(40.0, None)
        assert "afade=t=in:st=0:d=1" in f
        assert "afade=t=out:st=38.000:d=2" in f

    def test_fade_out_follows_the_voice_length(self):
        f = build_audio_filter(25.0, None)
        assert "afade=t=out:st=23.000:d=2" in f

    def test_input_numbers_follow_the_arguments(self):
        f = build_audio_filter(40.0, None, voice_index=0, music_index=1)
        assert "[1:a]atrim=start=" in f
        assert "[0:a][music]amix=inputs=2:duration=first" in f

    def test_default_inputs_are_voice_1_and_music_2(self):
        f = build_audio_filter(40.0, None)
        assert "[2:a]atrim=" in f
        assert "[1:a][music]amix=inputs=2:duration=first" in f


def _write_wav(path: Path, samples: np.ndarray, rate: int = 44100) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def _rms_db(samples: np.ndarray, rate: int, start: float, end: float) -> float:
    seg = samples[int(start * rate):int(end * rate)]
    return 10 * math.log10(np.mean(seg ** 2) + 1e-20)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
class TestRenderedMusic:
    """Renders a synthetic track through the real filter with ffmpeg and measures it.

    The synthetic track is quiet (-60 dBFS) for its first 5 s, then loud (about
    -10 dBFS) from 5 s to 12 s. A track that starts at a quiet section would
    therefore be audibly wrong, and a loop back to 0 s would bring the quiet part back.
    """

    RATE = 44100

    @pytest.fixture
    def assets(self, tmp_path):
        t = np.arange(int(12 * self.RATE)) / self.RATE
        track = np.where(t < 5.0, 0.001, 0.3) * np.sin(2 * np.pi * 440 * t)
        _write_wav(tmp_path / "track.wav", track, self.RATE)
        voice = 0.1 * np.sin(2 * np.pi * 220 * np.arange(int(14 * self.RATE)) / self.RATE)
        _write_wav(tmp_path / "voice.wav", voice, self.RATE)
        return tmp_path

    def _render(self, assets, start_at, gain_db=0.0, duck=None, seconds=14.0):
        out = assets / f"out_{start_at}_{gain_db}.wav"
        f = build_audio_filter(seconds, duck, start_at=start_at, gain_db=gain_db,
                               voice_index=0, music_index=1).split(";", 1)[0]
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-t", str(seconds), "-i", str(assets / "voice.wav"),
            "-i", str(assets / "track.wav"), "-filter_complex", f, "-map", "[music]",
            "-ac", "1", "-ar", str(self.RATE), str(out),
        ], check=True)
        return _read_wav(out)

    def test_render_has_the_voice_length(self, assets):
        samples = self._render(assets, start_at=5.0, seconds=14.0)
        assert len(samples) / self.RATE == pytest.approx(14.0, abs=0.02)

    def test_starting_at_start_at_avoids_the_quiet_intro(self, assets):
        samples = self._render(assets, start_at=5.0)
        loud = _rms_db(samples, self.RATE, 1.5, 3.5)  # after the 1 s fade-in
        assert loud == pytest.approx(-13.5, abs=1.5)

    def test_starting_at_zero_plays_the_quiet_intro(self, assets):
        samples = self._render(assets, start_at=0.0)
        quiet = _rms_db(samples, self.RATE, 1.5, 3.5)
        assert quiet < -50

    def test_loop_restarts_at_start_at_not_at_zero(self, assets):
        # Segment from 5 s is 7 s long, so the loop restarts at 7 s of output.
        # It must restart at 5 s of the track (loud), not at 0 s (quiet).
        samples = self._render(assets, start_at=5.0)
        after_loop = _rms_db(samples, self.RATE, 7.2, 7.8)
        assert after_loop == pytest.approx(-13.5, abs=1.5)

    def test_gain_db_raises_the_level(self, assets):
        base = self._render(assets, start_at=5.0, gain_db=0.0)
        boosted = self._render(assets, start_at=5.0, gain_db=6.0)
        diff = _rms_db(boosted, self.RATE, 1.5, 3.5) - _rms_db(base, self.RATE, 1.5, 3.5)
        assert diff == pytest.approx(6.0, abs=0.3)

    def test_fade_out_brings_the_end_to_silence(self, assets):
        samples = self._render(assets, start_at=5.0, seconds=14.0)
        assert _rms_db(samples, self.RATE, 13.85, 13.98) < _rms_db(samples, self.RATE, 3.0, 4.0) - 15

    def test_duck_level_applied_to_the_render(self, assets):
        speech, gap = music.duck_levels()
        duck = build_duck_filter([(0.0, 14.0)], buffer=0.0, vol_speech=speech, vol_gap=gap)
        base = self._render(assets, start_at=5.0)
        ducked = self._render(assets, start_at=5.0, duck=duck)
        diff = _rms_db(ducked, self.RATE, 1.5, 3.5) - _rms_db(base, self.RATE, 1.5, 3.5)
        assert diff == pytest.approx(20 * math.log10(speech), abs=0.3)


class _SavedState:
    """Stands in for PipelineState: only get_artifact is used by the resume helper."""

    def __init__(self, artifacts):
        self._artifacts = artifacts

    def get_artifact(self, stage, key, default=None):
        return self._artifacts.get(stage, {}).get(key, default)


class TestRejectedStartAndGain:
    """A start_at or gain_db that is negative, infinite, nan or out of range makes the track unusable."""

    @pytest.mark.parametrize("entry", [
        "  start_at: -1.0\n",
        "  start_at: .inf\n",
        "  start_at: .nan\n",
        "  start_at: 46.0\n",
        "  gain_db: .inf\n",
        "  gain_db: -.inf\n",
        "  gain_db: .nan\n",
        "  gain_db: 30.0\n",
        "  gain_db: -25.0\n",
    ])
    def test_bad_value_makes_the_track_unusable_and_logs_it(self, fake_music, tmp_path, monkeypatch, entry):
        fake_music("a.mp3:\n  tags: [ambient]\n" + entry, ["a.mp3"])
        messages = []
        monkeypatch.setattr(music, "log", messages.append)
        assert select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"]) == {}
        assert any("a.mp3" in m and "not a finite number" in m for m in messages)

    def test_a_bad_track_is_skipped_but_a_good_one_is_still_picked(self, fake_music, tmp_path):
        fake_music("a.mp3:\n  tags: [ambient]\n  start_at: .nan\nb.mp3:\n  tags: [ambient]\n  start_at: 5.0\n",
                   ["a.mp3", "b.mp3"])
        picks = {_pick(["cosmic"], tmp_path) for _ in range(30)}
        assert picks == {"b.mp3"}

    def test_limits_themselves_are_accepted(self, fake_music, tmp_path):
        fake_music("a.mp3:\n  tags: [ambient]\n  start_at: 45.0\n  gain_db: -24.0\n", ["a.mp3"])
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=["cosmic"])
        assert result["start_at"] == 45.0
        assert result["gain_db"] == -24.0


class TestAllowedTracksValidation:
    """A niche's allowed_tracks must be a list of file names; anything else is refused or logged."""

    def test_a_plain_string_is_refused_not_matched_as_a_substring(self, fake_music, tmp_path, monkeypatch):
        fake_music("a.mp3: [ambient]\nab.mp3: [ambient]\n", ["a.mp3", "ab.mp3"])
        messages = []
        monkeypatch.setattr(music, "log", messages.append)
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work",
                                          mood_keywords=["cosmic"], allowed_tracks="a.mp3")
        assert result == {}
        assert any("must be a list of file names" in m for m in messages)

    def test_a_non_string_entry_is_refused(self, fake_music, tmp_path, monkeypatch):
        fake_music("a.mp3: [ambient]\n", ["a.mp3"])
        messages = []
        monkeypatch.setattr(music, "log", messages.append)
        result = select_and_prepare_music(tmp_path / "vo.mp3", tmp_path / "work",
                                          mood_keywords=["cosmic"], allowed_tracks=["a.mp3", 5])
        assert result == {}
        assert any("must be a list of file names" in m for m in messages)

    def test_a_name_with_no_file_is_logged_and_ignored(self, fake_music, tmp_path, monkeypatch):
        fake_music("b.mp3: [ambient]\n", ["b.mp3"])
        messages = []
        monkeypatch.setattr(music, "log", messages.append)
        picks = {_pick_allowed(["cosmic"], ["b", "b.mp3"], tmp_path) for _ in range(20)}
        assert picks == {"b.mp3"}
        assert any("allowed_tracks lists b but there is no such file in music/" in m for m in messages)

    def test_a_tuple_of_names_is_accepted(self, fake_music, tmp_path):
        fake_music("a.mp3: [ambient]\nb.mp3: [ambient]\n", ["a.mp3", "b.mp3"])
        picks = {_pick_allowed(["cosmic"], ("b.mp3",), tmp_path) for _ in range(20)}
        assert picks == {"b.mp3"}


class TestResumedMusicState:
    def test_old_saved_music_without_start_at_logs_a_warning(self, monkeypatch):
        from verticals import __main__ as cli
        messages = []
        monkeypatch.setattr(cli, "log", messages.append)
        state = _SavedState({"music": {"track_path": "x/pulsar.mp3", "duck_filter": "volume=0.1"}})
        result = cli._music_result_from_state(state)
        assert result["start_at"] == 0.0
        assert result["gain_db"] == 0.0
        assert any("WARNING" in m and "--force" in m for m in messages)

    def test_saved_music_with_start_at_has_no_warning(self, monkeypatch):
        from verticals import __main__ as cli
        messages = []
        monkeypatch.setattr(cli, "log", messages.append)
        state = _SavedState({"music": {"track_path": "x/pulsar.mp3", "duck_filter": "volume=0.1",
                                       "start_at": 16.6, "gain_db": -5.3}})
        result = cli._music_result_from_state(state)
        assert result["start_at"] == 16.6
        assert result["gain_db"] == -5.3
        assert messages == []

    def test_saved_track_outside_the_niche_list_logs_a_warning(self, monkeypatch):
        from verticals import __main__ as cli
        messages = []
        monkeypatch.setattr(cli, "log", messages.append)
        state = _SavedState({"music": {"track_path": "x/pulsar.mp3", "duck_filter": "volume=0.1",
                                       "start_at": 16.6, "gain_db": -5.3}})
        result = cli._music_result_from_state(state, ["nebula.mp3"])
        assert result["track_path"] == "x/pulsar.mp3"  # kept, but flagged
        assert any("not in this niche's allowed_tracks" in m and "pulsar.mp3" in m for m in messages)

    def test_saved_track_inside_the_niche_list_has_no_warning(self, monkeypatch):
        from verticals import __main__ as cli
        messages = []
        monkeypatch.setattr(cli, "log", messages.append)
        state = _SavedState({"music": {"track_path": "x/pulsar.mp3", "duck_filter": "volume=0.1",
                                       "start_at": 16.6, "gain_db": -5.3}})
        cli._music_result_from_state(state, ["pulsar.mp3", "nebula.mp3"])
        assert messages == []

    def test_no_niche_list_has_no_allowed_track_warning(self, monkeypatch):
        from verticals import __main__ as cli
        messages = []
        monkeypatch.setattr(cli, "log", messages.append)
        state = _SavedState({"music": {"track_path": "x/pulsar.mp3", "duck_filter": "volume=0.1",
                                       "start_at": 16.6, "gain_db": -5.3}})
        cli._music_result_from_state(state, None)
        assert messages == []

    def test_no_music_saved_has_no_warning(self, monkeypatch):
        from verticals import __main__ as cli
        messages = []
        monkeypatch.setattr(cli, "log", messages.append)
        state = _SavedState({"music": {"track_path": "", "duck_filter": ""}})
        cli._music_result_from_state(state)
        assert messages == []


class TestTechIsNotLimited:
    """Other niches behave as before: they pick by their own tags and mood, with no track limit."""

    @pytest.fixture
    def real_music(self, tmp_path, monkeypatch):
        monkeypatch.setattr(music, "MUSIC_DIR", REAL_MUSIC_DIR)
        monkeypatch.setattr(music, "_get_speech_regions", lambda *a, **k: [(0.0, 1.0)])
        return REAL_MUSIC_DIR

    @staticmethod
    def _tech_pick(tmp_path, video_mood):
        mc = get_music_config(load_niche("tech"))
        kw = [w.strip() for w in mc.get("mood", "").replace(",", " ").split() if w.strip()]
        kw += [str(t).strip() for t in mc.get("tags", [])]
        result = select_and_prepare_music(
            tmp_path / "vo.mp3", tmp_path / "work", mood_keywords=kw, video_mood=video_mood,
            allowed_tracks=mc["allowed_tracks"],
        )
        return Path(result["track_path"]).name

    def test_tech_can_pick_a_non_cosmic_track_that_matches_its_tags(self, real_music, tmp_path):
        picks = {self._tech_pick(tmp_path, None) for _ in range(300)}
        assert "paulyudin-technology-tech-technology-484304.mp3" in picks

    def test_tech_quirky_mood_still_picks_quirky_tracks(self, real_music, tmp_path):
        quirky = {"caribbean-hide-and-seek.mp3", "fun-house-trap.mp3", "critter-cruise.mp3"}
        picks = {self._tech_pick(tmp_path, "quirky") for _ in range(200)}
        assert picks <= quirky

    def test_tech_hopeful_mood_still_picks_hopeful_tracks(self, real_music, tmp_path):
        hopeful = {"spaghettification.mp3", "frame-dragging.mp3", "turn-in-the-sun.mp3", "way-back-home.mp3"}
        picks = {self._tech_pick(tmp_path, "hopeful") for _ in range(200)}
        assert picks <= hopeful
