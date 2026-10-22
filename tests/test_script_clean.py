"""Tests for removing bracketed visual notes from scripts (verticals/script_clean.py).

Covers the helper itself, the call sites in verticals/draft.py, verticals/tts.py
and verticals/__main__.py (cmd_produce and the long-form join), and the caption
alignment. Every LLM, TTS, Whisper, Pexels, music and assembly call is mocked.
Nothing touches the real media or drafts folders: they are redirected to tmp_path.
"""

import argparse
import json
import logging
from unittest.mock import MagicMock, call, patch

import pytest

from verticals import __main__ as M
from verticals import draft as draft_mod
from verticals import log as log_mod
from verticals.captions import _align_to_script
from verticals.draft import (
    FALLBACK_BROLL_PROMPT_COUNT,
    generate_draft,
    generate_long_form_draft,
)
from verticals.script_clean import clean_script, strip_script_markers
from verticals.tts import generate_voiceover


@pytest.fixture(autouse=True)
def _keep_real_folders_untouched(tmp_path, monkeypatch):
    """Every test in this file works only on temp files.

    - The variation history is redirected to tmp_path through VARIATION_HISTORY_PATH
      (read at call time). record_variation is mocked, so nothing is appended.
    - The pipeline logger is replaced by a silent one, so log() never creates a file
      in the project's logs folder.
    - The media and drafts folders used by cmd_produce are redirected to tmp_path.
    - The draft tests mock research and the LLM, and the TTS tests mock the engines.
    """
    monkeypatch.setenv("VARIATION_HISTORY_PATH", str(tmp_path / "variation_history.json"))
    monkeypatch.setattr(draft_mod, "record_variation", MagicMock())
    silent = logging.getLogger("script-clean-tests-silent")
    silent.handlers = [logging.NullHandler()]
    silent.propagate = False
    monkeypatch.setattr(log_mod, "get_logger", lambda: silent)
    monkeypatch.setattr(M, "MEDIA_DIR", tmp_path / "media")
    monkeypatch.setattr(M, "DRAFTS_DIR", tmp_path / "drafts")

MARKER_LOG = "Removed {} bracketed marker(s) from the script."
DIRTY_SCRIPT = (
    "[B-ROLL: person yawning]\n"
    "Ever wondered why we yawn? Yawning is [B-ROLL: close-up of a yawn] contagious."
)
CLEAN_SCRIPT = "Ever wondered why we yawn? Yawning is contagious."


# 1. Marker removal gives clean narration; word count equals the spoken words.
def test_markers_removed_and_word_count_matches_spoken_words():
    cleaned = clean_script(DIRTY_SCRIPT)
    assert cleaned == CLEAN_SCRIPT
    assert "[" not in cleaned and "]" not in cleaned
    assert len(cleaned.split()) == len(CLEAN_SCRIPT.split()) == 8


# 2. A marker in the middle of a sentence reads naturally afterwards.
def test_marker_mid_sentence_reads_naturally():
    cleaned = clean_script("The yawn [B-ROLL: close-up] spreads fast.")
    assert cleaned == "The yawn spreads fast."
    assert "  " not in cleaned
    assert " ." not in clean_script("Yawning is contagious [B-ROLL: x].")
    assert clean_script("Yawning is contagious [B-ROLL: x].") == "Yawning is contagious."


# 3. Caption alignment still works with a cleaned script.
def test_caption_alignment_works_with_cleaned_script():
    script = clean_script(DIRTY_SCRIPT)
    whisper = [
        {"word": "Ever", "start": 0.0, "end": 0.3},
        {"word": "wondered", "start": 0.3, "end": 0.8},
        {"word": "why", "start": 0.8, "end": 1.0},
        {"word": "we", "start": 1.0, "end": 1.2},
        {"word": "yon?", "start": 1.2, "end": 1.6},  # Whisper mishears "yawn"
        {"word": "Yawning", "start": 1.8, "end": 2.3},
        {"word": "is", "start": 2.3, "end": 2.5},
        {"word": "contagious.", "start": 2.5, "end": 3.2},
    ]
    out = _align_to_script(whisper, script)
    words = [w["word"] for w in out]
    assert words == script.split()
    assert not any("B-ROLL" in w or "[" in w for w in words)
    assert out[4]["word"] == "yawn?"
    assert out[4]["start"] == 1.2 and out[4]["end"] == 1.6  # Whisper's timing kept


# 4. A script without markers is returned exactly unchanged.
def test_script_without_markers_is_returned_exactly_unchanged():
    odd = "  Odd   spacing\t here (keep this) *and this*.\n\n\n  Second line  \n"
    with patch("verticals.script_clean.log") as mock_log:
        assert clean_script(odd) == odd
        assert strip_script_markers(odd) == (odd, 0)
    mock_log.assert_not_called()


# 5. The log line appears only when something was removed, with the right N, once per call.
def test_log_line_only_when_removed_with_right_count_once_per_call():
    with patch("verticals.script_clean.log") as mock_log:
        clean_script("Plain sentence with nothing to remove.")
        mock_log.assert_not_called()

        clean_script("A [x] b [y] c.")
        assert mock_log.call_args_list == [call(MARKER_LOG.format(2))]


def test_strip_script_markers_does_not_log():
    with patch("verticals.script_clean.log") as mock_log:
        cleaned, n = strip_script_markers("A [x] b.")
    assert (cleaned, n) == ("A b.", 1)
    mock_log.assert_not_called()


# 10. Edge cases.
class TestEdgeCases:
    def test_nested_brackets_removed_as_one_marker(self):
        assert strip_script_markers("a [x [y] z] b") == ("a b", 1)

    def test_unmatched_open_bracket_removed_and_rest_kept(self):
        cleaned, n = strip_script_markers("Before [ after the bracket.")
        assert cleaned == "Before after the bracket."
        assert n == 0

    def test_unmatched_close_bracket_removed(self):
        cleaned, n = strip_script_markers("Before ] after.")
        assert cleaned == "Before after."
        assert n == 0

    def test_marker_alone_on_a_line_drops_that_line(self):
        cleaned, n = strip_script_markers("First line.\n[B-ROLL: x]\nSecond line.")
        assert cleaned == "First line.\nSecond line."
        assert n == 1

    def test_multiple_markers_on_one_line(self):
        cleaned, n = strip_script_markers("[A] one [B] two [C] three")
        assert cleaned == "one two three"
        assert n == 3

    def test_text_with_only_a_marker_becomes_empty(self):
        assert strip_script_markers("[B-ROLL: person yawning]") == ("", 1)

    def test_none_becomes_empty_string(self):
        assert strip_script_markers(None) == ("", 0)
        assert clean_script(None) == ""

    def test_empty_string_unchanged(self):
        assert clean_script("") == ""

    def test_non_string_input_does_not_crash(self):
        assert strip_script_markers(42) == (42, 0)

    def test_parentheses_asterisks_and_sfx_labels_are_kept(self):
        text = "SFX: (whoosh) *dramatic* [B-ROLL: x] done."
        assert clean_script(text) == "SFX: (whoosh) *dramatic* done."


# 7. Shorts draft sanitizer cleans markers, and the prompt rule is sent to the model.
class TestShortsDraft:
    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_shorts_draft_script_is_cleaned(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({
            "script": DIRTY_SCRIPT,
            "broll_prompts": ["Prompt 1", "Prompt 2"],
            "youtube_title": "T",
            "youtube_description": "D",
            "youtube_tags": "t",
            "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })
        with patch("verticals.script_clean.log") as mock_log:
            draft = generate_draft("Yawning")
        assert draft["script"] == CLEAN_SCRIPT
        mock_log.assert_called_once_with(MARKER_LOG.format(2))

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_shorts_prompt_tells_model_script_has_no_visual_notes(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({"script": "Hello.", "broll_prompts": ["p"]})
        generate_draft("Yawning")
        sent_prompt = mock_claude.call_args_list[0][0][0]  # the draft prompt (the fact check call comes after it)
        assert 'The "script" field contains ONLY the words to be spoken aloud' in sent_prompt
        assert "no square-bracket notes" in sent_prompt
        assert "All visuals belong in broll_prompts" in sent_prompt

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft.call_llm")
    def test_long_form_prompt_does_not_get_the_new_rule_but_narration_is_cleaned(
        self, mock_llm, mock_research
    ):
        mock_research.return_value = "research"
        mock_llm.return_value = json.dumps({
            "broll_subject": "yawn",
            "sections": [{
                "id": "s1",
                "heading": "Hook",
                "narration": "[B-ROLL: person yawning]\nYawns spread fast. [B-ROLL: crowd]",
                "broll_subject": "yawn",
                "broll_prompts": ["a yawn"],
            }],
            "shorts_cutpoints": [],
        })
        with patch("verticals.script_clean.log") as mock_log:
            draft = generate_long_form_draft("Yawning")
        assert draft["sections"][0]["narration"] == "Yawns spread fast."
        mock_log.assert_called_once_with(MARKER_LOG.format(2))
        assert "ONLY the words to be spoken aloud" not in mock_llm.call_args[0][0]


# 8. generate_voiceover cleans its input before the engine sees it.
def test_generate_voiceover_cleans_input_before_engine(tmp_path):
    captured = {}

    def fake_engine(script, out_dir, lang, voice_override):
        captured["script"] = script
        return tmp_path / "vo.mp3"

    with patch("verticals.tts.get_tts_provider", return_value="edge"), \
         patch("verticals.tts._generate_edge_tts", side_effect=fake_engine), \
         patch("verticals.script_clean.log") as mock_log:
        generate_voiceover(DIRTY_SCRIPT, tmp_path, "en")
    assert captured["script"] == CLEAN_SCRIPT
    mock_log.assert_called_once_with(MARKER_LOG.format(2))


def test_generate_voiceover_clean_text_logs_nothing(tmp_path):
    captured = {}

    def fake_engine(script, out_dir, lang, voice_override):
        captured["script"] = script
        return tmp_path / "vo.mp3"

    with patch("verticals.tts.get_tts_provider", return_value="edge"), \
         patch("verticals.tts._generate_edge_tts", side_effect=fake_engine), \
         patch("verticals.script_clean.log") as mock_log:
        generate_voiceover(CLEAN_SCRIPT, tmp_path, "en")
    assert captured["script"] == CLEAN_SCRIPT
    mock_log.assert_not_called()


# 6 and 9. cmd_produce: cleaned text reaches the b-roll estimate, the voice, the
# captions and the file-name title; the Hindi path is cleaned; the fallback count is used.
def _write_draft(tmp_path, fields):
    draft = {
        "job_id": "777",
        "niche": "general",
        "news": "Yawning",
        "platform": "shorts",
        "broll_subject": "yawn",
        "broll_prompts": ["yawn close-up", "crowd"],
    }
    draft.update(fields)
    draft = {k: v for k, v in draft.items() if v is not None}  # None means "leave this key out"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    return path


def _run_produce(tmp_path, fields, lang="en", script_override=None):
    draft_path = _write_draft(tmp_path, fields)
    args = argparse.Namespace(
        draft=str(draft_path), lang=lang, force=False, voice=None, script=script_override
    )

    def fake_voiceover(script, out_dir, lang_, provider=None, voice_config=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "vo.mp3"
        path.write_bytes(b"x")
        return path

    def fake_assemble(**kwargs):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"x")
        return path

    mocks = {
        "estimate": MagicMock(return_value=1.0),
        "broll": MagicMock(return_value=[]),
        "voice": MagicMock(side_effect=fake_voiceover),
        "captions": MagicMock(return_value={"srt_path": "", "ass_path": "", "words": []}),
        "music": MagicMock(return_value={"track_path": "", "duck_filter": ""}),
        "assemble": MagicMock(side_effect=fake_assemble),
    }
    with patch.object(M, "MEDIA_DIR", tmp_path / "media"), \
         patch("verticals.broll.estimate_slot_seconds", mocks["estimate"]), \
         patch("verticals.broll.generate_broll", mocks["broll"]), \
         patch("verticals.tts.generate_voiceover", mocks["voice"]), \
         patch("verticals.captions.generate_captions", mocks["captions"]), \
         patch("verticals.music.select_and_prepare_music", mocks["music"]), \
         patch("verticals.assemble.assemble_video", mocks["assemble"]), \
         patch("verticals.script_clean.log") as mock_log:
        M.cmd_produce(args)
    return mocks, mock_log


def test_produce_uses_cleaned_script_everywhere_including_first_line_marker(tmp_path):
    script = "[B-ROLL: person yawning]\nEver wondered why we yawn? Yawning is [B-ROLL: close-up of a yawn] contagious."
    mocks, mock_log = _run_produce(tmp_path, {"script": script})

    est_script, est_slots = mocks["estimate"].call_args.args
    assert est_script == CLEAN_SCRIPT
    assert est_slots == 2

    assert mocks["voice"].call_args.args[0] == CLEAN_SCRIPT
    assert mocks["captions"].call_args.kwargs["script_text"] == CLEAN_SCRIPT
    assert mocks["assemble"].call_args.kwargs["title"] == "Ever wondered why we yawn"

    # Exactly one "Removed N" line for the whole produce run.
    assert mock_log.call_args_list == [call(MARKER_LOG.format(2))]


def test_produce_hindi_path_is_cleaned(tmp_path):
    fields = {
        "script": "Ever wondered why we yawn? [B-ROLL: person yawning]",
        "script_hi": "[B-ROLL: person yawning]\nयह एक परीक्षण है।",
    }
    mocks, mock_log = _run_produce(tmp_path, fields, lang="hi")

    assert mocks["voice"].call_args.args[0] == "यह एक परीक्षण है।"
    assert mocks["estimate"].call_args.args[0] == "यह एक परीक्षण है।"
    # Captions and the file name keep using the (now cleaned) English draft script.
    assert mocks["captions"].call_args.kwargs["script_text"] == "Ever wondered why we yawn?"
    assert mocks["assemble"].call_args.kwargs["title"] == "Ever wondered why we yawn"
    assert mock_log.call_args_list == [call(MARKER_LOG.format(1))]


def test_produce_script_override_is_cleaned(tmp_path):
    mocks, mock_log = _run_produce(
        tmp_path,
        {"script": "Plain English draft."},
        script_override="[B-ROLL: a crowd] Override line spoken aloud.",
    )
    assert mocks["voice"].call_args.args[0] == "Override line spoken aloud."
    assert mocks["captions"].call_args.kwargs["script_text"] == "Plain English draft."
    assert mock_log.call_args_list == [call(MARKER_LOG.format(1))]


def test_produce_clean_english_script_logs_nothing(tmp_path):
    mocks, mock_log = _run_produce(tmp_path, {"script": CLEAN_SCRIPT})
    assert mocks["voice"].call_args.args[0] == CLEAN_SCRIPT
    mock_log.assert_not_called()


def test_produce_without_broll_prompts_uses_fallback_count(tmp_path):
    # cmd_produce imports the constant from verticals.draft when it runs, so patching
    # it there with a value other than 12 proves the fallback really comes from the
    # constant. A hard-coded 12 would give 12 prompts and fail this test.
    fields = {"script": CLEAN_SCRIPT, "broll_prompts": None}
    with patch("verticals.draft.FALLBACK_BROLL_PROMPT_COUNT", 7):
        mocks, _ = _run_produce(tmp_path, fields)

    prompts = mocks["broll"].call_args.args[0]
    assert prompts == ["Cinematic landscape"] * 7
    assert len(prompts) != 12


# 2c. A marker at the edge of a quoted phrase leaves no stray space inside the quotes.
class TestQuoteEdges:
    def test_marker_after_opening_double_quote(self):
        assert clean_script('"[x] hi"') == '"hi"'

    def test_marker_before_closing_double_quote(self):
        assert clean_script('"Hi [x]"') == '"Hi"'

    def test_marker_edges_single_quotes(self):
        assert clean_script("'[x] hi'") == "'hi'"
        assert clean_script("'Hi [x]'") == "'Hi'"

    def test_marker_edges_curly_quotes(self):
        assert clean_script("“[x] hi”") == "“hi”"
        assert clean_script("“Hi [x]”") == "“Hi”"

    def test_marker_edges_guillemets(self):
        assert clean_script("«[x] hi»") == "«hi»"
        assert clean_script("«Hi [x]»") == "«Hi»"

    def test_apostrophe_inside_word_unaffected(self):
        assert clean_script("don't [x] won't") == "don't won't"
        assert clean_script("rock [x]'n roll") == "rock 'n roll"


# 2f. A space the author wrote next to a removal is kept, so a separate token stays a word.
class TestSeparateTokensKeepTheirSpace:
    def test_space_before_period_after_marker_is_kept(self):
        assert clean_script("yawn [x] .") == "yawn ."

    def test_space_before_comma_after_marker_is_kept(self):
        assert clean_script("a [M] ,") == "a ,"

    def test_space_before_closing_paren_after_marker_is_kept(self):
        assert clean_script("Hello [A] ) next") == "Hello ) next"

    def test_standalone_ellipsis_token_keeps_word_count(self):
        cleaned = clean_script("(note) [M] …")
        assert cleaned == "(note) …"
        assert cleaned.split() == ["(note)", "…"]

    def test_standalone_period_token_keeps_word_count(self):
        assert clean_script("(note) [M] .").split() == ["(note)", "."]

    def test_attached_period_still_joins(self):
        assert clean_script("Hello [A].") == "Hello."

    def test_attached_closing_quote_still_joins(self):
        assert clean_script('"Hi [x]"') == '"Hi"'

    def test_attached_parentheses_still_join(self):
        assert clean_script("Hello ([A]) next") == "Hello () next"
        assert clean_script("([A] hi)") == "(hi)"


# 2d. A real U+E000 character in the text is left untouched next to a removed marker.
def test_real_private_use_character_beside_a_marker_is_kept():
    assert strip_script_markers("x \ue000 [y] z") == ("x \ue000 z", 1)


# 2e. Parentheses, the ellipsis character and lone carriage returns at a removal site.
class TestBracketsEllipsisAndCR:
    def test_marker_inside_parentheses_leaves_no_space_inside(self):
        assert clean_script("Hello ([A]) next") == "Hello () next"

    def test_marker_after_opening_parenthesis(self):
        assert clean_script("([A] hi)") == "(hi)"

    def test_marker_before_closing_parenthesis(self):
        assert clean_script("Hello [A]) next") == "Hello) next"

    def test_marker_before_ellipsis_character(self):
        assert clean_script("Hello [A]…") == "Hello…"

    def test_marker_before_three_dots_still_attached(self):
        assert clean_script("Hello [A]...") == "Hello..."

    def test_lone_carriage_return_marker_line_is_dropped(self):
        assert strip_script_markers("Hello\r[A]\rWorld") == ("Hello\rWorld", 1)

    def test_lone_carriage_return_before_text_still_joins_cleanly(self):
        assert clean_script("Hello\r[A] World") == "Hello\rWorld"


# 3b. Only the spacing at a removal point changes. Everything else is untouched.
class TestLocalTidy:
    def test_space_before_punctuation_elsewhere_is_kept(self):
        assert clean_script("Wait : what [B-ROLL: x] ok.") == "Wait : what ok."

    def test_question_and_exclamation_spacing_elsewhere_is_kept(self):
        assert clean_script("Quoi ? Oui ! [B-ROLL: x] fin") == "Quoi ? Oui ! fin"

    def test_comma_spacing_elsewhere_is_kept(self):
        assert clean_script("Hello , world [x]") == "Hello , world"

    def test_indentation_elsewhere_is_kept(self):
        text = "  indented line\n[B-ROLL: x]\n    more indented\n"
        assert clean_script(text) == "  indented line\n    more indented\n"

    def test_blank_lines_elsewhere_are_kept(self):
        assert clean_script("A\n\n\n\nB [x] C") == "A\n\n\n\nB C"

    def test_marker_line_between_blank_lines_leaves_one_blank_line(self):
        assert clean_script("Line one.\n\n[B-ROLL: x]\n\n\nLine two.") == "Line one.\n\nLine two."

    def test_marker_at_text_start_before_indented_line(self):
        assert clean_script("[B-ROLL: x]\n  Indented start.") == "  Indented start."


# 2b. A marker between matching quote marks leaves no stray space between them.
class TestQuotes:
    def test_double_quotes_around_marker(self):
        assert clean_script('He said "[x]" ok.') == 'He said "" ok.'

    def test_single_quotes_around_marker(self):
        assert clean_script("He said '[x]' ok.") == "He said '' ok."

    def test_curly_quotes_around_marker(self):
        assert clean_script("He said “[x]” ok.") == "He said “” ok."

    def test_marker_elsewhere_keeps_adjacent_quoted_words(self):
        assert clean_script('a "b" [x] "c" d') == 'a "b" "c" d'

    def test_marker_inside_quoted_pair_among_other_quotes(self):
        assert clean_script('"a" "[x]" "b"') == '"a" "" "b"'

    def test_text_with_quotes_and_no_brackets_unchanged(self):
        assert clean_script('"a" "b"') == '"a" "b"'


# 4. Long-form draft step: markers in several sections give exactly one log line.
@patch("verticals.draft.research_topic")
@patch("verticals.draft.call_llm")
def test_long_form_draft_logs_once_with_total_across_sections(mock_llm, mock_research):
    mock_research.return_value = "research"
    mock_llm.return_value = json.dumps({
        "broll_subject": "yawn",
        "sections": [
            {"id": "s1", "heading": "A", "narration": "[B-ROLL: a] One. [B-ROLL: b] Two.",
             "broll_subject": "yawn", "broll_prompts": ["p"]},
            {"id": "s2", "heading": "B", "narration": "Plain section with nothing to remove.",
             "broll_subject": "yawn", "broll_prompts": ["p"]},
            {"id": "s3", "heading": "C", "narration": "Three [B-ROLL: c] here.",
             "broll_subject": "yawn", "broll_prompts": ["p"]},
            {"id": "s4", "heading": "D", "narration": "[B-ROLL: d] Four.\n[B-ROLL: e]\nFive.",
             "broll_subject": "yawn", "broll_prompts": ["p"]},
        ],
        "shorts_cutpoints": [],
    })
    with patch("verticals.script_clean.log") as mock_log:
        draft = generate_long_form_draft("Yawning")
    assert [s["narration"] for s in draft["sections"]] == [
        "One. Two.", "Plain section with nothing to remove.", "Three here.", "Four.\nFive."]
    # 2 + 0 + 1 + 2 = 5 markers, one line for the whole draft step.
    assert mock_log.call_args_list == [call(MARKER_LOG.format(5))]


# 3. Long-form: an older draft whose section narration still has markers.
def test_long_form_old_draft_with_markers_is_cleaned_everywhere(tmp_path):
    from verticals.section_timing import compute_section_timings

    sections = [
        {"id": "s1", "heading": "Hook", "narration": "[B-ROLL: person yawning]\nYawns spread fast.",
         "broll_subject": "yawn", "broll_prompts": ["a yawn"]},
        {"id": "s2", "heading": "Why", "narration": "Yawning is [B-ROLL: crowd] contagious.",
         "broll_subject": "yawn", "broll_prompts": ["crowd"]},
    ]
    draft = {
        "job_id": "888", "niche": "general", "news": "Yawning", "platform": "long_form",
        "youtube_title": "Yawning", "broll_subject": "yawn", "sections": sections,
    }
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    args = argparse.Namespace(draft=str(draft_path), lang="en", force=False, voice=None, script=None)

    # Fake Whisper: one word per script word, one second apart, so timings exist.
    def fake_captions(vo_path, work_dir, lang, **kwargs):
        words = [{"word": w, "start": float(i), "end": float(i) + 0.9}
                 for i, w in enumerate(kwargs["script_text"].split())]
        return {"srt_path": "", "ass_path": "", "words": words}

    def fake_voiceover(script, out_dir, lang_, provider=None, voice_config=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "vo.mp3"
        path.write_bytes(b"x")
        return path

    def fake_assemble(**kwargs):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"x")
        return path

    voice = MagicMock(side_effect=fake_voiceover)
    captions = MagicMock(side_effect=fake_captions)
    broll = MagicMock(return_value=[])
    timings = MagicMock(side_effect=compute_section_timings)

    with patch.object(M, "MEDIA_DIR", tmp_path / "media"), \
         patch("verticals.tts.generate_voiceover", voice), \
         patch("verticals.captions.generate_captions", captions), \
         patch("verticals.broll.generate_broll_long_form", broll), \
         patch("verticals.music.select_and_prepare_music",
               MagicMock(return_value={"track_path": "", "duck_filter": ""})), \
         patch("verticals.assemble.assemble_video", MagicMock(side_effect=fake_assemble)), \
         patch("verticals.section_timing.compute_section_timings", timings), \
         patch("verticals.script_clean.log") as mock_log:
        M.cmd_produce_long_form(args)

    clean_full = "Yawns spread fast.\n\nYawning is contagious."
    assert voice.call_args.args[0] == clean_full
    assert captions.call_args.kwargs["script_text"] == clean_full

    # The timing and b-roll code both receive the cleaned narration.
    timed_sections = timings.call_args.args[0]
    assert [s["narration"] for s in timed_sections] == ["Yawns spread fast.", "Yawning is contagious."]
    assert [s["narration"] for s in broll.call_args.args[0]] == [
        "Yawns spread fast.", "Yawning is contagious."]

    # Word counts agree with the spoken words.
    assert [len(s["narration"].split()) for s in timed_sections] == [3, 3]

    # Exactly one "Removed N" line for the whole long-form produce, N = total markers.
    assert mock_log.call_args_list == [call(MARKER_LOG.format(2))]

    # The stored draft keeps its original narration (cleaned copies were used).
    stored = json.loads(draft_path.read_text(encoding="utf-8"))
    assert stored["sections"][0]["narration"].startswith("[B-ROLL: person yawning]")


# 5. Long-form: an older draft with a section that is only a marker.
def test_long_form_marker_only_section_is_dropped_everywhere(tmp_path, capsys):
    from verticals.section_timing import compute_section_timings

    sections = [
        {"id": "s1", "heading": "Hook", "narration": "Yawns spread fast.",
         "broll_subject": "yawn", "broll_prompts": ["a yawn"]},
        {"id": "s2", "heading": "Gap", "narration": "[B-ROLL: crowd]",
         "broll_subject": "yawn", "broll_prompts": ["crowd"]},
        {"id": "s3", "heading": "Why", "narration": "Yawning is contagious.",
         "broll_subject": "yawn", "broll_prompts": ["yawn"]},
    ]
    draft = {
        "job_id": "889", "niche": "general", "news": "Yawning", "platform": "long_form",
        "youtube_title": "Yawning", "broll_subject": "yawn", "total_words": 99,
        "sections": sections,
    }
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    args = argparse.Namespace(draft=str(draft_path), lang="en", force=False, voice=None, script=None)

    def fake_captions(vo_path, work_dir, lang, **kwargs):
        words = [{"word": w, "start": float(i), "end": float(i) + 0.9}
                 for i, w in enumerate(kwargs["script_text"].split())]
        return {"srt_path": "", "ass_path": "", "words": words}

    def fake_voiceover(script, out_dir, lang_, provider=None, voice_config=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "vo.mp3"
        path.write_bytes(b"x")
        return path

    def fake_assemble(**kwargs):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"x")
        return path

    voice = MagicMock(side_effect=fake_voiceover)
    captions = MagicMock(side_effect=fake_captions)
    broll = MagicMock(return_value=[])
    timings = MagicMock(side_effect=compute_section_timings)

    with patch.object(M, "MEDIA_DIR", tmp_path / "media"), \
         patch("verticals.tts.generate_voiceover", voice), \
         patch("verticals.captions.generate_captions", captions), \
         patch("verticals.broll.generate_broll_long_form", broll), \
         patch("verticals.music.select_and_prepare_music",
               MagicMock(return_value={"track_path": "", "duck_filter": ""})), \
         patch("verticals.assemble.assemble_video", MagicMock(side_effect=fake_assemble)), \
         patch("verticals.section_timing.compute_section_timings", timings), \
         patch("verticals.script_clean.log") as mock_log:
        M.cmd_produce_long_form(args)
    printed = capsys.readouterr().out

    # No empty paragraph in the joined script or the captions.
    clean_full = "Yawns spread fast.\n\nYawning is contagious."
    assert voice.call_args.args[0] == clean_full
    assert captions.call_args.kwargs["script_text"] == clean_full

    # Timings and b-roll get the reduced section list, with no empty narration.
    timed = timings.call_args.args[0]
    assert [s["narration"] for s in timed] == ["Yawns spread fast.", "Yawning is contagious."]
    assert [s["id"] for s in timed] == ["s1", "s3"]
    assert [s["narration"] for s in broll.call_args.args[0]] == [
        "Yawns spread fast.", "Yawning is contagious."]

    # Timing starts are distinct (no duplicate start_idx).
    words = [{"word": w, "start": float(i), "end": float(i) + 0.9}
             for i, w in enumerate(clean_full.split())]
    starts = [t["start_idx"] for t in compute_section_timings(timed, words)]
    assert len(set(starts)) == len(starts)

    # The printed word count is the cleaned one (6), not the stored total (99).
    assert "2 sections, 6 words" in printed
    # Check only the word-count line, so a temp path elsewhere in the output
    # cannot make this fail.
    count_line = next(line for line in printed.splitlines() if "sections," in line)
    assert "99" not in count_line

    assert mock_log.call_args_list == [call(MARKER_LOG.format(1))]

    # The stored draft is unchanged.
    stored = json.loads(draft_path.read_text(encoding="utf-8"))
    assert len(stored["sections"]) == 3
    assert stored["sections"][1]["narration"] == "[B-ROLL: crowd]"


# 6. Long-form produce: the "nothing to produce" error.
def _run_long_form_with_sections(tmp_path, sections):
    """Run cmd_produce_long_form with every heavy stage mocked. Returns the mocks."""
    draft = {
        "job_id": "890", "niche": "general", "news": "Yawning", "platform": "long_form",
        "youtube_title": "Yawning", "broll_subject": "yawn", "sections": sections,
    }
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    args = argparse.Namespace(draft=str(draft_path), lang="en", force=False, voice=None, script=None)

    def fake_captions(vo_path, work_dir, lang, **kwargs):
        return {"srt_path": "", "ass_path": "", "words": []}

    def fake_voiceover(script, out_dir, lang_, provider=None, voice_config=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "vo.mp3"
        path.write_bytes(b"x")
        return path

    def fake_assemble(**kwargs):
        path = tmp_path / "video.mp4"
        path.write_bytes(b"x")
        return path

    mocks = {
        "voice": MagicMock(side_effect=fake_voiceover),
        "captions": MagicMock(side_effect=fake_captions),
        "broll": MagicMock(return_value=[]),
        "music": MagicMock(return_value={"track_path": "", "duck_filter": ""}),
        "assemble": MagicMock(side_effect=fake_assemble),
    }
    with patch.object(M, "MEDIA_DIR", tmp_path / "media"), \
         patch("verticals.tts.generate_voiceover", mocks["voice"]), \
         patch("verticals.captions.generate_captions", mocks["captions"]), \
         patch("verticals.broll.generate_broll_long_form", mocks["broll"]), \
         patch("verticals.music.select_and_prepare_music", mocks["music"]), \
         patch("verticals.assemble.assemble_video", mocks["assemble"]), \
         patch("verticals.script_clean.log"):
        M.cmd_produce_long_form(args)
    return mocks


def test_long_form_all_marker_only_does_not_call_stages(tmp_path):
    sections = [
        {"id": "s1", "heading": "A", "narration": "[B-ROLL: a]", "broll_subject": "x", "broll_prompts": []},
        {"id": "s2", "heading": "B", "narration": "[B-ROLL: b] [B-ROLL: c]", "broll_subject": "x", "broll_prompts": []},
    ]
    voice = MagicMock()
    captions = MagicMock()
    broll = MagicMock()
    assemble = MagicMock()
    draft = {"job_id": "891", "niche": "general", "news": "n", "platform": "long_form",
             "youtube_title": "t", "broll_subject": "x", "sections": sections}
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    args = argparse.Namespace(draft=str(draft_path), lang="en", force=False, voice=None, script=None)
    with patch.object(M, "MEDIA_DIR", tmp_path / "media"), \
         patch("verticals.tts.generate_voiceover", voice), \
         patch("verticals.captions.generate_captions", captions), \
         patch("verticals.broll.generate_broll_long_form", broll), \
         patch("verticals.music.select_and_prepare_music", MagicMock()), \
         patch("verticals.assemble.assemble_video", assemble), \
         patch("verticals.script_clean.log"):
        with pytest.raises(ValueError, match="nothing to produce"):
            M.cmd_produce_long_form(args)
    voice.assert_not_called()
    captions.assert_not_called()
    broll.assert_not_called()
    assemble.assert_not_called()


def test_long_form_single_normal_section_does_not_raise(tmp_path):
    sections = [
        {"id": "s1", "heading": "A", "narration": "Just one normal section here.",
         "broll_subject": "x", "broll_prompts": ["p"]},
    ]
    mocks = _run_long_form_with_sections(tmp_path, sections)
    assert mocks["voice"].call_args.args[0] == "Just one normal section here."
    mocks["assemble"].assert_called_once()


# 7. Hindi danda, symbols attached to a removal, and straight-quote pairing.
class TestHindiSymbolsAndQuotePairing:
    def test_hindi_danda_attached_after_marker_joins(self):
        assert clean_script("\u092f\u0939 \u0939\u0948 [B-ROLL: x]\u0964") == "\u092f\u0939 \u0939\u0948\u0964"

    def test_hindi_danda_with_space_before_stays_separate(self):
        assert clean_script("\u092f\u0939 \u0939\u0948 [B-ROLL: x] \u0964") == "\u092f\u0939 \u0939\u0948 \u0964"

    def test_double_danda_attached_after_marker_joins(self):
        assert clean_script("\u0913\u092e [B-ROLL: x]\u0965") == "\u0913\u092e\u0965"

    def test_hindi_script_with_markers_is_cleaned(self):
        text = "[B-ROLL: person yawning]\n\u092f\u0939 \u090f\u0915 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0939\u0948\u0964 [B-ROLL: crowd] \u0905\u0902\u0924\u0964"
        assert clean_script(text) == "\u092f\u0939 \u090f\u0915 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0939\u0948\u0964 \u0905\u0902\u0924\u0964"

    def test_percent_attached_joins(self):
        assert clean_script("50 [x]%") == "50%"

    def test_percent_with_space_stays_separate(self):
        assert clean_script("50 [x] %") == "50 %"

    def test_em_dash_attached_joins(self):
        assert clean_script("wait [x]\u2014what") == "wait\u2014what"

    def test_en_dash_attached_joins(self):
        assert clean_script("wait [x]\u2013what") == "wait\u2013what"

    def test_dash_with_space_stays_separate(self):
        assert clean_script("wait [x] \u2014 what") == "wait \u2014 what"

    def test_straight_quote_after_marker_is_opening_when_no_quote_is_open(self):
        assert clean_script('He said [x]"...no"') == 'He said "...no"'

    def test_straight_single_quote_after_marker_is_opening_when_none_open(self):
        assert clean_script("He said [x]'...no'") == "He said '...no'"

    def test_straight_quote_after_marker_closes_an_open_quote(self):
        assert clean_script('say "hi [x]" ok') == 'say "hi" ok'

    def test_straight_quote_after_marker_opens_after_a_closed_pair(self):
        assert clean_script('"a" [x]"b" ') == '"a" "b" '

    def test_quote_pairing_word_count_property(self):
        cleaned = clean_script('He said [x]"...no"')
        assert cleaned.split() == ["He", "said", '"...no"']


class TestApostropheInsideWord:
    def test_apostrophe_in_word_does_not_count_as_open_quote(self):
        assert clean_script("don't [x]' hi") == "don't ' hi"

    def test_apostrophe_word_then_quoted_word(self):
        assert clean_script("don't [x]'hi'") == "don't 'hi'"
