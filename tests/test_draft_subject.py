"""Tests for the broll_subject wording in the draft prompts (mocked model, no API calls)."""

import json
from unittest.mock import patch

from verticals.draft import generate_draft, generate_long_form_draft


# The long-form prompt still uses the original wording. These must stay exactly as they are.
LONGFORM_TOP_LEVEL_LINE = (
    '  "broll_subject": "the literal main visual subject in 1-2 words '
    '(e.g. octopus, cat, volcano, black hole)",'
)
LONGFORM_SECTION_LINE = '      "broll_subject": "main visual subject for this section, 1-2 words",'
LONGFORM_REUSE_RULE = (
    '- Reuse the video\'s main subject (the top-level broll_subject) as the\n'
    '  broll_subject for EVERY section, unless that section is about something\n'
    '  genuinely different from it, such as a different animal, a lab, or a\n'
    '  machine (1-2 words, e.g. "research laboratory").'
)


def _shorts_payload(**overrides):
    payload = {
        "script": "A made-up script about skin.",
        "broll_prompts": ["goosebumps on skin close up", "hair standing up on an arm"],
        "youtube_title": "T",
        "youtube_description": "D",
        "youtube_tags": "t",
        "instagram_caption": "C",
        "tiktok_caption": "C",
        "thumbnail_prompt": "P",
        "broll_subject": "goosebumps on skin",
    }
    payload.update(overrides)
    return payload


def _longform_payload(section_subject=None):
    section = {
        "id": "s1",
        "heading": "Made up heading",
        "narration": "Made up narration.",
        "broll_prompts": ["goosebumps on skin close up"],
    }
    if section_subject is not None:
        section["broll_subject"] = section_subject
    return {
        "youtube_title": "T",
        "youtube_description": "D",
        "youtube_tags": "t",
        "thumbnail_prompt": "P",
        "broll_subject": "goosebumps on skin",
        "sections": [section],
        "shorts_cutpoints": [],
    }


class TestShortsBrollSubjectPrompt:
    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft._call_claude")
    def _shorts_prompt(self, mock_claude, mock_research, mock_record):
        mock_claude.return_value = json.dumps(_shorts_payload())
        generate_draft("why do we get goosebumps")
        return mock_claude.call_args[0][0]

    def test_asks_for_two_to_four_word_visual_phrase(self):
        prompt = self._shorts_prompt()
        assert "2-4 word concrete visual search phrase" in prompt
        assert "what the camera would actually see" in prompt

    def test_gives_goosebumps_on_skin_example_and_single_word_warning(self):
        prompt = self._shorts_prompt()
        assert "goosebumps on skin" in prompt
        assert "Never a single word on its own" in prompt
        assert "rather than goosebumps" in prompt

    def test_no_old_one_to_two_word_wording_or_example_list(self):
        prompt = self._shorts_prompt()
        assert "1-2 words" not in prompt
        assert "(e.g. octopus, cat, volcano, black hole)" not in prompt

    def test_every_example_phrase_is_two_to_four_words(self):
        prompt = self._shorts_prompt()
        line = next(l for l in prompt.splitlines() if l.strip().startswith('"broll_subject": "a 2-4 word'))
        examples_text = line.split("(e.g. ", 1)[1].split(")", 1)[0]
        examples = [e.strip() for e in examples_text.split(",")]
        assert len(examples) >= 3
        for example in examples:
            assert 2 <= len(example.split()) <= 4, f"example {example!r} is not 2-4 words"

    def test_subject_line_sits_inside_the_json_output_block(self):
        prompt = self._shorts_prompt()
        output_start = prompt.index("Output JSON exactly:")
        subject_pos = prompt.index('"broll_subject": "a 2-4 word')
        prompts_pos = prompt.index('"broll_prompts"', output_start)
        assert output_start < subject_pos < prompts_pos


class TestLongFormPromptUnchanged:
    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft.call_llm")
    def _longform_prompt(self, mock_llm, mock_research, mock_record):
        mock_llm.return_value = json.dumps(_longform_payload())
        generate_long_form_draft("why do we get goosebumps")
        return mock_llm.call_args[0][0]

    def test_keeps_original_top_level_line(self):
        assert LONGFORM_TOP_LEVEL_LINE in self._longform_prompt()

    def test_keeps_original_section_line(self):
        assert LONGFORM_SECTION_LINE in self._longform_prompt()

    def test_keeps_original_reuse_rule(self):
        assert LONGFORM_REUSE_RULE in self._longform_prompt()

    def test_does_not_get_new_wording(self):
        prompt = self._longform_prompt()
        assert "2-4 word concrete visual search phrase" not in prompt
        assert "Never a single word on its own" not in prompt


class TestBrollSubjectHandling:
    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft._call_claude")
    def test_two_to_four_word_phrase_is_kept_as_is(self, mock_claude, mock_research, mock_record):
        mock_claude.return_value = json.dumps(_shorts_payload(broll_subject="goosebumps on skin"))
        draft = generate_draft("why do we get goosebumps")
        assert draft["broll_subject"] == "goosebumps on skin"

    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft._call_claude")
    def test_missing_broll_subject_is_not_added(self, mock_claude, mock_research, mock_record):
        payload = _shorts_payload()
        del payload["broll_subject"]
        mock_claude.return_value = json.dumps(payload)
        draft = generate_draft("why do we get goosebumps")
        assert "broll_subject" not in draft

    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft._call_claude")
    def test_empty_broll_subject_stays_empty(self, mock_claude, mock_research, mock_record):
        mock_claude.return_value = json.dumps(_shorts_payload(broll_subject=""))
        draft = generate_draft("why do we get goosebumps")
        assert draft["broll_subject"] == ""

    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft._call_claude")
    def test_non_string_broll_subject_is_still_a_string(self, mock_claude, mock_research, mock_record):
        mock_claude.return_value = json.dumps(_shorts_payload(broll_subject=["goosebumps", "skin"]))
        draft = generate_draft("why do we get goosebumps")
        assert isinstance(draft["broll_subject"], str)

    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft.call_llm")
    def test_longform_section_without_subject_falls_back_to_top_level(self, mock_llm, mock_research, mock_record):
        mock_llm.return_value = json.dumps(_longform_payload(section_subject=None))
        draft = generate_long_form_draft("why do we get goosebumps")
        assert draft["sections"][0]["broll_subject"] == "goosebumps on skin"

    @patch("verticals.draft.record_variation")
    @patch("verticals.draft.research_topic", return_value="research")
    @patch("verticals.draft.call_llm")
    def test_longform_section_with_own_subject_is_kept(self, mock_llm, mock_research, mock_record):
        mock_llm.return_value = json.dumps(_longform_payload(section_subject="research laboratory"))
        draft = generate_long_form_draft("why do we get goosebumps")
        assert draft["sections"][0]["broll_subject"] == "research laboratory"
