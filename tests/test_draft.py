"""Tests for pipeline/draft.py — draft generation with mocked Claude API."""

import json
from unittest.mock import patch, MagicMock

from verticals.draft import generate_draft, FALLBACK_BROLL_PROMPT_COUNT, MAX_BROLL_PROMPTS


class TestGenerateDraft:
    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_basic_draft_generation(self, mock_claude, mock_research):
        mock_research.return_value = "Some research data about the topic."
        mock_claude.return_value = json.dumps({
            "script": "This is a test script about AI.",
            "broll_prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
            "youtube_title": "AI Revolution 2026",
            "youtube_description": "All about AI.",
            "youtube_tags": "ai,tech,2026",
            "instagram_caption": "AI is changing the world!",
            "thumbnail_prompt": "Futuristic AI image",
        })

        draft = generate_draft("AI is changing everything in 2026")

        assert draft["script"] == "This is a test script about AI."
        assert len(draft["broll_prompts"]) == 3
        assert draft["youtube_title"] == "AI Revolution 2026"
        assert draft["news"] == "AI is changing everything in 2026"
        assert draft["research"] == "Some research data about the topic."

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_handles_code_block_wrapper(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = '```json\n{"script":"test","broll_prompts":["p1","p2","p3"],"youtube_title":"T","youtube_description":"D","youtube_tags":"t","instagram_caption":"C","thumbnail_prompt":"P"}\n```'

        draft = generate_draft("Test topic")
        assert draft["script"] == "test"

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_sanitizes_non_string_fields(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({
            "script": 12345,  # non-string
            "broll_prompts": "not a list",  # non-list
            "youtube_title": "T",
            "youtube_description": "D",
            "youtube_tags": "t",
            "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })

        draft = generate_draft("Test")
        assert isinstance(draft["script"], str)
        assert isinstance(draft["broll_prompts"], list)
        assert len(draft["broll_prompts"]) == FALLBACK_BROLL_PROMPT_COUNT  # fallback list

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_includes_channel_context(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({
            "script": "s", "broll_prompts": ["p1", "p2", "p3"],
            "youtube_title": "T", "youtube_description": "D",
            "youtube_tags": "t", "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })

        draft = generate_draft("Test", channel_context="esports news channel")
        # Verify the channel context was passed to Claude
        call_args = mock_claude.call_args_list[0][0][0]  # the draft prompt (the fact check call comes after it)
        assert "esports news channel" in call_args

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_truncates_broll_prompts(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({
            "script": "s",
            "broll_prompts": [f"p{i}" for i in range(MAX_BROLL_PROMPTS + 5)],  # too many
            "youtube_title": "T", "youtube_description": "D",
            "youtube_tags": "t", "instagram_caption": "C",
            "thumbnail_prompt": "P",
        })

        draft = generate_draft("Test")
        assert len(draft["broll_prompts"]) == MAX_BROLL_PROMPTS  # truncated to the maximum


class TestMusicMood:
    BASE = {
        "script": "A short test script.",
        "broll_prompts": ["p1", "p2", "p3"],
        "youtube_title": "T", "youtube_description": "D",
        "youtube_tags": "t", "instagram_caption": "C",
        "thumbnail_prompt": "P",
    }

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_valid_mood_is_saved_normalised(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({**self.BASE, "music_mood": "  Hopeful "})

        draft = generate_draft("Test")
        assert draft["music_mood"] == "hopeful"

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_draft_without_music_mood_is_accepted(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps(self.BASE)

        draft = generate_draft("Test")
        assert draft["script"] == "A short test script."
        assert draft["music_mood"] is None

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_invalid_mood_is_accepted_as_none(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({**self.BASE, "music_mood": "jazzy"})

        draft = generate_draft("Test")
        assert draft["script"] == "A short test script."
        assert draft["music_mood"] is None

    @patch("verticals.draft.research_topic")
    @patch("verticals.draft._call_claude")
    def test_prompt_asks_for_music_mood(self, mock_claude, mock_research):
        mock_research.return_value = "research"
        mock_claude.return_value = json.dumps({**self.BASE, "music_mood": "cosmic"})

        generate_draft("Test")
        prompt = mock_claude.call_args_list[0][0][0]
        assert '"music_mood": "cosmic, mysterious, hopeful or quirky"' in prompt
        assert "MUSIC MOOD" in prompt
