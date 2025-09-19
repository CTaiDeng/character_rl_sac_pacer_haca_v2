import unittest

from src.train_demo import (
    ArticleEnvironment,
    CharTokenizer,
    TextAction,
    _combine_summary_and_chapter,
)


class PreviousSummaryPenaltyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chapters = [
            "第一章内容摘要。",
            "第二章详细描述当前情节。",
        ]
        self.tokenizer = CharTokenizer(self.chapters)
        self.previous_summary = "第一章内容摘要。"
        self.next_chapter = self.chapters[1]

    def _make_action(self, text: str) -> TextAction:
        return TextAction(
            token_ids=self.tokenizer.encode_action_text(text),
            text=text,
            length=len(text),
        )

    def _prepare_environment(self) -> ArticleEnvironment:
        env = ArticleEnvironment(self.chapters, tokenizer=self.tokenizer)
        env.reset()
        env._current_summary = self.previous_summary
        env._cursor = 1
        env._last_metrics = {}
        return env

    def test_missing_previous_summary_is_penalized(self) -> None:
        without_prev_env = self._prepare_environment()
        only_chapter_action = self._make_action(self.next_chapter)
        without_prev_reward = without_prev_env.step(only_chapter_action).reward
        without_prev_metrics = without_prev_env.last_metrics

        with_prev_env = self._prepare_environment()
        combined_text = _combine_summary_and_chapter(
            self.previous_summary, self.next_chapter
        )
        combined_action = self._make_action(combined_text)
        with_prev_reward = with_prev_env.step(combined_action).reward
        with_prev_metrics = with_prev_env.last_metrics

        self.assertGreater(with_prev_reward, without_prev_reward)
        self.assertLess(without_prev_metrics["coverage_ratio"], with_prev_metrics["coverage_ratio"])
        self.assertLess(without_prev_metrics["similarity"], with_prev_metrics["similarity"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
