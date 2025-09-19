import unittest

from src.train_demo import (
    ArticleEnvironment,
    CharTokenizer,
    TextAction,
    analyze_summary,
)


class GarbledPenaltyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chapter = "你好世界，这是一段测试文本。"
        self.tokenizer = CharTokenizer([self.chapter])
        self.environment = ArticleEnvironment([self.chapter], tokenizer=self.tokenizer)

    def test_analyze_summary_reports_garbled_ratio(self) -> None:
        summary = "<unk>摘录"
        metrics = analyze_summary(
            summary,
            self.chapter,
            tokenizer=self.tokenizer,
            word_checker=self.environment.word_checker,
            chapter_text=self.chapter,
        )
        self.assertGreater(metrics["garbled_ratio"], 0.0)
        self.assertAlmostEqual(
            metrics["garbled_ratio"], metrics["garbled_penalty"], places=6
        )
        self.assertGreater(metrics["unk_char_ratio"], 0.0)

    def test_environment_penalizes_garbled_text(self) -> None:
        clean_text = "你好世界"
        garbled_text = "<unk>你好世界"

        clean_action = TextAction(
            token_ids=self.tokenizer.encode_action_text(clean_text),
            text=clean_text,
            length=len(clean_text),
        )
        garbled_action = TextAction(
            token_ids=self.tokenizer.encode_action_text(garbled_text),
            text=garbled_text,
            length=len(garbled_text),
        )

        self.environment.reset()
        clean_reward = self.environment.step(clean_action).reward

        self.environment.reset()
        garbled_transition = self.environment.step(garbled_action)
        garbled_reward = garbled_transition.reward

        self.assertLess(garbled_reward, clean_reward)
        metrics = self.environment.last_metrics
        self.assertGreater(metrics["garbled_ratio"], 0.0)
        self.assertGreater(metrics["garbled_penalty"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
