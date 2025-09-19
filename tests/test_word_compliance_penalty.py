import unittest

from src.train_demo import (
    ArticleEnvironment,
    CharTokenizer,
    TextAction,
    analyze_summary,
)


class WordCompliancePenaltyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chapter = "你好世界，这是一段测试文本。"
        self.tokenizer = CharTokenizer([self.chapter])
        self.environment = ArticleEnvironment([self.chapter], tokenizer=self.tokenizer)

    def test_analyze_summary_reports_word_penalty(self) -> None:
        summary = "你好世界"
        metrics = analyze_summary(
            summary,
            self.chapter,
            tokenizer=self.tokenizer,
            word_checker=self.environment.word_checker,
            chapter_text=self.chapter,
        )
        self.assertIn("word_noncompliance_ratio", metrics)
        self.assertAlmostEqual(metrics["word_noncompliance_ratio"], 0.0, places=6)

        scrambled = "界世你好"
        scrambled_metrics = analyze_summary(
            scrambled,
            self.chapter,
            tokenizer=self.tokenizer,
            word_checker=self.environment.word_checker,
            chapter_text=self.chapter,
        )
        self.assertGreater(scrambled_metrics["word_noncompliance_ratio"], 0.0)
        self.assertAlmostEqual(
            scrambled_metrics["word_noncompliance_ratio"],
            scrambled_metrics["word_penalty"],
            places=6,
        )

    def test_environment_penalizes_noncompliant_wording(self) -> None:
        clean_text = "你好世界"
        scrambled_text = "界世你好"

        clean_action = TextAction(
            token_ids=self.tokenizer.encode_action_text(clean_text),
            text=clean_text,
            length=len(clean_text),
        )
        scrambled_action = TextAction(
            token_ids=self.tokenizer.encode_action_text(scrambled_text),
            text=scrambled_text,
            length=len(scrambled_text),
        )

        self.environment.reset()
        clean_reward = self.environment.step(clean_action).reward

        self.environment.reset()
        scrambled_reward = self.environment.step(scrambled_action).reward

        self.assertLess(scrambled_reward, clean_reward)
        metrics = self.environment.last_metrics
        self.assertGreater(metrics["word_noncompliance_ratio"], 0.0)
        self.assertGreater(metrics["word_penalty"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
