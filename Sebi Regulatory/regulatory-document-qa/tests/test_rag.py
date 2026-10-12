import unittest
from src.rag import clean_text, chunk_text


class TextProcessingTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text(" RBI   rules\n\n apply "), "RBI rules apply")

    def test_empty_chunks(self):
        self.assertEqual(chunk_text("  "), [])

    def test_overlap(self):
        chunks = chunk_text("one two three four five six seven eight", 5, 2)
        self.assertEqual(chunks[0], "one two three four five")
        self.assertEqual(chunks[1], "four five six seven eight")

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", 4, 4)


if __name__ == "__main__":
    unittest.main()
