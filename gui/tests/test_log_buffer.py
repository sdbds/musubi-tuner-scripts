import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils.log_buffer import LogBuffer  # noqa: E402


class TestLogBuffer(unittest.TestCase):
    def test_retains_more_than_previous_default_limit(self):
        buffer = LogBuffer()

        for index in range(6001):
            buffer.push(f"line {index}")

        lines = buffer.get_all_lines()
        self.assertEqual(len(lines), 6001)
        self.assertEqual(lines[0], (1, "line 0"))
        self.assertEqual(lines[-1], (6001, "line 6000"))

    def test_clear_empties_history_without_resetting_sequence(self):
        buffer = LogBuffer()
        seen = []
        buffer.subscribe(lambda seq, line: seen.append((seq, line)))

        buffer.push("before")
        buffer.clear()
        buffer.push("after")

        self.assertEqual(buffer.get_all_lines(), [(2, "after")])
        self.assertEqual(seen, [(1, "before"), (2, "after")])


if __name__ == "__main__":
    unittest.main()
