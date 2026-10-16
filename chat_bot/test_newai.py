import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import newai


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path_patch = patch.object(
            newai, "DB_PATH", str(Path(self.temp_dir.name) / "test.db")
        )
        self.db_path_patch.start()
        self.client = newai.app.test_client()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_health_check_is_public(self):
        response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_chat_page_is_public(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AURA", response.data)

    def test_chat_page_and_api(self):
        with patch.object(newai.Aura, "answer_directly", return_value="Hello"):
            response = self.client.post("/api/chat", json={"message": "Hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"response": "Hello"})
        memory = newai.SimpleMemory(newai.DB_PATH)
        try:
            self.assertEqual(
                memory.recent_history(),
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            )
        finally:
            memory.close()

    def test_chat_rejects_empty_message(self):
        response = self.client.post("/api/chat", json={"message": "  "})

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
