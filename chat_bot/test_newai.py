import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import newai


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory = newai.SimpleMemory(str(Path(self.temp_dir.name) / "test.db"))
        self.old_aura = newai.aura
        newai.aura = newai.Aura(self.memory)
        self.client = newai.app.test_client()
        with newai._rate_lock:
            newai._rate_events.clear()

    def tearDown(self):
        newai.aura = self.old_aura
        self.memory.close()
        self.temp_dir.cleanup()
        with newai._rate_lock:
            newai._rate_events.clear()

    def test_health_check_is_json(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_chat_page_is_available(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AURA", response.data)

    def test_chat_saves_complete_exchange(self):
        with patch.object(newai.Aura, "answer_directly", return_value="Hello"):
            response = self.client.post("/api/chat", json={"message": "Hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"response": "Hello"})
        self.assertEqual(
            self.memory.recent_history(),
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )

    def test_chat_rejects_invalid_json_and_non_text_messages(self):
        invalid_json = self.client.post(
            "/api/chat", data="not json", content_type="application/json"
        )
        non_text = self.client.post("/api/chat", json={"message": ["hello"]})
        empty = self.client.post("/api/chat", json={"message": "  "})
        self.assertEqual(invalid_json.status_code, 400)
        self.assertEqual(non_text.status_code, 400)
        self.assertEqual(empty.status_code, 400)

    def test_chat_rejects_oversized_input(self):
        response = self.client.post(
            "/api/chat", json={"message": "x" * (newai.MAX_INPUT_CHARS + 1)}
        )
        self.assertEqual(response.status_code, 413)

    def test_chat_rate_limit_is_enforced(self):
        with (
            patch.object(newai, "CHAT_RATE_LIMIT", 1),
            patch.object(newai.Aura, "answer_directly", return_value="Hello"),
        ):
            first = self.client.post("/api/chat", json={"message": "Hi"})
            second = self.client.post("/api/chat", json={"message": "Again"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_model_failure_returns_service_unavailable_without_provider_details(self):
        with patch.object(
            newai.Aura,
            "answer_directly",
            side_effect=newai.ModelUnavailable("private provider detail"),
        ):
            response = self.client.post("/api/chat", json={"message": "Hi"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"private provider detail", response.data)

    def test_groq_failure_falls_back_to_ollama(self):
        with (
            patch.object(newai, "GROQ_API_KEY", "configured"),
            patch.object(newai.Aura, "_complete_with_groq", side_effect=RuntimeError("failure")),
            patch.object(newai.Aura, "_complete_with_ollama", return_value="Ollama reply") as ollama,
        ):
            answer = newai.Aura(self.memory)._complete([])
        self.assertEqual(answer, "Ollama reply")
        ollama.assert_called_once()

    def test_both_model_providers_fail_with_sanitized_error(self):
        with (
            patch.object(newai, "GROQ_API_KEY", "configured"),
            patch.object(newai.Aura, "_complete_with_groq", side_effect=RuntimeError("secret detail")),
            patch.object(newai.Aura, "_complete_with_ollama", side_effect=RuntimeError("private detail")),
        ):
            with self.assertRaises(newai.ModelUnavailable) as error:
                newai.Aura(self.memory)._complete([])
        self.assertNotIn("secret detail", str(error.exception))
        self.assertNotIn("private detail", str(error.exception))

    def test_context_contains_recent_user_and_assistant_turns(self):
        self.memory.record_exchange("Question", "Answer")
        messages = newai.Aura(self.memory)._build_messages("Next")
        self.assertEqual(messages[-3:], [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Next"},
        ])

    def test_history_cap_prunes_whole_turns(self):
        with patch.object(newai, "MAX_HISTORY_RECORDS", 4):
            for index in range(4):
                self.memory.record_exchange(f"q{index}", f"a{index}")
        history = self.memory.recent_history(limit=10)
        self.assertEqual(len(history), 4)
        self.assertEqual([item["content"] for item in history], ["q2", "a2", "q3", "a3"])

    def test_clear_command_leaves_history_empty(self):
        self.memory.record_exchange("old question", "old answer")
        answer = newai.Aura(self.memory).answer_and_record("/clear")
        self.assertEqual(answer, "Conversation history cleared.")
        self.assertEqual(self.memory.recent_history(), [])


if __name__ == "__main__":
    unittest.main()
