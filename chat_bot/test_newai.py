from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import newai
from docx import Document
from openpyxl import Workbook


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

    def test_txt_upload_is_used_but_not_saved_to_history(self):
        with patch.object(
            newai.Aura, "answer_directly", return_value="It says hello."
        ) as answer_directly:
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "What does this say?",
                    "document": (BytesIO(b"Hello from a file."), "note.txt"),
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"response": "It says hello."})
        answer_directly.assert_called_once_with(
            "What does this say?",
            [newai.ChatAttachment("note.txt", text="Hello from a file.")],
        )
        memory = newai.SimpleMemory(newai.DB_PATH)
        try:
            self.assertEqual(
                memory.recent_history(),
                [
                    {"role": "user", "content": "What does this say?"},
                    {"role": "assistant", "content": "It says hello."},
                ],
            )
        finally:
            memory.close()

    def test_pdf_upload_extracts_text(self):
        page = Mock()
        page.extract_text.return_value = "PDF document text."
        reader = Mock(is_encrypted=False, pages=[page])

        with (
            patch.object(newai, "PdfReader", return_value=reader),
            patch.object(newai.Aura, "answer_directly", return_value="Answer."),
        ):
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "Summarize this.",
                    "document": (BytesIO(b"%PDF-test"), "document.pdf"),
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"response": "Answer."})

    def test_upload_rejects_unsupported_files(self):
        response = self.client.post(
            "/api/chat",
            data={
                "message": "Read this.",
                "document": (BytesIO(b"data"), "document.unsupported"),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json["error"])

    def test_python_file_is_read_as_text(self):
        with patch.object(
            newai.Aura, "answer_directly", return_value="This prints hello."
        ) as answer_directly:
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "What does this code do?",
                    "documents": (BytesIO(b'print("hello")'), "example.py"),
                },
            )

        self.assertEqual(response.status_code, 200)
        answer_directly.assert_called_once_with(
            "What does this code do?",
            [newai.ChatAttachment("example.py", text='print("hello")')],
        )

    def test_xlsx_upload_extracts_cells(self):
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Sales"
        worksheet.append(["Product", "Units"])
        worksheet.append(["Apples", 12])
        content = BytesIO()
        workbook.save(content)

        with patch.object(
            newai.Aura, "answer_directly", return_value="There are 12 apples."
        ) as answer_directly:
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "How many apples?",
                    "documents": (BytesIO(content.getvalue()), "sales.xlsx"),
                },
            )

        self.assertEqual(response.status_code, 200)
        attachment = answer_directly.call_args.args[1][0]
        self.assertEqual(attachment.filename, "sales.xlsx")
        self.assertIn("Sheet: Sales", attachment.text)
        self.assertIn("Apples\t12", attachment.text)

    def test_docx_upload_extracts_paragraphs_and_tables(self):
        document = Document()
        document.add_paragraph("Quarterly report")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Revenue"
        table.cell(0, 1).text = "$100"
        content = BytesIO()
        document.save(content)

        with patch.object(
            newai.Aura, "answer_directly", return_value="Revenue is $100."
        ) as answer_directly:
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "What is the revenue?",
                    "documents": (BytesIO(content.getvalue()), "report.docx"),
                },
            )

        self.assertEqual(response.status_code, 200)
        attachment = answer_directly.call_args.args[1][0]
        self.assertEqual(attachment.filename, "report.docx")
        self.assertIn("Quarterly report", attachment.text)
        self.assertIn("Revenue\t$100", attachment.text)

    def test_image_upload_requires_vision_model_setting(self):
        response = self.client.post(
            "/api/chat",
            data={
                "message": "Describe this.",
                "documents": (BytesIO(b"image-bytes"), "photo.png"),
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Set GROQ_VISION_MODEL", response.json["error"])

    def test_image_upload_uses_configured_vision_model(self):
        with (
            patch.object(newai, "GROQ_VISION_MODEL", "vision-model"),
            patch.object(newai.Aura, "answer_directly", return_value="A photo.")
            as answer_directly,
        ):
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "Describe this.",
                    "documents": (BytesIO(b"image-bytes"), "photo.png"),
                },
            )

        self.assertEqual(response.status_code, 200)
        attachment = answer_directly.call_args.args[1][0]
        self.assertEqual(attachment.filename, "photo.png")
        self.assertTrue(attachment.data_url.startswith("data:image/png;base64,"))

    def test_image_and_text_attachments_use_multimodal_vision_request(self):
        memory = newai.SimpleMemory(str(Path(self.temp_dir.name) / "vision.db"))
        try:
            aura = newai.Aura(memory)
            with (
                patch.object(newai, "GROQ_VISION_MODEL", "vision-model"),
                patch.object(aura, "_complete", return_value="Answer.") as complete,
            ):
                answer = aura.answer_directly(
                    "Explain these files.",
                    [
                        newai.ChatAttachment("script.py", text="print('hello')"),
                        newai.ChatAttachment(
                            "photo.png", data_url="data:image/png;base64,aGVsbG8="
                        ),
                    ],
                )

            self.assertEqual(answer, "Answer.")
            messages = complete.call_args.args[0]
            user_content = messages[-1]["content"]
            self.assertIn("script.py", user_content[1]["text"])
            self.assertEqual(user_content[2]["type"], "image_url")
            self.assertEqual(complete.call_args.kwargs["model"], "vision-model")
        finally:
            memory.close()

    def test_upload_rejects_non_utf8_text(self):
        response = self.client.post(
            "/api/chat",
            data={
                "message": "Read this.",
                "document": (BytesIO(b"\xff"), "document.txt"),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json, {"error": "document.txt must use UTF-8 encoding."}
        )

    def test_upload_rejects_extracted_text_over_limit(self):
        response = self.client.post(
            "/api/chat",
            data={
                "message": "Read this.",
                "document": (
                    BytesIO(b"a" * (newai.MAX_DOCUMENT_CHARACTERS + 1)),
                    "document.txt",
                ),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json,
            {
                "error": (
                    "Extracted file text must be 100,000 characters or less "
                    "per file."
                )
            },
        )


if __name__ == "__main__":
    unittest.main()
