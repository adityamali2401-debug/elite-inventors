import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.ai_service import AIServiceError, OllamaAIService


class FakeOllamaClient:
    def __init__(self, model_name, output):
        self.model_name = model_name
        self.output = output
        self.chat_arguments = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=self.model_name)])

    async def chat(self, **kwargs):
        self.chat_arguments = kwargs
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.output)))


class OllamaServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_configured_model_and_filters_unsupported_evidence(self):
        readme_text = "# Demo\nInstall with pip install demo."
        model_output = {
            "project_overview": "A beginner demo project.",
            "main_purpose": "Demonstrates a small Python project.",
            "technologies_used": [
                {"name": "Python", "evidence_quote": "Python"},
                {"name": "demo", "evidence_quote": "Install with pip install demo."},
                {"name": "UnlistedTool", "evidence_quote": "This quote is not in the README."},
            ],
            "important_files": [
                {"path": "setup.py", "purpose": "Project packaging configuration.", "evidence_quote": "setup.py"},
                {"path": "src/missing.py", "purpose": "Unsupported nested file.", "evidence_quote": "missing file"},
            ],
            "setup_instructions": [
                {"beginner_explanation": "Install the package using the documented pip command.", "documentation_quote": "Install with pip install demo."},
                {"beginner_explanation": "Run pytest.", "documentation_quote": "Run pytest."},
            ],
            "potential_contribution_areas": [
                {"suggestion": "Improve setup guidance.", "evidence_quote": "Install with pip install demo."},
                {"suggestion": "Add a new API endpoint.", "evidence_quote": "This phrase is not in the evidence."},
            ],
            "missing_information": ["The README does not describe a test command."],
            "hypotheses": ["A setup guide may help new contributors."],
        }
        model_tag = "gemma-custom:local-tag"
        fake_client = FakeOllamaClient(model_tag, model_output)
        repository = {
            "repository": {"name": "demo", "full_name": "octo/demo", "description": "Demo project", "html_url": "https://github.com/octo/demo"},
            "readme": {"available": True, "path": "README.md", "html_url": "https://github.com/octo/demo/blob/main/README.md", "content": readme_text},
            "root_contents": [
                {"name": "README.md", "path": "README.md", "type": "file", "html_url": "https://github.com/octo/demo/blob/main/README.md"},
                {"name": "setup.py", "path": "setup.py", "type": "file", "html_url": "https://github.com/octo/demo/blob/main/setup.py"},
            ],
            "languages": {"Python": 1000},
            "issues": [],
        }

        with patch.dict(os.environ, {"GEMMA_MODEL": model_tag}), patch("backend.ai_service.AsyncClient", return_value=fake_client):
            result = await OllamaAIService().analyze_repository(repository)

        analysis = result["analysis"]
        self.assertEqual(result["model"], model_tag)
        self.assertEqual([item["path"] for item in analysis["important_files"]], ["setup.py"])
        self.assertEqual(len(analysis["setup_instructions"]), 1)
        self.assertEqual(len(analysis["potential_contribution_areas"]), 1)
        self.assertEqual(analysis["technologies_used"][0]["name"], "Python")
        self.assertEqual([item["name"] for item in analysis["technologies_used"]], ["Python", "demo"])
        self.assertEqual(fake_client.chat_arguments["model"], model_tag)
        self.assertEqual(fake_client.chat_arguments["options"]["num_ctx"], 8192)

    async def test_missing_configured_model_returns_clear_error(self):
        with patch.dict(os.environ, {"GEMMA_MODEL": ""}), self.assertRaisesRegex(AIServiceError, "GEMMA_MODEL"):
            await OllamaAIService().analyze_repository({})

    async def test_uninstalled_model_tag_is_not_guessed_or_substituted(self):
        fake_client = FakeOllamaClient("gemma-other:tag", {})
        with patch.dict(os.environ, {"GEMMA_MODEL": "gemma-configured:tag"}), patch("backend.ai_service.AsyncClient", return_value=fake_client):
            with self.assertRaisesRegex(AIServiceError, "gemma-configured:tag"):
                await OllamaAIService().analyze_repository({})


if __name__ == "__main__":
    unittest.main()
