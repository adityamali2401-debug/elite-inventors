import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.ai_service import AIServiceError
from backend.main import app


REPOSITORY_URL = "https://github.com/octo/demo"
REPOSITORY_CONTEXT = {
    "repository": {"full_name": "octo/demo", "html_url": REPOSITORY_URL},
    "readme": {"available": True, "path": "README.md", "content": "Demo project README."},
    "root_contents": [{"path": "README.md", "name": "README.md", "type": "file"}],
    "languages": {"Python": 1000},
    "issues": [{"number": 7, "title": "Improve docs", "state": "open", "html_url": f"{REPOSITORY_URL}/issues/7", "body": "Add a documented example.", "labels": []}],
    "warnings": [],
}


class AIRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
        self.context_patch = patch("backend.main._get_ai_repository_context", new=AsyncMock(return_value=REPOSITORY_CONTEXT))
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_all_ai_routes_use_the_new_payload_contract(self):
        with (
            patch("backend.main.ai_service.analyze_repository", new=AsyncMock(return_value={"model": "local-gemma", "analysis": {"project_overview": "Demo."}})),
            patch("backend.main.ai_service.explain_issue", new=AsyncMock(return_value={"model": "local-gemma", "explanation": {"simple_explanation": "Improve docs."}})),
            patch("backend.main.ai_service.chat", new=AsyncMock(return_value={"model": "local-gemma", "answer": "The README describes the project."})) as chat,
        ):
            analyze = await self.client.post("/api/analyze", json={"repository_url": REPOSITORY_URL})
            explain = await self.client.post("/api/explain-issue", json={"repository_url": REPOSITORY_URL, "issue_number": 7})
            chat_response = await self.client.post(
                "/api/chat",
                json={"repository_url": REPOSITORY_URL, "question": "What is this?", "history": [{"role": "user", "content": "Hello"}]},
            )

        self.assertEqual(analyze.status_code, 200)
        self.assertEqual(analyze.json()["model"], "local-gemma")
        self.assertEqual(explain.status_code, 200)
        self.assertEqual(explain.json()["explanation"]["simple_explanation"], "Improve docs.")
        self.assertEqual(chat_response.status_code, 200)
        self.assertEqual(chat.await_args.args[2], [{"role": "user", "content": "Hello"}])

    async def test_ai_unavailable_error_does_not_change_github_context_route(self):
        with patch("backend.main.ai_service.analyze_repository", new=AsyncMock(side_effect=AIServiceError("Ollama is unavailable."))):
            response = await self.client.post("/api/analyze", json={"repository_url": REPOSITORY_URL})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Ollama is unavailable.")

    async def test_issue_must_be_present_in_recent_backend_github_results(self):
        with patch("backend.main.ai_service.explain_issue", new=AsyncMock()) as explain:
            response = await self.client.post("/api/explain-issue", json={"repository_url": REPOSITORY_URL, "issue_number": 404})
        self.assertEqual(response.status_code, 404)
        explain.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
