import base64
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from backend.ai_service import AIServiceError, OllamaAIService
from backend.main import _fetch_contribution_guidelines, app


REPOSITORY_URL = "https://github.com/octo/demo"
ISSUE = {
    "number": 27,
    "title": "Add parser validation",
    "state": "open",
    "html_url": f"{REPOSITORY_URL}/issues/27",
    "body": "Reject invalid parser input and add a regression test.",
    "labels": [{"name": "help wanted", "color": "008672"}],
}
GUIDE_TEXT = "# Contributing\nCreate a focused branch before you work.\nRun tests with python -m pytest."
REPOSITORY_CONTEXT = {
    "repository": {
        "full_name": "octo/demo",
        "name": "demo",
        "html_url": REPOSITORY_URL,
        "default_branch": "main",
    },
    "readme": {
        "available": True,
        "name": "README.md",
        "path": "README.md",
        "html_url": f"{REPOSITORY_URL}/blob/main/README.md",
        "content": "# Demo\nInstall instructions are documented in the contributing guide.",
        "truncated": False,
    },
    "contribution_guidelines": [{
        "available": True,
        "name": "CONTRIBUTING.md",
        "path": "CONTRIBUTING.md",
        "html_url": f"{REPOSITORY_URL}/blob/main/CONTRIBUTING.md",
        "content": GUIDE_TEXT,
        "truncated": False,
    }],
    "root_contents": [
        {"path": "README.md", "name": "README.md", "type": "file", "html_url": f"{REPOSITORY_URL}/blob/main/README.md"},
        {"path": "CONTRIBUTING.md", "name": "CONTRIBUTING.md", "type": "file", "html_url": f"{REPOSITORY_URL}/blob/main/CONTRIBUTING.md"},
        {"path": "src/parser.py", "name": "parser.py", "type": "file", "html_url": f"{REPOSITORY_URL}/blob/main/src/parser.py"},
        {"path": "tests/test_parser.py", "name": "test_parser.py", "type": "file", "html_url": f"{REPOSITORY_URL}/blob/main/tests/test_parser.py"},
    ],
    "languages": {"Python": 1200},
    "issues": [ISSUE],
    "warnings": [],
}


def roadmap_step(key, *, paths=None, commands=None, quotes=None):
    return {
        "key": key,
        "guidance": f"For issue #27, review the evidence before completing {key.replace('_', ' ')}.",
        "required": True,
        "optional_reason": "",
        "evidence_quotes": quotes or ["Reject invalid parser input"],
        "source_paths": paths or [],
        "relevant_files": [],
        "commands": commands or [],
    }


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


def complete_steps():
    keys = [
        "understand_issue", "expected_result", "locate_files", "read_guidelines",
        "setup_environment", "create_branch", "investigate", "implement",
        "run_tests", "review_diff", "prepare_pull_request",
    ]
    steps = [roadmap_step(key) for key in keys]
    steps[2] = roadmap_step(
        "locate_files",
        paths=["src/parser.py", "src/not-real.py"],
    )
    steps[3] = roadmap_step("read_guidelines", paths=["CONTRIBUTING.md"])
    steps[5] = roadmap_step(
        "create_branch",
        commands=[
            {
                "command": "git switch -c fix/parser-validation",
                "explanation": "Creates and switches to a new local branch; suggested only.",
                "evidence_quote": "Create a focused branch",
                "source_path": "General Git guidance",
            },
            {
                "command": "git reset --hard main",
                "explanation": "Destructive command; must never be suggested.",
                "evidence_quote": "Create a focused branch",
                "source_path": "CONTRIBUTING.md",
            },
        ],
        quotes=["Create a focused branch"],
    )
    steps[8] = roadmap_step(
        "run_tests",
        commands=[
            {
                "command": "python -m pytest",
                "explanation": "Runs the test tool named in the guide.",
                "evidence_quote": "Run tests with python -m pytest.",
                "source_path": "CONTRIBUTING.md",
            },
            {
                "command": "python invented_test_runner.py",
                "explanation": "Not present in project documentation.",
                "evidence_quote": "Run tests with python -m pytest.",
                "source_path": "CONTRIBUTING.md",
            },
        ],
        paths=["tests/test_parser.py"],
        quotes=["add a regression test"],
    )
    steps[8]["required"] = False
    steps[8]["optional_reason"] = "The guide does not specify a test command."
    return steps


class ContributionRoadmapServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_issue_specific_roadmap_and_filters_unsupported_commands_and_paths(self):
        output = {
            "steps": complete_steps(),
            "missing_information": ["The issue does not specify the exact error message."],
            "assumptions_to_verify": ["Confirm how invalid parser input is represented."],
        }
        client = FakeOllamaClient("gemma-configured:tag", output)
        with patch.dict(os.environ, {"GEMMA_MODEL": "gemma-configured:tag"}), patch("backend.ai_service.AsyncClient", return_value=client):
            result = await OllamaAIService().generate_contribution_roadmap(
                REPOSITORY_CONTEXT,
                ISSUE,
                {"known_languages": ["Python"], "experience_level": "beginner", "skills_description": "Learning tests."},
            )

        self.assertEqual(result["model"], "gemma-configured:tag")
        self.assertEqual(result["issue"]["number"], 27)
        self.assertEqual(len(result["steps"]), 11)
        self.assertEqual(result["steps"][0]["key"], "understand_issue")
        self.assertTrue(result["steps"][0]["required"])
        self.assertTrue(result["steps"][8]["optional_reason"])
        locate = result["steps"][2]
        self.assertEqual(locate["source_paths"], ["src/parser.py"])
        self.assertEqual(locate["source_links"][0]["url"], f"{REPOSITORY_URL}/blob/main/src/parser.py")
        branch_commands = result["steps"][5]["commands"]
        self.assertEqual([item["command"] for item in branch_commands], ["git switch -c fix/parser-validation"])
        self.assertEqual(branch_commands[0]["status"], "suggested; not executed")
        self.assertTrue(result["steps"][5]["safety_warnings"])
        test_commands = result["steps"][8]["commands"]
        self.assertEqual([item["command"] for item in test_commands], ["python -m pytest"])
        self.assertTrue(all("reset --hard" not in item["command"] for step in result["steps"] for item in step["commands"]))
        self.assertTrue(any(link["label"] == "CONTRIBUTING.md" for link in result["source_links"]))
        self.assertIn("did not run", result["execution_notice"])
        prompt = client.chat_arguments["messages"][1]["content"]
        self.assertIn("CONTRIBUTING.md", prompt)
        self.assertIn('"number": 27', prompt)

    async def test_missing_model_raises_ai_service_error(self):
        with patch.dict(os.environ, {"GEMMA_MODEL": ""}), self.assertRaisesRegex(AIServiceError, "GEMMA_MODEL"):
            await OllamaAIService().generate_contribution_roadmap(REPOSITORY_CONTEXT, ISSUE, {})

    async def test_contribution_guides_are_discovered_only_from_listed_repository_directories(self):
        class ClientContext:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        guide_content = base64.b64encode(b"Create a focused branch.").decode("ascii")

        async def fake_github_get(_client, api_path, *, params=None, max_bytes=0):
            if api_path == "/repos/octo/demo/contents/.github":
                return [{"name": "CONTRIBUTING.md", "path": ".github/CONTRIBUTING.md", "type": "file"}]
            if api_path == "/repos/octo/demo/contents/docs":
                return [
                    {"name": "CONTRIBUTING.md", "path": "../CONTRIBUTING.md", "type": "file"},
                    {"name": "guide.txt", "path": "docs/guide.txt", "type": "file"},
                ]
            if api_path == "/repos/octo/demo/contents/.github/CONTRIBUTING.md":
                return {"name": "CONTRIBUTING.md", "path": ".github/CONTRIBUTING.md", "html_url": f"{REPOSITORY_URL}/blob/main/.github/CONTRIBUTING.md", "encoding": "base64", "content": guide_content}
            self.fail(f"Unexpected or unverified contribution-guide path requested: {api_path}")

        with (
            patch("backend.main.httpx.AsyncClient", return_value=ClientContext()),
            patch("backend.main._github_get", new=AsyncMock(side_effect=fake_github_get)) as github_get,
        ):
            guides, warnings = await _fetch_contribution_guidelines(
                ClientContext(),
                "/repos/octo/demo",
                [
                    {"name": ".github", "path": ".github", "type": "dir"},
                    {"name": "docs", "path": "docs", "type": "dir"},
                ],
            )
        self.assertEqual(warnings, [])
        self.assertEqual(len(guides), 1)
        self.assertEqual(guides[0]["path"], ".github/CONTRIBUTING.md")
        self.assertEqual(guides[0]["content"], "Create a focused branch.")
        self.assertEqual(github_get.await_count, 3)


class ContributionRoadmapRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
        self.context_patch = patch("backend.main._get_ai_repository_context", new=AsyncMock(return_value=REPOSITORY_CONTEXT))
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_route_passes_selected_issue_and_profile_to_generator(self):
        expected = {"model": "gemma-configured:tag", "issue": ISSUE, "steps": []}
        with patch("backend.main.ai_service.generate_contribution_roadmap", new=AsyncMock(return_value=expected)) as generate:
            response = await self.client.post("/api/roadmap", json={
                "repository_url": REPOSITORY_URL,
                "issue_number": 27,
                "known_languages": ["Python"],
                "experience_level": "beginner",
                "skills_description": "I have written a few scripts.",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(generate.await_args.args[1]["number"], 27)
        self.assertEqual(generate.await_args.args[2]["experience_level"], "beginner")

    async def test_closed_or_unselected_issue_is_not_sent_to_model(self):
        context = {**REPOSITORY_CONTEXT, "issues": [{**ISSUE, "state": "closed"}]}
        with patch("backend.main._get_ai_repository_context", new=AsyncMock(return_value=context)), patch("backend.main.ai_service.generate_contribution_roadmap", new=AsyncMock()) as generate:
            response = await self.client.post("/api/roadmap", json={"repository_url": REPOSITORY_URL, "issue_number": 27})
        self.assertEqual(response.status_code, 404)
        generate.assert_not_awaited()

    async def test_invalid_experience_and_extra_fields_are_rejected(self):
        response = await self.client.post("/api/roadmap", json={
            "repository_url": REPOSITORY_URL,
            "issue_number": 27,
            "experience_level": "expert-plus",
        })
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
