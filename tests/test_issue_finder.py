import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from backend.ai_service import AIServiceError, OllamaAIService
from backend.main import (
    _open_issues,
    app,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTAPI_ISSUE_FIXTURE = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "github_fastapi_issue.json").read_text(encoding="utf-8"))
REPO_URL = "https://github.com/fastapi/fastapi"


def repository_context(issues=None):
    return {
        "repository": {"full_name": "octo/demo", "html_url": "https://github.com/octo/demo"},
        "readme": {
            "available": True,
            "path": "README.md",
            "html_url": "https://github.com/octo/demo/blob/main/README.md",
            "content": "# Demo\nThe parser validates contributor input and renders a report.",
        },
        "root_contents": [
            {"path": "README.md", "name": "README.md", "type": "file", "html_url": "https://github.com/octo/demo/blob/main/README.md"},
            {"path": "src/parser.py", "name": "parser.py", "type": "file", "html_url": "https://github.com/octo/demo/blob/main/src/parser.py"},
        ],
        "languages": {"Python": 1200, "JavaScript": 300},
        "issues": list(issues or []),
        "warnings": [],
    }


def candidate(number, title, body, labels=None):
    return {
        "number": number,
        "title": title,
        "state": "open",
        "html_url": f"https://github.com/octo/demo/issues/{number}",
        "body": body,
        "labels": [{"name": name, "color": "7057ff"} for name in (labels or [])],
    }


def assessment(number, *, difficulty="beginner", confidence=4, skills=4, clarity=4, scope=4, evidence=4, quote=""):
    return {
        "issue_number": number,
        "difficulty_estimate": difficulty,
        "difficulty_confidence": confidence,
        "skills_needed": ["Python"],
        "skill_compatibility_score": skills,
        "task_clarity_score": clarity,
        "scope_score": scope,
        "evidence_score": evidence,
        "reasons": ["The issue describes a focused change."],
        "not_suitable_reasons": ["Some project-specific context may still be needed."],
        "evidence_quotes": [quote] if quote else [],
        "relevant_files": [
            {"path": "src/parser.py", "reason": "The issue mentions parser validation.", "evidence_quote": quote}
        ] if quote else [],
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
        return {"models": [{"model": self.model_name}]}

    async def chat(self, **kwargs):
        self.chat_arguments = kwargs
        return {"message": {"content": json.dumps(self.output)}}


class GitHubIssueFixtureTests(unittest.TestCase):
    def test_captured_real_github_issue_response_passes_strict_filter(self):
        filtered = _open_issues({"items": [FASTAPI_ISSUE_FIXTURE]}, "fastapi", "fastapi")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["number"], 10370)
        self.assertEqual(filtered[0]["html_url"], "https://github.com/fastapi/fastapi/issues/10370")
        self.assertEqual(filtered[0]["state"], "open")
        self.assertEqual(filtered[0]["labels"], [])

    def test_controlled_invalid_closed_and_pull_request_records_are_excluded(self):
        records = [
            candidate(1, "Valid issue", "A short valid task."),
            {**candidate(2, "Closed issue", "No longer open."), "state": "closed"},
            {**candidate(3, "PR", "A pull request."), "pull_request": {"url": "https://api.github.com/repos/octo/demo/pulls/3"}},
            {**candidate(4, "Spoof URL", "Do not trust the link."), "html_url": "https://github.com.evil.test/octo/demo/issues/4"},
            {"number": 0, "title": "Invalid number", "state": "open", "html_url": "https://github.com/octo/demo/issues/0"},
            {"number": 6, "title": "", "state": "open", "html_url": "https://github.com/octo/demo/issues/6"},
        ]
        filtered = _open_issues({"items": records}, "octo", "demo")
        self.assertEqual([item["number"] for item in filtered], [1])


class IssueFinderServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ranks_model_assessments_but_does_not_treat_label_as_easy_proof(self):
        issue_with_label = candidate(10, "Broad refactor", "This changes parser behavior across multiple modules.", ["good first issue"])
        unlabeled_issue = candidate(11, "Document a concrete parser example", "The README says the parser validates contributor input and renders a report.")
        repository = repository_context([issue_with_label, unlabeled_issue])
        model_output = {
            "assessments": [
                assessment(10, difficulty="advanced", confidence=4, skills=1, clarity=2, scope=1, evidence=4, quote="changes parser behavior across multiple modules"),
                assessment(11, difficulty="beginner", confidence=5, skills=5, clarity=5, scope=5, evidence=5, quote="parser validates contributor input"),
                assessment(999, difficulty="beginner", quote="fabricated issue number"),
            ]
        }
        fake_client = FakeOllamaClient("gemma-installed:tag", model_output)
        with patch.dict(os.environ, {"GEMMA_MODEL": "gemma-installed:tag"}), patch("backend.ai_service.AsyncClient", return_value=fake_client):
            result = await OllamaAIService().recommend_issues(
                repository,
                [issue_with_label, unlabeled_issue],
                {"known_languages": ["Python"], "experience_level": "beginner", "preferred_contribution_type": "documentation", "skills_description": "I have read Python tutorials."},
            )

        recommendations = result["recommendations"]
        self.assertEqual([item["issue"]["number"] for item in recommendations], [11, 10])
        labeled = recommendations[1]
        self.assertEqual(labeled["difficulty_estimate"], "advanced")
        self.assertEqual(labeled["github_label_signals"], ["good first issue"])
        self.assertIn("not proof", labeled["label_note"])
        self.assertEqual(labeled["fit_status"], "unlikely-match")
        self.assertEqual(recommendations[0]["fit_status"], "likely-match")
        self.assertEqual(recommendations[0]["relevant_files"][0]["path"], "src/parser.py")
        self.assertNotIn(999, [item["issue"]["number"] for item in recommendations])
        self.assertEqual(result["ranking_method"]["score_range"], "0–100; each component is Gemma-estimated on a 1–5 scale.")
        self.assertEqual(fake_client.chat_arguments["model"], "gemma-installed:tag")

    async def test_unmatched_or_unsupported_quotes_force_uncertain_difficulty(self):
        issue = candidate(12, "Underspecified task", "Please improve this sometime.", ["beginner"])
        fake_client = FakeOllamaClient("gemma-installed:tag", {"assessments": [assessment(12, difficulty="beginner", confidence=5, quote="A fabricated evidence quote.")]})
        with patch.dict(os.environ, {"GEMMA_MODEL": "gemma-installed:tag"}), patch("backend.ai_service.AsyncClient", return_value=fake_client):
            result = await OllamaAIService().recommend_issues(repository_context([issue]), [issue], {"known_languages": []})
        recommendation = result["recommendations"][0]
        self.assertEqual(recommendation["difficulty_estimate"], "uncertain")
        self.assertEqual(recommendation["evidence_quotes"], [])
        self.assertEqual(recommendation["score_breakdown"]["evidence"], 1)
        self.assertEqual(recommendation["github_label_signals"], ["beginner"])

    async def test_no_valid_candidates_does_not_call_ollama(self):
        with patch("backend.ai_service.AsyncClient") as client:
            result = await OllamaAIService().recommend_issues(repository_context(), [], {})
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["suitable_count"], 0)
        client.assert_not_called()


class IssueFinderRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
        self.context = repository_context([
            candidate(10, "Broad change", "This changes parser behavior across modules.", ["good first issue"]),
            candidate(11, "Docs example", "Document the parser example.", []),
        ])
        self.context_patch = patch("backend.main._get_ai_repository_context", new=AsyncMock(return_value=self.context))
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_route_passes_preferences_and_returns_gemma_recommendations(self):
        expected = {"model": "gemma-installed:tag", "candidate_count": 2, "suitable_count": 1, "recommendations": []}
        with patch("backend.main.ai_service.recommend_issues", new=AsyncMock(return_value=expected)) as recommend:
            response = await self.client.post("/api/issues/recommend", json={
                "repository_url": "https://github.com/octo/demo",
                "known_languages": ["Python", "JavaScript"],
                "experience_level": "beginner",
                "contribution_type": "documentation",
                "skills_description": "I have used Python for small projects.",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "gemma-installed:tag")
        self.assertEqual(recommend.await_args.args[1][0]["number"], 10)
        self.assertEqual(recommend.await_args.args[2]["preferred_contribution_type"], "documentation")

    async def test_candidate_cap_keeps_unlabeled_issues_in_the_ai_assessment(self):
        self.context["issues"] = [candidate(number, f"Labeled task {number}", "A scoped change.", ["good first issue"]) for number in range(1, 19)]
        self.context["issues"].extend([
            candidate(100, "Unlabeled documentation task", "Document the example."),
            candidate(101, "Another unlabeled issue", "Clarify parser output."),
        ])
        with patch("backend.main.ai_service.recommend_issues", new=AsyncMock(return_value={"recommendations": [], "candidate_count": 15, "suitable_count": 0})) as recommend:
            response = await self.client.post("/api/issues/recommend", json={"repository_url": "https://github.com/octo/demo"})
        self.assertEqual(response.status_code, 200)
        assessed_numbers = [issue["number"] for issue in recommend.await_args.args[1]]
        self.assertEqual(len(assessed_numbers), 15)
        self.assertIn(100, assessed_numbers)
        self.assertIn(101, assessed_numbers)

    async def test_invalid_profile_values_are_rejected(self):
        response = await self.client.post("/api/issues/recommend", json={
            "repository_url": "https://github.com/octo/demo",
            "experience_level": "expert-plus",
        })
        self.assertEqual(response.status_code, 422)

    async def test_missing_gemma_keeps_real_github_issues_available_unranked(self):
        with patch("backend.main.ai_service.recommend_issues", new=AsyncMock(side_effect=AIServiceError("Set GEMMA_MODEL to the installed model."))):
            response = await self.client.post("/api/issues/recommend", json={"repository_url": "https://github.com/octo/demo"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("GEMMA_MODEL", response.json()["analysis_error"])
        self.assertEqual(response.json()["issues"][0]["html_url"], "https://github.com/octo/demo/issues/10")
        self.assertEqual(response.json()["recommendations"], [])

    async def test_refresh_fetches_only_issue_search_when_repo_context_is_cached(self):
        self.context_patch.stop()
        with (
            patch("backend.main._cached_repository_context", return_value=self.context),
            patch("backend.main._cache_repository_context") as cache_updated,
            patch("backend.main._fetch_open_issues", new=AsyncMock(return_value=[candidate(12, "Fresh issue", "Updated from GitHub.")])) as refresh,
            patch("backend.main.ai_service.recommend_issues", new=AsyncMock(return_value={"recommendations": [], "candidate_count": 1, "suitable_count": 0})) as recommend,
        ):
            response = await self.client.post("/api/issues/recommend", json={
                "repository_url": "https://github.com/octo/demo",
                "refresh": True,
            })
        self.assertEqual(response.status_code, 200)
        refresh.assert_awaited_once_with("octo", "demo")
        self.assertEqual(recommend.await_args.args[1][0]["number"], 12)
        self.assertEqual(cache_updated.call_args.args[2]["issues"][0]["number"], 12)


@unittest.skipUnless(os.getenv("REPORESCUE_LIVE_GITHUB_TESTS") == "1", "Set REPORESCUE_LIVE_GITHUB_TESTS=1 to make one live GitHub request.")
class LiveGitHubResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_fastapi_issue_search_response_is_accepted(self):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RepoRescue-AI-tests",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.github.com/search/issues",
                params={"q": "repo:fastapi/fastapi is:open is:issue", "sort": "created", "order": "desc", "per_page": 2},
                headers=headers,
            )
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            self.skipTest("GitHub rate limit was exhausted for this live integration test.")
        self.assertEqual(response.status_code, 200, response.text[:250])
        issues = _open_issues(response.json(), "fastapi", "fastapi")
        self.assertTrue(issues)
        self.assertTrue(all(issue["html_url"].startswith("https://github.com/fastapi/fastapi/issues/") for issue in issues))
        self.assertTrue(all(issue["state"] == "open" for issue in issues))


if __name__ == "__main__":
    unittest.main()
