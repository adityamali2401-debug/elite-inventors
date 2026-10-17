import base64
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from backend.main import GitHubAPIError, app, parse_github_repository_url


class RepositoryUrlTests(unittest.TestCase):
    def test_accepts_only_canonical_repository_urls(self):
        self.assertEqual(parse_github_repository_url("https://github.com/octo/demo"), ("octo", "demo"))
        self.assertEqual(parse_github_repository_url("https://github.com/octo/demo.git/"), ("octo", "demo"))

    def test_rejects_non_github_and_ambiguous_urls(self):
        invalid_urls = (
            "http://github.com/octo/demo",
            "https://github.com.evil.test/octo/demo",
            "https://user@github.com/octo/demo",
            "https://github.com:443/octo/demo",
            "https://github.com/octo/demo/issues",
            "https://github.com/octo/demo?redirect=https://evil.test",
            "https://github.com/octo%2Fother/demo",
        )
        for value in invalid_urls:
            with self.subTest(value=value), self.assertRaises(HTTPException):
                parse_github_repository_url(value)


class RepositoryEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_returns_readme_languages_root_and_non_pr_issues(self):
        readme = base64.b64encode(b"# Demo project\n\nREADME body.").decode("ascii")
        metadata = {
            "id": 42,
            "name": "demo",
            "full_name": "octo/demo",
            "owner": {"login": "octo"},
            "description": "A demo repository",
            "html_url": "https://github.com/octo/demo",
            "private": False,
            "default_branch": "main",
            "stargazers_count": 12,
            "forks_count": 2,
            "open_issues_count": 11,
            "license": {"spdx_id": "MIT"},
        }
        api_issues = [
            {"number": 99, "title": "A pull request", "state": "open", "html_url": "https://github.com/octo/demo/pull/99", "pull_request": {}},
            {"number": 98, "title": "Closed issue", "state": "closed", "html_url": "https://github.com/octo/demo/issues/98"},
            {"number": 97, "title": "Spoofed URL", "state": "open", "html_url": "https://github.com.evil.test/octo/demo/issues/97"},
        ]
        api_issues.extend(
            {
                "number": number,
                "title": f"Issue {number}",
                "state": "open",
                "html_url": f"https://github.com/octo/demo/issues/{number}",
                "body": "Issue details",
                "labels": [{"name": "good first issue", "color": "7057ff"}],
            }
            for number in range(1, 13)
        )

        async def fake_github_get(_client, api_path, *, params=None, max_bytes=0):
            if api_path == "/repos/octo/demo":
                return metadata
            if api_path == "/repos/octo/demo/readme":
                return {"name": "README.md", "path": "README.md", "encoding": "base64", "content": readme}
            if api_path == "/repos/octo/demo/contents":
                return [{"name": "README.md", "path": "README.md", "type": "file", "size": 29}]
            if api_path == "/repos/octo/demo/languages":
                return {"Python": 1000, "HTML": 400}
            if api_path == "/search/issues":
                self.assertIn("is:open is:issue", params["q"])
                self.assertEqual(params["sort"], "created")
                return {"total_count": 12, "items": api_issues}
            self.fail(f"Unexpected GitHub API path: {api_path}")

        with patch("backend.main._github_get", new=AsyncMock(side_effect=fake_github_get)) as github_get:
            response = await self.client.get("/api/repository", params={"url": "https://github.com/octo/demo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["repository"]["full_name"], "octo/demo")
        self.assertEqual(payload["readme"]["content"], "# Demo project\n\nREADME body.")
        self.assertEqual(payload["languages"], {"Python": 1000, "HTML": 400})
        self.assertEqual(len(payload["root_contents"]), 1)
        self.assertEqual(len(payload["issues"]), 12)
        self.assertNotIn("pull_request", payload["issues"][0])
        self.assertTrue(all(issue["state"] == "open" for issue in payload["issues"]))
        self.assertEqual(github_get.await_count, 5)

    async def test_repository_without_readme_is_still_successful(self):
        metadata = {"name": "demo", "full_name": "octo/demo", "owner": {"login": "octo"}}

        async def fake_github_get(_client, api_path, *, params=None, max_bytes=0):
            if api_path == "/repos/octo/demo":
                return metadata
            if api_path == "/repos/octo/demo/readme":
                raise GitHubAPIError(404, "This repository is empty.")
            if api_path == "/repos/octo/demo/contents":
                return []
            if api_path == "/repos/octo/demo/languages":
                return {}
            if api_path == "/search/issues":
                return {"total_count": 0, "items": []}
            self.fail(f"Unexpected GitHub API path: {api_path}")

        with patch("backend.main._github_get", new=AsyncMock(side_effect=fake_github_get)):
            response = await self.client.get("/api/repository", params={"url": "https://github.com/octo/demo"})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["readme"]["available"])
        self.assertEqual(response.json()["readme"]["content"], "")
        self.assertEqual(response.json()["warnings"], [])

    async def test_rejects_external_url_without_calling_github(self):
        with patch("backend.main._github_get", new=AsyncMock()) as github_get:
            response = await self.client.get("/api/repository", params={"url": "https://example.com/octo/demo"})
        self.assertEqual(response.status_code, 422)
        github_get.assert_not_awaited()

    async def test_reports_github_rate_limit_clearly(self):
        with patch(
            "backend.main._github_get",
            new=AsyncMock(side_effect=GitHubAPIError(403, "rate limit", rate_limited=True, retry_after="60")),
        ):
            response = await self.client.get("/api/repository", params={"url": "https://github.com/octo/demo"})
        self.assertEqual(response.status_code, 429)
        self.assertIn("rate limit", response.json()["detail"].lower())
        self.assertEqual(response.headers["retry-after"], "60")


if __name__ == "__main__":
    unittest.main()
