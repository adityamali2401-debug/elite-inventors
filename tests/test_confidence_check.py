"""Tests for the RepoRescue Confidence Check feature.

Covers:
  - Missing evidence (no README, no evidence quotes, no file paths)
  - Invalid or spoofed issue URLs
  - Unsupported file path claims (AI-inferred paths not in root listing)
  - Full evidence (all criteria met → Verified Source)
  - Partial evidence (some criteria met → Partially Supported)
  - Score explanation is always present and does not describe a probability
  - Verification steps are always returned
  - Integration: confidence_check is included in recommend_issues output
"""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.ai_service import OllamaAIService, compute_confidence_check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_NAME = "octo/demo"
REPO_URL = f"https://github.com/{FULL_NAME}"


def _issue(number=1, body="Reject invalid parser input and return a clear error.", url=None):
    return {
        "number": number,
        "title": "Fix parser validation",
        "state": "open",
        "html_url": url or f"{REPO_URL}/issues/{number}",
        "body": body,
        "labels": [],
    }


def _repository_data(
    full_name=FULL_NAME,
    readme_available=True,
    readme_url=None,
    root_paths=None,
):
    return {
        "repository": {
            "full_name": full_name,
            "html_url": f"https://github.com/{full_name}",
        },
        "readme": {
            "available": readme_available,
            "path": "README.md",
            "html_url": readme_url or (
                f"https://github.com/{full_name}/blob/main/README.md"
                if readme_available else None
            ),
            "content": "# Demo\nThe parser validates contributor input.",
            "truncated": False,
        },
        "root_contents": [
            {"path": p, "name": p.split("/")[-1], "type": "file"}
            for p in (root_paths or ["README.md", "src/parser.py"])
        ],
        "languages": {"Python": 1200},
        "issues": [],
        "contribution_guidelines": [],
        "warnings": [],
    }


def _source_texts(repository_data):
    readme = repository_data.get("readme", {})
    return [str(readme.get("content") or "")]


def _root_paths(repository_data):
    return {
        str(entry.get("path"))
        for entry in repository_data.get("root_contents", [])
        if entry.get("path")
    }


# ---------------------------------------------------------------------------
# Unit tests: compute_confidence_check
# ---------------------------------------------------------------------------


class ConfidenceCheckAllCriteriaMetTests(unittest.TestCase):
    """All five criteria are met → score = 100 → Verified Source."""

    def setUp(self):
        self.issue = _issue()
        self.repo = _repository_data()
        self.source_texts = _source_texts(self.repo)
        self.root_paths = _root_paths(self.repo)
        # Evidence quote is present verbatim in the README content
        self.evidence_quotes = ["parser validates contributor input"]
        self.relevant_files = [
            {"path": "src/parser.py", "reason": "Handles parsing logic."}
        ]

    def _run(self):
        return compute_confidence_check(
            issue=self.issue,
            repository_data=self.repo,
            evidence_quotes=self.evidence_quotes,
            relevant_files=self.relevant_files,
            source_texts=self.source_texts,
            root_paths=self.root_paths,
        )

    def test_score_is_100_when_all_criteria_are_met(self):
        result = self._run()
        self.assertEqual(result["score"], 100)

    def test_status_label_is_verified_source(self):
        result = self._run()
        self.assertEqual(result["status_label"], "Verified Source")

    def test_all_criteria_are_true(self):
        result = self._run()
        for key, value in result["criteria"].items():
            self.assertTrue(value, f"Criterion '{key}' should be met")

    def test_verified_file_paths_contains_confirmed_path(self):
        result = self._run()
        self.assertIn("src/parser.py", result["verified_file_paths"])

    def test_inferred_file_paths_is_empty(self):
        result = self._run()
        self.assertEqual(result["inferred_file_paths"], [])

    def test_supported_evidence_quotes_is_populated(self):
        result = self._run()
        self.assertTrue(result["supported_evidence_quotes"])

    def test_verification_steps_always_present(self):
        result = self._run()
        self.assertIsInstance(result["verification_steps"], list)
        self.assertGreater(len(result["verification_steps"]), 0)

    def test_score_explanation_does_not_claim_probability(self):
        result = self._run()
        explanation = result["score_explanation"].lower()
        self.assertNotIn("probability", explanation)
        self.assertNotIn("chance", explanation)
        self.assertNotIn("likelihood", explanation)

    def test_missing_information_is_empty_when_all_criteria_met(self):
        result = self._run()
        self.assertEqual(result["missing_information"], [])


# ---------------------------------------------------------------------------
# Missing evidence tests
# ---------------------------------------------------------------------------


class MissingEvidenceTests(unittest.TestCase):
    """Various missing evidence scenarios."""

    def _run(self, issue, repo, evidence_quotes=None, relevant_files=None):
        return compute_confidence_check(
            issue=issue,
            repository_data=repo,
            evidence_quotes=evidence_quotes or [],
            relevant_files=relevant_files or [],
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo),
        )

    def test_no_readme_lowers_score_and_marks_criterion_false(self):
        repo = _repository_data(readme_available=False)
        result = self._run(_issue(), repo)
        self.assertFalse(result["criteria"]["readme_available"])
        self.assertLess(result["score"], 100)

    def test_no_readme_adds_to_missing_information(self):
        repo = _repository_data(readme_available=False)
        result = self._run(_issue(), repo)
        missing_lower = [m.lower() for m in result["missing_information"]]
        self.assertTrue(any("readme" in m for m in missing_lower))

    def test_no_evidence_quotes_forces_evidence_agreement_false(self):
        repo = _repository_data()
        result = self._run(_issue(), repo, evidence_quotes=[])
        self.assertFalse(result["criteria"]["evidence_agreement"])
        self.assertEqual(result["supported_evidence_quotes"], [])

    def test_unsupported_quote_not_in_source_still_returns_false(self):
        repo = _repository_data()
        result = self._run(_issue(), repo, evidence_quotes=["This phrase is not in the README at all"])
        self.assertFalse(result["criteria"]["evidence_agreement"])
        self.assertEqual(result["supported_evidence_quotes"], [])

    def test_no_relevant_files_forces_file_paths_verified_false(self):
        repo = _repository_data()
        result = self._run(_issue(), repo, relevant_files=[])
        self.assertFalse(result["criteria"]["file_paths_verified"])

    def test_very_short_issue_body_forces_issue_clarity_false(self):
        repo = _repository_data()
        short_issue = _issue(body="Fix it.")
        result = self._run(short_issue, repo)
        self.assertFalse(result["criteria"]["issue_clarity"])

    def test_empty_issue_body_forces_issue_clarity_false(self):
        repo = _repository_data()
        result = self._run(_issue(body=""), repo)
        self.assertFalse(result["criteria"]["issue_clarity"])

    def test_score_zero_when_all_criteria_fail(self):
        repo = _repository_data(readme_available=False)
        bad_issue = _issue(number=1, body="x", url="https://github.com/other/repo/issues/99")
        result = self._run(bad_issue, repo)
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["status_label"], "Needs Verification")

    def test_needs_verification_label_below_40_percent(self):
        repo = _repository_data(readme_available=False)
        short_issue = _issue(body="Fix it.", url="https://github.com/other/repo/issues/1")
        result = self._run(short_issue, repo)
        self.assertLess(result["score"], 40)
        self.assertEqual(result["status_label"], "Needs Verification")


# ---------------------------------------------------------------------------
# Invalid issue URL tests
# ---------------------------------------------------------------------------


class InvalidIssueURLTests(unittest.TestCase):
    """Issue URLs that do not match the expected GitHub format."""

    def _run(self, issue_url, repo=None):
        repo = repo or _repository_data()
        issue = _issue(url=issue_url)
        return compute_confidence_check(
            issue=issue,
            repository_data=repo,
            evidence_quotes=[],
            relevant_files=[],
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo),
        )

    def test_spoofed_github_subdomain_url_fails_criterion(self):
        result = self._run("https://github.com.evil.test/octo/demo/issues/1")
        self.assertFalse(result["criteria"]["issue_available"])

    def test_http_instead_of_https_fails_criterion(self):
        result = self._run("http://github.com/octo/demo/issues/1")
        self.assertFalse(result["criteria"]["issue_available"])

    def test_wrong_repo_fails_criterion(self):
        result = self._run("https://github.com/someone-else/other-repo/issues/1")
        self.assertFalse(result["criteria"]["issue_available"])

    def test_pull_request_url_fails_criterion(self):
        result = self._run("https://github.com/octo/demo/pull/1")
        self.assertFalse(result["criteria"]["issue_available"])

    def test_missing_url_fails_criterion(self):
        repo = _repository_data()
        issue = {
            "number": 1,
            "title": "Something",
            "state": "open",
            "html_url": None,
            "body": "A valid body with enough text here to be clear.",
        }
        result = compute_confidence_check(
            issue=issue,
            repository_data=repo,
            evidence_quotes=[],
            relevant_files=[],
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo),
        )
        self.assertFalse(result["criteria"]["issue_available"])

    def test_missing_url_adds_to_missing_information(self):
        repo = _repository_data()
        issue = {
            "number": 1,
            "title": "Broken",
            "state": "open",
            "html_url": "",
            "body": "A valid body with enough text here to be clear.",
        }
        result = compute_confidence_check(
            issue=issue,
            repository_data=repo,
            evidence_quotes=[],
            relevant_files=[],
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo),
        )
        self.assertTrue(any("url" in m.lower() for m in result["missing_information"]))

    def test_correct_url_passes_criterion(self):
        result = self._run(f"{REPO_URL}/issues/1")
        self.assertTrue(result["criteria"]["issue_available"])

    def test_case_insensitive_url_match_passes(self):
        result = self._run("https://GITHUB.COM/octo/demo/issues/1")
        self.assertTrue(result["criteria"]["issue_available"])


# ---------------------------------------------------------------------------
# Unsupported file path tests
# ---------------------------------------------------------------------------


class UnsupportedFilePathTests(unittest.TestCase):
    """File paths that are not in the repository root listing."""

    def _run(self, relevant_files, root_paths=None, evidence_quotes=None, repo=None):
        repo = repo or _repository_data(root_paths=root_paths or ["README.md", "src/parser.py"])
        return compute_confidence_check(
            issue=_issue(),
            repository_data=repo,
            evidence_quotes=evidence_quotes or [],
            relevant_files=relevant_files,
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo) if root_paths is None else set(root_paths),
        )

    def test_path_not_in_root_listing_is_inferred(self):
        files = [{"path": "src/deep/module/secret.py", "reason": "Invented path."}]
        result = self._run(files)
        self.assertIn("src/deep/module/secret.py", result["inferred_file_paths"])
        self.assertNotIn("src/deep/module/secret.py", result["verified_file_paths"])

    def test_path_in_root_listing_is_verified(self):
        files = [{"path": "src/parser.py", "reason": "Parser module."}]
        result = self._run(files)
        self.assertIn("src/parser.py", result["verified_file_paths"])
        self.assertNotIn("src/parser.py", result["inferred_file_paths"])

    def test_all_inferred_paths_force_file_paths_verified_false(self):
        files = [{"path": "made/up/path.py", "reason": "Invented."}]
        result = self._run(files)
        self.assertFalse(result["criteria"]["file_paths_verified"])

    def test_mixed_paths_still_passes_criterion_if_any_verified(self):
        files = [
            {"path": "src/parser.py", "reason": "Verified path."},
            {"path": "made/up/path.py", "reason": "Inferred path."},
        ]
        result = self._run(files)
        self.assertTrue(result["criteria"]["file_paths_verified"])
        self.assertIn("src/parser.py", result["verified_file_paths"])
        self.assertIn("made/up/path.py", result["inferred_file_paths"])

    def test_empty_relevant_files_marks_criterion_false(self):
        result = self._run([])
        self.assertFalse(result["criteria"]["file_paths_verified"])

    def test_missing_information_includes_path_note_when_unverified(self):
        files = [{"path": "invented/path.py", "reason": "Made up."}]
        result = self._run(files)
        missing_combined = " ".join(result["missing_information"]).lower()
        self.assertIn("file", missing_combined)


# ---------------------------------------------------------------------------
# Partial evidence tests
# ---------------------------------------------------------------------------


class PartialEvidenceTests(unittest.TestCase):
    """Two or three criteria met → Partially Supported."""

    def test_three_criteria_met_gives_partial_status(self):
        # issue_available=True, readme_available=True, issue_clarity=True → 3/5 = 60%
        repo = _repository_data()
        result = compute_confidence_check(
            issue=_issue(),
            repository_data=repo,
            evidence_quotes=[],
            relevant_files=[],
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo),
        )
        self.assertEqual(result["score"], 60)
        self.assertEqual(result["status_label"], "Partially Supported")

    def test_score_is_between_40_and_79_for_partial(self):
        repo = _repository_data()
        result = compute_confidence_check(
            issue=_issue(),
            repository_data=repo,
            evidence_quotes=[],
            relevant_files=[],
            source_texts=_source_texts(repo),
            root_paths=_root_paths(repo),
        )
        self.assertGreaterEqual(result["score"], 40)
        self.assertLess(result["score"], 80)


# ---------------------------------------------------------------------------
# Integration: confidence_check in recommend_issues output
# ---------------------------------------------------------------------------


class ConfidenceCheckRecommendationsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """confidence_check appears in every recommendation from recommend_issues."""

    def _context(self):
        return {
            "repository": {"full_name": FULL_NAME, "html_url": REPO_URL},
            "readme": {
                "available": True,
                "path": "README.md",
                "html_url": f"{REPO_URL}/blob/main/README.md",
                "content": "# Demo\nThe parser validates contributor input and renders a report.",
                "truncated": False,
            },
            "root_contents": [
                {
                    "path": "README.md",
                    "name": "README.md",
                    "type": "file",
                    "html_url": f"{REPO_URL}/blob/main/README.md",
                },
                {
                    "path": "src/parser.py",
                    "name": "parser.py",
                    "type": "file",
                    "html_url": f"{REPO_URL}/blob/main/src/parser.py",
                },
            ],
            "languages": {"Python": 1200},
            "issues": [],
            "contribution_guidelines": [],
            "warnings": [],
        }

    def _candidate(self, number=1):
        return {
            "number": number,
            "title": f"Fix issue {number}",
            "state": "open",
            "html_url": f"{REPO_URL}/issues/{number}",
            "body": "Reject invalid parser input and return a clear error message to the user.",
            "labels": [],
        }

    def _fake_client(self, output):
        class FakeClient:
            model_name = "gemma-test:tag"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def list(self):
                return SimpleNamespace(models=[SimpleNamespace(model=FakeClient.model_name)])

            async def chat(self, **kwargs):
                return SimpleNamespace(message=SimpleNamespace(content=json.dumps(output)))

        return FakeClient()

    async def test_confidence_check_present_in_each_recommendation(self):
        model_output = {
            "assessments": [{
                "issue_number": 1,
                "difficulty_estimate": "beginner",
                "difficulty_confidence": 4,
                "skills_needed": ["Python"],
                "skill_compatibility_score": 4,
                "task_clarity_score": 4,
                "scope_score": 4,
                "evidence_score": 4,
                "reasons": ["Good fit."],
                "not_suitable_reasons": [],
                "evidence_quotes": ["parser validates contributor input"],
                "relevant_files": [{
                    "path": "src/parser.py",
                    "reason": "Parser module.",
                    "evidence_quote": "parser validates contributor input",
                }],
            }]
        }
        with (
            patch.dict(os.environ, {"GEMMA_MODEL": "gemma-test:tag"}),
            patch("backend.ai_service.AsyncClient", return_value=self._fake_client(model_output)),
        ):
            result = await OllamaAIService().recommend_issues(
                self._context(),
                [self._candidate(1)],
                {
                    "known_languages": ["Python"],
                    "experience_level": "beginner",
                    "preferred_contribution_type": "any",
                    "skills_description": "",
                },
            )

        self.assertEqual(len(result["recommendations"]), 1)
        rec = result["recommendations"][0]
        self.assertIn("confidence_check", rec)
        cc = rec["confidence_check"]
        self.assertIsInstance(cc["score"], int)
        self.assertIn(cc["status_label"], ("Verified Source", "Partially Supported", "Needs Verification"))
        self.assertIsInstance(cc["criteria"], dict)
        self.assertEqual(len(cc["criteria"]), 5)
        self.assertGreater(len(cc["verification_steps"]), 0)

    async def test_confidence_check_marks_missing_evidence_as_low_score(self):
        model_output = {
            "assessments": [{
                "issue_number": 1,
                "difficulty_estimate": "uncertain",
                "difficulty_confidence": 1,
                "skills_needed": [],
                "skill_compatibility_score": 1,
                "task_clarity_score": 1,
                "scope_score": 1,
                "evidence_score": 1,
                "reasons": [],
                "not_suitable_reasons": ["No evidence."],
                "evidence_quotes": [],
                "relevant_files": [],
            }]
        }
        # Short body → issue_clarity=False, no quotes, no files
        short_issue = {
            "number": 1,
            "title": "Vague issue",
            "state": "open",
            "html_url": f"{REPO_URL}/issues/1",
            "body": "Fix it.",
            "labels": [],
        }
        with (
            patch.dict(os.environ, {"GEMMA_MODEL": "gemma-test:tag"}),
            patch("backend.ai_service.AsyncClient", return_value=self._fake_client(model_output)),
        ):
            result = await OllamaAIService().recommend_issues(
                self._context(), [short_issue], {},
            )

        rec = result["recommendations"][0]
        cc = rec["confidence_check"]
        self.assertFalse(cc["criteria"]["issue_clarity"])
        self.assertFalse(cc["criteria"]["evidence_agreement"])
        self.assertFalse(cc["criteria"]["file_paths_verified"])
        self.assertLess(cc["score"], 80)


# ---------------------------------------------------------------------------
# Score output guarantees
# ---------------------------------------------------------------------------


class ScoreOutputGuaranteesTests(unittest.TestCase):
    """Score and output field invariants across scenarios."""

    def _run(self, **kwargs):
        repo = _repository_data()
        defaults = {
            "issue": _issue(),
            "repository_data": repo,
            "evidence_quotes": [],
            "relevant_files": [],
            "source_texts": _source_texts(repo),
            "root_paths": _root_paths(repo),
        }
        defaults.update(kwargs)
        return compute_confidence_check(**defaults)

    def test_score_is_always_an_integer(self):
        self.assertIsInstance(self._run()["score"], int)

    def test_score_is_between_0_and_100(self):
        result = self._run()
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)

    def test_status_label_is_one_of_three_valid_values(self):
        result = self._run()
        self.assertIn(result["status_label"], ("Verified Source", "Partially Supported", "Needs Verification"))

    def test_criteria_always_has_five_keys(self):
        result = self._run()
        self.assertEqual(len(result["criteria"]), 5)

    def test_all_output_fields_present(self):
        result = self._run()
        expected = {
            "score", "status_label", "criteria", "score_explanation",
            "reasons", "missing_information", "verification_steps",
            "supported_evidence_quotes", "verified_file_paths", "inferred_file_paths",
        }
        self.assertTrue(expected.issubset(result.keys()))

    def test_reasons_list_has_one_entry_per_criterion(self):
        result = self._run()
        self.assertEqual(len(result["reasons"]), 5)

    def test_score_explanation_mentions_total_count(self):
        result = self._run()
        self.assertIn("5", result["score_explanation"])

    def test_score_explanation_does_not_use_probability_language(self):
        result = self._run()
        explanation = result["score_explanation"].lower()
        for word in ("probability", "chance", "likelihood"):
            self.assertNotIn(word, explanation, f"score_explanation must not use '{word}'")


if __name__ == "__main__":
    unittest.main()
