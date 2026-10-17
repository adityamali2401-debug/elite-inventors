"""RepoRescue AI GitHub repository API.

The only user-controlled URL is parsed and reduced to a GitHub owner/repository
pair. All outbound requests are then built against the fixed api.github.com host.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote, urlsplit
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ai_service import AIServiceError, ai_service, explicit_beginner_label_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_JSON_BYTES = 768 * 1024
MAX_README_RESPONSE_BYTES = 192 * 1024
MAX_README_TEXT_BYTES = 48 * 1024
MAX_ISSUES = 30
MAX_ROOT_ENTRIES = 100
MAX_ISSUE_BODY_CHARS = 700
MAX_RECOMMENDATION_CANDIDATES = 15
REPOSITORY_CACHE_TTL_SECONDS = 300
REPOSITORY_CACHE_MAX_ENTRIES = 64

OWNER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepositoryAIRequest(StrictRequest):
    repository_url: str = Field(min_length=1, max_length=2048)


class ChatMessage(StrictRequest):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1200)


class RepositoryChatRequest(RepositoryAIRequest):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank.")
        return cleaned


class IssueExplanationRequest(RepositoryAIRequest):
    issue_number: int = Field(strict=True, ge=1, le=2_147_483_647)


class IssueFinderRequest(RepositoryAIRequest):
    known_languages: list[str] = Field(default_factory=list, max_length=12)
    experience_level: Literal["beginner", "some-experience", "experienced"] = "beginner"
    contribution_type: Literal["any", "bug-fix", "documentation", "tests", "accessibility", "performance", "feature", "other"] = "any"
    skills_description: str = Field(default="", max_length=800)
    refresh: bool = False

    @field_validator("known_languages")
    @classmethod
    def clean_known_languages(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip()[:80] for value in values if value.strip()))

    @field_validator("skills_description")
    @classmethod
    def clean_skills_description(cls, value: str) -> str:
        return value.strip()


class ContributionRoadmapRequest(IssueExplanationRequest):
    known_languages: list[str] = Field(default_factory=list, max_length=12)
    experience_level: Literal["beginner", "some-experience", "experienced"] = "beginner"
    skills_description: str = Field(default="", max_length=800)

    @field_validator("known_languages")
    @classmethod
    def clean_roadmap_languages(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip()[:80] for value in values if value.strip()))

    @field_validator("skills_description")
    @classmethod
    def clean_roadmap_skills(cls, value: str) -> str:
        return value.strip()


_repository_context_cache: dict[tuple[str, str], tuple[float, dict[str, object]]] = {}


def _cache_repository_context(owner: str, repository: str, payload: dict[str, object]) -> None:
    key = (owner.casefold(), repository.casefold())
    now = time.monotonic()
    expired = [cache_key for cache_key, (stored_at, _) in _repository_context_cache.items() if now - stored_at >= REPOSITORY_CACHE_TTL_SECONDS]
    for cache_key in expired:
        _repository_context_cache.pop(cache_key, None)
    if key not in _repository_context_cache and len(_repository_context_cache) >= REPOSITORY_CACHE_MAX_ENTRIES:
        oldest_key = min(_repository_context_cache, key=lambda cache_key: _repository_context_cache[cache_key][0])
        _repository_context_cache.pop(oldest_key, None)
    _repository_context_cache[key] = (now, copy.deepcopy(payload))


def _cached_repository_context(owner: str, repository: str) -> dict[str, object] | None:
    key = (owner.casefold(), repository.casefold())
    cached = _repository_context_cache.get(key)
    if cached is None:
        return None
    stored_at, payload = cached
    if time.monotonic() - stored_at >= REPOSITORY_CACHE_TTL_SECONDS:
        _repository_context_cache.pop(key, None)
        return None
    return copy.deepcopy(payload)


def parse_github_repository_url(value: str) -> tuple[str, str]:
    """Accept only a canonical HTTPS GitHub repository URL; return owner/name."""
    try:
        parsed = urlsplit(value.strip())
        # Accessing .port also validates malformed ports; all ports are rejected.
        port = parsed.port
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="Enter a valid GitHub repository URL.") from None

    if (
        parsed.scheme.lower() != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.hostname is None
        or parsed.hostname.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or "%" in parsed.path
    ):
        raise HTTPException(
            status_code=422,
            detail="URL must use https://github.com/owner/repository (optional trailing slash or .git).",
        )

    match = re.fullmatch(r"/([^/]+)/([^/]+?)(?:\.git)?/?", parsed.path)
    if not match:
        raise HTTPException(
            status_code=422,
            detail="URL must point directly to one GitHub repository: /owner/repository.",
        )

    owner, repository = match.groups()
    if (
        not OWNER_PATTERN.fullmatch(owner)
        or not REPOSITORY_PATTERN.fullmatch(repository)
        or repository in {".", ".."}
    ):
        raise HTTPException(status_code=422, detail="The GitHub owner or repository name is invalid.")

    return owner, repository


class GitHubAPIError(Exception):
    def __init__(self, status_code: int, detail: str, *, rate_limited: bool = False, retry_after: str | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.rate_limited = rate_limited
        self.retry_after = retry_after


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "RepoRescue-AI/1.0",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _read_limited(response: httpx.Response, limit: int, *, tolerate_overflow: bool = False) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        remaining = limit - size
        if len(chunk) > remaining:
            if tolerate_overflow:
                chunks.append(chunk[:remaining])
                break
            raise GitHubAPIError(502, "GitHub returned a response larger than the allowed limit.")
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def _github_error(response: httpx.Response, body: bytes) -> GitHubAPIError:
    message = "GitHub could not complete the repository request."
    try:
        payload = json.loads(body) if body else {}
        if isinstance(payload, dict) and isinstance(payload.get("message"), str):
            message = payload["message"][:300]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    remaining = response.headers.get("X-RateLimit-Remaining")
    lowered_message = message.lower()
    rate_limited = (
        response.status_code == 429
        or remaining == "0"
        or "rate limit exceeded" in lowered_message
        or "secondary rate limit" in lowered_message
        or "abuse detection" in lowered_message
    )
    return GitHubAPIError(
        response.status_code,
        message,
        rate_limited=rate_limited,
        retry_after=response.headers.get("Retry-After"),
    )


async def _github_get(
    client: httpx.AsyncClient,
    api_path: str,
    *,
    params: dict[str, str | int] | None = None,
    max_bytes: int = MAX_JSON_BYTES,
) -> object:
    """GET a fixed-host GitHub API resource while bounding response bytes."""
    async with client.stream("GET", f"{GITHUB_API}{api_path}", params=params, headers=_github_headers()) as response:
        if response.status_code >= 400:
            body = await _read_limited(response, 8 * 1024, tolerate_overflow=True)
            raise _github_error(response, body)
        if response.is_redirect:
            raise GitHubAPIError(502, "GitHub redirected an API request unexpectedly.")
        body = await _read_limited(response, max_bytes)

    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GitHubAPIError(502, "GitHub returned an invalid JSON response.") from error


def _api_repository_path(owner: str, repository: str) -> str:
    return f"/repos/{quote(owner, safe='')}/{quote(repository, safe='')}"


def _readme_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"available": False, "name": None, "path": None, "html_url": None, "content": "", "truncated": False}

    encoded = payload.get("content")
    content = ""
    truncated = False
    if isinstance(encoded, str) and payload.get("encoding") == "base64":
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
            truncated = len(raw) > MAX_README_TEXT_BYTES
            content = raw[:MAX_README_TEXT_BYTES].decode("utf-8", errors="replace")
        except (binascii.Error, ValueError):
            content = ""
    elif isinstance(encoded, str):
        raw = encoded.encode("utf-8")
        truncated = len(raw) > MAX_README_TEXT_BYTES
        content = raw[:MAX_README_TEXT_BYTES].decode("utf-8", errors="replace")

    return {
        "available": True,
        "name": payload.get("name"),
        "path": payload.get("path"),
        "html_url": payload.get("html_url"),
        "content": content,
        "truncated": truncated,
    }


def _root_contents(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list):
        return []
    entries: list[dict[str, object]] = []
    for item in payload[:MAX_ROOT_ENTRIES]:
        if not isinstance(item, dict):
            continue
        entries.append(
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "type": item.get("type"),
                "size": item.get("size"),
                "html_url": item.get("html_url"),
            }
        )
    return entries


async def _fetch_contribution_guidelines(
    client: httpx.AsyncClient,
    repository_path: str,
    root_payload: object,
) -> tuple[list[dict[str, object]], list[str]]:
    """Fetch at most two contribution guides, only after GitHub directory evidence lists them."""
    guide_names = {"contributing", "contributing.md", "contributing.markdown", "contributing.rst", "contributing.txt"}
    candidate_paths: list[str] = []
    warnings: list[str] = []
    root_entries = root_payload if isinstance(root_payload, list) else []
    for entry in root_entries:
        if isinstance(entry, dict) and entry.get("type") == "file" and str(entry.get("name", "")).casefold() in guide_names:
            path = entry.get("path")
            if isinstance(path, str) and path == entry.get("name"):
                candidate_paths.append(path)

    root_directories = {
        str(entry.get("name"))
        for entry in root_entries
        if isinstance(entry, dict) and entry.get("type") == "dir" and entry.get("name") in {".github", "docs", "doc"}
    }
    for directory in (".github", "docs", "doc"):
        if directory not in root_directories or len(candidate_paths) >= 2:
            continue
        try:
            directory_entries = await _github_get(client, f"{repository_path}/contents/{quote(directory, safe='')}")
            if isinstance(directory_entries, list):
                for entry in directory_entries:
                    if isinstance(entry, dict) and entry.get("type") == "file" and str(entry.get("name", "")).casefold() in guide_names:
                        path = entry.get("path")
                        if isinstance(path, str) and path.startswith(f"{directory}/") and ".." not in path.split("/"):
                            candidate_paths.append(path)
        except GitHubAPIError as error:
            if error.rate_limited:
                _raise_github_http_error(error)
            if error.status_code != 404:
                warnings.append(f"GitHub could not provide files from the {directory} directory.")
        except (httpx.TimeoutException, httpx.RequestError):
            warnings.append(f"GitHub could not provide files from the {directory} directory.")

    guidelines: list[dict[str, object]] = []
    for path in list(dict.fromkeys(candidate_paths))[:2]:
        try:
            payload = await _github_get(
                client,
                f"{repository_path}/contents/{quote(path, safe='/')}",
                max_bytes=MAX_README_RESPONSE_BYTES,
            )
        except GitHubAPIError as error:
            if error.rate_limited:
                _raise_github_http_error(error)
            if error.status_code != 404:
                warnings.append(f"GitHub could not provide the contribution guide at {path}.")
            continue
        except (httpx.TimeoutException, httpx.RequestError):
            warnings.append(f"GitHub could not provide the contribution guide at {path}.")
            continue
        decoded = _readme_payload(payload)
        if decoded["available"]:
            guidelines.append(decoded)
    return guidelines, warnings


def _language_stats(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(name)[:100]: max(0, int(byte_count))
        for name, byte_count in list(payload.items())[:20]
        if isinstance(name, str) and isinstance(byte_count, (int, float)) and not isinstance(byte_count, bool)
    }


def _open_issues(payload: object, owner: str | None = None, repository: str | None = None) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        payload = payload.get("items")
    if not isinstance(payload, list):
        return []
    issues: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict) or "pull_request" in item:
            continue
        number = item.get("number")
        title = item.get("title")
        issue_url = item.get("html_url")
        if item.get("state") != "open" or not isinstance(number, int) or isinstance(number, bool) or number < 1:
            continue
        if not isinstance(title, str) or not title.strip() or not isinstance(issue_url, str):
            continue
        try:
            parsed_issue_url = urlsplit(issue_url)
            issue_port = parsed_issue_url.port
        except ValueError:
            continue
        expected_path = f"/{owner}/{repository}/issues/{number}" if owner and repository else None
        if (
            parsed_issue_url.scheme != "https"
            or parsed_issue_url.hostname != "github.com"
            or parsed_issue_url.username is not None
            or parsed_issue_url.password is not None
            or issue_port is not None
            or parsed_issue_url.query
            or parsed_issue_url.fragment
            or (expected_path is not None and parsed_issue_url.path.casefold() != expected_path.casefold())
        ):
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), list) else []
        issues.append(
            {
                "number": number,
                "title": title.strip()[:300],
                "html_url": issue_url,
                "state": "open",
                "body": str(item.get("body") or "")[:MAX_ISSUE_BODY_CHARS],
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "comments": item.get("comments", 0),
                "labels": [
                    {"name": str(label.get("name") or "")[:100], "color": label.get("color")}
                    for label in labels[:8]
                    if isinstance(label, dict)
                ],
            }
        )
        if len(issues) >= MAX_ISSUES:
            break
    return issues


def _optional_warning(name: str, error: Exception) -> str:
    if isinstance(error, GitHubAPIError) and error.status_code == 404 and name.lower() == "readme":
        return ""
    return f"GitHub could not provide {name.lower()} for this repository."


def _raise_github_http_error(error: GitHubAPIError) -> None:
    if error.rate_limited:
        headers = {"Retry-After": error.retry_after} if error.retry_after else None
        raise HTTPException(
            status_code=429,
            detail="GitHub API rate limit reached. Try again later or configure a GitHub token in the backend environment.",
            headers=headers,
        ) from None
    if error.status_code == 404:
        raise HTTPException(status_code=404, detail="Repository not found or not accessible. It may be private or misspelled.") from None
    if error.status_code in {401, 403}:
        raise HTTPException(status_code=403, detail="GitHub denied access. Check repository visibility and the configured token permissions.") from None
    raise HTTPException(status_code=502, detail="GitHub could not complete the request. Please try again later.") from None


app = FastAPI(
    title="RepoRescue AI API",
    version="1.0.0",
    description="Read-only GitHub repository metadata for the RepoRescue AI frontend.",
)

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Content-Type"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "RepoRescue AI API"}


@app.get("/api/repository")
async def get_repository(
    url: str = Query(..., min_length=1, max_length=2048, description="HTTPS GitHub repository URL"),
) -> dict[str, object]:
    owner, repository = parse_github_repository_url(url)
    base_path = _api_repository_path(owner, repository)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=4.0)
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=5)
    contribution_guidelines: list[dict[str, object]] = []
    guideline_warnings: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False) as client:
            metadata = await _github_get(client, base_path)
            if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
                raise GitHubAPIError(502, "GitHub returned incomplete repository metadata.")

            jobs = {
                "readme": _github_get(client, f"{base_path}/readme", max_bytes=MAX_README_RESPONSE_BYTES),
                "root_contents": _github_get(client, f"{base_path}/contents", params={"per_page": MAX_ROOT_ENTRIES}),
                "languages": _github_get(client, f"{base_path}/languages"),
                "issues": _github_get(
                    client,
                    "/search/issues",
                    params={
                        "q": f"repo:{owner}/{repository} is:open is:issue",
                        "sort": "created",
                        "order": "desc",
                        "per_page": MAX_ISSUES,
                    },
                    max_bytes=256 * 1024,
                ),
            }
            names = list(jobs)
            results = await asyncio.gather(*jobs.values(), return_exceptions=True)
            root_result = results[names.index("root_contents")]
            if not isinstance(root_result, Exception):
                contribution_guidelines, guideline_warnings = await _fetch_contribution_guidelines(client, base_path, root_result)

    except GitHubAPIError as error:
        _raise_github_http_error(error)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GitHub took too long to respond. Please try again.") from None
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not connect to GitHub. Please try again later.") from None

    fetched: dict[str, object] = {}
    warnings: list[str] = []
    for name, result in zip(names, results, strict=True):
        if isinstance(result, Exception):
            if isinstance(result, GitHubAPIError) and result.rate_limited:
                _raise_github_http_error(result)
            warning = _optional_warning(name, result)
            if warning:
                warnings.append(warning)
            continue
        fetched[name] = result

    readme_data = _readme_payload(fetched.get("readme")) if "readme" in fetched else {
        "available": False,
        "name": None,
        "path": None,
        "html_url": None,
        "content": "",
        "truncated": False,
    }
    repository_data = {
        "id": metadata.get("id"),
        "name": metadata.get("name"),
        "full_name": metadata.get("full_name"),
        "owner": (metadata.get("owner") or {}).get("login") if isinstance(metadata.get("owner"), dict) else owner,
        "description": metadata.get("description"),
        "html_url": metadata.get("html_url"),
        "private": bool(metadata.get("private", False)),
        "default_branch": metadata.get("default_branch"),
        "stargazers_count": metadata.get("stargazers_count", 0),
        "forks_count": metadata.get("forks_count", 0),
        "open_issues_count": metadata.get("open_issues_count", 0),
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "license": (metadata.get("license") or {}).get("spdx_id") if isinstance(metadata.get("license"), dict) else None,
    }

    result: dict[str, object] = {
        "repository": repository_data,
        "readme": readme_data,
        "root_contents": _root_contents(fetched.get("root_contents")),
        "languages": _language_stats(fetched.get("languages")),
        "issues": _open_issues(fetched.get("issues"), owner, repository),
        "contribution_guidelines": contribution_guidelines,
        "warnings": warnings + guideline_warnings,
    }
    _cache_repository_context(owner, repository, result)
    return result


async def _fetch_open_issues(owner: str, repository: str) -> list[dict[str, object]]:
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=4.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            payload = await _github_get(
                client,
                "/search/issues",
                params={
                    "q": f"repo:{owner}/{repository} is:open is:issue",
                    "sort": "created",
                    "order": "desc",
                    "per_page": MAX_ISSUES,
                },
                max_bytes=256 * 1024,
            )
    except GitHubAPIError as error:
        _raise_github_http_error(error)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GitHub took too long to refresh issues. Please try again.") from None
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not connect to GitHub to refresh issues.") from None
    return _open_issues(payload, owner, repository)


async def _get_ai_repository_context(repository_url: str, *, refresh_issues: bool = False) -> dict[str, object]:
    owner, repository = parse_github_repository_url(repository_url)
    cached = _cached_repository_context(owner, repository)
    if cached is not None and refresh_issues:
        cached["issues"] = await _fetch_open_issues(owner, repository)
        cached["warnings"] = [warning for warning in cached.get("warnings", []) if "issues" not in warning.lower()]
        _cache_repository_context(owner, repository, cached)
        return cached
    if cached is not None:
        return cached
    return await get_repository(url=repository_url)


def _ai_http_error(error: AIServiceError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.message)


@app.post("/api/analyze")
async def analyze_repository(request: RepositoryAIRequest) -> dict[str, object]:
    repository_context = await _get_ai_repository_context(request.repository_url)
    try:
        generated = await ai_service.analyze_repository(repository_context)
    except AIServiceError as error:
        raise _ai_http_error(error) from None
    return {
        "repository_url": request.repository_url,
        "repository": repository_context.get("repository", {}),
        **generated,
    }


@app.post("/api/explain-issue")
async def explain_issue(request: IssueExplanationRequest) -> dict[str, object]:
    repository_context = await _get_ai_repository_context(request.repository_url)
    issues = repository_context.get("issues", [])
    issue = next(
        (item for item in issues if isinstance(item, dict) and item.get("number") == request.issue_number),
        None,
    ) if isinstance(issues, list) else None
    if issue is None:
        raise HTTPException(
            status_code=404,
            detail="That issue is not among the recent open issues retrieved for this repository. Analyze the repository again and select a listed issue.",
        )
    try:
        return await ai_service.explain_issue(repository_context, issue)
    except AIServiceError as error:
        raise _ai_http_error(error) from None


@app.post("/api/roadmap")
async def generate_contribution_roadmap(request: ContributionRoadmapRequest) -> dict[str, object]:
    repository_context = await _get_ai_repository_context(request.repository_url)
    issues = repository_context.get("issues", [])
    issue = next(
        (item for item in issues if isinstance(item, dict) and item.get("number") == request.issue_number),
        None,
    ) if isinstance(issues, list) else None
    if issue is None or issue.get("state") != "open" or "pull_request" in issue:
        raise HTTPException(
            status_code=404,
            detail="That issue is not an available open GitHub issue in this repository. Refresh repository issues and choose a listed issue.",
        )
    preferences = {
        "known_languages": request.known_languages,
        "experience_level": request.experience_level,
        "skills_description": request.skills_description,
    }
    try:
        return await ai_service.generate_contribution_roadmap(repository_context, issue, preferences)
    except AIServiceError as error:
        print("ROADMAP AI ERROR:", repr(error))
        raise _ai_http_error(error) from None


@app.post("/api/issues/recommend")
async def recommend_issues(request: IssueFinderRequest) -> dict[str, object]:
    repository_context = await _get_ai_repository_context(request.repository_url, refresh_issues=request.refresh)
    issues = repository_context.get("issues", [])
    owner, repository = parse_github_repository_url(request.repository_url)
    valid_issues = _open_issues(issues, owner, repository)

    # Put explicitly labeled candidates first, while still evaluating unlabeled issues.
    def has_beginner_signal(issue: dict[str, object]) -> bool:
        labels = issue.get("labels", [])
        return bool(explicit_beginner_label_signals(labels)) if isinstance(labels, list) else False

    labeled_candidates = [issue for issue in valid_issues if has_beginner_signal(issue)]
    other_candidates = [issue for issue in valid_issues if not has_beginner_signal(issue)]
    label_quota = min(len(labeled_candidates), max(1, MAX_RECOMMENDATION_CANDIDATES // 2))
    candidate_issues = labeled_candidates[:label_quota] + other_candidates[:MAX_RECOMMENDATION_CANDIDATES - label_quota]
    remaining_slots = MAX_RECOMMENDATION_CANDIDATES - len(candidate_issues)
    candidate_issues += labeled_candidates[label_quota:label_quota + remaining_slots]
    preferences = {
        "known_languages": request.known_languages,
        "experience_level": request.experience_level,
        "preferred_contribution_type": request.contribution_type,
        "skills_description": request.skills_description,
    }
    try:
        return await ai_service.recommend_issues(repository_context, candidate_issues, preferences)
    except AIServiceError as error:
        return {
            "model": None,
            "analysis_error": error.message,
            "repository": repository_context.get("repository", {}),
            "issues": valid_issues[:MAX_RECOMMENDATION_CANDIDATES],
            "candidate_count": min(len(valid_issues), MAX_RECOMMENDATION_CANDIDATES),
            "suitable_count": 0,
            "recommendations": [],
            "ranking_method": {
                "formula": "No ranking available because Gemma could not assess the issue text.",
                "note": "These are unranked GitHub issues. No difficulty or skill match is implied.",
            },
        }


@app.post("/api/chat")
async def repository_chat(request: RepositoryChatRequest) -> dict[str, object]:
    repository_context = await _get_ai_repository_context(request.repository_url)
    history = [message.model_dump() for message in request.history]
    try:
        return await ai_service.chat(repository_context, request.question, history)
    except AIServiceError as error:
        raise _ai_http_error(error) from None
