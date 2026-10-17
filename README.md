# RepoRescue AI

RepoRescue AI is a small FastAPI service plus a static frontend for exploring public GitHub repositories and asking a locally installed Gemma model for grounded contributor guidance.

## Requirements

- Python 3.12 (Python 3.10+ should also work)
- Ollama installed and running locally for AI features
- A Gemma model already available in Ollama; its version/tag is not assumed

## Configure and run

From this project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama list
```

Edit `.env` and set `GEMMA_MODEL` to the exact model tag displayed by `ollama list`. Optionally set `OLLAMA_HOST`, `OLLAMA_TIMEOUT_SECONDS`, and `OLLAMA_NUM_CTX`. If `GITHUB_TOKEN` is needed, keep it in `.env` only; never put it in browser code. Ollama must be running (on Windows, starting the Ollama application normally starts its local service).

Start the backend in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Serve the existing static UI from another terminal:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Open <http://127.0.0.1:5500>. The browser calls the existing GitHub route at `/api/repository` and the AI routes at `/api/analyze`, `/api/explain-issue`, and `/api/chat` on port 8000. The GitHub endpoint remains available if Ollama is stopped or the model is not configured.

## API routes

### `POST /api/analyze`

Request body:

```json
{"repository_url":"https://github.com/owner/repository"}
```

Returns the verified GitHub repository metadata and a separate `analysis` object with a beginner overview, purpose, language stats, evidence-checked root paths, README-quoted setup instructions, contribution hypotheses, missing information, and GitHub source links. It uses the repository evidence cached by `/api/repository` for five minutes or retrieves it itself when absent.

### `POST /api/explain-issue`

Request body:

```json
{"repository_url":"https://github.com/owner/repository","issue_number":123}
```

The issue number must belong to the recent open issue results returned by GitHub for that repository. The response includes explanation, hypotheses, suggested steps, testing checklist, relevant root files only when evidence supports them, and GitHub links.

### `POST /api/issues/recommend`

Request body:

```json
{
	"repository_url":"https://github.com/owner/repository",
	"known_languages":["Python"],
	"experience_level":"beginner",
	"contribution_type":"documentation",
	"skills_description":"I have written small Python scripts.",
	"refresh":false
}
```

`experience_level` accepts `beginner`, `some-experience`, or `experienced`. `contribution_type` accepts `any`, `bug-fix`, `documentation`, `tests`, `accessibility`, `performance`, `feature`, or `other`. Set `refresh` to `true` to refresh only the GitHub issues search when the repository context is cached; this uses one GitHub API request. Otherwise the five-minute repository cache is reused.

The finder evaluates recent, validated open issue candidates (pull requests, closed issues, malformed records, and non-canonical source URLs are excluded). Label-signaled issues are included, but unlabeled issues are also assessed. Labels are returned as actual GitHub metadata and are never treated as proof of difficulty. Gemma produces separate estimated difficulty and 1–5 skill-fit, clarity, scope, and evidence scores. The server calculates the visible 0–100 match score as `20 × (skills × 0.40 + clarity × 0.25 + scope × 0.20 + evidence × 0.15)`. Exact evidence quotes are checked against fetched issue/README text; if there is insufficient evidence, difficulty is marked uncertain. A local model failure returns actual GitHub issues unranked with a clear error rather than hiding them.

Test the finder after configuring Ollama and `GEMMA_MODEL`:

```powershell
$repo = 'https://github.com/fastapi/fastapi'
$finder = @{
	repository_url = $repo
	known_languages = @('Python')
	experience_level = 'beginner'
	contribution_type = 'documentation'
	skills_description = 'I have written small Python scripts.'
	refresh = $false
}
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/issues/recommend' -ContentType 'application/json' -Body ($finder | ConvertTo-Json -Depth 5)
```

### `POST /api/roadmap`

Request body:

```json
{
	"repository_url":"https://github.com/owner/repository",
	"issue_number":123,
	"known_languages":["Python"],
	"experience_level":"beginner",
	"skills_description":"I have written a few Python scripts."
}
```

The issue must still be an open issue returned by GitHub for the selected repository. The roadmap has the ordered steps for understanding the issue, expected result, locating files, reading verified contribution guides, documented environment setup, branch suggestion, investigation, implementation, tests, diff review, and preparing a pull request. It includes actual README and contribution-guide links when present, and marks guidance that could not be tied to repository evidence. Commands are returned as copyable suggestions, never executed. Only exact documentation-backed commands or a safe branch-creation suggestion are included; destructive and remote-mutating commands are withheld and identified in the relevant step.

The browser saves completed steps under a separate local-storage key for each repository and issue. Copying a command places it on the clipboard only after the user clicks; a defensive confirmation appears if a potentially destructive command is ever returned. No Git command is automatically run.

To test the roadmap route directly, use an issue number from `/api/repository` for the same repository:

```powershell
$roadmap = @{
	repository_url = 'https://github.com/fastapi/fastapi'
	issue_number = 10370
	known_languages = @('Python')
	experience_level = 'beginner'
	skills_description = 'I am learning Python and tests.'
}
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/roadmap' -ContentType 'application/json' -Body ($roadmap | ConvertTo-Json -Depth 5)
```

Replace issue `10370` with a currently open issue returned for the repository. The roadmap test fixtures and mocked Gemma response run with the normal test suite and do not require a local model.

### `POST /api/chat`

Request body:

```json
{"repository_url":"https://github.com/owner/repository","question":"How is this repository organized?","history":[]}
```

Returns a repository-grounded answer, evidence-checked facts, hypotheses, missing information, the model tag, and source links. Chat history is optional and limited to eight short user/assistant messages.

## Test the endpoints

With both backend and UI setup complete, test repository context in PowerShell:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/repository?url=https%3A%2F%2Fgithub.com%2Ffastapi%2Ffastapi'
```

Call repository analysis:

```powershell
$repo = 'https://github.com/fastapi/fastapi'
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/analyze' -ContentType 'application/json' -Body (@{ repository_url = $repo } | ConvertTo-Json)
```

Call issue explanation using an issue number shown by the repository response:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/explain-issue' -ContentType 'application/json' -Body (@{ repository_url = $repo; issue_number = 10370 } | ConvertTo-Json)
```

Call repository chat:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/chat' -ContentType 'application/json' -Body (@{ repository_url = $repo; question = 'How is this repository organized?'; history = @() } | ConvertTo-Json -Depth 5)
```

Issue `10370` is just an example value: use a current issue number returned by the repository endpoint. AI calls need Ollama, a populated `GEMMA_MODEL`, and that exact tag installed. A missing/unavailable local model returns an actionable API error without disabling GitHub metadata, README, language, or issue data.

Run the automated suite (these tests mock external GitHub/Ollama services and need no network or local model):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`tests/fixtures/github_fastapi_issue.json` is a captured real GitHub issue response; the controlled tests additionally include closed issues, pull requests, malformed URLs, missing evidence, and mismatched skills. To run the one-request live GitHub integration check as well:

```powershell
$env:REPORESCUE_LIVE_GITHUB_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_issue_finder.py -v
Remove-Item Env:REPORESCUE_LIVE_GITHUB_TESTS
```

The live test requires network access and can be skipped if GitHub's unauthenticated rate limit is exhausted. The normal suite does not make live GitHub or Ollama requests.