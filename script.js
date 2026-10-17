/*
 * RepoRescue AI frontend
 * ---------------------
 * All live data comes from the API service below. Sample fixtures are isolated
 * in SAMPLE_PREVIEW and are never presented as live GitHub repository data.
 * Configure window.REPORESCUE_CONFIG = { apiBaseUrl: 'http://127.0.0.1:8000/api' }
 * before this file loads to point the UI at a deployed Python backend.
 *
 * Suggested API contract:
 *   GET  /repository?url=https://github.com/owner/repository
 *   POST /issues/explain             { repository, issue }
 *   POST /assistant/chat             { repository, question, history }
 */
(() => {
  'use strict';

  const API_BASE_URL = (window.REPORESCUE_CONFIG?.apiBaseUrl || 'http://127.0.0.1:8000/api').replace(/\/$/, '');
  const ROADMAP_STORAGE_PREFIX = 'reporescue-roadmap-v2:';
  const SAMPLE_PREVIEW = Object.freeze({
    name: 'frontend-playground',
    fullName: 'example-studio/frontend-playground',
    description: 'Illustrative sample only — this is not a real GitHub repository or repository analysis.',
    htmlUrl: '',
    defaultBranch: 'main · example',
    overview: 'This sample panel demonstrates where a plain-language project overview will appear. No live repository was analyzed for this preview.',
    technologies: ['Example language', 'Example framework', 'Sample tooling'],
    importantFiles: [
      { path: 'README.md', purpose: 'Example: project overview and setup guide' },
      { path: 'src/', purpose: 'Example: application source folder' },
      { path: 'tests/', purpose: 'Example: test folder' },
    ],
    setupInstructions: ['Read the project README.', 'Install dependencies using the documented command.', 'Run the project checks before making changes.'],
    health: [
      { label: 'Repository metadata', status: 'example' },
      { label: 'Setup documentation', status: 'example' },
      { label: 'Test instructions', status: 'example' },
      { label: 'Contribution guide', status: 'example' },
    ],
  });

  const state = {
    currentView: 'dashboard',
    lastRepositoryUrl: '',
    repository: null,
    issues: [],
    issuesLoaded: false,
    issueRecommendations: null,
    finderPreferencesKey: '',
    finderRequestId: 0,
    selectedIssue: null,
    chatHistory: [],
    isPreviewAnalysis: false,
    lastAnalysisError: '',
    analysisRequestId: 0,
    aiRequestId: 0,
    chatRequestId: 0,
    roadmapRequestId: 0,
    roadmapIssue: null,
    roadmapData: null,
    roadmapProgressKey: '',
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const APIService = {
    async request(path, options = {}) {
      let response;
      try {
        response = await fetch(`${API_BASE_URL}${path}`, {
          ...options,
          headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
        });
      } catch (error) {
        throw new Error(`Could not reach the RepoRescue API at ${API_BASE_URL}. Start the Python backend or set window.REPORESCUE_CONFIG.apiBaseUrl. No live data was loaded.`);
      }

      const contentType = response.headers.get('content-type') || '';
      let payload = null;
      if (contentType.includes('application/json')) {
        payload = await response.json().catch(() => null);
      } else {
        const text = await response.text().catch(() => '');
        payload = text ? { detail: text.slice(0, 240) } : null;
      }
      if (!response.ok) {
        const detail = payload?.detail || payload?.message;
        const message = Array.isArray(detail)
          ? detail.map((item) => item.msg || item.message || 'Invalid request').join(' ')
          : detail || `The API returned ${response.status} ${response.statusText}.`;
        throw new Error(String(message));
      }
      return payload;
    },

    getRepository(repositoryUrl) {
      const query = new URLSearchParams({ url: repositoryUrl });
      return this.request(`/repository?${query.toString()}`);
    },

    analyzeRepository(repositoryUrl) {
      return this.request('/analyze', { method: 'POST', body: JSON.stringify({ repository_url: repositoryUrl }) });
    },

    explainIssue(repositoryUrl, issueNumber) {
      return this.request('/explain-issue', { method: 'POST', body: JSON.stringify({ repository_url: repositoryUrl, issue_number: issueNumber }) });
    },

    generateRoadmap(repositoryUrl, issueNumber, preferences) {
      return this.request('/roadmap', { method: 'POST', body: JSON.stringify({ repository_url: repositoryUrl, issue_number: issueNumber, ...preferences }) });
    },

    askAssistant(repositoryUrl, question, history) {
      return this.request('/chat', { method: 'POST', body: JSON.stringify({ repository_url: repositoryUrl, question, history: history.slice(-8) }) });
    },

    recommendIssues(repositoryUrl, preferences) {
      return this.request('/issues/recommend', { method: 'POST', body: JSON.stringify({ repository_url: repositoryUrl, ...preferences }) });
    },
  };

  function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  }

  function safeGitHubUrl(value) {
    try {
      const url = new URL(value);
      return url.protocol === 'https:' && (url.hostname === 'github.com' || url.hostname.endsWith('.github.com')) ? url.href : '';
    } catch {
      return '';
    }
  }

  function parseRepositoryUrl(value) {
    let url;
    try { url = new URL(value.trim()); } catch { return null; }
    if (url.protocol !== 'https:' || url.hostname.toLowerCase() !== 'github.com' || url.username || url.password || url.port || url.search || url.hash || url.pathname.includes('%')) return null;
    const match = url.pathname.match(/^\/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)\/([A-Za-z0-9_.-]{1,100})\/?$/);
    if (!match) return null;
    const owner = match[1];
    const repository = match[2].replace(/\.git$/i, '');
    if (!repository || repository === '.' || repository === '..') return null;
    return { owner, repository, url: `https://github.com/${owner}/${repository}` };
  }

  function showToast(message, type = 'info') {
    const region = $('#toast-region');
    const toast = document.createElement('div');
    toast.className = `toast${type === 'error' ? ' error' : ''}`;
    toast.textContent = message;
    region.append(toast);
    window.setTimeout(() => toast.remove(), 4800);
  }

  function setApiStatus(label, status = 'neutral') {
    const labelElement = $('#api-status-label');
    labelElement.textContent = label;
    const statusElement = labelElement.closest('.connection-status');
    statusElement.classList.toggle('is-connected', status === 'connected');
    statusElement.classList.toggle('has-error', status === 'error');
  }

  function resetIssueView() {
    state.finderRequestId += 1;
    state.issueRecommendations = null;
    state.finderPreferencesKey = '';
    state.issuesLoaded = false;
    $('#issues-context').hidden = false;
    $('.active-filter span').textContent = '0';
    $('#issues-list').innerHTML = '<div class="state-panel empty-state issue-empty"><span class="state-icon">◎</span><h2>Pick a repository to get started</h2><p>Analyze a GitHub repository to see recent open issues returned by GitHub.</p><button class="button button-primary" type="button" data-nav="analysis">Analyze a repository</button></div>';
    $('#finder-status').textContent = 'Analyze a repository to search its open GitHub issues.';
    $('#find-issues').disabled = true;
    $('#refresh-issues').disabled = true;
    $('#finder-language-options').innerHTML = '<span class="finder-muted">Analyze a repository to load its detected languages.</span>';
  }

  function populateFinderLanguages(languages) {
    const root = $('#finder-language-options');
    const entries = Object.entries(languages || {}).sort((left, right) => Number(right[1]) - Number(left[1]));
    if (!entries.length) {
      root.innerHTML = '<span class="finder-muted">GitHub did not report languages. Add any relevant skills below.</span>';
    } else {
      root.innerHTML = entries.map(([language], index) => `<label class="finder-language-choice"><input type="checkbox" value="${escapeHTML(language)}" id="finder-language-${index}"><span>${escapeHTML(language)}</span></label>`).join('');
    }
    $('#find-issues').disabled = false;
    $('#refresh-issues').disabled = false;
  }

  function issueFinderPreferences(refresh = false) {
    return {
      known_languages: $$('#finder-language-options input:checked').map((input) => input.value),
      experience_level: $('#finder-experience').value,
      contribution_type: $('#finder-contribution').value,
      skills_description: $('#finder-skills').value.trim(),
      refresh,
    };
  }

  function resetChatUI(contextLabel = 'No repository selected') {
    state.chatHistory = [];
    state.chatRequestId += 1;
    $('#chat-context').textContent = contextLabel;
    $('#chat-error').textContent = '';
    $('#ai-assistant-status').textContent = 'Local model status is checked when you send.';
    $('#chat-input').disabled = false;
    $('.send-button', $('#chat-form')).disabled = false;
    $('#chat-messages').innerHTML = '<div class="chat-date">YOUR CONVERSATION</div><div class="message assistant-message"><span class="message-avatar">✳</span><div class="message-bubble"><p>Analyze a repository and configure a local Gemma model to get grounded answers about its structure, issues, and contribution steps.</p><small>RepoRescue assistant · Local model</small></div></div><div class="chat-system-note"><span>ⓘ</span> AI responses are suggestions; verified repository details remain separate.</div>';
  }

  function clearRoadmapSelection() {
    state.roadmapRequestId += 1;
    state.roadmapIssue = null;
    state.roadmapData = null;
    state.roadmapProgressKey = '';
    if ($('#roadmap-empty')) setRoadmapView('empty');
  }

  function navigate(viewName) {
    const target = $(`#view-${viewName}`);
    if (!target) return;
    if (viewName === 'roadmap' && state.selectedIssue && validRoadmapIssue(state.selectedIssue) && state.roadmapIssue?.number !== state.selectedIssue.number) {
      void buildRoadmapForIssue(state.selectedIssue);
      return;
    }
    state.currentView = viewName;
    $$('.view').forEach((view) => {
      const active = view === target;
      view.classList.toggle('is-visible', active);
      view.hidden = !active;
    });
    $$('[data-nav]').forEach((button) => {
      const active = button.dataset.nav === viewName && button.classList.contains('nav-link');
      button.classList.toggle('is-active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    const currentLink = $(`.nav-link[data-nav="${viewName}"]`);
    const navLabel = [...(currentLink?.childNodes || [])].find((node) => node.nodeType === Node.TEXT_NODE)?.textContent.trim();
    $('#page-breadcrumb').textContent = navLabel || viewName[0].toUpperCase() + viewName.slice(1);
    closeMobileNav();
    if (viewName === 'issues') loadIssuesIfReady();
    if (viewName === 'assistant') $('#chat-input').focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function closeMobileNav() {
    $('#sidebar').classList.remove('is-open');
    $('#menu-toggle').setAttribute('aria-expanded', 'false');
    document.body.classList.remove('nav-open');
  }

  function validateRepositoryInput(input, errorElement) {
    const value = input.value.trim();
    if (!value) {
      errorElement.textContent = 'Enter a GitHub repository URL to continue.';
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      return null;
    }
    const parsed = parseRepositoryUrl(value);
    if (!parsed) {
      errorElement.textContent = 'Use a public GitHub URL in the format https://github.com/owner/repository.';
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      return null;
    }
    errorElement.textContent = '';
    input.removeAttribute('aria-invalid');
    return parsed;
  }

  function setAnalysisState(view) {
    $('#analysis-loading').hidden = view !== 'loading';
    $('#analysis-empty').hidden = view !== 'empty';
    $('#analysis-error-state').hidden = view !== 'error';
    $('#analysis-result').hidden = view !== 'result';
  }

  function normalizeRepository(payload, fallback) {
    const source = payload?.repository || payload?.data?.repository || payload?.data || payload || {};
    const readme = payload?.readme || {};
    const languages = payload?.languages && typeof payload.languages === 'object' ? payload.languages : {};
    const languageEntries = Object.entries(languages)
      .map(([name, bytes]) => ({ name, bytes: Number(bytes) || 0 }))
      .sort((left, right) => right.bytes - left.bytes);
    const issues = Array.isArray(payload?.issues) ? payload.issues : [];
    const rootContents = Array.isArray(payload?.root_contents) ? payload.root_contents : [];
    return {
      name: source.name || fallback.repository,
      fullName: source.full_name || source.fullName || `${fallback.owner}/${fallback.repository}`,
      description: source.description || 'This repository does not have a GitHub description.',
      htmlUrl: safeGitHubUrl(source.html_url || source.htmlUrl || fallback.url),
      defaultBranch: source.default_branch || source.defaultBranch || 'Not provided',
      overview: source.description || 'No repository description is available. See the repository README for project details.',
      technologies: languageEntries,
      importantFiles: rootContents,
      setupInstructions: [],
      health: [
        { label: 'README', status: readme.available ? 'available' : 'not found' },
        { label: 'Root directory', status: `${rootContents.length} entries returned` },
        { label: 'Recent open issues', status: `${issues.length} returned` },
      ],
      metadata: source,
      readme,
      issues,
      warnings: Array.isArray(payload?.warnings) ? payload.warnings : [],
    };
  }

  function renderAnalysis(repository, isSample = false) {
    const root = $('#analysis-result');
    const technologies = Array.isArray(repository.technologies) ? repository.technologies : [];
    const files = Array.isArray(repository.importantFiles) ? repository.importantFiles : [];
    const health = Array.isArray(repository.health) ? repository.health : [];
    const originalLink = safeGitHubUrl(repository.htmlUrl);
    const readme = repository.readme || {};
    const metadata = repository.metadata || {};
    const languageBytes = technologies.reduce((total, item) => total + (Number(item.bytes) || 0), 0);
    const formattedNumber = (value) => new Intl.NumberFormat().format(Number(value) || 0);
    const warningMarkup = (repository.warnings || []).map((warning) => `<div class="analysis-warning">ⓘ ${escapeHTML(warning)}</div>`).join('');
    const readmeMarkup = readme.available
      ? `<details class="data-card readme-card"><summary><span><strong>README · ${escapeHTML(readme.path || readme.name || 'project documentation')}</strong><small>Content fetched from GitHub</small></span><span class="readme-expand">Expand</span></summary><pre>${escapeHTML(readme.content || 'GitHub reports a README, but did not return readable text.')}</pre>${readme.truncated ? '<small class="readme-truncated">README preview shortened to keep the response small.</small>' : ''}</details>`
      : `<article class="data-card"><h3>README</h3><p>${isSample ? 'README information is not included in this illustrative preview.' : 'No README was found in this repository.'}</p></article>`;

    root.innerHTML = `
      ${isSample ? '<div class="sample-ribbon"><strong>Illustrative sample preview.</strong> These details are fictional placeholders, not a real repository analysis.</div>' : ''}
      ${warningMarkup}
      <div class="result-banner"><div class="result-title-group"><span class="repo-avatar repo-avatar-blue">⌘</span><div><h2>${escapeHTML(repository.name)}</h2><p>${escapeHTML(repository.description)}</p></div></div>${originalLink ? `<a class="result-open" href="${escapeHTML(originalLink)}" target="_blank" rel="noopener noreferrer">View on GitHub ↗</a>` : '<span class="sample-pill">SAMPLE</span>'}</div>
      <div class="analysis-panels">
        <article class="data-card"><h3>Project overview</h3><p>${escapeHTML(repository.overview)}</p></article>
        <article class="data-card"><h3>Detected languages</h3>${technologies.length ? `<div class="tech-tags">${technologies.map((item) => { const percent = languageBytes ? Math.round((item.bytes / languageBytes) * 100) : 0; return `<span class="tech-tag">${escapeHTML(typeof item === 'string' ? item : item.name || 'Unspecified')}${typeof item === 'string' ? '' : ` · ${percent}%`}</span>`; }).join('')}</div><p class="data-footnote">Language percentages are calculated from GitHub’s byte counts.</p>` : '<p>GitHub did not report language statistics for this repository.</p>'}</article>
        <article class="data-card"><h3>Repository root contents</h3>${files.length ? `<ul class="file-list">${files.map((item) => `<li><span>${escapeHTML(typeof item === 'string' ? item : item.path || item.name || 'Entry')}</span><small>${escapeHTML(typeof item === 'string' ? '' : `${item.type || ''}${item.size ? ` · ${formattedNumber(item.size)} bytes` : item.purpose ? ` · ${item.purpose}` : ''}`)}</small></li>`).join('')}</ul>` : '<p>No root directory entries were returned.</p>'}</article>
        <article class="data-card"><h3>Repository health checklist</h3>${health.length ? `<ul class="health-list">${health.map((item) => { const label = typeof item === 'string' ? item : item.label || item.name || 'Check'; const status = typeof item === 'string' ? 'available' : item.status || item.result || 'unknown'; return `<li><span class="health-dot ${/missing|unknown|example/i.test(status) ? 'neutral' : ''}"></span><span>${escapeHTML(label)} — ${escapeHTML(status)}</span></li>`; }).join('')}</ul>` : '<p>No health checklist was returned by the API.</p>'}</article>
        ${readmeMarkup}
        <article class="data-card"><h3>Repository metadata</h3><ul class="health-list"><li><span>Owner: ${escapeHTML(repository.fullName)}</span></li><li><span>Default branch: ${escapeHTML(repository.defaultBranch)}</span></li><li><span>${formattedNumber(metadata.stargazers_count)} stars · ${formattedNumber(metadata.forks_count)} forks</span></li><li><span>GitHub open-item count (includes pull requests): ${formattedNumber(metadata.open_issues_count)}</span></li>${metadata.license ? `<li><span>License: ${escapeHTML(typeof metadata.license === 'string' ? metadata.license : metadata.license.spdx_id || 'Not specified')}</span></li>` : ''}</ul></article>
        <article class="data-card issue-summary-card"><h3>Recent open issues</h3><p>${formattedNumber(repository.issues?.length || 0)} real issue${repository.issues?.length === 1 ? '' : 's'} returned by GitHub (pull requests excluded).</p><button class="text-button" type="button" data-nav="issues">View issue details <span aria-hidden="true">→</span></button></article>
      </div>
      <section class="ai-analysis-panel" id="ai-analysis-panel" aria-live="polite" hidden></section>`;
    setAnalysisState('result');
  }

  function renderSourceLinks(links) {
    if (!Array.isArray(links) || !links.length) return '';
    return `<div class="ai-source-links"><strong>Sources</strong>${links.map((source) => {
      const url = safeGitHubUrl(source.url);
      return url ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(source.label || 'GitHub source')} ↗</a>` : '';
    }).join('')}</div>`;
  }

  // ── Confidence Check Panel ─────────────────────────────────────────────
  const CC_CRITERION_LABELS = {
    issue_available: 'Issue URL verifiable',
    readme_available: 'README fetched from GitHub',
    file_paths_verified: 'File paths in root listing',
    issue_clarity: 'Issue has sufficient context',
    evidence_agreement: 'Evidence quotes traced to source',
  };

  function renderConfidenceCheck(check, issueUrl) {
    if (!check || typeof check !== 'object') return '';
    const score = Number(check.score) || 0;
    const label = check.status_label || 'Unknown';
    const badgeClass = label === 'Verified Source' ? 'cc-status-verified' : label === 'Partially Supported' ? 'cc-status-partial' : 'cc-status-needs';
    const barClass = score >= 80 ? 'cc-score-high' : score >= 40 ? 'cc-score-mid' : 'cc-score-low';
    const criteria = check.criteria && typeof check.criteria === 'object' ? check.criteria : {};
    const missing = Array.isArray(check.missing_information) ? check.missing_information : [];
    const steps = Array.isArray(check.verification_steps) ? check.verification_steps : [];
    const reasons = Array.isArray(check.reasons) ? check.reasons : [];
    const verifiedPaths = Array.isArray(check.verified_file_paths) ? check.verified_file_paths : [];
    const inferredPaths = Array.isArray(check.inferred_file_paths) ? check.inferred_file_paths : [];
    const supportedQuotes = Array.isArray(check.supported_evidence_quotes) ? check.supported_evidence_quotes : [];
    const panelId = `cc-body-${Math.random().toString(36).slice(2, 7)}`;
    const safeIssueUrl = safeGitHubUrl(issueUrl || '');

    const criteriaMarkup = Object.entries(criteria).map(([key, met]) => {
      const dotClass = met ? 'cc-criterion-met' : 'cc-criterion-unmet';
      return `<div class="cc-criterion"><span class="cc-criterion-dot ${dotClass}" aria-label="${met ? 'Met' : 'Not met'}"></span><span class="cc-criterion-label">${escapeHTML(CC_CRITERION_LABELS[key] || key.replace(/_/g, ' '))}</span></div>`;
    }).join('');

    const pathsMarkup = (verifiedPaths.length || inferredPaths.length) ? `
      <div>
        <div class="cc-section-label">File paths</div>
        <div class="cc-file-tags">
          ${verifiedPaths.map((p) => `<span class="cc-file-tag cc-file-tag-verified" title="Confirmed in GitHub root listing">✓ ${escapeHTML(p)}</span>`).join('')}
          ${inferredPaths.map((p) => `<span class="cc-file-tag cc-file-tag-inferred" title="Inferred by AI — not confirmed in root listing">~ ${escapeHTML(p)} (inferred)</span>`).join('')}
        </div>
      </div>` : '';

    const quotesMarkup = supportedQuotes.length ? `
      <div>
        <div class="cc-section-label">Evidence quotes traced to source</div>
        <ul class="cc-list">${supportedQuotes.map((q) => `<li>"${escapeHTML(q)}"</li>`).join('')}</ul>
      </div>` : '';

    const missingMarkup = missing.length ? `
      <div>
        <div class="cc-section-label">Missing information</div>
        <ul class="cc-list">${missing.map((m) => `<li>${escapeHTML(m)}</li>`).join('')}</ul>
      </div>` : '';

    const stepsMarkup = steps.length ? `
      <div>
        <div class="cc-section-label">Suggested verification steps</div>
        <ul class="cc-list">${steps.map((s) => `<li>${escapeHTML(s)}</li>`).join('')}</ul>
      </div>` : '';

    const reasonsMarkup = reasons.length ? `
      <div>
        <div class="cc-section-label">Why this score was assigned</div>
        <ul class="cc-list">${reasons.map((r) => `<li>${escapeHTML(r)}</li>`).join('')}</ul>
      </div>` : '';

    const unverifiedNotice = label === 'Needs Verification' ? `<p class="cc-unverified-notice">This recommendation is not fully supported by the retrieved evidence. Review the original issue and repository before proceeding.</p>` : '';

    const sourceLink = safeIssueUrl ? `<a class="cc-source-link" href="${escapeHTML(safeIssueUrl)}" target="_blank" rel="noopener noreferrer">Inspect original GitHub issue ↗</a>` : '';

    return `<div class="confidence-check" aria-label="Confidence Check panel">
      <button class="confidence-check-header" type="button" aria-expanded="false" aria-controls="${panelId}"
        onclick="(function(btn){var body=document.getElementById(btn.getAttribute('aria-controls'));var open=body.classList.toggle('is-open');btn.setAttribute('aria-expanded',String(open));})(this)">
        <span class="confidence-check-title">
          <strong>Confidence Check</strong>
          <span class="cc-status-badge ${badgeClass}">${escapeHTML(label)}</span>
          <span class="cc-score-pill">${score}/100</span>
        </span>
        <span class="cc-chevron" aria-hidden="true">▾</span>
      </button>
      <div class="confidence-check-body" id="${panelId}">
        ${unverifiedNotice}
        <div class="cc-score-bar-wrap ${barClass}">
          <div class="cc-score-bar-label"><span>Evidence coverage score</span><strong>${score}/100</strong></div>
          <div class="cc-score-bar"><div class="cc-score-bar-fill" style="width:${score}%"></div></div>
        </div>
        <p class="cc-explanation">${escapeHTML(check.score_explanation || '')}</p>
        <div>
          <div class="cc-section-label">Evidence criteria</div>
          <div class="cc-criteria-grid">${criteriaMarkup}</div>
        </div>
        ${reasonsMarkup}
        ${pathsMarkup}
        ${quotesMarkup}
        ${missingMarkup}
        ${stepsMarkup}
        ${sourceLink}
      </div>
    </div>`;
  }


  function renderAIAnalysis(result) {
    const panel = $('#ai-analysis-panel');
    const analysis = result.analysis || {};
    const files = Array.isArray(analysis.important_files) ? analysis.important_files : [];
    const technologies = Array.isArray(analysis.technologies_used) ? analysis.technologies_used : [];
    const setup = Array.isArray(analysis.setup_instructions) ? analysis.setup_instructions : [];
    const contributions = Array.isArray(analysis.potential_contribution_areas) ? analysis.potential_contribution_areas : [];
    const missing = Array.isArray(analysis.missing_information) ? analysis.missing_information : [];
    const hypotheses = Array.isArray(analysis.hypotheses) ? analysis.hypotheses : [];
    const list = (items) => items.length ? `<ul class="ai-bullet-list">${items.map((item) => `<li>${escapeHTML(typeof item === 'string' ? item : item.text || item.statement || '')}</li>`).join('')}</ul>` : '<p class="ai-empty-note">The model did not return this information.</p>';

    panel.hidden = false;
    panel.innerHTML = `
      <div class="ai-panel-heading"><div><span class="ai-badge">AI-GENERATED INTERPRETATION</span><h2>Repository guide</h2><p>Generated by ${escapeHTML(result.model || 'the configured local model')}. Suggestions are not verified repository facts.</p></div></div>
      <div class="ai-response-grid">
        <article class="ai-section"><h3>Beginner-friendly overview</h3><p>${escapeHTML(analysis.project_overview || 'The model did not provide an overview.')}</p></article>
        <article class="ai-section"><h3>Main purpose <span class="ai-inline-label">interpretation</span></h3><p>${escapeHTML(analysis.main_purpose || 'The model could not determine the purpose from available evidence.')}</p></article>
        <article class="ai-section"><h3>Technologies <span class="ai-inline-label">evidence checked</span></h3>${technologies.length ? `<div class="tech-tags">${technologies.map((item) => `<span class="tech-tag">${escapeHTML(item.name)} · ${escapeHTML(item.source || 'repository evidence')}</span>`).join('')}</div>` : '<p class="ai-empty-note">No technologies were supported by the returned language statistics or README text.</p>'}</article>
        <article class="ai-section"><h3>Important files <span class="ai-inline-label">AI purpose; paths checked against GitHub root</span></h3>${files.length ? `<ul class="ai-bullet-list">${files.map((item) => `<li><strong>${escapeHTML(item.path)}</strong> — ${escapeHTML(item.purpose)}<span class="ai-item-note">${escapeHTML(item.status || 'AI interpretation')}</span></li>`).join('')}</ul>` : '<p class="ai-empty-note">The model did not identify root entries with enough evidence.</p>'}</article>
        <article class="ai-section"><h3>Setup instructions <span class="ai-inline-label">README-backed</span></h3>${setup.length ? `<ul class="ai-bullet-list">${setup.map((item) => `<li>${escapeHTML(item.beginner_explanation)}<blockquote>“${escapeHTML(item.documentation_quote)}”</blockquote><span class="ai-item-note">${escapeHTML(item.source || 'README')} · ${escapeHTML(item.status || 'AI interpretation')}</span></li>`).join('')}</ul>` : '<p class="ai-empty-note">No setup steps could be tied to an exact README quote.</p>'}</article>
        <article class="ai-section"><h3>Potential contribution areas <span class="ai-inline-label">hypotheses</span></h3>${contributions.length ? `<ul class="ai-bullet-list">${contributions.map((item) => `<li>${escapeHTML(item.suggestion)}<blockquote>Evidence: “${escapeHTML(item.evidence_quote)}”</blockquote><span class="ai-item-note">${escapeHTML(item.status || 'Hypothesis')}</span></li>`).join('')}</ul>` : '<p class="ai-empty-note">The model did not find evidence-backed contribution ideas.</p>'}</article>
        <article class="ai-section"><h3>Information to investigate</h3>${list(missing)}</article>
        <article class="ai-section"><h3>Other hypotheses</h3>${list(hypotheses)}</article>
      </div>${renderSourceLinks(result.source_links)}`;
  }

  async function loadRepositoryAI(repositoryUrl, analysisRequestId) {
    const panel = $('#ai-analysis-panel');
    if (!panel) return;
    const aiRequestId = ++state.aiRequestId;
    panel.hidden = false;
    panel.innerHTML = '<div class="state-panel loading-state ai-loading"><span class="spinner" aria-hidden="true"></span><strong>Preparing your repository guide…</strong><p>Asking the configured local Gemma model to summarize the fetched GitHub evidence.</p></div>';
    try {
      const result = await APIService.analyzeRepository(repositoryUrl);
      if (aiRequestId !== state.aiRequestId || analysisRequestId !== state.analysisRequestId || repositoryUrl !== state.lastRepositoryUrl) return;
      renderAIAnalysis(result);
    } catch (error) {
      if (aiRequestId !== state.aiRequestId || analysisRequestId !== state.analysisRequestId || repositoryUrl !== state.lastRepositoryUrl) return;
      panel.innerHTML = `<div class="ai-error-state"><span class="ai-badge">AI GUIDE UNAVAILABLE</span><h2>GitHub details are still available</h2><p>${escapeHTML(error.message)}</p><p>Repository metadata, README, languages, and issue links remain loaded above.</p><button class="button button-secondary" type="button" data-action="retry-ai-analysis">Retry AI guide</button></div>`;
    }
  }

  async function runAnalysis(repositoryUrl) {
    const parsed = parseRepositoryUrl(repositoryUrl);
    if (!parsed) return;
    const analysisRequestId = ++state.analysisRequestId;
    state.aiRequestId += 1;
    state.lastRepositoryUrl = parsed.url;
    state.lastAnalysisError = '';
    state.repository = null;
    state.issues = [];
    state.issuesLoaded = false;
    state.selectedIssue = null;
    clearRoadmapSelection();
    resetChatUI();
    state.isPreviewAnalysis = false;
    resetIssueView();
    $('#analysis-url').value = parsed.url;
    $('#analysis-error').textContent = '';
    navigate('analysis');
    setAnalysisState('loading');
    try {
      const payload = await APIService.getRepository(parsed.url);
      if (analysisRequestId !== state.analysisRequestId) return;
      state.repository = normalizeRepository(payload, parsed);
      state.isPreviewAnalysis = false;
      state.issues = state.repository.issues;
      state.issuesLoaded = true;
      $('#chat-context').textContent = state.repository.fullName;
      populateFinderLanguages(Object.fromEntries(state.repository.technologies.map((language) => [language.name, language.bytes])));
      $('.active-filter span').textContent = String(state.issues.length);
      $('#issues-context').hidden = true;
      setApiStatus('Connected to GitHub', 'connected');
      renderAnalysis(state.repository);
      addRecentRepository(state.repository, false);
      showToast(`Analysis ready for ${state.repository.fullName}.`);
      void loadRepositoryAI(parsed.url, analysisRequestId);
    } catch (error) {
      if (analysisRequestId !== state.analysisRequestId) return;
      state.lastAnalysisError = error.message;
      $('#analysis-error-message').textContent = error.message;
      setApiStatus('API unavailable', 'error');
      resetIssueView();
      setAnalysisState('error');
    }
  }

  function previewAnalysis() {
    state.analysisRequestId += 1;
    state.aiRequestId += 1;
    state.repository = { ...SAMPLE_PREVIEW, technologies: [...SAMPLE_PREVIEW.technologies], importantFiles: [...SAMPLE_PREVIEW.importantFiles], setupInstructions: [...SAMPLE_PREVIEW.setupInstructions], health: [...SAMPLE_PREVIEW.health] };
    state.isPreviewAnalysis = true;
    state.issues = [];
    state.issuesLoaded = false;
    state.selectedIssue = null;
    clearRoadmapSelection();
    resetIssueView();
    $('#analysis-url').value = '';
    resetChatUI('Sample preview only — not a real repository');
    navigate('analysis');
    renderAnalysis(state.repository, true);
  }

  function addRecentRepository(repository, isSample) {
    const row = document.createElement('div');
    row.className = 'recent-row';
    const owner = repository.fullName.split('/')[0] || 'repository';
    const analysedAt = isSample ? 'UI preview only' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date());
    row.innerHTML = `<span class="repo-avatar repo-avatar-blue">⌘</span><span class="recent-repo"><strong>${escapeHTML(repository.name)}</strong><small>${escapeHTML(repository.fullName)}</small></span>${isSample ? '<span class="sample-pill">SAMPLE</span>' : ''}<span class="recent-date">${escapeHTML(analysedAt)}</span>${isSample ? '<button class="row-chevron" type="button" data-action="preview-analysis" aria-label="Preview sample repository">→</button>' : `<a class="row-chevron" href="${escapeHTML(repository.htmlUrl || `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repository.name)}`)}" target="_blank" rel="noopener noreferrer" aria-label="Open repository on GitHub">→</a>`}`;
    $('#recent-empty').hidden = true;
    $('#recent-repositories').prepend(row);
  }


  async function loadIssuesIfReady() {
    if (!state.repository || state.isPreviewAnalysis) {
      resetIssueView();
      return;
    }
    $('#find-issues').disabled = false;
    $('#refresh-issues').disabled = false;
    if (state.issueRecommendations) {
      if (state.finderPreferencesKey === JSON.stringify(issueFinderPreferences(false))) {
        renderIssueRecommendations(state.issueRecommendations);
        return;
      }
    }
    void requestIssueRecommendations(false);
  }

  function renderIssueRecommendations(result) {
    const list = $('#issues-list');
    const recommendations = Array.isArray(result?.recommendations) ? result.recommendations : [];
    const suitableCount = Number(result?.suitable_count) || 0;
    state.issueRecommendations = result;
    state.issues = recommendations.map((recommendation) => recommendation.issue).filter(Boolean);
    state.issuesLoaded = true;
    $('#issues-context').hidden = true;
    $('.active-filter span').textContent = String(recommendations.length);
    $('#issue-sort-note').textContent = 'Ranked by fit, clarity, scope, and evidence';
    if (result?.analysis_error) {
      state.issues = Array.isArray(result.issues) ? result.issues : state.issues;
      clearRoadmapIfIssueUnavailable();
      $('.active-filter span').textContent = String(state.issues.length);
      const unranked = state.issues.map((issue, index) => {
        const sourceUrl = safeGitHubUrl(issue.html_url);
        const labels = Array.isArray(issue.labels) ? issue.labels.map((label) => `<span class="issue-tag">${escapeHTML(label.name || 'GitHub label')}</span>`).join('') : '';
        const openLink = sourceUrl ? `<a class="source-link" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noopener noreferrer">Open original GitHub issue ↗</a>` : '<span class="no-source">GitHub link unavailable</span>';
        const issueActions = sourceUrl ? `<div class="issue-card-actions"><button class="text-button" type="button" data-action="explain-issue" data-issue-index="${index}">Explain issue</button><button class="text-button" type="button" data-action="build-roadmap" data-issue-index="${index}">Build roadmap →</button></div>` : '';
        return `<article class="issue-card"><div class="issue-card-top"><span class="issue-icon">◎</span><span class="issue-number">#${escapeHTML(issue.number)}</span><span class="fit-badge fit-stretch">Unranked · AI unavailable</span></div><h3>${escapeHTML(issue.title)}</h3><p>${escapeHTML(String(issue.body || 'No description provided.').slice(0, 280))}</p><div class="issue-tags">${labels}</div><p class="finder-uncertain-note">No skill match or difficulty estimate was generated. Actual GitHub labels are shown above but do not prove this is beginner-friendly.</p><div class="issue-card-footer">${openLink}${issueActions}</div></article>`;
      }).join('');
      list.innerHTML = `<div class="ai-error-state finder-ai-error"><span class="ai-badge">GEMMA RANKING UNAVAILABLE</span><h2>Showing actual GitHub issues without AI estimates</h2><p>${escapeHTML(result.analysis_error)}</p><p>Difficulty and skill suitability are unassessed; these issues are not ranked.</p></div>${unranked || '<div class="state-panel empty-state"><h2>No valid open issues were returned</h2><p>Refresh GitHub issues to try again.</p></div>'}`;
      $('#finder-status').textContent = 'GitHub issues are available, but Gemma could not assess them.';
      return;
    }
    if (!recommendations.length) {
      clearRoadmapIfIssueUnavailable();
      list.innerHTML = '<div class="state-panel empty-state issue-empty"><span class="state-icon">◎</span><h2>No valid open issues were found</h2><p>GitHub did not return any recent, valid open issues for this repository. Refresh later to check for new issues.</p><button class="button button-secondary" type="button" data-action="refresh-issues">Refresh GitHub issues</button></div>';
      return;
    }
    clearRoadmapIfIssueUnavailable();
    const emptyMatch = suitableCount === 0
      ? '<div class="state-panel empty-state no-match-state"><span class="state-icon">⌕</span><h2>No strong beginner match for these preferences</h2><p>The available issues are still listed below as possible stretches. Try adjusting your profile or add more skills context. GitHub beginner labels are hints, not proof of difficulty.</p></div>'
      : `<div class="finder-results-summary"><strong>${suitableCount} likely match${suitableCount === 1 ? '' : 'es'}</strong><span>Other assessed issues are shown below as possible stretches or lower-fit tasks.</span></div>`;
    const cards = recommendations.map((item, index) => {
      const issue = item.issue || {};
      const labels = Array.isArray(issue.labels) ? issue.labels : [];
      const sourceUrl = safeGitHubUrl(issue.html_url);
      const body = String(issue.body || '').trim();
      const excerpt = body ? `${body.slice(0, 280)}${body.length > 280 ? '…' : ''}` : 'This GitHub issue has no description.';
      const difficulty = item.difficulty_estimate || 'uncertain';
      const fitClass = item.fit_status === 'likely-match' ? 'fit-good' : item.fit_status === 'possible-stretch' ? 'fit-stretch' : 'fit-low';
      const fitLabel = item.fit_status === 'likely-match' ? 'Likely match' : item.fit_status === 'possible-stretch' ? 'Possible stretch' : 'May not fit yet';
      const labelBadges = labels.map((label) => `<span class="issue-tag">${escapeHTML(typeof label === 'string' ? label : label.name || 'GitHub label')}</span>`).join('');
      const labelNote = item.github_label_signals?.length ? `<p class="github-label-note"><strong>GitHub label signal:</strong> ${escapeHTML(item.github_label_signals.join(', '))}. ${escapeHTML(item.label_note || 'This label is not proof that the issue is easy.')}</p>` : `<p class="github-label-note">${escapeHTML(item.label_note || 'No recognized beginner-oriented GitHub label. Gemma assessed this issue anyway.')}</p>`;
      const neededSkills = Array.isArray(item.skills_needed) && item.skills_needed.length ? item.skills_needed.map(escapeHTML).join(', ') : 'Not clear from available evidence';
      const reasons = Array.isArray(item.reasons) ? item.reasons : [];
      const mismatch = Array.isArray(item.not_suitable_reasons) ? item.not_suitable_reasons : [];
      const evidence = Array.isArray(item.evidence_quotes) ? item.evidence_quotes : [];
      const files = Array.isArray(item.relevant_files) ? item.relevant_files : [];
      const breakdown = item.score_breakdown || {};
      const openLink = sourceUrl ? `<a class="source-link" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noopener noreferrer">Open original GitHub issue ↗</a>` : '<span class="no-source">GitHub issue link not available</span>';
      const explainButton = sourceUrl ? `<div class="issue-card-actions"><button class="text-button" type="button" data-action="explain-issue" data-issue-index="${index}">Explain issue</button><button class="text-button" type="button" data-action="build-roadmap" data-issue-index="${index}">Build roadmap →</button></div>` : '';
      const confidencePanel = renderConfidenceCheck(item.confidence_check, issue.html_url);
      return `<article class="issue-card finder-issue-card ${fitClass}">
        <div class="issue-card-top"><span class="issue-icon">◎</span><span class="issue-number">#${escapeHTML(issue.number ?? '—')}</span><span class="fit-badge ${fitClass}">${escapeHTML(fitLabel)}</span><span class="issue-rank">Rank ${escapeHTML(item.rank ?? index + 1)}</span></div>
        <h3>${escapeHTML(issue.title || 'Untitled GitHub issue')}</h3><p>${escapeHTML(excerpt)}</p>
        <div class="issue-tags">${labelBadges}<span class="issue-tag difficulty-tag">Gemma estimate: ${escapeHTML(difficulty)}${difficulty === 'uncertain' ? '' : ` · confidence ${escapeHTML(item.difficulty_confidence ?? '—')}/5`}</span></div>
        ${labelNote}
        <div class="finder-score-row"><strong>Match score: ${escapeHTML(item.match_score ?? '—')}/100</strong><span>AI-estimated, weighted using the breakdown</span></div>
        <div class="finder-score-breakdown"><span>Skills ${escapeHTML(breakdown.skill_compatibility ?? '—')}/5</span><span>Clarity ${escapeHTML(breakdown.task_clarity ?? '—')}/5</span><span>Scope ${escapeHTML(breakdown.scope ?? '—')}/5</span><span>Evidence ${escapeHTML(breakdown.evidence ?? '—')}/5</span></div>
        <div class="finder-assessment-grid"><div><strong>Gemma-estimated skills needed</strong><p>${neededSkills}</p></div><div><strong>Gemma's match reasons</strong>${reasons.length ? `<ul>${reasons.map((reason) => `<li>${escapeHTML(reason)}</li>`).join('')}</ul>` : '<p>No match reasons were returned.</p>'}</div>${mismatch.length ? `<div class="finder-mismatch"><strong>Why it may not suit your current skills</strong><ul>${mismatch.map((reason) => `<li>${escapeHTML(reason)}</li>`).join('')}</ul></div>` : ''}</div>
        ${evidence.length ? `<details class="finder-evidence"><summary>Evidence used <span>${evidence.length} quote${evidence.length === 1 ? '' : 's'}</span></summary><ul>${evidence.map((quote) => `<li>"${escapeHTML(quote)}"</li>`).join('')}</ul></details>` : '<p class="finder-uncertain-note">Evidence was not sufficient to verify a difficulty estimate; treat this assessment as uncertain.</p>'}
        ${files.length ? `<div class="finder-relevant-files"><strong>Possible relevant files <span>AI-suggested, path checked against repository root</span></strong><ul>${files.map((file) => `<li><code>${escapeHTML(file.path)}</code> — ${escapeHTML(file.reason)}</li>`).join('')}</ul></div>` : ''}
        ${confidencePanel}
        <div class="issue-card-footer">${openLink}${explainButton}</div>
      </article>`;
    }).join('');
    list.innerHTML = `${emptyMatch}<div class="finder-ranking-note"><strong>Transparent ranking:</strong> ${escapeHTML(result?.ranking_method?.formula || '20 × (skills 40% + clarity 25% + scope 20% + evidence 15%)')}. Labels do not increase the score and are not treated as proof of ease. Scores and difficulty are Gemma estimates.</div>${cards}`;
  }


  async function requestIssueRecommendations(refresh = false) {
    if (!state.repository || state.isPreviewAnalysis || !state.lastRepositoryUrl) {
      resetIssueView();
      return;
    }
    const requestId = ++state.finderRequestId;
    const repositoryUrl = state.lastRepositoryUrl;
    const preferences = issueFinderPreferences(refresh);
    const form = $('#finder-form');
    const submitButton = $('#find-issues');
    const refreshButton = $('#refresh-issues');
    submitButton.disabled = true;
    refreshButton.disabled = true;
    $('#issues-context').hidden = true;
    $('#finder-status').textContent = refresh
      ? 'Refreshing open issues from GitHub, then asking Gemma to rank them…'
      : 'Gemma is comparing real open issues with your preferences…';
    $('#issues-list').innerHTML = `<div class="state-panel loading-state"><span class="spinner" aria-hidden="true"></span><strong>${refresh ? 'Refreshing GitHub issues…' : 'Finding issues that fit you…'}</strong><p>Filtering to real open issues, then evaluating skills, clarity, scope, and evidence.</p></div>`;
    try {
      const result = await APIService.recommendIssues(repositoryUrl, preferences);
      if (requestId !== state.finderRequestId || repositoryUrl !== state.lastRepositoryUrl) return;
      state.finderPreferencesKey = JSON.stringify({ ...preferences, refresh: false });
      renderIssueRecommendations(result);
      $('#finder-status').textContent = result.analysis_error
        ? `Gemma unavailable · ${result.candidate_count || 0} real GitHub issue${result.candidate_count === 1 ? '' : 's'} shown unranked.`
        : `${result.suitable_count || 0} likely match${result.suitable_count === 1 ? '' : 'es'} from ${result.candidate_count || 0} open issue${result.candidate_count === 1 ? '' : 's'} · ${result.model ? `assessed by ${result.model}` : 'no AI assessment needed'}.`;
      if (refresh) showToast('GitHub issues refreshed.');
    } catch (error) {
      if (requestId !== state.finderRequestId || repositoryUrl !== state.lastRepositoryUrl) return;
      $('#finder-status').textContent = 'Could not complete the personalized issue search.';
      $('#issues-list').innerHTML = `<div class="state-panel error-state"><span class="state-icon">!</span><h2>Issue recommendations are unavailable</h2><p>${escapeHTML(error.message)}</p><p>Your repository analysis and GitHub issue links are not changed.</p><button class="button button-secondary" type="button" data-action="retry-finder">Try again</button></div>`;
    } finally {
      if (requestId === state.finderRequestId && form) {
        submitButton.disabled = false;
        refreshButton.disabled = false;
      }
    }
  }

  function showExplainer(issue, explanation, isSample = false) {
    const empty = $('#explainer-empty');
    const content = $('#explainer-content');
    const files = explanation?.relevant_files || explanation?.relevantFiles || [];
    const approach = explanation?.implementation_approach || explanation?.suggested_approach || explanation?.suggestedApproach || [];
    const testing = explanation?.testing_checklist || explanation?.testingChecklist || [];
    const relevantFiles = Array.isArray(files) ? files : [];
    const approachItems = Array.isArray(approach) ? approach : [approach];
    const testingItems = Array.isArray(testing) ? testing : [testing];
    const sourceUrl = safeGitHubUrl(issue?.html_url || issue?.url);
    const overview = explanation?.simple_explanation || explanation?.explanation || explanation?.summary || (isSample ? 'This is an illustrative explanation layout. In a connected app, the issue description would be rewritten in plain language. No real issue was selected.' : 'The API did not return a plain-language explanation.');
    const why = explanation?.why_it_matters || explanation?.whyItMatters || (isSample ? 'This area explains the user or maintainer impact, based on the verified issue details.' : 'The API did not return why this issue matters.');
    content.innerHTML = `${isSample ? '<div class="sample-ribbon"><strong>Illustrative sample only.</strong> No real issue or repository is represented below.</div>' : `<div class="analysis-warning">AI-generated interpretation${explanation?.interpretation_notice ? ` — ${escapeHTML(explanation.interpretation_notice)}` : '. Verify suggestions against the linked GitHub issue.'}</div>`}<div class="explainer-layout"><div class="explainer-main"><div class="explainer-issue-head"><span class="issue-tag tag-blue">${isSample ? 'SAMPLE PREVIEW' : `ISSUE #${escapeHTML(issue.number ?? '—')}`}</span><h2>${escapeHTML(issue.title || 'Example issue explanation')}</h2><p>${isSample ? 'No real GitHub issue link is available in this preview.' : `Repository: ${escapeHTML(state.repository?.fullName || 'Connected repository')}`}</p>${sourceUrl ? `<a class="source-link" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noopener noreferrer">View original issue on GitHub ↗</a>` : ''}</div><article class="explainer-block"><h3>In simple terms</h3><p>${escapeHTML(overview)}</p></article><article class="explainer-block"><h3>Why this matters <span class="ai-inline-label">interpretation</span></h3><p>${escapeHTML(why)}</p></article><article class="explainer-block"><h3>Expected outcome <span class="ai-inline-label">interpretation</span></h3><p>${escapeHTML(explanation?.expected_outcome || 'The expected outcome is not clear from the available issue evidence.')}</p></article><article class="explainer-block"><h3>Suggested approach <span class="ai-inline-label">hypothesis</span></h3>${approachItems.filter(Boolean).length ? `<ul class="checklist">${approachItems.filter(Boolean).map((item) => `<li>${escapeHTML(typeof item === 'string' ? item : item.step || item.description || 'Review the issue requirements.')}</li>`).join('')}</ul>` : '<p>No implementation approach was provided by the model.</p>'}</article></div><aside class="explainer-sidebar"><article class="explainer-block"><h3>Relevant files <span class="ai-inline-label">AI candidates</span></h3>${relevantFiles.length ? `<ul class="checklist">${relevantFiles.map((file) => `<li><span><span class="file-reference">${escapeHTML(typeof file === 'string' ? file : file.path || file.name || 'File')}</span>${typeof file === 'string' || !file.reason ? '' : `<br>${escapeHTML(file.reason)}${file.status ? `<br><small>${escapeHTML(file.status)}</small>` : ''}`}</span></li>`).join('')}</ul>` : `<p>${isSample ? 'File paths are intentionally not invented for the sample.' : 'The model found no file paths supported by both the issue/README evidence and GitHub root listing.'}</p>`}</article><article class="explainer-block"><h3>Investigation steps</h3>${renderExplainerList(explanation?.investigation_steps)}</article><article class="explainer-block"><h3>Testing checklist <span class="ai-inline-label">suggestions</span></h3>${testingItems.filter(Boolean).length ? `<ul class="checklist">${testingItems.filter(Boolean).map((item) => `<li>${escapeHTML(typeof item === 'string' ? item : item.check || item.description || 'Run the relevant project tests.')}</li>`).join('')}</ul>` : '<p>No testing checklist was provided by the model.</p>'}</article><article class="explainer-block"><h3>Questions to clarify</h3>${renderExplainerList(explanation?.questions_to_clarify)}</article><article class="explainer-block"><h3>Information still missing</h3>${renderExplainerList(explanation?.missing_information)}</article><button class="button button-primary" type="button" data-nav="roadmap">Add to contribution roadmap →</button></aside></div>`;
    if (!isSample) {
      const sourceMarkup = renderSourceLinks(explanation?.source_links || []);
      if (sourceMarkup) content.insertAdjacentHTML('beforeend', sourceMarkup);
      const confidenceCheck = explanation?.confidence_check;
      if (confidenceCheck) {
        content.insertAdjacentHTML('beforeend', renderConfidenceCheck(confidenceCheck, issue?.html_url));
      }
    }
    empty.hidden = true;
    content.hidden = false;
    navigate('explainer');
  }

  function renderExplainerList(items) {
    if (!Array.isArray(items) || !items.length) return '<p class="ai-empty-note">No items were returned for this section.</p>';
    return `<ul class="ai-bullet-list">${items.map((item) => `<li>${escapeHTML(item)}</li>`).join('')}</ul>`;
  }

  async function explainSelectedIssue(index) {
    const issue = state.issues[index];
    if (!issue || !state.repository) return;
    state.selectedIssue = issue;
    navigate('explainer');
    $('#explainer-empty').hidden = true;
    $('#explainer-content').hidden = false;
    $('#explainer-content').innerHTML = '<div class="state-panel loading-state"><span class="spinner" aria-hidden="true"></span><strong>Making this issue easier to understand…</strong><p>Checking the issue context with the API.</p></div>';
    try {
      const result = await APIService.explainIssue(state.lastRepositoryUrl, issue.number);
      showExplainer(result.issue || issue, { ...result.explanation, source_links: result.source_links, confidence_check: result.confidence_check }, false);
    } catch (error) {
      $('#explainer-content').innerHTML = `<div class="state-panel error-state"><span class="state-icon">!</span><h2>Couldn’t explain this issue</h2><p>${escapeHTML(error.message)}</p><button class="button button-secondary" type="button" data-action="retry-explainer">Try again</button></div>`;
      state.selectedIssue = issue;
    }
  }

  function previewExplainer() {
    state.selectedIssue = null;
    showExplainer({ title: 'Example issue explanation', number: '—' }, {
      simple_explanation: 'This illustrative card shows how a technical issue could be restated in everyday language. It is not based on a real repository or GitHub issue.',
      why_it_matters: 'A clear explanation helps a new contributor understand the goal before touching project code.',
      relevant_files: [],
      suggested_approach: ['Read the original issue carefully (a real issue would be linked here).', 'Ask a maintainer a focused question if the expected behavior is unclear.', 'Make one small change and compare it with the project conventions.'],
      testing_checklist: ['Run the checks documented by the project.', 'Review the final diff and confirm the change addresses the requested behavior.'],
    }, true);
  }

  const ROADMAP_STEP_KEYS = [
    'understand_issue', 'expected_result', 'locate_files', 'read_guidelines', 'setup_environment',
    'create_branch', 'investigate', 'implement', 'run_tests', 'review_diff', 'prepare_pull_request',
  ];

  function roadmapStorageKey(issue = state.roadmapIssue) {
    if (!issue || !state.repository?.fullName) return '';
    return `${ROADMAP_STORAGE_PREFIX}${encodeURIComponent(state.repository.fullName.toLowerCase())}:${issue.number}`;
  }

  function setRoadmapView(view) {
    $('#roadmap-empty').hidden = view !== 'empty';
    $('#roadmap-loading').hidden = view !== 'loading';
    $('#roadmap-error').hidden = view !== 'error';
    $('#roadmap-content').hidden = view !== 'content';
  }

  function roadmapProfile() {
    const finderProfile = issueFinderPreferences(false);
    return {
      known_languages: finderProfile.known_languages,
      experience_level: finderProfile.experience_level,
      skills_description: finderProfile.skills_description,
    };
  }

  function validRoadmapIssue(issue) {
    if (!issue || issue.state !== 'open' || !Number.isInteger(issue.number) || issue.number < 1) return false;
    const issueUrl = safeGitHubUrl(issue.html_url);
    if (!issueUrl || !state.repository?.fullName) return false;
    try {
      const parsed = new URL(issueUrl);
      const ownerAndRepo = state.repository.fullName.toLowerCase();
      return parsed.pathname.toLowerCase() === `/${ownerAndRepo}/issues/${issue.number}`;
    } catch {
      return false;
    }
  }
  function clearRoadmapIfIssueUnavailable() {
    const selected = state.roadmapIssue || state.selectedIssue;
    if (selected && !state.issues.some((issue) => issue.number === selected.number && issue.state === 'open')) {
      state.selectedIssue = null;
      clearRoadmapSelection();
    }
  }

  function sourceLinksMarkup(links, allowedPaths = null) {
    if (!Array.isArray(links)) return '';
    const filtered = links.filter((source) => {
      const url = safeGitHubUrl(source?.url);
      if (!url) return false;
      if (!allowedPaths) return true;
      return allowedPaths.includes(source.label) || allowedPaths.some((path) => source.label === path);
    });
    if (!filtered.length) return '';
    return `<div class="roadmap-source-list">${filtered.map((source) => `<a href="${escapeHTML(safeGitHubUrl(source.url))}" target="_blank" rel="noopener noreferrer">${escapeHTML(source.label || 'GitHub documentation')} ↗</a>`).join('')}</div>`;
  }

  function renderRoadmap(result) {
    const stepsRoot = $('#roadmap-steps');
    const generatedSteps = Array.isArray(result.steps) ? result.steps : [];
    const byKey = new Map(generatedSteps.filter((step) => ROADMAP_STEP_KEYS.includes(step.key)).map((step) => [step.key, step]));
    const sources = Array.isArray(result.source_links) ? result.source_links : [];
    const issue = result.issue || state.roadmapIssue;
    const repository = result.repository || state.repository || {};
    const issueLink = safeGitHubUrl(issue?.html_url);

    state.roadmapData = result;
    state.roadmapProgressKey = roadmapStorageKey(issue);
    $('#roadmap-repo-name').textContent = repository.full_name || state.repository?.fullName || 'Selected repository';
    $('#roadmap-repo-caption').textContent = `Issue #${issue?.number ?? '—'} · ${issue?.title || 'Selected GitHub issue'} · tailored to your finder profile.`;
    $('#roadmap-issue-source').innerHTML = issueLink
      ? `<a class="source-link" href="${escapeHTML(issueLink)}" target="_blank" rel="noopener noreferrer">View the selected issue on GitHub ↗</a>`
      : '';
    $('#roadmap-execution-notice').textContent = result.execution_notice || 'Commands are suggestions only. RepoRescue AI did not run them.';
    $('#roadmap-model-label').textContent = result.model ? `Gemma roadmap · ${result.model} · all steps are suggestions` : 'AI-generated roadmap · verify against sources';
    $('#roadmap-source-links').innerHTML = sourceLinksMarkup(sources);

    stepsRoot.innerHTML = ROADMAP_STEP_KEYS.map((key, index) => {
      const step = byKey.get(key);
      if (!step) {
        const missingStepId = `roadmap-step-${key}`;
        return `<li class="roadmap-step roadmap-step-missing"><input id="${missingStepId}" type="checkbox" data-roadmap-step="${key}" aria-label="Mark step ${index + 1}: ${escapeHTML(key.replaceAll('_', ' '))} complete"><label class="step-check" for="${missingStepId}" aria-hidden="true"></label><span class="step-number">${String(index + 1).padStart(2, '0')}</span><div class="step-body"><strong>${escapeHTML(key.replaceAll('_', ' '))}</strong><small>This step was not returned by the model. Check repository guidance and the issue manually.</small></div><span class="roadmap-step-status">Needs review</span></li>`;
      }

      const stepId = `roadmap-step-${key}`;
      const requiredBadge = step.required
        ? '<span class="step-tag">REQUIRED</span>'
        : `<span class="step-tag optional-step-tag">OPTIONAL${step.optional_reason ? ` · ${escapeHTML(step.optional_reason)}` : ''}</span>`;
      const evidence = Array.isArray(step.evidence_quotes) ? step.evidence_quotes : [];
      const safetyWarnings = Array.isArray(step.safety_warnings) ? step.safety_warnings : [];
      const sourcePaths = Array.isArray(step.source_paths) ? step.source_paths : [];
      const fileReferences = Array.isArray(step.relevant_files) ? step.relevant_files : [];
      const linkedPathLabels = [...new Set([...sourcePaths, ...fileReferences.map((file) => file.path)])];
      const pathLinks = sourceLinksMarkup(Array.isArray(step.source_links) ? step.source_links : sources, linkedPathLabels);
      const commandMarkup = (Array.isArray(step.commands) ? step.commands : []).map((command) => {
        const value = String(command.command || '');
        const destructive = Boolean(command.destructive) || /(?:\bgit\s+(?:reset|clean|push|restore|worktree\s+remove)\b|\bgit\s+branch\s+-[a-z]*d\b|\bgit\s+checkout\s+--\b|\b(?:rm|rmdir|del|remove-item|sudo|format(?:\.com)?)\b|--force(?:-with-lease)?|--hard)/i.test(value);
        return `<div class="roadmap-command ${destructive ? 'is-destructive' : ''}"><code>${escapeHTML(value)}</code><div><small>${escapeHTML(command.explanation || '')}</small><button class="button button-secondary copy-command" type="button" data-action="copy-roadmap-command" data-command="${escapeHTML(value)}" data-destructive="${destructive}" aria-label="Copy suggested command for ${escapeHTML(step.title)}">Copy command</button></div><span class="command-notice">Suggested only · not executed${command.source_path ? ` · ${escapeHTML(command.source_path)}` : ''}</span>${destructive ? '<strong class="destructive-warning">Potentially destructive: review carefully before copying or running.</strong>' : ''}</div>`;
      }).join('');
      const relevantMarkup = fileReferences.length
        ? `<div class="roadmap-relevant-files"><strong>Relevant files · AI-suggested, source-checked</strong><ul>${fileReferences.map((file) => `<li><code>${escapeHTML(file.path)}</code> — ${escapeHTML(file.reason)}</li>`).join('')}</ul></div>`
        : '';
      const evidenceMarkup = evidence.length
        ? `<details class="roadmap-evidence"><summary>Repository evidence <span>${evidence.length} quote${evidence.length === 1 ? '' : 's'}</span></summary><ul>${evidence.map((quote) => `<li>“${escapeHTML(quote)}”</li>`).join('')}</ul></details>`
        : '';
      const safetyMarkup = safetyWarnings.map((warning) => `<p class="roadmap-safety-warning"><strong>Safety filter:</strong> ${escapeHTML(warning)}</p>`).join('');
      const missingCommandNote = ['setup_environment', 'create_branch', 'run_tests'].includes(key) && !commandMarkup
        ? '<p class="roadmap-no-command">No repository-specific command was supported by available setup documentation. Check the linked guide; no command is assumed.</p>'
        : '';
      const stepProgress = `<input id="${stepId}" type="checkbox" data-roadmap-step="${escapeHTML(key)}" aria-label="Mark step ${index + 1}: ${escapeHTML(step.title)} complete"><label class="step-check" for="${stepId}" aria-hidden="true"></label>`;
      return `<li class="roadmap-step" data-roadmap-item="${escapeHTML(key)}">${stepProgress}<span class="step-number">${String(index + 1).padStart(2, '0')}</span><div class="step-body"><div class="roadmap-step-heading"><strong>${escapeHTML(step.title || key.replaceAll('_', ' '))}</strong>${requiredBadge}</div><p class="roadmap-guidance">${escapeHTML(step.guidance || 'Review the issue and repository instructions before this step.')}</p><span class="roadmap-step-status">${escapeHTML(step.status || 'AI suggestion · review evidence')}</span>${step.optional_reason && !step.required ? `<p class="roadmap-optional-reason">Optional because: ${escapeHTML(step.optional_reason)}</p>` : ''}${pathLinks}${relevantMarkup}${evidenceMarkup}${safetyMarkup}${commandMarkup}${missingCommandNote}</div></li>`;
    }).join('');

    const missing = Array.isArray(result.missing_information) ? result.missing_information : [];
    const assumptions = Array.isArray(result.assumptions_to_verify) ? result.assumptions_to_verify : [];
    const notes = [...missing.map((item) => `Still unknown: ${item}`), ...assumptions.map((item) => `Clarify: ${item}`)];
    const missingRoot = $('#roadmap-missing-info');
    missingRoot.hidden = notes.length === 0;
    missingRoot.innerHTML = notes.length ? `<strong>Details to verify</strong><ul>${notes.map((note) => `<li>${escapeHTML(note)}</li>`).join('')}</ul>` : '';
    loadRoadmapProgress();
    setRoadmapView('content');
  }

  async function buildRoadmapForIssue(issue) {
    if (!state.repository || state.isPreviewAnalysis || !validRoadmapIssue(issue)) {
      state.roadmapIssue = null;
      setRoadmapView('empty');
      showToast('Choose a real, open GitHub issue from this repository first.', 'error');
      return;
    }
    const requestId = ++state.roadmapRequestId;
    const repositoryUrl = state.lastRepositoryUrl;
    state.roadmapIssue = issue;
    state.selectedIssue = issue;
    state.roadmapData = null;
    state.roadmapProgressKey = roadmapStorageKey(issue);
    navigate('roadmap');
    $('#roadmap-repo-name').textContent = state.repository.fullName;
    $('#roadmap-repo-caption').textContent = `Issue #${issue.number} · ${issue.title}`;
    $('#roadmap-issue-source').innerHTML = `<a class="source-link" href="${escapeHTML(safeGitHubUrl(issue.html_url))}" target="_blank" rel="noopener noreferrer">View the selected issue on GitHub ↗</a>`;
    setRoadmapView('loading');
    try {
      const result = await APIService.generateRoadmap(repositoryUrl, issue.number, roadmapProfile());
      if (requestId !== state.roadmapRequestId || repositoryUrl !== state.lastRepositoryUrl || state.roadmapIssue?.number !== issue.number) return;
      renderRoadmap(result);
    } catch (error) {
      if (requestId !== state.roadmapRequestId) return;
      $('#roadmap-error-message').textContent = error.message;
      setRoadmapView('error');
    }
  }

  function updateRoadmapProgress() {
    const checkboxes = $$('[data-roadmap-step]', $('#roadmap-steps'));
    const completed = checkboxes.filter((checkbox) => checkbox.checked).length;
    const total = checkboxes.length;
    const percentage = total ? Math.round((completed / total) * 100) : 0;
    $('#progress-count').innerHTML = `${completed} <small>/ ${total}</small>`;
    $('#progress-fill').style.width = `${percentage}%`;
    $('#progress-percent').textContent = `${percentage}% complete`;
    $('.progress-track').setAttribute('aria-valuenow', String(percentage));
    checkboxes.forEach((checkbox) => checkbox.closest('.roadmap-step').classList.toggle('is-complete', checkbox.checked));
    if (state.roadmapProgressKey) {
      const saved = Object.fromEntries(checkboxes.map((checkbox) => [checkbox.dataset.roadmapStep, checkbox.checked]));
      try { localStorage.setItem(state.roadmapProgressKey, JSON.stringify(saved)); } catch { /* Private browsing may block local storage. */ }
    }
  }

  function loadRoadmapProgress() {
    let saved = {};
    try {
      if (state.roadmapProgressKey) saved = JSON.parse(localStorage.getItem(state.roadmapProgressKey) || '{}');
    } catch { /* Start with an unchecked roadmap if storage is unavailable or invalid. */ }
    $$('[data-roadmap-step]', $('#roadmap-steps')).forEach((checkbox) => { checkbox.checked = Boolean(saved[checkbox.dataset.roadmapStep]); });
    updateRoadmapProgress();
  }

  function resetRoadmap() {
    if (!state.roadmapIssue || !state.roadmapProgressKey) {
      showToast('Choose an issue to reset its roadmap.', 'error');
      return;
    }
    $$('[data-roadmap-step]', $('#roadmap-steps')).forEach((checkbox) => { checkbox.checked = false; });
    updateRoadmapProgress();
    showToast(`Progress reset for issue #${state.roadmapIssue.number}.`);
  }

  async function copyRoadmapCommand(button) {
    const command = button.dataset.command || '';
    if (!command) return;
    if (button.dataset.destructive === 'true') {
      const confirmed = window.confirm('This command may be destructive. Copy only if you understand the possible effects. RepoRescue will not run it. Continue copying?');
      if (!confirmed) return;
    }
    try {
      await navigator.clipboard.writeText(command);
      showToast('Suggested command copied. It was not executed.');
    } catch {
      showToast('Clipboard access was denied. Select and copy the command manually.', 'error');
    }
  }

  function addChatMessage(text, role, options = {}) {
    const messages = $('#chat-messages');
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
    const avatar = document.createElement('span');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'G' : '✳';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    const paragraph = document.createElement('p');
    paragraph.textContent = text;
    bubble.append(paragraph);
    if (options.error) bubble.classList.add('message-error');
    if (role !== 'user') {
      const meta = document.createElement('small');
      meta.textContent = options.error ? 'Connection error' : 'RepoRescue assistant';
      bubble.append(meta);
    }
    wrapper.append(avatar, bubble);
    messages.append(wrapper);
    messages.scrollTop = messages.scrollHeight;
  }

  function addAssistantResponse(response) {
    const messages = $('#chat-messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant-message';
    const avatar = document.createElement('span');
    avatar.className = 'message-avatar';
    avatar.textContent = '✳';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble ai-chat-bubble';
    const answer = document.createElement('p');
    answer.className = 'ai-chat-answer';
    answer.textContent = response.answer || 'The model returned no answer.';
    bubble.append(answer);

    const addList = (heading, entries, includeEvidence = false) => {
      if (!Array.isArray(entries) || !entries.length) return;
      const section = document.createElement('section');
      section.className = 'ai-chat-section';
      const title = document.createElement('strong');
      title.textContent = heading;
      section.append(title);
      const list = document.createElement('ul');
      entries.forEach((entry) => {
        const item = document.createElement('li');
        if (typeof entry === 'string') {
          item.textContent = entry;
        } else {
          item.append(document.createTextNode(entry.statement || ''));
          if (includeEvidence && entry.evidence_quote) {
            const quote = document.createElement('small');
            quote.textContent = `Evidence: “${entry.evidence_quote}”`;
            item.append(quote);
          }
        }
        list.append(item);
      });
      section.append(list);
      bubble.append(section);
    };

    addList('Facts matched to repository evidence', response.verified_facts, true);
    addList('Hypotheses — verify before acting', response.hypotheses);
    addList('Information not available', response.missing_information);
    const sources = (Array.isArray(response.source_links) ? response.source_links : [])
      .map((source) => ({ ...source, url: safeGitHubUrl(source.url) }))
      .filter((source) => source.url);
    if (sources.length) {
      const sourceList = document.createElement('div');
      sourceList.className = 'ai-chat-sources';
      const label = document.createElement('strong');
      label.textContent = 'GitHub sources';
      sourceList.append(label);
      sources.forEach((source) => {
        const link = document.createElement('a');
        link.href = source.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = `${source.label || 'GitHub source'} ↗`;
        sourceList.append(link);
      });
      bubble.append(sourceList);
    }
    const meta = document.createElement('small');
    meta.textContent = `${response.model || 'Local Gemma'} · AI-generated; verify against sources`;
    bubble.append(meta);
    wrapper.append(avatar, bubble);
    messages.append(wrapper);
    messages.scrollTop = messages.scrollHeight;
  }

  async function submitChat(question) {
    const trimmed = question.trim();
    if (!trimmed) return;
    const requestId = ++state.chatRequestId;
    const input = $('#chat-input');
    const sendButton = $('.send-button', $('#chat-form'));
    const error = $('#chat-error');
    error.textContent = '';
    $('#ai-assistant-status').textContent = 'Asking the configured local model…';
    addChatMessage(trimmed, 'user');
    input.value = '';
    input.disabled = true;
    sendButton.disabled = true;
    const loading = document.createElement('div');
    loading.className = 'chat-loading';
    loading.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Checking the connected assistant…</span>';
    $('#chat-messages').append(loading);
    try {
      if (!state.repository || state.isPreviewAnalysis) throw new Error('Analyze a real repository first. Sample preview data cannot be sent to the assistant.');
      const response = await APIService.askAssistant(state.lastRepositoryUrl, trimmed, state.chatHistory);
      const answer = response?.answer || response?.message || response?.response;
      if (!answer) throw new Error('The API response did not include an answer.');
      if (requestId !== state.chatRequestId) return;
      state.chatHistory.push({ role: 'user', content: trimmed }, { role: 'assistant', content: answer });
      addAssistantResponse(response);
      $('#ai-assistant-status').textContent = `Connected · ${response.model || 'local model'}`;
    } catch (requestError) {
      if (requestId !== state.chatRequestId) return;
      addChatMessage(requestError.message, 'assistant', { error: true });
      error.textContent = 'The assistant could not answer this message.';
      $('#ai-assistant-status').textContent = 'Check Ollama and the configured model';
    } finally {
      if (requestId === state.chatRequestId) {
        loading.remove();
        input.disabled = false;
        sendButton.disabled = false;
        input.focus();
      }
    }
  }

  function bindEvents() {
    $('#repository-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const input = $('#repository-url');
      const parsed = validateRepositoryInput(input, $('#repository-error'));
      if (parsed) runAnalysis(parsed.url);
    });
    $('#analysis-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const input = $('#analysis-url');
      const parsed = validateRepositoryInput(input, $('#analysis-error'));
      if (parsed) {
        $('#repository-url').value = parsed.url;
        runAnalysis(parsed.url);
      }
    });
    $('#finder-form').addEventListener('submit', (event) => {
      event.preventDefault();
      if (state.repository && !state.isPreviewAnalysis) void requestIssueRecommendations(false);
    });
    $('#repository-url').addEventListener('input', () => {
      $('#repository-error').textContent = '';
      $('#repository-url').removeAttribute('aria-invalid');
      $('#clear-url').hidden = !$('#repository-url').value;
    });
    $('#clear-url').addEventListener('click', () => {
      $('#repository-url').value = '';
      $('#clear-url').hidden = true;
      $('#repository-url').focus();
    });

    document.addEventListener('click', (event) => {
      if (document.body.classList.contains('nav-open') && event.target === document.body) {
        closeMobileNav();
        return;
      }
      const navigation = event.target.closest('[data-nav]');
      if (navigation) {
        event.preventDefault();
        navigate(navigation.dataset.nav);
        return;
      }
      const action = event.target.closest('[data-action]');
      if (!action) return;
      switch (action.dataset.action) {
        case 'preview-analysis': previewAnalysis(); break;
        case 'preview-explainer': previewExplainer(); break;
        case 'retry-analysis': if (state.lastRepositoryUrl) runAnalysis(state.lastRepositoryUrl); break;
        case 'retry-ai-analysis': if (state.lastRepositoryUrl && state.repository) void loadRepositoryAI(state.lastRepositoryUrl, state.analysisRequestId); break;
        case 'refresh-issues': void requestIssueRecommendations(true); break;
        case 'retry-finder': void requestIssueRecommendations(false); break;
        case 'explain-issue': explainSelectedIssue(Number(action.dataset.issueIndex)); break;
        case 'build-roadmap': void buildRoadmapForIssue(state.issues[Number(action.dataset.issueIndex)]); break;
        case 'build-roadmap-selected': if (state.selectedIssue) void buildRoadmapForIssue(state.selectedIssue); else showToast('Choose a real GitHub issue before building a roadmap.', 'error'); break;
        case 'retry-roadmap': if (state.roadmapIssue) void buildRoadmapForIssue(state.roadmapIssue); break;
        case 'regenerate-roadmap': if (state.roadmapIssue) void buildRoadmapForIssue(state.roadmapIssue); break;
        case 'copy-roadmap-command': void copyRoadmapCommand(action); break;
        case 'retry-explainer': if (state.selectedIssue) explainSelectedIssue(state.issues.indexOf(state.selectedIssue)); break;
        case 'reset-roadmap': resetRoadmap(); break;
      }
    });

    $('#roadmap-steps').addEventListener('change', (event) => {
      if (event.target.matches('[data-roadmap-step]')) updateRoadmapProgress();
    });
    $('#chat-form').addEventListener('submit', (event) => {
      event.preventDefault();
      submitChat($('#chat-input').value);
    });
    $('#chat-input').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        $('#chat-form').requestSubmit();
      }
    });
    $$('.suggestion-chip').forEach((button) => button.addEventListener('click', () => {
      $('#chat-input').value = button.dataset.question;
      $('#chat-input').focus();
    }));
    $('#menu-toggle').addEventListener('click', () => {
      const isOpen = $('#sidebar').classList.toggle('is-open');
      $('#menu-toggle').setAttribute('aria-expanded', String(isOpen));
      document.body.classList.toggle('nav-open', isOpen);
    });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeMobileNav(); });
  }

  function initialize() {
    bindEvents();
    loadRoadmapProgress();
  }

  initialize();
})();
