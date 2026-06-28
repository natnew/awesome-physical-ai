# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository is a public, maintained awesome list for Physical AI and embodied AI, not an application codebase. The `README.md` is the product. There is no application build or unit-test suite; the only checks are content checks (link checking, markdown lint, entry counts) and the standalone documentation site under `website/`.

Claude Code should read this file first, then use `AGENTS.md` as the shared repository operating protocol.

## North Star

* Preserve `README.md` as the canonical public artefact.
* Keep the list selective, durable, technically useful, neutral, and easy to scan.
* Help the maintainer make fast, consistent, low-friction decisions.
* Prefer small, precise edits over broad rewrites.
* Do not broaden the list beyond Physical AI, embodied AI, robotics, and clearly adjacent technical areas already represented in the README.

## Claude's Role

Claude may assist with:

* PR review
* Issue triage
* README entry review
* Broken-link investigation
* Duplicate detection
* Section placement
* Neutral description rewrites
* Maintainer comment drafts
* Small safe maintainer edits when explicitly asked
* Improvements to agent instruction files when asked

Claude must not:

* Add entries without checking scope, link quality, duplicates, and placement
* Invent facts about a resource
* Preserve promotional claims
* Add ranking, pricing, novelty, adoption, or performance claims without strong evidence
* Rewrite the taxonomy without explicit instruction
* Move or add a second **Start here** marker without instruction
* Edit `README.md` and the matching `website/docs/` page out of sync
* Edit unrelated files
* Touch protected areas unless instructed
* Ask contributors to make trivial fixes the maintainer can safely make

## Commands

There is no application build. The repository runs content checks; CI mirrors these in `.github/workflows/`.

* Entry-count band — every canonical category must hold 15–25 entries:
  `python scripts/check_entry_counts.py`
* Markdown lint (advisory, matches the `lint` workflow):
  `npx --yes -p remark-cli -p remark-preset-lint-recommended -p remark-mdx remark --use remark-mdx --use remark-preset-lint-recommended --quiet --frail README.md "website/docs/**/*.md" "website/docs/**/*.mdx"`
* Link check (matches the `link-check` workflow; requires the `lychee` binary):
  `lychee --no-progress --max-retries 2 README.md "website/docs/**/*.md" "website/docs/**/*.mdx"`
  Allowed exceptions live in `.lycheeignore`.
* Documentation site (Docusaurus, Node ≥ 18, run from `website/`):
  `cd website && npm install && npm start` to preview, `npm run build` to build.

## Repository Facts

* `AGENTS.md` contains the full tool-agnostic operating protocol.
* `CONTRIBUTING.md` contains contributor-facing rules, the per-type inclusion gates, formatting conventions, the tag scheme, and the **Start here** update process.
* `website/docs/` is a standalone Docusaurus site. `website/docs/categories/<slug>.mdx` mirrors each README category and must stay in sync with it. `website/docs/workflow.mdx` and `workflow-review.mdx` document the PR and curation processes.
* `.github/ISSUE_TEMPLATE/` contains the public issue forms (new resource, remove resource, category proposal, curation review).
* The `README.md` contains a Get Started block, Contents, the canonical categories, and an Appendices block.
* The main list uses bullet entries with an em dash separator: `- [Name](URL) — Description.` Many entries carry a `<!-- tags: ... -->` comment on the next line. Match the surrounding section exactly.
* Each category has exactly one **Start here** entry, updated deliberately and changed in both `README.md` and the matching `website/docs/categories/<slug>.mdx`.
* New entries usually go to the bottom of the relevant category unless local ordering clearly indicates otherwise.
* New categories require a Category proposal issue with at least three seed entries and are handled separately.
* The list is reviewed monthly via a scheduled curation-review issue; `REVIEW.md` (local only) holds the full process.
* Protected areas include badges, the Get Started block, Contents, banners, images under `assets/`, **Start here** markers, contributor blocks, generated sections, and licence text.

## Always-Loaded Context

Keep this file short. It is an orientation layer, not a manual.

Use this routing:

* Need general agent rules → read `AGENTS.md`
* Need contribution rules → read `CONTRIBUTING.md`
* Need PR or curation process → read `website/docs/workflow.mdx` and `workflow-review.mdx`
* Need style examples → inspect the target section in `README.md`
* Need contributor expectations → inspect `.github/ISSUE_TEMPLATE/`
* Need maintainer precedent → inspect recent issues and merged PRs where available

Do not duplicate long sections from those files here.

## First-Pass Workflow

For any PR, issue, or README task:

1. Read the user request.
2. Read the relevant issue, PR, diff, or target README section.
3. Check the repository scope.
4. Check `CONTRIBUTING.md` if the task concerns a submission.
5. Check neighbouring entries for style and placement.
6. Search for duplicates.
7. Verify the link where tools allow.
8. Inspect the resource enough to understand what it is.
9. Choose the smallest useful action.
10. Produce a concise decision, edit, or maintainer comment.

## Entry Checklist

Before recommending acceptance or adding an entry, confirm:

* In scope
* Meets the resource-type gate in `CONTRIBUTING.md`
* Technically useful
* Credible source
* Canonical URL
* Durable link
* No duplicate
* Correct section
* Local format matched, including the tag comment where used
* Neutral description
* No hype
* No unsupported claims
* No avoidable tracking parameters
* No unnecessary new section
* Category stays within the 15–25 entry band

## Source Preference

Prefer:

* Official repositories
* Official documentation
* Papers
* Technical reports
* Benchmarks
* Datasets
* Durable project pages
* Maintained tools and libraries
* High-quality reference material

Treat cautiously:

* Launch posts
* Vendor pages
* Thin wrappers
* Newsletter posts
* Social posts
* Unmaintained repositories
* Link farms
* Pages dominated by sales language
* Time-sensitive comparisons

## Description Rules

Default pattern:

`- [Name](URL) — Clear factual description.`

Preserve any `<!-- tags: ... -->` comment line in sections that use one.

Descriptions should:

* Start with a capital letter
* End with a full stop
* Be short and specific
* Avoid title case
* Avoid starting with "A" or "An"
* Avoid marketing taglines
* Explain what the resource is, not why it is exciting

Remove or neutralise:

* "best"
* "latest"
* "most advanced"
* "powerful"
* "revolutionary"
* "cutting-edge"
* "game-changing"
* "industry-leading"
* "fastest"
* Unsupported performance, adoption, maturity, or pricing claims

## Section Placement

| Situation                             | Action                                                |
| ------------------------------------- | ----------------------------------------------------- |
| Exact fit in an existing section      | Place there.                                          |
| Fits two sections                     | Choose the more specific or more discoverable one.    |
| Similar to neighbouring entries       | Place near those entries if local ordering allows.    |
| New theme with one entry              | Park, or place in the nearest broader section.        |
| New theme with several strong entries | Suggest a new section; do not create it unless asked. |
| Unclear placement                     | Explain the options briefly and recommend one.        |

## PR Triage

| Decision        | Use when                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------- |
| Accept as-is    | Scope, gate, link, placement, format, and description are all sound.                       |
| Maintainer edit | Strong resource needing only minor wording, link, placement, or formatting fixes.         |
| Request changes | Relevance, evidence, link quality, or placement is materially unclear.                    |
| Close           | Out of scope, duplicate, promotional, broken with no replacement, or low technical value. |
| Park            | Promising but immature, below the gate, needs a taxonomy decision, or needs judgement.    |

## Issue Triage

Suggestion issues:

* Strong, in scope, canonical → draft entry and recommend acceptance.
* Strong but wording or placement needs work → recommend maintainer edit.
* Missing evidence → ask for minimal clarification.
* Duplicate → close with a pointer to the existing entry.
* Out of scope → close politely.
* Premature or taxonomy-dependent → park.

Broken-link issues:

* Verify the link.
* Find a canonical replacement first.
* Prefer official sources over mirrors.
* Remove only when no durable replacement exists.
* Leave a concise note explaining the action.

## Small Safe Fix Rule

Protect contributor goodwill.

When a resource is suitable and the issue is minor, make or recommend a maintainer edit rather than asking the contributor to revise.

Small safe fixes include:

* Tightening a description
* Removing hype
* Fixing punctuation
* Correcting placement
* Replacing a non-canonical URL
* Matching the bullet and tag-comment format
* Removing tracking parameters

## Stop and Ask

Stop before:

* Creating a new top-level category
* Reordering large parts of the README
* Editing Contents
* Moving or replacing a **Start here** marker
* Editing visual assets
* Changing contribution rules
* Removing several entries
* Making broad scope decisions
* Editing unrelated files

## Protected Areas

Do not edit unless explicitly instructed:

* Badges
* Contents
* The Get Started block
* **Start here** markers
* Banner and live-docs assets under `assets/`
* Gallery images
* Announcement or roadmap blocks
* Contributor sections
* Generated indexes
* Licence text
* `REVIEW.md` and other local-only files
* `.github/workflows/` automation
* Repository metadata unrelated to the task
* Private, draft, or scratch files

## Maintainer Comment Templates

Accept:

"Thank you — this looks relevant, the link is canonical, and the placement works. I would accept this."

Maintainer edit:

"Thank you — this is a useful resource. I would accept it with a small maintainer edit to tighten the description and keep the wording neutral."

Request changes:

"Thank you for the suggestion. I think this could fit, but I would ask for a little more context on why this is the canonical source and where it belongs."

Duplicate:

"Thank you — I would close this as a duplicate because the resource already appears under [section]."

Out of scope:

"Thank you for sharing this. I would close it because it sits outside the current scope of the list."

Park:

"Thank you — this may be worth revisiting, but I would park it for now until the list has a clearer section for this category."

## Output Format

For PR or issue review, respond with:

* **Decision**: accept, maintainer edit, request changes, close, or park
* **Reason**: 1–3 bullets
* **Suggested README entry**, if useful
* **Suggested maintainer comment**
* **Files changed**, if any
* **Remaining uncertainty**, if any

## Editing Rule

Do not modify `README.md`, `CONTRIBUTING.md`, `.github` templates, the `website/docs/` pages, or other files unless explicitly asked.
