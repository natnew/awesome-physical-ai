# AGENTS.md

Operating protocol for AI coding agents working in this repository.

All agents should start here, then read `README.md` and `CONTRIBUTING.md` before reviewing or editing. Follow repository-local guidance over generic awesome-list assumptions.

## Repository North Star

This is a public, maintained awesome list for Physical AI and embodied AI. The `README.md` is the product: a durable, high-signal, navigable map of robot learning, foundation models for robotics, vision-language-action models, world models, simulation and sim-to-real, datasets, benchmarks, and patterns for safe, production-grade systems.

The list is curated, not accumulated. Each entry should help a reader understand the Physical AI landscape, find a credible resource, or compare related tools. Selectivity, durability, clear placement, and neutral description quality matter more than volume.

## Agent Role

Agents may help with:

* README maintenance when explicitly asked
* New entry review
* Pull request review
* Issue triage
* Broken-link checks
* Duplicate detection
* Section placement
* Description tightening
* Maintainer comment drafts
* Small, safe cleanup edits when explicitly requested

Agents must not:

* Add speculative or low-signal entries
* Inflate claims or preserve promotional wording
* Reorganise the list without explicit instruction
* Run broad formatting sweeps
* Edit unrelated files
* Rewrite the maintainer's style unnecessarily
* Turn one contribution into a broad structural change
* Move or add a second **Start here** marker without explicit instruction
* Edit `README.md` and the matching `website/docs/` page out of sync
* Touch protected areas unless explicitly instructed

## Read Order

Before reviewing or editing, read in this order:

1. `README.md` — scope, taxonomy, tags, **Start here** markers, formatting, protected areas, and existing examples
2. `CONTRIBUTING.md` — inclusion criteria, formatting conventions, tags, and the **Start here** update process
3. `.github/ISSUE_TEMPLATE/` — contributor expectations (new resource, remove resource, category proposal, curation review)
4. `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist
5. `website/docs/categories/<slug>.mdx` — the docs-site mirror that must stay in sync with the README
6. Recent issues and merged PRs, where available, for maintainer precedent

Do not assume the generic awesome-list pattern overrides this repository's existing structure.

## Repository Facts

* The `README.md` contains introductory sections, a Get Started block, Contents, the canonical category list, and an Appendices block.
* The taxonomy is split into **Canonical categories** (Simulators, Datasets, Benchmarks, Evaluation Methodology, Robotics Foundation Models, World Models, Manipulation, Locomotion, Sim-to-Real, Safety & Robustness, Governance & Policy, Production Patterns / Reference Architectures, Courses, Companies) and **Appendices** (Books, Tutorials & Guides, Key Papers, Survey Papers, Hardware Platforms, Conferences, Community, Newsletters & Blogs, People to Follow, Related Awesome Lists, Contributing).
* Main list sections use bullet lists. Entries use an em dash separator: `- [Name](URL) — Description ending with a period.` Match the local section style exactly.
* Many entries carry a tag comment on the following line: `<!-- tags: ... -->`. Preserve and match this convention where the surrounding section uses it.
* Each category has exactly one **Start here** entry. It is updated deliberately, not casually, and must be changed in both `README.md` and `website/docs/categories/<slug>.mdx` together. Never add a second one.
* Some sections include explanatory text before entries. Preserve it.
* `CONTRIBUTING.md` sets inclusion gates by resource type: software/tools maintained within the last 12 months with documentation and ideally over 100 stars; papers at recognised venues or influential preprints with over 50 citations; hardware that is commercially available or has open reproduction guides; courses from recognised institutions; companies with public products, demos, or research output.
* `CONTRIBUTING.md` asks for one PR per suggestion.
* New entries should be added to the bottom of the relevant category unless local ordering clearly indicates otherwise.
* New categories require a Category proposal issue with at least three vetted seed entries in the canonical format, handled separately from single-entry PRs.
* For package-style submissions, prefer the GitHub repository over a package registry or marketing page.
* Descriptions should be short, simple, descriptive, and non-promotional.
* The list is reviewed monthly; a scheduled GitHub Action opens a curation-review issue. The full review process lives in `REVIEW.md` (local only) and on the docs site.

## Scope Rules

Belongs:

* Official repositories
* Official documentation
* Papers and technical reports
* Datasets and benchmarks
* Evaluation and robustness tools
* Simulators and sim-to-real frameworks
* Robotics foundation models, VLA models, and generalist policies
* World models for embodied agents
* Manipulation and locomotion resources
* AI-native robotics development tools
* Durable tutorials, courses, books, and technical explainers
* Safety, alignment, governance, observability, testing, and security resources relevant to embodied AI
* High-signal platforms, libraries, frameworks, standards, hardware platforms, and reference resources

Does not belong:

* Thin wrapper pages with little original technical value
* Pure marketing pages
* Broken or inaccessible links
* Duplicate or near-duplicate resources
* Speculative entries
* Low-signal directories or link farms
* Unsupported ranking, performance, adoption, or novelty claims
* Pricing claims unless the surrounding section already tracks pricing
* Time-sensitive claims such as "latest", "best", "leading", "fastest", or "most advanced"
* Content outside Physical AI, embodied AI, robotics, or adjacent technical areas already represented in the README

## Quality Bar

An entry qualifies when all are true:

* It is clearly relevant to Physical AI or an adjacent technical area already represented in the list.
* It meets the resource-type gate in `CONTRIBUTING.md` (maintenance, stars, citations, or availability, as applicable).
* The source is credible and useful to a technical reader.
* The link is canonical, durable, and reachable.
* The resource adds something distinct from existing entries.
* The entry fits an existing section without forcing a taxonomy change.
* The description is neutral, concise, specific, and non-promotional.
* The formatting matches the surrounding section, including the tag comment where used.
* No duplicate or stronger existing equivalent is already present.

## README Formatting Rules

Infer format from the surrounding section before editing.

* Preserve the existing heading structure.
* Preserve all anchors and Contents links.
* Preserve badges, banners, the Get Started block, Contents, **Start here** markers, contributor sections, and other protected areas.
* Match the section's existing format: bullet entry, em dash separator, and tag comment where present.
* Use HTTPS links where available.
* Use canonical names with correct capitalisation, for example `ROS 2`, `PyTorch`, `MuJoCo`.
* Keep descriptions short.
* Start descriptions with a capital letter.
* End descriptions with a full stop.
* Do not use title case for descriptions.
* Do not start descriptions with "A" or "An".
* No trailing slashes on URLs.
* Do not perform broad formatting changes unless explicitly asked.

## Link Quality Rules

Verify that:

* The link resolves.
* The link points to the canonical source.
* Repository links point to the main project, not an arbitrary fork.
* Paper links prefer official publisher pages, arXiv, DOI, or project pages.
* Documentation links prefer official docs.
* Dataset links prefer official dataset pages or maintained repositories.
* URLs do not include avoidable tracking parameters.
* Login-gated resources are avoided unless the list already accepts that kind of resource.
* Shortened links are avoided.

## Description Style

Descriptions should be:

* Neutral
* Factual
* Specific
* Short
* Present tense where possible
* Free of hype
* Free of unsupported claims
* Useful to a reader scanning the list quickly

Prefer:

* "GPU-accelerated robotics simulator with photorealistic rendering."
* "Cross-embodiment dataset of 1M+ trajectories for training robot policies."
* "Vision-language-action model connecting perception, language, and robot control."
* "Benchmark suite for language-conditioned manipulation."

Avoid:

* "Powerful"
* "Revolutionary"
* "Cutting-edge"
* "Best"
* "Latest"
* "Industry-leading"
* "Game-changing"
* "Fastest"
* Unsupported claims about performance, adoption, or maturity

## Section Placement Rules

1. Identify the closest existing section.
2. Compare the candidate with neighbouring entries.
3. Prefer the narrowest accurate section.
4. If two sections fit, choose the one where readers would most naturally look first.
5. Avoid creating a new section for a single item.
6. Do not move many existing entries unless explicitly asked.
7. If placement is uncertain, state the trade-off and recommend one option.
8. New category proposals should be separate from single-entry PRs and require three seed entries.

## Duplicate Checking Rules

Before adding or approving, check for:

* Same URL
* Same project under a different URL
* Same paper title
* Same organisation and product name
* Renamed repositories
* Existing entry in a nearby section
* Existing issue or PR suggesting the same resource
* A stronger canonical source already listed

If a duplicate exists, recommend closing, editing, or redirecting rather than adding another entry.

## Decision Matrix

| Decision           | Use when                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| Accept as-is       | In scope, meets the gate, canonical link, correct placement, matching format, neutral description, no duplicate. |
| Edit as maintainer | Strong resource needing small fixes: wording, punctuation, canonical URL, placement, tags, or local formatting. |
| Request changes    | Resource may fit but evidence, link quality, relevance, gate, or placement is materially unclear.         |
| Close              | Out of scope, duplicate, promotional, broken with no replacement, or no durable technical substance.      |
| Park               | Promising but immature, below the gate, not yet supported by the taxonomy, or requires maintainer judgement. |

## Issue-to-Entry Workflow

For suggestion issues:

1. Check scope.
2. Check the resource-type gate in `CONTRIBUTING.md`.
3. Check source quality.
4. Check link quality.
5. Check duplicates.
6. Identify the best section.
7. Draft a neutral entry, with tag comment where the section uses one, only if the resource qualifies.
8. Recommend accept, maintainer edit, request changes, close, or park.
9. Keep the maintainer comment concise.

For broken-link issues:

1. Verify the reported link.
2. Search for a canonical replacement.
3. Prefer official replacements over mirrors.
4. Preserve the entry if a durable replacement exists.
5. Recommend removal only when no credible replacement exists.
6. State the action clearly.

## Pull Request Review Workflow

1. Read the PR title, description, and diff.
2. Confirm it changes only relevant files.
3. Confirm README and any matching `website/docs/` page stay in sync.
4. Check each added or changed link.
5. Check scope, the resource-type gate, and source quality.
6. Check duplicates.
7. Check section placement.
8. Check local formatting, including the tag comment.
9. Neutralise description language where needed.
10. Decide: accept, maintainer edit, request changes, close, or park.
11. Draft a concise maintainer comment.

Minimise contributor friction. If the resource is clearly suitable and the issue is minor, recommend a maintainer edit rather than asking the contributor to revise.

## Stop and Ask

Stop and ask the maintainer before:

* Creating a new top-level category
* Reordering large parts of the README
* Changing the Contents structure
* Moving or replacing a **Start here** marker
* Editing the Get Started block, badges, banners, or visual assets
* Changing contribution rules
* Removing multiple entries
* Making judgement-heavy scope changes
* Editing files unrelated to the stated task

## Protected Areas

Do not edit unless explicitly instructed:

* Badges
* Banner and live-docs assets under `assets/`
* The Get Started block
* Contents
* **Start here** markers
* Announcement or roadmap blocks
* Contributor lists
* Generated indexes
* Licence text
* Repository metadata unrelated to the task
* `REVIEW.md` and other local-only files
* `.github/workflows/` automation
* Private notes, draft files, scratch files, and local-only files

## Maintainer Comment Style

Comments should be warm, concise, respectful, and decision-oriented.

Prefer:

* "Thank you for the suggestion. This is relevant, the link is canonical, and I would place it under X with a shorter neutral description."
* "Thank you — useful resource. I would accept this with a small maintainer edit to remove the ranking claim."
* "Thank you for raising this. I would close it as a duplicate because the resource already appears under X."
* "Thank you — I would park this for now; it is below the inclusion gate, so let us revisit once it is more established."

Avoid:

* Long explanations
* Harsh rejection wording
* Defensive language
* Asking contributors for trivial edits the maintainer can safely make

## Final Response Pattern

When finishing a task, summarise:

* What was reviewed
* Decision or recommended decision
* What changed, if anything
* Any risks or uncertainties
* Suggested maintainer comment, if relevant
* Follow-up needed, if any

Do not modify `README.md` or other files unless explicitly asked.
