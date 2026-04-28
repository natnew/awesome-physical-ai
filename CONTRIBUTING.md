# Contributing to Awesome Physical AI

Thank you for your interest in contributing! This list aims to be a high-quality, curated collection of resources for Physical AI and Embodied AI—helping people learn, build, deploy, and stay current in the field.

## Guidelines

### What Makes Something "Awesome"?

Only submit resources that you have personally used or can genuinely recommend. Ask yourself:

- Is this tool/paper/resource actively maintained?
- Does it provide unique value not covered by existing entries?
- Would a robotics researcher or practitioner benefit from knowing about this?
- Is it accessible (good documentation, open-source, or has a public demo)?

### Quality Standards

Resources should meet these criteria:

- **Software/Tools**: Actively maintained (commits within last 12 months), has documentation, and ideally >100 GitHub stars.
- **Papers**: Published at peer-reviewed venues (ICRA, RSS, CoRL, NeurIPS, ICML, ICLR) or influential arXiv preprints with >50 citations.
- **Hardware**: Commercially available or has open-source designs with reproduction guides.
- **Courses**: From recognized institutions or with substantial community adoption.
- **Companies**: Established organizations with public products, demos, or research output.

### Formatting

Follow these formatting conventions:

```markdown
- [Name](URL) - Description ending with a period.
```

- Use `- ` (dash with space) for list items.
- Descriptions should be concise (one sentence).
- Descriptions start with uppercase, end with period.
- Use proper capitalization: `ROS 2` not `ros2`, `PyTorch` not `pytorch`.
- No trailing slashes on URLs.
- No period after URLs.

Do / don't:

- Do: `- [PyTorch](https://pytorch.org) - Tensor library with autograd.`
- Don't: `- [pytorch](https://pytorch.org/) Tensor library with autograd` (lowercase name, trailing slash, missing dash separator and terminal period).

### Tags

Each entry is tagged with 1–3 labels indicating resource type, maturity, and availability model:

**Type & Purpose:**
- `tool` — Software, simulator, framework, or library for robotics development.
- `paper` — Research paper or academic publication.
- `dataset` — Dataset for training or evaluation.
- `benchmark` — Benchmark suite or evaluation harness.
- `simulator` — Physics engine or simulation environment.
- `framework` — Architectural or algorithmic methodology.
- `course` — Educational program or lecture series.

**Maturity & Deployment:**
- `production-ready` — Suitable for production deployment; actively maintained and documented.
- `research-only` — Experimental or research-focused; may require substantial adaptation for production.

**Licensing & Availability:**
- `open-source` — Released under an open-source license.
- `commercial` — Commercial product or closed-source.

**How to tag entries:**

When adding or updating an entry in the README, include tags in an HTML comment on the next line:

```markdown
- [Resource Name](https://url.example) — Description of the resource.
<!-- tags: tool, open-source, production-ready -->
```

If a resource doesn't fit cleanly, omit tags rather than over-tagging. For example:
- A research paper: `paper` only.
- A commercial simulation platform: `simulator, commercial`.
- An open-source RL library: `tool, open-source, research-only`.

### Categories

- Add entries to the most specific applicable category.
- Entries should be alphabetically ordered within their category.
- If a resource fits multiple categories, add it to the primary one only.
- New categories require at least 3 quality entries.

### Updating the "Start here" entry for a category

Each category in `README.md` has a single **Start here** entry — the one resource a newcomer should look at first. Because that designation is opinionated, it is updated deliberately, not casually:

1. Open a pull request titled `Update Start here for <Category>`.
2. In the PR description, name the **incumbent** Start-here entry and the **proposed** replacement, and explain why the replacement is a better entry point. Strong reasons include: incumbent is no longer maintained, the new entry is materially clearer for beginners, or the field has moved (e.g., a new canonical tutorial or survey).
3. Move the `Start here` marker line in `README.md` (and the corresponding callout on the matching page under `website/docs/categories/<slug>.mdx`) so README and the docs site stay in sync.
4. Do **not** add a second Start-here entry; there is exactly one per category.
5. If you would like to suggest a change without opening a PR, file a [**New resource**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue and check the *"This is a Start-here proposal"* box; a maintainer will pick it up during the next curation review.

## How to Contribute

### Adding a Resource

1. Fork the repository.
2. Create a new branch: `git checkout -b add-resource-name`
3. Add your resource in the appropriate category.
4. Ensure formatting follows guidelines.
5. Submit a pull request with:
   - Clear title: "Add [Resource Name]"
   - Brief explanation of why it's awesome.
   - Your relationship to the project (if any).

### Reporting Issues

- Broken links.
- Outdated information.
- Miscategorized resources.
- Duplicate entries.

### How to propose a removal

Open a [**Remove resource**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue. Include the link to the existing entry, its current category, the reason (dead link / abandoned / low-signal / duplicate / miscategorised), and supporting evidence (last commit date, archive notice, replacement entry, etc.). Removals are accepted when the entry no longer meets the curation standards above.

### How to propose a new category

Open a [**Category proposal**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue. Include the proposed name, why it is needed, how it differs from existing categories, and **at least three seed entries** in the canonical entry format. Proposals without three vetted seed entries will be deferred.

### Suggesting Improvements

- Restructuring proposals.
- Documentation improvements.

## Pull Request Checklist

- [ ] I have read and followed the guidelines above.
- [ ] The resource is not a duplicate.
- [ ] The resource meets the quality standards for its category.
- [ ] Formatting follows the conventions in the Formatting section.
- [ ] Entry is placed in the most specific applicable category, in alphabetical order.
- [ ] Links are working and have no trailing slash.
- [ ] I have provided a clear explanation of why this resource is awesome.

## Continuous integration

Every pull request runs two required checks:

- **`site-build`** — runs `npm ci && npm run build` in `website/` against Node.js 20. Fails on any Docusaurus build error or broken internal docs link.
- **`link-check`** — runs [lychee](https://github.com/lycheeverse/lychee) against `README.md` and `website/docs/**/*.{md,mdx}`. Fails on any unresolved external link.

A third advisory check (`lint`) runs `remark-lint` on Markdown/MDX and validates the YAML structure of issue forms. It does not block merge during this phase.

### Reproducing locally

```bash
# Site build
cd website
npm ci
npm run build

# Link check (install lychee once: https://lychee.cli.rs/installation/)
cd ..
lychee README.md "website/docs/**/*.md" "website/docs/**/*.mdx"
```

### Suppressing a known-flaky external link

If a link is verified-good in a browser but consistently fails in CI (e.g. a host that returns HTTP 999 to HEAD requests, or aggressively rate-limits), add a regex to `.lycheeignore` at the repo root **with a one-line comment explaining why**. Narrow patterns are preferred over broad ones. Entries are reviewed during the periodic curation pass.

## Periodic review

The list is reviewed **monthly** to prune stale entries, rebalance category depths, and refresh "Start here" highlights. A scheduled GitHub Action (`.github/workflows/curation-review.yml`) opens a `Curation review — YYYY-MM` issue on the first of each month at 09:00 UTC; the same workflow can be triggered manually via **workflow_dispatch** from the Actions tab. The full review process — checklist, staleness criteria, how to run a pass — is documented in `REVIEW.md` at the repository root (local only) and mirrored at the [Review process](https://natnew.github.io/awesome-physical-ai/docs/workflow-review) docs page.

To kick off an ad-hoc review between scheduled runs, open a [**Curation review**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue.

## Awesome list inclusion

This repository carries the [Awesome](https://awesome.re) badge at the top of `README.md`. Submission to the [`sindresorhus/awesome`](https://github.com/sindresorhus/awesome) meta-list is intentionally deferred: the project meets the structural criteria, but submission is a one-shot external event best done after sustained traction signals (incoming PRs, external links, established review history). Contributors do not need to do anything for this — the criteria authority remains the project's own [curation standards](https://natnew.github.io/awesome-physical-ai/docs/curation-standards).

## Announcements

Draft copy for community announcements (Hacker News, Reddit, Discord, Slack) is kept in a gitignored `ANNOUNCEMENTS.md` at the repository root. **Do not commit announcement copy to the repo.** The drafts file is intentionally local so marketing language never enters the public history; publication is a manual, deliberate act outside the spec lifecycle.

## Code of Conduct

- Be respectful and constructive.
- Focus on quality over quantity.
- Disclose conflicts of interest.
- Help maintain the list by reviewing others' contributions.

## Questions?

Open an issue for discussion before making significant changes.

---

Thank you for helping make this a valuable resource for the robotics AI community! 🤖