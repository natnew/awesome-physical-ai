# Contributing to Awesome Physical AI

Thank you for your interest in contributing! This list aims to be a high-quality, curated collection of resources for Physical AI and Embodied AI—helping people learn, build, deploy, and stay current in the field.

Resource suggestions, link repairs, factual corrections, and documentation improvements are welcome. Keep each contribution focused and follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately using the [Security Policy](SECURITY.md).

## Guidelines

### What Makes Something "Awesome"?

Only submit resources that you have personally used or can genuinely recommend. Ask yourself:

- Is this software actively maintained, or is this paper, dataset, or reference still useful and accessible?
- Does it provide unique value not covered by existing entries?
- Would a robotics researcher or practitioner benefit from knowing about this?
- Is it accessible (good documentation, open-source, or has a public demo)?

Inspect the resource before submitting it, and support your recommendation with evidence. Stay within Physical AI, embodied AI, robotics, and the adjacent technical topics already covered by the catalog. Pure marketing, thin wrappers, link farms, speculative entries, and unrelated general AI material do not qualify.

### Quality Standards

Resources should meet these criteria:

- **Software/Tools**: Actively maintained (commits within last 12 months), has documentation, and ideally >100 GitHub stars.
- **Papers**: Published at peer-reviewed venues (ICRA, RSS, CoRL, NeurIPS, ICML, ICLR) or influential arXiv preprints with >50 citations.
- **Hardware**: Commercially available or has open-source designs with reproduction guides.
- **Courses**: From recognized institutions or with substantial community adoption.
- **Companies**: Established organizations with public products, demos, or research output.

The star threshold is a preference, not a requirement. Stable papers, datasets, and reference material do not need recent commits, and research contributions do not need a code release. For technical reports or preprints whose eligibility is unclear, explain the evidence gap for maintainer review rather than assuming an exception.

### Formatting

Follow these formatting conventions:

Use an em dash in the 14 canonical README categories:

```markdown
- [Name](URL) — Description ending with a period.
```

Appendices and issue-form examples retain the hyphen separator:

```markdown
- [Name](URL) - Description ending with a period.
```

- Use `- ` (dash with space) for list items.
- Descriptions should be concise (one sentence).
- Descriptions start with uppercase, end with period.
- Use proper capitalization: `ROS 2` not `ros2`, `PyTorch` not `pytorch`.
- Explain what the resource does; avoid hype, unsupported claims, and descriptions beginning with "A" or "An".
- Prefer canonical upstream repositories, official documentation, dataset pages, or research pages. Use HTTPS where available and remove tracking parameters.
- No trailing slashes on URLs.
- No period after URLs.
- Open each added or changed link and confirm that redirects lead to the intended resource. Report any verification you could not complete.

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

Use only tags supported by evidence; popularity alone does not establish production readiness. On the matching website page, preserve its existing tag presentation (such as `TagList`) instead of adding HTML comments to MDX.

### Categories

- Add entries to the most specific applicable category.
- Entries should be alphabetically ordered within their category.
- Some existing sections are not fully sorted. Use an appropriate local insertion point and note ambiguity in the PR; do not reorder unrelated entries.
- If a resource fits multiple categories, add it to the primary one only.
- New categories require a separate Category proposal with at least 3 vetted seed entries and maintainer agreement before implementation.

The 14 categories in [website/sidebars.js](website/sidebars.js) are the authoritative taxonomy. README appendices complement them. Update catalog entries in both [README.md](README.md) and the matching page under [website/docs/categories](website/docs/categories); synchronization is manual. Keep names, URLs, descriptions, and applicable tags consistent while preserving each file's presentation.

When changing the catalog, update the README status-line date and total. The total counts canonical categories only, excluding appendices. Check any affected website summaries too.

### Updating the "Start here" entry for a category

Each category is intended to have one **Start here** recommendation for newcomers. The website currently has these callouts, but README category markers are absent. Do not add markers as part of an unrelated resource contribution. Propose a replacement deliberately:

1. Open a pull request titled `Update Start here for <Category>`.
2. In the PR description, name the **incumbent** Start-here entry and the **proposed** replacement, and explain why the replacement is a better entry point. Strong reasons include: incumbent is no longer maintained, the new entry is materially clearer for beginners, or the field has moved (e.g., a new canonical tutorial or survey).
3. Once the maintainer agrees on the replacement and README presentation, update the matching website callout and README together so both views agree.
4. Do **not** add a second Start-here entry; there is exactly one per category.
5. If you would like to suggest a change without opening a PR, file a [**New resource**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue and check the *"This is a Start-here proposal"* box; a maintainer will pick it up during the next curation review.

## How to Contribute

### Adding a Resource

1. Search the README, website docs, and open or closed issues and PRs for duplicates, including alternate names and URLs. Explain any distinct value of related paper, code, or dataset artifacts.
2. Fork the repository and create a branch for your change.
3. Add the resource to the most specific category in both the README and matching website page, following the formatting and count rules above.
4. Run the applicable checks below and inspect your diff for unrelated changes.
5. Submit a pull request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md), with:
   - Clear title, such as `Add Resource Name`.
   - Brief explanation of its unique technical value and eligibility evidence, such as venue, citation count, documentation, or maintenance date.
   - Your relationship to the project (if any).
   - Checks performed and any unresolved verification gaps.

You can also suggest a resource through the **New resource** issue form without preparing a PR. Discuss broad restructuring, category changes, or changes to protected README sections before implementing them.

### Reporting Issues

- Broken links.
- Outdated information.
- Miscategorized resources.
- Duplicate entries.

Use the appropriate issue form and include the affected entry, supporting evidence, and a proposed correction if available. Security reports belong in the [private reporting process](SECURITY.md), and conduct reports belong in the [Code of Conduct reporting process](CODE_OF_CONDUCT.md#reporting).

### How to propose a removal

Open a [**Remove resource**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue. Include the link to the existing entry, its current category, the reason (dead link / abandoned / low-signal / duplicate / miscategorised), and supporting evidence (last commit date, archive notice, replacement entry, etc.). Removals are accepted when the entry no longer meets the curation standards above.

For broken links, look for a durable canonical replacement first. Do not recommend removing stable research or reference material solely because it has no recent commits.

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
- [ ] Matching README and website entries agree, and the status-line date and total are updated where applicable.
- [ ] I have disclosed any relationship to the resource and included eligibility evidence.
- [ ] I have run relevant checks, reported any gaps, and kept the diff focused.

## Continuous integration

Checks are path-filtered; not every PR triggers every workflow. The [workflow files](.github/workflows) define the current triggers:

- **`site-build`** — runs for website or site-build workflow changes, using Node.js 20 and `npm ci` followed by `npm run build` in `website/`. Build errors and broken internal docs links fail the check.
- **`link-check`** — runs for README, website docs, or link-check configuration changes. It checks catalog and docs links with lychee, subject to `.lycheeignore`.
- **`lint`** — runs for README, website docs, template, or lint workflow changes. Markdown/MDX lint is advisory; issue-form YAML validation is not.

The entry-count script is a local check and is not currently run by CI. Changes only to this guide, `CODE_OF_CONDUCT.md`, or `SECURITY.md` do not trigger these three workflows.

### Reproducing locally

Run only the checks relevant to your change. From the repository root:

```bash
# Canonical category counts (15–25 each) and status-line total
python scripts/check_entry_counts.py

# Catalog and docs links; requires lychee to be installed
lychee --no-progress --max-retries 2 README.md "website/docs/**/*.md" "website/docs/**/*.mdx"

# Whitespace errors
git diff --check
```

For website or MDX changes, use Node.js 20 and preserve the committed lockfile:

```bash
cd website
npm ci
npm run build
npm run lint:docs
```

The count script does not compare README and MDX content; check that consistency manually. Report existing count or lint problems rather than adding weak resources or making unrelated fixes. For policy-only edits, verify local links and referenced commands, then inspect the diff; a site build is unnecessary.

### Suppressing a known-flaky external link

If a link is verified-good in a browser but consistently fails in CI (e.g. a host that returns HTTP 999 to HEAD requests, or aggressively rate-limits), add a regex to `.lycheeignore` at the repo root **with a one-line comment explaining why**. Narrow patterns are preferred over broad ones. Entries are reviewed during the periodic curation pass.

## Periodic review

The list is reviewed **monthly** to prune stale entries, rebalance category depths, and refresh "Start here" highlights. The [curation-review workflow](.github/workflows/curation-review.yml) runs on the first of each month at 09:00 UTC and can also be triggered manually. It keeps one review issue open, noting later cycles on an existing review rather than creating duplicates. Follow the [Review process](website/docs/workflow-review.mdx) and submit one review PR. A local, gitignored `REVIEW.md` may contain maintainer notes; contributors do not need that file.

To kick off an ad-hoc review between scheduled runs, open a [**Curation review**](https://github.com/natnew/awesome-physical-ai/issues/new/choose) issue.

## Awesome list inclusion

This repository carries the [Awesome](https://awesome.re) badge at the top of `README.md`. Submission to the [`sindresorhus/awesome`](https://github.com/sindresorhus/awesome) meta-list is intentionally deferred: the project meets the structural criteria, but submission is a one-shot external event best done after sustained traction signals (incoming PRs, external links, established review history). Contributors do not need to do anything for this — the criteria authority remains the project's own [curation standards](https://natnew.github.io/awesome-physical-ai/docs/curation-standards).

## Announcements

Draft copy for community announcements (Hacker News, Reddit, Discord, Slack) is kept in a gitignored `ANNOUNCEMENTS.md` at the repository root. **Do not commit announcement copy to the repo.** The drafts file is intentionally local so marketing language never enters the public history; publication is a manual, deliberate act outside the spec lifecycle.

## Code of Conduct

All participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md), which describes community expectations, reporting, and enforcement.

## Questions?

Open an issue for discussion before making significant changes.

---

Thank you for helping make this a valuable resource for the robotics AI community! 🤖
