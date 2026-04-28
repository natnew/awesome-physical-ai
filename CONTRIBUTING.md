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

### Categories

- Add entries to the most specific applicable category.
- Entries should be alphabetically ordered within their category.
- If a resource fits multiple categories, add it to the primary one only.
- New categories require at least 3 quality entries.

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

## Code of Conduct

- Be respectful and constructive.
- Focus on quality over quantity.
- Disclose conflicts of interest.
- Help maintain the list by reviewing others' contributions.

## Questions?

Open an issue for discussion before making significant changes.

---

Thank you for helping make this a valuable resource for the robotics AI community! 🤖