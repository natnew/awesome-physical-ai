# Awesome Physical AI Docs Site

This Docusaurus project provides a standalone documentation site for the repository.

## Local development

```bash
cd website
npm install
npm start
```

## Production build

```bash
cd website
npm run build
```

## Publishing

GitHub Pages deployment is defined in `.github/workflows/deploy-docs.yml`.

### Deployment variables

Configure the following repository variables to control the published URL:

- `DOCUSAURUS_CUSTOM_DOMAIN` (optional): if set, deploy writes `CNAME` and uses `https://<domain>` as site URL.
- `DOCUSAURUS_SITE_URL` (optional fallback): used when no custom domain is configured.
- `DOCUSAURUS_BASE_URL` (optional): defaults to `/` when a custom domain is set, otherwise `/awesome-physical-ai/`.

### DNS checklist (custom domain)

1. Add DNS records at your provider for the chosen host (apex/`www`) pointing to GitHub Pages.
2. Set `DOCUSAURUS_CUSTOM_DOMAIN` in repository variables.
3. Trigger `Deploy Docs` workflow and verify that `website/build/CNAME` is present in the published artifact.
4. In repository settings, confirm Pages serves the custom domain and HTTPS is enabled.

### Rollback

1. Remove `DOCUSAURUS_CUSTOM_DOMAIN` variable.
2. Re-run `Deploy Docs` workflow.
3. Verify site resolves at GitHub Pages URL (`https://natnew.github.io/awesome-physical-ai/`).

The source catalog for project content remains `README.md` in the repository root.

## npm overrides

`package.json` declares `overrides` to clear transitive vulnerabilities surfaced by `npm audit` against the Docusaurus 3.x dependency tree. Each pin should be revisited when its parent dependency releases a clean version:

- `serialize-javascript ^7.0.5` — XSS fix (CVE in older 6.x reachable via webpack/terser).
- `follow-redirects ^1.16.0` — closes credential/auth-header leak on cross-origin redirect.
- `postcss ^8.5.12` — line-return parsing CVE.
- `uuid ^14.0.0` — pulls forward to a maintained major; older `uuid@3` and `uuid@8` lines reach Docusaurus through several transitive paths.

When upgrading Docusaurus, run `npm audit` and remove any override whose root cause is no longer present.