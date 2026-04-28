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