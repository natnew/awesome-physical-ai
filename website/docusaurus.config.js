/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Awesome Physical AI',
  tagline:
    'Repository-grounded navigation for the curated Physical AI / Embodied AI resource list.',
  favicon: 'img/favicon.svg',
  url: 'https://natnew.github.io',
  baseUrl: process.env.DOCUSAURUS_BASE_URL || '/awesome-physical-ai/',
  organizationName: 'natnew',
  projectName: 'awesome-physical-ai',
  onBrokenLinks: 'throw',
  trailingSlash: false,
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/natnew/awesome-physical-ai/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'Awesome Physical AI',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'guideSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            href: 'https://github.com/natnew/awesome-physical-ai/blob/main/README.md',
            label: 'README',
            position: 'right',
          },
          {
            href: 'https://github.com/natnew/awesome-physical-ai/blob/main/CONTRIBUTING.md',
            label: 'Contribute',
            position: 'right',
          },
          {
            href: 'https://github.com/natnew/awesome-physical-ai',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Catalog',
            items: [
              {
                label: 'Overview',
                to: '/docs/overview',
              },
              {
                label: 'Quick Start',
                to: '/docs/quick-start',
              },
              {
                label: 'Workflow',
                to: '/docs/workflow',
              },
            ],
          },
          {
            title: 'Repository',
            items: [
              {
                label: 'README',
                href: 'https://github.com/natnew/awesome-physical-ai/blob/main/README.md',
              },
              {
                label: 'Contributing',
                href: 'https://github.com/natnew/awesome-physical-ai/blob/main/CONTRIBUTING.md',
              },
              {
                label: 'License',
                href: 'https://github.com/natnew/awesome-physical-ai/blob/main/LICENSE',
              },
            ],
          },
          {
            title: 'Operations',
            items: [
              {
                label: 'Site README',
                href: 'https://github.com/natnew/awesome-physical-ai/blob/main/website/README.md',
              },
              {
                label: 'Issue Triage Workflow',
                href: 'https://github.com/natnew/awesome-physical-ai/blob/main/.github/workflows/issue-triage-agent.lock.yml',
              },
              {
                label: 'Docs Deploy Workflow',
                href: 'https://github.com/natnew/awesome-physical-ai/blob/main/.github/workflows/deploy-docs.yml',
              },
            ],
          },
        ],
        copyright: 'MIT License. Source catalog remains README-first.',
      },
      docs: {
        sidebar: {
          hideable: false,
          autoCollapseCategories: false,
        },
      },
    }),
};

module.exports = config;