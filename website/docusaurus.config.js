/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Awesome Physical AI',
  tagline:
    'A curated Physical AI roadmap: robotics resources, embodied AI, world models, robotics simulation, sim-to-real, Physical AI benchmarks, foundation models for robotics, and production-grade Physical AI systems.',
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
      metadata: [
        {
          name: 'description',
          content:
            'Awesome Physical AI — a curated, engineering-oriented map of Physical AI resources: robotics resources, robot learning, embodied AI, embodied agents, embodied intelligence, robotics simulation, sim-to-real, world models, vision-language-action (VLA) models, Physical AI benchmarks, robotics datasets, robotics benchmarks, foundation models for robotics, generalist robot policies, and production-grade, safe embodied AI systems.',
        },
        {
          name: 'keywords',
          content:
            'awesome physical AI, Physical AI resources, Physical AI roadmap, robotics resources, robot learning, robotics foundation models, embodied AI, embodied agents, embodied intelligence, robotics simulation, sim-to-real, simulation environments, world models, vision-language-action models, VLA models, Physical AI benchmarks, robotics datasets, robotics benchmarks, foundation models for robotics, generalist robot policies, production-grade Physical AI systems, safe embodied AI systems',
        },
        { property: 'og:title', content: 'Awesome Physical AI' },
        {
          property: 'og:description',
          content:
            'A curated Physical AI roadmap covering robotics resources, embodied AI, world models, robotics simulation, sim-to-real, VLA models, Physical AI benchmarks, foundation models for robotics, generalist robot policies, and production-grade, safe embodied AI systems.',
        },
        { property: 'og:type', content: 'website' },
        { name: 'twitter:card', content: 'summary_large_image' },
        { name: 'twitter:title', content: 'Awesome Physical AI' },
        {
          name: 'twitter:description',
          content:
            'Curated Physical AI resources: robotics, embodied AI, simulation, sim-to-real, world models, VLA models, benchmarks, and production-grade Physical AI systems.',
        },
      ],
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