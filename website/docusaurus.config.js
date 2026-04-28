/** @type {import('@docusaurus/types').Config} */
const baseUrl = process.env.DOCUSAURUS_BASE_URL || '/awesome-physical-ai/';
const siteUrl = process.env.DOCUSAURUS_SITE_URL || 'https://natnew.github.io';
const customDomain = process.env.DOCUSAURUS_CUSTOM_DOMAIN || '';
const canonicalSiteUrl = customDomain ? `https://${customDomain}` : siteUrl;

const joinUrl = (origin, pathPrefix = '', relativePath = '') => {
  const normalizedOrigin = origin.replace(/\/+$/, '');
  const normalizedPrefix = pathPrefix.replace(/^\/+|\/+$/g, '');
  const normalizedPath = relativePath.replace(/^\/+/, '');
  const segments = [normalizedOrigin];

  if (normalizedPrefix) {
    segments.push(normalizedPrefix);
  }

  if (normalizedPath) {
    segments.push(normalizedPath);
  }

  return segments.join('/');
};

const siteRootUrl = `${joinUrl(canonicalSiteUrl, baseUrl)}/`;
const ogImageUrl = joinUrl(canonicalSiteUrl, baseUrl, 'img/og-card.svg');

const config = {
  title: 'Awesome Physical AI',
  tagline:
    'A curated Physical AI roadmap: robotics resources, embodied AI, world models, robotics simulation, sim-to-real, Physical AI benchmarks, foundation models for robotics, and production-grade Physical AI systems.',
  favicon: 'img/favicon.svg',
  url: canonicalSiteUrl,
  baseUrl,
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
      image: ogImageUrl,
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
        { property: 'og:url', content: siteRootUrl },
        { property: 'og:image', content: ogImageUrl },
        { property: 'og:type', content: 'website' },
        { name: 'twitter:card', content: 'summary_large_image' },
        { name: 'twitter:title', content: 'Awesome Physical AI' },
        {
          name: 'twitter:description',
          content:
            'Curated Physical AI resources: robotics, embodied AI, simulation, sim-to-real, world models, VLA models, benchmarks, and production-grade Physical AI systems.',
        },
        { name: 'twitter:image', content: ogImageUrl },
        { rel: 'canonical', href: siteRootUrl },
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
            type: 'docSidebar',
            sidebarId: 'categoriesSidebar',
            position: 'left',
            label: 'Categories',
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
  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        searchResultLimits: 10,
        searchBarShortcut: true,
        searchBarShortcutHint: true,
      }),
    ],
  ],
};

module.exports = config;