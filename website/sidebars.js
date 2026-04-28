module.exports = {
  guideSidebar: [
    {
      type: 'category',
      label: 'Guide',
      items: [
        'overview',
        'quick-start',
        'discovery',
        'workflow',
        'workflow-review',
        'architecture',
        'curation-standards',
        'scope-and-limits',
      ],
    },
  ],
  categoriesSidebar: [
    {
      type: 'category',
      label: 'Categories',
      link: { type: 'doc', id: 'categories/index' },
      items: [
        'categories/simulators',
        'categories/datasets',
        'categories/benchmarks',
        'categories/evaluation-methodology',
        'categories/robotics-foundation-models',
        'categories/world-models',
        'categories/manipulation',
        'categories/locomotion',
        'categories/sim-to-real',
        'categories/safety-and-robustness',
        'categories/governance-and-policy',
        'categories/production-patterns',
        'categories/courses',
        'categories/companies',
      ],
    },
  ],
};