import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

import styles from './index.module.css';

const coverage = [
  {
    title: 'Learn',
    description:
      'Courses, books, tutorials, key papers, and survey papers for building a working mental model of robot learning and embodied AI.',
    link: '/docs/quick-start#learn-first',
  },
  {
    title: 'Build',
    description:
      'Foundation models, world models, simulation platforms, learning frameworks, robot software stacks, perception, and control tooling.',
    link: '/docs/quick-start#build-systems',
  },
  {
    title: 'Deploy',
    description:
      'Hardware platforms, datasets and benchmarks, and safety or evaluation references for moving closer to real-world systems.',
    link: '/docs/quick-start#prepare-for-real-world-work',
  },
  {
    title: 'Stay Current',
    description:
      'Conferences, communities, labs, company sources, newsletters, and people to follow.',
    link: '/docs/quick-start#track-the-field',
  },
  {
    title: 'Projects',
    description:
      'Beginner, intermediate, advanced, and optional hardware tracks pulled directly from the README.',
    link: '/docs/quick-start#use-the-project-ladder',
  },
  {
    title: 'Contribution Bar',
    description:
      'Contribution policy, formatting rules, quality thresholds, and pull request checklist summarized from CONTRIBUTING.md.',
    link: '/docs/curation-standards',
  },
];

const proofSignals = [
  {
    label: 'Source of truth',
    value: 'README-first catalog',
  },
  {
    label: 'Content lanes',
    value: '6 major areas',
  },
  {
    label: 'README structure',
    value: '24 subsection headings',
  },
  {
    label: 'Repository signals',
    value: 'MIT + contribution guide + issue triage workflow',
  },
];

const fastPaths = [
  {
    title: 'New to the field',
    description:
      'Start in Courses, Books, Tutorials & Guides, then move into Key Papers and Survey Papers.',
    link: '/docs/quick-start#learn-first',
  },
  {
    title: 'Selecting a stack',
    description:
      'Use the Build branches to compare model families, simulators, learning frameworks, robot middleware, and control libraries.',
    link: '/docs/quick-start#build-systems',
  },
  {
    title: 'Planning hands-on work',
    description:
      'Follow the staged project ladder from beginner exercises to foundation-model fine-tuning and optional hardware builds.',
    link: '/docs/quick-start#use-the-project-ladder',
  },
  {
    title: 'Improving the list',
    description:
      'Use the curation standards, formatting rules, and pull request steps before editing the root README.',
    link: '/docs/workflow#contribute-to-the-list',
  },
];

const repoAnatomy = [
  {
    title: 'README.md',
    description:
      'Primary catalog with the curated sections, subsections, project ladder, and related lists.',
    href: 'https://github.com/natnew/awesome-physical-ai/blob/main/README.md',
  },
  {
    title: 'CONTRIBUTING.md',
    description:
      'Maintainer policy for quality standards, formatting, alphabetical ordering, and pull request expectations.',
    href: 'https://github.com/natnew/awesome-physical-ai/blob/main/CONTRIBUTING.md',
  },
  {
    title: 'LICENSE',
    description: 'MIT license file at the repository root.',
    href: 'https://github.com/natnew/awesome-physical-ai/blob/main/LICENSE',
  },
  {
    title: '.github/workflows/issue-triage-agent.lock.yml',
    description:
      'Scheduled weekday issue triage automation already present in the repository.',
    href: 'https://github.com/natnew/awesome-physical-ai/blob/main/.github/workflows/issue-triage-agent.lock.yml',
  },
];

const workflowSteps = [
  {
    title: 'Choose a lane',
    description:
      'Pick Learn, Build, Deploy, Stay Current, or the project ladder based on whether you need orientation, tools, deployment context, or practice.',
  },
  {
    title: 'Drop to the right subsection',
    description:
      'The README already separates topics like Foundation Models, Simulation Platforms, Datasets & Benchmarks, and People to Follow.',
  },
  {
    title: 'Validate with the source files',
    description:
      'This site compresses and routes; the root README and CONTRIBUTING guide remain the authoritative source files.',
  },
  {
    title: 'Contribute carefully',
    description:
      'New entries should satisfy the documented quality bar, match the Markdown format, and land in the most specific category.',
  },
];

const docsNavigation = [
  {
    title: 'Overview',
    description: 'What the repository contains and why the docs site exists.',
    link: '/docs/overview',
  },
  {
    title: 'Quick Start',
    description: 'Task-based routes into the catalog.',
    link: '/docs/quick-start',
  },
  {
    title: 'Workflow',
    description: 'How to use the list and how to contribute changes.',
    link: '/docs/workflow',
  },
  {
    title: 'Architecture',
    description: 'Repository anatomy, content structure, and site layout.',
    link: '/docs/architecture',
  },
  {
    title: 'Curation Standards',
    description: 'The actual inclusion and formatting rules from CONTRIBUTING.md.',
    link: '/docs/curation-standards',
  },
  {
    title: 'Scope & Limits',
    description: 'What the repository does not claim and where drift can happen.',
    link: '/docs/scope-and-limits',
  },
];

export default function Home() {
  return (
    <Layout
      title="Awesome Physical AI — Curated Physical AI Resources & Roadmap"
      description="Awesome Physical AI: a curated, engineering-oriented map of Physical AI resources, robotics resources, robot learning, embodied AI, embodied agents, robotics simulation, sim-to-real, world models, vision-language-action (VLA) models, Physical AI benchmarks, foundation models for robotics, generalist robot policies, and production-grade, safe embodied AI systems."
    >
      <main className={styles.main}>
          <header className={styles.hero}>
            <p className={styles.eyebrow}>Curated Physical AI resources — a Physical AI roadmap for robotics, embodied AI, simulation, world models, and VLA models</p>
            <h1 className={styles.title}>
              <span className={styles.titleLine}>Physical AI &amp; Embodied AI:</span>
              <span className={styles.titleLine}>Find resources faster.</span>
            </h1>
            <p className={styles.lead}>
              Awesome Physical AI is a curated, engineering-oriented map of Physical AI resources —
              robotics resources, robot learning, embodied agents, robotics simulation, sim-to-real,
              world models, vision-language-action (VLA) models, Physical AI benchmarks, foundation
              models for robotics, generalist robot policies, and patterns for production-grade,
              safe embodied AI systems. The repository is centered on a single README plus
              contribution policy and GitHub workflow files, so this site focuses on orientation,
              navigation, and the curation model rather than a fictional software runtime.
            </p>
            <div className={styles.actions}>
              <Link className="button button--primary button--lg" to="/docs/overview">
                Explore Docs
              </Link>
              <Link
                className="button button--secondary button--lg"
                href="https://github.com/natnew/awesome-physical-ai/blob/main/README.md"
              >
                Main README
              </Link>
            </div>
            <div className={styles.signalGrid}>
              {proofSignals.map((signal) => (
                <div key={signal.label} className={styles.signalCard}>
                  <span className={styles.signalLabel}>{signal.label}</span>
                  <strong>{signal.value}</strong>
                </div>
              ))}
            </div>
          </header>

          <section className={styles.sectionPanel}>
            <div className={styles.sectionHeader}>
              <p className={styles.sectionEyebrow}>Core coverage</p>
              <h2>Use the repository by intent, not by scrolling.</h2>
              <p>
                The root README already separates the field into learning material, build-time
                tools, deployment context, current-awareness sources, and staged hands-on projects.
              </p>
            </div>
            <div className={styles.cardGrid}>
              {coverage.map((item) => (
                <Link key={item.title} className={styles.card} to={item.link}>
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </Link>
              ))}
            </div>
          </section>

          <section className={styles.sectionSplit}>
            <div className={styles.sectionPanel}>
              <div className={styles.sectionHeader}>
                <p className={styles.sectionEyebrow}>Fast paths</p>
                <h2>Shortest routes into the catalog.</h2>
              </div>
              <div className={styles.stack}>
                {fastPaths.map((item) => (
                  <Link key={item.title} className={styles.pathCard} to={item.link}>
                    <div>
                      <h3>{item.title}</h3>
                      <p>{item.description}</p>
                    </div>
                    <span className={styles.arrow}>/</span>
                  </Link>
                ))}
              </div>
            </div>

            <div className={styles.sectionPanel}>
              <div className={styles.sectionHeader}>
                <p className={styles.sectionEyebrow}>Repository anatomy</p>
                <h2>Every major statement traces back to a tracked file.</h2>
              </div>
              <div className={styles.stack}>
                {repoAnatomy.map((item) => (
                  <a key={item.title} className={styles.pathCard} href={item.href}>
                    <div>
                      <h3>{item.title}</h3>
                      <p>{item.description}</p>
                    </div>
                    <span className={styles.arrow}>/</span>
                  </a>
                ))}
              </div>
            </div>
          </section>

          <section className={styles.sectionPanel}>
            <div className={styles.sectionHeader}>
              <p className={styles.sectionEyebrow}>Workflow</p>
              <h2>How this repository is meant to be used.</h2>
              <p>
                The site mirrors the editorial workflow already implied by the README and the
                contribution guide: navigate, compare, validate, then improve the list carefully.
              </p>
            </div>
            <div className={styles.workflowGrid}>
              {workflowSteps.map((step, index) => (
                <div key={step.title} className={styles.workflowCard}>
                  <span className={styles.workflowIndex}>0{index + 1}</span>
                  <h3>{step.title}</h3>
                  <p>{step.description}</p>
                </div>
              ))}
            </div>
          </section>

          <section className={styles.sectionSplit}>
            <div className={styles.sectionPanel}>
              <div className={styles.sectionHeader}>
                <p className={styles.sectionEyebrow}>Docs map</p>
                <h2>Jump directly to the useful pages.</h2>
              </div>
              <div className={styles.cardGrid}>
                {docsNavigation.map((item) => (
                  <Link key={item.title} className={styles.card} to={item.link}>
                    <h3>{item.title}</h3>
                    <p>{item.description}</p>
                  </Link>
                ))}
              </div>
            </div>

            <div className={styles.sectionPanel}>
              <div className={styles.sectionHeader}>
                <p className={styles.sectionEyebrow}>Scope</p>
                <h2>What this repository is not.</h2>
              </div>
              <div className={styles.scopeCard}>
                <p>
                  Outside this docs layer, the original repository content is editorial rather than
                  executable: it does not ship an SDK, service, training pipeline, or deployment
                  package. The value is the selection and structure of external resources.
                </p>
                <p>
                  External links, maintenance status, pricing, and claims made by third-party
                  projects can drift. The docs site keeps those limitations visible instead of
                  converting the README into generic product marketing.
                </p>
                <Link className={styles.inlineLink} to="/docs/scope-and-limits">
                  Read the scope notes
                </Link>
              </div>
            </div>
          </section>
      </main>
    </Layout>
  );
}