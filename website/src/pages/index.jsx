import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

import styles from './index.module.css';

// Order mirrors `categoriesSidebar` in website/sidebars.js.
// Blurbs adapted from website/docs/categories/index.mdx for voice consistency.
const categories = [
  {
    title: 'Simulators',
    blurb:
      'Physics engines and rendering stacks where policies are trained and stress-tested before touching hardware.',
    href: '/docs/categories/simulators',
  },
  {
    title: 'Datasets',
    blurb:
      'Recorded robot behaviour — teleoperation, demonstrations, egocentric video — that feeds imitation learning and pretraining.',
    href: '/docs/categories/datasets',
  },
  {
    title: 'Benchmarks',
    blurb:
      'Fixed task suites for comparing policies under controlled conditions across manipulation, locomotion, and embodied reasoning.',
    href: '/docs/categories/benchmarks',
  },
  {
    title: 'Evaluation Methodology',
    blurb:
      'How to measure policies correctly — harnesses, metrics, and statistical practice that turn rollouts into defensible claims.',
    href: '/docs/categories/evaluation-methodology',
  },
  {
    title: 'Robotics Foundation Models',
    blurb:
      'Generalist pretrained policies and vision-language-action (VLA) models that map perception and language to robot actions.',
    href: '/docs/categories/robotics-foundation-models',
  },
  {
    title: 'World Models',
    blurb:
      'Learned predictive models of physical dynamics used for imagination-based planning, sample-efficient RL, and synthetic data.',
    href: '/docs/categories/world-models',
  },
  {
    title: 'Manipulation',
    blurb:
      'Grasping, dexterous, and contact-rich methods — analytic planners, diffusion and transformer policies, and the rigs that feed them.',
    href: '/docs/categories/manipulation',
  },
  {
    title: 'Locomotion',
    blurb:
      'Legged, bipedal, and humanoid controllers and the GPU-parallel training stacks that make modern sim-to-real practical.',
    href: '/docs/categories/locomotion',
  },
  {
    title: 'Sim-to-Real',
    blurb:
      'Techniques for closing the reality gap — domain randomisation, system identification, residual learning, and real-robot fine-tuning.',
    href: '/docs/categories/sim-to-real',
  },
  {
    title: 'Safety & Robustness',
    blurb:
      'Constrained training, adversarial evaluation, and failure-mode analysis that decide whether a system is shippable.',
    href: '/docs/categories/safety-and-robustness',
  },
  {
    title: 'Governance & Policy',
    blurb:
      'Standards, regulation, and risk frameworks that shape how Physical AI systems are evaluated and deployed.',
    href: '/docs/categories/governance-and-policy',
  },
  {
    title: 'Production Patterns',
    blurb:
      'Middleware, planners, fleet orchestration, and observability for shipping robots in production.',
    href: '/docs/categories/production-patterns',
  },
  {
    title: 'Courses',
    blurb:
      'Structured learning paths into robot learning and embodied AI for individuals and team onboarding.',
    href: '/docs/categories/courses',
  },
  {
    title: 'Companies',
    blurb:
      'Organisations actively shaping Physical AI in industry — foundation-model labs, platform builders, and applied deployers.',
    href: '/docs/categories/companies',
  },
];

const nextSteps = [
  {
    title: 'Read the overview',
    description:
      'Mission, audience, and how the catalog is organised — the fastest way to decide whether this resource fits your work.',
    link: '/docs/overview',
  },
  {
    title: 'Curation standards',
    description:
      'The selection bar every entry must clear: maintained, relevant, distinctive, and load-bearing for real engineering work.',
    link: '/docs/curation-standards',
  },
  {
    title: 'Scope & limits',
    description:
      'What this list does not claim. Useful for technical leaders evaluating where the catalog ends and your own diligence begins.',
    link: '/docs/scope-and-limits',
  },
];

export default function Home() {
  return (
    <Layout
      title="Awesome Physical AI — Curated Physical AI Resources & Roadmap"
      description="Awesome Physical AI: a curated, engineering-oriented map of Physical AI resources — simulators, datasets, benchmarks, evaluation methodology, foundation models, world models, manipulation, locomotion, sim-to-real, safety, governance, production patterns, courses, and companies — for practitioners and technical leaders building or evaluating embodied AI systems."
    >
      <main className={styles.main}>
        <header className={styles.hero}>
          <p className={styles.eyebrow}>For practitioners and technical leaders</p>
          <h1 className={styles.title}>
            <span className={styles.titleLine}>An engineering map</span>
            <span className={styles.titleLine}>of Physical AI.</span>
          </h1>
          <p className={styles.lead}>
            Awesome Physical AI is a curated, engineering-oriented catalog across 14 categories —
            simulators, datasets, benchmarks, evaluation methodology, foundation models, world
            models, manipulation, locomotion, sim-to-real, safety, governance, production
            patterns, courses, and companies. Researchers and ML/robotics engineers use it to
            shortlist tooling and reference work; technical leaders use it to scope evaluation,
            risk, and deployment for embodied AI systems. Selection lens over hype.
          </p>
          <div className={styles.actions}>
            <Link className="button button--primary button--lg" to="/docs/categories">
              Browse categories
            </Link>
            <Link className="button button--secondary button--lg" to="/docs/overview">
              Read the overview
            </Link>
          </div>
        </header>

        <section className={styles.sectionPanel}>
          <div className={styles.sectionHeader}>
            <p className={styles.sectionEyebrow}>Browse by category</p>
            <h2>14 categories, each with a selection lens.</h2>
            <p>
              Every category page opens with what it is, why it matters for evaluation or
              deployment, and how to choose between entries — followed by a short, vetted list.
            </p>
          </div>
          <div className={styles.categoryGrid}>
            {categories.map((item) => (
              <Link key={item.title} className={styles.categoryCard} to={item.href}>
                <h3>{item.title}</h3>
                <p>{item.blurb}</p>
              </Link>
            ))}
          </div>
        </section>

        <section className={styles.sectionPanel}>
          <div className={styles.sectionHeader}>
            <p className={styles.sectionEyebrow}>Next steps</p>
            <h2>Decide whether this fits your work.</h2>
          </div>
          <div className={styles.cardGrid}>
            {nextSteps.map((item) => (
              <Link key={item.title} className={styles.card} to={item.link}>
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </Link>
            ))}
          </div>
        </section>
      </main>
    </Layout>
  );
}