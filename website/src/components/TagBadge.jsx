import React from 'react';
import styles from './TagBadge.module.css';

const TAG_COLORS = {
  'open-source': '#10b981',
  'paper': '#8b5cf6',
  'tool': '#0ea5e9',
  'framework': '#f59e0b',
  'benchmark': '#ef4444',
  'dataset': '#ec4899',
  'simulator': '#06b6d4',
  'production-ready': '#22c55e',
  'research-only': '#6366f1',
  'commercial': '#d97706',
};

export default function TagBadge({ tag }) {
  if (!tag) return null;

  const color = TAG_COLORS[tag] || '#6b7280';
  const normalizedTag = tag.replace(/-/g, ' ');

  return (
    <span
      className={styles.badge}
      style={{ backgroundColor: color }}
      title={`Tag: ${normalizedTag}`}
    >
      {normalizedTag}
    </span>
  );
}

export function TagList({ tags }) {
  if (!tags || tags.length === 0) return null;

  return (
    <div className={styles.tagContainer}>
      {tags.map((tag) => (
        <TagBadge key={tag} tag={tag} />
      ))}
    </div>
  );
}
