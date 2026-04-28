import { TagList } from '../components/TagBadge';

/**
 * EntryWithTags component for use in MDX files.
 * Displays an entry with an optional list of tags.
 *
 * Usage in MDX:
 * <EntryWithTags
 *   name="Tool Name"
 *   url="https://example.com"
 *   description="Description of the tool"
 *   tags={['open-source', 'tool', 'production-ready']}
 * />
 */
export default function EntryWithTags({ name, url, description, tags = [] }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div style={{ marginBottom: '0.5rem' }}>
        <strong>
          <a href={url} target="_blank" rel="noopener noreferrer">
            {name}
          </a>
        </strong>{' '}
        — {description}
      </div>
      {tags && tags.length > 0 && <TagList tags={tags} />}
    </div>
  );
}
