/**
 * Extract tags from a README entry line and the next line.
 * Tags are stored in HTML comments: <!-- tags: tag1, tag2, tag3 -->
 */
export function extractTagsFromEntry(entryLine, nextLine = '') {
  if (!nextLine) return [];

  const tagMatch = nextLine.match(/<!--\s*tags:\s*([^-]*?)-->/);
  if (!tagMatch || !tagMatch[1]) return [];

  return tagMatch[1]
    .split(',')
    .map((tag) => tag.trim())
    .filter((tag) => tag.length > 0);
}

/**
 * Canonical tag taxonomy with descriptions.
 */
export const TAG_TAXONOMY = {
  'tool': 'Software, simulator, framework, or library for robotics development',
  'paper': 'Research paper or academic publication',
  'dataset': 'Dataset for training or evaluation',
  'benchmark': 'Benchmark suite or evaluation harness',
  'simulator': 'Physics engine or simulation environment',
  'framework': 'Architectural or algorithmic methodology',
  'course': 'Educational program or lecture series',
  'production-ready': 'Suitable for production deployment; actively maintained',
  'research-only': 'Experimental or research-focused; may require adaptation',
  'open-source': 'Released under an open-source license',
  'commercial': 'Commercial product or closed-source',
};

export function getTagDescription(tag) {
  return TAG_TAXONOMY[tag] || tag;
}

export const VALID_TAGS = Object.keys(TAG_TAXONOMY);
