/*
  Local remark plugin: automatically number document headings (h2, h3, ...).

  Numbers are injected into the heading text at build time, so they also show
  up in the on-page TOC and the local search index, and the heading anchors
  are derived from the numbered text (consistent with the manual numbering that
  used to live in the sources).

  Numbering restarts at 1 for every document, matching the previous hand-written
  numbers. Depth-1 headings (page titles) are never numbered.
*/

import { visit } from 'unist-util-visit';

function remarkHeadingNumbering(options = {}) {
  const startDepth = options.startDepth ?? 2;
  const maxDepth = options.maxDepth ?? 6;

  return function transformer(tree) {
    // Counters are reset for every document, so numbering restarts at 1 per file.
    // counters[depth] = how many headings have been seen at that depth so far.
    const counters = Array(maxDepth + 1).fill(0);

    visit(tree, 'heading', (node) => {
      const depth = node.depth;
      if (depth < startDepth || depth > maxDepth) {
        return;
      }
      // Entering a shallower level invalidates all deeper counters.
      for (let d = depth + 1; d <= maxDepth; d++) {
        counters[d] = 0;
      }
      counters[depth] += 1;

      const parts = [];
      for (let d = startDepth; d <= depth; d++) {
        // Missing intermediate levels (malformed nesting) fall back to 1.
        parts.push(counters[d] || 1);
      }

      // Trailing dot only on top-level (startDepth) headings, e.g. "1. " vs "1.1 ".
      const trailingDot = depth === startDepth ? '.' : '';
      node.children.unshift({ type: 'text', value: `${parts.join('.')}${trailingDot} ` });
    });
  };
}

export default remarkHeadingNumbering;
