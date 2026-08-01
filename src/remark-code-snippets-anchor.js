/*
  Local remark plugin: import source-file snippets into fenced code blocks.

  Forked from remark-code-snippets (https://github.com/jknoxville/remark-code-snippets)
  with the added `anchor=` mechanism (compatible with The Rust Book's
  `// ANCHOR: <name>` ... `// ANCHOR_END: <name>` markers), which is what the
  RSTSR book relies on. The published npm package only supports `start=`/`end=`,
  so this local copy must be kept.

  Usage in a code fence:
    ```rust file=../../listings/features-default/tests/foo.rs anchor=bar
    ```
  ...pulls the lines between `// ANCHOR: bar` and `// ANCHOR_END: bar`.
*/

import fs from 'fs';
import path from 'path';
import { visit } from 'unist-util-visit';

function resolveShortName(shortName, listingsDir) {
  const results = [];
  function walk(dir) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile()) {
        const stem = path.basename(entry.name, path.extname(entry.name));
        if (stem === shortName) {
          results.push(fullPath);
        }
      }
    }
  }
  walk(listingsDir);

  if (results.length === 0) {
    throw new Error(
      `Short name "${shortName}" not found in listings directory "${listingsDir}"`,
    );
  }
  if (results.length > 1) {
    throw new Error(
      `Ambiguous short name "${shortName}". Found multiple files:\n` +
        results.map((r) => '  - ' + path.relative(listingsDir, r)).join('\n'),
    );
  }
  return results[0];
}

function parseArgs(meta) {
  const result = {};
  meta.split(' ').forEach((arg) => {
    const keyLength = arg.indexOf('=');
    if (keyLength < 0) {
      return;
    }
    const key = arg.slice(0, keyLength);
    const value = arg.slice(keyLength + 1);
    result[key] = value;
  });
  return result;
}

const FERRIS_TITLES = {
  panics: {
    en: 'This code will panic!',
    'zh-hans': '此代码将会 panic！',
  },
  does_not_compile: {
    en: 'This code does not compile!',
    'zh-hans': '此代码无法编译！',
  },
  not_desired_behavior: {
    en: 'Not desired behavior!',
    'zh-hans': '非期望行为！',
  },
};

function detectLocale(history) {
  // history is an array of file paths; i18n files contain "i18n/<locale>/"
  if (history && history.length) {
    const m = history[0].match(/i18n\/([^/]+)\//);
    if (m) return m[1];
  }
  return 'en';
}

const referencedFiles = new Set();

function makeUseBaseUrlExpr(pathStr) {
  return {
    type: 'mdxJsxAttributeValueExpression',
    value: `useBaseUrl("${pathStr}")`,
    data: {
      estree: {
        type: 'Program',
        body: [
          {
            type: 'ExpressionStatement',
            expression: {
              type: 'CallExpression',
              callee: { type: 'Identifier', name: 'useBaseUrl' },
              arguments: [
                {
                  type: 'Literal',
                  value: pathStr,
                  raw: JSON.stringify(pathStr),
                },
              ],
              optional: false,
            },
          },
        ],
        sourceType: 'module',
        comments: [],
      },
    },
  };
}

function applyFerrisOverlay(node, ferrisType, parent, index, locale) {
  if (!ferrisType) {
    return;
  }
  if (!(ferrisType in FERRIS_TITLES)) {
    throw new Error(
      `Invalid ferris type "${ferrisType}". Valid types: ${Object.keys(FERRIS_TITLES).join(', ')}`,
    );
  }

  const title = FERRIS_TITLES[ferrisType][locale] ?? FERRIS_TITLES[ferrisType].en;

  const wrapper = {
    type: 'mdxJsxFlowElement',
    name: 'div',
    attributes: [
      { type: 'mdxJsxAttribute', name: 'className', value: 'ferris-overlay' },
    ],
    children: [
      node,
      {
        type: 'mdxJsxFlowElement',
        name: 'span',
        attributes: [
          { type: 'mdxJsxAttribute', name: 'className', value: 'ferris-icon' },
          { type: 'mdxJsxAttribute', name: 'title', value: title },
        ],
        children: [
          {
            type: 'mdxJsxFlowElement',
            name: 'img',
            attributes: [
              {
                type: 'mdxJsxAttribute',
                name: 'src',
                value: makeUseBaseUrlExpr(`/img/ferris/${ferrisType}.svg`),
              },
              { type: 'mdxJsxAttribute', name: 'alt', value: ferrisType },
            ],
            children: [],
          },
        ],
      },
    ],
  };

  if (parent && typeof index === 'number') {
    parent.children.splice(index, 1, wrapper);
  } else {
    // Should not happen for block-level code nodes, but handle gracefully.
    throw new Error('Cannot apply ferris overlay: missing parent or index');
  }
}

function codeImport(options = {}) {
  return function transformer(tree, file) {
    const codes = [];
    const promises = [];

    visit(tree, 'code', (node, index, parent) => {
      codes.push([node, index, parent]);
    });

    for (const [node, index, parent] of codes) {
      // If someone forgets the language tag, the meta string is read as the
      // language; detect `file=` there and give a helpful error.
      if (hasLang(node) && node.lang.startsWith('file=')) {
        throw new Error(`Language tag missing on code block snippet in ${file.history}`);
      }
      if (!node.meta) {
        continue;
      }
      const args = parseArgs(node.meta);
      if (!args.file) {
        // Standalone ferris overlay (no file import) — wrap inline code block
        if (args.ferris) {
          applyFerrisOverlay(node, args.ferris, parent, index, detectLocale(file.history));
        }
        continue;
      }
      // Short-name lookup: if file= has no path separator and listingsDir is
      // configured, search the listings directory recursively for a unique
      // file whose stem (name without extension) matches.
      let fileAbsPath;
      if (!args.file.includes('/') && options.listingsDir) {
        const listingsAbsDir = path.resolve(
          options.baseDir ?? process.cwd(),
          options.listingsDir,
        );
        fileAbsPath = resolveShortName(args.file, listingsAbsDir);
      } else {
        fileAbsPath = path.resolve(options.baseDir ?? (file.dirname || ''), args.file);
      }
      logReferencedFile(fileAbsPath);

      if (options.async) {
        promises.push(
          new Promise((resolve, reject) => {
            fs.readFile(fileAbsPath, 'utf8', (err, fileContent) => {
              if (err) {
                if (options.ignoreMissingFiles) {
                  node.value = `Referenced file from ${file.name} (${args.file}) not found.`;
                  resolve();
                  return;
                }
                reject(err);
                return;
              }
              node.value = getSnippet(fileContent, args);
              applyFerrisOverlay(node, args.ferris, parent, index, detectLocale(file.history));
              resolve();
            });
          }),
        );
      } else {
        if (!fs.existsSync(fileAbsPath)) {
          if (options.ignoreMissingFiles) {
            node.value = `Referenced file from ${file.name} (${args.file}) not found.`;
            continue;
          }
          throw new Error(`File not found: ${args.file}`);
        }
        const fileContent = fs.readFileSync(fileAbsPath, 'utf8');
        node.value = getSnippet(fileContent, args);
        applyFerrisOverlay(node, args.ferris, parent, index, detectLocale(file.history));
      }
    }

    if (promises.length) {
      return Promise.all(promises).then(() => {});
    }
  };
}

function getSnippet(fileContent, args) {
  let lines = fileContent.trim().split('\n');

  if (args.anchor === undefined) {
    return removeCommonIndentation(lines).join('\n');
  }

  let startingLine = 0;
  let endingLine = undefined;

  {
    const numbers = getLineNumbersOfOccurrence(lines, 'ANCHOR: ' + args.anchor);
    if (numbers.length === 0) {
      throw new Error(`Code block start marker "${args.anchor}" not found in file ${args.file}`);
    }
    if (numbers.length > 1) {
      throw new Error(`Ambiguous code block start marker. Found more than once in ${args.file}, at lines ${numbers}`);
    }
    startingLine = numbers[0] + 1;
  }

  {
    const numbers = getLineNumbersOfOccurrence(lines, 'ANCHOR_END: ' + args.anchor);
    if (numbers.length === 0) {
      throw new Error(`Code block end marker "${args.anchor}" not found in file ${args.file}`);
    }
    if (numbers.length > 1) {
      throw new Error(`Ambiguous code block end marker. Found more than once in ${args.file}, at lines ${numbers}`);
    }
    endingLine = numbers[0];
  }

  lines = lines.slice(startingLine, endingLine);
  return removeCommonIndentation(lines).join('\n');
}

function removeCommonIndentation(lines) {
  const commonIndentation = lines.reduce((minIndentation, line) => {
    if (line === '') {
      return minIndentation;
    }
    const m = line.match(/^( *)/);
    if (!m) {
      return 0;
    }
    return Math.min(m[1].length, minIndentation);
  }, Number.MAX_VALUE);

  return lines.map((line) => line.slice(commonIndentation));
}

function getLineNumbersOfOccurrence(lines, searchTerm) {
  const lineNumbers = [];
  lines.forEach((line, index) => {
    if (line.endsWith(searchTerm)) {
      lineNumbers.push(index);
    }
  });
  return lineNumbers;
}

function hasLang(node) {
  return Boolean(node.lang) && typeof node.lang === 'string';
}

function logReferencedFile(filepath) {
  const relativePath = path.relative(process.cwd(), filepath);
  referencedFiles.add(relativePath);
}

export function getReferencedFiles() {
  return Array.from(referencedFiles);
}

export default codeImport;
