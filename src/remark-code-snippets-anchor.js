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
import visit from 'unist-util-visit';

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

const referencedFiles = new Set();

function codeImport(options = {}) {
  return function transformer(tree, file) {
    const codes = [];
    const promises = [];

    visit(tree, 'code', (node, index, parent) => {
      codes.push([node, index, parent]);
    });

    for (const [node] of codes) {
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
        continue;
      }
      const fileAbsPath = path.resolve(options.baseDir ?? (file.dirname || ''), args.file);
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
