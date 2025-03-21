/*
    This file is modified from
    https://github.com/jknoxville/remark-code-snippets/blob/master/src/index.ts
*/

import fs from 'fs';
import path from 'path';
import visit from 'unist-util-visit';
import {Node, Parent} from 'unist';
import {Transformer} from 'unified';
const {parseArgs} = require('./arguments');

const referencedFiles = new Set<string>();

type Options = {
  async?: Boolean,
  baseDir?: string,
  ignoreMissingFiles?: boolean,
};

export default function codeImport(options: Options = {}): Transformer {
  return function transformer(tree, file): Promise<void> | void {
    const codes: [Node, number, Parent | undefined][] = [];
    const promises = [];

    visit(tree, 'code', (node, index, parent) => {
      codes.push([node, index, parent]);
    });

    for (const [node] of codes) {

      // If someone tries to import a file, but forgets to add a language tag e.g ```json
      // then the meta string will be interpreted as the language. So check the lang prop for file=
      // and show a helpful error if this is the case, or else importing wont work for them.
      if (hasLang(node) && node.lang.startsWith('file=')) {
        throw new Error(`Language tag missing on code block snippet in ${file.history}`)
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
          new Promise<void>((resolve, reject) => {
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
          })
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

function getSnippet(fileContent: string, args: { anchor: any; file: any; }) {
  let lines = fileContent.trim().split('\n');

  if (args.anchor === undefined) {
    return removeCommonIndentation(lines).join('\n');
  }

  let startingLine = 0;
  let endingLine = undefined;

  {
    const numbers = getLineNumbersOfOccurrence(lines, "ANCHOR: " + args.anchor);
    if (numbers.length === 0) {
      throw new Error(`Code block start marker "${args.anchor}" not found in file ${args.file}`);
    }
    if (numbers.length > 1) {
      throw new Error(`Ambiguous code block start marker. Found more than once in ${args.file}, at lines ${numbers}`);
    }
    startingLine = numbers[0] + 1;
  }

  {
    const numbers = getLineNumbersOfOccurrence(lines, "ANCHOR_END: " + args.anchor);
    if (numbers.length === 0) {
      throw new Error(`Code block end marker "${args.anchor}" not found in file ${args.file}`);
    }
    if (numbers.length > 1) {
      throw new Error(`Ambiguous code block end marker. Found more than once in ${args.file}, at lines ${numbers}`);
    }
    endingLine = numbers[0];
  }

  lines = lines.slice(startingLine, endingLine);

  let lines_rmident = removeCommonIndentation(lines).join('\n');

  return lines_rmident;
}

function removeCommonIndentation(lines: string[]): string[] {
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

  return lines.map(line => line.slice(commonIndentation));
}

function getLineNumbersOfOccurrence(lines: string[], searchTerm: string) {
  let lineNumbers: number[] = [];
  lines.forEach((line, index) => {
    const searched = line.endsWith(searchTerm);
    if (searched) {
      lineNumbers.push(index);
    }
  });
  return lineNumbers;
}

function hasLang(node: Node): node is Node & {lang: string} {
  return Boolean(node.lang) && typeof node.lang === 'string';
}

function logReferencedFile(filepath: string): void {
  const relativePath = path.relative(process.cwd(), filepath);
  referencedFiles.add(relativePath);
}

export function getReferencedFiles(): string[] {
  return Array.from(referencedFiles);
}