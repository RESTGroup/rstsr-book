// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import { themes as prismThemes } from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import parseArgs from './src/remark-code-spinnets';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// https://docusaurus.io/docs/api/docusaurus-config#using-environment-variables
// https://github.com/facebook/docusaurus/discussions/4480
function getSiteTitle() {
  switch(process.env.DOCUSAURUS_CURRENT_LOCALE) {
    case "zh-hans": return "RSTSR: n-维 Rust 张量程序库";
    default: return "RSTSR: An n-Dimensional Rust Tensor Toolkit";
  }
}

function getSiteTagline() {
  switch(process.env.DOCUSAURUS_CURRENT_LOCALE) {
    case "zh-hans": return "快速、直观、可扩展的原生 Rust 科学计算工具包";
    default: return "Fast, Intuitive, Extensible for scientific computation in native Rust.";
  }
}

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: getSiteTitle(),
  tagline: getSiteTagline(),
  favicon: 'img/logo-64.ico',

  // Set the production url of your site here
  url: 'https://rstsr-book.readthedocs.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/rstsr-book/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'RSTSR developer(s)', // Usually your GitHub org/user name.
  projectName: 'rstsr-book', // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-hans'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          remarkPlugins: [remarkMath, parseArgs],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],
  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'dev',
        path: 'dev',
        routeBasePath: 'dev',
        sidebarPath: require.resolve('./sidebars-dev.js'),
      },
    ],
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      crossorigin: 'anonymous',
    },
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        // title: 'Document',
        logo: {
          alt: 'My Site Logo',
          src: 'img/logo-2.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'userDocSidebar',
            position: 'left',
            label: 'User Document',
          },
          {
            type: 'docSidebar',
            sidebarId: 'devDocSidebar',
            docsPluginId: 'dev',
            position: 'left',
            label: 'Dev Document',
          },
          // {to: '/blog', label: 'Blog', position: 'left'},
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/RESTGroup/rstsr',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          /*{
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'X',
                href: 'https://x.com/docusaurus',
              },
            ],
          },
          */
          {
            title: 'Git Repo Links',
            items: [
              /* {
                label: 'Blog',
                to: '/blog',
              }, */
              { label: 'RSTSR', href: 'https://github.com/RESTGroup/rstsr' },
              { label: 'RSTSR Document', href: 'https://github.com/RESTGroup/rstsr-book' },
              { label: 'REST', href: 'https://gitee.com/RESTGroup/rest' },
            ],
          },
          {
            title: 'API Document',
            items: [
              { label: 'rstsr', href: 'https://docs.rs/rstsr' },
              { label: 'rstsr-core', href: 'https://docs.rs/rstsr-core' },
              { label: 'rstsr-openblas', href: 'https://docs.rs/rstsr-openblas' },
            ]
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} RSTSR developer(s). Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
  themes: [
    // ... Your other themes.
    [
      // @ts-ignore
      require.resolve("@easyops-cn/docusaurus-search-local"),
      /** @type {import("@easyops-cn/docusaurus-search-local").PluginOptions} */
      // @ts-ignore
      ({
        // ... Your options.
        // `hashed` is recommended as long-term-cache of index file is possible.
        hashed: true,

        // For Docs using Chinese, it is recomended to set:
        language: ["en", "zh"],

        // If you're using `noIndex: true`, set `forceIgnoreNoIndex` to enable local index:
        // forceIgnoreNoIndex: true,
      }),
    ],
  ],
};

export default config;
