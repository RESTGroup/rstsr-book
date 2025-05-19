import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';
import MainMarkdown from "./main-markdown.mdx";
import MDXContent from "@theme/MDXContent";

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <img
          alt="RSTSR logo"
          src="img/logo-3-white.png"
          style={{ maxHeight: '200px' }}
        ></img>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/fundamentals">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

function QuickStartSection() {
  return (
    <div className={clsx(styles.section, styles.sectionAlt)}>
      <div className="container">
        {/* <Heading as="h2" className={clsx("margin-bottom--lg", "text--center")}>
          It works locally, too 👩‍💻
        </Heading> */}
        <MDXContent>
          <MainMarkdown />
        </MDXContent>
      </div>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
      <QuickStartSection/>
    </Layout>
  );
}