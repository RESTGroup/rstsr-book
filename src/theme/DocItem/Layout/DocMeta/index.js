import React from 'react';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

// ── i18n labels ─────────────────────────────────────────────────────────
const LABELS = {
  en: {
    written: 'Written',
    version: 'Version',
    generated: 'Generated',
    translated: 'Translated from',
    reviewed: 'Reviewed',
    ai: 'AI',
    human: 'Human',
    partial: 'Partial',
    yes: 'Yes',
    no: 'No',
  },
  'zh-hans': {
    written: '编写时间',
    version: '版本',
    generated: '生成方式',
    translated: '翻译自',
    reviewed: '审阅状态',
    ai: 'AI',
    human: '人工',
    partial: '部分',
    yes: '是',
    no: '否',
  },
};

const LANG_NAMES = {
  en: { en: 'English', 'zh-hans': '简体中文' },
  'zh-hans': { en: 'English', 'zh-hans': '简体中文' },
};

function useT() {
  const { i18n } = useDocusaurusContext();
  const locale = i18n?.currentLocale || 'en';
  return LABELS[locale] || LABELS.en;
}

// ── Helpers ─────────────────────────────────────────────────────────────

/** Format a date-like value for display. */
function formatDate(value) {
  if (!value) return null;
  const s = String(value);
  const parsed = new Date(s);
  if (!isNaN(parsed.getTime())) {
    return parsed.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  }
  return s;
}

// ── Sub-components ──────────────────────────────────────────────────────

/** Badge with label + value. */
function Field({ label, children }) {
  return (
    <span className={styles.metaField}>
      <span className={styles.metaLabel}>{label}</span>
      {children}
    </span>
  );
}

/** Colored badge pill for status values. */
function Badge({ value, kind }) {
  if (value === undefined || value === null) return null;
  let text, cls;
  if (kind === 'generated') {
    if (value === true || value === 'true') {
      text = 'ai';
      cls = styles.badgeWarn;
    } else if (value === false || value === 'false') {
      text = 'human';
      cls = styles.badgeOk;
    } else if (value === 'partial') {
      text = 'partial';
      cls = styles.badgeInfo;
    } else {
      return <span className={styles.metaValue}>{String(value)}</span>;
    }
  } else if (kind === 'reviewed') {
    if (value === true || value === 'true') {
      text = 'yes';
      cls = styles.badgeOk;
    } else if (value === false || value === 'false') {
      text = 'no';
      cls = styles.badgeWarn;
    } else if (value === 'partial') {
      text = 'partial';
      cls = styles.badgeInfo;
    } else {
      return <span className={styles.metaValue}>{String(value)}</span>;
    }
  } else {
    return <span className={styles.metaValue}>{String(value)}</span>;
  }
  const t = useT();
  return <span className={`${styles.badge} ${cls}`}>{t[text]}</span>;
}

// ── Main ────────────────────────────────────────────────────────────────

/** Extract metadata. All fields are optional — null when not set. */
function extractMeta(frontMatter) {
  const m = frontMatter.rstsr_meta || {};
  return {
    written: m.written ?? frontMatter.rstsr_written ?? null,
    version: m.rstsr_version ?? frontMatter.rstsr_version ?? null,
    translated: m.translated ?? frontMatter.translated ?? null,
    reviewed: m.reviewed ?? frontMatter.reviewed ?? null,
    ai_generated: m.ai_generated ?? frontMatter.ai_generated ?? null,
  };
}

export default function DocMeta() {
  const { frontMatter } = useDoc();
  const meta = extractMeta(frontMatter);
  const t = useT();
  const langNames = LANG_NAMES[useDocusaurusContext().i18n?.currentLocale || 'en'] || LANG_NAMES.en;

  // Fields are built in display order; null values are skipped.
  const fields = [];

  if (meta.written) {
    fields.push(
      <Field key="written" label={t.written}>
        <span className={styles.metaValue}>{formatDate(meta.written)}</span>
      </Field>,
    );
  }

  if (meta.version) {
    fields.push(
      <Field key="version" label={t.version}>
        <span className={styles.metaValue}>v{meta.version}</span>
      </Field>,
    );
  }

  if (meta.ai_generated !== null) {
    fields.push(
      <Field key="ai_generated" label={t.generated}>
        <Badge value={meta.ai_generated} kind="generated" />
      </Field>,
    );
  }

  // translated: only render when explicitly set
  if (meta.translated) {
    fields.push(
      <Field key="translated" label={t.translated}>
        <span className={styles.metaValue}>{langNames[meta.translated] || meta.translated}</span>
      </Field>,
    );
  }

  // reviewed: only render when explicitly set
  if (meta.reviewed !== null) {
    fields.push(
      <Field key="reviewed" label={t.reviewed}>
        <Badge value={meta.reviewed} kind="reviewed" />
      </Field>,
    );
  }

  // If no fields were set, render nothing.
  if (fields.length === 0) return null;

  // ── Assemble with · separators ──────────────────────────────────────
  const withSeparators = fields.reduce((acc, field, i) => {
    if (i > 0) {
      acc.push(
        <span key={`sep-${i}`} className={styles.separator}>
          ·
        </span>,
      );
    }
    acc.push(field);
    return acc;
  }, []);

  return <div className={styles.metaBar}>{withSeparators}</div>;
}
