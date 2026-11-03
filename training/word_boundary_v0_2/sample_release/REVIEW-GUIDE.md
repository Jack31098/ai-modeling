# Human review of independent candidates

1. Copy a `reviews/*.json` file to your review workspace. Keep the original candidate
   ID, raw text and excerpt hash. Files start `pending`, with no annotator or units.
   They are templates, not accepted gold. Do not edit an immutable release in place.
2. Read the raw excerpt before opening the separately stored automatic suggestions.
   Add every unit as `{"text":"...","cp":[start,end],"byte":[start,end]}`. Spans are
   half-open. `cp` counts Unicode code points; use `cp_to_byte` to obtain UTF-8 offsets.
   Include whitespace and punctuation; concatenating units must recover the text exactly.
3. Follow v0 orthographic conventions for English abbreviations, internal apostrophes,
   hyphens, numbers and URLs. For Chinese, record your chosen word boundary convention.
   Put competing analyses in `ambiguities`, each with `span_cp`, `options`, `reason`,
   and `resolution`. Useful review points: 字符集/字符编码, two-character verbs and their
   complements, technical compounds, 中文 mixed with ASCII/UTF-8, numeric ranges and
   parentheses. These are prompts to inspect, not supplied gold answers.
4. Set your actual `annotator` and `review_status: "accepted"` only after resolving all
   ambiguities and covering the entire excerpt. `candidates.validate_review` rejects
   pending, unnamed, incomplete, misaligned or unresolved files. The validator checks
   structure, not whether a purported annotator is a real human.
5. Keep both topics and both languages isolated from training and tuning. Future
   ingestion must enforce `holdout-registry.json` and repeat overlap/lineage audits.
   Candidate discovery or automatic labels do not constitute gold acceptance. Review
   changes require a new review version/manifest. Do not report gold accuracy before
   accepted human files exist; this increment does not compute one.
