# Data and software notices

The 24 original pilot fixtures and 72 deterministic template variations are project-authored examples released under CC0-1.0. They are diagnostics, not 96 independent natural documents. All records have `training_eligible=false`, no assigned training split, and no accepted human gold.

Four unchanged prose excerpts reuse the existing `training/word_boundary_v0_2` review candidates. They are attributed to **MDN contributors** under [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/). Each input's nested `source` object preserves the original repository, pinned revision, path, source/excerpt hashes, character/byte ranges, attribution history URL, and change notice. English and Chinese parallel articles retain the same family and must not be split into train/evaluation independently. The excerpts are truncated views, not true document EOF.

The original licenses are copied to `licenses/MDN-en-us.md` and `licenses/MDN-zh-cn.md`. Result annotations of these excerpts retain the source attribution and CC BY-SA terms; the changes are word/phrase boundaries and derived span metadata. No paraphrasing or translation is intended; any such changes in raw model output are preserved failures, not corrected source documents. The full source snapshots remain in the original v0.2 package.

`inputs/rule-references.jsonl` contains offline jieba and bounded FMM diagnostics. jieba's MIT notice is copied to `licenses/jieba-MIT.txt`. The known `研究生｜命` failure belongs to FMM; the old offline jieba reference is `研究｜生命｜起源`. Rules are never inserted into the teacher prompt or the student Observation Tape.

llama.cpp is MIT-licensed; models retain their own upstream licenses (Qwen Apache-2.0, GLM MIT). Exact source repositories, revisions, conversion publishers, hashes and local filenames are in `assets.lock.json`. Q8_0 files are third-party Unsloth conversions, not official BF16/FP8 releases. Weights and executable binaries are ignored by Git. Hash verification establishes artifact identity, not an independent supply-chain or quality audit.
