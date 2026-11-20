# 粗粒度语言接口的中英文语料小样

[mdn.jsonl](mdn.jsonl) 含 7 个 MDN 术语的英文与简体中文连续正文摘录，共 14 条、1,991 个 Unicode 码点和 2,825 个 UTF-8 字节。每行一个文档摘录，`text` 是可用于编解码器自由重建的输入和目标。保留原始空格、换行、大小写和标点；尚未按 tokenizer 分块，也没有词边界标签。它适合核对数据读取、切块与逐字还原，大规模训练和质量评估需要更多文档、领域与长度。

同一词条的译本共用 `family_id` 与 `split`。训练集有像素、视口、纵横比、带宽四个家族（8 条），开发集为延迟（2 条），测试集为有损与无损压缩（4 条）。只对 `split=train` 拟合。这里按词条分组，基于同一主题收录的中英文摘录没有逐句对齐标签，开发与测试的词条数也不足以估算泛化性能。

各行记录标题、文章与固定版本的原文 URL、源文件和摘录的 SHA-256，以及从完整 Markdown 首字节起算的半开 UTF-8 区间 `source_byte`。署名 Mozilla Contributors。所选正文依 [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/) 提供；仅取连续区间，没有改写。英文与中文均沿用[此前锁定的版本](../word_boundary_v0_2/sources.lock.json)及其[英文仓库许可](../word_boundary_v0_2/sources/en-us/LICENSE.md)、[中文仓库许可](../word_boundary_v0_2/sources/zh-cn/LICENSE.md)。转载时保留每条的标题、来源、署名、许可及摘录说明。[MDN 复用说明](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license)

在仓库根目录使用 Python 3.8 或以上运行下列命令。第一条离线检查数据与历史输入，第二条从固定地址取回 14 份源文件重建并逐条比较现有发布物。若发布物不存在，第二条会生成它。

```sh
python -B -m training.coarse_codec_corpus_v0.prepare
python -B -m training.coarse_codec_corpus_v0.prepare --upstream
```

本机已用 Python 3.12.10 复核。若 Windows 的 `python` 指向旧版 Anaconda，PowerShell 可运行 `& "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe" -B -m training.coarse_codec_corpus_v0.prepare`。

查重范围包括旧词边界管线的 29 条输入、4 条隔离审核候选、100 条本地标注试跑输入，以及本包不同 split 的摘录。对照家族标识、NFKC/大小写/空白归一后的全文、至少 24 个归一化字符的整条包含、字符 5-gram Jaccard ≥ 0.85，未发现命中。归一化仅用于查重。这项审计对未知释义和跨语种改写没有自动识别能力。
