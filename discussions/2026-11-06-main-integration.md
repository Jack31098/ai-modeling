# 2026-11-06：训练数据、H-Net ROCm 与本地 teacher 工作链合入主干

## 范围与依据

用户恢复 MZHOU 的 D 盘后授权将已有 AI modeling 工作合入远端 main。恢复后的原仓库是 `D:\ai-modeling`，当前分支为 `codex/v1.2-local-teacher-pilot`。检查了原仓库和研究 worktree，两者均无未提交或未跟踪文件；stash 为空。所有相关本地分支与 fetch 后的远端分支一致，没有发现额外的本地代码提交。模型权重和虚拟环境仍由原有忽略规则排除，没有搬移或删除。

集成前 main 为 `ce54d66a7be7337db8d19fd7a95fa805683d06bf`，它是以下五个提交的祖先；这是一条递增工作链，四条功能分支不应重复计数：

1. `87d01ba4a7acf1867c26c274113ed7ac0d87a934`：固定词边界数据管线和第零步样例。
2. `0073a2df91afd5895fd0aa52bb798a0873ab692e`：词边界夹具、有界流式参考和独立审核候选。
3. `250cbb10896c359ec843346bbc35918ad796a245`：H-Net 完整 ROCm 参考前后向及一步优化验收。
4. `17272cdff730fd7a6de8b51eb74bdeaba8b1bcbc`：v1.2 teacher 候选和 MI300 生产／本地测试分工。
5. `620f1789b9334f97a82c1bdf9513fe881a95bac5`：三模型本地词／词组标注 pilot、原始输出和质量边界。

研究分支 `research/v1.2-weight-space-2026-11-03` 的 `1ff300b` 已由旧 main 包含。本轮只集成上述工作链及本次记录和索引，使用独立 worktree 准备保留原提交的 merge commit，没有修改原工作区或切换其分支。合并没有冲突，`git diff --check` 通过。

## 本轮验证

- `training.word_boundary_v0_2.verify`：46 项基线／增量测试通过；数据包、实现、配置和旧基线的 hash 检查均无错误。五组压力输入共 1,120,000 个字符，原文重建通过；最大暂存 64 code points／244 UTF-8 bytes，最大提交等待 63 次到达。
- pilot 的 `test_validate` 和 `test_model_evidence`：8 项测试通过。
- 对 pilot 原 manifest 的 852 个文件逐一校验字节数和 SHA-256；在内存中重跑现有审计和统计，300 份记录、499 次尝试全部通过，重算指标与归档一致。原审计文件和 manifest 未重写。
- 原 v0 数据包的 90 个文件及实现／输入指纹通过校验。
- `experiments.hnet_rocm.audit --verify experiments/hnet_rocm/manifest-2026-11-03.json`：11 组历史验收证据、26 个归档文件通过审计，无错误。

验证使用现有 Python 虚拟环境。数据管线和 pilot 检查在独立集成 worktree 运行；H-Net 归档审计在原仓库运行，以读取原有受忽略的上游源码。没有重新生成 teacher 输出、运行新的 GPU 前后向或启动 MI300 任务，因此这次集成不产生新的推理性能或语义质量结论。

## 接续与边界

本次主干集成保存已有实验和失败证据，不改变其原验收范围：人工金标仍 pending，pilot 输出未获训练准入，H-Net fused／缓存及完整 v1.2 学生训练仍未验证。后续从 main 读取项目 README、讨论索引和各实验 README；不得把已经 push 的功能分支误称为此前已合入主干。远端主干的最终提交和包含关系由集成后的 Git refs／祖先检查确认。
