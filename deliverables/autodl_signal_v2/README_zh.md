# 第二轮：同题多位置、长前缀与ASAG式信号定义

## 本次进度

第一轮返回包 `infty_signal_first_ia1mUMM1_return.zip` 已核验。真实checkpoint上4/4短前缀成功，分块/完整eager参考的H与logits一致；GPU环境、6项旧自测、7项新数值/逻辑测试通过。原始脚本、case、prefix与模型config哈希匹配。详细分析见 `analysis_first.md` 与 `analysis_first.json`。

当前已达到：能在真实模型上取得所定义的attention与置信度信号。尚未验证：摘要动作的条件性收益、联合信号的选择能力、在线总成本及泛化。短前缀4个答案均错，只反映这些早期探测，不能判定联合方案失败。

本次观察到题6 raw entropy sum上升而归一化H下降，且题1的闭合格式token概率显著抬高C。因此第二轮保留原始/归一化H，并明确记录内容/格式token范围。

GPU报告是RTX2080Ti、21.49GiB，和历史RTX4090不同。继续使用当前可用环境；新版记录硬件capability及BF16原生/模拟支持，不把时延与旧卡直接计算加速比。

## 本包这次做什么

1. 从原native20的保存轨迹，找1024/2048/4096目标后128 token范围内的首个自然段落边界，要求尚未结束思考。
2. 先取在三个目标都有合法位置的题目交集，再按题号选择前三题。题目选择不读答案对错，固定同一题集避免不同长度采用不同题目。
3. 先跑第一个题目的最长前缀，即题1/cut4103。比较chunk128与chunk64的H和logits，记录峰值显存、耗时与工程容差。
4. canary通过后完成其余8点，canary不作为新样本重复测量。每点采集两个H定义和一次ASAG式短答案probe。计算ΔH时按本题实际cut排序，不能把首先执行的4k canary当作基准。
5. 所有位置只做离线测量，没有摘要/重置动作、自动早停控制或阈值校准。保留旧A/B/C结果，后续按精确prefix哈希与4k位置对齐。

默认共同候选为 `[1,6,8,10,11,12,15,19]`，按题号取前三题：

| 题号 | 约1k | 约2k | 约4k |
|---:|---:|---:|---:|
| 1 | 1041 | 2104 | 4103 |
| 6 | 1055 | 2092 | 4203 |
| 8 | 1084 | 2055 | 4173 |

这是3道开发题×3位置共9点，不是9道独立题。该交集偏向会推理至4k的题，不能代表完整MATH-500。已有题目已被探索过，本轮不做显著性/泛化结论。

## 信号定义与ASAG的关系

| 列 | 定义 | 用途/限制 |
|---|---|---|
| raw_tail32 | 原推理前缀最后32个query，最后4层×全部12个query heads | 与v1兼容的诊断；不追加probe指令 |
| asag_style_probe_window | 原前缀后显式追加原文 `\n\n Final Answer\n\n \\boxed`，取最近m个思考token与m个指令token；本地m=6，因此q=12 | 按ASAG窗口与聚合式测量；两种窗口的q不同，原始sum不可直接横比 |
| raw entropy | 所选层、头、query的 `-sum p log p`，并保存平均值所需的q/k | 检查长度归一化导致的变化方向 |
| normalized H | raw entropy除以log(当前全局key数)，另保存query/头/层均值 | 保持因果mask，未来key权重为0；不是词表熵 |
| ΔH | 相对本题最早选中位置的变化，同时保存raw和normalized两种 | 此处是计划段落边界，不是ASAG第一个Wait ATP；初始化0不是证据 |
| 主C | 使用上述ASAG指令，贪心生成至第一个闭合非空答案盒，取全部实际生成token概率的算术均值 | 包含生成的开/闭合格式token；与v1的提示和token范围不同，不能当同一协议比较 |
| 内容C诊断 | 仅使用解码字符区间完全落在答案内容中的token，记录覆盖率、混合格式token及无法判定情况 | 辅助分析格式影响；不是论文官方C，也不保证语义正确 |

probe上限128；未闭合、空答案盒的主C为null，有明确失败状态。逐token ID、概率和解码范围均保留。三次独立prefill（两个H及C）的成本全部记录，canary额外重复回放单列。当前eager离线实现服务于数值与信息验证，不能由此宣称在线加速。

本轮仍标记ASAG-style：探测点为计划自然边界，没有完整实现Wait ATP、首ATP基准、logits/probability mixture和jump机制，也没有解决论文所有停止范围歧义。暂不套用0.95/-0.1作为摘要条件。InftyThink v5/Plus v3指定PDF与checkpoint训练阶段映射仍待补齐。

## 上传与运行

原AutoDL目录和虚拟环境已通过首轮检查。**只上传本包 `autodl_signal_v2.zip` 到 `/root/autodl-tmp`。** 不用传回旧结果，不用重装环境，也不用重新下载模型。

在AutoDL终端执行：

```bash
set -euo pipefail
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_signal_v2_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_signal_v2.zip -d "$UPLOAD_DIR"
bash "$UPLOAD_DIR/autodl_signal_v2/run_phase2.sh"
```

脚本打印 `OUTPUT_DIRECTORY=...` 和结束时的 `RETURN_ZIP=...`。它先核对包内校验和，运行CPU测试、生成不可覆盖的plan.json，再加载一次模型测9点。新输出全部写进独立目录，旧模型/代码/数据/结果只读。

只运行默认三题即可。脚本支持第二个参数增至8题，但本轮不建议扩展；先看3题×3位置的返回结果再决定。

## 工程门限与失败处理

4k canary使用两个有限chunk实现比较，避免导出长序列所有层的完整方阵。默认H相对差容差1%、平均logit绝对差0.05；这些是明确标记的工程一致性门限，不是方法的H/C触发阈值，不能解释为误差上界或准确率保证。

脚本另检查canary用时不超过120秒、峰值allocated显存不超过整卡85%；用时包含额外chunk检查，排除模型加载。这是完成该点后判断是否继续，不是运行中强制超时。参数及检查结果写入run_config和canary记录。超过门限、非有限值或OOM均停止本轮并保留诊断，不静默换精度/预算。BF16模拟支持不会仅因显卡型号而被拒绝。

如果失败，把同样的RETURN_ZIP传回；不要反复重跑、修改旧结果或升级依赖。工程门限调整必须根据返回日志另建新run；本包拒绝覆盖已有plan或输出目录。

## 交回什么与验收

把脚本打印的 `RETURN_ZIP` 对应ZIP上传本聊天。它应包含：

- plan.json：相同题集、实际cut与来源哈希；
- signals/run_config.json、summary.json、9个逐点文件；
- canary对比结果、峰值显存、每部分耗时、硬件信息；
- 逐token置信度及范围、raw/normalized H、按时间排序的ΔH；
- 原样执行代码副本、测试日志、exit_status.txt。

成功应满足exit_status=0、summary.completed=true、planned_positions=executed_positions=9、canary_passed=true。C未闭合允许记录为null：测量完成不等于答对或C可用，更不等于策略有效。

返回后会先分析：长前缀数值/成本是否可靠；窗口定义改变是否影响趋势；C中格式token贡献；同题多个位置的H/ΔH/C与已有4k配对动作结果是否值得进一步验证。三个题的相关性仅用于提出下一步假设，不能调完阈值再宣称测试有效。若信号/动作有迹象，才补公平预算的小规模配对，并保留直接回答作为替代动作。

## 当前研究进度

| 阶段 | 状态 |
|---|---|
| 已上传代码、实际执行配置及可读论文来源审计 | 已完成可获得材料；指定两篇PDF及权重训练阶段映射待补 |
| 真实checkpoint短前缀数值验证 | 本次返回包已验收，4/4完成 |
| 长前缀、相同题集、ASAG式窗口与成本测量 | 本包准备；真实GPU结果尚待运行 |
| 摘要动作收益、联合信号的增量 | 尚未验证 |
| 在线端到端效率和独立题目/模型泛化 | 尚未验证 |

初步研究判断不变：值得小规模推进，但顶会潜力依赖后续动作收益、强基线、公平成本和泛化证据。信号可测是必要工程进展，不等于方法已成立。
