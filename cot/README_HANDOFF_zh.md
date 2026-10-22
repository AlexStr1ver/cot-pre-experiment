# InftyThink 免训练摘要时机预实验：Codex 交接说明

整理日期：2026-10-21。本文根据本次会话中实际可读取的上传文件与代码快照整理；未访问用户的 AutoDL 当前磁盘，未在本机重新运行 GPU 推理。

## 0. 给接手 Codex 的任务

请继续“基于 attention 熵变化与答案置信度，免训练地选择 InftyThink 式摘要／重置时机”的原始方向。先阅读本说明、原始结果包中的 run_config.json 和随结果保存的实际执行脚本，再核对当前 checkpoint 与论文实验设置。此前只做了固定位置的 A/B/C 分支试验，尚未采集 attention 熵，尚未实现联合信号控制器，不能把这些结果解释为该控制器已经失败。

用户偏好：免训练、单卡可做、算力少、利用模型内部信号、与推理增强／加速相关。不要擅自改为训练隐藏状态探针、跨层判别器或另一个项目。用户尚未同意转方向。不要把继续建议当成已完成实验。

本文优先于旧分析包里的方向性建议。旧报告的原始数字保留，但其中“暂停摘要时机研究、改做跨层信号”等建议属于此前过早外推，不作为本次交接结论。

## 1. 我们真正要验证的 idea

在已经训练好的、支持分段推理和自生成摘要的 InftyThink 系列模型上，不更新模型权重。利用推理过程中的 attention 分布熵及其变化，加上短答案生成置信度等低成本信号，判断当前有效进展是否适合被摘要，并由模型生成摘要后重置上下文、继续推理。

目标：相对模型原生的摘要时机或固定长度摘要，在保持最终答案准确率的同时，降低实际推理成本；也可能通过减少后续错误／冗余推理改善准确率。这些收益目前都是待验证假设。

三个问题需要分别回答：

1. **动作是否有价值**：是否存在“此处摘要／重置”优于“继续推理”的前缀位置？收益可能是正确率、成本或二者的权衡。
2. **信号能否选位置**：attention 熵及其变化、答案置信度的联合规则，能否优于固定长度、随机位置、单一信号或原生策略？需要控制前缀长度与推理阶段。
3. **总成本是否划算**：计入监测、短答案试探、摘要生成、重置后的 prefill，再看端到端耗时、显存和准确率。

用户还提出过可选判断器：如果答案已经可靠，可以直接结束；如果仅有可保留的阶段进展，可以摘要后继续。**答案可提前输出**与**阶段进展适合摘要**是两个标签，不应混同。当前没有自动可靠性保证。

模型原本经过训练，不影响“新提出的控制方法不再训练”这一约束。现阶段也没有“顶会级创新已成立”的证据；尤其需要与 ASAG、InftyThink+ 对齐比较，明确新增的是摘要动作选择及其效果，而非简单拼接名称。

## 2. 参考文章与代码

### 当前最直接的三篇

| 文章 | 在本项目中的作用 | 原文／代码 |
|---|---|---|
| InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models | 分段推理 → 模型生成摘要 → 用摘要重建输入继续推理的基础框架。2025 年初版，ICLR 2026。 | https://arxiv.org/abs/2503.06692 ；https://github.com/ZJU-REAL/InftyThink |
| InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning | 学习摘要边界、摘要内容和继续策略的直接相关工作。2026 年文章，ICML 2026。核对当前权重归属时必须读。我们的新增方法拟免训练，不是重跑其 RL 训练。 | https://arxiv.org/abs/2602.06960 ；https://github.com/ZJU-REAL/InftyThink-Plus |
| Stop When Further Reasoning Won't Help: Attention-State Adaptive Generation in Reasoning Models（ASAG） | 使用 attention 状态与答案置信度进行免训练自适应生成，是用户提出联合信号的直接启发。2026 年文章，arXiv 标注 ICML 2026 Spotlight。 | https://arxiv.org/abs/2606.15070 ；https://arxiv.org/html/2606.15070v1 |

ASAG 的 attention 熵不能偷换成词表输出分布的熵。需按原文核对答案试探、最后若干层的聚合、熵变化窗口和触发条件。此前讨论过置信度阈值 0.95、熵变化阈值 -0.1 等论文示例，但未证明能直接迁移到当前模型／摘要动作，不能据此假定阈值已经确定。

### 早期启发与相关工作

| 文章／资料 | 作用与区别 | 链接 |
|---|---|---|
| Limited Reasoning Space: The cage of long-horizon reasoning in LLMs（Halo） | 早期关于长推理退化、熵驱动控制与重置的启发。未作为当前已实现 baseline。 | https://arxiv.org/abs/2602.19281 |
| Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks | 用户反复讨论过的 latent-CoT 机制论文。当前实验的文字摘要／重置不等同于 latent 步计算研究。 | https://arxiv.org/abs/2602.00449 ；https://github.com/jialiang19/latent-cot-thinking |
| When Chain-of-Thought Fails, the Solution Hides in the Hidden States | 早期隐藏状态可恢复信息的启发；并非当前控制器已经实现的技术。 | https://arxiv.org/abs/2604.23351 ；https://github.com/hmehrafarin/when-cot-fails |
| Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure | 早期 latent 机制相关工作。 | https://arxiv.org/abs/2602.08783 |
| Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification | 训练隐藏状态正确性探针的相关工作，不能据此断言免训练“attention 熵＋答案置信度 → 摘要时机”已被完整覆盖。 | https://arxiv.org/abs/2504.05419 |
| Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits（Gnosis） | 使用内部表征训练预测模块的相关工作。不能简单说它“完全不涉及熵／置信度”，也不能直接等同于本项目。 | https://arxiv.org/abs/2512.20578 |
| Awesome-Efficient-Reasoning | 用户用于筛选高效推理方向的论文清单。 | https://github.com/hemingkx/Awesome-Efficient-Reasoning |

早期会话还提过简称 “Learning When to Stop”，没有足够材料唯一确定是哪篇，本文不补造论文编号。更早的隐藏状态方案和图片作为历史材料保留，不能替代当前原始 idea。

## 3. AutoDL 环境、模型与数据

以下是已有运行记录，不保证 AutoDL 当前环境未被改动；打包时会另采集版本信息。

| 项目 | 已知值 |
|---|---|
| 主目录 | `/root/autodl-tmp` |
| 虚拟环境 | `/root/autodl-tmp/venvs/infty-official` |
| Python | 3.10；当前实验解释器记录为该环境的 `bin/python` |
| PyTorch / CUDA 构建 | `2.6.0+cu124` / `12.4` |
| Transformers / vLLM | `4.51.3` / `0.8.5` |
| 当前结果记录 GPU | NVIDIA GeForce RTX 4090，24 GB；早期调试曾使用 4090 D |
| 模型仓库 | `yanyc/InftyThink-1.5B` |
| 模型目录 | `/root/autodl-tmp/inftythink_day1/models/InftyThink-1.5B` |
| 模型 revision | `2ffdd84494667fbba35525550f9396492b264e23` |
| 数据集 | `HuggingFaceH4/MATH-500` 的 `test.jsonl` |
| 数据 revision | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| 数据清单 | `/root/autodl-tmp/inftythink_day1/manifest.json` |
| 已跑的 20 题 | `/root/autodl-tmp/inftythink_day1/data/day1_20.jsonl` |

数据划分由 `random.Random(2026)` 打乱 500 个索引：前 80 为 `dev80.jsonl`，接着 120 为 `test120.jsonl`，剩余 300 为 `reserve300.jsonl`。`day1_20.jsonl` 是 dev80 的前 20 题。已探索的 20 题不可再当作未看过的测试集。完整划分和模型配置需由 AutoDL 打包补齐；上传结果中含本轮题目、参考答案与执行轨迹。

**重要待核对项**：模型名 `InftyThink-1.5B` 不能直接证明它对应原版 InftyThink 论文的某行结果。该仓库模型卡标题指向 InftyThink+；已检查的固定 revision README 也如此。需要进一步核对权重训练阶段、1.5B 对应的论文表格、提示格式及生成预算。当前没有完成这项对应关系的闭环核验，不能宣称已经复现某篇论文的完整性能。

模型卡：https://huggingface.co/yanyc/InftyThink-1.5B 。固定 revision 模型卡 URL 写入 `references/links.md`。

## 4. 我们实际用了什么代码

不是把官方完整评测流水线原样执行了一遍。当前代码经历了以下阶段：

1. `inftythink_day1/day1.py`：助手编写的 Transformers／SDPA 接口调试程序，验证下载、格式、自然摘要、强制摘要等。早期单题结果属于这一阶段。
2. `inftythink_official_check`：固定了官方仓库提交，保留上游示例，对本地模型与 checkpoint 格式做兼容检查。
3. `native_batch_check.py` ＋实际保存的 `reference_runner.py`：基于官方“模型自生摘要、摘要后重启”机制编写批量记录器，得到 native20。
4. `infty_branch_pilot/branch_pilot.py`：固定 4096 附近的贪心 A/B/C 分支。
5. `infty_sampling_check/branch_sampling_check.py`：将 A/B 改成 temperature=0.7、top_p=0.95，并增加显式请求 seed，得到后续三 seed 和 5120 结果。

最重要的脚本以**原始结果 ZIP 内的保存副本**为准，不能只看旧安装包里较早的脚本。采样结果 ZIP 内脚本文件仍叫 `branch_pilot.py`，其内容 hash 对应 AutoDL 的 `branch_sampling_check.py`。

### 已知路径

```text
/root/autodl-tmp/inftythink_day1/
    day1.py
    manifest.json
    data/day1_20.jsonl, dev80.jsonl, test120.jsonl, reserve300.jsonl
    models/InftyThink-1.5B/
    outputs/                         # 早期独立调试结果

/root/autodl-tmp/inftythink_official_check/
    outputs/native5/                  # 名为 native5，最终实际已有 20 题
        native_batch_check.py         # 随运行保存的批处理器
        reference_runner.py           # 随运行保存的推理逻辑
        run_config.json, environment.json, summary.json
        cases/001.json ... 020.json, 对应 txt, review.csv 等
    outputs/branch4096/
    outputs/branch4096_sampled/
    outputs/branch4096_sampled_seed2027/
    outputs/branch4096_sampled_seed2028/
    outputs/branch5120_sampled/

/root/autodl-tmp/infty_branch_pilot/branch_pilot.py
/root/autodl-tmp/infty_sampling_check/branch_sampling_check.py
```

native 批处理器最初的实时入口文件名应由 AutoDL 现有目录核对；本文不猜测。保存的脚本副本足够审计当时的执行逻辑。

### 官方来源与兼容差异

- 官方仓库：`https://github.com/ZJU-REAL/InftyThink`。
- 固定提交：`780035a0dc4999c2210d7c7396a39bc71651e9cc`。
- 上游文件：`inference/infer_single.py`，保留原文件及许可。
- 上游示例使用 `User:` / `Assistant:` 文本提示、temperature=0.7、top_p=0.95、单次最多 8192 token，循环上限 50。它是推理示例，不能自动代表论文完整评测设置。
- 原摘要解析正则要求 `</think>` 与 `<summary>` 中间存在换行；checkpoint 会生成相邻标签，兼容处理将该处改成允许空白 `\s*`。这个变化仅解决解析，不保证推理正确。
- 初始原格式调用出现过重复输出至长度上限。随后改用 checkpoint chat template、BF16、显式 token IDs、编码时 `add_special_tokens=False` 避免重复 BOS，并补 `<think>\n` 前缀。相关日志和 diff 保留。
- native20 调试协议为最多 3 轮，不是原样沿用 50 轮。官方论文另有最多 10 次迭代和 16 次采样取均值的设置，需接手方逐项核对，不能声称本实验与论文预算完全相同。

### 关键 SHA256

| 对象 | SHA256 |
|---|---|
| 官方上游 infer_single.py | `c25ed9ee20ca77915ee90867cfdb4afe571fc12e5a5fcdb810adb840edcb86a4` |
| native20 的 native_batch_check.py | `d0ad25b1029207444748d80956acf85080934dd489a9ecbee58d8051c2c177a6` |
| native20 的 reference_runner.py | `21ee2ffe505d2316db7aab946d7209af663277cb3a66dfbfac908e2feb19ee32` |
| day1_20 数据 | `48f689e8b83fa9624e17890163f1c6fb76b8313471e01e5cb566bacf18d7eb35` |
| 贪心分支脚本 | `a6e23f9fd3e6854a41750286fc540cb2d0cbfb39ee9987f2a693f5b6c182cf7b` |
| 采样分支脚本 | `97edd04a252d26cde768b8cce020ad401f340f34093aaa6e5cdd9a424f8e5313` |
| 分支引用的 baseline config | `b3c9f763e42e89ebae064066fc0c5ba06edce1213ecfd741789447ecb3dded35` |
| 分支引用的 baseline case 集合 | `a4374874c2c071e003bea1d784097fbec220989eb0e34d7d168d6a47ce29a969` |

## 5. 已有实验的实际设计

### native20：模型自己决定摘要

同一模型、同一组 20 个开发题，使用 checkpoint 格式，BF16，temperature=0.7，top_p=0.95，单轮最多 8192，最多 3 轮。没有外部强制摘要。模型生成有效摘要后，输入被重建为原题＋最新摘要的 `<history>...</history>`＋`<think>\n`，再继续生成。

请求随机 seed：`2026 + zero_based_case_index * 1000 + zero_based_round_index`。

### A/B/C：从同一个既有前缀分叉

用 native20 第一轮中实际保存的 token IDs，不重新生成前缀。在目标长度 4096 或 5120 附近，寻找之后 128 token 窗口中的首个段落边界 `\n\n`；位置必须还在推理区，筛选不使用答案正确性。

| 分支 | 操作 | 注意 |
|---|---|---|
| A_continue | 从该前缀继续原生推理／摘要机制 | 仍允许模型随后自然摘要，不是“永不摘要”的 baseline。 |
| B_summary | 外部插入 `\n</think><summary>\n`，让同一模型生成摘要；有效后重置输入继续推理 | 外部控制时机，摘要内容由模型生成，不是人工代写。 |
| C_answer | 插入 `\n</think>\nFinal answer: \\boxed{`，贪心生成短答案，并记录所选 token 的概率 | 是答案试探 baseline；额外解码本身可能产生新的计算，不能证明答案此前已在隐藏状态完整形成。 |

预算与判定：

- A/B 单轮上限 8192；共享前缀计入总生成预算 16384；最多 3 个 segment，**B 的强制摘要 segment 也占一个名额**。
- B 摘要最多生成 1024 token；停止 token 配置中含 151651，并保留标签用于解析。
- C 最多 128 token。第一个括号平衡且闭合的答案盒用于评分，即使后面的解释被截断；成本依然算实际生成的全部 128 token，不能只算答案中的几 token。
- A 的第一个 continuation 上限是 `8192 - cut`。因此可能先碰到单段长度上限，即使总预算还有余量。
- 首轮 4096 贪心 A/B：temperature=0、top_p=1。后续采样 A/B：temperature=0.7、top_p=0.95；C 始终贪心。
- 采样分支 seed：`seed + 1000 * case + zero_based_call_index`。它与 native20 的 seed 规则不同；即使 temperature/top_p 相同，也不能要求 A 重现原始 native 的随机轨迹。
- `record_finished=true` 只表示记录写完，不表示答对或正常结束。`reasoning_token_limit` 等截断输出在当前预算协议下算未成功；不能由此断言模型在更大预算下一定解不出来。

### 内部信号目前收集到了什么

所有分支配置均写明 `attention_entropy_collected: false`。没有收集 hidden states，也没有实现 entropy＋confidence 联合触发策略。

当前只有 C 分支的“所选答案 token 概率均值”，且范围包括闭合答案盒的分隔符。它不是语义正确概率，也未与 ASAG 定义完成逐项对齐。高值可能来自格式 token、常见错误答案或其他因素，不能直接用它证明任何联合方法无效。

### 成本记录的边界

`extra_generated_tokens` 不含分叉前的共享前缀。`model_call_seconds` 不含共有前缀生成、引擎初始化和离线分析。每次 generate 会重新 prefill；分支没有启用 prefix caching。当前统计可用于局部分支诊断，不能直接当作端到端加速比。

若未来在线策略先执行 C 再选择 A/B，必须为 A/B 额外计入 C 的全部解码和 prefill。旧 CSV 中附加的 probe token 列仅补了解码 token，尚未完整补上所有耗时和 attention 监测开销。

## 6. 已有结果

准确率由已有人工最终答案评分汇总，不是脚本自动证明。原始 summary.json 中 `accuracy: null` 属于预期；不要用空白 review.csv 推翻人工标签，也不要将盒子提取率当正确率。接手方可重新复核人工评分。

| 条件 | 题目／重复 | A 继续 | B 摘要后继续 | C 短答案 | A / B 平均额外生成 token |
|---|---:|---:|---:|---:|---:|
| native20，自然摘要 | 20 题 | 最终答案 16/20 | 不适用 | 不适用 | 每题总生成 4943.2 |
| 4096 贪心初试 | 8 题 | 4/8 | 4/8 | 4/8 | 3608.625 / 5828.750 |
| 4096，seed 2026 | 同 8 题 | 4/8 | 4/8 | 4/8 | 4320.500 / 4162.000 |
| 4096，seed 2027 | 同 8 题 | 5/8 | 4/8 | 同一批贪心结果 4/8 | 4339.750 / 3371.500 |
| 4096，seed 2028 | 同 8 题 | 6/8 | 4/8 | 同一批贪心结果 4/8 | 4675.625 / 5342.500 |
| 4096 三 seed 汇总 | 8 题 × 3 | 15/24 | 12/24 | 每次相同，独立题目仍是 4/8 | 4445.292 / 4292.000 |
| 5120，seed 2026 | 6 个符合位置条件的题 | 2/6 | 1/6 | 2/6 | 4154.667 / 6111.833 |

4096 贪心与采样不能合并成同一设置。4096 与 5120 的题目集合不同，不能把两行准确率差当作位置变化的因果效应。两个采样位置合计覆盖 9 个独立题目、14 个题目位置，不是 14 个互不相关题目。

### native20 细节

- 错题为 1、6、11、18；其余最终答案判对。
- 第 16 题答案 2220 正确，但未使用 boxed；若只按 boxed 提取评分会漏判。
- 6 次有效自然摘要／重启，涉及题 6、8、11、18、19，其中第 19 题有两次。
- 总生成 98864 token，模型调用诊断时间合计约 524.52 秒；26 次调用均观察到 EOS。
- 题 4、8、13、19 等存在“最终答案正确但论证有问题”的情况，不能把 16/20 称为完整推理正确率。
- 未运行配对的普通长 CoT 模型 baseline，不能据此测出 InftyThink 相对原模型的论文收益。

### 4096 三 seed 细节

8 个题号：1、6、8、10、11、12、15、19。

实际 cut 分别为 4103、4203、4173、4105、4100、4148、4106、4096。

与同 seed 的 A 配对比较，B 单独答对的运行是 `(2026, 11)`、`(2028, 6)`；A 单独答对的运行有 5 个。A 未完成 1 次，B 未完成 2 次。B 平均额外 token 比 A 少约 3.45%，但正确率更低；若策略每次先付 C 的 128 token，按旧解码 token 口径节约仅约 0.57%，还没有算完整监测／prefill 成本。

题 11 的 seed2026 摘要仍保留错误值 16，后续得到正确答案 8；不能解释为摘要已经完整包含正确答案。题 8 的某次摘要包含 -1，后续却答成 1；摘要后仍可能推理出错。

### 5120 细节

题号／cut：1/5121、6/5207、8/5189、11/5133、18/5134、19/5157。

A 正确题 8、19；B 仅题 19；C 正确题 8、19。B 额外 token 比 A 多约 47.11%。A 题 18 触及首段 8192 上限；B 题 8、11 耗尽当前预算。这些失败须与预算限制一起解释。

### 置信度反例

- 4096 题 11：C 输出 16，参考答案 8，记录均值约 0.98547。
- 5120 题 11：C 仍输出 16，均值约 0.99599。
- 5120 题 1：C 输出 4，参考答案 6，均值约 0.99010。

这些反例表明当前定义下“高短答案置信度”不能独立保证正确；没有验证过 attention 熵是否能提供额外区分能力。

### 当前可以／不可以下的结论

可以：当前兼容框架能够生成自然摘要并执行重置；盲目在 4k／5k 附近强制摘要没有呈现稳定整体优势；当前单一置信度不足以充当正确性保证。

不可以：不能说 InftyThink 论文不成立，不能说已完成严格复现但论文效果虚假，不能说 attention 熵＋置信度已失败，也不能说训练探针论文已经完整覆盖我们的免训练摘要动作选择。

## 7. 本交接包包含哪些已有文件

### 原始实验结果：完整保留，不覆盖重写

| 本包路径 | 内容 |
|---|---|
| `raw_results/native20_results.zip` | 最重要自然摘要基线，目录名 native5 但包含 20 题；配置、脚本、逐题 JSON/TXT、日志。 |
| `raw_results/native5_results.zip` | 较早 5 题结果，是 native20 的重叠子集，不能再算 5 个新样本。 |
| `raw_results/branch4096_results.zip` | 4096 贪心 A/B/C，8 题。 |
| `raw_results/branch4096_sampled_results.zip` | 4096 seed2026 A/B/C，8 题。 |
| `raw_results/branch4096_seedcheck_results.zip` | 4096 seed2027 和 seed2028，仍是上述 8 题。 |
| `raw_results/branch5120_sampled_results.zip` | 5120 seed2026 A/B/C，6 题。 |
| `historical_debug/official_case1.log` | 初始官方示例格式调用的重复输出等诊断。 |
| `historical_debug/checkpoint_case1.log`、`checkpoint_compat.diff` | 后续 checkpoint 格式检查与差异。 |
| `historical_debug/summary.json`、`summary_1.json` | 更早独立 Transformers 调试的单题统计，不与 vLLM 分支结果混合。 |
| `historical_debug/001.txt`、`001_1.txt`、`001_2.txt` | 更早单题完整文本；依次为 normal、force_once 无效摘要回退、force_once 成功版本。原上传名映射见文件清单。 |

早期 `001_2.txt` 对应会话里 force_4096_1：最终提取答案 3（参考 6），8381 token、约 96.04 秒，强制摘要格式成功但最终答案错误。它仅是独立 debug runner 的一条记录，不计入后续 vLLM 对比。

### 代码、人工评分和历史材料

- `code_packages/`：四个已发过的源代码包，包括 day1、official_check、branch_pilot、sampling_check。
- `code_snapshots/`：从**原始结果 ZIP**提取的实际执行脚本和原始配置，以及固定上游文件。看此处和原始 ZIP，优先于旧安装包的入口文件。
- `grading/`：已有 baseline20 人工评分、采样三 seed 和 5120 的评分 JSON；并提供整合的 `native20_scores.csv`、`branch_scores.csv` 与 `results_summary.csv`。
- `prior_analysis/`：三个已有分析 ZIP 完整保留。其历史建议不代表本次重新确认的方向；以本文第 0、1、6、8 节为准。
- `references/`：参考链接、上传的论文 PDF、早期图片与历史方案。重复 PDF 可能只保存一份，映射记录于 `original_upload_inventory.json`。这些历史方案不能被误当成当前已同意的实验目标。
- `SOURCE_MANIFEST.json`、`checksums.sha256`：追踪原始包／脚本与文件完整性。

这份包只包含会话中已经上传或已生成的可访问文件，不含模型权重、AutoDL 上未上传的完整代码目录、完整数据划分或当前运行环境。请另外执行 `PACK_AUTODL.sh` 获取 AutoDL 实际目录快照，再把两个包一起交给 Codex。

## 8. 接手后的工作顺序

1. **先核对，不重装环境、不立即重跑整批。** 解包阅读实际脚本、配置、20 题人工评分，检查模型 revision 和模型卡；核对它对应 InftyThink／InftyThink+ 的哪一阶段，论文的提示、预算、采样方式与当前有何差异。给出明确的对齐表。当前环境已经跑通，避免无故升级依赖。
2. **厘清预算与 baseline。** 保留已有结果；新协议若改变 max_segments、A 首段上限、摘要上限或概率定义，使用新版本、新输出目录，不能混合续跑覆盖旧配置。A 必须明确是原生可摘要策略；如另加永不摘要策略，单独命名。
3. **真正测联合信号的前提。** 按 ASAG 原文定义 attention 熵与答案置信度，先做极小规模一致性与显存／时间开销检查。24 GB 卡上不要默认导出长序列全部层的全量 attention 矩阵；若用选定层、末尾 query、窗口或离线 replay，需要说明采样定义和额外计算，不能声称零成本。
4. **构造动作收益标签。** 在原前缀的多个自然边界，记录触发前可获得的 H、ΔH、C、长度／阶段；配对比较继续、摘要重置和短答案。答案只用于离线评分，不能泄漏到在线规则。不要只在原先已看过的少数成功例子上调整阈值。
5. **先看信息价值，再调简洁规则。** 比较长度／阶段、C、H／ΔH、联合信号；区分“答案已经可直接交付”与“局部进展适合摘要”。阈值只能在开发集校准，测试集固定后评估；按题目划分，避免同一题不同 seed／位置跨集合。
6. **有证据后再扩展。** 联合信号如能识别净收益位置，再做小规模在线控制器与端到端计时，计入全部监测、探测、摘要和 prefill。若没有改善，报告究竟是动作缺乏收益、信号无区分力，还是开销抵消收益，不把不同失败机制混在一起。

下一步的交付应是事实核对与最小可运行验证，不是没有数据支撑的成功／失败宣告，也不是再扩大固定 4k／5k 扫描来代替内部信号试验。

## 9. AutoDL 打包使用方法

把本包里的 `PACK_AUTODL.sh` 放到 AutoDL 任意目录，执行：

```bash
bash PACK_AUTODL.sh /root/autodl-tmp
```

脚本读取四个已知项目目录、根 outputs 和可能存在的官方仓库目录，打包代码、数据、manifest、原始结果、日志、模型配置和 tokenizer。排除模型权重、虚拟环境和缓存，不安装包、不做推理、不改原始结果。环境信息独立保存到新建的 metadata 目录。

成功后会打印 `PACK_OK: /root/autodl-tmp/inftythink_autodl_handoff_时间_PID.tar.gz` 和 SHA256。缺失的可选目录记录在 metadata 的 missing_paths.txt；若四个项目及其他候选目录都不存在则退出，避免产生空交接包。

要发给接手 Codex 的材料：**本交接 ZIP＋AutoDL 生成的 tar.gz**。不必重新传几 GB 的模型权重；原卡继续跑时可沿用原模型目录，换机器则按固定 repo/revision 获取权重。
