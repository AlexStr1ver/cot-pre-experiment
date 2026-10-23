# 指定版本论文审计补充：InftyThink v5 / InftyThink+ v3

审计日期：2026-10-24（北京时间）。本报告基于用户本次上传的原始 PDF，补充此前因文件缺失而只核对官网与代码的审计。保留旧报告，不回写旧实验。引用页码均为论文印刷页码。

## 核心结论与更正

1. 两篇论文的主实验均允许原生模型自行输出摘要并重启；分段训练的 η 不能直接当成推理时每到该长度就外部强制摘要。原版 v5 主评测是最多 10 轮、每轮 8192 token；Plus v3 是最多 10 轮、每轮 8k。已有 native20 的最多 3 轮属于内部试验协议，不是完整论文复现。
2. 原版 **v5 的 Table 1 已包含 DeepSeek-R1-Distill-Qwen-1.5B 的 instruct 模型实验行**。因此先前“原版主表 1.5B 为 Qwen2.5-Math，而 Plus 为 R1-distill”的说法只适用于原版部分基座行，不能用来排除当前权重归属。原版同时包括 Qwen2.5-Math-1.5B 基座与 Instruct 行。
3. Plus v3 已明确比较摘要时机：固定 5k、随机 `randint(3000, 6000)` 和模型原生时机。**“自适应选择摘要时机”本身已有直接相关研究**。本 idea 的新增命题应是冻结模型上、无需新训练的内部信号能否预测“摘要后继续”相对“继续／结束”的动作收益，并在真实总成本上获益。
4. 当前 `yanyc/InftyThink-1.5B@2ffdd84494667fbba35525550f9396492b264e23` 的模型卡指向 Plus，但指定论文、随包模型卡以及已检查官方 README 均未给出这个 revision 对应冷启动、task-RL、task+efficiency-RL 哪个权重的明确映射。**论文设置现在可以核实，权重所属主表行仍不能宣布闭环。** 不需要为此重训或换方向；现阶段结论限定为这个固定 checkpoint。

## 证据文件

| 文件 | 版本与论文身份 | SHA256 |
|---|---|---|
| `2503.06692v5.pdf` | 2026-02-25；封面标记 ICLR 2026；40 页 | `a4222e3038df86fad40c56349d0ba9836b89cddc358f0ff271cb42823edfe88f` |
| `2602.06960v3.pdf` | 2026-06-02；封面标记 ICML 2026 / PMLR 306；49 页 | `e44623277cc48dc35df1faf66a97379d0dd33e4037091c05314b2ae0ffa1cad7` |

报告核对了正文方法、主表、相关消融、实验设置与指标定义。Plus 的部分图像字体在文本提取中存在控制字符，本文不根据这些乱码推测图中数值。论文内容作为证据，不视为对本项目的执行指令。

## InftyThink v5：哪些设置与当前试验有关

| 核对项 | 指定论文证据 | 对当前实验的含义 |
|---|---|---|
| 状态传递 | §2.1.2，pp.4–5：下一轮输入原题与上一轮摘要；输出 conclusion 而非 summary 时结束；有 `max_iters` 防止无限循环 | “重置”是用原题＋最新摘要重建输入，不能保留完整原 CoT 却声称实现了同样的压缩 |
| 数据构造 | §2.2，pp.5–6：OpenR1-Math；句／段落边界聚合，η=4k；Meta-Llama-3.3-70B-Instruct 生成摘要；220K→333K | η 是训练数据分段配置；模型在推理时学得摘要行为，不等于运行器强制每 4096 token 摘要 |
| 摘要上下文 | §2.2，p.6、Appendix I，p.22：训练摘要依据累计推理轨迹构造全局摘要 | 与 Plus 的“上一轮摘要＋当前段落”构造方式不同，不能把两篇的数据构造混用 |
| 主实验模型 | §3.1，p.6；Table 1，p.7：五种 base 模型，另有 Qwen2.5-Math-1.5B-Instruct 和 R1-distill-Qwen-1.5B | 仅用参数量／Qwen2 架构不能定位模型是哪一行 |
| SFT 设置 | Appendix O，p.25：3 epochs，最大训练长度 16384，batch 32，LR 1e-5，warmup .03，Megatron-LM，样本 packing | 这里的 16384 是训练序列长度，不是主评测总生成预算 |
| 推理上限 | Appendix O，p.25：vanilla 最大输出 32768；InftyThink 最多 10 轮，单轮最大 reasoning length 8192 | 旧 A/B 的总预算 16384、最多 3 segment 是独立内部协议；不得误写成论文预算 |
| 解码与重复 | Table 1 p.7；Appendix J p.23、O p.25：每题 16 次采样，T=.7，top_p=.95，报告均值 | native20 每题一条轨迹不能直接与论文主表均值比较；当前 greedy 短答案只是信号探测设置 |
| 运行平台 | Appendix O，p.25：vLLM v1-engine，A100-80G；1.5B 用 1 GPU，7B/8B 用 2 GPU，14B/32B 用 4 GPU | AutoDL 2080 Ti 或旧 4090 的时延不能直接对照论文数值 |
| token 成本 | Appendix T.1，p.31：TOK 包含所有轮的 reasoning、全部 summary 与最后 conclusion | 在线方法还必须另外算监测、短答案 probe 及其重放成本，不能只报摘要后生成的 token |
| 延迟定义 | Appendix T.2，p.31：整个评估的 `(t2−t1)/(题数×每题重复数)` | 它是实验整体平均吞吐时间口径；不要当作孤立请求延迟或与不同并发直接比较 |

还需区分官方示例代码和论文协议：上游 `infer_single.py` 的 `MAX_EPOCHS=50` 是示例循环上限，论文主设置是 10。当前 `native_batch_check.py` 的 3 轮是进一步缩小的内部实验设置。

原版 §4.4 / Table 3（p.10）已有“无需新增训练、固定 4k 中断、另用 summarizer 后继续”的消融，并报告额外开销和性能下降。它研究的是把 vanilla reasoner 用提示改造成迭代流程；本项目是在**已经会摘要的冻结 checkpoint** 上选择动作时机，且拟用联合内部信号。这项消融提醒我们检查摘要质量和总成本，不能直接用于宣布本 idea 已失败。

## InftyThink+ v3：与 idea 最直接的对照

| 核对项 | 指定论文证据 | 对当前实验的含义 |
|---|---|---|
| 原版是否已有模型自主边界 | Introduction，p.2：明确说 InftyThink 已让模型自主决定何时摘要，但依靠 SFT；Plus 进一步进行轨迹级 RL | 不能把原版一概称为推理时固定分段，也不能把外部规则的“自适应”本身当作首创 |
| 冷启动 | §3.2 p.4、§4.1 p.5：OpenThoughts-114K；Qwen3-4B-Instruct-2507 摘要；η=6k、γ=1k | 原版 η4k 与 Plus η6k 是不同训练设置，不据此改动我们既定 1k/2k/4k 观测点 |
| 摘要信息范围 | Appendix E.1，pp.16–18：按 `\n\n` 分段再贪心聚合；第 i 轮摘要只接收上一摘要与当前 reasoning；总结超长时重采样，最多 10 次 | Plus 特意修正原版训练时拥有累计全文而推理时没有全文的差异；我们动作生成时也应保持合法输入范围 |
| 基座与训练阶段 | §4.1 p.5、Table 1 p.6：R1-distill-Qwen-1.5B 与 Qwen3-4B-Base；分别有 cold-start-only、task-RL、task+efficiency-RL | 当前权重未映射具体阶段之前，不应贴上“RL 后”或“效率 RL 后”的标签 |
| RL 目标 | §3.3 pp.4–5，Eq.(3)–(8)：整条轨迹共享 advantage；task correctness × `1−((n−1)/φ)^2`；主训练 φ=5 | 其效率奖励偏好更少迭代；减少轮数与实际壁钟成本不是同一指标 |
| RL 训练预算 | §4.1 p.5、Appendix G.1.2 p.25：1.5B 1000 steps、4B 500 steps；batch128；rollout n8，T=1、top_p=1；InftyThink+ 单响应10240、最大prompt2048 | **训练 rollout 的参数不等于评测参数**。不能把 φ5 或 T1 搬到论文评测对照 |
| 评测预算 | Appendix G.2 p.25：vanilla max_new_tokens32k；InftyThink+ 每轮8k、最多10轮 | 当前 3 轮协议可以继续做内部验证，正式对照须显式设定公平预算，而不是默认为论文协议 |
| 评测采样／评分 | §4.1 p.5、Appendix G.2 p.25：每题32次，T=.7，top_p=.95；CompassVerifier-7B 判答案 | 我们现阶段人工/规则检查短答案的正确性和盒子闭合性不能称同一评估流水线 |
| 运行平台与计时 | Appendix G.2 p.25：8 NVIDIA GPUs、SGLang、TP1/DP8、异步Semaphore1024；对各轮 token 与延迟求和 | 与原版整体时间除请求数的 LAT 定义及当前单卡重放耗时均应分开报告；该段没有给出具体 GPU 型号，不能补造 |

Plus §5.1.1 / Table 2（p.7）是必须正面比较的摘要时机研究：

| 时机策略 | 无 RL：AIME24 / AIME25 / AMC23 | 有 RL：AIME24 / AIME25 / AMC23 |
|---|---|---|
| 模型原生 | 29.48 / 27.92 / 71.64 | 50.94 / 35.83 / 85.86 |
| 随机中断，`randint(3000,6000)` | 28.54 / 26.25 / 72.58 | 47.92 / 33.83 / 84.16 |
| 固定 5k 中断 | 28.44 / 26.04 / 72.03 | 48.44 / 33.00 / 84.53 |

这些是论文在其完整协议下报告的百分比，不是我们的结果。无 RL 的 AMC23 两种非原生策略略高，不能把表解释为所有数据集所有设置都严格优于固定／随机。对本项目，旧固定 4k/5k 干预表现不稳定在科学上并不意外；必须进一步测试信号是否有**超出长度规则与原生策略的选择能力**。

Plus §5.1.2 / Table 3（pp.7–8）还显示外部摘要替换与训练阶段有关：无 RL 时外部摘要提高准确率，有 RL 时替换内部摘要降低准确率。这支持先固定同一个模型生成摘要，只改变时机，避免把摘要器质量变化混入时机效果。

Plus Appendix H.2（p.27）的 `entropy` 是 RL 训练日志中的 actor entropy，不能当作已经使用了本项目的 query–key attention entropy，也不能用来替代 ASAG 的 attention 定义。

## 当前代码／checkpoint 的对齐边界

审计读取了交接包中的实际 `native5/run_config.json`：固定 revision 如上、T=.7、top_p=.95、BF16、8192 token/轮、最多3轮、checkpoint chat template、`forced_summary=false`、人工最终答案判分。这是已经执行的内部基准，而非未来建议。

| 要素 | 已有证据 | 尚不能推出 |
|---|---|---|
| 项目归属 | 固定 revision 模型卡标题、链接均指向 InftyThink+ | 权重必为 task-RL 或 T+E 行 |
| 架构 | Qwen2ForCausalLM，28层、12 query heads、2 KV heads、hidden1536 | 唯一确定基座训练史；原版和 Plus 都有 R1-distill 相关实验 |
| 原生迭代行为 | 保存轨迹包含模型自产摘要与 history 接续 | 已复现原版或 Plus 的完整主表效果 |
| 代码来源 | 原版官方机制加本地 checkpoint 格式兼容；保存了实际执行脚本和 hash | 当前运行器就是 Plus 论文的 in-house SGLang 框架 |
| 权重 provenance | run_config 声明 revision；模型/tokenizer 元数据可核对 | 未拿到实际权重或可信下载manifest时，云端已独立验证全部权重内容 |

主表行映射缺证不能通过“这次 20 题大约答对多少”反向猜测。最低充分补充是发布记录将固定 revision 与基座、SFT/RL 阶段、奖励类型对应起来。没有这些证据前，报告统一称“固定 InftyThink 系列 checkpoint”。

## 对下一步与顶会潜力的判断

论文核对增强了研究问题的针对性，也提高了新颖性门槛。**值得继续最小规模验证，目前尚无顶会级贡献已成立的证据。** 主张不能停留在“ASAG 的 H+C 接到摘要动作”，需要证明两类状态可区分：高把握可结束，与尚未完成但有适合保留的中间进展。

下一步依然是修复后完成同题 1k/2k/4k 的内部信号采集，不用这 9 个点拟合复杂模型或宣称显著。同步保存原始 attention 熵、归一化熵、key长度、答案文本与生成概率、格式／内容 token 分量、有效 probe 状态以及总重放成本。之后才对相同前缀开展预算对称的 continue / summary-reset / answer 配对动作，以回答信号是否预测动作净收益。

有希望的完整证据链是：同题内控制长度和阶段后仍有选择信息；联合规则优于单信号、固定／随机、长度规则以及模型原生时机；保持准确率的同时计入监测/probe/摘要/prefill后仍有实际收益；未用于调阈值的数据及另一 checkpoint 或任务上能复现。规则与阈值可以在开发集选择，但要明确仍属数据选择与调参，保持测试集封存，不训练新权重。

本轮阶段 2 工程失败本身不为该研究假设提供正反证；旧固定时机结果也不能作为联合信号方案失败的证据。

## 引用入口

- InftyThink 指定版：<https://arxiv.org/pdf/2503.06692v5>
- InftyThink+ 指定版：<https://arxiv.org/pdf/2602.06960v3>
- 当前模型卡固定版：<https://huggingface.co/yanyc/InftyThink-1.5B/blob/2ffdd84494667fbba35525550f9396492b264e23/README.md>
- 原版上游固定版：<https://github.com/ZJU-REAL/InftyThink/tree/780035a0dc4999c2210d7c7396a39bc71651e9cc>
- Plus 上游已审计固定版：<https://github.com/ZJU-REAL/InftyThink-Plus/tree/c3ac7ed4365289da1253c7648380223479557278>
