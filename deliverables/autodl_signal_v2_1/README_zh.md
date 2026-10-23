# 阶段 2.1：长前缀数值复核与原定九点续跑

2026-10-24（北京时间）。本包修复阶段 2 的数值诊断与运行可观测性。研究方向仍为：冻结 InftyThink 系列模型，以 attention 熵变化＋答案置信度选择摘要／重置时机，不训练新探针或判别器。

## 为什么上一轮停止

返回包 `infty_signal_phase2_ZTj8Ywpr_return.zip` 完整且 CRC 通过；实际执行的六个文件与发布的 v2 一致。退出码 1，错误是程序主动触发 `Long-prefix chunk consistency gate failed`。不是日志已证明的显存不足或 CUDA 崩溃；终端为何关闭尚无独立证据。

在 case 1 / cut 4103 上，将 prefill chunk 从 128 改为 64：

|窗口|H 相对差|logit 平均绝对差|结果|
|---|---:|---:|---|
|raw tail 32|0.269%|0.119981|logit 差超过 0.05，停止|
|ASAG-style 12 queries|0.569%|0.030022|两项通过|

H 门槛是 1%，两种窗口的贪心下一 token 相同；仍不能据此宣布概率一致。C 尚未开始，完整测点 **0/9**。实际 GPU 是 **RTX 4090 D**，支持原生 BF16。长前缀 BF16 对矩阵形状的数值敏感性是候选解释，尚未被 GPU 复核证明。本次工程中止不构成 idea 失败的证据。

## 修复内容与实验边界

- 显式用 **FP32，关闭 TF32** 重跑原计划；保留原 H≤1%、平均 logit 差≤0.05 的数值检查。没有自动放宽门槛、没有静默精度切换。FP32 是数值诊断口径，不能与旧 BF16 数据混算变化，也不是论文 BF16 配置的复现。
- 新增真正答案 probe 的 chunk128/64 对比：生成 token IDs 与状态一致，逐 token 概率最大差≤0.01。此为工程门槛，不是摘要策略阈值。两次均截断／空盒时只有逐 token 检查通过才算数值一致，C 仍为 null，不计有效答案。
- 记录 logit 整体平移、去平移差及概率分布总变差，帮助判断差异是否影响概率；这些诊断不替换旧门槛。
- 仍按最后四层、全部 query heads 测 raw tail32 与 ASAG-style 窗口；保存原始熵、归一化熵、keys、probe 内容与格式概率。原 `v1_core.py` 和选点脚本保持字节不变。
- 逐阶段原子保存 JSON，未缓冲日志；后台启动可承受普通 SSH／浏览器断开。普通程序失败会保留退出码并打包。断电、SIGKILL 无法执行退出钩子，可在任务停止后补打包已有文件。
- 每次写全新目录，旧输出不覆盖。这次没有已完成测点可复用，因此从同一计划重新测九点；本包不宣称跨进程自动断点续算。

固定问题与位置：case 1 → 1041 / 2104 / 4103；case 6 → 1055 / 2092 / 4203；case 8 → 1084 / 2055 / 4173。先运行 case 1 的 4103-token canary，通过 H、C 和资源检查后自动完成其余八点。ΔH 的参考点仍是同题最早选点，不是执行顺序最先跑的 4k，也不是 ASAG 的首次 Wait ATP。

## 上传什么

只需将新文件 **`autodl_signal_v2_1.zip`** 上传到 `/root/autodl-tmp/`。模型、native20 数据和原虚拟环境继续使用已有位置。不需上传三篇 PDF 到 AutoDL；论文审计已在本包中，不安装新依赖、不改旧环境。

下载入口是 GitHub 分支 `codex/autodl-signal-delivery` 的 `deliverables/autodl_signal_v2_1/autodl_signal_v2_1.zip`，打开后点 **Download raw file**。代码另以 `source/` 目录保存，便于逐文件检查。

## 在 AutoDL 执行

在终端逐行执行下列三行；不要 `source` 脚本，不要关闭实例：

```bash
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_signal_v2_1_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_signal_v2_1.zip -d "$UPLOAD_DIR"
bash "$UPLOAD_DIR/autodl_signal_v2_1/launch_phase2.sh"
```

启动器立即打印 `OUTPUT_DIRECTORY`、PID、日志与状态查询的完整命令。**PID 只表示已启动，不代表实验成功。** 复制打印出的 `tail -f .../launcher.log` 查看进度；`Ctrl+C` 只退出查看，不停止后台实验。后台仍依赖 AutoDL 实例保持开机。

也可执行（把路径换成启动器打印的实际目录）：

```bash
RUN_DIR=/root/autodl-tmp/infty_signal_phase2fix_实际后缀
/root/autodl-tmp/venvs/infty-official/bin/python "$UPLOAD_DIR/autodl_signal_v2_1/status_phase2.py" "$RUN_DIR"
```

成功应同时满足 `exit_status=0`、`completed=true`、`executed_positions=9`、`canary_passed=true`。有效闭合答案可能少于 9；盒子闭合不代表答案正确。通常先预留约 10–20 分钟完成上传、测试与观察，实际 GPU 时间以日志为准；资源检查在 canary 完成后进行，120 秒不是强制超时。

运行正常结束或普通 Python 报错都会生成相邻的：

```text
/root/autodl-tmp/infty_signal_phase2fix_XXXXXXXX_return.zip
```

请从 AutoDL 文件浏览器下载并将这个 **return.zip** 回传。失败也回传，勿自行改阈值或运行更大规模。若实例硬中断导致没有 ZIP，确认任务已停止后补打包：

```bash
/root/autodl-tmp/venvs/infty-official/bin/python "$UPLOAD_DIR/autodl_signal_v2_1/status_phase2.py" "$RUN_DIR" --pack
```

辅助工具不会覆盖已有 ZIP；重复打包会产生带时间后缀的新文件。若重新打开终端后 `UPLOAD_DIR` 变量丢失，使用解压包的实际完整路径替换它。

## 云端验证与尚待完成的 GPU 验证

见 `VALIDATION.json`。云端已执行 CPU 数值与运行生命周期测试，并以随机小型 Qwen2、真实九点 token IDs 跑通 plan→collector 的集成流程；该模型仅为测试夹具，不是用户 checkpoint。当前云端没有 CUDA 和真实权重，**未替你运行 AutoDL 的 FP32 模型实验，也不能预先保证它通过 canary**。

若 FP32 通过，这支持“旧失败与数值精度／计算路径有关”的解释，并提供同口径内部信号；仍不能宣称已准确定位某个 CUDA kernel。若继续失败，保留新日志，检查新的 H／C／资源失败项，先不扩大规模。

## 当前进度与后续顺序

|阶段|状态／下一步|需要什么证据|
|---|---|---|
|实际代码、原始结果、论文核对|三篇指定版本已审计；权重主表行归属仍未闭环|固定 revision 的发布记录，不能从分数反猜 SFT/RL 阶段|
|阶段 1：短前缀采集|4/4 完成，四个短答案均错误|只能说明接口可采到信号，不足以判断选时机价值|
|阶段 2：原 BF16 九点|0/9 完整完成，留下长前缀 H 诊断|数值门槛中止，不是联合方案效果为负|
|阶段 2.1：本次|云端修复验证完成，等待 AutoDL 九点结果|先数值可靠，再看同题不同位置的 H、C、正确性与长度效应|
|阶段 3：下一轮再开展|设计同一前缀 continue / summary-reset / answer 配对动作|严格对称预算、失败摘要处理、相同随机种子；判断动作净收益|
|阶段 4：有早期证据再扩展|开发集选择简单规则，测试集封存|联合信号对单信号、长度、固定／随机、原生策略的增益及含监测成本的净收益|

本次不增加题目、不训练任何新模型、不以九点拟合复杂规则。阶段 3 的实验指令应基于阶段 2.1 返回结果制定，避免把尚未验证的测量直接变为大规模干预。

## 论文核对后的方向判断

值得继续最小实验，但尚无顶会级贡献已成立的证据。ASAG 已覆盖 H+C 联合控制早停／继续；InftyThink+ 已有固定 5k、随机 3k–6k 和模型原生摘要时机比较。因此创新要落在：**联合内部信号能否区分“可结束”与“适合摘要后继续”，并预测动作的条件性净收益，在包含监测／probe／摘要／重建 prefill 成本后仍获益。**

原版 v5 主表也包含 R1-distill-Qwen-1.5B，不能用基座名称直接区分原版与 Plus。两篇主评测均最多 10 轮×约8k，原版每题16次、Plus32次采样；现有3轮内部协议不等于完整论文复现。当前 checkpoint 的 cold-start、task-RL 或 task+efficiency-RL 主表行映射仍缺证。详情见 `PAPERS_AUDIT_zh.md`、`ASAG_AUDIT_zh.md` 和 `FAILURE_REVIEW.md`。
