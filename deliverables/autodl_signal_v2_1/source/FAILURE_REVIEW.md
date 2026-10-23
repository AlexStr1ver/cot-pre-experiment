# 阶段 2 失败的独立复核

证据源：`infty_signal_phase2_ZTj8Ywpr_return.zip`，SHA-256 `486cf2f5b482c9b0971c659635e14a66a52674bb21cd96120730be4512f29802`。ZIP CRC 通过；返回的 6 个代码文件与 v2 发布件逐字节相同；包校验及 14 项 CPU 测试均通过。

## 已确认的直接原因

`exit_status.txt` 为 1。`signals.log` 给出 `collect_phase2.py:361` 主动抛出的 `RuntimeError("Long-prefix chunk consistency gate failed; inspect canary, do not expand")`。这是程序的数值一致性门槛中止，不是已证实的显存 OOM、CUDA 崩溃或连接中断。终端窗口为何关闭，返回包没有独立证据。

程序先执行 case 1、cut 4103 的长前缀 canary，再准备执行其余 8 个点。两个 H 窗口分别采用 prefill chunk 128 与 64 测量：

|窗口|H 相对差|原门槛|logit 平均绝对差|原门槛|贪心下一 token|
|---|---:|---:|---:|---:|---|
|raw_tail32|0.00269441（0.269%）|≤ 0.01|0.119981|≤ 0.05|相同|
|ASAG-style 12 queries|0.00569084（0.569%）|≤ 0.01|0.0300224|≤ 0.05|相同|

仅 raw_tail32 的平均 logit 差超过门槛。其余三项门槛通过。H 值不是 NaN，注意力行概率质量最大偏差约 0.0023。首次 C 探测尚未运行；9 个完整 H+C 测点完成数为 0。已有部分 H 必须保留作数值诊断，不能当作联合信号效能证据。

本次硬件是 **NVIDIA GeForce RTX 4090 D**，CUDA capability 8.9，25,250,627,584 bytes，原生 BF16 支持；不同于上一返回包的 RTX 2080 Ti。模型加载约 2.508 秒；四次 H replay 合计约 3.292 秒。中止发生在资源门槛和峰值显存记录之前，故不能从返回包给出实际峰值显存或资源门槛已通过的结论。

## 机制审计与不确定性

v1_core.py 每次 replay 新建 cache；HF 4.51.3 Qwen2 从 cache.get_seq_length() 派生 cache_position 与 position_ids，未见把位置重置或跨 replay 复用 KV 的代码。checkpoint 配置无动态 RoPE scaling、无 sliding window，attention dropout 为 0。CPU tiny-Qwen2 的 chunk/full 测试通过，但不能代替实际长前缀、BF16、GPU 验证。

改变 chunk 会改变 GEMM 的矩阵形状；BF16 舍入误差沿 28 层传播，是合理候选机制，当前日志不能独立证明因果。平均绝对 logit 差也可能含整体平移，整体平移对 softmax 不变；目前日志没有保存整份 logits 或分布距离，无法回算平移部分。不能凭“下一 token 一样”宣布置信度一致，更不能直接把 0.05 改为 0.2 以绕过门槛。

## 最小修复建议

在新的 v2.1 包中显式提供 FP32 内部信号诊断模式，关闭 TF32 并设置 float32 matmul precision 为 highest，保持原有 H 与 logit 数值门槛。重新按同一固定计划采集全部 9 个点；原 BF16 失败结果保留。FP32 推理改变数值口径，须在结果配置及报告中注明，不与旧 BF16 H/C 混合计算变化，也不宣称已复现论文 BF16 设置。

1.5B 模型 FP32 权重约 6.62 GiB，采用有界 chunk 的 eager replay，24 GB GPU 具有合理余量，但真实峰值仍以实测为准。不得用 4k 序列的所有 28 层完整注意力张量作为默认参考，以免引入可避免的显存压力。

新增实际置信度路径的 canary 对比：H 的 ASAG replay 最后固定为 12 queries，而 C 的 prefill 最后块是长度余数（4164 对 128 的余数为 68，对 64 为 4）；H 的 logit 比较不能替代 C 路径验证。对两种 chunk 各运行一次实际 greedy probe，记录 token 序列、status、C、逐 token 概率差，并以明确工程容差检查。空盒或截断产生 C=null，不能当成置信度一致性“通过”的单独依据。

## 建议验证

- 已上传失败值的回归用例：旧门槛仍拒绝 0.119981 的 raw-tail logit 差；H 差必须仍通过。
- FP32/TF32 配置写入真实 run_config；不偷偷自动切换 dtype。
- 长度不整除 chunk 的 tiny-Qwen2 FP32 测试，检查缓存长度与 raw/ASAG 窗口的 full-reference 一致性，并验证真实 probe-prefill 与 ASAG replay 在同前缀上的 logits 一致。
- C canary 对不同 token 序列、明显概率差、空/截断状态有明确处理；相同内容不等同相同 token 概率。
- 运行器保留非零退出码、日志、部分结果，并输出返回 ZIP；新目录写入，不覆盖旧实验。
- 真正的 4090 D FP32 canary + 9 点采集仍需要 AutoDL 执行。当前云环境没有该模型权重和 CUDA，不能将 CPU 测试包装成实际 GPU 成功。
