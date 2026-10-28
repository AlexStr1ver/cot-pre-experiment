# v3.2 修复包：GPU 总容量微小差异不再误拦截

2026-10-28（北京时间）。本包接替 v3.1，继续九个固定前缀的 seeds 2027、2028 配对实验；不是新实验方向。旧代码包和结果均保留。

## 本次确切原因

`infty_paired_v3_FCuKe8gI_return.zip` 完整，43 个返回代码文件与 v3.1 发布版一致。退出码 1，停在 environment_preparation：尚未执行 CPU 测试、依赖安装或模型推理，没有新的实验分支。

本次 Python 3.10.8、torch 2.6.0+cu124、Transformers 4.51.3、CUDA 12.4、RTX 4090 D 名称及 [8,9] 算力均通过检查。唯一差异是驱动报告的总显存：

| 记录 | 总显存字节数 |
| --- | ---: |
| 成功 main kt2eZTc2 | 25,252,724,736 |
| 本次 FCuKe8gI | 25,250,627,584 |
| 差值 | -2,097,152（-2 MiB，约 -0.0083%） |

v3.1 错把总显存要求为逐字节相等。环境探测和后续旧结果复用各有一道相同检查，两处都要修复。不能从这些记录确定容量变化的具体原因，也不能证明两次使用同一张物理 GPU。

## 修复内容与保留检查

- 两处使用同一个硬件兼容模块。GPU 型号、计算能力、torch/CUDA 构建、设备类型仍精确匹配；总显存允许差值同时不超过 **16 MiB 和原容量的 0.1%**。缺失信息、非法容量或超限差异仍拒绝。
- 真实总显存原样保存，并记录新旧容量、差值、容差与判定理由。容差用于容量报告兼容，不代替数值一致性验证。
- 修改协调脚本后如实生成**新代码指纹**，不假装仍为原指纹。只允许已知原 v3 指纹迁移到审核过的修复版；其他五个科学文件、协议、权重、前缀、tokenizer 与 phase2 evidence 必须完全一致。所有旧分支仍核验 SHA 和记账。
- actions.py、decoder.py、collect_phase2.py、v1_core.py、prepare_plan.py 均逐字节不变。摘要动作、预算、FP32、温度、seed、评分、H/C 数值阈值均不变。协调脚本差异见 COORDINATOR_DIFF.txt。
- 新版保留隔离导入、自动归档、后台运行和只复用完整分支的恢复逻辑。按本次诊断，原环境健康，应直接复用，不重新安装依赖。若未来依赖损坏，仍只新建隔离环境，不覆盖原环境。
- GPU 运行前必须通过原 24 项核心测试及新增兼容性测试，再进行三个最长前缀的实际 H/C 数值审计。云端 CPU 验证不能替代 AutoDL 上这一步。

## 上传和执行

GitHub 文件页点击 **Download raw file** 下载 `autodl_paired_v3_2.zip`，上传到 `/root/autodl-tmp/`。不需要重新上传模型、数据或论文，也不需要重跑 main18。

```bash
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_paired_v3_2_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_paired_v3_2.zip -d "$UPLOAD_DIR"
PKG="$UPLOAD_DIR/autodl_paired_v3_2"

bash "$PKG/launch_paired.sh" /root/autodl-tmp replicate \
  /root/autodl-tmp/infty_paired_v3_kt2eZTc2
```

**来源必须是成功的 main：kt2eZTc2。不要使用失败的 FCuKe8gI 或 gddcYOhk，它们没有模型结果。** 每次启动会创建新目录，不覆盖旧结果。不要同时重复启动；后台任务可承受普通终端断开，实例仍须开机。

启动后复制终端打印的状态命令。也可以用下面命令，将 RUN 替换为本次实际目录：

```bash
RUN=/root/autodl-tmp/infty_paired_v3_本次实际后缀
python3 -I "$PKG/isolated_entry.py" script status_paired.py "$RUN"
```

环境准备正常时，`environment/preparation.json` 应显示 `selection=existing_healthy` 和 `completed=true`。若模块遮蔽仅在普通模式发生，也可以选择 `isolated_import_recovery`。这只是准备完成，不代表模型实验结束。

## 完成、回传与恢复

首次完整运行应满足 `exit_status.txt=0`，以及 results/summary.json：

```text
mode=replicate
completed=true
completed_branches=36
new_branches=36
reused_branches=0
numerical_audit_passed=true
```

36 = 九点 × 两个 seed × A/B。加上旧 main18 才是总计 54；新包不应要求显示 54。新旧指纹与硬件兼容审计在 results/reuse_manifest.json，旧来源元数据只读复制到 prior_provenance。计时保留运行来源，不把相同型号当作同一物理卡或完全相同速度的证明。

结束后上传**本次新生成的** `/root/autodl-tmp/infty_paired_v3_XXXXXXXX_return.zip`。发生错误也上传这个 ZIP，内含原始日志、代码、环境与退出状态。

若仅环境/CPU 检查失败，下次仍以成功 main kt2eZTc2 为来源。若新版运行已通过数值审计并产生完整分支，停止后可从该新目录继续：

```bash
bash "$PKG/launch_paired.sh" /root/autodl-tmp replicate "$RUN"
```

恢复时 new/reused 数量可以不同，总数应为 36。只复用完整、校验通过的分支；未完成分支从原前缀重跑，不恢复中途 KV。普通终止会归档；关机或 SIGKILL 后可在确认进程停止后为状态命令追加 `--pack` 补打包，不覆盖已有 ZIP。

## 当前科研进度

本次失败新增 **0 条模型分支**。现有证据仍是三个题、九个前缀、一个 seed 的 A/B 共 18 条：人工处理角度单位后 A/B 都是 4/9，B 救回 1 对、损害 1 对，整体少 15.87% 追加 token。动作存在潜在收益；H+C 是否能可靠选时机、端到端成本是否划算仍未证实。

下一步仍完成全部九点的两个 seed，检查收益和损害是否可重复。保持免训练方向，不在这九点上拟合阈值，不打开 test120。本次是工程误拦截，不能作为 idea 成功或失败的科研证据。

完整验证范围见 VALIDATION.json。原研究协议保留在 PROTOCOL_NOTES_v3_zh.md；旧文档里的严格硬件字节相等及指纹完全相等的复用要求，由本版明确、受限且留痕的兼容规则替代。

## 下载校验

[修复代码包 autodl_paired_v3_2.zip](autodl_paired_v3_2.zip)，文件页点击 Download raw file。

SHA256：`21712a1bf23928ee3781d394587f7ab3db6aecb05eb9fa5397d2d51348107dca`。共 50 个包内文件；校验和随包保存。完整日志在 validation_logs/，原失败 ZIP 原样保存于 evidence/。
