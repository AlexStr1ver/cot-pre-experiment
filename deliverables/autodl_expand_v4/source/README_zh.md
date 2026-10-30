# 下一步 v4：扩五道开发题，采集信号并运行60条动作分支

计划日期：2026-10-29；交付日期：2026-10-30（北京时间）。本包继续冻结模型、免训练的 attention 熵变化＋答案置信度选择摘要／重置时机方向。它扩展条件动作收益的数据，不是已经验证的 H+C 在线控制器。

## 固定范围与目的

原 native20 满足三个长度目标的题目为 [1,6,8,10,11,12,15,19]，已有配对研究使用 [1,6,8]。本轮纳入全部剩余五题 **[10,11,12,15,19]**，不按答案或旧收益挑题。

- 目标仍为1024、2048、4096，取之后128 token内首个段落边界；共15个新前缀。
- 全部15点重新采集 raw/normalized attention H、相对最早点变化、短答案 C、完整probe输出与成本。
- 每点固定seeds **2026、2027**，运行A继续／B立即摘要重置，共 **60条**。每个seed30条。
- 原actions、decoder、collect_phase2、v1_core、prepare_plan保持原字节。FP32、TF32关闭、temperature0.7、top_p0.95、摘要cap1024、共享前缀计入16384总生成预算、失败扣费回退均不变。
- 旧54条不重跑、不混入新结果、不覆盖。新旧题和seed数量不同，分别报告，按题聚合；不能把114条分支称作114道独立题。
- 这些五题属于此前看过native20的开发题，新增的是H/C动作配对，不是未见测试集。test120继续封存。

计划已经冻结在expansion_plan.json。所有15点和60分支均预先指定；不会因为中途答错或收益不好而删除、换题或只重复赢家。下一轮开发决策标准见PREREGISTRATION_zh.md。

## 下载、上传和启动

下载GitHub上的autodl_expand_v4.zip，文件页点击 **Download raw file**，上传到 `/root/autodl-tmp/`。无需上传模型、数据或论文。

```bash
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_expand_v4_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_expand_v4.zip -d "$UPLOAD_DIR"
PKG="$UPLOAD_DIR/autodl_expand_v4"

bash "$PKG/launch_expanded.sh" /root/autodl-tmp \
  /root/autodl-tmp/infty_paired_v3_rsiFEcc2
```

第二个参数rsiFEcc2用于核验健康环境，不复用它的分支。每次创建新的infty_expand_v4_XXXXXXXX目录，后台执行；不要重复启动，实例须保持开机。

执行顺序：包校验 → 原健康环境／隔离导入检查 → CPU测试 → 权重SHA与15个前缀核验 → 15点H/C采集及collector canary → 封存信号 → 五题最长前缀数值审计 → 全部60条A/B。

只在工程检查失败时停止。摘要失败、答错或用尽预算会按原协议记录并继续，不会根据答案质量筛掉结果。复用已经健康的原环境；依赖异常时才按已有恢复逻辑新建环境，原环境不覆盖。

## 查看状态和完成条件

启动器打印实际目录、日志、状态命令和最终ZIP路径。也可将RUN替换为实际新目录：

```bash
RUN=/root/autodl-tmp/infty_expand_v4_本次实际后缀
python3 -I "$PKG/isolated_expand_entry.py" script status_expanded.py "$RUN"
```

首次完整运行应满足：

```text
exit_status.txt = 0
evidence/signals/summary.json: completed=true, executed_positions=15, canary_passed=true
results/summary.json: mode=expand, completed=true, completed_branches=60
results/summary.json: new_branches=60, reused_branches=0, numerical_audit_passed=true
```

预算截断计入完成的实验记录，但不等于完成有效答案；最终正确性另行人工复核。新题可能有分数或表达式答案，自动整数判分的null须人工处理，不能把null自动判错或当程序故障。

按本次36条的平均轨迹长度及吞吐外推，60条约 **126分钟**；全部用满预算最多837400个追加生成token，按本次54.27token/s外推约 **257分钟**。两者均为资源规划情景，不是时间承诺或墙钟上限；新题长度、硬件负载、信号采集和数值审计会影响耗时。

结束或报错后，上传本次新的 `/root/autodl-tmp/infty_expand_v4_XXXXXXXX_return.zip`，不重复上传rsiFEcc2。

## 中断恢复

若只在环境、采集信号或数值审计阶段失败，还没有可复用的完整动作分支，重新执行首次启动命令并回传失败包即可，不修改旧目录。

若v4已通过五题数值审计且产生完整分支，可以从停止后的v4目录恢复：

```bash
bash "$PKG/launch_expanded.sh" /root/autodl-tmp \
  /root/autodl-tmp/infty_paired_v3_rsiFEcc2 "$RUN"
```

恢复只接受同版v4：原15点信号和manifest逐字节复制以保持身份；当前权重和前缀重新核验；已完成分支逐条检查hash后复用，未完成分支从原前缀重跑；五题数值审计仍重新执行。不做token/KV级续跑。此时new/reused可以变化，最终唯一分支数应为60。

普通终止自动归档。关机或SIGKILL后，确认所有进程已停止，可在状态命令末尾加 `--pack` 补打包，不覆盖原ZIP。

## 验证边界

云端已做原CPU核心测试、扩展完整流程模拟、错误与恢复检查，以及真实tokenizer和原始15个前缀校验，具体计数见VALIDATION.json。云端没有实际新GPU推理。AutoDL上的实时依赖、权重、数值和资源检查仍必须通过。

v4采集的C是同前缀短答案诊断：每状态一次，不是两个独立seed样本；其时间含独立prefill和生成。它支持判断“摘要收益是否只是答案早已就绪”，但没有实现或复现完整ASAG早停控制器。正式在线A/B/STOP策略及所有监测总成本留到下一阶段，不以离线动作数据直接宣称端到端加速。
