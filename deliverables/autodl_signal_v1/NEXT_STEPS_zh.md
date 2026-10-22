# InftyThink 内部信号预实验：第一轮执行说明

日期：2026-10-22。目标是冻结已有模型，研究 attention 熵变化＋答案置信度能否选择有净收益的摘要/重置时机。本包先完成审计和低成本信号测量；不包含已验证的联合控制器。

## 证据范围与当前判断

本轮已读 ASAG 2606.15070v1 原文，已审计两个交接包的实际执行脚本、配置、checkpoint 元数据、原始结果，以及 InftyThink/InftyThink+ 官方仓库与项目页。无法读取用户 Windows 的 C:、D: 盘；指定的 2503.06692v5.pdf 和 2602.06960v3.pdf 不在上传包内，arXiv 访问被当前代理以403拒绝。这两篇指定版本全文仍待上传后逐项核对。D:\cot 如在打包后有更新，需要上传新增或修改的文件。

**合理性：值得检验。顶会潜力：尚无足够证据，简单组合的创新性偏弱。** ASAG 已使用 attention 状态和答案置信度做生成干预；InftyThink+ 已研究何时摘要、保留什么、如何继续。更有价值的可证伪命题是：控制长度和阶段后，联合信号能区分“可直接结束”和“适合摘要后继续”，并识别摘要相对继续的净收益。必须用实验支持，不能预先证明必然有效。

关键审计结论：

- 旧分支未采集attention，不能否定联合方案。当前C是所选token概率算术均值，包含闭合答案分隔符。
- ASAG方法阶段H使用最后4层、所有query heads，在probe指令及等长最近思考token上求和，除log(key数)。本地模型是28层、12 query heads、2 KV heads。
- ASAG在线ΔH相对第一次ATP；正文方法probe窗口不等于附录观察实验的固定32。完整ASAG还包括概率混合与jump prompt。
- checkpoint模型卡指向InftyThink+，具体cold-start/task-RL/efficiency-RL阶段及论文主表行未闭环。metadata声明不能替代权重字节来源校验。
- native20每轮8192、最多3轮，没有A/B的16384总token限制。A首段受8192-cut限制，B强制摘要占一个segment；旧协议需作为独立版本保留。
- tokenizer BOS=151646，但model config BOS=151643。信号回放直接使用保存IDs，不解码再编码。

详见同包 audit/ 下三份审计报告。

## 阶段与交付

| 阶段 | 做什么 | 指令/入口 | 进入下一阶段的条件 |
|---|---|---|---|
| 0 来源审计 | 核对代码、模型、论文；补齐两份指定PDF | 阅读audit，运行下面preflight | 差异和未确认项有记录；GPU环境可用 |
| 1 数值与接口 | 小模型比较完整attention与分块KV回放；真模型仅2题×2短前缀 | test_signal_smoke.py、signal_smoke.py | 数值自测通过、输出完整、成本和显存可接受 |
| 2 真实长度信号 | 从旧native20提取2k/4k的自然边界，采集触发前H和C | 下面prepare-only与candidates命令 | 实际prefix与分支位置匹配；定义与ASAG差异明确；截断probe单独标注 |
| 3 配对动作诊断 | 在同一前缀比较A继续、B摘要、C短答案；复用旧结果或新目录运行 | 下面原脚本命令；仅作为legacy协议诊断 | 确认动作存在条件性收益，检查长度/阶段和预算混淆 |
| 4 新协议与规则 | 固定公平预算、ASAG式probe定义、dev题目划分；比较长度、C、H、H+C | 需根据阶段1–3结果编写并测试新driver；本包没有此命令 | 规则仅在开发题校准，预注册阈值搜索与成本口径 |
| 5 冻结后的在线评估 | 相同引擎比较原生、固定/随机、单信号、联合及早停替代；记录全部成本 | 新driver验收后提供命令；当前不声称已实现 | 题目级置信区间、独立题目/模型泛化和误差分类 |

前3阶段约2–4个工作日可得到一次go/no-go判断，取决于GPU可用性和首次实测速度。后两阶段是研究工作，不能把3–5日的预实验估计当作完整论文周期。先完成下面首轮小试，再确定扩展规模。

## 上传什么

**原AutoDL仍有旧环境、模型和实验目录：只需上传本包 autodl_signal_v1.zip 到 /root/autodl-tmp。** 论文不用传GPU机器；旧交接ZIP、tar.gz、模型权重不用重复上传。

本包包含preflight.py（只读检查）、signal_smoke.py（离线attention＋短答案诊断）、test_signal_smoke.py（数值自测）、run_first.sh（首轮入口）、本说明、审计记录和校验和。各脚本的新输出写到独立目录；不安装或升级旧环境。

**新AutoDL实例：**另需 inftythink_autodl_handoff_20261021T132447Z_1396.tar.gz 恢复代码/数据/历史记录，以及固定revision的模型权重与兼容环境。tar.gz不含权重。模型下载/环境恢复应单独验收；不要直接把下述“原机器路径”命令套到缺失文件的新机器。当前采集器接受显式--model、--baseline；旧分支器的模型路径来自历史run_config，不能原地改旧配置来迁就新路径。

## 原AutoDL机器：首轮直接执行

最简方式：上传ZIP后执行下面三条，脚本自动运行环境检查、数值自测和2题×2短前缀测量，结束时打印OUTPUT_DIRECTORY和RETURN_ZIP。把RETURN_ZIP对应文件带回即可。检查失败会停在失败步骤；不会继续长前缀或分支实验。

```bash
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_signal_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_signal_v1.zip -d "$UPLOAD_DIR"
bash "$UPLOAD_DIR/autodl_signal_v1/run_first.sh"
```

上面首轮入口与下面的手动首轮命令二选一，不要重复跑。手动方式可逐条检查，后续命令沿用其EXP变量；自动方式后续先把EXP设为打印的OUTPUT_DIRECTORY，并设置ROOT/PY/MODEL/NATIVE/PKG。

以下命令在AutoDL终端运行，而不是Windows PowerShell。所有变量沿用同一个终端；开始后会显示本次工作目录。若失败，先保留日志和preflight.json，不执行大规模推理。

```bash
set -euo pipefail
ROOT=/root/autodl-tmp
PY="$ROOT/venvs/infty-official/bin/python"
MODEL="$ROOT/inftythink_day1/models/InftyThink-1.5B"
NATIVE="$ROOT/inftythink_official_check/outputs/native5"
EXP=$(mktemp -d "$ROOT/infty_signal_XXXXXXXX")
unzip -q "$ROOT/autodl_signal_v1.zip" -d "$EXP"
PKG="$EXP/autodl_signal_v1"
cd "$PKG"
sha256sum -c checksums.sha256
printf '本次目录: %s\n' "$EXP"
nvidia-smi
"$PY" preflight.py --root "$ROOT" --output "$EXP/preflight" --require-gpu
"$PY" test_signal_smoke.py 2>&1 | tee "$EXP/unit_tests.log"
```

验收：preflight.json的audit_ok及gpu_prerequisites_ok为true；数值测试无失败。preflight也检查vLLM0.8.5，因为后续旧分支使用它；signal_smoke自身使用Transformers4.51.3与PyTorch2.6。GPU测试命令已按API审阅，但云端没有真实权重/GPU，需以AutoDL实测为准。

接着仅跑2题、128和256生成token处的前缀。它们用于工程一致性，不用于选择有利题目或调策略阈值：

```bash
"$PY" "$PKG/signal_smoke.py" \
  --model "$MODEL" --baseline "$NATIVE" \
  --cases 1,6 --cuts 128,256 \
  --query-window 32 --prefill-chunk 128 --probe-tokens 32 --check-short-reference \
  --device cuda --output "$EXP/signals_short" \
  2>&1 | tee "$EXP/signals_short.log"
```

这里H是**前缀末32个query的attention熵**，C使用旧分支格式的boxed短答案探测；它是测量诊断，不是完整ASAG复现，不可直接套用0.95/-0.1触发摘要。仅观察实际值、稳定性、显存和成本。probe未闭合需记为截断/无完整C，不把它当作低置信错误答案。

--check-short-reference会在总输入不超过512 token时，用相同真实权重、设备和dtype的完整eager attention作为参考，报告H/logits差异和下一token是否相同。参考计算的额外时间单独保存；峰值显存包含该参考检查，因此首轮峰值不能当作仅监测开销。超过512会明确记录跳过，不构造长序列完整矩阵。BF16差异需实际检查，不把CPU数值测试通过等同GPU数值已验证。

首轮运行后，把本次目录中的preflight、数值测试日志和signals_short输出带回。根据这些实测，才决定下面长前缀步骤是否继续。

## 后续：同一自然边界的H/C与旧A/B/C配对

下面命令是可选的下一轮；不要和首轮一次性整批运行。它调用已有脚本，仍用旧budget协议，用于诊断和关联，不作为最终公平性/加速证据。

先只准备候选，不加载模型。选前3个符合边界条件的开发题，不依据对错筛选：

```bash
BRANCH="$ROOT/infty_sampling_check/branch_sampling_check.py"
"$PY" "$BRANCH" --baseline "$NATIVE" --cut 2048 --boundary-window 128 \
  --limit 3 --seed 2026 --temperature 0.7 --top-p 0.95 \
  --output "$EXP/legacy_2048" --prepare-only
"$PY" "$BRANCH" --baseline "$NATIVE" --cut 4096 --boundary-window 128 \
  --limit 3 --seed 2026 --temperature 0.7 --top-p 0.95 \
  --output "$EXP/legacy_4096" --prepare-only
```

使用候选文件中实际cut而不是2048/4096名义值采集，核对保存prefix IDs后再运行：

本次离线复算：2048的前三题为1/2104、4/2069、5/2079；4096为1/4103、6/4203、8/4173。这两个小集合只重叠题1，不能把两组均值差解释为位置效应。该轮主要检查采集接口；只有同题至少两个有效位置才有有意义的相对变化，单点的首点ΔH=0只是初始化。正式动作分析必须另外固定同一题目集合和多个前缀位置。

```bash
"$PY" "$PKG/signal_smoke.py" \
  --model "$MODEL" --baseline "$NATIVE" \
  --candidates "$EXP/legacy_2048/candidates.json" "$EXP/legacy_4096/candidates.json" \
  --query-window 32 --prefill-chunk 128 --probe-tokens 128 \
  --device cuda --output "$EXP/signals_boundaries" \
  2>&1 | tee "$EXP/signals_boundaries.log"
```

若要取得新A/B/C动作结果，保持相同参数，仅将prepare-only替换为resume。每个调用是独立进程，Transformers与vLLM不会同时占用GPU：

```bash
"$PY" "$BRANCH" --baseline "$NATIVE" --cut 2048 --boundary-window 128 \
  --limit 3 --seed 2026 --temperature 0.7 --top-p 0.95 \
  --output "$EXP/legacy_2048" --resume \
  2>&1 | tee "$EXP/legacy_2048.log"
"$PY" "$BRANCH" --baseline "$NATIVE" --cut 4096 --boundary-window 128 \
  --limit 3 --seed 2026 --temperature 0.7 --top-p 0.95 \
  --output "$EXP/legacy_4096" --resume \
  2>&1 | tee "$EXP/legacy_4096.log"
```

4096已有同seed结果时优先复用对应case/cut的保存记录，可跳过重复GPU生成，但必须核对prefix/model/script/config哈希与采样设置。跨引擎数值差异要保留：新HF的C不能假定与旧vLLM的C逐token一致；新C闭合即停，旧C可能继续解码到上限，成本也不同。

以上只产生小规模动作诊断。正式新协议须明确允许摘要失败回退与否、按何种统一budget控制A/B、摘要是否占segment、probe频率和花费。不把新协议和legacy结果混合续跑。尚未实现的阶段4/5不列假命令。

## 应交回哪些输出

在本次EXP目录执行：

```bash
"$PY" - "$EXP" <<'PY'
import pathlib, sys, zipfile
p=pathlib.Path(sys.argv[1]).resolve()
out=p.parent/(p.name+'_return.zip')
with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(p.rglob('*')):
        if f.is_file() and 'autodl_signal_v1' not in f.relative_to(p).parts:
            z.write(f, f.relative_to(p))
print(out)
PY
```

返回ZIP包含本次配置、逐点信号/概率/时耗/显存、日志，以及可选分支输出。不需要传模型权重、缓存或整个虚拟环境。保留新输出目录，原始结果一律保留。

## 后续判断idea的证据标准

1. 动作价值：在同一前缀、明确budget下，摘要至少在一类状态中改善准确率/成本折中；以多seed估计不确定性，不能从少数事后选中的成功例子外推。
2. 信号增量：H/ΔH+C应优于长度/阶段、C-only、H-only；如果收益只来自直接早停，需承认摘要动作本身没有得到支持。
3. 真正效率：所有probe、监测、摘要、reset prefill和失败回退都算入在线总时间。离线回放开销单列，不能把它当免费在线信号。
4. 泛化：按题目划分校准与评估，同题多位置/seed不能跨集合；按题目聚类bootstrap/配对比较。8–12题是排错与go/no-go，120题也可能不足以证明小幅准确率不劣，需据实际方差估计功效。
5. 论文贡献：若有稳定证据，再验证第二个checkpoint/任务，报告准确率-时间曲线和失败机制。顶会潜力依赖这些证据与更完整相关工作检索，当前不提供录用概率或成功承诺。

test120和reserve300在规则冻结前不用于调阈值；已有20题属于开发材料。免训练允许预先限定的开发集阈值校准，但应公开搜索范围，避免把大量调参的成本藏起来。
