# 阶段3：同一前缀的继续／摘要重置配对实验

2026-10-24（北京时间）。本包在固定 InftyThink 系列 checkpoint 上做免训练动作验证。不是新训练、不是已部署的 H+C 控制器，也不是完整论文复现。阶段2的九点信号和旧结果全部保留。

## 本轮要回答的问题

在同一个已保存推理前缀上，A=继续原生推理，B=立即生成摘要、用原题＋摘要重建上下文后继续。比较最终答案、完成状态和实际成本；用已采集H/C解释动作差异，而不是先根据九点调一个规则。

- 同九点：case1 cuts1041/2104/4103；case6 1055/2092/4203；case8 1084/2055/4173。
- 同模型、FP32、关闭TF32、HF eager；temperature=.7、top_p=.95，使用明确的torch nucleus采样，无隐含top_k。与旧vLLM不保证逐token一致。
- A也允许之后自然摘要；B只在当前点额外强制一次摘要，之后双方使用同一原生规则。
- 两支共同剩余生成预算为`16384-cut`，含全部推理、答案、摘要、EOS、摘要闭合token和失败摘要。外部控制token作为输入成本另记。
- B强制摘要最多1024生成token；失败后扣除已花费预算，从原始前缀继续。失败不免费、不剔除、不根据正确答案决定回退。
- 持续KV解码，不再在`8192-cut`处把A提前结束，不让B额外占掉一个有限segment；只有真实摘要／回退才重建输入。
- 摘要是模型自己生成，不使用参考答案；参考答案只用于运行后的保守判分。seed由case/cut/replicate/角色/次数确定，不因日志心跳重置；相同seed不意味着A/B具有相同随机轨迹。

## 上传内容与环境

只需上传 **`autodl_paired_v3.zip`** 到 `/root/autodl-tmp/`。包里已附原样的阶段2 plan、九点信号及指纹，无须重新上传阶段2结果、论文或权重。

继续使用：

```text
/root/autodl-tmp/venvs/infty-official/bin/python
/root/autodl-tmp/inftythink_day1/models/InftyThink-1.5B
/root/autodl-tmp/inftythink_official_check/outputs/native5
```

不安装或升级依赖。要求既有Transformers4.51.3和可用CUDA，建议保持当前RTX4090D。阶段2已测得约7.1GiB峰值，但长动作实际峰值与耗时仍以本轮为准。若改用别的GPU，不复用旧GPU的动作耗时记录。

GitHub下载路径：`codex/autodl-signal-delivery` 分支下 `deliverables/autodl_paired_v3/autodl_paired_v3.zip`，打开后点击 **Download raw file**。解压后不要source脚本。

## 第一步：数值补检＋一个A/B检查对

```bash
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_paired_v3_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_paired_v3.zip -d "$UPLOAD_DIR"
PKG="$UPLOAD_DIR/autodl_paired_v3"
bash "$PKG/launch_paired.sh" /root/autodl-tmp smoke
```

启动后打印`OUTPUT_DIRECTORY`、日志查看和状态查询命令。任务在后台运行；普通SSH/浏览器断开不会停止任务，实例仍须保持开机。PID只表示启动，不能当作实验通过。

smoke会依次：

1. 校验代码包、输入指纹和九点计划，执行CPU测试。
2. 流式计算完整本地权重SHA256，固定当前内容身份；此指纹不能单独证明它对应论文哪一训练阶段。
3. 对三题各自最长前缀（1/4103、6/4203、8/4173）执行chunk128/64的H和真实C比较；前一题用于当前运行复核，后两题补原覆盖缺口。同时与已上传FP32信号比较，防止当前数值自洽却与旧信号不一致。保持H1%、logit均差.05、token概率差.01的工程阈值。
4. 数值检查通过后，预热解码路径；从case1/cut4103执行完整预算的A、B各一条，seed2026。不是缩小预算的假对照。

请将打印的实际目录保存为变量，例如（把示例后缀替换成实际值）：

```bash
SMOKE_RUN=/root/autodl-tmp/infty_paired_v3_实际smoke后缀
/root/autodl-tmp/venvs/infty-official/bin/python "$PKG/status_paired.py" "$SMOKE_RUN"
```

继续条件为退出码0、`completed=true`、`completed_branches=2`、`numerical_audit_passed=true`。**不要求两支答对，不按答案选择是否继续。** `generation_budget_limit`、无有效摘要等模型结果也会保留并正常结束记录；CUDA/OOM/进程错误则阻断扩展。

这两支最坏追加生成24562token，可能明显长于阶段2的13秒测量。看日志每256token的心跳和最终吞吐字段；不能承诺未经实测的完成分钟数。

## 第二步：补齐九点18条，复用已完成首对

第一步成功后可直接执行，无须重新跑首对：

```bash
bash "$PKG/launch_paired.sh" /root/autodl-tmp main "$SMOKE_RUN"
```

该模式验证前一目录的代码／协议、权重内容、前缀、GPU及运行时身份、完成记录校验和。将首对的完整结果**逐字节复制**进新的结果目录，保留原运行配置和成本来源，然后只运行余下16条。额外数值复核与预热另计，不混入A/B成本。

main最终应有`completed_branches=18`、`reused_branches=2`、`new_branches=16`，退出码0。若从一个中断的main恢复，复用数可能大于2；仍以最终18条唯一记录为准。完整首批最大追加生成上限251092token，已完成首对不会重跑。

如果main普通中断且已保存完整首对，可把那个main目录作为新的main来源：

```bash
PREVIOUS_MAIN=/root/autodl-tmp/infty_paired_v3_中断main实际后缀
bash "$PKG/launch_paired.sh" /root/autodl-tmp main "$PREVIOUS_MAIN"
```

每次写新目录，只复用有完整记录和正确hash的分支。未完成的单支从原前缀重新执行；**不是token级KV断点恢复**。旧部分结果仍留在旧目录。smoke若未完成首对，先回传失败包定位，不绕过检查扩大运行。

若重新打开终端，`PKG`变量会丢失；请重新设置为解压包的实际完整路径。

## 回传什么

运行结束或普通程序失败都会产生相邻文件：

```text
/root/autodl-tmp/infty_paired_v3_XXXXXXXX_return.zip
```

建议先保留smoke包，main结束后回传 **main的return.zip**（已含复用首对和来源配置），方便一起分析全部18条。发生错误时回传失败包，不自行放宽数值门槛或改预算。

查看日志用启动器打印的`tail -f .../launcher.log`，`Ctrl+C`只停止查看。若需主动停止任务，对输出目录中`runner.pid`所指进程发送TERM；运行器会通知工作进程停止并尽量打包。关机/SIGKILL无法保证退出钩子执行，确认进程已停后可补打包：

```bash
/root/autodl-tmp/venvs/infty-official/bin/python "$PKG/status_paired.py" "$SMOKE_RUN" --pack
```

`--pack`会检查主进程与工作进程，不覆盖已有ZIP。

## 第二批seed（首批运行和成本核对后）

本包同时提供扩展入口，默认不会自动运行：

```bash
MAIN_RUN=/root/autodl-tmp/infty_paired_v3_已完成main实际后缀
bash "$PKG/launch_paired.sh" /root/autodl-tmp replicate "$MAIN_RUN"
```

它要求main的18条已完整结束，在所有九点统一追加seed2027、2028，共36条；不挑选赢家。首批18＋第二批36合计54条，上限753276追加生成token。若replicate中断，可将中断的replicate目录作为该命令第三参数，只重跑尚未完成的分支；原完成记录仍需匹配全部身份与校验。

本轮优先获得首批18条结果和实际成本，再决定是否投入第二批。三题、三seed仍是开发诊断，不能用于宣称统计显著或泛化准确率。

## 结果字段与预先规定的评分口径

`results/summary.json`的completed只表示执行与预算核算完成；正确性在`branch_review.csv`与`paired_review.json`中另列。

- `branches/*.json`：每次真实prefill的输入IDs、采样seed、生成IDs/text、摘要、重置输入、状态与时间，逐256token更新。失败也尽量保存。
- 所有样本预算按**采样token数**计费。缓存解码最后一个采样token无需再forward，所以forward次数可能少1，不能用它代替预算。
- `model_call_seconds`为会话时间，含主机采样／处理／检查点开销；`branch_wall_seconds`另计分支总耗时。模型加载、权重hash、数值检查、预热和历史H/C重放另列，不假称纯GPUkernel时间或在线控制器延迟。
- 只有EOS且`</think>`后的最终区域非空才记`answer_completed=true`。最终盒子从该区域最后一个完整balanced box提取，思考／摘要中的盒子不计。
- 预算耗尽但最终区域已有完整盒子时，保留`candidate_boxed`和文本，标记截断，并给单独的候选答案诊断；不悄悄丢弃，也不冒充EOS完整结束。
- 当前自动判分只处理明确整数；单位、分式或复杂等价表达式留待人工核查，不自动判错。主报告并列严格完成整数判分与截断候选判分，不能事后挑选更有利口径。
- 强制摘要失败回退仍属于B组（按原分配统计），失败成本保留；只分析成功重置者属于另一个有选择偏差的诊断，不能代替总体B结果。
- C复用阶段2的短答案及H/C；不是又产生了一批独立C样本，也不是ASAG插入`</think>`后的最终答案动作。

`completed_artifacts.json`固定完成分支的内容hash；`reuse_manifest.json`与`prior_provenance/`保留复用来源。兼容性检查失败时不静默合并。

## 验证范围与研究边界

见`VALIDATION.json`与`PROTOCOL.json`。本包已在CPU上验证缓存采样、真实tiny-Qwen接口、预算与状态机、smoke→main→replicate复用、失败阻断及后台运行生命周期；还用实际九个前缀和真实tokenizer检查了18条脚本化输出的摘要重建。测试夹具不是实际checkpoint的实验结果。

当前云端没有CUDA/真实权重，实际AutoDL动作实验尚未执行。包中阶段2测量是原样输入证据，不能误当成本包新跑的数据。完整权重hash是本轮新增，历史阶段2仍缺权重hash，跨轮信号比较是数值核验而非密码学同一性证明。

当前不拟合H+C阈值、不训练探针、不覆盖旧结果。下一步分析首先看B相对A是否有可复核的动作收益，再问H+C是否比C-only、长度、固定／随机和原生策略更能选位置；正式效率结论还需真实在线监测与总成本核算。
