# v3.1 环境恢复包：继续全部九点的两个seed

交付日期：2026-10-28（北京时间）。本包修复环境检查、隔离导入和恢复入口。原模型、九个前缀、预算、FP32、采样、seed与判分协议保持不变；六个进入实验指纹的核心文件与v3逐字节相同。旧环境和结果不覆盖。

## 本次失败原因

`infty_paired_v3_gddcYOhk_return.zip`校验通过、代码与成功main一致，但退出码1。CPU测试导入时出现：

```text
cannot import name 'Qwen2Config' from 'transformers' (unknown location)
module 'torch' has no attribute 'zeros'
No module named 'transformers.AutoTokenizer'
```

没有results目录或experiment.log，说明模型实验尚未开始，36条新分支均未生成。日志里的20个测试条目、6个错误来自导入失败；部分纯Python测试通过不能证明依赖正常。失败包完整保存了现场，不是ZIP损坏。

问题确定在导入／依赖环境层。旧日志缺少模块来源，尚不能区分安装残缺、同名路径遮蔽、namespace残留或环境变化。标准库来自`/root/miniconda3`可以是正常venv行为，不能单凭该路径判断解释器错误。

## 修复流程与边界

1. 对现有Python进行普通与`-I`隔离导入检查，记录解释器、模块来源、API、版本和GPU身份。
2. 隔离导入正常且与成功main兼容时，继续使用原环境，不下载依赖。测试及推理以`-I`启动，先导入已安装库再加入本包目录。
3. 若依赖仍损坏或版本不符，新建`/root/autodl-tmp/venvs/infty-paired-v3_1_XXXXXXXX/`，安装PyTorch **2.6.0+cu124**、Transformers **4.51.3**及明确列出的兼容依赖。原`infty-official`不安装、不卸载、不覆盖。无需vLLM、torchvision或torchaudio。
4. Python主次版本或GPU身份不兼容、CUDA不可用时停止并保留诊断。不会放宽门槛；依赖健康但GPU信息不可核验时也不会重复安装。
5. 通过环境检查后，仍执行原24项CPU测试、数据／模型／权重身份检查和三题H/C数值检查，最后才启动replicate。

新环境和下载缓存位于结果目录外，不装入回传ZIP。需要重建时会下载CUDA依赖，并要求`/root/autodl-tmp`至少12GiB可用空间；不会自动删文件。健康环境不触发安装空间检查。新环境通过后记录可复用路径，后续仍重新核验。

安装期间查看`ENVIRONMENT_STAGE`及pip进度。下载失败同样归档。云端已验证恢复流程及CPU测试，没有实际安装CUDA环境或执行新GPU推理，验证范围见VALIDATION.json。

## 上传与启动

GitHub文件页点击 **Download raw file**，只上传`autodl_paired_v3_1.zip`到`/root/autodl-tmp/`。无需重传模型、论文或旧结果。

```bash
UPLOAD_DIR=$(mktemp -d /root/autodl-tmp/infty_paired_v3_1_package_XXXXXXXX)
unzip -q /root/autodl-tmp/autodl_paired_v3_1.zip -d "$UPLOAD_DIR"
PKG="$UPLOAD_DIR/autodl_paired_v3_1"

bash "$PKG/launch_paired.sh" /root/autodl-tmp replicate \
  /root/autodl-tmp/infty_paired_v3_kt2eZTc2
```

**来源填成功main `kt2eZTc2`，不要填失败的`gddcYOhk`。** 成功main只读核验，不重跑seed2026。

任务后台运行，普通终端断开不影响，实例仍须开机；不要重复启动。启动器打印实际输出目录、日志与状态命令。保存目录后可查询：

```bash
RUN=/root/autodl-tmp/infty_paired_v3_本次实际后缀
python3 -I "$PKG/isolated_entry.py" script status_paired.py "$RUN"
```

没有results目录时也会显示runner_status和environment/preparation.json，区分环境准备、CPU测试与模型阶段。PID或安装完成不等于实验完成。重新打开终端后需恢复PKG和RUN实际路径。

## 完成与回传

首次完整replicate应满足退出码0，以及：

```text
mode=replicate
completed=true
completed_branches=36
new_branches=36
reused_branches=0
numerical_audit_passed=true
```

九点×两个seed×A/B共36条。与已经保存的main18条合计54条；新包不要求显示54。答对不是运行通过门槛，摘要失败、错误答案或预算截断另列保留。

结束或普通失败后，回传**新生成的**`/root/autodl-tmp/infty_paired_v3_XXXXXXXX_return.zip`。环境失败也回传该ZIP，包含environment.log、探针JSON、状态、代码和退出码；不要再次上传旧gddcYOhk包。

若新replicate已进入模型阶段且数值检查通过，普通中断后可从该目录恢复：

```bash
bash "$PKG/launch_paired.sh" /root/autodl-tmp replicate "$RUN"
```

若仅环境／CPU测试失败，仍以成功main `kt2eZTc2`为来源。每次写新目录，只复用完整且hash匹配的分支；未完成分支从原前缀重跑，不做token级KV恢复。恢复运行的new/reused数量可能变化，以36条唯一完成记录为准。

主动停止可对启动器打印的runner.pid发送TERM；工作进程组（含安装子进程）会停止并归档。SIGKILL／关机不能保证归档，确认进程已停后可在状态命令末尾加`--pack`补打包，不覆盖已有ZIP。环境目录和缓存保留。

## 不考虑创新性，idea是否还有可行性

**有继续验证的理由，但H+C能否选时机尚未证实。** 本次没有新增科研结果，不改变已有18条结论：人工处理角度单位后A/B均4/9；B救回1对、伤害1对、双对3对、双错4对。B整体少15.87%追加token，是三个题九个相关前缀的描述性统计，不能保证总体质量不下降。

- 动作层有初步证据：case6/4203中C=56错、A=26错、B=28对；case8两个中后段位置双方正确而B少42%–46%生成token。也有case8早段B损害正确性、case6中段B更贵的负例。
- 信号选择尚未证明：获益位置H变化方向不统一，最晚位置B均更省token，长度规则可能已解释部分收益。两个case8节省位置C已正确，需要直接早停对照。C不是语义正确概率，归一化H下降也不一定意味着原始熵下降。
- 总成本尚未证明：离线分支耗时未包含真正在线监测、probe和完整共享推理成本。

下一步保持全部九点统一追加两个seed，检验救回、伤害和成本是否可重复。稳定后按预定、与答案无关的规则扩独立开发题，比较长度、H-only、C-only、H+C、原生机制与直接早停；test120封存。额外seed不增加独立题数，暂不训练探针或在九点上调阈值。

原研究协议保存在`PROTOCOL_NOTES_v3_zh.md`；其中旧“无需安装依赖”的操作说明由本文件的有条件恢复流程替代，研究协议不变。
