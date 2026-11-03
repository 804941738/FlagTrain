# Ascend 910B Bias + GELU Forward 开发与验收

当前验收：**21 项三轮平均 0.826428×（PASS）**；完整正确性 **132 passed、6 skipped、0 failed**；
运行时及 Fast GELU 回归 **40 passed**。第一轮 0.705412× 的失败数据保留为历史基线，
当前结果与源码哈希见“第二轮性能优化”末尾。

## 接口和数学功能

`flag_train.megatron.bias_gelu_impl(input, bias)` 接收同设备、同 dtype 的
FP16、BF16 或 FP32 Tensor，输出广播后 shape 的连续 Tensor。支持标量、向量、
一般广播、非连续视图和空 Tensor；空输出不启动 kernel。输出不传播梯度，
`requires_grad=False`、`grad_fn=None`。参数类型错误抛出 TypeError，dtype、设备、
广播不合法抛出 ValueError。

计算为 `torch.nn.functional.gelu(input + bias, approximate="tanh")` 对应的近似式：

\[
z=\operatorname{round}_{dtype}(input+bias),\quad
 y=\operatorname{cast}_{dtype}\left[\frac z2(1+\tanh(0.79788456z(1+0.044715z^2)))\right].
\]

先将加法结果舍入回原 dtype，再以 FP32 计算激活，避免改变低精度加法语义。
例如 input=[0,1]、bias=0 时，输出约为 [0,0.841192]。
正无穷输出正无穷；负无穷和 NaN 按公式输出 NaN；有限极值保持饱和行为。

## 实际服务器与环境

服务器 `bm-ctyun-wq-zone1-d-910b4-64g-0-13`，8 张 Ascend910B4-1，
单卡 20 cube cores、40 vector cores，HBM 标称 64 GB。
Python 3.11.12，PyTorch 2.8.0+cpu（通过 torch_npu 使用 NPU），
torch_npu 2.8.0.post4.dev20260409，FlagTree 0.7.0+ascend3.5，
Triton 3.5.1，实际 CANN 9.0.0，驱动/npu-smi 25.2.0。
`flag_train.vendor_name=ascend`、`flag_train.device=npu`，NPU 可用。
原环境未设置 ASCEND_RT_VISIBLE_DEVICES，本次使用默认逻辑设备 npu:0，未更改设备可见性。
未升级或修改全局软件栈。日志均在 `/home/secure/lizhenghao/results/bias_gelu_910b/`。

## 失败复现与根因

历史记录为 77 passed、22 failed、6 skipped；本服务器当前基线提交
`a9c185edd7a731f8cf1f5d9adbe3a3cb53be55db` 实际完整复现得到
**48 passed、51 failed、6 skipped**（baseline.log）。不能将历史 22 项逐条等同于本次 51 项。

失败按根因处理，而非放宽容差：

1. Grid/coreDim 上限：首个错误位于 FP16 (4096,4096)，256 元素/tile
   导致 coreDim=65536，超过 65535。之后异步报错污染同进程结果，许多普通 shape、
   广播、非连续和极值用例也出现未写入输出。基线失败分布：常规 shape 17、
   广播 15、非连续 3、数值及空输入 3、有限极值 3、Forward-only 8、
   大 hidden 1、非法参数 1。
2. FP64 非法参数测试：清除 grid 错误后仅剩该项失败。
   torch_npu 的 `x.to(torch.float64)` 发出警告并实际生成 FP32，原测试误将合法参数
   当作非法参数。仅 Ascend 的这一步改用 CPU 上真正的 FP64 Tensor，保留原预期
   ValueError；算子先检查 dtype 再检查设备，拒绝路径真实执行，没有增加跳过。
3. 数值精度：独立进程运行原 kernel 的普通小 shape 和特殊值，三种 dtype 的
   最大绝对误差分别为 FP16 6.103515625e-5、BF16 1.52587890625e-5、
   FP32 4.76837158203125e-7；Inf/NaN、有限极值行为正确。
   没有发现需要修改 tanh、舍入公式或 CPU reference 的独立数值错误。
4. Megatron：当前没有安装 `megatron`，保留原依赖条件的 3 项 skip，未安装替代栈。
   仓库 `baseline/megatron/bias_gelu_impl.py:native()` 已提供禁用 JIT fuser 后
   重载参考模块的方案；当前缺少包，无法进入该路径，不能声称完成 Megatron 对照。
   其余 3 项 skip 为原 CUDA stream 测试。

第一轮修复实际结果 **98 passed、1 failed、6 skipped**，见 persistent.log；
剩余非法参数错误独立复现见 bad_arguments.log。原 kernel 的独立数值隔离见 isolated_baseline.log。

## 调度实现

继续使用一个 `@libentry()` + Triton kernel，标量、共享向量及逐元素 bias 复用
`_bias_gelu_tile` 内联计算，不引入额外激活 kernel、同步或中间 Tensor。
Ascend 使用 num_warps=4、num_stages=1，BLOCK 按元素量选择 128 至 16384
之间的二次幂（约 N/40 的下一二次幂，封顶 16384），最多启动 CORE_NUM 个
program，本设备为 40。每个 program 处理 `pid, pid+grid, pid+2*grid, ...`
的逻辑 tile，运行时循环避免按总元素数静态展开。

共享向量 bias 在 H>=128、H 为二次幂且 H<=BLOCK 时，一次加载连续 bias 行并在 tile
内广播到多行，input/output 仍使用连续平坦索引。H<128 保留原平坦读取；其他向量 bias 使用按行分块：
每行 `ceil(H/BLOCK)` 个 tile，tile 编号除以该值确定行号，余数确定起始列，
列 mask 排除行尾 padding。这样避免 `offsets % H` 向量地址被当前编译器降成
逐元素离散 gather。标量和逐元素 bias 保持平坦 tile。所有分支使用宽整数索引，
N 和列边界 mask 保护尾部；仅当完整 tile 有效时以 constexpr 消除 mask。
逻辑 tile 按 `tile % grid` 唯一分配，不遗漏或重复处理输出。
本次选择仓库 Ascend fast_gelu 已采用的持久循环方案，而非二维 grid。
实际错误来自 runtime 的 coreDim 参数，单纯拆分维度不足以证明满足物理启动限制；
固定物理 program 数既规避该上限，也避免每 256 元素一个 program 的调度负担。
真实大 shape 和尾块验收验证此方案，而非仅依赖推导。

其他平台仍使用原 BLOCK=256、num_warps=4 和一维普通调度，constexpr 分支直接
内联一个 tile，不经过 Ascend 循环。未改 public API、布局处理、公式或误差容限。
未在本机重新验收其他硬件，不把先前其他平台成绩冒充本次验证；尤其平头哥旧结果
属于系统 Triton 3.5.0，不能视为 FlagTree 0.7.0rc1 指定环境通过。

## 正确性命令和范围

```bash
python -m pytest tests/megatron/test_bias_gelu_impl.py \
  tests/megatron/test_bias_gelu_triton_regression.py --ref cpu -q -rs --tb=short
```

覆盖普通 shape、FP16/BF16/FP32、CPU 高精度 reference、dtype-rounded 加法、
广播及反向广播、非连续和非对齐视图、空输入、标量、Inf/NaN、有限极值、
非法参数、Forward-only 及标准 LibEntry 调度。
新增 12 项 Ascend 大输入回归：三种 dtype 分别测试 `65535*256-1`、`65535*256`、
`65535*256+1`、`2^26+1` 元素；非零随机输入配标量 bias，逐段全量 CPU 比较，
既检测漏 tile、重复 tile，也验证最后一个不整除元素。原有超大向量 bias 用例保留。

## 性能测量方法

```bash
python tools/bench_bias_gelu_910b.py --rounds 3 \
  --output /home/secure/lizhenghao/results/bias_gelu_910b/forward
```

严格复用 H20 脚本的 7 个 shape × 3 种 dtype，共 21 项，每项 3 轮。
真实 NPU PyTorch Eager `F.gelu(x+bias, approximate="tanh")` 对比公共算子接口，
输入来自相同随机分布，无梯度。每次计时前预热 25 次并同步（包含首次 JIT）。
使用 torch.npu.Event(enable_timing=True)，11 批，每批连续调用 100 次，
end Event 同步后读取设备毫秒数，换算为单次微秒并取批次中位数。
轮次交替基线/算子测量顺序；不使用 CUDA do_bench 或 CPU 秒表。
该计时包含公共接口分配和主机提交间隙对设备时间线的影响，属于 Eager 接口吞吐延迟，
不是设备 kernel 指令本体的纯耗时，微小 shape 尤其受 launch 影响。

每轮原始结果写入 JSON/CSV；汇总 CSV 保存每项三轮延迟和 speedup 中位数。
验收为 21 项 speedup 中位数的算术平均 >=0.80；同时报告几何平均和最低值，
不以总延迟比替代。报告记录 commit、算子源码 SHA256 和软件环境。

## 性能优化记录与正式结果

初始 1024-lane 持久 kernel 的 NPU profiler 设备记录（FP32，4096×4096）
约 3738 µs；不是单纯 CPU 提交问题。编译器 ttadapter 中可以看到 bias 的
`DiscreteMemAccess` / `ExtractedLoadOrStore` 标量循环。连续 bias 加载实验
降至约 186 µs，随后采用多行 tile 复用。相关日志和 trace：

- `diagnostic_block1024.log`：中止的初始性能诊断，不作为完整验收。
- `profile_block1024/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`、`trace_view.json`：真实设备 profiling。
- `tune.log`：1024/4096/8192 lane、40/512/2048 program 扫描，单纯调度配置无法修复 gather。
- `tune_math.log`：索引宽度、exp 和 tanh 变体隔离，gather 主导时无收益。
- `diagnostic_rowtiles.json`：4096-lane 按行 tile，完整 21 项但仅一轮，平均 0.657806×，不作正式通过。
- `tune_row.log`：program 数、num_stages 的对照，没有明显收益。
- `row_group_large.log`、`group16384.log`：FP32 16384×4096 从约 655 µs
  改善至约 591 / 558 µs；保留 16384 lane 版本。
- `group32768.log`：更大 tile 编译失败，未采用。
- `row_hoist_large.log`：bias 循环外提升未改善，未采用。
- `constexpr_n*.log`：N 编译常量实验约 137 / 560 µs，未显示足够收益，未采用。
- `profile_cpu.log`：cProfile 显示小输入成本主要为 LibEntry、Triton runner/launcher、
  分配和设备上下文；cProfile 会增加主机开销，不能当作正式延迟数据。

未采用改变低精度数学语义、跳过参数检查、替换 Eager 基线、捕获 NPU Graph 或
改变计时口径的方法。性能结论仅适用于当前普通 Eager 公共接口。

第一轮适配正式结果（21 项，各 3 轮，每项 speedup 中位数）：

| Shape | dtype | Eager 中位延迟 µs | FlagTrain 中位延迟 µs | Speedup 中位数 |
|---|---|---:|---:|---:|
| (64, 64) | float16 | 33.247 | 138.804 | 0.242409× |
| (64, 64) | bfloat16 | 32.779 | 138.907 | 0.235975× |
| (64, 64) | float32 | 33.501 | 139.010 | 0.240999× |
| (1024, 1024) | float16 | 33.629 | 140.774 | 0.238884× |
| (1024, 1024) | bfloat16 | 30.998 | 138.893 | 0.234203× |
| (1024, 1024) | float32 | 33.362 | 139.457 | 0.239228× |
| (4096, 4096) | float16 | 114.635 | 142.938 | 0.806991× |
| (4096, 4096) | bfloat16 | 129.926 | 144.974 | 0.896197× |
| (4096, 4096) | float32 | 209.576 | 142.490 | 1.470807× |
| (16384, 4096) | float16 | 490.286 | 567.548 | 0.863439× |
| (16384, 4096) | bfloat16 | 512.513 | 584.252 | 0.877037× |
| (16384, 4096) | float32 | 889.094 | 560.855 | 1.585248× |
| (2048, 11008) | float16 | 174.393 | 286.490 | 0.608725× |
| (2048, 11008) | bfloat16 | 189.236 | 295.493 | 0.640750× |
| (2048, 11008) | float32 | 311.789 | 251.466 | 1.238844× |
| (16, 128, 8192) | float16 | 115.254 | 144.676 | 0.796639× |
| (16, 128, 8192) | bfloat16 | 123.489 | 146.175 | 0.845577× |
| (16, 128, 8192) | float32 | 206.454 | 145.314 | 1.420359× |
| (256, 32768) | float16 | 56.011 | 140.716 | 0.398145× |
| (256, 32768) | bfloat16 | 65.027 | 139.853 | 0.465797× |
| (256, 32768) | float32 | 67.770 | 145.232 | 0.467408× |

算术平均 **0.705412×**，几何平均 **0.576392×**，最低 **0.234203×**。
**第一轮基线性能 FAIL（0.705412×）。用户随后授权推送阶段进展，对应提交 b6eccb9；本轮后续优化与验收见下文。**

报告目录 `/home/secure/lizhenghao/results/bias_gelu_910b/`：
`forward.json` 保存 63 条原始轮次数据及中位数、环境、SHA256、验收判断；
`forward.csv` 为原始数据，`forward_medians.csv` 为 21 项中位数；
`forward.log` 为完整测量日志。原始试验、编译产物、profile 不加入 Git。
`forward_before_small_hidden_fix*` 保留此前三轮记录（算术平均 0.719804×），
它对应极值修复前源码，不能替代最终验收。

第一轮测量的算子 SHA256：`35fc7bc0f1d54ce0364e710ba55e3c6405058326013356cb8ba309e138f5a1ed`，对应 b6eccb9 中的文件；本轮源码哈希以新报告为准。

## 第一轮兼容性、限制与后续方向

仅 Ascend 启用持久循环、按行/多行 tile、mask 特化和单缓冲；其他平台保持原
256-lane 普通调度。计算函数在同一个源文件内共享，保留标准 LibEntry 调度。
其他硬件没有在本服务器重新验证，不能保证其性能结果不变。

第一轮的限制是普通公共接口的小 shape 提交开销，以及大 shape 低精度路径相对 Eager
仍不够快；当时 21 项均值不达标，后续本轮已达到均值门槛。后续需在独立任务中研究 Ascend LibEntry/launcher
开销、向量流水线及 UB 数据搬运，并继续对相同 Eager 和相同计时方法进行验证。
本次未改动通用调度基础设施来扩大风险面；没有 backward。
Megatron 缺失和 CUDA stream 的 6 项原始条件跳过仍然存在。

分支 `platform/ascend-910b-bias-gelu-forward` 从指定 a9c185e 基线创建。
原未跟踪 `.lock` 文件保留，不暂存或清理用户工作。用户后续明确授权在性能
未达标的情况下提交当前进展并推送此独立分支，供继续优化和审阅；不推送其他分支，
不将本次进展标记为性能验收通过。原始性能报告中的 commit 为测量时的基线 HEAD，
对应的未提交算子源码由报告 SHA256 精确标识；此次提交未修改已测量的算子源码。

性能优化后的首次完整回归曾出现 3 项有限极值失败（optimized_correctness.log：
108 passed、3 failed、6 skipped）。隔离到 H=2 的多行 tile 路径，其正有限极大值
错误变为 0。将 H<128 限制回原平坦加载路径后，三种 dtype 的该回归全部通过
（extreme_fixed.log：3 passed）；没有修改数学公式或预期值。
另外新增三种 dtype 在 H=128、4096、11008 上的 9 项有限极值回归，验证按行 tile
分支。最终验证和性能报告以修复后的记录为准。

## 第一轮正确性验收

最终完整命令执行结果：**120 passed、6 skipped、0 failed，272.37 秒**。
日志：`/home/secure/lizhenghao/results/bias_gelu_910b/final_correctness_v2.log`。
相较原 105 项收集用例增加 21 项实际执行测试（12 项大输入、9 项行分块极值），
跳过仍只有原先 Megatron 缺失和 CUDA stream 的 6 项。未放宽容差，未删除失败用例。

## 第二轮性能优化（2026-11-04）

基线为已推送的 `b6eccb92b39543272770e3de86dacbb112d49e13`，继续在同一
`platform/ascend-910b-bias-gelu-forward` 分支开发。个人目录未找到可确认属于
此前毕昇杯的 FA 项目、历史 Tiling 或实验记录，因此没有声称复用了历史 FA
实现，也没有读取或修改其他用户项目。仅借鉴“主机/设备拆分 profiling →
资源瓶颈 → 单变量候选 → 正确性 → 全量验收”的方法；没有 Cube/MMAD、
split dQ/dK/dV、workspace 或 NPU Graph 路径。

### 固定开销隔离

原基线的 FP16 隔离数据（µs）如下。Event 沿用正式脚本的批次方法；host 列
为 `perf_counter` 测量 1000 次提交的平均值，计时结束后再同步，因此是提交成本，
不是替代正式设备计时。预分配/固定 stream 只用于诊断，不用于正式验收。

| Shape | 阶段 | Event µs | 主机提交 µs |
|---|---|---:|---:|
| 64×64 | 公共接口 | 128.893 | 128.693 |
| 64×64 | 预分配 LibEntry | 62.825 | 62.485 |
| 64×64 | 编译后 runner | 32.402 | 31.706 |
| 64×64 | runner 固定 stream | 23.581 | 23.326 |
| 64×64 | launcher 固定 stream | 18.311 | 17.648 |
| 64×64 | Tensor 分配 | 7.397 | 6.853 |
| 64×64 | device context | 4.186 | 3.847 |
| 16384×4096 | 公共接口 | 566.197 | 130.795 |
| 16384×4096 | 预分配 LibEntry | 565.864 | 68.171 |
| 16384×4096 | 编译后 runner | 565.147 | 37.592 |

小输入 kernel 设备本体约 4–6 µs，1024×1024 约 12.5 µs；大输入约 569 µs。
小 shape 的 Event 批次包含真实的主机提交间隙，不能将它解释为 129 µs 的设备计算。
大 shape 的 Event 与设备本体接近，必须进一步优化计算。

Level1 PipeUtilization：大 FP16 融合 kernel 的 AIV Vector ratio 均值 0.8444，
MTE2 0.0894、MTE3 0.0560，表明该配置主要受 Vector 运算限制，而非简单的 HBM
带宽上限。比例可以重叠，不能直接相加为 100%。Cube 没有参与这个逐元素算子。
真实设备证据为 `pipes_baseline/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`。

### 候选与取舍

以下文件均在 `/home/secure/lizhenghao/results/bias_gelu_910b/optimization_v2/`。
诊断数字不冒充完整性能验收；所有正式结果仍使用原 Eager 基线和原 Event 方法。

| 候选/证据文件 | 实测结果 | 判断 |
|---|---|---|
| `isolate.log`、`cpu_*.txt`、`profile_{64,1024,16384}` | 上述三种 shape 的主机与设备隔离 | 确立双重瓶颈 |
| `properties.log` | 单次 Tensor property 约 0.24–0.35 µs，分配约 6.82 µs | 参数检查没有足够大收益，不删检查 |
| `python_tiling.json` | 64×64 约 110–112 µs | 用 Python 整数代替 host 上的 FlagTree JIT 整数辅助函数 |
| `direct_key.json` | 64×64 约 108–110 µs | Ascend 直接构造等价 specialization key，避免 storage/指针签名 memo |
| `plain_jit_hit.json` | 64×64 约 98–99 µs | 普通 JIT 命中路径不再建立 tuner 字典 |
| `explicit_stream.json`、`cached_launch.json` | Python Stream 候选约 122–142 µs，反而退化 | 拒绝构造 Stream 对象；保留后端 NoWait 查询语义 |
| `nowait_launch.json` | 64×64 约 83–84 µs | 保留真实当前 stream 查询、NPULauncher 和动态 hooks |
| `runtime_all21.json` | 完整 21 项，仅一轮，均值 0.761191× | 诊断，不能判定 PASS |
| `math_tanh.log` | 三个重点 shape 的 9 项无明显整体收益 | 不替换 tanh |
| `hidden4096.log` | H=11008 的 FP16/BF16 约 242/249 µs；FP32 约 277 µs 有回退 | 仅低精度、高 padding 配置缩小 tile |
| `native_add.log` | 大 FP16/BF16 初步约 512/542 µs；H=11008 约 241/256 µs | BF16 后续被精确舍入测试拒绝，仅保留 FP16 |
| `native_h4096.log` | H=11008 初步约 218/221 µs，但包含不可接受的 BF16 native add；错误缩小二次幂 hidden 会回退 | 不直接合入；排除跨行复用的二次幂 hidden，BF16 保留原舍入 |
| `candidate_correctness.log` | 25 passed、1 failed、3 skipped（另 34 deselected），BF16 half-ULP tie 不符 | 真实拒绝 BF16 native add，不删测试或放宽容差 |
| `candidate_correctness_v2.log` | 26 passed、3 skipped（另 34 deselected） | 修复后相关测试通过 |
| `forward_candidate.json` | 完整 21×3：0.795666×，三轮约 0.7960/0.7960/0.7974× | FAIL，继续优化，不靠重复测试碰运气 |
| `horner.log` | 大 FP16/BF16 约 496/566 µs；FP32 约 570 µs 回退 | 同一多项式的等价重排仅用于 Ascend 低精度 |

### 最终实现与兼容性

1. **Host tiling**：Ascend 的 `ceildiv` 和下一二次幂由普通 Python 整数运算计算。
   实测 FlagTree 的 `next_power_of_2(cdiv(...))` 约 5.98 µs，而等价 Python 约
   0.16 µs。非 Ascend 的原 256-lane 调度保留。
2. **LibEntry**：`LibEntry.run` 仍是入口。仅 Ascend、普通 JIT、Triton 3.3–3.6
   启用简化的缓存命中参数绑定；缺参数、callable grid、autotuner/heuristics 等
   仍走原路径。直接 key 使用原 `_spec_arg`、descriptor 和 DNS 规则，保持 dtype、
   对齐、整数宽度与 constexpr specialization；缓存仍由 LibEntry 按设备管理。
   每次重新绑定实际 Tensor 与输出，绝不缓存输出或跳过运算。
3. **Launcher**：仅 Triton 3.5 且存在 `torch_npu._C._npu_getCurrentRawStreamNoWait`
   时，复用 `CompiledKernel` 的同一 NPULauncher 调用约定，减少 runner closure 和
   backend registry 重复查询；每次读取当前设备的当前 stream 和动态 enter/exit hooks。
   仍经 NPULauncher，保留 profiling/debug 的处理。不缓存 stream；能力不满足时回退
   原编译后 runner。未改安装的 Triton/torch_npu/CANN。
4. **低精度加法**：仅 Ascend H>=128 的 FP16 用原 dtype 加法再转 FP32 激活，
   避免冗余转换。BF16 必须保留显式 FP32 加法 → BF16 舍入 → FP32 激活屏障；
   原生 BF16 候选在当前编译器下未保留 tie 行为，已拒绝。其他平台保持原运算路径。
5. **尾块**：低精度行分块若 padding 超过有效列的 1/8，且原 BLOCK>4096，使用
   4096-lane tile；能够在 tile 内复用多行的二次幂 H 不适用此规则，FP32 保持原配置。
   H=11008 从 16384-lane 的 48.8% 额外 padding 降至三个 4096-lane tile 的 11.6%。
6. **等价多项式**：低精度 Ascend 路径将 tanh 参数从
   `a*z*(1+b*z*z)` 重排为 `(a+(a*b)*z*z)*z`，a=0.79788456、b=0.044715，
   减少一次 FP32 Vector 乘法，未引入新的 GELU 近似。FP32 及非 Ascend 保持原表达式。
   浮点重排仍使用原测试容差；低精度加法舍入发生在激活之前，没有移动舍入屏障。

运行时回归额外覆盖新指针、dtype/非对齐、constexpr、默认/关键字参数、DNS 整数宽度、
非默认 stream、callable grid、heuristics、launch metadata 和 enter/exit hooks；
非 Ascend 不启用此命中路径。其他硬件未在本机实测，不能声称完成其性能验收。
新增两个 dtype 的 half-ULP tie 测试采用严格相等，明确区分未舍入 FP32 加法；
新增 subnormal 测试也采用严格相等，未放宽任何已有容差。

复现工具：

```bash
python tools/profile_bias_gelu_910b.py --output /home/secure/lizhenghao/results/bias_gelu_910b/optimization_v2/optimized_profile
python -m pytest tests/test_libentry_ascend.py tests/megatron/test_fast_gelu.py --ref cpu -q -rs --tb=short
python -m pytest tests/megatron/test_bias_gelu_impl.py tests/megatron/test_bias_gelu_triton_regression.py --ref cpu -q -rs --tb=short
python tools/bench_bias_gelu_910b.py --rounds 3 --output /home/secure/lizhenghao/results/bias_gelu_910b/optimization_v2/forward_final
```

正式脚本保留算子 SHA256，补充 LibEntry/benchmark 源码哈希及 dirty 标记，不修改测量区间、
预热、轮数、输入分布或 Eager 基线。所有正式数据为普通公共接口，未使用 NPU Graph。

### 主机追加优化前的正确性与设备 profiling

主机追加优化前的完整正确性命令：**124 passed、6 skipped、0 failed，338.98 秒**，
见 `optimization_v2/final_correctness.log`。保留原 Megatron 缺失 3 项、CUDA stream
3 项条件跳过；原 120 项实际通过用例之外新增 4 项严格舍入/subnormal 测试。
运行时及其他算子回归：**40 passed，61.65 秒**，见 `runtime_smoke_final.log`；
包括 10 项 LibEntry 回归和 30 项 Fast GELU 回归，唯一 warning 是原 Fast GELU
测试在 Megatron 未安装时使用 CPU PyTorch 组合参考，没有新增 skip。

最终诊断工具真实执行成功，证据 `optimized_profile/breakdown.json`、`cpu_*.txt`，
及三个 shape 对应的 `profile_*/ASCEND_PROFILER_OUTPUT/{kernel_details.csv,trace_view.json}`。
小 shape 公共接口 Event 从 128.893 µs 降至 83.478 µs，预分配 LibEntry 从
62.825 µs 降至 41.137 µs；大 FP16 公共接口从 566.197 µs 降至 496.399 µs。
同一大 shape 的 Level1 设备 kernel 均值为 496.017 µs，Vector ratio 0.8260，
MTE2/MTE3 为 0.1008/0.0620，仍以 Vector 计算为主。这些是诊断，正式验收
以下文的新进程 21 项三轮 Event 数据为准，未将固定 stream 或预分配结果混入验收。

### 主机热路径追加优化

`forward_final.json` 是上一候选的完整三轮记录，算术平均 **0.791605×**，
各轮约 0.7944/0.7911/0.7910×，仍为 FAIL。设备侧已改善，但该轮小输入公共接口
约 91–93 µs，继续消除主机热路径成本，没有调整计时或挑选某轮结果。

`allocation.log` 实测：原 `torch.empty(shape,dtype,device)` 约 7.78 µs、
缓存 options 约 7.10 µs、`input.new_empty(shape)` 约 5.88 µs。
最终 Ascend 输出采用 `input.new_empty`，输出仍为广播 shape、原 dtype/device、
连续且不传播梯度；非 Ascend 仍用原 factory。单次调用缓存 dtype/device 元数据，
所有检查及错误顺序保留，不缓存输入验证结果或输出。

仅当当前 NPU 已等于输入设备时使用空 context，避免无用的 device guard；
设备不一致时及所有非 Ascend 路径仍用原 `torch_device_fn.device`。
新增三种 dtype 的公共 NPU stream 用例和强制进入 guard 的回退测试，不占用其他 NPU。
`host_refinement_small.json` 的小 shape 诊断为 77.43–78.75 µs；只有单轮三个用例，
不能作为正式验收。最终完整三轮结果见下面的新报告，不覆盖任何失败轮次的数据。

### 最终正式性能验收

7 个 shape × 3 种 dtype × 3 轮完整运行，63 条原始记录。每项先取三轮 speedup 中位数，
再对 21 项作算术平均，保持原公共接口和计时方法。结果如下：

| Shape | dtype | Eager 中位延迟 µs | FlagTrain 中位延迟 µs | Speedup 中位数 |
|---|---|---:|---:|---:|
| (64, 64) | float16 | 34.077 | 82.278 | 0.409067× |
| (64, 64) | bfloat16 | 34.520 | 84.823 | 0.405881× |
| (64, 64) | float32 | 34.782 | 83.108 | 0.409971× |
| (1024, 1024) | float16 | 32.427 | 84.824 | 0.388205× |
| (1024, 1024) | bfloat16 | 32.639 | 86.353 | 0.391459× |
| (1024, 1024) | float32 | 34.473 | 86.208 | 0.394993× |
| (4096, 4096) | float16 | 108.250 | 123.022 | 0.879728× |
| (4096, 4096) | bfloat16 | 129.067 | 140.105 | 0.920672× |
| (4096, 4096) | float32 | 212.531 | 143.336 | 1.482748× |
| (16384, 4096) | float16 | 479.861 | 495.917 | 0.967546× |
| (16384, 4096) | bfloat16 | 509.920 | 566.611 | 0.900269× |
| (16384, 4096) | float32 | 889.153 | 561.233 | 1.585781× |
| (2048, 11008) | float16 | 180.522 | 210.636 | 0.857768× |
| (2048, 11008) | bfloat16 | 195.120 | 233.274 | 0.835668× |
| (2048, 11008) | float32 | 315.317 | 252.924 | 1.246623× |
| (16, 128, 8192) | float16 | 107.064 | 124.029 | 0.863216× |
| (16, 128, 8192) | bfloat16 | 116.316 | 141.511 | 0.821675× |
| (16, 128, 8192) | float32 | 209.138 | 144.531 | 1.447485× |
| (256, 32768) | float16 | 55.933 | 87.116 | 0.642826× |
| (256, 32768) | bfloat16 | 64.998 | 85.585 | 0.759907× |
| (256, 32768) | float32 | 62.917 | 86.808 | 0.743497× |

**正式性能 PASS：算术平均 0.826428× >=0.80×**；
几何平均 0.748153×，最低 0.388205×。
三轮各自算术平均：0.824795× / 0.824626× / 0.828252×。
相对第一轮基线 0.705412×，均值提升约 17.16%。

正式证据位于 `optimization_v2/forward_host_refined.json`、`.csv`、`_medians.csv`、`.log`。
这不是 kernel-only 或 Graph 成绩，仍包含输入检查、分配、LibEntry、当前 stream 查询、
CPU 提交间隙和真实设备执行。小 shape 虽明显改善，仍慢于 Eager，不能将均值通过
表述为每一项都快于 Eager。最低值也是 21 项三轮 speedup 中位数的最低值。

测量时 Git HEAD 为 b6eccb9，`measured_sources_dirty=true`，实际验收源码由以下
三个哈希精确标识；提交时不修改这些源码，因此不将父提交误写为已测量的最终代码：

- 算子 SHA256：`600ecd23f02e95b97ff372a1a8ff64c00e62c2dfdf4f8f436404b5fba14805ef`。

- LibEntry SHA256：`e1cf2f2e6d3136b76b81e11e918caf2e2921ce25c6aef11ec3c9283e6b46ee0e`。

- 计时脚本 SHA256：`11f8277ec57328e10c6cbdd0ae0fa12ef29ff88fa9522986eb82d03a47e023db`。

### 已知限制与风险

Triton 3.5 的直接已编译调用约定属于版本相关的优化，能力检查及 fallback 必须保留；
升级 FlagTree 后应复测 stream、hooks、debug 和 specialization。其他平台只执行原路径，
但没有真实硬件复测。小 shape 仍受 Python/allocator/launcher 限制；大低精度仍以
Vector 计算为主；本轮没有扩大到 NPU Graph、backward 或通用调度框架重构。
原始日志、报告和 profiling 产物均留在个人结果目录，不加入提交。

### 最终完整正确性确认与交付

最终提交源码的完整命令再次执行，结果 **132 passed、6 skipped、0 failed，84.35 秒**，
见 `optimization_v2/accepted_correctness.log`。在原 120 个通过用例上新增 8 个严格
加法舍入/subnormal 用例（包含标量、向量、逐元素 bias），以及 4 个公共 stream/guard
用例。跳过仍仅为原 Megatron 缺失和 CUDA stream 的 6 项；没有删除失败用例。

`host_refined_correctness.log` 的 127 passed、1 failed 是新 guard 测试的监测器错误：
把 `torch.npu.device` 类替换为普通函数，破坏了 torch_npu 内部的 isinstance 检查。
改为继承原类的记录器后，`guard_test_fixed.log` 为 1 passed；随后上述完整命令全部通过。
没有为了修复该测试而改变算子或期望值。最后的三种源码 SHA256 均已与正式报告核对一致。

本轮按仓库 DCO 规则使用已配置的 Git 身份签署，提交主题：
`perf(ascend-910b): optimize bias_gelu forward performance`。
仅将有效源码、测试、工具和本文档提交到现有 910B 分支，保留 b6eccb9 历史；
原 `.lock` 文件、原始报告、日志和编译缓存均不提交。普通推送，不使用 force。
