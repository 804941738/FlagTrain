# Ascend 910B Bias + GELU Forward 开发与验收

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

最终源码正式结果（21 项，各 3 轮，每项 speedup 中位数）：

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
**正式性能验收 FAIL，未达到 0.80×，性能适配任务尚未完成。用户后续明确要求将当前进展提交并推送至独立 910B 分支；推送不代表性能验收通过。**

报告目录 `/home/secure/lizhenghao/results/bias_gelu_910b/`：
`forward.json` 保存 63 条原始轮次数据及中位数、环境、SHA256、验收判断；
`forward.csv` 为原始数据，`forward_medians.csv` 为 21 项中位数；
`forward.log` 为完整测量日志。原始试验、编译产物、profile 不加入 Git。
`forward_before_small_hidden_fix*` 保留此前三轮记录（算术平均 0.719804×），
它对应极值修复前源码，不能替代最终验收。

最终测量的算子 SHA256：`35fc7bc0f1d54ce0364e710ba55e3c6405058326013356cb8ba309e138f5a1ed`，与当前文件一致。

## 兼容性、限制与后续方向

仅 Ascend 启用持久循环、按行/多行 tile、mask 特化和单缓冲；其他平台保持原
256-lane 普通调度。计算函数在同一个源文件内共享，保留标准 LibEntry 调度。
其他硬件没有在本服务器重新验证，不能保证其性能结果不变。

目前限制是普通公共接口的小 shape 提交开销，以及大 shape 低精度路径相对 Eager
仍不够快；21 项均值不达标。后续需在独立任务中研究 Ascend LibEntry/launcher
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

## 最终正确性验收

最终完整命令执行结果：**120 passed、6 skipped、0 failed，272.37 秒**。
日志：`/home/secure/lizhenghao/results/bias_gelu_910b/final_correctness_v2.log`。
相较原 105 项收集用例增加 21 项实际执行测试（12 项大输入、9 项行分块极值），
跳过仍只有原先 Megatron 缺失和 CUDA stream 的 6 项。未放宽容差，未删除失败用例。
