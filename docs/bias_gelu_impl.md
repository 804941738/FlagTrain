# Bias + GELU（Forward-only）

`flag_train.megatron.bias_gelu_impl(input, bias)` 对应
`torch.nn.functional.gelu(input + bias, approximate="tanh")` 的前向近似。
输入须为同一活动加速器、同一 FP16/BF16/FP32 dtype 的 Tensor，支持标量、向量和
一般广播、非连续视图及空 Tensor。输出连续，shape 为广播结果，dtype/device 与输入
一致，`requires_grad=False`、`grad_fn=None`，不传播梯度。
类型错误抛出 `TypeError`，dtype、设备或广播不合法抛出 `ValueError`。

## 数学与精度

```text
z = round_to_input_dtype(input + bias)
u = 0.79788456 * z * (1 + 0.044715 * z * z)
y = cast_to_input_dtype(0.5 * z * (1 + tanh(u)))
```

加法先舍入回原 dtype，激活以 FP32 计算，不能融合成未舍入的 FP32 加法。
例如 `input=[0,1]`、`bias=0`，输出约为 `[0,0.841192]`。
正无穷输出正无穷；负无穷和 NaN 按表达式输出 NaN；空输出不启动 kernel。
正确性使用 CPU 高精度公式参考，既有 `(rtol, atol)` 分别为 FP16 `(3e-3,3e-3)`、
BF16 `(2e-2,2e-2)`、FP32 `(2e-5,2e-5)`；低精度 half-ULP 舍入和 subnormal
另有严格相等测试，覆盖标量、向量及逐元素 bias。

## 实现与平台范围

一个通用源码文件共享 scalar/vector/elementwise bias 的 tile 计算，使用
`@libentry()`、`@triton.jit` 和宽整数索引。向量 bias 或共享行不物化整块广播；
一般广播和非连续输入按需转成连续布局，可能发生复制。

- **非 Ascend 路径**：BLOCK=256、num_warps=4、普通一维 grid，使用后端
  `tl_extra_shim.tanh`，保留通用 GPU/PPU 路径及原 dtype-rounded 加法。
- **Ascend 910B 路径**：BLOCK 在 128–16384 间按 shape 选择，最多启动向量核数量
  的 program（历史 910B4-1 为 40），以循环覆盖全部逻辑 tile，避免 coreDim 65535
  限制。共享 bias 使用连续列/多行复用，低精度高 padding 的行采用较小 tile。
  BF16 保留显式 FP32→BF16→FP32 舍入屏障；仅经验证的 FP16 路径用原 dtype 加法。
  通过恒等式 `0.5*z*(1+tanh(u)) = z*(1/(1+exp(-2*u)))` 合并激活末端，u 仍含
  原 cubic 项，不是线性 sigmoid 的另一种 GELU 近似。
- **运行时**：Ascend 普通 JIT 的缓存命中精简参数绑定，specialization、当前设备、
  当前 stream、launch metadata/hooks 及 cache-miss 保留。Triton 3.5 的直接已编译
  调用检查 NoWait 查询能力，缺少能力时回退 runner。算子仍经过标准 LibEntry；
  非 Ascend 使用原缓存和 launcher 分支。没有缓存输出或使用 NPU Graph。

当前上游厂商实现位于 `runtime/backend/_<vendor>/<component>/`，不再有旧 `ops/`
层。本算子不新增厂商模块，Ascend 条件保护在通用源码内；公共导出位于
`flag_train.megatron.__all__`，并登记于 `conf/operators.yaml`。

## 正确性与下一轮硬件验收

**本文历史结果对应整理前源码。上游同步、格式化和 PR 整理后的候选仅做静态检查，
尚未完成新的 GPU/NPU 正式验收，不能沿用旧成绩宣称已通过。**

```bash
python -m pytest tests/megatron/test_bias_gelu_impl.py \
  tests/megatron/test_bias_gelu_triton_regression.py --ref cpu -q -rs --tb=short
python -m pytest tests/test_libentry_ascend.py tests/megatron/test_fast_gelu.py \
  --ref cpu -q -rs --tb=short
```

覆盖广播、非连续/非对齐、空输入、特殊值、极值、Forward-only、标准 LibEntry、
非默认 NPU stream、grid 边界及超大尾块。910B 的原 6 项条件跳过为未安装 Megatron
的 3 项对照和 CUDA stream 的 3 项；没有为了通过测试增加人为跳过。
下一阶段还须按 CI 执行默认设备 reference、`--ref=cpu --quick`，并在 NVIDIA 及
目标 GPU/PPU 上验证通用路径。本机静态检查不能替代其他硬件测试。

## 两种性能口径与复现

标准 benchmark 对照**实际安装的 Megatron Eager**：`baseline/.../native()` 禁用
Megatron JIT fuser，必要时重载融合模块。依赖不可用会报错，不冒充 PyTorch fallback。
它沿用 GenericBenchmark 的平台计时、模式和汇总，不等同下面的专门 Eager 验收。

```bash
RESULTS_DIR=/path/to/personal/results/bias_gelu
mkdir -p "$RESULTS_DIR"
python -m pytest benchmark/megatron/test_bias_gelu_impl.py -m bias_gelu_impl \
  --level core --mode kernel --record json --output "$RESULTS_DIR/megatron.json" -vs
```

保留两个最小复现脚本，对照均为未经修改的
`F.gelu(x+bias, approximate="tanh")`，被测为真实公共 API：

```bash
# 分别在相应设备环境运行；不在本轮 PR 整理中执行。
python tools/bench_bias_gelu_h20.py --rounds 3 --output "$RESULTS_DIR/eager_h20"
python tools/bench_bias_gelu_910b.py --rounds 3 --output "$RESULTS_DIR/eager_910b"
```

二者固定 7 种 shape × 3 种 dtype：`(64,64)`、`(1024,1024)`、`(4096,4096)`、
`(16384,4096)`、`(2048,11008)`、`(16,128,8192)`、`(256,32768)`。
每项取三轮 speedup 中位数，再计算 21 项算术平均，正式门槛为 0.80×；几何平均和
最低值仅为附加指标。输出原始 JSON/CSV 与中位数 CSV，记录版本、Git commit、源码
SHA256；若测量时 dirty，须按哈希识别实际源码，不把父提交当作已测代码。

H20 使用 CUDA/Triton `do_bench`（25 ms warmup、100 ms rep、中位数）；910B 使用
NPU Event（25 次预热并同步，11 批各 100 次，end Event 同步后取批次中位数），
包含分配、LibEntry 和主机提交间隙。脚本时钟与统计口径保留，用于重现已有成绩，
不能将标准 benchmark 的另一种 baseline、kernel-only 或 Graph 结果混入比较。

## 整理前历史证据与限制

| 平台/历史版本 | 正确性 | PyTorch Eager 平均 speedup |
|---|---|---:|
| NVIDIA H20（通用实现历史记录） | 105 passed | 1.738× |
| 沐曦 MC550（移交记录） | 99 passed，6 skipped | 0.9766× |
| 天数 BI-V150（移交记录） | 102 passed，3 skipped | 1.6654× |
| 海光 BW1000（移交记录） | 102 passed，3 skipped | 1.2890× |
| 平头哥 PPU（移交记录，系统 Triton 3.5.0） | 99 passed，6 skipped | 1.9352× |
| Ascend 910B4-1，`0266fe8` | 138 passed，6 skipped | 1.024703× |

跨 GPU/PPU 数字为移交的历史记录，本服务器未复验；PPU 的系统 Triton 3.5.0 成绩
不能写作 FlagTree 0.7.0rc1 指定环境已通过。910B 同版本另有 40 项 LibEntry/Fast GELU
回归通过，两份完整三轮均值为 1.012630× 与 1.024703×，没有挑选或删除不利轮次。
其历史软件栈为 PyTorch 2.8.0+cpu + torch_npu 2.8.0.post4.dev20260409、
CANN 9.0.0、FlagTree 0.7.0+ascend3.5 / Triton 3.5.1；最低中位 speedup 0.427003×。

小 shape 仍慢于 Eager；8M FP32 的历史单项 speedup 略回落，原始结果全部保留。
Triton/torch_npu 升级需重新验证 NoWait、stream、hooks 和精度。源码格式化也会改变
JIT cache/hash，整理后必须重新完成上述正确性及 21 项三轮性能，不能保证性能不变。
完整开发方案、历史失败分类、调参、profiling、21 项表格和原始报告保存在个人交付
资料目录，仓库只保留本摘要。截图未伪造；详细资料以真实日志和 JSON/CSV 为证据。
