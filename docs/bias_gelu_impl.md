# Bias + GELU（Forward-only）

`flag_train.megatron.bias_gelu_impl(input, bias)` 实现 Bias 加法与 tanh 近似 GELU
的融合前向，对应 Megatron 的 `bias_gelu_impl` 前向接口。

## 接口与数值约定

两个参数须为同一活动加速器、同一 FP16/BF16/FP32 dtype 的 Tensor。支持标量、
向量 Bias、一般 PyTorch 广播、非连续视图和空 Tensor。输出连续，shape 为广播结果，
dtype/device 与输入一致；`requires_grad=False`、`grad_fn=None`，不支持 Backward。
类型错误抛出 `TypeError`，非法 dtype、设备或广播抛出 `ValueError`。

```text
z = round_to_input_dtype(input + bias)
u = 0.79788456 * z * (1 + 0.044715 * z * z)
y = cast_to_input_dtype(0.5 * z * (1 + tanh(u)))
```

加 Bias 后先舍入回原 dtype，激活使用 FP32；不能省略该舍入屏障，将加法与激活
融合为未舍入的 FP32 中间值。一般广播和非连续输入可能需要复制为连续布局。
空输出不启动 Kernel；正无穷输出正无穷，负无穷和 NaN 按表达式输出 NaN。
有限极值、正负零、次正规数和加法溢出沿用 dtype 舍入及后端浮点语义。

## GPU 与 Ascend 实现

一个通用 Triton 源码共享 scalar/vector/elementwise Bias 的 tile 计算，通过
`@libentry()` 启动。向量 Bias 或共享行不物化整块广播。

- **通用 GPU/PPU**：H20 使用 BLOCK=256、num_warps=4、普通一维 Grid，
  PERSISTENT=False、ROWWISE=False，调用后端 `tl_extra_shim.tanh`。
- **Ascend 910B**：按 shape 在 128–16384 间选择 BLOCK，Persistent 调度将物理
  program 数限制在向量核数量内，再循环覆盖全部逻辑 tile。连续列读取、多行复用
  和低精度行 padding 控制减少 Bias 访问开销。BF16 保留显式
  FP32→BF16→FP32 舍入；经验证的 FP16 路径使用原 dtype 加法。
- **Ascend 运行时**：普通 JIT 缓存命中精简参数绑定，保留 specialization、当前
  设备/Stream、cache miss 和 launch metadata/hooks。Triton 3.5 的已编译 Launcher
  在能力检查通过时使用 NoWait Stream 查询；缺少能力时回退标准 runner。
  非 Ascend 不启用这些缓存和 Launcher 快速路径。

Ascend 的 ROWWISE 分支通过
`0.5*z*(1+tanh(u)) = z*(1/(1+exp(-2*u)))` 合并 GELU 末端，`u` 仍含原三次项。
该恒等式在实数上成立，但浮点运算顺序、舍入、饱和及次正规数行为可能不同，
不能宣称逐位相等。非 Ascend 仍计算原 tanh 表达式。

## 正确性与精度

以下完整验收均对应同步前提交
`3973e99f91d4097dc036dc938e2e941fdc3fb99b`。同步上游后的 H20 回归单独记录在
个人结果目录；本机不能重新证明同步后 910B 的设备行为。

| 验证项目 | NVIDIA H20 | Ascend 910B |
|---|---|---|
| CPU reference | 105 passed，39 skipped | 141 passed，3 skipped |
| 默认设备 reference | 105 passed，39 skipped | 141 passed，3 skipped |
| CPU quick | 47 passed，39 skipped | 本摘要未提供 |
| Megatron 原生直接对照 | 21 passed | 21 passed |
| Fast GELU / LibEntry | 30 passed，10 skipped | 40 passed |
| 数值专项 | 额外17项通过 | exp/tanh 等价变换专项通过 |

910B 数据来自该设备的正式验收移交记录；H20 日志来自本机。跳过项为平台条件，
不能计为通过。覆盖广播、尾块、非连续/非对齐、空输入、极值、NaN/Inf、
Forward-only、标准 LibEntry、非默认 Stream 和超大 Grid 边界。

| dtype | dtype 舍入后数学参考 (rtol, atol) | Megatron 原生对照 (rtol, atol) |
|---|---|---|
| FP16 | (3e-3, 3e-3) | (5e-3, 1e-2) |
| BF16 | (2e-2, 2e-2) | (4e-2, 8e-2) |
| FP32 | (2e-5, 2e-5) | (2e-5, 2e-5) |

数学参考先执行原 dtype 加法，再上转精度计算；CPU reference 使用独立高精度路径。
Megatron 原生低精度表达式存在逐步舍入，故与融合 FP32 激活按上述容差比较。
H20 正式21项最大绝对误差依次为 FP16 0.001953125、BF16 0.015625、
FP32 4.768372e-7；容差失败及 NaN/Inf 差异均为0。参考为0而实际为微小非零时，
相对误差可为无穷；零附近的相对误差或 ULP 数不能单独代表计算错误。
本算子不承诺所有输入上固定 ULP 上界，也不承诺跨平台次正规数逐位一致。
half-ULP 加法舍入及次正规数另有专项用例，不能将这些有限用例外推为全域逐位相等。

## 正式 Megatron 性能

基线为实际安装的
`megatron.core.fusions.fused_bias_gelu.bias_gelu_impl`，即原生
`GeLUFunction.apply`。仓库 `baseline/megatron/bias_gelu_impl.py::native()` 关闭
JIT fuser，必要时重载已导入融合模块；依赖错误会抛出，不使用公式 fallback。
没有 `torch.compile` 或 CUDA/NPU Graph。

**下表均为同步前 `3973e99` 的结果，不是新 HEAD 的全平台复验。**

| 平台 | Megatron Core | 完整确认平均加速比 |
|---|---|---:|
| NVIDIA H20 | 0.17.1（PyPI） | 5.186078× |
| Ascend 910B | 0.17.1+0.2.3（FlagOS） | 5.734440× |

固定7个 shape × FP16/BF16/FP32 × 3轮，共63条记录：
`(64,64)`、`(1024,1024)`、`(4096,4096)`、`(16384,4096)`、
`(2048,11008)`、`(16,128,8192)`、`(256,32768)`。
每项先取三轮 Speedup（Megatron 延迟 / FlagTrain 延迟）的中位数，再对21项计算
算术平均。H20 确认实验几何平均5.086083×、最低单项4.216024×；首份平均5.189451×，
因单次小 shape 波动追加完整确认，两份原始记录均保留。910B 首份平均5.759483×，
确认结果如表。历史0.80×为既有算术平均验收阈值，不代表性能不退化的逐项保证。

H20 使用原 Triton `do_bench`（warmup=25 ms、rep=100 ms、median）；910B 使用
NPU Event（25次预热并同步，11批各100次，end Event 同步后取批次中位数）。
双方相同输入/Bias、设备和 dtype，交替顺序，禁用梯度，保留真实公共接口调用。
平台的计时工具和软件栈不同，不能直接比较绝对性能，也不能外推为训练模型整体加速。

标准 GenericBenchmark 各覆盖21项，单独列示：

| 平台 | Kernel 算术平均 | Operator 算术平均 |
|---|---:|---:|
| NVIDIA H20 | 5.456916× | 4.557153× |
| Ascend 910B | 6.762998× | 5.697379×（复测） |

H20 Kernel 使用 Triton Event 区间，Operator 使用 CPU 端调用循环与边界同步；
910B 标准框架使用其平台计时实现。不同模式不能混合平均。输出中的
“Torch Latency”列绑定真实 Megatron native，而非 `F.gelu`。

## 平台状态与软件环境

H20、910B 已完成上述同步前版本的正式 Megatron 基线验证。其余平台仅保留历史事实：

| 平台 | 历史正确性 | 历史 PyTorch Eager 平均加速比 |
|---|---|---:|
| 沐曦 MC550（移交记录） | 99 passed，6 skipped | 0.9766× |
| 天数 BI-V150（移交记录） | 102 passed，3 skipped | 1.6654× |
| 海光 BW1000（移交记录） | 102 passed，3 skipped | 1.2890× |
| 平头哥 PPU（移交记录） | 99 passed，6 skipped | 1.9352× |

这些数字不是 Megatron 正式成绩，也不是同步后本机复验。PPU 记录使用系统
Triton 3.5.0，不能写成 FlagTree 0.7.0rc1 指定环境已通过。

H20：Python 3.12.3，Driver 570.158.01，CUDA Runtime 13.1，
torch 2.11.0a0+eb65b36914.nv26.02，Triton 3.6.0
（pytorch-triton 3.6.0+git9844da95.nv26.2），未单独安装 FlagTree。
910B 移交环境记录为 PyTorch 2.8.0+cpu、torch_npu 2.8.0.post4.dev20260409、
CANN 9.0.0、FlagTree 0.7.0+ascend3.5 / Triton 3.5.1；本次 H20 不重新核验其环境。

## 复现与限制

复用厂商兼容的 CUDA/NPU 软件，不能为安装 Megatron 替换 torch、Triton 或驱动。
如已有兼容0.17.1构建则复用，否则在个人 venv 内最小安装：

```bash
python -m venv --system-site-packages /path/to/personal/venv
source /path/to/personal/venv/bin/activate
python -m pip install --no-deps 'megatron-core==0.17.1' \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
export PYTHONPATH="$PWD/src:$PWD"
```

H20 实际使用 PyPI 0.17.1；因系统 nvidia-resiliency-ext 0.5.0 不满足 Megatron 导入
要求，仅在该 venv 以 `--no-deps` 安装0.6.0。记录实际构建、导入路径和依赖变化，
不强求910B本地版本后缀，不伪装为同一wheel。

```bash
python -m pytest tests/megatron/test_bias_gelu_impl.py \
  tests/megatron/test_bias_gelu_triton_regression.py --ref cpu -q -rs --tb=short
# 默认设备参考：去掉 --ref cpu；CI快速验证：添加 --quick。
python -m pytest tests/megatron/test_fast_gelu.py tests/test_libentry_ascend.py \
  --ref cpu -q -rs --tb=short

RESULT_DIR=/path/to/personal/results
mkdir -p "$RESULT_DIR"
python -m pytest benchmark/megatron/test_bias_gelu_impl.py -m bias_gelu_impl \
  --level core --mode kernel --record json \
  --output "$RESULT_DIR/megatron_kernel.json" -vs
# 独立运行operator模式：改为 --mode operator，并使用新的输出文件。
```

若正确性测试也需限定原生 Eager 对照，可先在同一进程调用仓库 `native()`，再调用
`pytest.main([...])`，参数与上述命令一致。标准 Benchmark 自行调用 `native()`。
不使用 `--skip-native`、`--ref-as-candidate`、`--dummy-candidate` 或动态 override
生成正式性能成绩。

仓库两个 `tools/bench_bias_gelu_{h20,910b}.py` 仍是 PyTorch Eager 历史复现脚本，
不能直接将其输出标记为 Megatron 成绩。正式三轮实验在个人结果目录复制脚本，
仅将 baseline 换成以下原生入口，保留7个 shape、3种 dtype、平台计时和统计方式：

```python
from baseline.megatron.bias_gelu_impl import native
import flag_train

megatron_op = native()
flagtrain_op = flag_train.megatron.bias_gelu_impl
# 在 torch.no_grad() 中对相同 x、bias 交替测量真实公开调用。
```

个人脚本输出全部63条原始 JSON/CSV、中位数 CSV、版本、Git HEAD、源码及
Megatron融合模块/测量脚本SHA256；不得挑选最优轮次。H20 完整证据目录名为
`results/bias_gelu_h20/megatron_baseline_validation/`，含个人
`bench_megatron_h20.py`、首份及确认数据，均不纳入提交。
同步后的日志另存个人目录；核心哈希不变不等于已经重新完成全平台实测。
Triton/torch_npu 升级或公共运行时行为变化后，仍需验证 Stream、hooks、NoWait 和精度。
