# bias_gelu_impl 集成交付说明

本轮完成通用 Triton 实现与工程接线，等待总工程师审核及正式硬件验收。
当前实现没有 H20 或国产 GPU/NPU 的正式 FlagTrain 实测结果。

## 接口与语义

`flag_train.megatron.bias_gelu_impl(input, bias)` 返回与 `input + bias`
广播结果同 Shape、dtype、device 的张量。两个参数必须是同设备、同 dtype 的
FP16/BF16/FP32 加速器张量，支持标量和任意合法 PyTorch 广播；不支持混合 dtype。
非连续输入和 Bias 会按需要连续化。空结果不启动 Kernel，但仍执行参数检查。
例如全零 input 和 bias 的输出为零，输入的逐元素导数为 0.5。

令 `z = input + bias`，前向为
`y = z/2 * (1 + tanh(0.79788456*z*(1+0.044715*z*z)))`。
反向逐元素导数为
`0.5*(1+t) + 0.5*z*(1-t*t)*(0.79788456+0.1070322243*z*z)`，其中 `t` 是前向的 tanh。
两个参数都接收逐元素梯度，由 Autograd 按各自广播维归约；空广播的 Bias 梯度为零。
仅提供一阶梯度，二阶梯度通过 `once_differentiable` 显式禁用。

官方语义依据：[Megatron fused_bias_gelu.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_bias_gelu.py)。
Bias Add 先舍入到输入 dtype，激活和导数在 FP32 计算，输出/梯度转换回输入 dtype。
这与 eager Megatron 在低精度下多次舍入的结果并非逐位相同，测试分别比较高精度公式参考
和真实安装的 Megatron。FP16/BF16 对官方参考使用独立的容差，需要总工程师审核。
反向在绝对值超过 10 时使用 tanh 饱和值，避免巨大有限输入出现 `0*inf`；
非有限 z 的梯度保留为 NaN。极端有限幅值下与直接低精度多项式溢出的行为可能不同。

## 修改文件

- `src/flag_train/megatron/bias_gelu_impl.py`：前向/反向 Kernel、Autograd 与参数检查。
- `src/flag_train/megatron/__init__.py`：导出公开算子，沿用顶层平台注册机制。
- `tests/megatron/test_bias_gelu_impl.py`：三种 dtype、小中大 Shape、非对齐维度、
  标量/向量/多维广播、输入自身被广播、空张量、非连续输入/Bias/上游梯度、
  单侧梯度、异常参数，以及已安装 Megatron 的比较。
- `benchmark/megatron/test_bias_gelu_impl.py`：GenericBenchmark 前向与反向。
- `baseline/megatron/bias_gelu_impl.py`：自包含的原生 Megatron eager 基线采集定义。
- 本文：设计、验证边界及硬件复现命令。

## Kernel 与平台方案

一维 grid、mask 保护尾块；Bias 分为标量、hidden 向量和完整连续广播结果三种寻址模式。
通用实现使用 `libentry`、`tle.program_id`、`tl_extra_shim.tanh` 和平台 device context。
广播 Bias 为向量时不复制成完整输入规模；一般广播和非连续输入可能额外分配内存。
反向重算 z/tanh，不保存完整激活；Bias 归约交给 Torch Autograd。

Forward 为 `BLOCK=256, num_warps=4`。Backward 的 FP32 固定 256；
FP16/BF16 的元素数 `<2^20` 用 256，`[2^20,2^23)` 用 1024，`>=2^23` 用 2048，
均为 4 warps。总工程师已核对 H20 原始 V1 源码并确认这两个分界；本次修正与其一致。
这仅确认分档来源，正式 FlagTrain 环境的硬件正确性和性能仍待验证。

GPU/NPU 共用同一实现，未添加平台特化。国产平台均待测试，特别需要验证 tanh shim、
编译参数、较大 grid 的调度限制和 Broadcast/Autograd 归约。没有实测证据前不添加专属优化。

Benchmark 与 baseline 都要求安装 Megatron；不回退到 PyTorch 公式基线。
加载 eager 基线会禁用 Megatron jit_fuser，必要时重新加载 fusion 模块；该影响限于基线工具进程。
反向采用 GenericBenchmark 的 Autograd 计时（包括 Bias 归约，前向建立图不计入该项），
因此与独立 V1 的缓存 z ATen microbenchmark 口径不同。
此前独立 H20 V1 的正确性和加速比仅是用户提供的历史信息，不能作为本次提交的验收成绩。

## 本地已执行检查

- Python AST/语法检查，flake8、isort、Black、Git whitespace 和导出检查。
- 使用系统已有 Torch 2.5.1+cu121，在 CPU 上以替代启动器检查真实 Autograd 包装代码：
  138 个 Shape/dtype/梯度组合通过，另检查三种 dtype 的 tiling 分界。
  此项验证的是 host 数据准备与 Autograd 广播，不执行也不验证 Triton Kernel。
- 本次阈值修正后，通过 AST 隔离执行生产分档函数和新增回归测试：三种 dtype
  共 27 个分档边界用例通过，覆盖 `2^20`、`2^23` 两侧与边界，以及原 `2^24` 分界。
  此项不依赖加速器，也不代表完整 pytest 或 Kernel 验证。
- pytest collection 已尝试，因 Windows 缺少 `tests/conftest.py` 所需 `fcntl` 阻断。
  本地缺少 Triton；未安装 CUDA/厂商工具链，未执行 GPU Kernel。
- H20/国产 GPU/NPU 正确性、编译、性能、显存峰值均待验证。

## H20 执行命令

在独立的 Linux FlagTrain checkout 和已准备好的官方兼容环境中运行，不覆盖独立 V1 实验目录。
以下命令中的 git fetch/switch 假设该 checkout 工作树干净，origin 为用户 Fork。

```bash
git fetch origin feat/bias-gelu-impl
git switch feat/bias-gelu-impl
git pull --ff-only origin feat/bias-gelu-impl
mkdir -p bias_gelu_results
git rev-parse HEAD | tee bias_gelu_results/commit.txt
nvidia-smi > bias_gelu_results/nvidia-smi.txt
python -m pip show torch triton flagtree megatron-core > bias_gelu_results/packages.txt
python -m pip install --no-deps -e .

python -m pytest tests/megatron/test_bias_gelu_impl.py --quick -q
python -m pytest tests/megatron/test_bias_gelu_impl.py -q --record json \
  --output bias_gelu_results/accuracy.json
python -m pytest tests/megatron/test_bias_gelu_impl.py --quick --ref cpu -q
python -m pytest benchmark/megatron/test_bias_gelu_impl.py -m bias_gelu_impl \
  --level core --mode kernel --record json \
  --output bias_gelu_results/benchmark.json -vs
```

必须一并回传 commit.txt、硬件/软件环境、Shape/dtype、正确性及前向/反向性能结果；
若官方 Megatron 比较被 skip，须说明原因，不能将其算作通过。
测试结果不要自动提交到源码分支。按三种 dtype 分析误差，先验证正确性再评估 tiling。
若显存不足，应保留 OOM 的 Shape/dtype 和环境记录，不得将缺失项算作性能通过。

## 待总工程师决策

1. 审核 FP32 内部运算、低精度官方参考容差及极端有限输入的饱和策略。
2. H20 正式集成实测复核原始 V1 分档；观察 Bias 归约在训练反向中的占比，必要时另行优化。
3. 审核仅一阶梯度、同 dtype 的接口约束；需要扩展时新增对应验证。
4. 国产平台逐一验收；大 grid 或编译参数问题必须用真实失败记录驱动修复。
