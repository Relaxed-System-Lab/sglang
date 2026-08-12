# Mooncake 分段与带宽口径说明

## 本轮问题

解释为什么 A100 上单块 `cudaMemcpyPeerAsync` 达到 260.7 GiB/s，而 Mooncake native
CUDA-event copy sum 只有约 189 GiB/s，以及为什么 250 MiB 校准 payload 使用 50 个
5 MiB entries。

## 由代码确认的事实

- `cuda_peer_copy_probe.cu` 对连续 256 MiB buffer 只执行一次
  `cudaMemcpyPeerAsync`。
- `measure_mooncake_native_copy.py` 将一次 batch 构造成 source/destination/length 数组；
  A100 校准显式使用 50 个 5 MiB entries。
- Mooncake v0.3.9 intra-node transport 当前逐 entry 执行同步
  `cudaMemcpy(..., cudaMemcpyDefault)`。HexAgenT patch 对每个真实 copy 分别记录 CUDA
  events，再汇总 `cuda_copy_sum_ms`。
- SGLang 的 MHA transfer path 按 layer 组织 K/V pointers。Llama 3.1 8B 为 32 层，连续
  token run 通常至少形成 64 个 K/V-layer regions；不连续 token slots 会产生更多 regions。

## 数值解释

250 MiB 按 direct 260.7 GiB/s 换算约为 0.936ms；Mooncake 多 entry copy sum 为
1.293ms，差值约 0.357ms，即每个 5 MiB entry 约 7.1us。差值混合了小 copy 的固定
提交/启动成本、短 copy 带宽利用不足和逐 entry telemetry 扰动。189 GiB/s 因此表示当前
Mooncake 多 entry copy 形状的有效 payload 吞吐，不是 NVLink hardware peak。

最大 ShareGPT payload 254.625 MiB 对应约 2,037 tokens。每个 K 或 V layer region 为
`2 KiB/token x 2,037 = 约 3.98 MiB`，64 个 regions 合计 254.625 MiB。`50 x 5 MiB`
只是在相似 entry 尺寸和总量上的受控校准，不是生产固定 layout。

## 后续校准

补测 1、32、64 entries，并从 production telemetry 记录每个请求真实的 entry count、length
分布和 contiguous-run 数量。不要用单块 direct P2P 带宽替代 Mooncake 多 entry copy 模型，
也不要把 Python API wall-clock 当作 raw-copy 带宽。
