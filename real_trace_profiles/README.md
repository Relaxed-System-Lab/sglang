# Curated HexAgenT Profile Assets

Only validated profile/result archives belong in this directory. Debug runs,
revision-0 fallback results, wrong-SLO reports and intermediate topology sweeps
must remain outside version control.

| Asset | Contents | Integrity |
| --- | --- | --- |
| `h100_real_trace_profiles_final.tar.gz` | H100 TP4 Llama-3.1-70B and Qwen3-235B-A22B FP8; ShareGPT 100, BFCL 100 and HotpotQA 20 workflows; raw rows, TTFT/decode profiles and service tables | SHA256 `530aad92ba35cc74207bf2ce146a5142b3c8d2dbe1cc43dd5bbb8bad644e21ab` |
| `llama8b_h100_2p2d_exclusive_profile_20260802.jsonl` | H100 TP1 Llama-8B FP8 model-matched exclusive profile used by the 2P+2D study; 456 calls / 30 workflows | See adjacent `.sha256` |
| `a100_hexgent_final_valid_assets_20260810.tar.gz` | A100 TP1 Llama-8B BF16 exclusive profiles for all three traces; fixed replay DAGs; final ShareGPT/BFCL telemetry; A100 Mooncake NVLink calibration | SHA256 `9732893bbf171a12e130941f02cb866b92e76ab5c9d5ab06815c5f0858206a4b` |
| `a100_hotpotqa_hexgent_valid_20260811.tar.gz` | Two final paired HotpotQA runs with per-mode warmup and isolated runtime IDs | SHA256 `879bb6a0e14b1460f0f31be14edc70d513576dad43227655755fd8a8ec8c53ae` |

Every profile row includes `input_len`, `output_len`, `ttft_ms` and
`total_latency_ms`. The H100 final archive additionally keeps original response
records and merged prefill/decode service tables.

Do not compare SLO ratios across assets unless model, GPU, TP, dtype and cache
configuration match. Token lengths may be replayed on another platform, but its
latency must be profiled again.
