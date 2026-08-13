# CAGRA 单卡/多卡瓶颈验证

使用 `sift-locality` 环境。所有路径与 CAGRA 参数集中在 `cagra_config.json`；默认配置使用 SIFT100M 的 `.bbin`、`.bvecs`、`uint8` 与 `sqeuclidean`。值为 `null` 的图和搜索参数会保留 cuVS 默认值。命令行同名选项优先于配置，适合 batch/merge 扫描。

服务器尚未安装依赖时，先创建独立环境。默认环境名为 `cagra-bench`；可用第一个参数改名。对于多卡 RTX 5090（Blackwell，SM 12.0），脚本默认安装 CUDA 12.9；CUDA 12.8 是 Blackwell 的最低支持版本。创建完成后会输出每张可见卡的名称和 compute capability。 [CUDA 架构兼容矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)

```bash
bash create_cagra_env.sh
# 默认创建 cagra-bench；例如改名：bash create_cagra_env.sh sift-locality
```

创建实验前先复制并修改配置，例如：

```bash
cp cagra_config.json my_cagra_config.json
# 在 my_cagra_config.json 中设置 paths、graph_degree、build_algo 等字段
```

先生成 Top-10 真值：

```bash
conda run -n sift-locality python make_top10_groundtruth.py --config my_cagra_config.json
```

## 1. 构建完整单卡图

完整图必须能装入一张 GPU，才可进行“同图”强扩展实验。

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n sift-locality python build_graph.py \
  --config my_cagra_config.json
```

`build_graph.py` 首次运行建图并保存；后续运行加载同一 index。传入 `--rebuild` 才会覆盖重建。`search_seconds` 和 `qps` 只包含同步后的 CAGRA 搜索，不含建图、加载、warmup 或输出。

## 2. 同图多卡搜索扩展性

`same-index` 会把上一步的同一份完整图复制到全部可见 GPU。它用于定位多卡查询分发、同步和传输开销，不改变图结构。

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n sift-locality python cagra_mg_benchmark.py \
  --config my_cagra_config.json --mode same-index \
  --metrics metrics/same_index_2gpu.json
```

以 `CUDA_VISIBLE_DEVICES=0`、`0,1`、`0,1,2,3` 和相应的 8 卡集合重复运行。所有 GPU 数都调用同一多卡 API，便于比较强扩展性。

## 3. 原生分片多卡实验

`sharded` 会按连续行将数据分到各卡，各卡独立构建局部 CAGRA 图；搜索时 cuVS 合并局部候选为全局 Top-10。它能容纳跨卡数据，但图与单卡全局图不同。

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n sift-locality python cagra_mg_benchmark.py \
  --config my_cagra_config.json --mode sharded \
  --metrics metrics/sharded_2gpu_default.json
```

针对同一 GPU 集合，扫描批大小与归并方式：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n sift-locality python cagra_mg_benchmark.py \
  --config my_cagra_config.json --mode sharded \
  --n-rows-per-batch 1000 --merge-mode tree_merge \
  --metrics metrics/sharded_2gpu_b1000_tree.json

CUDA_VISIBLE_DEVICES=0,1 conda run -n sift-locality python cagra_mg_benchmark.py \
  --config my_cagra_config.json --mode sharded \
  --n-rows-per-batch 1000 --merge-mode merge_on_root_rank \
  --metrics metrics/sharded_2gpu_b1000_root.json
```

每个 JSON 包含设备、模式、index 准备时间、warmup、实际搜索时间、QPS、参数和 Recall@10。不要把 sharded 的 QPS 当作同图加速比；将其与 `same-index` 结果并列，才能观察分片建图、查询广播和候选归并的额外成本。

## Smoke test

格式测试不需要 GPU：

```bash
conda run -n sift-locality python smoke_test.py
```

在 GPU 节点上可额外运行极小的 CAGRA 建图/保存/加载/单卡分发测试；有至少两张卡时还会测试 sharded 路径：

```bash
conda run -n sift-locality python smoke_test.py --with-gpu
```
