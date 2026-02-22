# Green Love 💚

A PyTorch plugin that benchmarks your training loop locally, estimates total training time, electricity cost, and CO₂ emissions, then compares against all [Crusoe Cloud](https://www.crusoe.ai/) GPU options — presented in a polished interactive HTML report.

The motivation for this project originated from a Crusoe workshop, where we got insight into the company's mission and how its infrastructure operates using 100% renewable energy sources. Our team found this concept both inspiring and highly practical — beneficial for users while also reducing environmental impact.

Our goal is to make this idea accessible to a wider audience by providing a tool that helps users **save time and resources** while making more **environmentally conscious decisions**.

> **This plugin uses data in very smart ways** — it samples only a tiny fraction of your dataset and epochs, then leverages linear scaling laws and statistical inference to accurately predict full-training costs, time, and carbon emissions without ever running the full workload.

---

## What It Does

Green Love takes your existing PyTorch training loop and, with just a few lines of code, provides:

- **Estimated total training time** with confidence intervals
- **Estimated electricity cost** based on your location
- **Estimated CO₂ emissions** during training
- **Crusoe Cloud GPU comparisons** — estimated time, cost, and CO₂ for each available GPU
- **Savings overview** — how much time, money, and carbon you save by switching to Crusoe Cloud
- **Interactive HTML dashboard** with all metrics visualized

---

## Recommended Models

This estimator works best with models whose training time scales **linearly** with both the number of data points and the number of epochs:

| Architecture | Why It Works |
|---|---|
| **Linear / Logistic Regression** | Fixed per-sample cost with SGD |
| **MLPs (Feedforward NNs)** | Constant cost per sample per epoch |
| **CNNs** (ResNet, VGG, EfficientNet, etc.) | Fixed conv ops per sample |
| **RNNs / LSTMs / GRUs** (fixed seq length) | Constant per-sample cost |
| **Transformers** (fixed seq length) | Fixed attention cost per sample |
| **Fine-tuning** (BERT, GPT, etc.) | Standard SGD/Adam iteration |

In general, this covers all models trained with **gradient descent** or **stochastic gradient descent (SGD)**, which accounts for the vast majority of modern machine learning models.

**Not recommended** for: variable-length Transformers, Graph Neural Networks, KNN/kernel methods, models with dynamic computation graphs.

---

## Installation

```bash
pip install green-love
```

Or install from source:

```bash
git clone <repo-url>
cd green-love
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NVIDIA GPU with drivers (for power monitoring via NVML)
- `jinja2`, `requests`, `nvidia-ml-py`

---

## Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from green_love import GreenLoveEstimator

# Your model, optimizer, criterion, and dataset
model = nn.Linear(784, 10).cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

dataset = TensorDataset(torch.randn(10000, 784), torch.randint(0, 10, (10000,)))

# Create estimator — pass model, optimizer, criterion, and dataset
estimator = GreenLoveEstimator(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataset=dataset,
    total_epochs=100,
    batch_size=64,
)

# Run estimation — automatically benchmarks at multiple sample sizes,
# generates an HTML report, and prompts whether to continue
results = estimator.estimate()

# If user chose to continue training locally
if estimator.user_chose_continue:
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(100):
        model.train()
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

estimator.cleanup()
```

After estimation completes:
1. An **HTML report** opens in your browser with full cost/time/CO₂ analysis
2. The **terminal** shows a summary and prompts: `Continue training locally? [y/N]`
3. Model and optimizer are **restored** to their pre-benchmark state

---

## Mathematical Foundations

### Core Problem: Estimating Training Time

We model training time as a random variable:

$$T(n, B_e)$$

where $n$ is the sample size and $B_e$ is the number of training epochs.

The expected training time is approximately **linear** in both variables — multiplying either by $k$ results in roughly $k$ times longer training. We validated this hypothesis empirically by training multiple model architectures on MNIST across a range of sample sizes.

### Empirical Evidence: Neural Networks

The plots below show that training time scales linearly with sample size, and that individual epoch durations stabilize after an initial warmup:

<p align="center">
  <img src="images/NN-tt.png" alt="Neural Network — training time vs sample size" width="600"/>
</p>

<p align="center">
  <em>Total training time grows linearly with dataset size — the foundation of our estimation model.</em>
</p>

<p align="center">
  <img src="images/NN-epoch.png" alt="Neural Network — per-epoch timing" width="600"/>
</p>

<p align="center">
  <em>Per-epoch timing: the first 1–3 epochs are slower (data loading & initialization overhead), then times stabilize. Occasional spikes are caused by RAM saturation triggering disk swaps.</em>
</p>

### Other Model Architectures

We repeated the experiment on SVMs, RNNs, and Random Forests. All exhibited the same linear scaling behavior:

<details>
<summary><strong>SVM results</strong></summary>
<p align="center">
  <img src="images/SVM-tt.png" alt="SVM — training time vs sample size" width="600"/>
  <br/>
  <img src="images/SVM-epoch.png" alt="SVM — per-epoch timing" width="600"/>
</p>
</details>

<details>
<summary><strong>RNN results</strong></summary>
<p align="center">
  <img src="images/RNN-tt.png" alt="RNN — training time vs sample size" width="600"/>
  <br/>
  <img src="images/RNN-epoch.png" alt="RNN — per-epoch timing" width="600"/>
</p>
</details>

<details>
<summary><strong>Random Forest results</strong></summary>
<p align="center">
  <img src="images/RF-tt.png" alt="Random Forest — training time vs sample size" width="600"/>
  <br/>
  <img src="images/RF-epoch.png" alt="Random Forest — per-epoch timing" width="600"/>
</p>
</details>

All models confirmed our two key assumptions: (1) linear time scaling with sample size, and (2) warmup epochs being consistently slower than steady-state epochs.

### The Training Time Formula

Based on these observations, we separate the first three (slower) epochs from the rest:

$$T(n, B_e) \approx e_1(n) + e_2(n) + e_3(n) + (B_e - 3) \cdot \bar{A}_e(n)$$

where $\bar{A}_e(n)$ is the average epoch duration after the third epoch. The warmup epochs are modeled independently because they include one-time costs (JIT compilation, CUDA context setup, data loader prefetching) that don't repeat.

### Linear Scaling with Sample Size

Since per-epoch time is proportional to the number of samples processed:

$$e_i(N) \approx e_i(n) \cdot \frac{N}{n}$$

Green Love trains on **multiple sample sizes** $k_1, k_2, \ldots, k_m$ and scales each measurement to the full dataset size $N$. This cross-sample averaging produces more robust estimates than a single sample size:

$$\bar{A}_e(N) = \text{mean}\!\left(e_{i,k} \cdot \frac{N}{k}\right) \quad \text{for all sample sizes } k \text{ and epochs } i > 3$$

Similarly, the three warmup epochs are estimated independently:

$$e_j(N) = \text{mean}\!\left(e_{j,k} \cdot \frac{N}{k}\right) \quad \text{for } j \in \{1, 2, 3\}$$

The standard deviation used for confidence intervals is also computed from these cross-sample scaled values:

$$\sigma(N) = \text{std}\!\left(e_{i,k} \cdot \frac{N}{k}\right)$$

### Confidence Intervals

The dataset of scaled epoch times $e_{i,k} \cdot N/k$ (for all epochs $i$ and all sample sizes $k$) is large — typically hundreds of data points. By the Central Limit Theorem, the normal distribution is sufficient and no t-distribution correction is needed.

The 95% confidence interval for total training time is:

$$\bar{A}_e(N) \cdot B_e \;\pm\; \sqrt{B_e} \cdot \sigma(N) \cdot z_{0.025}$$

where $z_{0.025} = 1.96$ is the standard normal critical value and $\sigma(N)$ is the standard deviation of $e_{i,k} \cdot N/k$ over all post-warmup epochs $i$ and all sample sizes $k$.

### Choosing a Representative Sample Size

The sample size must be large enough to be representative while still allowing fast estimation. Green Love automatically finds a good sample size using this algorithm:

Start with $(n, B_e) = (1000, 30)$ where $n$ is the initial sample size and $B_e$ is the number of exploration epochs.

**Algorithm (iterative):**

1. If first epoch takes longer than **0.3 seconds**, stop and set $n_{\text{new}} = n / 10$, then retry from step 1.
2. If the number of completed epochs is between **1 and 29** (i.e., at least one epoch completed but not all 30): $n_{\text{new}} = n \cdot \text{completed\_epochs}$. This is the **exit condition** — proceed to the multi-sample phase with $n_{\text{new}}$.
3. If **all 30 epochs complete** (training is too fast): $n_{\text{new}} = n \cdot \frac{10}{\text{total\_training\_time\_in\_seconds}}$, then retry from step 1.

The iteration repeats steps 1 and 3 until step 2 is reached, which produces the representative sample size.

### Sampling Strategy

Once the representative $n$ is found (with its epoch times already saved), Green Love collects data at additional sample sizes in two directions:

**Scaling up** — starting from $n$, multiply by 1.5 repeatedly:
$$n,\; \lfloor n \cdot 1.5 \rfloor,\; \lfloor n \cdot 1.5^2 \rfloor,\; \ldots$$
Train and log at each size. Repeat while total training time stays **below 20 seconds**. When training time **exceeds 20 seconds**, log that final run too, then stop scaling up.

**Scaling down** — starting from the original $n$, divide by 1.5 repeatedly:
$$\lfloor n / 1.5 \rfloor,\; \lfloor n / 1.5^2 \rfloor,\; \ldots$$
Train and log at each size. Repeat while total training time stays **above 1 second**. When training time **falls below 1 second**, log that final run too, then stop scaling down.

This typically produces **9–10 data points**, sufficient for reliable estimation.

### Maximum Runtime Bound

Assuming worst case where the upper limit (20 seconds) is included and scaled by 1.5:

$$30 + 20 + 13.3 + \ldots + 20 \cdot 1.5^{-8} < 90 \text{ seconds} \approx 1.5 \text{ minutes}$$

The entire multi-sample benchmark completes in **under 90 seconds**.

### Final Validation

Green Love verifies the linear relationship between sample size and training time using **Spearman rank correlation**. A high correlation ($\rho > 0.9$) confirms the linear scaling assumption holds for your specific model and dataset. If correlation is low, a warning is displayed.

---

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | **required** | PyTorch model to benchmark |
| `optimizer` | `Optimizer` | **required** | Optimizer used for training |
| `criterion` | `nn.Module` | **required** | Loss function |
| `train_dataset` | `Dataset` | **required** | Full training dataset |
| `total_epochs` | `int` | **required** | Total planned training epochs |
| `batch_size` | `int` | `64` | Batch size for training |
| `device` | `str` | auto-detect | Device (`"cuda"` or `"cpu"`) |
| `train_step` | `callable` | `None` | Custom training step function |
| `exploration_epochs` | `int` | `30` | Epochs per sample size during exploration |
| `warmup_epochs` | `int` | `3` | First N epochs treated as warmup |
| `target_benchmark_time` | `float` | `10.0` | Target seconds for representative sample |
| `initial_sample_size` | `int` | `1000` | Starting sample size for exploration |
| `single_epoch_budget` | `float` | `0.3` | Max seconds per epoch in exploration |
| `country_code` | `str` | auto-detect | ISO country code (e.g., `"US"`, `"DE"`) |
| `carbon_intensity` | `float` | auto-lookup | Grid carbon intensity (gCO₂/kWh) |
| `electricity_price` | `float` | auto-lookup | Electricity price ($/kWh) |
| `electricity_maps_api_key` | `str` | `None` | API key for live carbon intensity |
| `gpu_name` | `str` | auto-detect | GPU name override |
| `gpu_index` | `int` | `0` | NVIDIA GPU device index |
| `manual_tdp_watts` | `float` | `None` | Manual TDP if NVML unavailable |
| `manual_gpu_utilization` | `float` | `0.70` | GPU utilization fraction |
| `benchmark_task` | `str` | `None` | Task-specific benchmark comparison |
| `precision` | `str` | `"fp16"` | `"fp16"` or `"fp32"` benchmark table |
| `custom_speedup` | `dict` | `None` | Custom speedup ratios per Crusoe GPU |
| `report_dir` | `str` | `"./crusoe_reports"` | HTML report output directory |
| `auto_open_report` | `bool` | `True` | Auto-open report in browser |
| `power_poll_interval` | `float` | `1.0` | Seconds between power readings |

### Benchmark Task Options

For task-specific speed comparison, set `benchmark_task` to one of:
- `"resnet50"` — Image classification (best for CNN workloads)
- `"bert_base_squad"` — NLP fine-tuning (best for small Transformers)
- `"bert_large_squad"` — NLP fine-tuning (best for large Transformers)
- `"gnmt"` — Machine translation (best for seq2seq models)
- `"tacotron2"` — Text-to-speech
- `"waveglow"` — Audio generation

If not set, uses **geometric mean** across all tasks (recommended for general workloads).

### Custom Speedup Ratios

Override benchmark-based speed estimation for specific GPUs:

```python
estimator = GreenLoveEstimator(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataset=dataset,
    total_epochs=100,
    custom_speedup={
        "H100 HGX 80GB": 2.5,    # your measured speedup
        "A100 SXM 80GB": 1.8,
    }
)
```

### Manual Environment Configuration

```python
estimator = GreenLoveEstimator(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataset=dataset,
    total_epochs=100,
    country_code="DE",           # Germany
    carbon_intensity=332,        # gCO₂/kWh
    electricity_price=0.40,      # $/kWh
    manual_tdp_watts=350,        # if NVML unavailable
    manual_gpu_utilization=0.75, # estimated utilization
)
```

---

## How It Works

1. **Save State**: Model and optimizer states are deep-copied for restoration after benchmarking
2. **Find Representative Sample Size**: Starting from $n = 1000$, iteratively adjusts sample size until training 30 epochs takes ~10 seconds (see algorithm above)
3. **Multi-Sample Training**: From representative $n$, scales up (×1.5) until training > 20s, then scales down (÷1.5) until training < 1s, logging all runs
4. **Warmup Separation**: First 3 epochs of each run are modeled separately (they include initialization overhead)
5. **Cross-Sample Scaling**: Each measured epoch time is scaled to the full dataset: $e_{i,k} \cdot N/k$. The mean across all sample sizes and post-warmup epochs gives $\bar{A}_e(N)$
6. **Total Estimate**: $T(N, B_e) = e_1(N) + e_2(N) + e_3(N) + (B_e - 3) \cdot \bar{A}_e(N)$
7. **Confidence Intervals**: 95% CI via normal distribution: $T \pm \sqrt{B_e} \cdot \sigma(N) \cdot 1.96$
8. **Spearman Validation**: Verifies linear scaling assumption with rank correlation
9. **Restore State**: Model and optimizer are restored to pre-benchmark state
10. **Power Monitoring**: GPU power sampled via NVML → total energy (kWh)
11. **CO₂ & Cost**: Energy × grid carbon intensity / electricity price (offline tables for 70+ countries, or live via Electricity Maps API)
12. **Crusoe Comparison**: Speed ratios from Lambda Labs benchmarks (or TFLOPS+bandwidth weighted fallback) → estimated time, cost, CO₂ for each Crusoe GPU
13. **Report**: Interactive HTML dashboard with CSS-only visualizations
14. **Prompt**: Terminal prompt to continue training or stop

---

## GPU Speed Estimation

Green Love uses two strategies to estimate how fast Crusoe Cloud GPUs would train your model:

### Strategy 1: Lambda Labs Benchmark Ratios (preferred)
When your local GPU exists in the benchmark table, speed ratios are computed directly from published benchmark data across multiple tasks (ResNet-50, BERT, GNMT, etc.).

### Strategy 2: TFLOPS + Bandwidth Fallback
When your local GPU is NOT in the benchmark table, Green Love uses a weighted formula:

$$\text{speedup} = 0.7 \times \frac{\text{TFLOPS}_{\text{cloud}}}{\text{TFLOPS}_{\text{local}}} + 0.3 \times \frac{\text{BW}_{\text{cloud}}}{\text{BW}_{\text{local}}}$$

This covers 50+ GPU models including consumer cards (RTX 3050, 3060, etc.) that lack published benchmarks.

---

## Future Directions

Our architecture is designed to extend naturally — with more time, proper data, and a Crusoe partnership, these are realistic next steps:

- **Live Crusoe benchmarking via API**: Instead of relying on published benchmark ratios, we could run actual training on Crusoe GPUs through their API and feed real timing data back into our estimator. We built the infrastructure for this but were unable to test it without an API key.
- **Hardware-specific Crusoe profiles**: With access to detailed Crusoe hardware specs (interconnect topology, memory bandwidth under load, multi-GPU scaling factors), our estimation model could produce significantly more accurate cloud predictions.
- **Multi-framework support**: The linear-scaling model generalizes beyond PyTorch — extending to TensorFlow, scikit-learn, Keras, and other frameworks is a natural next step with additional engineering effort.
- **Full pipeline estimation**: Our approach can be extended to cover validation time, data preprocessing overhead, and CPU usage — completing the picture from raw data to trained model.

---

## Data Sources

| Data | Source | Last Updated |
|---|---|---|
| GPU Benchmarks | [Lambda Labs](https://github.com/lambdal/deeplearning-benchmark) | Dec 2025 |
| Crusoe Pricing | [crusoe.ai/cloud/pricing](https://www.crusoe.ai/cloud/pricing) | Dec 2025 |
| Carbon Intensity | Ember 2025, IEA 2023, EPA eGRID 2023 | 2025 |
| Electricity Prices | GlobalPetrolPrices Q4 2025 | 2025 |
| CO₂ Equivalences | EPA, IEA | 2024 |

---

## License

MIT
