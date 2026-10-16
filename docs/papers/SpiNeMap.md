---
title: 【论文精读】SpiNeMap： Mapping Spiking Neural Networks to Neuromorphic Hardware
tags:
  - SNN
  - Mapping
  - Neuromorphic Hardware
  - SpiNeMap
  - ISI
---

## 1.任务背景与挑战

### 1.1 SNN Mapping

SNN 训练完成后，还不能直接运行在神经形态芯片上，需要把：

神经元 / 突触 → 芯片上的 Crossbar / Core

进行映射。

问题在于单个 Crossbar 的容量有限，因此一个较大的 SNN 必须拆到多个 Crossbar 上。于是产生：

- Local synapse：连接的神经元在同一 Crossbar 内
- Global synapse：连接跨越不同 Crossbar，需要经过共享互连网络

论文指出，随着 SNN 规模增大，共享互连可能成为主要瓶颈。映射策略不仅决定 Crossbar 的使用方式，还决定 Global spike 的通信量、路径和拥塞程度。

### 1.2 Global Spike

由于多个 Crossbar 共用片上网络，同时通信 NoC / Interconnect 造成通信 congestion。

Congestion 进一步导致：

Global spikes ↑
      ↓
Interconnect congestion ↑
      ↓
Spike latency ↑
      ↓
ISI distortion ↑
      ↓
SNN temporal information 被破坏
      ↓
Accuracy ↓

论文还通过公式定义了 **ISI distortion**，本质上就是不同 spike 因网络拥塞产生不同延迟，使原本的 spike 时间关系发生变化。

对于 rate-based SNN，论文还讨论了 **spike disorder**：即硬件通信延迟改变了 spike 到达速率分布。该指标在论文的扩展讨论中提出，首轮主要实验仍以 ISI distortion 为主要 SNN 时序指标。

### 1.3 已有方法概括

| 方法 | Partition | Placement | 主要目标 |
|---|---:|---:|---|
| NEUTRAMS | ✓ | × | 减少 Crossbar 使用 |
| SCO | ✓ | × | 平衡 Crossbar 利用率 |
| PSOPART | ✓ | × | 减少 Global spikes |
| **SpiNeMap** | ✓ | ✓ | **同时优化通信与物理放置** |

不同的物理放置会直接改变 spike 需要经过的 hop 数，从而改变 latency 和 energy。

---

## 2.方法

论文的核心流程可以压缩成：

Trained SNN
    ↓
CARLsim
    ↓
Spike Trace
    ↓
SpiNeCluster
    ↓
Partition
    ↓
Local / Global Synapses
    ↓
SpiNePlacer
    ↓
Physical Placement
    ↓
Noxim++
    ↓
Latency / Energy / ISI

### 2.1 SpiNeCluster：Partition

**核心思想**

把 SNN 看成一个带权图：

Neuron = Node
Synapse = Edge
Spike count = Edge weight

若两个神经元之间通信很多，则最好把两个神经元放进同一个 Cluster。

最终目标：

> **在满足 Crossbar 容量约束的情况下，尽可能减少 Cluster 之间的 spike 数。**

论文将该问题视为经典 Graph Partitioning Problem。由于图划分是 NP-complete，论文采用启发式方法，而不是试图求全局最优解。

#### 2.1.1 为什么不用 PSO 做 Partition？

论文之前的 PSOPART 使用 PSO 进行 SNN Partition，但随着 SNN 规模增加，搜索空间迅速膨胀。

在固定 16 个 Crossbar、每个 Crossbar 容纳 256 个 neurons 的情况下：

| SNN neurons | Partition 的 PSO 维度 | Placement 的 PSO 维度 |
|---:|---:|---:|
| 1,000 | 16,000 | 64 |
| 2,000 | 32,000 | 128 |
| 3,000 | 48,000 | 192 |
| 4,000 | 64,000 | 256 |

因此论文采用：

- Partition → **Greedy / Kernighan–Lin-inspired heuristic**
- Placement → **Binary PSO**

核心原因不是“PSO 不好”，而是：

> **不同子问题的搜索空间规模不同，应选择与搜索空间匹配的优化方法。**

#### 2.1.2 SpiNeCluster 的实现

将 N 个 neurons 划分为：

\[
k=\lceil |N|/n_c\rceil
\]

个 Cluster，其中 \(n_c\) 为单个 Crossbar 的容量。

然后不断考察 Cluster 之间的 neuron exchange：

1. 将一个 neuron 从 Cluster A 移动到 Cluster B；
2. 将一个 neuron 从 Cluster B 移动到 Cluster A；
3. 交换两个 Cluster 中的 neuron。

对每一种候选操作计算新的 inter-cluster spike 数，选择能够最大程度减少 Global spikes 的操作。

循环执行：

Global spikes ↓

直到不存在进一步的改进。

该方法本质上属于 **Greedy Local Search**，思路与 Kernighan–Lin Graph Partitioning 相近。论文报告其时间复杂度约为：

\[
O(|N|^2)
\]

### 2.2 SpiNePlacer：Placement

Partition 完成后，只知道：

Cluster A
Cluster B
Cluster C
Cluster D

还不知道它们具体放在哪些物理 Crossbar：

A → Core 0
B → Core 1
C → Core 2
D → Core 3

因此需要进一步解决 **Cluster-to-Core Placement**。

#### 2.2.1 核心思想

如果两个 Cluster 之间存在大量 Global spikes，那么：

距离远
→ hop 数更多
→ latency 更高
→ energy 更高

因此 Placement 的目标是：

> **将已经划分好的 Cluster 映射到物理 Crossbar，使共享互连上的通信成本尽可能低。**

论文采用 **Binary Particle Swarm Optimization（PSO）**。

映射可表示为：

\[
M:C\rightarrow V
\]

其中：

- \(C\)：Cluster 集合
- \(V\)：物理 Crossbar 集合

并满足：

- 每个 Cluster 必须映射到一个 Crossbar；
- 一个 Crossbar 最多承载一个 Cluster。

#### 2.2.2 为什么 Placement 可以使用 PSO？

Placement 的搜索空间远小于 Partition。

Partition 的维度约为：

\[
|N|\times|C|
\]

而 Placement 的维度约为：

\[
|C|\times|V|
\]

随着 SNN 规模增加：

```text
Partition search space  → 快速膨胀
Placement search space  → 相对可控
```

因此论文将计算量较大的 PSO 保留给 Placement，而使用更具扩展性的 Greedy 方法完成 Partition。

#### 2.2.3 Noxim++：让 Mapping 有真实的通信代价

Placement 不能只按照静态 Manhattan distance 判断。

因为实际 spike 在 NoC 中会受到：

- routing algorithm
- link contention
- congestion
- dynamic traffic

等因素影响。

因此论文将 **Noxim++** 集成到 PSO 的 fitness evaluation 中。

整体过程：

Candidate Mapping
      ↓
Noxim++
      ↓
Cycle-accurate Interconnect Simulation
      ↓
Spike latency / Energy / ISI distortion
      ↓
Fitness
      ↓
PSO 更新 Mapping

Noxim++ 使用 CARLsim 产生的 spike trace 作为输入，从而模拟不同 Mapping 下真实 spike traffic 的变化。

这也是论文中非常重要的设计：

> **Mapping 的好坏最终由统一的硬件级模拟结果决定，而不是由算法自己估计。**

---

## 3.实验结果

### 3.1 实验设置

论文使用：

- **7 个 synthetic SNN**
  - S_1000
  - S_1500
  - S_2000
  - S_2500
  - S_3000
  - S_3500
  - S_4000
- **8 个 realistic SNN**
  - ImgSmooth
  - EdgeDet
  - MLP-MNIST
  - HeartEstm
  - HeartClass
  - CNN-MNIST
  - LeNet-MNIST
  - LeNet-CIFAR

硬件模型采用 **DYNAP-SE**，Crossbar 之间通过片上互连连接。

实验系统使用 CARLsim 获取 spike trace，再使用扩展后的 Noxim++ 进行 trace-driven、cycle-accurate 的互连仿真。

主要比较：

1. Baseline：以减少 Crossbar 使用为目标；
2. SCO：平衡 Crossbar 利用率；
3. PSOPART：减少 Shared Interconnect 上的 Global spikes；
4. SpiNeMap：SpiNeCluster + SpiNePlacer。

### 3.2 总体结果

论文报告 SpiNeMap 相对于不同方法的平均改进如下：

| 对比方法 | Energy Consumption ↓ | Spike Latency ↓ | ISI Distortion ↓ | Application Accuracy ↑ |
|---|---:|---:|---:|---:|
| vs. Baseline | **45%** | **21%** | **36%** | **12%** |
| vs. SCO | **40%** | **27%** | **39%** | **20%** |
| vs. PSOPART | **20%** | **13%** | **23%** | **5%** |

这里的 Accuracy 提升是论文按 Baseline 归一化后报告的**相对平均提升**。

### 3.3 为什么 SpiNeMap 有效？

#### 3.3.1 Energy

PSOPART 已经通过减少 Global spikes 降低了通信能耗，但 SpiNeMap 进一步通过 Placement 减少通信路径成本。

平均结果：

```text
Baseline
   ↓ 45%
SpiNeMap
```

相较 PSOPART：

```text
Energy ↓ 20%
```

核心原因：

1. SpiNeCluster 减少 Global spikes；
2. SpiNePlacer 将 Cluster 放到更合适的物理位置，降低通信成本。

#### 3.3.2 Spike Latency

平均：

```text
Baseline
   ↓ 21%
SpiNeMap
```

相较 PSOPART：

```text
Latency ↓ 13%
```

PSOPART 已经降低 Global spikes，从而缓解 congestion；SpiNeMap 进一步优化 Cluster 的物理位置，使 spike 平均 hop count 更低。

因此：

> **减少 spike 数量 + 缩短通信路径** 是降低 latency 的两个主要来源。

#### 3.3.3 ISI Distortion

平均：

```text
Baseline
   ↓ 36%
SpiNeMap
```

相较 PSOPART：

```text
ISI Distortion ↓ 23%
```

论文观察到：

```text
Global spikes ↑
      ↓
Congestion ↑
      ↓
Latency ↑
      ↓
ISI distortion ↑
      ↓
Application accuracy ↓
```

SpiNeMap 的 Placement 阶段进一步降低 spike latency，因此相较只优化 Global spike count 的 PSOPART，能够继续降低 ISI distortion。

#### 3.3.4 Application Accuracy

论文发现 Application Accuracy 与 ISI Distortion 呈直接相关趋势。

平均结果：

```text
SpiNeMap
Accuracy ↑ 12% vs. Baseline
Accuracy ↑ 5%  vs. PSOPART
```

因此 Mapping 并不仅仅是一个“硬件效率”问题：

> **硬件通信产生的 temporal distortion 最终会传递到 SNN 的应用层性能。**

### 3.4 SpiNeCluster 的单独验证

论文进一步验证了为什么需要新的 Partition 方法。

Global spike 数量：

| 方法 | 相对 Baseline |
|---|---:|
| SCO | **+6%** |
| PSOPART | **−23%** |
| **SpiNeMap** | **−26%** |

SpiNeMap 相比 PSOPART：

> Global spikes 进一步降低 **9%**。

同时，SpiNeCluster 的优化时间平均约为 PSOPART 的 **1/3**，即约 **3× faster**。

因此论文认为：

> Greedy-based SpiNeCluster 在大规模 SNN 上不仅更快，而且在该实验中获得了更低的 Global spike count。

### 3.5 Interconnect Design-Space Exploration

论文还利用 SpiNeMap 探索不同 Interconnect / Routing：

- XY routing
- NorthLast routing
- WestFirst routing
- Segmented Bus

结果：

- NorthLast：平均 latency 比 XY 高约 **7%**
- WestFirst：平均 latency 比 XY 高约 **4%**
- Segmented Bus：平均 latency 比 XY NoC 低约 **54%**

说明：

> 一个好的 Mapping Framework 不应只服务于单一硬件配置，还应该能够用于 Hardware / Interconnect Design-Space Exploration。

---

## 4.贡献及创新点

### 4.1 SpiNeCluster

提出一种可扩展的 heuristic-based clustering 方法：

> 将 SNN partition 成 Local / Global synapses，并以减少 Shared Interconnect 上的 Global spikes 为目标。

相比 PSO-based partition：

- 搜索空间更小；
- 运行时间更低；
- 更适合大规模 SNN。

### 4.2 SpiNePlacer

提出基于 PSO 的 Cluster-to-Crossbar placement：

> 在已经完成 Partition 的基础上进一步优化物理放置，降低 Shared Interconnect 的 Energy 和 Spike Latency。

这补足了 PSOPART 只做 Partition、不处理物理 Placement 的问题。

### 4.3 两阶段 Mapping

论文的重要设计思想：

```text
SNN
 ↓
Partition
 ↓
去除大量明显不理想的搜索空间
 ↓
Placement
 ↓
精细优化剩余空间
```

相比直接对完整 Partition + Placement 空间进行联合搜索：

> **先用廉价方法解决大搜索空间问题，再把昂贵优化算法用于较小的搜索空间。**

这使得设计空间探索更加可控，同时保持较好的解质量。

### 4.4 Trace-driven + Cycle-accurate Evaluation

论文将：

**CARLsim → Spike Trace → Noxim++**

连接起来。

这样 Mapping 的评价能够同时考虑：

- Spike traffic
- routing
- congestion
- hop count
- latency
- energy
- ISI distortion

而不是只使用一个静态的图距离指标。

### 4.5 Hardware / Interconnect Design-Space Exploration

SpiNeMap 不仅用于寻找一个 Mapping，还可以用于比较：

```text
不同 Mapping
不同 Routing
不同 Interconnect
不同 Hardware configuration
```

因此其定位更接近一个 **Mapping / Architecture Exploration Methodology**，而不只是一个单独的 Mapping Algorithm。

---

## 5. Task Design Insight

由于现阶段正在做SNN Benchmark Design 的任务才开始阅读这篇论文，故特此专门设置了这个版块，以期实现一套 **Benchmark Design Pattern**。

### 5.1 将问题拆成不同层级的 Optimization

论文没有让一个算法同时直接解决所有问题：

```text
Partition
    ↓
Placement
```

两个阶段的搜索空间不同，因此使用不同算法。

**对 D2 的启示：**

首版 Task 不宜同时开放：

```text
Partition + Placement + Routing + Scheduling
```

否则搜索空间过大，难以判断 Agent 到底优化了什么。

更合理的演进方式：

```text
D2-v0
固定 Partition
→ Agent 优化 Placement

D2-v1
Agent 优化 Partition + Placement

D2-v2
进一步开放 Routing / Scheduling
```

### 5.2 明确区分“逻辑优化”和“物理优化”

论文中：

- Partition：决定哪些神经元 / 突触属于同一 Cluster；
- Placement：决定 Cluster 在物理硬件中的位置。

二者作用不同。

因此 D2 Task 的输出也应尽量结构化：

```text
Partition
+
Cluster → Core Mapping
```

后续再增加：

```text
Routing
+
Scheduling
```

避免不同优化变量混在一起。

### 5.3 Objective 必须能够被验证

论文首先通过 Figure 7 验证：

```text
Global spikes ↑
→ Latency ↑
→ ISI distortion ↑
→ Accuracy ↓
```

之后才把 Global spike count 作为 Partition 的优化目标。

这对 Benchmark 的启示：

> **不能只规定一个看起来合理的 metric，还需要用实验验证它确实与最终任务性能相关。**

因此 D2 应先验证：

```text
Global traffic
Average hop
Congestion
Latency
```

之间是否存在稳定关系，再决定最终 Score。

### 5.4 Judge 统一化，不允许Agent自己提交

论文使用 Noxim++ 对 Candidate Mapping 重新模拟：

```text
Agent
 ↓
Mapping
 ↓
Judge / Simulator
 ↓
Latency
Energy
Congestion
ISI
 ↓
Score
```

因此 D2 Benchmark 应遵循：

> **Agent 只提交 Mapping；所有性能指标由统一 Judge 重新计算。**


否则容易产生不可验证的结果。

### 5.5 Hard Constraint 与 Optimization Objective 分离

D2 中首先必须判断：

```text
Mapping 是否合法？
```

例如：

- Cluster 是否全部被映射；
- 一个 Cluster 是否只对应一个 Core；
- Core capacity 是否超限；
- Routing 是否满足硬件规则；
- 是否出现非法通信。

只有合法解才进入：

```text
Latency / Energy / Congestion
```

评分。

即：

```text
Feasibility
    ↓
Performance
    ↓
Score
```

而不是用一个巨大的 penalty 把所有违规情况混在一起。

### 5.6 需要同时准备 Baseline 与 Reference Solution

论文中的 Baseline 不是“随便给一个随机方案”，而是具有明确优化目标的已有方法：

```text
Baseline
SCO
PSOPART
SpiNeMap
```

对 D2：

- **Baseline**：顺序 / 按层 Placement；
- **Strong baseline**：通信加权 Greedy；
- **Reference solution**：SpiNeMap-style Partition + Placement；
- **Agent solution**：由 Agent 自己搜索。

其中论文方法可以作为一个 **强参考解**，但不能直接视为全局最优解。

### 5.7 Workload 需要具有 Scaling

论文同时使用：

```text
S_1000
S_1500
...
S_4000
```

以及真实工作负载。

D2 可以设计：

```text
Small
Medium
Large
```

逐渐增加：

- neuron 数；
- synapse 数；
- spike density；
- communication hotspot；
- Core 数。

这样才能判断 Agent 是否真正学到了 Mapping strategy，而不是记住某一个实例。

### 5.8 最重要：D2 应该测试“泛化”，而不是只测试“搜索”

如果 Agent 只针对一个固定 SNN 做大量搜索，它可能只是找到：

> **这个实例的特殊答案。**

因此后续 Benchmark 应考虑：

```text
Public workload
      ↓
Agent development

Hidden workload
      ↓
Judge
      ↓
Generalization
```

甚至可以固定：

- Hardware specification
- Core capacity
- Routing rule

只更换：

- SNN topology
- Spike traffic
- Network scale

来测试 Mapping algorithm 是否真正具有泛化能力。

---

## 6.总结

SpiNeMap 真正解决的问题可以概括为：

> **在 Crossbar 容量和片上互连约束下，先利用 SNN 的 spike communication pattern 减少跨 Cluster 通信，再优化 Cluster 的物理放置，从而同时降低 Global spike、通信拥塞、Spike latency 和 Energy，并减少由通信延迟造成的 ISI distortion。**

对当前 D2 Task 最重要的启发是：

> **提供了Mapping Benchmark Pattern的一套范式，要把问题拆成清晰的优化空间，冻结硬件与工作负载约束，由统一 Judge 重新计算通信成本，并通过标准的 Baseline、Reference Solution、Scaling Workload 和 Hidden Test 流程验证 Agent 是否真的学会了 Mapping，跳出单一实例验证的局限性。**
