---
title: 【论文精读】SENMap：面向可扩展神经形态系统的多目标映射与综合
tags:
  - SNN
  - Neuromorphic Computing
  - Mapping
  - SENMap
  - Hardware-Software Co-design
---

# SENMap：面向可扩展神经形态系统的多目标映射与综合

> 论文：*SENMap: Multi-objective data-flow mapping and synthesis for hybrid scalable neuromorphic systems*  
> 作者：Prithvish V. Nembhani et al.  
> 2025，IJCNN accepted / arXiv:2506.03450v2

## 1. 总述

这篇论文讨论一个很实际的问题：**一个已经训练好的 SNN / ANN，怎样映射到多核神经形态芯片上，并同时兼顾能耗、延迟、面积、吞吐量和输出质量？**

作者给出的方案叫 **SENMap**。它建立在两层基础之上：

- **SENECA**：一类可配置的数字神经形态处理器，内部包含 RISC-V 核、神经处理单元（NPE）、存储层次和事件驱动 NoC。
- **SENSIM**：面向 SENECA 的事件驱动硬件模拟器，用硬件参数估算能耗、延迟、通信等指标。

SENMap 位于二者之上，承担“**怎么分、怎么放、芯片参数怎么配**”这一层优化。论文中的整体流程可以理解为：

```text
Pretrained SNN / ANN
        ↓
Clustering / Partitioning
        ↓
Mapping to cores
        ↓
SENSIM hardware simulation
        ↓
Energy / Latency / Area / Accuracy
        ↓
GA / NSGA-II / PSO 等优化器继续搜索
```

所以，SENMap 更像一个 **mapping + design-space exploration + hardware/software co-design framework**。它把“模型部署”转化成一个带资源约束的多目标优化问题。

论文在 PilotNet 等模型上展示了映射、事件速率和 NPE 数量之间的耦合关系，并报告约 **40% 的能耗改进**。并行元启发式搜索还把部分实验的 time-to-solution 从约 3 天压缩到约 3 小时。

---

## 2. 重要概念辨析

### 2.1 Mapping、Clustering 与 Partitioning

可以把一个大 SNN 想成一张图：

- 神经元 / 层：计算节点；
- 突触：连接；
- Spike / event：运行时通信。

芯片则由多个 Core 组成，每个 Core 的存储和计算资源都有上限。

部署时通常会经历：

```text
大网络
  ↓
分成若干组
  ↓
每组分配给某个 Core
  ↓
Core 之间通过 NoC 传递事件
```

**Clustering / Partitioning** 解决“哪些神经元或计算块放在一组”。

**Mapping** 解决“这些组最终放到哪些 Core，以及使用怎样的硬件配置”。

这两步共同决定核间通信量、存储占用、负载分布和执行效率。

### 2.2 NoC

**NoC（Network-on-Chip）** 是多核芯片内部的通信网络。

SNN 天然带有大量事件通信。当相互联系紧密的计算块被放到距离更远或更拥挤的 Core 上时，事件需要经过更多链路，延迟和能耗随之变化。

SENECA 采用事件驱动 NoC，并支持 multicast、source-based routing 和数据压缩。SENMap 因此可以把通信特征一起纳入映射优化。

### 2.3 Event rate


对于事件驱动系统来说，模型结构只是静态的一部分；真实运行时还有一个动态变量：

> 单位时间里到底产生多少事件？

事件率会影响：

- Core 的计算压力；
- NoC 的通信压力；
- 延迟；
- 能耗；
- 输出时间关系。

论文实验表明，frame / event rate 会明显改变最佳 Mapping，因此“模型 + 输入活动模式 + 硬件”需要一起看。

### 2.4 Multi-objective optimization

芯片映射通常同时关心多个目标：

\[
\text{Energy},\quad
\text{Latency},\quad
\text{Area},\quad
\text{Accuracy}
\]

这些目标之间经常存在 trade-off。

SENMap 使用 PyMOO 接入 GA、PSO、NSGA-II、NSGA-III、MOEA/D 等优化器。论文实验主要使用 GA 和 NSGA-II。

对于多目标问题，一个常见思路是寻找 **Pareto Front**：一组分别代表各类权衡关系的优质方案，再由设计者按需求选取。

### 2.5 Inter-spike distortion

SNN 的信息包含时间结构。

映射到多核芯片后，通信和调度会给各类事件带来差异化延迟，从而改变最终输出信号的时序关系。

SENMap 用输出信号之间的相关性与时间偏移来量化这类变化。这个指标把“硬件映射效果”和“模型输出质量”连接起来。

---

## 3. 主要方法介绍

### 3.1 Clustering 与 Partitioning

SENMap 支持多种拆分方式，例如：

- layer-wise；
- channel-wise；
- height-wise；
- width-wise；
- homogeneous / greedy 分配。

这些方式决定每个 Core 接收多少神经元、权重和状态，也决定 Core 之间形成怎样的通信关系。

### 3.2 Remapping 与 Mesh 压缩

完成初步映射后，SENMap 会进一步调整 Mesh 形状。

论文给出三种典型策略：

**Strict area optimal**

从总 Core 数的因数对里寻找更紧凑的矩形布局。例如 30 个 Core 可形成：

\[
(1,30),(2,15),(3,10),(5,6)
\]

最终选择 \(5\times6\) 或 \(6\times5\)。

**Loose area optimal**

当 Core 数对应的 Mesh 形状较长时，可以增加少量 Core，使二维布局更规整。

**Strict square**

直接寻找更接近正方形的 Mesh。

这一步体现了一个重要思想：

> Mapping 的优化对象既包括“模型怎么放”，也包括“硬件怎样配”。

### 3.3 Core Memory Constraint

SENMap 把每个 Core 的存储容量作为硬约束。

论文 Eq.(1) 给出的 Core memory 模型为：

\[
M_{pc}
=
2\left(
N_{npc}
+
(N_{tpc}F_{snn})
(BW_{states}+BW_{outputs})
\right)
+
(N_{wpc}+N_{bpc})BW_{weights}
\]

其中：

- \(N_{npc}\)：每 Core 的 neuron 数；
- \(N_{wpc}\)：每 Core 的 weight 数；
- \(N_{bpc}\)：每 Core 的 bias 数；
- \(N_{tpc}\)：每 Core 的 threshold 数；
- \(BW_{states}\)：neuron state 位宽；
- \(BW_{outputs}\)：输出位宽；
- \(BW_{weights}\)：weight / bias 位宽；
- \(F_{snn}\)：SNN 标志。

优化器搜索出的方案需要落在硬件容量允许的范围内。

因此，Mapping 本质上属于 **resource-constrained optimization**。

### 3.4 Metaheuristic Optimization

映射问题包含离散变量、排列变量、混合变量和多目标函数，搜索空间增长很快。

SENMap 借助 PyMOO 使用多种元启发式方法：

\[
\text{GA, PSO, ISRES, NSGA-II, NSGA-III, MOEA/D}
\]

论文实验主要采用 GA 和 NSGA-II。

可以把这个过程理解成：

```text
生成一批 Mapping
      ↓
SENSIM 评估
      ↓
得到 Energy / Latency / Accuracy 等指标
      ↓
优化器保留更优方案
      ↓
交叉 / 变异 / 更新
      ↓
继续搜索
```

因此，SENSIM 实际上承担了 objective evaluator 的角色。

### 3.5 Inter-spike distortion 与输出相关性

论文通过两个输出信号 \(x(t)\) 与 \(y(t)\) 的归一化互相关衡量输出的一致程度：

\[
z(t)
=
\frac{\mathrm{corr}(x(t),y(t))}
{\sqrt{(x(t)\cdot y(t))(x(t)\cdot y(t))}}
\]

再通过：

\[
t_{\text{shift}}
=
\arg\max_t z(t)
\]

估计两个输出信号之间的时间偏移。

直觉上：

- correlation 越高，整体输出形状越接近；
- \(t_{\text{shift}}\) 描述输出在时间轴上的偏移量。

这样，SENMap 可以同时观察硬件效率和时序信号质量。

### 3.6 Closed-loop Synthesis

SENMap 的进一步目标是把 Mapping 和芯片参数放进同一个优化环：

\[
\text{Mapping}
+
\text{NPE count}
+
\text{Memory}
+
\text{Bit-width}
+
\text{Clock frequency}
+
\text{Flit width}
\]

共同参与：

\[
\text{Energy / Latency / Area / Accuracy}
\]

的优化。

这已经从“软件部署”走向了 **algorithm-hardware-software co-design**。

---

## 4. Pull Requests：贡献、延伸问题与值得思考的点

### PR #1：把 Mapping 扩展成完整设计空间搜索

SENMap 的核心贡献之一，是把网络划分、Core 映射、事件率和硬件参数放进同一套优化框架。

一个自然的延伸方向是：

> 怎样定义清晰的变量边界，让 Placement、Routing、Scheduling 和硬件参数逐层开放，同时保持每一步的收益都可解释？

这对 Benchmark 设计尤其重要。

### PR #2：Simulator 的可信度决定优化结果的可信度

SENMap 很依赖 SENSIM。

优化器真正看到的是：

```text
Candidate
   ↓
Simulator
   ↓
Score
```

因此可以继续追问：

> Simulator 的 energy / latency 模型怎样通过真实硬件测量进行校准？

以及：

> 当硬件版本变化时，成本模型怎样快速更新？

这关系到 simulation-driven optimization 的外部有效性。

### PR #3：Event rate 应成为 Benchmark 中的一等变量

论文显示事件率会显著影响映射和能耗。

因此更完整的评测可以把 workload 表示成：

\[
\text{Model topology}
+
\text{Event activity}
+
\text{Hardware configuration}
\]

这样更接近真实部署场景。

进一步可以研究：

> 在公开 event pattern 上优化出的 Mapping，对新的 event pattern 能保持多少收益？

这会自然引出 Mapping generalization 问题。

### PR #4：搜索成本本身也值得进入评价体系

论文通过并行元启发式把部分搜索时间从约 3 天降到约 3 小时。

这说明“找到一个好 Mapping 的成本”的重要性。

因此一个更完整的评价可以同时报告：

\[
\text{Deployment quality}
\quad+\quad
\text{Search time}
\]

甚至研究：

> 在固定 10 分钟、1 小时、3 小时预算下，哪个算法可以找到更好的 Pareto 解？

这个方向很适合 Agent Benchmark。

### PR #5：从输出相关性走向任务级正确性

论文使用 cross-correlation 和 time shift 衡量输出信号变化。

进一步可以把评估分成三层：

```text
Signal consistency
      ↓
Task accuracy
      ↓
System efficiency
```

这样可以更系统地回答：

> 一个更快、更省电的 Mapping，对最终任务性能带来了怎样的影响？

### PR #6：从 Mapper 走向真正的闭环 Co-design

SENMap 已经开始同时搜索 NPE 数量、memory、bit-width、clock frequency 和 flit width。

下一步很自然地可以加入：

- Routing policy；
- Scheduling policy；
- NoC topology；
- 2.5D / 3D memory；
- 3D core stacking。

这会把问题进一步推进到：

> 给定一个模型和 workload，自动找到更合适的 neuromorphic architecture。

---

## 5. 五条总结

1. **SENMap 的核心任务是把大型 SNN / ANN 映射到可配置神经形态硬件，并同时优化能耗、延迟、面积和输出质量。**

2. **SENMap + SENSIM 形成“搜索器 + 硬件模拟器”的闭环：优化器提出 Mapping，模拟器给出性能，再继续搜索。**

3. **Mapping 同时受静态结构和动态事件率影响，因此真实部署需要联合考虑 Model、Event Activity 与 Hardware。**

4. **论文把 Core memory、Mesh layout、Inter-spike distortion 和多目标优化纳入同一框架，并使用 GA、NSGA-II 等元启发式方法进行搜索。**

5. **SENMap 最值得延伸的方向是 simulator 校准、跨 workload 泛化、搜索成本评估，以及 Mapping 与 Routing / Scheduling / Hardware 参数的联合 Co-design。**
