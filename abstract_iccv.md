## [1] GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting

### GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting

We introduce GaussianOcc, a systematic method that investigates the two usages of Gaussian splatting for fully self-supervised and efficient 3D occupancy estimation in surround views. First, traditional methods for self-supervised 3D occupancy estimation still require ground truth 6D poses from sensors during training. To address this limitation, we propose Gaussian Splatting for Projection (GSP) module to provide accurate scale information for fully self-supervised training from adjacent view projection. Additionally, existing methods rely on volume rendering for final 3D voxel representation learning using 2D signals (depth maps, semantic maps), which is both time-consuming and less effective. We propose Gaussian Splatting from Voxel space (GSV) to leverage the fast rendering properties of Gaussian splatting. As a result, the proposed GaussianOcc method enables fully self-supervised (no ground truth pose) 3D occupancy estimation in competitive performance with low computational cost (2.7 times faster in training and 5 times faster in rendering).

我们介绍了一种系统化方法——GaussianOcc，该方法研究了高斯喷涂在完全自监督和高效的环绕视角三维占用估计中的两种用法。首先，传统的自监督三维占用估计方法在训练期间仍需要传感器提供的真实6D位姿数据。为了解决这一局限性，我们提出了用于投影的高斯喷涂（GSP）模块，通过相邻视角投影提供准确的尺度信息，从而实现完全自监督的训练。此外，现有方法依赖于体渲染来使用二维信号（深度图、语义图）进行最终三维体素表示学习，这不仅耗时而且效果较差。我们提出了从体素空间进行高斯喷涂（GSV），以利用高斯喷涂的快速渲染特性。结果表明，提出的 GaussianOcc 方法在没有真实位姿的情况下实现了完全自监督的三维占用估计，并且在性能具有竞争力的同时，计算成本较低（训练速度提高了2.7倍，渲染速度提高了5倍）。



---

## [2] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

### 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

We present 3DGS-LM, a new method that accelerates the reconstruction of 3D Gaussian Splatting (3DGS) by replacing its ADAM optimizer with a tailored Levenberg-Marquardt (LM). Existing methods reduce the optimization time by decreasing the number of Gaussians or by improving the implementation of the differentiable rasterizer. However, they still rely on the ADAM optimizer to fit Gaussian parameters of a scene in thousands of iterations, which can take up to an hour. To this end, we change the optimizer to LM that runs in conjunction with the 3DGS differentiable rasterizer. For efficient GPU parallization, we propose a caching data structure for intermediate gradients that allows us to efficiently calculate Jacobian-vector products in custom CUDA kernels. In every LM iteration, we calculate update directions from multiple image subsets using these kernels and combine them in a weighted mean. Overall, our method is 30% faster than the original 3DGS while obtaining the same reconstruction quality. Our optimization is also agnostic to other methods that acclerate 3DGS, thus enabling even faster speedups compared to vanilla 3DGS.

我们提出了3DGS-LM，一种通过替换ADAM优化器为定制的Levenberg-Marquardt（LM）方法来加速3D Gaussian Splatting（3DGS）重建的新方法。现有方法通过减少高斯数量或改进可微光栅器的实现来缩短优化时间。然而，这些方法仍然依赖于ADAM优化器来调整场景的高斯参数，需要数千次迭代，可能耗时长达一个小时。为此，我们将优化器更换为与3DGS可微光栅器结合运行的LM。为了实现高效的GPU并行化，我们提出了一种用于缓存中间梯度的数据结构，能够通过自定义的CUDA内核高效计算雅可比矩阵与向量的乘积。在每次LM迭代中，我们使用这些内核从多个图像子集计算更新方向，并通过加权均值将它们结合起来。总体而言，我们的方法比原始3DGS快30%，同时保持相同的重建质量。此外，我们的优化对其他加速3DGS的方法保持兼容，因此相较于基础版3DGS能够实现更快的速度提升。


---

## [3] Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats

### Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats

We propose Long-LRM, a generalizable 3D Gaussian reconstruction model that is capable of reconstructing a large scene from a long sequence of input images. Specifically, our model can process 32 source images at 960x540 resolution within only 1.3 seconds on a single A100 80G GPU. Our architecture features a mixture of the recent Mamba2 blocks and the classical transformer blocks which allowed many more tokens to be processed than prior work, enhanced by efficient token merging and Gaussian pruning steps that balance between quality and efficiency. Unlike previous feed-forward models that are limited to processing 1~4 input images and can only reconstruct a small portion of a large scene, Long-LRM reconstructs the entire scene in a single feed-forward step. On large-scale scene datasets such as DL3DV-140 and Tanks and Temples, our method achieves performance comparable to optimization-based approaches while being two orders of magnitude more efficient.

我们提出了Long-LRM，这是一个可扩展的3D高斯重建模型，能够从长序列的输入图像中重建大规模场景。具体来说，我们的模型能够在一块A100 80G GPU上仅用1.3秒处理32张分辨率为960x540的源图像。我们的架构结合了近期的Mamba2模块和经典的Transformer模块，能够处理比以往工作更多的tokens，并通过高效的token合并和高斯修剪步骤在质量与效率之间取得平衡。与之前受限于处理1至4张输入图像、只能重建场景一小部分的前馈模型不同，Long-LRM能够在单次前馈步骤中重建整个场景。在像DL3DV-140和Tanks and Temples这样的大规模场景数据集上，我们的方法在性能上与基于优化的方法相当，但效率却高出两个数量级。


---

## [4] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes

### MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes

4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190× and 125× on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field.

4D高斯散射（4DGS）作为捕捉复杂动态3D场景的高保真技术，最近获得了广泛关注。它利用4D高斯表示和GPU友好的光栅化器，实现了快速渲染速度。尽管具有诸多优势，4DGS仍面临显著挑战，尤其是需要数百万个4D高斯，每个高斯都附带大量属性，导致巨大的内存和存储成本。本文提出了一种内存高效的4DGS框架。我们通过将颜色属性分解为每个高斯的直接颜色分量（仅需3个参数）和一个共享的轻量级交流色彩预测器，从而简化了颜色表示。这一方法消除了传统4DGS中常见的球谐函数系数，后者通常需要多达144个参数，创建了一种内存高效的4D高斯表示。此外，我们引入了一种受限熵的高斯变形技术，该技术使用变形场来扩展每个高斯的作用范围，并结合基于不透明度的熵损失，限制高斯数量，从而迫使模型使用尽可能少的高斯点来很好地拟合动态场景。通过简单的半精度存储和zip压缩，我们的框架在Technicolor和Neural 3D Video数据集上分别实现了约190倍和125倍的存储压缩，相比原始4DGS，在保持相似的渲染速度和场景表示质量的同时，设立了该领域的新标准。


---

## [5] LUDVIG: Learning-free Uplifting of 2D Visual features to Gaussian Splatting scenes

### LUDVIG: Learning-free Uplifting of 2D Visual features to Gaussian Splatting scenes

We address the task of uplifting visual features or semantic masks from 2D vision models to 3D scenes represented by Gaussian Splatting. Whereas common approaches rely on iterative optimization-based procedures, we show that a simple yet effective aggregation technique yields excellent results. Applied to semantic masks from Segment Anything (SAM), our uplifting approach leads to segmentation quality comparable to the state of the art. We then extend this method to generic DINOv2 features, integrating 3D scene geometry through graph diffusion, and achieve competitive segmentation results despite DINOv2 not being trained on millions of annotated masks like SAM.

我们研究了将2D视觉模型的视觉特征或语义掩码提升到由高斯散射表示的3D场景中的任务。与常见的基于迭代优化的方法不同，我们展示了一种简单但有效的聚合技术能够产生出色的结果。应用于来自Segment Anything（SAM）的语义掩码时，我们的提升方法在分割质量上可与当前最先进的方法媲美。随后，我们将该方法扩展到通用的DINOv2特征，通过图扩散集成3D场景几何信息，尽管DINOv2没有像SAM那样在数百万标注掩码上进行训练，但仍然取得了有竞争力的分割结果。


---

## [6] Multimodal LLM Guided Exploration and Active Mapping using Fisher Information

### AG-SLAM: Active Gaussian Splatting SLAM

We present AG-SLAM, the first active SLAM system utilizing 3D Gaussian Splatting (3DGS) for online scene reconstruction. In recent years, radiance field scene representations, including 3DGS have been widely used in SLAM and exploration, but actively planning trajectories for robotic exploration is still unvisited. In particular, many exploration methods assume precise localization and thus do not mitigate the significant risk of constructing a trajectory, which is difficult for a SLAM system to operate on. This can cause camera tracking failure and lead to failures in real-world robotic applications. Our method leverages Fisher Information to balance the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.

我们提出了 AG-SLAM，这是首个利用三维高斯喷涂 (3D Gaussian Splatting, 3DGS) 进行在线场景重建的主动 SLAM 系统。近年来，辐射场场景表示（包括 3DGS）在 SLAM 和环境探索中得到了广泛应用，但主动规划机器人探索的轨迹仍未被深入研究。尤其是，许多探索方法假设精确定位，从而未能解决构建难以用于 SLAM 系统的轨迹的显著风险。这可能导致摄像机跟踪失败，从而影响现实中的机器人应用。我们的方法利用费舍尔信息，在最大化环境信息增益和最小化定位误差成本的双重目标间实现平衡。基于 Gibson 和 Habitat-Matterport 3D 数据集的实验结果表明，所提出的方法达到了最新的技术水平。


---

## [7] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering

### GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering

We consider the problem of physically-based inverse rendering using 3D Gaussian Splatting (3DGS) representations. While recent 3DGS methods have achieved remarkable results in novel view synthesis (NVS), accurately capturing high-fidelity geometry, physically interpretable materials and lighting remains challenging, as it requires precise geometry modeling to provide accurate surface normals, along with physically-based rendering (PBR) techniques to ensure correct material and lighting disentanglement. Previous 3DGS methods resort to approximating surface normals, but often struggle with noisy local geometry, leading to inaccurate normal estimation and suboptimal material-lighting decomposition. In this paper, we introduce GeoSplatting, a novel hybrid representation that augments 3DGS with explicit geometric guidance and differentiable PBR equations. Specifically, we bridge isosurface and 3DGS together, where we first extract isosurface mesh from a scalar field, then convert it into 3DGS points and formulate PBR equations for them in a fully differentiable manner. In GeoSplatting, 3DGS is grounded on the mesh geometry, enabling precise surface normal modeling, which facilitates the use of PBR frameworks for material decomposition. This approach further maintains the efficiency and quality of NVS from 3DGS while ensuring accurate geometry from the isosurface. Comprehensive evaluations across diverse datasets demonstrate the superiority of GeoSplatting, consistently outperforming existing methods both quantitatively and qualitatively.

我们研究基于物理的逆向渲染问题，使用三维高斯分裂（3DGS）表示。尽管最新的3DGS方法在新视角合成（NVS）中取得了显著成果，但要精确捕捉高保真几何、物理可解释的材质和光照仍然具有挑战性，因为这需要精确的几何建模以提供准确的表面法线，并且需要基于物理的渲染（PBR）技术来确保材质和光照的正确解耦。现有的3DGS方法通常通过近似表面法线来解决该问题，但在处理噪声较大的局部几何时往往会遇到困难，导致法线估计不准和次优的材质光照分解。在本文中，我们提出了一种名为GeoSplatting的新型混合表示方法，通过显式几何引导和可微PBR方程扩展3DGS。具体而言，我们将等值面与3DGS相结合，首先从标量场中提取等值面网格，然后将其转换为3DGS点，并为其构建全可微的PBR方程。在GeoSplatting中，3DGS基于网格几何，使得表面法线建模更加精确，从而支持PBR框架用于材质分解。此方法在保持3DGS的NVS效率和质量的同时，确保了来自等值面的精确几何表现。在多样化数据集上的全面评估表明，GeoSplatting在定量和定性上均显著优于现有方法。


---

## [8] Self-Ensembling Gaussian Splatting for Few-shot Novel View Synthesis

### Self-Ensembling Gaussian Splatting for Few-shot Novel View Synthesis

3D Gaussian Splatting (3DGS) has demonstrated remarkable effectiveness for novel view synthesis (NVS). However, the 3DGS model tends to overfit when trained with sparse posed views, limiting its generalization capacity for broader pose variations. In this paper, we alleviate the overfitting problem by introducing a self-ensembling Gaussian Splatting (SE-GS) approach. We present two Gaussian Splatting models named the Σ-model and the Δ-model. The Σ-model serves as the primary model that generates novel-view images during inference. At the training stage, the Σ-model is guided away from specific local optima by an uncertainty-aware perturbing strategy. We dynamically perturb the Δ-model based on the uncertainties of novel-view renderings across different training steps, resulting in diverse temporal models sampled from the Gaussian parameter space without additional training costs. The geometry of the Σ-model is regularized by penalizing discrepancies between the Σ-model and the temporal samples. Therefore, our SE-GS conducts an effective and efficient regularization across a large number of Gaussian Splatting models, resulting in a robust ensemble, the Σ-model. Experimental results on the LLFF, Mip-NeRF360, DTU, and MVImgNet datasets show that our approach improves NVS quality with few-shot training views, outperforming existing state-of-the-art methods.

3D Gaussian Splatting（3DGS）在新视图合成（NVS）中表现出了显著的效果。然而，3DGS模型在使用稀疏姿态视图训练时容易出现过拟合，限制了其对更广泛姿态变化的泛化能力。本文通过引入一种自集成的高斯散点方法（Self-Ensembling Gaussian Splatting，SE-GS）来缓解过拟合问题。我们提出了两个高斯散点模型，分别命名为Σ-模型和Δ-模型。Σ-模型作为主要模型，用于推理阶段生成新视图图像。在训练阶段，通过一种不确定性感知扰动策略将Σ-模型引导离开特定的局部最优解。我们基于不同训练步骤中新视图渲染的不确定性对Δ-模型进行动态扰动，从而在无需额外训练成本的情况下，从高斯参数空间中采样出多样的时间模型。通过惩罚Σ-模型与这些时间样本之间的几何差异，对Σ-模型进行正则化。因此，SE-GS在大量高斯散点模型上实现了高效而有效的正则化，最终形成一个稳健的集成模型，即Σ-模型。实验结果表明，在LLFF、Mip-NeRF360、DTU和MVImgNet数据集上，我们的方法在少样本训练视图下提升了NVS质量，超越了现有的最先进方法。


---

## [9] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

### TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

Dynamic scene reconstruction is a long-term challenge in 3D vision. Recent methods extend 3D Gaussian Splatting to dynamic scenes via additional deformation fields and apply explicit constraints like motion flow to guide the deformation. However, they learn motion changes from individual timestamps independently, making it challenging to reconstruct complex scenes, particularly when dealing with violent movement, extreme-shaped geometries, or reflective surfaces. To address the above issue, we design a plug-and-play module called TimeFormer to enable existing deformable 3D Gaussians reconstruction methods with the ability to implicitly model motion patterns from a learning perspective. Specifically, TimeFormer includes a Cross-Temporal Transformer Encoder, which adaptively learns the temporal relationships of deformable 3D Gaussians. Furthermore, we propose a two-stream optimization strategy that transfers the motion knowledge learned from TimeFormer to the base stream during the training phase. This allows us to remove TimeFormer during inference, thereby preserving the original rendering speed. Extensive experiments in the multi-view and monocular dynamic scenes validate qualitative and quantitative improvement brought by TimeFormer.

动态场景重建一直是3D视觉领域的长期挑战。近期的方法通过附加的变形场将3D高斯点扩展到动态场景，并应用显式约束（如运动流）来引导变形。然而，这些方法从单独的时间戳独立学习运动变化，这使得在重建复杂场景时面临挑战，尤其是在处理剧烈运动、极端几何形状或反射表面时。为了解决上述问题，我们设计了一个即插即用模块，称为 TimeFormer，使现有的可变形3D高斯重建方法能够从学习的角度隐式建模运动模式。
具体而言，TimeFormer 包括一个 跨时间 Transformer 编码器（Cross-Temporal Transformer Encoder），能够自适应地学习可变形3D高斯的时间关系。此外，我们提出了一种 双流优化策略，在训练阶段将 TimeFormer 学到的运动知识传递到基础流（base stream）。这样，在推理阶段可以移除 TimeFormer，从而保留原始的渲染速度。
在多视角和单目动态场景中的大量实验表明，TimeFormer 带来了定性和定量的显著改进。


---

## [10] Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting

### Automated 3D Physical Simulation of Open-world Scene with Gaussian Splatting

Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present Sim Anything, a physics-based approach that endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict mean physical properties of objects in a zero-shot manner. Based on the mean values and the object's geometry, the Material Property Distribution Prediction model (MPDP) model then estimates the full distribution, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in an open-world scene with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate our Sim Anything achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.

近期3D生成模型的进展为模拟动态3D对象运动和定制行为提供了新的可能性，但生成此类内容依然具有挑战性。现有方法通常需要手动指定精确的物理属性进行模拟，或者依赖视频生成模型进行预测，这对计算资源要求较高。
本文重新思考了多模态大语言模型（MLLM）在基于物理模拟中的应用，提出了 Sim Anything，一种赋予静态3D对象交互动态的物理模拟方法。我们从详细的场景重建和对象级 3D 开放词汇分割开始，逐步实现多视角图像修补。受人类视觉推理的启发，我们设计了 MLLM-based Physical Property Perception (MLLM-P3)，以零样本方式预测对象的平均物理属性。基于平均值和对象几何信息，Material Property Distribution Prediction (MPDP) 模型进一步估计完整分布，将问题重构为概率分布估计，从而显著降低计算成本。
最后，我们利用 Physical-Geometric Adaptive Sampling (PGAS) 策略在开放世界场景中对对象进行模拟，通过采样粒子高效捕捉复杂变形，并显著减少计算成本。大量实验和用户研究表明，Sim Anything 能够在单张 GPU 上于 2 分钟内 生成比现有最先进方法更真实的运动效果。


---

## [11] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting

### GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting

Gaze estimation encounters generalization challenges when dealing with out-of-distribution data. To address this problem, recent methods use neural radiance fields (NeRF) to generate augmented data. However, existing methods based on NeRF are computationally expensive and lack facial details. 3D Gaussian Splatting (3DGS) has become the prevailing representation of neural fields. While 3DGS has been extensively examined in head avatars, it faces challenges with accurate gaze control and generalization across different subjects. In this work, we propose GazeGaussian, a high-fidelity gaze redirection method that uses a two-stream 3DGS model to represent the face and eye regions separately. By leveraging the unstructured nature of 3DGS, we develop a novel eye representation for rigid eye rotation based on the target gaze direction. To enhance synthesis generalization across various subjects, we integrate an expression-conditional module to guide the neural renderer. Comprehensive experiments show that GazeGaussian outperforms existing methods in rendering speed, gaze redirection accuracy, and facial synthesis across multiple datasets. We also demonstrate that existing gaze estimation methods can leverage GazeGaussian to improve their generalization performance.

凝视估计在处理分布外数据时面临泛化挑战。为解决这一问题，近期方法尝试使用 NeRF（神经辐射场）生成增强数据。然而，基于 NeRF 的现有方法计算代价高昂且缺乏面部细节。随着 3D Gaussian Splatting (3DGS) 成为神经场的主流表示，其在头部头像建模中已有广泛应用，但在准确的凝视控制和跨主体的泛化方面仍存在挑战。
为此，我们提出 GazeGaussian，一种高保真凝视重定向方法，使用双流 3DGS 模型分别表示面部和眼睛区域。通过利用 3DGS 的非结构化特性，我们设计了一种基于目标凝视方向的刚性眼球旋转新颖表示方法。为增强在不同主体间的合成泛化能力，我们引入了一个 表情条件模块，以引导神经渲染器。
全面实验表明，GazeGaussian 在渲染速度、凝视重定向精度以及面部合成质量上均优于现有方法，并在多个数据集上表现出卓越的性能。此外，我们进一步证明，现有的凝视估计方法可以利用 GazeGaussian 提升其泛化能力，从而改进对分布外数据的适应性。


---

## [12] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation

### Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation

Existing feed-forward image-to-3D methods mainly rely on 2D multi-view diffusion models that cannot guarantee 3D consistency. These methods easily collapse when changing the prompt view direction and mainly handle object-centric prompt images. In this paper, we propose a novel single-stage 3D diffusion model, DiffusionGS, for object and scene generation from a single view. DiffusionGS directly outputs 3D Gaussian point clouds at each timestep to enforce view consistency and allow the model to generate robustly given prompt views of any directions, beyond object-centric inputs. Plus, to improve the capability and generalization ability of DiffusionGS, we scale up 3D training data by developing a scene-object mixed training strategy. Experiments show that our method enjoys better generation quality (2.20 dB higher in PSNR and 23.25 lower in FID) and over 5x faster speed (~6s on an A100 GPU) than SOTA methods. The user study and text-to-3D applications also reveals the practical values of our method.

现有的前馈式图像到3D方法主要依赖于二维多视角扩散模型，但这些模型难以保证三维一致性。在视角变化时，这些方法容易崩溃，并且主要适用于以物体为中心的提示图像。为解决这些问题，本文提出了一种新颖的单阶段三维扩散模型 DiffusionGS，用于从单视图生成物体和场景。
DiffusionGS 在每个时间步直接输出三维高斯点云，从而强化视角一致性，使模型能够稳健地生成来自任意方向的提示视图，而不仅限于以物体为中心的输入。此外，为了提高 DiffusionGS 的生成能力和泛化能力，我们开发了一种 场景-物体混合训练策略，大规模扩展了三维训练数据。
实验表明，与现有最先进方法相比，DiffusionGS 在生成质量上表现更佳（PSNR 提高 2.20 dB，FID 降低 23.25），并且速度提高超过 5 倍（在 A100 GPU 上约为 6 秒）。用户研究和文本到3D应用进一步展示了该方法的实用价值。


---

## [13] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting

### EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting

Photorealistic reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. While recent methods based on 3D/4D Gaussian Splatting (GS) have demonstrated promising results, they still encounter challenges in complex street scenes due to the unpredictable motion of dynamic objects. Current methods typically decompose street scenes into static and dynamic objects, learning the Gaussians in either a supervised manner (e.g., w/ 3D bounding-box) or a self-supervised manner (e.g., w/o 3D bounding-box). However, these approaches do not effectively model the motions of dynamic objects (e.g., the motion speed of pedestrians is clearly different from that of vehicles), resulting in suboptimal scene decomposition. To address this, we propose Explicit Motion Decomposition (EMD), which models the motions of dynamic objects by introducing learnable motion embeddings to the Gaussians, enhancing the decomposition in street scenes. The proposed EMD is a plug-and-play approach applicable to various baseline methods. We also propose tailored training strategies to apply EMD to both supervised and self-supervised baselines. Through comprehensive experimentation, we illustrate the effectiveness of our approach with various established baselines. The code will be released at: this https URL.

街景的逼真重建对开发自动驾驶的真实场景模拟器至关重要。尽管基于 3D/4D 高斯投影（Gaussian Splatting, GS）的最新方法在该领域展现了良好前景，但由于动态物体不可预测的运动，这些方法在复杂街景中仍面临挑战。当前的方法通常将街景分解为静态和动态物体，并通过有监督（例如基于 3D 边界框）或自监督（例如无需 3D 边界框）的方式学习高斯分布。然而，这些方法未能有效建模动态物体的运动特性（例如，行人的运动速度明显不同于车辆），导致场景分解效果不够理想。
为了解决这一问题，我们提出了显式运动分解（Explicit Motion Decomposition, EMD）方法，通过向高斯分布中引入可学习的运动嵌入（motion embeddings），对动态物体的运动进行建模，从而增强街景的分解效果。所提出的 EMD 方法是一种可即插即用的方案，适用于多种基线方法。此外，我们还设计了针对性的训练策略，使 EMD 能够应用于有监督和自监督的基线方法。
通过全面的实验，我们验证了在多种基线方法中应用 EMD 的有效性，并表明其显著改善了街景动态物体的分解与建模。


---

## [14] Sequential Gaussian Avatars with Hierarchical Motion Context

### GAST: Sequential Gaussian Avatars with Hierarchical Spatio-temporal Context

3D human avatars, through the use of canonical radiance fields and per-frame observed warping, enable high-fidelity rendering and animating. However, existing methods, which rely on either spatial SMPL(-X) poses or temporal embeddings, respectively suffer from coarse rendering quality or limited animation flexibility. To address these challenges, we propose GAST, a framework that unifies 3D human modeling with 3DGS by hierarchically integrating both spatial and temporal information. Specifically, we design a sequential conditioning framework for the non-rigid warping of the human body, under whose guidance more accurate 3D Gaussians can be obtained in the observation space. Moreover, the explicit properties of Gaussians allow us to embed richer sequential information, encompassing both the coarse sequence of human poses and finer per-vertex motion details. These sequence conditions are further sampled across different temporal scales, in a coarse-to-fine manner, ensuring unbiased inputs for non-rigid warping. Experimental results demonstrate that our method combined with hierarchical spatio-temporal modeling surpasses concurrent baselines, delivering both high-quality rendering and flexible animating capabilities.

通过使用规范辐射场和逐帧观察到的形变，3D 人体化身能够实现高保真的渲染和动画。然而，现有方法依赖空间上的 SMPL(-X) 姿态或时间嵌入，分别面临渲染质量粗糙或动画灵活性受限的问题。
为了解决这些挑战，我们提出了 GAST，一个将 3D 人体建模与 3D 高斯投影（3DGS）相统一的框架，通过层次化整合空间和时间信息实现高效建模。具体来说，我们设计了一种顺序条件框架，用于非刚体的人体形变，在其引导下，可以在观测空间中获得更精确的 3D 高斯。此外，高斯的显式属性允许我们嵌入更丰富的序列信息，涵盖人体姿态的粗略序列以及更细粒度的逐顶点运动细节。
这些序列条件以粗到细的方式在不同时间尺度上进行采样，从而确保非刚体形变的输入不带偏差。实验结果表明，我们结合层次化时空建模的方法，超越了现有的同期基线，实现了高质量渲染和灵活的动画能力，显著提升了 3D 人体建模的表现力和实用性。


---

## [15] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision

### A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision

We introduce a diffusion model for Gaussian Splats, SplatDiffusion, to enable generation of three-dimensional structures from single images, addressing the ill-posed nature of lifting 2D inputs to 3D. Existing methods rely on deterministic, feed-forward predictions, which limit their ability to handle the inherent ambiguity of 3D inference from 2D data. Diffusion models have recently shown promise as powerful generative models for 3D data, including Gaussian splats; however, standard diffusion frameworks typically require the target signal and denoised signal to be in the same modality, which is challenging given the scarcity of 3D data. To overcome this, we propose a novel training strategy that decouples the denoised modality from the supervision modality. By using a deterministic model as a noisy teacher to create the noised signal and transitioning from single-step to multi-step denoising supervised by an image rendering loss, our approach significantly enhances performance compared to the deterministic teacher. Additionally, our method is flexible, as it can learn from various 3D Gaussian Splat (3DGS) teachers with minimal adaptation; we demonstrate this by surpassing the performance of two different deterministic models as teachers, highlighting the potential generalizability of our framework. Our approach further incorporates a guidance mechanism to aggregate information from multiple views, enhancing reconstruction quality when more than one view is available. Experimental results on object-level and scene-level datasets demonstrate the effectiveness of our framework.

我们提出了一种针对高斯散射（Gaussian Splats）的扩散模型，称为SplatDiffusion，以从单张图像生成三维结构，解决将二维输入提升为三维的病态问题。现有方法依赖于确定性、前馈式预测，这限制了它们处理从二维数据推断三维固有模糊性的能力。
扩散模型最近被证明是三维数据（包括高斯散射）的强大生成模型。然而，标准的扩散框架通常要求目标信号和去噪信号处于相同模态中，这在三维数据稀缺的情况下具有挑战性。为了解决这一问题，我们提出了一种新的训练策略，将去噪模态与监督模态解耦。具体来说，我们利用一个确定性模型作为噪声教师，生成带噪信号，并从单步去噪过渡到通过图像渲染损失监督的多步去噪，大幅提升了相较于确定性教师的性能。
此外，我们的方法具有灵活性，可通过最小适配从不同的三维高斯散射（3DGS）教师中学习；实验表明，我们的方法优于两种不同的确定性教师模型，展现了框架的潜在泛化能力。我们的方法还结合了一种指导机制，以聚合来自多视角的信息，在可用多个视角时进一步提高重建质量。
在物体级和场景级数据集上的实验结果证明了我们框架的有效性。


---

## [16] InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models

### InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models

We present InfiniCube, a scalable method for generating unbounded dynamic 3D driving scenes with high fidelity and controllability. Previous methods for scene generation either suffer from limited scales or lack geometric and appearance consistency along generated sequences. In contrast, we leverage the recent advancements in scalable 3D representation and video models to achieve large dynamic scene generation that allows flexible controls through HD maps, vehicle bounding boxes, and text descriptions. First, we construct a map-conditioned sparse-voxel-based 3D generative model to unleash its power for unbounded voxel world generation. Then, we re-purpose a video model and ground it on the voxel world through a set of carefully designed pixel-aligned guidance buffers, synthesizing a consistent appearance. Finally, we propose a fast feed-forward approach that employs both voxel and pixel branches to lift the dynamic videos to dynamic 3D Gaussians with controllable objects. Our method can generate controllable and realistic 3D driving scenes, and extensive experiments validate the effectiveness and superiority of our model.

我们提出了 InfiniCube，一种可扩展的方法，用于生成高保真且可控的无限动态三维驾驶场景。以往的场景生成方法要么受限于生成规模，要么在生成序列中缺乏几何和外观的一致性。相比之下，我们利用了近期在可扩展三维表示和视频模型方面的进展，实现了大型动态场景生成，并通过高清地图（HD maps）、车辆边界框和文本描述实现灵活控制。
首先，我们构建了一个基于地图约束的稀疏体素三维生成模型，释放其在生成无限体素世界中的潜力。接着，我们重新设计了一个视频模型，并通过一组精心设计的像素对齐引导缓冲器将其锚定在体素世界中，以合成一致的外观。最后，我们提出了一种快速前馈方法，结合体素分支和像素分支，将动态视频提升为包含可控对象的动态三维高斯表示。
我们的方法能够生成可控且逼真的三维驾驶场景，并通过大量实验验证了模型的有效性和优越性。


---

## [17] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding

### EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding

3D occupancy prediction provides a comprehensive description of the surrounding scenes and has become an essential task for 3D perception. Most existing methods focus on offline perception from one or a few views and cannot be applied to embodied agents which demands to gradually perceive the scene through progressive embodied exploration. In this paper, we formulate an embodied 3D occupancy prediction task to target this practical scenario and propose a Gaussian-based EmbodiedOcc framework to accomplish it. We initialize the global scene with uniform 3D semantic Gaussians and progressively update local regions observed by the embodied agent. For each update, we extract semantic and structural features from the observed image and efficiently incorporate them via deformable cross-attention to refine the regional Gaussians. Finally, we employ Gaussian-to-voxel splatting to obtain the global 3D occupancy from the updated 3D Gaussians. Our EmbodiedOcc assumes an unknown (i.e., uniformly distributed) environment and maintains an explicit global memory of it with 3D Gaussians. It gradually gains knowledge through local refinement of regional Gaussians, which is consistent with how humans understand new scenes through embodied exploration. We reorganize an EmbodiedOcc-ScanNet benchmark based on local annotations to facilitate the evaluation of the embodied 3D occupancy prediction task. Experiments demonstrate that our EmbodiedOcc outperforms existing local prediction methods and accomplishes the embodied occupancy prediction with high accuracy and strong expandability.

三维占据预测能够全面描述周围场景，是三维感知领域的一项核心任务。目前大多数方法专注于基于单视图或少量视图的离线感知，无法满足具身智能体（embodied agents）逐步通过探索感知场景的需求。本文针对这一实际场景，提出了具身三维占据预测任务（embodied 3D occupancy prediction），并设计了基于高斯的 EmbodiedOcc 框架来实现。
我们以均匀分布的三维语义高斯初始化全局场景，并逐步更新具身智能体观测到的局部区域。对于每次更新，我们从观测图像中提取语义和结构特征，并通过高效的可变形跨注意力机制（deformable cross-attention）整合这些特征，以优化区域高斯表示。最终，通过高斯到体素的点绘（Gaussian-to-voxel splatting）将更新后的三维高斯转化为全局三维占据表示。
EmbodiedOcc 假设环境未知（即初始为均匀分布），并通过三维高斯显式维护全局记忆。它通过对局部区域的逐步优化来逐渐获取知识，这种方式与人类通过具身探索理解新场景的过程一致。我们基于局部标注重组了 EmbodiedOcc-ScanNet 基准，用于评估具身三维占据预测任务。
实验表明，EmbodiedOcc 超越了现有局部预测方法，在高精度和强扩展性方面表现出色，成功实现了具身占据预测任务。


---

## [18] FaceLift: Single Image to 3D Head with View Generation and GS-LRM

### FaceLift: Single Image to 3D Head with View Generation and GS-LRM

We present FaceLift, a feed-forward approach for rapid, high-quality, 360-degree head reconstruction from a single image. Our pipeline begins by employing a multi-view latent diffusion model that generates consistent side and back views of the head from a single facial input. These generated views then serve as input to a GS-LRM reconstructor, which produces a comprehensive 3D representation using Gaussian splats. To train our system, we develop a dataset of multi-view renderings using synthetic 3D human head as-sets. The diffusion-based multi-view generator is trained exclusively on synthetic head images, while the GS-LRM reconstructor undergoes initial training on Objaverse followed by fine-tuning on synthetic head data. FaceLift excels at preserving identity and maintaining view consistency across views. Despite being trained solely on synthetic data, FaceLift demonstrates remarkable generalization to real-world images. Through extensive qualitative and quantitative evaluations, we show that FaceLift outperforms state-of-the-art methods in 3D head reconstruction, highlighting its practical applicability and robust performance on real-world images. In addition to single image reconstruction, FaceLift supports video inputs for 4D novel view synthesis and seamlessly integrates with 2D reanimation techniques to enable 3D facial animation.

我们提出了FaceLift，这是一种前馈方法，用于从单张图像快速、高质量地进行360度头部重建。我们的流程首先使用多视图潜在扩散模型，从单一面部输入生成一致的头部侧面和背面视图。这些生成的视图随后作为GS-LRM重建器的输入，后者使用高斯点云生成全面的三维表示。为了训练我们的系统，我们开发了一个使用合成三维人头资产的多视图渲染数据集。基于扩散的多视图生成器仅在合成头部图像上进行训练，而GS-LRM重建器则先在Objaverse上进行初步训练，然后在合成头部数据上进行微调。FaceLift在保持身份特征和各视图之间的一致性方面表现出色。尽管仅在合成数据上进行训练，FaceLift在真实世界图像上的泛化能力表现出色。通过广泛的定性和定量评估，我们展示了FaceLift在三维头部重建方面优于最先进的方法，突显了其在实际应用中的可行性和在真实世界图像上的稳健表现。除了单张图像重建，FaceLift还支持视频输入用于4D新颖视图合成，并与二维再动画技术无缝集成，实现三维面部动画。


---

## [19] SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding

### Scaffold-SLAM: Structured 3D Gaussians for Simultaneous Localization and Photorealistic Mapping

3D Gaussian Splatting (3DGS) has recently revolutionized novel view synthesis in the Simultaneous Localization and Mapping (SLAM). However, existing SLAM methods utilizing 3DGS have failed to provide high-quality novel view rendering for monocular, stereo, and RGB-D cameras simultaneously. Notably, some methods perform well for RGB-D cameras but suffer significant degradation in rendering quality for monocular cameras. In this paper, we present Scaffold-SLAM, which delivers simultaneous localization and high-quality photorealistic mapping across monocular, stereo, and RGB-D cameras. We introduce two key innovations to achieve this state-of-the-art visual quality. First, we propose Appearance-from-Motion embedding, enabling 3D Gaussians to better model image appearance variations across different camera poses. Second, we introduce a frequency regularization pyramid to guide the distribution of Gaussians, allowing the model to effectively capture finer details in the scene. Extensive experiments on monocular, stereo, and RGB-D datasets demonstrate that Scaffold-SLAM significantly outperforms state-of-the-art methods in photorealistic mapping quality, e.g., PSNR is 16.76% higher in the TUM RGB-D datasets for monocular cameras.

3D 高斯点绘制（3D Gaussian Splatting, 3DGS）近年来在同步定位与建图（Simultaneous Localization and Mapping, SLAM）中的新视图合成任务中取得了革命性进展。然而，现有利用 3DGS 的 SLAM 方法尚未能够同时为单目、立体视觉和 RGB-D 摄像机提供高质量的新视图渲染。其中，一些方法在 RGB-D 摄像机中表现较好，但在单目摄像机中渲染质量显著下降。
本文提出了 Scaffold-SLAM，一种能够在单目、立体视觉和 RGB-D 摄像机中同时实现高质量光真实感建图和定位的系统。为实现这一最先进的视觉质量，我们提出了基于运动的外观嵌入（Appearance-from-Motion embedding），使得 3D 高斯能够更好地建模不同相机姿态下图像的外观变化。此外，我们引入了频率正则化金字塔（Frequency Regularization Pyramid），用于引导高斯分布，从而有效捕捉场景中的细节信息。
在单目、立体视觉和 RGB-D 数据集上的大量实验表明，Scaffold-SLAM 在光真实感建图质量上显著优于当前最先进的方法。例如，在 TUM RGB-D 数据集的单目摄像机实验中，PSNR 提高了 16.76%。


---

## [20] Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution

### Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution

Equipped with the continuous representation capability of Multi-Layer Perceptron (MLP), Implicit Neural Representation (INR) has been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, the limited receptive field of the linear layers in MLP restricts the representation capability of INR, while it is computationally expensive to query the MLP numerous times to render each pixel. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted contiguous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method.

基于多层感知机（MLP）的连续表示能力，隐式神经表示（INR）已被成功应用于任意比例超分辨率（ASR）。然而，MLP 中线性层的有限感受野限制了 INR 的表示能力，同时多次查询 MLP 来渲染每个像素的计算开销较高。最近，高斯散点（Gaussian Splatting, GS）在 3D 任务中展现了其在视觉质量和渲染速度上的优势，这促使我们探索 GS 是否可以被用于 ASR 任务。然而，直接将 GS 应用于 ASR 面临极大的挑战，因为原始 GS 是一种通过对每个单一场景进行过拟合的优化方法，而在 ASR 中，我们的目标是学习一个可以泛化到不同图像和缩放因子的单一模型。
我们通过开发两项新技术克服了这些挑战。首先，为了将 GS 泛化到 ASR，我们精心设计了一种架构，以前馈的方式预测与输入低分辨率图像相关的图像条件高斯分布。其次，我们实现了一种高效的基于 GPU/CUDA 的可微分二维缩放感知光栅化，通过从预测的连续高斯分布中采样离散的 RGB 值来渲染超分辨率图像。通过端到端的训练，我们优化的网络，即 GSASR，可以对任意图像和未见过的缩放因子执行 ASR。大量实验验证了我们提出方法的有效性。


---

## [21] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting

### OccGS: Zero-shot 3D Occupancy Reconstruction with Semantic and Geometric-Aware Gaussian Splatting

Obtaining semantic 3D occupancy from raw sensor data without manual annotations remains an essential yet challenging task. While prior works have approached this as a perception prediction problem, we formulate it as scene-aware 3D occupancy reconstruction with geometry and semantics. In this work, we propose OccGS, a novel 3D Occupancy reconstruction framework utilizing Semantic and Geometric-Aware Gaussian Splatting in a zero-shot manner. Leveraging semantics extracted from vision-language models and geometry guided by LiDAR points, OccGS constructs Semantic and Geometric-Aware Gaussians from raw multisensor data. We also develop a cumulative Gaussian-to-3D voxel splatting method for reconstructing occupancy from the Gaussians. OccGS performs favorably against self-supervised methods in occupancy prediction, achieving comparable performance to fully supervised approaches and achieving state-of-the-art performance on zero-shot semantic 3D occupancy estimation.

从原始传感器数据中获取语义3D占据信息而无需人工标注，依然是一个至关重要且具有挑战性的任务。尽管先前的研究将其视为感知预测问题，但我们将其表述为具有几何和语义的场景感知3D占据重建。在本研究中，我们提出了OccGS，一种利用语义和几何感知高斯溅射的零样本3D占据重建框架。OccGS通过利用从视觉-语言模型提取的语义信息和通过LiDAR点引导的几何信息，从原始多传感器数据中构建语义和几何感知高斯。我们还开发了一种累积的高斯到3D体素溅射方法，用于从高斯中重建占据信息。与自监督方法相比，OccGS在占据预测方面表现优异，达到了与完全监督方法相当的性能，并在零样本语义3D占据估计任务中实现了最先进的性能。


---

## [22] GaussRender: Learning 3D Occupancy with Gaussian Rendering

### GaussRender: Learning 3D Occupancy with Gaussian Rendering

Understanding the 3D geometry and semantics of driving scenes is critical for developing of safe autonomous vehicles. While 3D occupancy models are typically trained using voxel-based supervision with standard losses (e.g., cross-entropy, Lovasz, dice), these approaches treat voxel predictions independently, neglecting their spatial relationships. In this paper, we propose GaussRender, a plug-and-play 3D-to-2D reprojection loss that enhances voxel-based supervision. Our method projects 3D voxel representations into arbitrary 2D perspectives and leverages Gaussian splatting as an efficient, differentiable rendering proxy of voxels, introducing spatial dependencies across projected elements. This approach improves semantic and geometric consistency, handles occlusions more efficiently, and requires no architectural modifications. Extensive experiments on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate consistent performance gains across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), highlighting the robustness and versatility of our framework.

理解驾驶场景的3D几何和语义对于开发安全的自动驾驶车辆至关重要。虽然3D占据模型通常使用基于体素的监督与标准损失（例如交叉熵、Lovasz、dice）进行训练，但这些方法将体素预测视为独立的，忽视了它们之间的空间关系。本文提出了GaussRender，一种即插即用的3D到2D重投影损失，旨在增强基于体素的监督。我们的方法将3D体素表示投影到任意2D视角，并利用高斯溅射作为体素的高效、可微分渲染代理，引入了投影元素之间的空间依赖关系。这种方法改善了语义和几何一致性，更高效地处理了遮挡问题，并且不需要对架构进行修改。在多个基准测试（SurroundOcc-nuScenes、Occ3D-nuScenes、SSCBench-KITTI360）上的大量实验表明，我们的方法在各种3D占据模型（TPVFormer、SurroundOcc、Symphonies）中都表现出了稳定的性能提升，凸显了我们框架的鲁棒性和多样性。


---

## [23] Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction

### Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction

In this paper, we present a self-calibrating framework that jointly optimizes camera parameters, lens distortion and 3D Gaussian representations, enabling accurate and efficient scene reconstruction. In particular, our technique enables high-quality scene reconstruction from Large field-of-view (FOV) imagery taken with wide-angle lenses, allowing the scene to be modeled from a smaller number of images. Our approach introduces a novel method for modeling complex lens distortions using a hybrid network that combines invertible residual networks with explicit grids. This design effectively regularizes the optimization process, achieving greater accuracy than conventional camera models. Additionally, we propose a cubemap-based resampling strategy to support large FOV images without sacrificing resolution or introducing distortion artifacts. Our method is compatible with the fast rasterization of Gaussian Splatting, adaptable to a wide variety of camera lens distortion, and demonstrates state-of-the-art performance on both synthetic and real-world datasets.

在本文中，我们提出了一个自校准框架，该框架联合优化相机参数、镜头畸变和3D高斯表示，从而实现精确且高效的场景重建。特别地，我们的技术能够从大视场（FOV）图像中进行高质量场景重建，这些图像使用广角镜头拍摄，使得能够从较少的图像中建模场景。我们的方法引入了一种新颖的镜头畸变建模方法，采用混合网络，结合了可逆残差网络和显式网格。这一设计有效地规范化了优化过程，相比传统相机模型达到了更高的准确性。此外，我们提出了一种基于立方体贴图的重采样策略，支持大视场图像，同时不牺牲分辨率或引入畸变伪影。我们的方法与高斯溅射的快速光栅化兼容，可适应各种相机镜头畸变，并在合成和真实世界数据集上展示了最先进的性能。


---

## [24] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow

### GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow

Occupancy estimation has become a prominent task in 3D computer vision, particularly within the autonomous driving community. In this paper, we present a novel approach to occupancy estimation, termed GaussianFlowOcc, which is inspired by Gaussian Splatting and replaces traditional dense voxel grids with a sparse 3D Gaussian representation. Our efficient model architecture based on a Gaussian Transformer significantly reduces computational and memory requirements by eliminating the need for expensive 3D convolutions used with inefficient voxel-based representations that predominantly represent empty 3D spaces. GaussianFlowOcc effectively captures scene dynamics by estimating temporal flow for each Gaussian during the overall network training process, offering a straightforward solution to a complex problem that is often neglected by existing methods. Moreover, GaussianFlowOcc is designed for scalability, as it employs weak supervision and does not require costly dense 3D voxel annotations based on additional data (e.g., LiDAR). Through extensive experimentation, we demonstrate that GaussianFlowOcc significantly outperforms all previous methods for weakly supervised occupancy estimation on the nuScenes dataset while featuring an inference speed that is 50 times faster than current SOTA.

占用估计（Occupancy Estimation） 已成为 3D 计算机视觉 领域的重要任务，尤其在 自动驾驶 领域受到了广泛关注。在本文中，我们提出了一种新颖的占用估计算法 GaussianFlowOcc，该方法受 Gaussian Splatting 启发，用 稀疏 3D 高斯表示 替代了传统的 稠密体素网格。
我们基于 Gaussian Transformer 设计了一种高效的模型架构，大幅降低了计算和内存开销。相比传统 基于体素的表示方法 主要用于表示 空旷的 3D 空间，但需要高昂的 3D 卷积计算，我们的方法无需这些低效操作。GaussianFlowOcc 通过在整个网络训练过程中，为每个 3D Gaussian 估计 时序流（temporal flow），从而高效捕捉场景动态，为这一复杂问题提供了一种直观的解决方案，而该问题往往被现有方法所忽略。
此外，GaussianFlowOcc 具有良好的可扩展性，采用 弱监督 训练 （Weak Supervision），无需依赖额外数据（如 LiDAR）生成的 高成本稠密 3D 体素标注。
通过广泛的实验，我们表明 GaussianFlowOcc 在 nuScenes 数据集上的弱监督占用估计任务中，显著超越了所有已有方法，并且 推理速度比当前 SOTA 方法快 50 倍。


---

## [25] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars

### Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars

Traditionally, creating photo-realistic 3D head avatars requires a studio-level multi-view capture setup and expensive optimization during test-time, limiting the use of digital human doubles to the VFX industry or offline renderings.
To address this shortcoming, we present Avat3r, which regresses a high-quality and animatable 3D head avatar from just a few input images, vastly reducing compute requirements during inference. More specifically, we make Large Reconstruction Models animatable and learn a powerful prior over 3D human heads from a large multi-view video dataset. For better 3D head reconstructions, we employ position maps from DUSt3R and generalized feature maps from the human foundation model Sapiens. To animate the 3D head, our key discovery is that simple cross-attention to an expression code is already sufficient. Finally, we increase robustness by feeding input images with different expressions to our model during training, enabling the reconstruction of 3D head avatars from inconsistent inputs, e.g., an imperfect phone capture with accidental movement, or frames from a monocular video.
We compare Avat3r with current state-of-the-art methods for few-input and single-input scenarios, and find that our method has a competitive advantage in both tasks. Finally, we demonstrate the wide applicability of our proposed model, creating 3D head avatars from images of different sources, smartphone captures, single images, and even out-of-domain inputs like antique busts.

传统上，创建逼真的 3D 头像模型需要使用专业级的多视角捕捉设备，并在测试阶段进行昂贵的优化，这限制了数字人双胞胎技术的应用范围，仅能用于视觉特效（VFX）行业或离线渲染。
为了解决这一局限性，我们提出了 Avat3r，该方法能够仅凭少量输入图像回归出高质量且可动画化的 3D 头像模型，大幅降低推理时的计算成本。具体而言，我们使大规模重建模型（Large Reconstruction Models）具备动画能力，并从大规模多视角视频数据集中学习了强大的 3D 人头先验。为了实现更高质量的 3D 头部重建，我们结合了 DUSt3R 提供的位置映射（position maps）以及人类基础模型 Sapiens 的通用特征映射（generalized feature maps）。
在 3D 头部动画化方面，我们的关键发现是：简单的跨注意力（cross-attention）机制应用于表情编码（expression code）即可实现高效的动画驱动。此外，为了增强鲁棒性，我们在训练过程中输入了具有不同表情的图像，使模型能够从不一致的输入数据中重建 3D 头像，例如因意外运动导致的手机拍摄误差，或是单目视频中的不同帧图像。
我们将 Avat3r 与当前最先进的少量输入和单输入 3D 头像重建方法进行了比较，结果表明，我们的方法在这两种任务上均具有竞争优势。最后，我们展示了 Avat3r 的广泛适用性，它能够从不同来源的图像（如智能手机拍摄、单张图片，甚至是古代雕像）生成高质量的 3D 头像模型。


---

## [26] MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions

### MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions

Novel view synthesis (NVS) and surface reconstruction (SR) are essential tasks in 3D Gaussian Splatting (3D-GS). Despite recent progress, these tasks are often addressed independently, with GS-based rendering methods struggling under diverse light conditions and failing to produce accurate surfaces, while GS-based reconstruction methods frequently compromise rendering quality. This raises a central question: must rendering and reconstruction always involve a trade-off? To address this, we propose MGSR, a 2D/3D Mutual-boosted Gaussian splatting for Surface Reconstruction that enhances both rendering quality and 3D reconstruction accuracy. MGSR introduces two branches--one based on 2D-GS and the other on 3D-GS. The 2D-GS branch excels in surface reconstruction, providing precise geometry information to the 3D-GS branch. Leveraging this geometry, the 3D-GS branch employs a geometry-guided illumination decomposition module that captures reflected and transmitted components, enabling realistic rendering under varied light conditions. Using the transmitted component as supervision, the 2D-GS branch also achieves high-fidelity surface reconstruction. Throughout the optimization process, the 2D-GS and 3D-GS branches undergo alternating optimization, providing mutual supervision. Prior to this, each branch completes an independent warm-up phase, with an early stopping strategy implemented to reduce computational costs. We evaluate MGSR on a diverse set of synthetic and real-world datasets, at both object and scene levels, demonstrating strong performance in rendering and surface reconstruction.

新视角合成（Novel View Synthesis, NVS）和表面重建（Surface Reconstruction, SR）是 3D 高斯散点 (3D Gaussian Splatting, 3D-GS) 中的两项核心任务。尽管近年来取得了显著进展，这两项任务通常是独立处理的：基于 GS 的渲染方法在不同光照条件下表现不稳定，难以生成精确的表面，而基于 GS 的重建方法往往会牺牲渲染质量。这引发了一个核心问题：渲染与重建是否必须相互妥协？
为了解决这一问题，我们提出 MGSR，即一种 2D/3D 互增强的高斯散点表面重建方法 (Mutual-boosted Gaussian Splatting for Surface Reconstruction)，旨在同时提升渲染质量和 3D 重建精度。
MGSR 采用双分支架构：一个基于 2D-GS，另一个基于 3D-GS。其中，2D-GS 分支擅长表面重建，提供精确的几何信息以增强 3D-GS 分支。利用这一几何信息，3D-GS 分支引入几何引导的光照分解模块 (Geometry-Guided Illumination Decomposition Module)，能够分离反射与透射成分，从而在不同光照条件下实现逼真的渲染。同时，以透射成分作为监督信号，2D-GS 分支也能实现高保真的表面重建。
在优化过程中，2D-GS 和 3D-GS 分支采用交替优化机制，互相提供监督信息。在此之前，每个分支会独立完成预训练阶段 (warm-up phase)，并采用提前停止策略 (early stopping strategy) 以降低计算成本。
我们在多个合成和真实数据集上进行了实验，包括物体级和场景级评估，结果表明 MGSR 在渲染与表面重建任务上均表现出色。


---

## [27] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images

### CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images

3D Gaussian Splatting (3DGS) has gained significant attention for their high-quality novel view rendering, motivating research to address real-world challenges. A critical issue is the camera motion blur caused by movement during exposure, which hinders accurate 3D scene reconstruction. In this study, we propose CoMoGaussian, a Continuous Motion-Aware Gaussian Splatting that reconstructs precise 3D scenes from motion-blurred images while maintaining real-time rendering speed. Considering the complex motion patterns inherent in real-world camera movements, we predict continuous camera trajectories using neural ordinary differential equations (ODEs). To ensure accurate modeling, we employ rigid body transformations, preserving the shape and size of the object but rely on the discrete integration of sampled frames. To better approximate the continuous nature of motion blur, we introduce a continuous motion refinement (CMR) transformation that refines rigid transformations by incorporating additional learnable parameters. By revisiting fundamental camera theory and leveraging advanced neural ODE techniques, we achieve precise modeling of continuous camera trajectories, leading to improved reconstruction accuracy. Extensive experiments demonstrate state-of-the-art performance both quantitatively and qualitatively on benchmark datasets, which include a wide range of motion blur scenarios, from moderate to extreme blur.

3D 高斯散点 (3D Gaussian Splatting, 3DGS) 由于其高质量的新视角渲染能力，近年来受到广泛关注，推动了针对真实世界挑战的研究。其中，一个关键问题是相机运动模糊 (Camera Motion Blur)，即由于曝光过程中相机的运动导致的模糊现象，这一问题严重影响了准确的 3D 场景重建。
在本研究中，我们提出 CoMoGaussian，即一种连续运动感知的高斯散点 (Continuous Motion-Aware Gaussian Splatting)，能够从运动模糊图像中精确重建 3D 场景，同时保持实时渲染的性能。
考虑到真实世界相机运动的复杂性，我们采用神经常微分方程 (Neural Ordinary Differential Equations, ODEs) 来预测连续相机轨迹。为了确保建模的准确性，我们引入刚体变换 (Rigid Body Transformations)，在保留物体形状和大小的前提下，通过离散积分 (Discrete Integration) 处理采样帧。然而，为了更精确地逼近连续运动模糊 (Continuous Motion Blur)，我们进一步提出连续运动优化 (Continuous Motion Refinement, CMR) 变换，该方法在刚体变换的基础上引入额外的可学习参数，以优化相机运动建模。
通过重新审视基础相机理论 (Fundamental Camera Theory) 并结合先进的神经 ODE 技术，我们的模型能够对连续相机轨迹进行精确建模，从而提升 3D 重建的准确性。大量实验表明，在多个基准数据集上，我们的方法在定量指标 (Quantitative Metrics) 和定性效果 (Qualitative Results) 方面均达到了当前最先进的性能 (State-of-the-Art Performance)，能够处理从中等模糊 (Moderate Blur) 到极端模糊 (Extreme Blur) 的多种运动模糊场景。


---

## [28] SplatTalk: 3D VQA with Gaussian Splatting

### SplatTalk: 3D VQA with Gaussian Splatting

Language-guided 3D scene understanding is important for advancing applications in robotics, AR/VR, and human-computer interaction, enabling models to comprehend and interact with 3D environments through natural language. While 2D vision-language models (VLMs) have achieved remarkable success in 2D VQA tasks, progress in the 3D domain has been significantly slower due to the complexity of 3D data and the high cost of manual annotations. In this work, we introduce SplatTalk, a novel method that uses a generalizable 3D Gaussian Splatting (3DGS) framework to produce 3D tokens suitable for direct input into a pretrained LLM, enabling effective zero-shot 3D visual question answering (3D VQA) for scenes with only posed images. During experiments on multiple benchmarks, our approach outperforms both 3D models trained specifically for the task and previous 2D-LMM-based models utilizing only images (our setting), while achieving competitive performance with state-of-the-art 3D LMMs that additionally utilize 3D inputs.

三维高斯散点（3D Gaussian Splatting，3DGS）的出现推动了三维场景重建和新视角合成的发展。随着对需要即时反馈的交互式应用的关注不断增长，实时在线 3DGS 重建的需求也日益增加。然而，现有方法尚无法满足这一需求，主要受到以下三大挑战的限制：缺乏预设相机参数、需要可泛化的 3DGS 优化，以及减少冗余的必要性。
为此，我们提出 StreamGS，一种用于无位姿图像流的在线可泛化 3DGS 重建方法，该方法通过预测和聚合逐帧高斯点，逐步将图像流转换为 3D 高斯流。我们的方法克服了初始点重建方法 \cite{dust3r} 在处理域外（OOD）场景时的局限性，引入了一种内容自适应优化（content adaptive refinement）。该优化方法通过在相邻帧之间建立可靠的像素对应关系来增强跨帧一致性。这种对应关系进一步帮助通过跨帧特征聚合合并冗余高斯点，从而减少高斯点的密度，大幅降低计算和内存开销，使在线重建成为可能。
在多个不同数据集上的广泛实验表明，StreamGS 在重建质量上可与基于优化的方法相媲美，但速度提高 150 倍，同时在处理 OOD 场景时展现出更强的泛化能力。


---

## [29] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting

### 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting

Real-time rendering of dynamic scenes with view-dependent effects remains a fundamental challenge in computer graphics. While recent advances in Gaussian Splatting have shown promising results separately handling dynamic scenes (4DGS) and view-dependent effects (6DGS), no existing method unifies these capabilities while maintaining real-time performance. We present 7D Gaussian Splatting (7DGS), a unified framework representing scene elements as seven-dimensional Gaussians spanning position (3D), time (1D), and viewing direction (3D). Our key contribution is an efficient conditional slicing mechanism that transforms 7D Gaussians into view- and time-conditioned 3D Gaussians, maintaining compatibility with existing 3D Gaussian Splatting pipelines while enabling joint optimization. Experiments demonstrate that 7DGS outperforms prior methods by up to 7.36 dB in PSNR while achieving real-time rendering (401 FPS) on challenging dynamic scenes with complex view-dependent effects.

实时渲染具有视角相关效应的动态场景仍然是计算机图形学中的一项基本挑战。尽管高斯投影（Gaussian Splatting）技术的最新进展分别在处理动态场景（4DGS）和视角相关效应（6DGS）方面取得了显著成果，但目前尚无方法能够在保持实时性能的同时统一这两种能力。我们提出 7D 高斯投影（7D Gaussian Splatting, 7DGS），这是一种统一框架，将场景元素表示为七维高斯，涵盖 位置（3D）、时间（1D）和视角方向（3D）。我们的核心贡献是一种高效的 条件切片机制（conditional slicing mechanism），能够将 7D 高斯转换为 视角-时间条件化的 3D 高斯，从而在兼容现有 3D 高斯投影流水线的同时，实现联合优化。实验结果表明，7DGS 在 PSNR 方面比现有方法最高提升 7.36 dB，同时在具有复杂视角相关效应的动态场景中实现了实时渲染（401 FPS）。


---

## [30] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction

### DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction

Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments.

从真实世界捕捉中重建干净且无干扰的三维场景仍然是一个重大挑战，特别是在高度动态和杂乱的环境中，如自我中心的视频。为了解决这个问题，我们提出了DeGauss，这是一个简单且稳健的自监督框架，基于解耦的动态-静态高斯点云渲染设计进行动态场景重建。DeGauss使用前景高斯点来建模动态元素，使用背景高斯点来建模静态内容，采用概率掩膜来协调它们的组合，使得动态和静态部分能够独立但互补地优化。DeGauss在广泛的真实场景中表现出稳健的泛化能力，从随意的图像集合到长时间、动态的自我中心视频，均无需依赖复杂的启发式方法或大量的监督。我们在多个基准数据集（包括NeRF-on-the-go、ADT、AEA、Hot3D和EPIC-Fields）上的实验表明，DeGauss始终优于现有方法，为在高度动态且富有互动的环境中进行通用、无干扰的三维重建奠定了坚实的基准。


---

## [31] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation

### Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation

Recent advances in text-to-image diffusion models have been driven by the increasing availability of paired 2D data. However, the development of 3D diffusion models has been hindered by the scarcity of high-quality 3D data, resulting in less competitive performance compared to their 2D counterparts. To address this challenge, we propose repurposing pre-trained 2D diffusion models for 3D object generation. We introduce Gaussian Atlas, a novel representation that utilizes dense 2D grids, enabling the fine-tuning of 2D diffusion models to generate 3D Gaussians. Our approach demonstrates successful transfer learning from a pre-trained 2D diffusion model to a 2D manifold flattened from 3D structures. To support model training, we compile GaussianVerse, a large-scale dataset comprising 205K high-quality 3D Gaussian fittings of various 3D objects. Our experimental results show that text-to-image diffusion models can be effectively adapted for 3D content generation, bridging the gap between 2D and 3D modeling.

近年来，文本到图像的扩散模型（text-to-image diffusion models）取得了显著进展，主要得益于配对 2D 数据 的不断增加。然而，3D 扩散模型的发展受限于高质量 3D 数据 的匮乏，导致其与 2D 模型相比性能较为逊色。为了解决这一挑战，我们提出了将预训练的 2D 扩散模型重新用于 3D 物体生成的方法。
我们引入了 Gaussian Atlas，一种新的表示方式，利用密集的 2D 网格，使得 2D 扩散模型能够微调以生成 3D 高斯点。我们的方法展示了从预训练的 2D 扩散模型到从 3D 结构展开的 2D 流形的成功迁移学习。为了支持模型训练，我们编制了 GaussianVerse，一个大规模数据集，包含了 205K 个高质量的 3D 高斯拟合数据，涵盖各种 3D 物体。
实验结果表明，文本到图像的扩散模型可以有效地适应 3D 内容生成，从而弥合了 2D 和 3D 建模之间的差距。


---

## [32] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering

### OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering

In large-scale scene reconstruction using 3D Gaussian splatting, it is common to partition the scene into multiple smaller regions and reconstruct them individually. However, existing division methods are occlusion-agnostic, meaning that each region may contain areas with severe occlusions. As a result, the cameras within those regions are less correlated, leading to a low average contribution to the overall reconstruction. In this paper, we propose an occlusion-aware scene division strategy that clusters training cameras based on their positions and co-visibilities to acquire multiple regions. Cameras in such regions exhibit stronger correlations and a higher average contribution, facilitating high-quality scene reconstruction. We further propose a region-based rendering technique to accelerate large scene rendering, which culls Gaussians invisible to the region where the viewpoint is located. Such a technique significantly speeds up the rendering without compromising quality. Extensive experiments on multiple large scenes show that our method achieves superior reconstruction results with faster rendering speed compared to existing state-of-the-art approaches.

在使用 3D 高斯散点 (3DGS) 进行大规模场景重建时，通常会将场景划分为多个较小的区域，并分别进行重建。然而，现有的划分方法忽略了遮挡信息，导致每个区域可能包含大量严重遮挡的区域。由于这些区域内的相机之间相关性较低，使得它们对整体重建的贡献度较低，进而影响最终的重建质量。
为了解决这一问题，我们提出了一种基于遮挡感知的场景划分策略，该策略基于相机的位置和共视信息 (co-visibility) 进行聚类，从而生成多个子区域。在这些区域内，相机之间的相关性更强，平均贡献度更高，从而促进高质量的场景重建。
此外，我们进一步提出了一种基于区域的渲染技术，用于加速大规模场景的渲染。该方法能够剔除当前视角所在区域不可见的高斯点，从而在不影响渲染质量的前提下显著提升渲染速度。
在多个大规模场景上的实验表明，与现有最先进的方法相比，我们的方法能够实现更优的重建质量，同时显著提升渲染效率。


---

## [33] X^2-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction

### X2-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction

Four-dimensional computed tomography (4D CT) reconstruction is crucial for capturing dynamic anatomical changes but faces inherent limitations from conventional phase-binning workflows. Current methods discretize temporal resolution into fixed phases with respiratory gating devices, introducing motion misalignment and restricting clinical practicality. In this paper, We propose X2-Gaussian, a novel framework that enables continuous-time 4D-CT reconstruction by integrating dynamic radiative Gaussian splatting with self-supervised respiratory motion learning. Our approach models anatomical dynamics through a spatiotemporal encoder-decoder architecture that predicts time-varying Gaussian deformations, eliminating phase discretization. To remove dependency on external gating devices, we introduce a physiology-driven periodic consistency loss that learns patient-specific breathing cycles directly from projections via differentiable optimization. Extensive experiments demonstrate state-of-the-art performance, achieving a 9.93 dB PSNR gain over traditional methods and 2.25 dB improvement against prior Gaussian splatting techniques. By unifying continuous motion modeling with hardware-free period learning, X2-Gaussian advances high-fidelity 4D CT reconstruction for dynamic clinical imaging.

四维计算机断层扫描（4D CT）重建对于捕捉动态解剖变化至关重要，但传统的相位分箱流程存在固有局限。目前的方法通常将时间分辨率离散为固定相位，并依赖呼吸门控设备，从而引入运动错位，限制了临床实用性。
本文中，我们提出了一种新框架 X2-Gaussian，通过结合动态辐射高斯投影与自监督呼吸运动学习，实现了连续时间的 4D CT 重建。我们的方法基于时空编码-解码架构建模解剖动态，预测随时间变化的高斯形变，从而消除了相位离散化的需求。
为摆脱对外部门控设备的依赖，我们引入了一种基于生理节律的周期一致性损失函数，可通过可微优化从投影数据中直接学习患者特异性的呼吸周期。大量实验表明，X2-Gaussian 在性能上达到当前最优，相较于传统方法提升了 9.93 dB 的 PSNR，相较于现有高斯投影方法也提高了 2.25 dB。
通过将连续运动建模与无硬件周期学习统一起来，X2-Gaussian 推进了高保真 4D CT 重建在动态临床影像中的发展。


---

## [34] FlowR: Flowing from Sparse to Dense 3D Reconstructions

### FlowR: Flowing from Sparse to Dense 3D Reconstructions

3D Gaussian splatting enables high-quality novel view synthesis (NVS) at real-time frame rates. However, its quality drops sharply as we depart from the training views. Thus, dense captures are needed to match the high-quality expectations of some applications, e.g. Virtual Reality (VR). However, such dense captures are very laborious and expensive to obtain. Existing works have explored using 2D generative models to alleviate this requirement by distillation or generating additional training views. These methods are often conditioned only on a handful of reference input views and thus do not fully exploit the available 3D information, leading to inconsistent generation results and reconstruction artifacts. To tackle this problem, we propose a multi-view, flow matching model that learns a flow to connect novel view renderings from possibly sparse reconstructions to renderings that we expect from dense reconstructions. This enables augmenting scene captures with novel, generated views to improve reconstruction quality. Our model is trained on a novel dataset of 3.6M image pairs and can process up to 45 views at 540x960 resolution (91K tokens) on one H100 GPU in a single forward pass. Our pipeline consistently improves NVS in sparse- and dense-view scenarios, leading to higher-quality reconstructions than prior works across multiple, widely-used NVS benchmarks.

三维高斯喷溅（3D Gaussian Splatting）能够以实时帧率实现高质量的新视角合成（Novel View Synthesis, NVS）。然而，当观察角度偏离训练视角时，其渲染质量会显著下降。因此，为满足某些应用（如虚拟现实 VR）对高质量的要求，通常需要密集的数据采集，而这类采集过程往往代价高昂且极为繁琐。
已有研究尝试借助二维生成模型，通过蒸馏或生成额外训练视角来缓解对密集采集的依赖。然而，这些方法通常仅基于少量参考视图进行条件生成，未能充分利用可用的三维信息，导致生成结果存在不一致与重建伪影等问题。
为解决这一问题，我们提出了一种多视角流匹配模型，该模型学习一种映射流（flow），将来自稀疏重建的视角渲染结果对齐至期望的密集重建渲染结果，从而支持利用生成的新视角增强场景采集数据，提升重建质量。
我们的方法基于一个包含 360 万图像对的新数据集进行训练，并能在单张 H100 GPU 上以单次前向过程处理多达 45 个视角、540×960 分辨率（共 91K token）的输入。
该流程在稀疏视角与密集视角场景下均显著提升新视角合成效果，在多个主流 NVS 基准上实现了超过现有方法的高质量重建表现。


---

## [35] SIGMAN: Scaling 3D Human Gaussian Generation with Millions of Assets

### SIGMAN: Scaling 3D Human Gaussian Generation with Millions of Assets

3D human digitization has long been a highly pursued yet challenging task. Existing methods aim to generate high-quality 3D digital humans from single or multiple views, but remain primarily constrained by current paradigms and the scarcity of 3D human assets. Specifically, recent approaches fall into several paradigms: optimization-based and feed-forward (both single-view regression and multi-view generation with reconstruction). However, they are limited by slow speed, low quality, cascade reasoning, and ambiguity in mapping low-dimensional planes to high-dimensional space due to occlusion and invisibility, respectively. Furthermore, existing 3D human assets remain small-scale, insufficient for large-scale training. To address these challenges, we propose a latent space generation paradigm for 3D human digitization, which involves compressing multi-view images into Gaussians via a UV-structured VAE, along with DiT-based conditional generation, we transform the ill-posed low-to-high-dimensional mapping problem into a learnable distribution shift, which also supports end-to-end inference. In addition, we employ the multi-view optimization approach combined with synthetic data to construct the HGS-1M dataset, which contains 1 million 3D Gaussian assets to support the large-scale training. Experimental results demonstrate that our paradigm, powered by large-scale training, produces high-quality 3D human Gaussians with intricate textures, facial details, and loose clothing deformation.

三维人体数字化一直是一个高度追求但极具挑战性的任务。现有方法主要致力于从单视图或多视图生成高质量的三维数字人，但受限于当前范式和三维人体数据资产的稀缺性，始终面临瓶颈。具体而言，近期方法主要可归类为以下几种范式：基于优化的方法，以及前馈式的方法（包括单视图回归和结合重建的多视图生成）。然而，这些方法分别受到速度慢、质量低、推理流程冗长，以及由于遮挡和不可见性导致的低维到高维映射歧义等问题的限制。
此外，现有三维人体资产规模有限，难以满足大规模训练需求。为了解决上述挑战，我们提出了一种用于三维人体数字化的潜空间生成范式。该方法通过 UV 结构的变分自编码器（VAE） 将多视图图像压缩为高斯表示，并结合 DiT（Diffusion Transformer） 进行条件生成，将原本病态的低维到高维映射问题转化为可学习的分布迁移过程，同时支持端到端推理。
此外，我们结合多视图优化方法与合成数据构建了 HGS-1M 数据集，包含 一百万个三维高斯人体资产，以支持大规模训练。实验结果表明，得益于大规模数据支撑，我们的方法能够生成具有精细纹理、面部细节以及宽松衣物变形的高质量三维高斯人体表示。


---

## [36] DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting

### DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting

Creating relightable and animatable human avatars from monocular videos is a rising research topic with a range of applications, e.g. virtual reality, sports, and video games. Previous works utilize neural fields together with physically based rendering (PBR), to estimate geometry and disentangle appearance properties of human avatars. However, one drawback of these methods is the slow rendering speed due to the expensive Monte Carlo ray tracing. To tackle this problem, we proposed to distill the knowledge from implicit neural fields (teacher) to explicit 2D Gaussian splatting (student) representation to take advantage of the fast rasterization property of Gaussian splatting. To avoid ray-tracing, we employ the split-sum approximation for PBR appearance. We also propose novel part-wise ambient occlusion probes for shadow computation. Shadow prediction is achieved by querying these probes only once per pixel, which paves the way for real-time relighting of avatars. These techniques combined give high-quality relighting results with realistic shadow effects. Our experiments demonstrate that the proposed student model achieves comparable or even better relighting results with our teacher model while being 370 times faster at inference time, achieving a 67 FPS rendering speed.

从单目视频中构建可重光照、可驱动的人体虚拟化身，已成为当前备受关注的研究方向，广泛应用于虚拟现实、体育分析与电子游戏等场景。已有方法通常结合神经场与基于物理的渲染（Physically Based Rendering, PBR）框架，以估计人体几何并解耦外观属性。然而，这些方法普遍面临渲染速度缓慢的问题，原因在于其依赖计算开销极高的蒙特卡洛光线追踪。
为解决这一问题，我们提出将隐式神经场（teacher）中的知识蒸馏至显式的二维高斯投影（student）表示中，充分利用 Gaussian Splatting 的快速光栅化特性，实现高效渲染。为避免光线追踪，我们采用了 split-sum 近似算法以完成 PBR 外观建模，并提出了新颖的 部位级环境光遮蔽探针用于阴影计算。阴影预测仅需对每个像素查询一次这些探针，极大提升了实时重光照能力。
上述技术相结合，实现了具有逼真阴影效果的高质量重光照效果。实验表明，我们提出的 student 模型在保有与 teacher 模型相当甚至更佳的重光照表现的同时，推理速度提升达 370 倍，达到 67 FPS 实时渲染帧率，为可重光照虚拟人技术的实用化迈出了关键一步。


---

## [37] GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR

### GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR

We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature.

我们提出了 GaSLight，一种可从普通图像生成空间可变光照的方法。该方法首次将 HDR 高斯投影（HDR Gaussian Splats） 作为光源表示，使得普通图像得以在三维渲染器中直接作为光源使用。
我们的方法采用两阶段流程：第一阶段通过利用扩散模型中蕴含的先验信息，对图像的动态范围进行可信且准确的增强；第二阶段则使用高斯投影对三维光照进行建模，从而实现空间可变的照明效果。
在高动态范围估计及其在虚拟物体与场景照明中的应用方面，我们的方法取得了当前最优性能。为推动“图像作为光源”这一研究方向的评估标准建设，我们还构建了一个全新的标定且未过曝的 HDR 数据集，用于评估图像作为光源的表现。我们的方法通过该新数据集与已有文献中的公开数据集相结合进行评估，验证了其有效性。


---

## [38] HUG: Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction

### HUG: Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction

As urban 3D scenes become increasingly complex and the demand for high-quality rendering grows, efficient scene reconstruction and rendering techniques become crucial. We present HUG, a novel approach to address inefficiencies in handling large-scale urban environments and intricate details based on 3D Gaussian splatting. Our method optimizes data partitioning and the reconstruction pipeline by incorporating a hierarchical neural Gaussian representation. We employ an enhanced block-based reconstruction pipeline focusing on improving reconstruction quality within each block and reducing the need for redundant training regions around block boundaries. By integrating neural Gaussian representation with a hierarchical architecture, we achieve high-quality scene rendering at a low computational cost. This is demonstrated by our state-of-the-art results on public benchmarks, which prove the effectiveness and advantages in large-scale urban scene representation.

随着城市三维场景日益复杂，以及对高质量渲染需求的增长，高效的场景重建与渲染技术变得尤为关键。我们提出 HUG，一种基于三维高斯投影（3D Gaussian Splatting）的方法，旨在解决处理大规模城市环境与复杂细节时的效率问题。该方法通过引入分层神经高斯表示，优化了数据划分与重建流程。我们采用增强的基于块的重建管线，重点提升每个块内的重建质量，并减少块边界周围冗余训练区域的需求。通过将神经高斯表示与分层结构相结合，HUG 实现了低计算成本下的高质量场景渲染。我们在公开基准数据集上的实验结果表明，HUG 在大规模城市场景表达方面达到了当前最先进水平，验证了其有效性与优势。


---

## [39] Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptional Learning

### Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptional Learning

3D Gaussian Splatting (3DGS) has emerged as a powerful and efficient 3D representation for novel view synthesis. This paper extends 3DGS capabilities to inpainting, where masked objects in a scene are replaced with new contents that blend seamlessly with the surroundings. Unlike 2D image inpainting, 3D Gaussian inpainting (3DGI) is challenging in effectively leveraging complementary visual and semantic cues from multiple input views, as occluded areas in one view may be visible in others. To address this, we propose a method that measures the visibility uncertainties of 3D points across different input views and uses them to guide 3DGI in utilizing complementary visual cues. We also employ uncertainties to learn a semantic concept of scene without the masked object and use a diffusion model to fill masked objects in input images based on the learned concept. Finally, we build a novel 3DGI framework, VISTA, by integrating VISibility-uncerTainty-guided 3DGI with scene conceptuAl learning. VISTA generates high-quality 3DGS models capable of synthesizing artifact-free and naturally inpainted novel views. Furthermore, our approach extends to handling dynamic distractors arising from temporal object changes, enhancing its versatility in diverse scene reconstruction scenarios. We demonstrate the superior performance of our method over state-of-the-art techniques using two challenging datasets: the SPIn-NeRF dataset, featuring 10 diverse static 3D inpainting scenes, and an underwater 3D inpainting dataset derived from UTB180, including fast-moving fish as inpainting targets.

3D Gaussian Splatting（3DGS）作为一种高效而强大的三维表示形式，在新视角合成任务中表现突出。本文将 3DGS 的能力拓展至补全任务（inpainting），即将场景中被遮挡或移除的目标以与周围环境自然融合的内容进行替代。不同于二维图像补全，**三维高斯补全（3DGI）**面临更大挑战：如何有效利用多视图中互补的视觉和语义线索，尤其当某一视角中遮挡区域在其他视角中可见时。
为此，我们提出一种方法，通过衡量三维点在不同输入视角下的可见性不确定性，引导 3DGI 更好地利用互补视觉线索。同时，我们利用这些不确定性学习一个去除遮挡物后的场景语义概念，并基于该概念使用扩散模型对输入图像中的遮挡区域进行填补。最终，我们构建了一个新的三维补全框架 VISTA，将可见性不确定性引导的 3DGI与场景语义学习相结合，生成高质量的 3DGS 模型，能够合成无伪影、自然过渡的补全视图。
此外，我们的方法还能处理由于时间变化引起的动态干扰目标，提升其在多样化场景重建任务中的适应性。我们在两个具有挑战性的数据集上验证了方法的优越性：一是包含 10 个多样静态 3D 补全过程景的 SPIn-NeRF 数据集；二是从 UTB180 构建的水下三维补全数据集，其中以高速游动的鱼类为补全目标。实验结果表明，我们的方法在补全质量与视图一致性方面均优于现有最先进技术。


---

## [40] Sparfels: Fast Reconstruction from Sparse Unposed Imagery

### Sparfels: Fast Reconstruction from Sparse Unposed Imagery

We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.

我们提出了一种基于表面元素泼溅的稀疏视角重建方法，可在消费级 GPU 上于 3 分钟内完成运行。尽管已有少数方法尝试从噪声或未标定的稀疏相机中学习稀疏辐射场，但在该设定下的形状恢复问题仍相对较少被研究。现有部分辐射场与形状学习方法主要针对已知位姿的稀疏设定，通过学习数据先验或结合外部单目几何先验来实现。
与之不同，我们提出了一种高效且简洁的重建流程，仅依赖一个最新的三维基础模型。我们利用该基础模型提供的多任务输出头，特别是点图（point maps）和相机初始化，用于构建并调整一个二维高斯泼溅（2D Gaussian Splatting, 2DGS）模型，同时利用图像间的对应关系来辅助训练过程中的相机优化。
我们工作的关键贡献之一是提出了一种新的射线上泼溅颜色方差公式化方法，该方法可高效计算并在训练过程中最小化，从而提升形状重建的精度。
在多个标准多视图数据集上进行的稀疏未标定设定下的重建与新视角合成测试表明，我们的方法在精度上达到了当前最先进水平。


---

## [41] GUAVA: Generalizable Upper Body 3D Gaussian Avatar

### GUAVA: Generalizable Upper Body 3D Gaussian Avatar

Reconstructing a high-quality, animatable 3D human avatar with expressive facial and hand motions from a single image has gained significant attention due to its broad application potential. 3D human avatar reconstruction typically requires multi-view or monocular videos and training on individual IDs, which is both complex and time-consuming. Furthermore, limited by SMPLX's expressiveness, these methods often focus on body motion but struggle with facial expressions. To address these challenges, we first introduce an expressive human model (EHM) to enhance facial expression capabilities and develop an accurate tracking method. Based on this template model, we propose GUAVA, the first framework for fast animatable upper-body 3D Gaussian avatar reconstruction. We leverage inverse texture mapping and projection sampling techniques to infer Ubody (upper-body) Gaussians from a single image. The rendered images are refined through a neural refiner. Experimental results demonstrate that GUAVA significantly outperforms previous methods in rendering quality and offers significant speed improvements, with reconstruction times in the sub-second range (0.1s), and supports real-time animation and rendering.

从单张图像中重建具有面部与手部表情的高质量可动画三维人体头像，因其广泛的应用潜力而受到广泛关注。传统的三维人体头像重建方法通常依赖多视角或单目视频，并需针对每个个体进行训练，过程复杂且耗时。此外，受限于 SMPLX 模型的表达能力，此类方法虽能处理躯干运动，但在面部表情建模方面表现不足。
为解决上述问题，我们首先提出了一个增强面部表达能力的 EHM（Expressive Human Model），并在此基础上开发了精确的追踪方法。基于该模板模型，我们进一步提出 GUAVA，首个面向快速可动画上半身三维高斯头像重建的框架。
GUAVA 利用反向纹理映射与投影采样技术，从单张图像中推理出上半身（Ubody）高斯图元，并通过神经细化器对渲染结果进行优化。实验结果表明，GUAVA 在渲染质量上显著优于现有方法，重建速度达到亚秒级（0.1 秒），支持实时动画与渲染。


---

## [42] QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization

### QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization

Surface reconstruction is fundamental to computer vision and graphics, enabling applications in 3D modeling, mixed reality, robotics, and more. Existing approaches based on volumetric rendering obtain promising results, but optimize on a per-scene basis, resulting in a slow optimization that can struggle to model under-observed or textureless regions. We introduce QuickSplat, which learns data-driven priors to generate dense initializations for 2D gaussian splatting optimization of large-scale indoor scenes. This provides a strong starting point for the reconstruction, which accelerates the convergence of the optimization and improves the geometry of flat wall structures. We further learn to jointly estimate the densification and update of the scene parameters during each iteration; our proposed densifier network predicts new Gaussians based on the rendering gradients of existing ones, removing the needs of heuristics for densification. Extensive experiments on large-scale indoor scene reconstruction demonstrate the superiority of our data-driven optimization. Concretely, we accelerate runtime by 8x, while decreasing depth errors by up to 48% in comparison to state of the art methods.

表面重建是计算机视觉与图形学中的基础任务，支撑着三维建模、混合现实、机器人等多种应用。现有基于体渲染的方法已取得了令人瞩目的成果，但往往依赖每个场景独立优化，导致优化过程缓慢，并在观测不足或纹理稀缺区域表现不佳。
我们提出 QuickSplat，一种利用数据驱动先验为大规模室内场景的二维高斯泼溅（2D Gaussian Splatting）优化生成稠密初始化的新方法。该初始化为重建提供了强有力的起点，加速了优化收敛过程，并显著改善了平坦墙面等结构的几何质量。
此外，我们进一步设计了一种联合估计密化与场景参数更新的策略。我们提出的 densifier 网络 通过现有高斯的渲染梯度预测新高斯点，从而无需依赖传统的启发式密化策略。
在大规模室内场景重建上的广泛实验表明，我们的优化框架具有明显优势。具体而言，QuickSplat 在保持或提升重建质量的同时，将运行时间加速了 8 倍，并将深度误差最多降低了 48%，显著优于现有最先进方法。


---

## [43] SpatialCrafter: Unleashing the Imagination of Video Diffusion Models for Scene Reconstruction from Limited Observations

### SpatialCrafter: Unleashing the Imagination of Video Diffusion Models for Scene Reconstruction from Limited Observations

Novel view synthesis (NVS) boosts immersive experiences in computer vision and graphics. Existing techniques, though progressed, rely on dense multi-view observations, restricting their application. This work takes on the challenge of reconstructing photorealistic 3D scenes from sparse or single-view inputs. We introduce SpatialCrafter, a framework that leverages the rich knowledge in video diffusion models to generate plausible additional observations, thereby alleviating reconstruction ambiguity. Through a trainable camera encoder and an epipolar attention mechanism for explicit geometric constraints, we achieve precise camera control and 3D consistency, further reinforced by a unified scale estimation strategy to handle scale discrepancies across datasets. Furthermore, by integrating monocular depth priors with semantic features in the video latent space, our framework directly regresses 3D Gaussian primitives and efficiently processes long-sequence features using a hybrid network structure. Extensive experiments show our method enhances sparse view reconstruction and restores the realistic appearance of 3D scenes.

新视角合成（Novel View Synthesis, NVS）在计算机视觉与图形学中极大提升了沉浸式体验。尽管现有技术已取得显著进展，但普遍依赖密集多视角观测，限制了其在实际应用中的可用性。本文针对从稀疏甚至单视角输入中重建真实感三维场景的挑战，提出了 SpatialCrafter 框架。
该框架通过利用视频扩散模型中蕴含的丰富先验知识，生成合理的补充观测，从而缓解重建歧义问题。我们设计了一个可训练的摄像机编码器以及带有显式几何约束的极线注意力机制（epipolar attention mechanism），实现了精确的摄像机控制与三维一致性，并引入统一的尺度估计策略以应对不同数据集间的尺度差异。
此外，SpatialCrafter 将单目深度先验与语义特征融合到视频潜空间中，直接回归生成三维高斯图元（3D Gaussian primitives），并通过混合神经网络结构高效处理长序列特征。大量实验表明，该方法在稀疏视角重建任务中表现优异，能够有效还原三维场景的真实外观。


---

## [44] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting

### CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting

Recent advances in 3D reconstruction techniques and vision-language models have fueled significant progress in 3D semantic understanding, a capability critical to robotics, autonomous driving, and virtual/augmented reality. However, methods that rely on 2D priors are prone to a critical challenge: cross-view semantic inconsistencies induced by occlusion, image blur, and view-dependent variations. These inconsistencies, when propagated via projection supervision, deteriorate the quality of 3D Gaussian semantic fields and introduce artifacts in the rendered outputs. To mitigate this limitation, we propose CCL-LGS, a novel framework that enforces view-consistent semantic supervision by integrating multi-view semantic cues. Specifically, our approach first employs a zero-shot tracker to align a set of SAM-generated 2D masks and reliably identify their corresponding categories. Next, we utilize CLIP to extract robust semantic encodings across views. Finally, our Contrastive Codebook Learning (CCL) module distills discriminative semantic features by enforcing intra-class compactness and inter-class distinctiveness. In contrast to previous methods that directly apply CLIP to imperfect masks, our framework explicitly resolves semantic conflicts while preserving category discriminability. Extensive experiments demonstrate that CCL-LGS outperforms previous state-of-the-art methods.

近年来，三维重建技术与视觉-语言模型的快速发展，极大推动了三维语义理解的进步，这一能力对于机器人、自主驾驶以及虚拟/增强现实等领域至关重要。然而，依赖二维先验的方法普遍面临一个关键挑战：由遮挡、图像模糊及视角依赖变化所导致的跨视角语义不一致性。这些不一致性在投影监督过程中被传递，会严重影响三维高斯语义场的质量，并在渲染结果中引入伪影。
为缓解这一问题，我们提出了 CCL-LGS，一种通过融合多视角语义信息实现视角一致语义监督的新型框架。具体而言，我们首先使用零样本追踪器对一组由 SAM 生成的二维掩码进行对齐，并可靠地识别其所属类别。随后，利用 CLIP 提取跨视角的稳健语义编码。最后，我们引入对比式码本学习（Contrastive Codebook Learning, CCL）模块，通过增强类内聚合与类间分离，提炼判别性语义特征。
与以往直接将 CLIP 应用于不完美掩码的方法不同，我们的框架显式地解决了语义冲突问题，同时保留了类别区分性。大量实验证明，CCL-LGS 在各项指标上均优于现有的最新方法。


---

## [45] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion

### AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion

Existing methods for image-to-3D avatar generation struggle to produce highly detailed, animation-ready avatars suitable for real-world applications. We introduce AdaHuman, a novel framework that generates high-fidelity animatable 3D avatars from a single in-the-wild image. AdaHuman incorporates two key innovations: (1) A pose-conditioned 3D joint diffusion model that synthesizes consistent multi-view images in arbitrary poses alongside corresponding 3D Gaussian Splats (3DGS) reconstruction at each diffusion step; (2) A compositional 3DGS refinement module that enhances the details of local body parts through image-to-image refinement and seamlessly integrates them using a novel crop-aware camera ray map, producing a cohesive detailed 3D avatar. These components allow AdaHuman to generate highly realistic standardized A-pose avatars with minimal self-occlusion, enabling rigging and animation with any input motion. Extensive evaluation on public benchmarks and in-the-wild images demonstrates that AdaHuman significantly outperforms state-of-the-art methods in both avatar reconstruction and reposing.

现有的图像到三维头像生成方法在生成具有高度细节、可用于动画制作的三维头像方面仍存在明显不足，难以满足现实应用需求。我们提出了 AdaHuman，一个能够从单张自然图像中生成高保真、可动画化三维头像的全新框架。

AdaHuman 包含两个关键创新点：
(1)	姿态条件 3D 关节点扩散模型：该模型可在任意姿态下合成一致的多视角图像，并在每一步扩散过程中同步完成对应的 3D Gaussian Splatting（3DGS）重建；
(2)	组合式 3DGS 细化模块：该模块通过图像到图像的细化方式增强局部身体部位的细节，并结合一种新颖的**裁剪感知相机光线映射（crop-aware camera ray map）**机制，实现各局部区域的无缝整合，最终生成一个结构完整、细节丰富的三维头像。
得益于以上设计，AdaHuman 能够生成高度写实的标准 A 姿态头像，自遮挡极小，便于进行骨骼绑定与任意动作的动画驱动。
在多个公开基准与真实自然图像上的广泛评估表明，AdaHuman 在头像重建与姿态迁移两个方面均显著优于当前最先进的方法。


---

## [46] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS

### RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS

3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling scenes affected by transient objects, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances. To address this, we propose RobustSplat, a robust solution based on two critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method.

3D Gaussian Splatting（3DGS）因其在新视角合成和三维建模中的实时、照片级真实感渲染能力而受到广泛关注。然而，现有方法在建模受瞬时物体影响的场景时表现不佳，导致渲染图像中出现伪影。我们发现，尽管高斯密化过程有助于捕捉场景细节，但其也会无意中引入建模瞬时干扰的额外高斯，从而加剧伪影问题。为了解决这一问题，我们提出了 RobustSplat，一个基于两项关键设计的鲁棒解决方案。
首先，我们引入了一种延迟高斯生长策略，该策略优先优化静态场景结构，随后才允许高斯的拆分/克隆，从而在优化初期减少对瞬时物体的过拟合。其次，我们设计了一种尺度级联的掩码自举方法，该方法首先利用低分辨率特征相似性监督进行初始瞬时掩码的估计，借助其更强的语义一致性与抗噪性；随后再过渡到高分辨率监督，以实现更精确的掩码预测。
在多个具有挑战性的数据集上的大量实验表明，我们的方法优于现有方法，清晰地展示了其鲁棒性与有效性。


---

## [47] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization

### CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization

In dynamic 3D environments, accurately updating scene representations over time is crucial for applications in robotics, mixed reality, and embodied AI. As scenes evolve, efficient methods to incorporate changes are needed to maintain up-to-date, high-quality reconstructions without the computational overhead of re-optimizing the entire scene. This paper introduces CL-Splats, which incrementally updates Gaussian splatting-based 3D representations from sparse scene captures. CL-Splats integrates a robust change-detection module that segments updated and static components within the scene, enabling focused, local optimization that avoids unnecessary re-computation. Moreover, CL-Splats supports storing and recovering previous scene states, facilitating temporal segmentation and new scene-analysis applications. Our extensive experiments demonstrate that CL-Splats achieves efficient updates with improved reconstruction quality over the state-of-the-art. This establishes a robust foundation for future real-time adaptation in 3D scene reconstruction tasks.

在动态三维环境中，随着场景的不断演化，如何高效、准确地更新场景表示对于机器人技术、混合现实以及具身智能等应用至关重要。为了保持最新且高质量的重建效果，亟需一种能够高效引入场景变化的方法，同时避免对整个场景进行高开销的重新优化。本文提出了 CL-Splats，一种基于高斯投影的三维表示增量更新方法，可从稀疏的场景捕捉中逐步构建更新内容。CL-Splats 集成了一个强健的变化检测模块，能够将场景中的变动部分与静态部分进行分割，从而实现聚焦式的局部优化，避免不必要的重复计算。此外，CL-Splats 支持历史场景状态的存储与恢复，便于进行时间序列分割与新型场景分析任务。大量实验表明，CL-Splats 在更新效率和重建质量方面均优于现有方法，为未来实时三维场景重建任务中的自适应更新奠定了坚实基础。


---

## [48] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction

### Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction

This paper presents an end-to-end framework for reconstructing 3D parametric curves directly from multi-view edge maps. Contrasting with existing two-stage methods that follow a sequential "edge point cloud reconstruction and parametric curve fitting" pipeline, our one-stage approach optimizes 3D parametric curves directly from 2D edge maps, eliminating error accumulation caused by the inherent optimization gap between disconnected stages. However, parametric curves inherently lack suitability for rendering-based multi-view optimization, necessitating a complementary representation that preserves their geometric properties while enabling differentiable rendering. We propose a novel bi-directional coupling mechanism between parametric curves and edge-oriented Gaussian components. This tight correspondence formulates a curve-aware Gaussian representation, \textbf{CurveGaussian}, that enables differentiable rendering of 3D curves, allowing direct optimization guided by multi-view evidence. Furthermore, we introduce a dynamically adaptive topology optimization framework during training to refine curve structures through linearization, merging, splitting, and pruning operations. Comprehensive evaluations on the ABC dataset and real-world benchmarks demonstrate our one-stage method's superiority over two-stage alternatives, particularly in producing cleaner and more robust reconstructions. Additionally, by directly optimizing parametric curves, our method significantly reduces the parameter count during training, achieving both higher efficiency and superior performance compared to existing approaches.

本文提出了一种端到端框架，可直接从多视角边缘图中重建三维参数曲线。与现有采用“边缘点云重建 + 参数曲线拟合”两阶段顺序流程的方法不同，我们的一阶段方法直接从二维边缘图优化三维参数曲线，避免了由于阶段分离导致的优化间隙所引起的误差累积。然而，参数曲线本身并不适用于基于渲染的多视角优化，因此需要一种既能保留其几何属性，又支持可微渲染的补充表示。为此，我们提出了一种参数曲线与边缘导向高斯组件之间的双向耦合机制。该紧密对应关系构成了一种曲线感知的高斯表示，称为 CurveGaussian，使得三维曲线具备可微渲染能力，能够基于多视角证据进行直接优化。此外，我们还引入了一种训练过程中的动态自适应拓扑优化框架，可通过线性化、合并、分裂和裁剪操作精细调整曲线结构。在 ABC 数据集和真实世界基准上的全面评估表明，该一阶段方法相较于两阶段方案在生成更干净且更鲁棒的重建结果方面具有显著优势。同时，由于直接优化参数曲线，我们的方法在训练中显著减少了参数数量，在保持高效率的同时也实现了优于现有方法的性能。


---

## [49] GGTalker: Talking Head Systhesis with Generalizable Gaussian Priors and Identity-Specific Adaptation

### GGTalker: Talking Head Systhesis with Generalizable Gaussian Priors and Identity-Specific Adaptation

Creating high-quality, generalizable speech-driven 3D talking heads remains a persistent challenge. Previous methods achieve satisfactory results for fixed viewpoints and small-scale audio variations, but they struggle with large head rotations and out-of-distribution (OOD) audio. Moreover, they are constrained by the need for time-consuming, identity-specific training. We believe the core issue lies in the lack of sufficient 3D priors, which limits the extrapolation capabilities of synthesized talking heads. To address this, we propose GGTalker, which synthesizes talking heads through a combination of generalizable priors and identity-specific adaptation. We introduce a two-stage Prior-Adaptation training strategy to learn Gaussian head priors and adapt to individual characteristics. We train Audio-Expression and Expression-Visual priors to capture the universal patterns of lip movements and the general distribution of head textures. During the Customized Adaptation, individual speaking styles and texture details are precisely modeled. Additionally, we introduce a color MLP to generate fine-grained, motion-aligned textures and a Body Inpainter to blend rendered results with the background, producing indistinguishable, photorealistic video frames. Comprehensive experiments show that GGTalker achieves state-of-the-art performance in rendering quality, 3D consistency, lip-sync accuracy, and training efficiency.

生成高质量、具有泛化能力的语音驱动三维数字人面部动画一直是一个具有挑战性的课题。以往方法在固定视角和小尺度音频变化下可实现令人满意的效果，但在面对大幅度头部旋转和超出训练分布（OOD）的音频时表现不佳。此外，这些方法通常依赖于耗时的身份特定训练过程，限制了其实用性。我们认为，其核心问题在于缺乏充分的三维先验，导致生成数字人面部动画时的外推能力受限。为此，我们提出了 **GGTalker**，通过结合可泛化的先验知识与身份特定的自适应机制，实现数字人口型动画的合成。我们引入了一个两阶段的“先验-适应”训练策略，先学习高斯人头先验，再进行个体化特征适应。我们训练了**音频-表情先验**与**表情-视觉先验**，用于捕捉口型运动的通用模式和人头纹理的整体分布。在个性化适应阶段，我们精确建模个体的说话风格和纹理细节。此外，我们引入了**颜色 MLP** 用于生成细粒度、与动作对齐的纹理图，并设计了**背景修复模块（Body Inpainter）**以将渲染结果自然融合至背景中，生成高度逼真的视频帧。大量实验表明，GGTalker 在渲染质量、三维一致性、唇形同步精度以及训练效率方面均达到了当前最优性能。


---

## [50] RoboPearls: Editable Video Simulation for Robot Manipulation

### RoboPearls: Editable Video Simulation for Robot Manipulation

The development of generalist robot manipulation policies has seen significant progress, driven by large-scale demonstration data across diverse environments. However, the high cost and inefficiency of collecting real-world demonstrations hinder the scalability of data acquisition. While existing simulation platforms enable controlled environments for robotic learning, the challenge of bridging the sim-to-real gap remains. To address these challenges, we propose RoboPearls, an editable video simulation framework for robotic manipulation. Built on 3D Gaussian Splatting (3DGS), RoboPearls enables the construction of photo-realistic, view-consistent simulations from demonstration videos, and supports a wide range of simulation operators, including various object manipulations, powered by advanced modules like Incremental Semantic Distillation (ISD) and 3D regularized NNFM Loss (3D-NNFM). Moreover, by incorporating large language models (LLMs), RoboPearls automates the simulation production process in a user-friendly manner through flexible command interpretation and execution. Furthermore, RoboPearls employs a vision-language model (VLM) to analyze robotic learning issues to close the simulation loop for performance enhancement. To demonstrate the effectiveness of RoboPearls, we conduct extensive experiments on multiple datasets and scenes, including RLBench, COLOSSEUM, Ego4D, Open X-Embodiment, and a real-world robot, which demonstrate our satisfactory simulation performance.

通用型机器人操作策略的发展在近年取得了显著进展，这得益于跨多样环境的大规模示范数据。然而，现实世界中示范数据的采集成本高且效率低，严重制约了数据获取的可扩展性。尽管现有仿真平台可以为机器人学习提供可控环境，但“从仿真到现实”（sim-to-real）的鸿沟仍是关键挑战。为应对这一问题，我们提出了 **RoboPearls** —— 一个用于机器人操作任务的可编辑视频仿真框架。该框架基于三维高斯投影（3D Gaussian Splatting, 3DGS），能够从示范视频中构建具有照片级真实感和视角一致性的仿真场景，并支持广泛的仿真操作，包括多种对象操控行为，这些能力由诸如**增量语义蒸馏（ISD）**和**三维正则化 NNFM 损失（3D-NNFM）**等先进模块提供支持。此外，RoboPearls 集成了大型语言模型（LLMs），可通过灵活的指令理解与执行，自动化生成仿真过程，提升用户交互友好性。同时，RoboPearls 还结合视觉-语言模型（VLM）对机器人学习过程中的问题进行分析，实现仿真闭环优化与性能增强。我们在多个数据集和场景上进行了大量实验，包括 RLBench、COLOSSEUM、Ego4D、Open X-Embodiment 以及真实机器人平台，实验结果表明 RoboPearls 在仿真效果方面表现出色，验证了其有效性。


---

## [51] VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding

### VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding

3D Gaussian Splatting (3DGS) has become horsepower in high-quality, real-time rendering for novel view synthesis of 3D scenes. However, existing methods focus primarily on geometric and appearance modeling, lacking deeper scene understanding while also incurring high training costs that complicate the originally streamlined differentiable rendering pipeline. To this end, we propose VoteSplat, a novel 3D scene understanding framework that integrates Hough voting with 3DGS. Specifically, Segment Anything Model (SAM) is utilized for instance segmentation, extracting objects, and generating 2D vote maps. We then embed spatial offset vectors into Gaussian primitives. These offsets construct 3D spatial votes by associating them with 2D image votes, while depth distortion constraints refine localization along the depth axis. For open-vocabulary object localization, VoteSplat maps 2D image semantics to 3D point clouds via voting points, reducing training costs associated with high-dimensional CLIP features while preserving semantic unambiguity. Extensive experiments demonstrate effectiveness of VoteSplat in open-vocabulary 3D instance localization, 3D point cloud understanding, click-based 3D object localization, hierarchical segmentation, and ablation studies.

三维高斯投影（3D Gaussian Splatting，3DGS）已成为高质量、实时新视角合成渲染的重要基础。然而，现有方法主要聚焦于几何与外观建模，缺乏对场景的深层理解，并且训练成本高昂，违背了其原本简洁的可微渲染管线设计初衷。为此，我们提出了 **VoteSplat** —— 一种融合 Hough 投票机制与 3DGS 的新型三维场景理解框架。具体而言，我们引入 Segment Anything Model（SAM）进行实例分割，提取物体并生成二维投票图。随后，我们将空间偏移向量嵌入至高斯基元中，并通过将这些偏移与二维图像投票关联，构建三维空间投票。同时，引入深度畸变约束以优化沿深度方向的定位精度。在开放词汇的物体定位任务中，VoteSplat 通过投票点将二维图像语义映射到三维点云，有效降低高维 CLIP 特征的训练成本，同时保留语义判别性。大量实验表明，VoteSplat 在开放词汇三维实例定位、三维点云理解、基于点击的三维物体定位、层次化分割及消融分析等任务中均展现出优异表现。


---

## [52] RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors

### RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors

A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality.

单次驾驶采集常常无法完整扫描道路结构，因此对重建场景进行扩展已成为传感器模拟器有效回归驾驶行为的关键需求。尽管当代三维高斯投影（3D Gaussian Splatting，3DGS）技术在重建质量方面表现出色，但其通过直接集成扩散先验进行扩展，往往会引入累积的物理不一致性，并降低训练效率。为克服上述局限，我们提出了 **RGE-GS** —— 一种融合扩散生成与奖励引导高斯集成的全新扩展式重建框架。RGE-GS 包含两项核心创新：其一，我们设计了一个奖励网络，在进入重建阶段前学习识别并优先保留稳定生成的结构模式，从而实现对扩散结果的选择性保留，提升空间稳定性；其二，在重建过程中，我们提出了一种差异化训练策略，能够根据场景收敛指标自动调节高斯优化进程，从而实现比基线方法更优的收敛效果。我们在多个公开数据集上进行了全面评估，结果表明 RGE-GS 在重建质量方面达到了当前最先进水平。


---

## [53] From Coarse to Fine: Learnable Discrete Wavelet Transforms for Efficient 3D Gaussian Splatting

### From Coarse to Fine: Learnable Discrete Wavelet Transforms for Efficient 3D Gaussian Splatting

3D Gaussian Splatting has emerged as a powerful approach in novel view synthesis, delivering rapid training and rendering but at the cost of an ever-growing set of Gaussian primitives that strains memory and bandwidth. We introduce AutoOpti3DGS, a training-time framework that automatically restrains Gaussian proliferation without sacrificing visual fidelity. The key idea is to feed the input images to a sequence of learnable Forward and Inverse Discrete Wavelet Transforms, where low-pass filters are kept fixed, high-pass filters are learnable and initialized to zero, and an auxiliary orthogonality loss gradually activates fine frequencies. This wavelet-driven, coarse-to-fine process delays the formation of redundant fine Gaussians, allowing 3DGS to capture global structure first and refine detail only when necessary. Through extensive experiments, AutoOpti3DGS requires just a single filter learning-rate hyper-parameter, integrates seamlessly with existing efficient 3DGS frameworks, and consistently produces sparser scene representations more compatible with memory or storage-constrained hardware.

三维高斯投影（3D Gaussian Splatting）作为一种新视角合成的强大方法，能够实现快速训练与渲染，但代价是高斯基元数量不断膨胀，从而对内存与带宽造成压力。为了解决这一问题，我们提出了 **AutoOpti3DGS** —— 一个训练阶段框架，可在不降低视觉保真的前提下自动抑制高斯基元的过度增长。其核心思想是将输入图像输入到一组可学习的正向与逆向离散小波变换（Discrete Wavelet Transforms）中，其中低通滤波器保持固定，高通滤波器则可学习并初始化为零，通过辅助正交损失逐步激活细节频率。该基于小波驱动的由粗到细训练策略，延迟了冗余精细高斯的生成，使 3DGS 能够优先捕捉全局结构，仅在必要时才细化局部细节。大量实验表明，AutoOpti3DGS 仅需一个滤波器学习率的超参数，能够无缝集成到现有高效 3DGS 框架中，始终生成更稀疏的场景表示，更加适用于内存或存储受限的硬件环境。


---

## [54] Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space

### Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space

Feature matching plays a fundamental role in many computer vision tasks, yet existing methods heavily rely on scarce and clean multi-view image collections, which constrains their generalization to diverse and challenging scenarios. Moreover, conventional feature encoders are typically trained on single-view 2D images, limiting their capacity to capture 3D-aware correspondences. In this paper, we propose a novel two-stage framework that lifts 2D images to 3D space, named as **Lift to Match (L2M)**, taking full advantage of large-scale and diverse single-view images. To be specific, in the first stage, we learn a 3D-aware feature encoder using a combination of multi-view image synthesis and 3D feature Gaussian representation, which injects 3D geometry knowledge into the encoder. In the second stage, a novel-view rendering strategy, combined with large-scale synthetic data generation from single-view images, is employed to learn a feature decoder for robust feature matching, thus achieving generalization across diverse domains. Extensive experiments demonstrate that our method achieves superior generalization across zero-shot evaluation benchmarks, highlighting the effectiveness of the proposed framework for robust feature matching.

特征匹配在许多计算机视觉任务中发挥着基础性作用，但现有方法严重依赖稀缺且干净的多视图图像集合，这限制了其在多样化和具有挑战性的场景中的泛化能力。此外，传统的特征编码器通常在单视图二维图像上训练，限制了其捕捉具备三维感知的对应关系的能力。本文提出了一种将二维图像提升到三维空间的新型两阶段框架，称为 **Lift to Match (L2M)**，充分利用了大规模、多样化的单视图图像。具体而言，在第一阶段，我们通过结合多视图图像合成与三维特征高斯表示，学习一个具备三维感知的特征编码器，从而将三维几何知识注入编码器。在第二阶段，我们采用新视角渲染策略，并结合从单视图图像生成的大规模合成数据，来训练特征解码器以实现稳健的特征匹配，从而在不同领域中实现泛化。大量实验表明，我们的方法在零样本评估基准上表现出优越的泛化能力，凸显了所提框架在稳健特征匹配中的有效性。


---

## [55] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation

### 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation

Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA.

物理对抗攻击方法揭示了深度神经网络的脆弱性，并对自动驾驶等安全关键场景构成了重大威胁。相比基于补丁的攻击，基于伪装的物理攻击在复杂物理环境中具有更强的对抗效果，因此更具潜力。然而，大多数现有工作依赖于目标物体的网格先验以及由模拟器构建的虚拟环境，这些先验的获取过程耗时且不可避免地与真实世界存在差异。此外，由于训练图像背景的局限性，现有方法往往难以生成具备多视角鲁棒性的对抗伪装，并容易陷入次优解。基于上述原因，以往方法在多样化视角和物理环境下的对抗有效性与鲁棒性均不足。为此，我们提出了一种基于三维高斯投影（3D Gaussian Splatting, 3DGS）的物理攻击框架——PGA，该框架能够利用少量图像实现快速且精确的重建，并具备照片级逼真的渲染能力。我们的框架通过防止高斯之间的相互遮挡和自遮挡，并采用极小极大优化方法调整每个视角的成像背景，从而提升跨视角的鲁棒性与对抗效果，帮助算法过滤掉非鲁棒的对抗特征。大量实验验证了 PGA 的有效性与优越性。


---

## [56] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling

### LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling

Due to the complex and highly dynamic motions in the real world, synthesizing dynamic videos from multi-view inputs for arbitrary viewpoints is challenging. Previous works based on neural radiance field or 3D Gaussian splatting are limited to modeling fine-scale motion, greatly restricting their application. In this paper, we introduce LocalDyGS, which consists of two parts to adapt our method to both large-scale and fine-scale motion scenes: 1) We decompose a complex dynamic scene into streamlined local spaces defined by seeds, enabling global modeling by capturing motion within each local space. 2) We decouple static and dynamic features for local space motion modeling. A static feature shared across time steps captures static information, while a dynamic residual field provides time-specific features. These are combined and decoded to generate Temporal Gaussians, modeling motion within each local space. As a result, we propose a novel dynamic scene reconstruction framework to model highly dynamic real-world scenes more realistically. Our method not only demonstrates competitive performance on various fine-scale datasets compared to state-of-the-art (SOTA) methods, but also represents the first attempt to model larger and more complex highly dynamic scenes.

由于现实世界中存在复杂且高度动态的运动，从多视图输入合成任意视角的动态视频是一项具有挑战性的任务。基于神经辐射场或三维高斯投影的现有方法在建模细粒度运动方面存在局限性，从而极大地限制了其应用。本文提出了 LocalDyGS，该方法由两个部分组成，以同时适应大规模和细粒度运动场景：1）我们将复杂的动态场景分解为由种子定义的精简局部空间，通过捕捉每个局部空间内的运动，实现全局建模；2）我们将局部空间运动建模中的静态特征与动态特征解耦，时间步共享的静态特征用于捕捉静态信息，而动态残差场则提供特定时间的特征。二者结合并解码生成时序高斯（Temporal Gaussians），用于建模各局部空间内的运动。因此，我们提出了一种新颖的动态场景重建框架，以更真实地建模高度动态的现实场景。我们的方法不仅在多个细粒度数据集上与当前最先进（SOTA）方法相比表现出竞争力，还首次尝试对更大规模、更复杂的高度动态场景进行建模。


---

## [57] Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps

### Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps

3D Gaussian Splatting (3DGS) has become a popular solution in SLAM due to its high-fidelity and real-time novel view synthesis performance. However, some previous 3DGS SLAM methods employ a differentiable rendering pipeline for tracking, lack geometric priors in outdoor scenes. Other approaches introduce separate tracking modules, but they accumulate errors with significant camera movement, leading to scale drift. To address these challenges, we propose a robust RGB-only outdoor 3DGS SLAM method: S3PO-GS. Technically, we establish a self-consistent tracking module anchored in the 3DGS pointmap, which avoids cumulative scale drift and achieves more precise and robust tracking with fewer iterations. Additionally, we design a patch-based pointmap dynamic mapping module, which introduces geometric priors while avoiding scale ambiguity. This significantly enhances tracking accuracy and the quality of scene reconstruction, making it particularly suitable for complex outdoor environments. Our experiments on the Waymo, KITTI, and DL3DV datasets demonstrate that S3PO-GS achieves state-of-the-art results in novel view synthesis and outperforms other 3DGS SLAM methods in tracking accuracy.

三维高斯投影（3D Gaussian Splatting, 3DGS）因其高保真度和实时的新视角合成性能，已成为 SLAM 中广受欢迎的解决方案。然而，一些已有的 3DGS SLAM 方法在跟踪中采用可微渲染管线，但在室外场景中缺乏几何先验；另一些方法则引入独立的跟踪模块，但在相机大幅移动时会累积误差，导致尺度漂移。为应对这些挑战，我们提出了一种鲁棒的仅基于 RGB 的室外 3DGS SLAM 方法：S3PO-GS。在技术上，我们构建了一个基于 3DGS 点图（pointmap）锚定的自一致跟踪模块，避免了累积尺度漂移，并以更少迭代实现更精确、更鲁棒的跟踪。此外，我们设计了基于补丁的点图动态建图模块，引入几何先验的同时避免尺度歧义。这显著提升了跟踪精度和场景重建质量，使其特别适用于复杂的室外环境。在 Waymo、KITTI 和 DL3DV 数据集上的实验表明，S3PO-GS 在新视角合成方面实现了当前最先进的结果，并在跟踪精度上优于其他 3DGS SLAM 方法。


---

## [58] VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis

### VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis

Realistic, high-fidelity 3D facial animations are crucial for expressive avatar systems in human-computer interaction and accessibility. Although prior methods show promising quality, their reliance on the mesh domain limits their ability to fully leverage the rapid visual innovations seen in 2D computer vision and graphics. We propose VisualSpeaker, a novel method that bridges this gap using photorealistic differentiable rendering, supervised by visual speech recognition, for improved 3D facial animation. Our contribution is a perceptual lip-reading loss, derived by passing photorealistic 3D Gaussian Splatting avatar renders through a pre-trained Visual Automatic Speech Recognition model during training. Evaluation on the MEAD dataset demonstrates that VisualSpeaker improves both the standard Lip Vertex Error metric by 56.1% and the perceptual quality of the generated animations, while retaining the controllability of mesh-driven animation. This perceptual focus naturally supports accurate mouthings, essential cues that disambiguate similar manual signs in sign language avatars.

真实且高保真的三维面部动画对于在人机交互和无障碍应用中的富有表现力的虚拟形象系统至关重要。尽管已有方法在质量上取得了可喜的成果，但其对网格域的依赖限制了其充分利用二维计算机视觉与图形学中快速发展的视觉创新能力。我们提出了 **VisualSpeaker**，一种利用光真实可微渲染并结合视觉语音识别监督的新方法，以提升三维面部动画效果。我们的方法核心贡献是一种感知型唇读损失（perceptual lip-reading loss），该损失在训练过程中通过将光真实的三维高斯溅射虚拟形象渲染结果输入至预训练的视觉自动语音识别模型（Visual ASR）获得。在 MEAD 数据集上的评估表明，VisualSpeaker 在标准唇部顶点误差（Lip Vertex Error）指标上提升了 56.1%，并显著改善了生成动画的感知质量，同时保留了网格驱动动画的可控性。这种感知导向的设计天然支持精确的口型表达，这对于在手语虚拟形象中区分相似手势至关重要。


---

## [59] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration

### RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration

3D Gaussian Splatting (3DGS) has demonstrated its potential in reconstructing scenes from unposed images. However, optimization-based 3DGS methods struggle with sparse views due to limited prior knowledge. Meanwhile, feed-forward Gaussian approaches are constrained by input formats, making it challenging to incorporate more input views. To address these challenges, we propose RegGS, a 3D Gaussian registration-based framework for reconstructing unposed sparse views. RegGS aligns local 3D Gaussians generated by a feed-forward network into a globally consistent 3D Gaussian representation. Technically, we implement an entropy-regularized Sinkhorn algorithm to efficiently solve the optimal transport Mixture 2-Wasserstein (MW2) distance, which serves as an alignment metric for Gaussian mixture models (GMMs) in Sim(3) space. Furthermore, we design a joint 3DGS registration module that integrates the MW2 distance, photometric consistency, and depth geometry. This enables a coarse-to-fine registration process while accurately estimating camera poses and aligning the scene. Experiments on the RE10K and ACID datasets demonstrate that RegGS effectively registers local Gaussians with high fidelity, achieving precise pose estimation and high-quality novel-view synthesis.

三维高斯投影（3D Gaussian Splatting, 3DGS）在无位姿图像的场景重建中展现了潜力。然而，基于优化的 3DGS 方法在稀疏视图下由于缺乏足够的先验知识而表现不佳；与此同时，前馈式高斯方法受输入格式限制，难以灵活引入更多输入视图。为应对这些挑战，我们提出了 RegGS，这是一种基于三维高斯配准的无位姿稀疏视图重建框架。RegGS 将前馈网络生成的局部三维高斯对齐为全局一致的三维高斯表示。在技术上，我们实现了一种熵正则化的 Sinkhorn 算法，以高效求解最优传输的混合二阶 Wasserstein（MW2）距离，该距离被用作 Sim(3) 空间中高斯混合模型（GMM）的对齐度量。此外，我们设计了一个联合 3DGS 配准模块，将 MW2 距离、光度一致性与深度几何相结合，从而实现由粗到细的配准过程，同时精确估计相机位姿并对齐场景。在 RE10K 和 ACID 数据集上的实验表明，RegGS 能够高保真地配准局部高斯，实现精确的位姿估计与高质量的新视角合成。


---

## [60] Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling

### Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling

Recent advances in 3D neural representations and instance-level editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3D-GALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing.

近年来，三维神经表示与实例级编辑模型的进步使得高质量三维内容的高效创作成为可能。然而，在高斯投影（Gaussian Splatting）中实现精确的局部三维编辑仍然具有挑战性，主要原因在于多视图二维局部分割的不一致性以及得分蒸馏采样（Score Distillation Sampling, SDS）损失本身存在的固有歧义。为克服这些局限，我们提出了 RoMaP，这是一种新颖的局部三维高斯编辑框架，能够实现精确且大幅度的部件级修改。首先，我们引入了鲁棒的三维掩码生成模块——三维几何感知标签预测（3D-Geometry Aware Label Prediction, 3D-GALP），利用球谐函数（Spherical Harmonics, SH）系数建模视角相关的标签变化与软标签特性，从而在多视角下获得准确且一致的部件分割。其次，我们提出了一种正则化 SDS 损失，将标准 SDS 损失与额外的正则项相结合。具体来说，我们通过计划潜变量混合与部件（Scheduled Latent Mixing and Part, SLaMP）编辑方法引入 L1 锚点损失，该方法可生成高质量的局部编辑二维图像，并将修改限制在目标区域，同时保持上下文一致性。其他正则化方法（如高斯先验移除）进一步提升了灵活性，使编辑可突破现有上下文的限制，而鲁棒的三维掩码机制则可防止非预期的编辑。实验结果表明，RoMaP 在重建与生成的高斯场景和物体上都实现了当前最先进的局部三维编辑效果，无论在定性还是定量评估中均表现优异，从而实现了更稳健且灵活的部件级三维高斯编辑。


---

## [61] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update

### TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update

Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a δ &lt; 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images.

由于反射和折射等固有物理特性，从 RGB 图像中理解透明物体的三维几何形状是一项具有挑战性的任务。为应对这些困难，特别是在稀疏视图和动态环境场景下，我们提出了 TRAN-D，这是一种基于二维高斯投影（2D Gaussian Splatting）的透明物体深度重建新方法。我们的核心思想是将透明物体与背景分离，从而能够针对物体对应的高斯进行集中优化。我们通过引入物体感知损失（object-aware loss）来缓解伪影问题，该损失会在被遮挡区域放置高斯，以确保对不可见表面的覆盖，同时减少过拟合。此外，我们结合了基于物理的模拟，仅需数秒即可优化重建结果，有效处理物体移除及剩余物体的连锁反应式运动，而无需重新扫描。我们在合成和真实数据序列上对 TRAN-D 进行了评估，其在现有基于高斯投影的最先进方法之上表现出持续且稳健的提升。与基线方法相比，TRAN-D 在合成的 TRansPose 序列上将平均绝对误差降低了 39% 以上。此外，尽管仅使用一张图像进行更新，TRAN-D 在 δ &lt; 2.5 cm 的精度下达到了 48.46%，是使用六张图像的基线方法的 1.5 倍以上。


---

## [62] DCHM: Depth-Consistent Human Modeling for Multiview Detection

### DCHM: Depth-Consistent Human Modeling for Multiview Detection

Multiview pedestrian detection typically involves two stages: human modeling and pedestrian localization. Human modeling represents pedestrians in 3D space by fusing multiview information, making its quality crucial for detection accuracy. However, existing methods often introduce noise and have low precision. While some approaches reduce noise by fitting on costly multiview 3D annotations, they often struggle to generalize across diverse scenes. To eliminate reliance on human-labeled annotations and accurately model humans, we propose Depth-Consistent Human Modeling (DCHM), a framework designed for consistent depth estimation and multiview fusion in global coordinates. Specifically, our proposed pipeline with superpixel-wise Gaussian Splatting achieves multiview depth consistency in sparse-view, large-scaled, and crowded scenarios, producing precise point clouds for pedestrian localization. Extensive validations demonstrate that our method significantly reduces noise during human modeling, outperforming previous state-of-the-art baselines. Additionally, to our knowledge, DCHM is the first to reconstruct pedestrians and perform multiview segmentation in such a challenging setting.

多视角行人检测通常包括两个阶段：行人建模和行人定位。行人建模通过融合多视角信息在三维空间中表示行人，其质量对检测精度至关重要。然而，现有方法常引入噪声且精度较低。虽然一些方法通过拟合代价高昂的多视角三维标注来降低噪声，但往往难以在多样化场景中具备良好的泛化能力。为消除对人工标注的依赖并精确建模行人，我们提出了深度一致性行人建模（Depth-Consistent Human Modeling, DCHM），该框架旨在实现全局坐标系下的一致深度估计与多视角融合。具体而言，我们提出的基于超像素级高斯点渲染（superpixel-wise Gaussian Splatting）的处理流程，在稀视角、大规模及拥挤场景中实现了多视角深度一致性，生成用于行人定位的精确点云。大量验证结果表明，该方法在行人建模过程中显著降低了噪声，性能优于以往的最新基线方法。此外，据我们所知，DCHM 是首个在如此具有挑战性的环境中同时实现行人重建与多视角分割的方法。


---

## [63] ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting

### ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting

3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing. o

三维高斯点渲染（3D Gaussian Splatting）以其高保真重建和实时新视角合成而闻名，但缺乏语义理解能力，限制了其在物体级感知中的应用。在本研究中，我们提出了 ObjectGS，这是一种融合三维场景重建与语义理解的物体感知框架。与将整个场景视为统一整体的方式不同，ObjectGS 将单个物体建模为生成神经高斯并共享物体 ID 的局部锚点，从而实现精确的物体级重建。在训练过程中，我们动态地增加或剪除这些锚点并优化其特征，同时通过带有分类损失的独热编码（one-hot ID encoding）施加明确的语义约束。大量实验表明，ObjectGS 在开放词汇和全景分割任务上均优于当前最先进的方法，并且能够与网格提取、场景编辑等应用无缝结合。


---

## [64] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting

### SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting

Surface reconstruction and novel view rendering from sparse-view images are challenging. Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets.

从稀视角图像进行表面重建与新视角渲染是一项具有挑战性的任务。基于有符号距离函数（Signed Distance Function, SDF）的方法在捕捉细节方面存在不足，而基于三维高斯点渲染（3D Gaussian Splatting, 3DGS）的方法则缺乏全局几何一致性。为此，我们提出了一种结合两者优势的新型混合方法：SDF 用于捕捉粗略几何结构以提升基于 3DGS 的渲染效果，而 3DGS 渲染生成的新图像则用于细化 SDF，从而实现精确的表面重建。实验结果表明，该方法在 DTU 和 MobileBrick 数据集上的表面重建与新视角合成任务中均超越了当前最先进的方法。


---

## [65] Gaussian Splatting with Discretized SDF for Relightable Assets

### Gaussian Splatting with Discretized SDF for Relightable Assets

3D Gaussian splatting (3DGS) has shown its detailed expressive ability and highly efficient rendering speed in the novel view synthesis (NVS) task. The application to inverse rendering still faces several challenges, as the discrete nature of Gaussian primitives makes it difficult to apply geometry constraints. Recent works introduce the signed distance field (SDF) as an extra continuous representation to regularize the geometry defined by Gaussian primitives. It improves the decomposition quality, at the cost of increasing memory usage and complicating training. Unlike these works, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This approach allows us to link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, enabling rendering the SDF via splatting and avoiding the computational cost of ray this http URL key challenge is to regularize the discrete samples to be consistent with the underlying SDF, as the discrete representation can hardly apply the gradient-based constraints (e.g., Eikonal loss). For this, we project Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely a projection-based consistency loss. Thanks to the discretized SDF, our method achieves higher relighting quality, while requiring no extra memory beyond GS and avoiding complex manually designed optimization. The experiments reveal that our method outperforms existing Gaussian-based inverse rendering methods.

三维高斯点渲染（3D Gaussian Splatting, 3DGS）在新视角合成（NVS）任务中展现了细致的表达能力和高效的渲染速度。然而，其在逆向渲染中的应用仍面临诸多挑战，因为高斯基元的离散性使得几何约束难以施加。现有一些工作引入有符号距离场（Signed Distance Field, SDF）作为额外的连续表示，以对高斯基元定义的几何进行正则化，这虽然提升了解耦质量，但也增加了内存占用并使训练过程更加复杂。不同于这些方法，我们提出了一种离散化的 SDF，将连续的 SDF 以离散方式表示，即在每个高斯中编码一个采样值。这种方法通过 SDF 到不透明度的变换（SDF-to-opacity transformation）将 SDF 与高斯的不透明度关联起来，从而可通过点渲染（splatting）实现 SDF 的渲染，并避免光线追踪的计算开销。关键挑战在于如何使离散采样与底层 SDF 保持一致，因为离散表示难以直接应用基于梯度的约束（例如 Eikonal 损失）。为此，我们将高斯投影到 SDF 的零水平集，并通过投影一致性损失（projection-based consistency loss）强制其与点渲染得到的表面对齐。得益于离散化 SDF，我们的方法在无需额外 GS 之外的内存、且避免复杂人工设计优化的情况下，实现了更高质量的重光照效果。实验结果表明，该方法在性能上优于现有基于高斯的逆向渲染方法。


---

## [66] GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar

### GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar

Despite recent progress in 3D head avatar generation, balancing identity preservation, i.e., reconstruction, with novel poses and expressions, i.e., animation, remains a challenge. Existing methods struggle to adapt Gaussians to varying geometrical deviations across facial regions, resulting in suboptimal quality. To address this, we propose GeoAvatar, a framework for adaptive geometrical Gaussian Splatting. GeoAvatar leverages Adaptive Pre-allocation Stage (APS), an unsupervised method that segments Gaussians into rigid and flexible sets for adaptive offset regularization. Then, based on mouth anatomy and dynamics, we introduce a novel mouth structure and the part-wise deformation strategy to enhance the animation fidelity of the mouth. Finally, we propose a regularization loss for precise rigging between Gaussians and 3DMM faces. Moreover, we release DynamicFace, a video dataset with highly expressive facial motions. Extensive experiments show the superiority of GeoAvatar compared to state-of-the-art methods in reconstruction and novel animation scenarios.

尽管三维头部虚拟形象生成领域取得了显著进展，但在保持身份特征（即重建）与生成新姿态和表情（即动画）之间的平衡方面仍面临挑战。现有方法在适应面部不同区域几何偏差方面表现不足，导致质量欠佳。为此，我们提出了 GeoAvatar，这是一种自适应几何高斯点渲染框架。GeoAvatar 引入了自适应预分配阶段（Adaptive Pre-allocation Stage, APS），这是一种无监督方法，可将高斯划分为刚性集与柔性集，以实现自适应偏移正则化。随后，我们基于口腔的解剖结构与动态特性，提出了一种新颖的口腔结构及分部形变策略，以提升口部动画的保真度。最后，我们提出了一种正则化损失，用于在高斯与三维形状可变模型（3DMM）面部之间实现精确绑定。此外，我们还发布了 DynamicFace 数据集，该视频数据集包含高度丰富的面部表情动态。大量实验表明，GeoAvatar 在重建和新颖动画场景中均优于当前最先进的方法。


---

## [67] DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering

### DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering

Dynamic scene reconstruction is a long-term challenge in 3D vision. Existing plane-based methods in dynamic Gaussian splatting suffer from an unsuitable low-rank assumption, causing feature overlap and poor rendering quality. Although 4D hash encoding provides an explicit representation without low-rank constraints, directly applying it to the entire dynamic scene leads to substantial hash collisions and redundancy. To address these challenges, we present DASH, a real-time dynamic scene rendering framework that employs 4D hash encoding coupled with self-supervised decomposition. Our approach begins with a self-supervised decomposition mechanism that separates dynamic and static components without manual annotations or precomputed masks. Next, we introduce a multiresolution 4D hash encoder for dynamic elements, providing an explicit representation that avoids the low-rank assumption. Finally, we present a spatio-temporal smoothness regularization strategy to mitigate unstable deformation artifacts. Experiments on real-world datasets demonstrate that DASH achieves state-of-the-art dynamic rendering performance, exhibiting enhanced visual quality at real-time speeds of 264 FPS on a single 4090 GPU.

动态场景重建是三维视觉领域的长期挑战。现有基于平面的动态高斯点渲染方法依赖不合适的低秩假设，导致特征重叠和渲染质量下降。虽然四维哈希编码（4D hash encoding）能够在没有低秩约束的情况下提供显式表示，但直接将其应用于整个动态场景会引发大量哈希冲突与冗余。为解决这些问题，我们提出了 DASH，这是一种结合四维哈希编码与自监督分解的实时动态场景渲染框架。该方法首先采用自监督分解机制，在无需人工标注或预计算掩码的情况下，将动态与静态部分进行分离。随后，我们为动态元素引入多分辨率四维哈希编码，以提供避免低秩假设的显式表示。最后，我们提出时空平滑正则化策略，以缓解不稳定的形变伪影。在真实世界数据集上的实验表明，DASH 在动态渲染性能上达到了当前最优水平，在单张 4090 GPU 上实现了 264 FPS 的实时速度，同时具备更高的视觉质量。


---

## [68] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars

### HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars

We present a universal prior model for 3D head avatars with explicit hair compositionality. Existing approaches to build generalizable priors for 3D head avatars often adopt a holistic modeling approach, treating the face and hair as an inseparable entity. This overlooks the inherent compositionality of the human head, making it difficult for the model to naturally disentangle face and hair representations, especially when the dataset is limited. Furthermore, such holistic models struggle to support applications like 3D face and hairstyle swapping in a flexible and controllable manner. To address these challenges, we introduce a prior model that explicitly accounts for the compositionality of face and hair, learning their latent spaces separately. A key enabler of this approach is our synthetic hairless data creation pipeline, which removes hair from studio-captured datasets using estimated hairless geometry and texture derived from a diffusion prior. By leveraging a paired dataset of hair and hairless captures, we train disentangled prior models for face and hair, incorporating compositionality as an inductive bias to facilitate effective separation. Our model's inherent compositionality enables seamless transfer of face and hair components between avatars while preserving identity. Additionally, we demonstrate that our model can be fine-tuned in a few-shot manner using monocular captures to create high-fidelity, hair-compositional 3D head avatars for unseen subjects. These capabilities highlight the practical applicability of our approach in real-world scenarios, paving the way for flexible and expressive 3D avatar generation.

我们提出了一种具有显式头发组合性的三维头部虚拟形象通用先验模型。现有用于构建可泛化三维头部虚拟形象先验的方法通常采用整体建模，将面部与头发视为不可分割的整体。这种方法忽视了人类头部固有的组合性，使得模型难以自然地解耦面部与头发的表示，尤其是在数据集有限的情况下。此外，这类整体模型在支持三维面部与发型交换等需要灵活可控的应用时也存在困难。为了解决这些问题，我们引入了一种显式建模面部与头发组合性的先验模型，分别学习它们的潜在空间。实现这一方法的关键是我们提出的合成无发数据生成流程，该流程利用来自扩散先验估计的无发几何与纹理，从影棚采集的数据集中去除头发。通过利用成对的有发与无发数据集，我们为面部与头发训练了解耦的先验模型，并将组合性作为归纳偏置以促进有效分离。我们模型固有的组合性使得在保持身份一致性的前提下，实现面部与头发组件在虚拟形象之间的无缝迁移。此外，我们还展示了该模型能够通过单目采集的少量样本进行快速微调，为未见过的对象生成高保真、具备头发组合性的三维头部虚拟形象。这些能力突显了我们方法在真实场景中的实用性，为灵活且富有表现力的三维虚拟形象生成铺平了道路。


---

## [69] GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes from Unconstrained Photo Collections

### GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes from Unconstrained Photo Collections

We propose a 3D Gaussian splatting-based framework for outdoor relighting that leverages intrinsic image decomposition to precisely integrate sunlight, sky radiance, and indirect lighting from unconstrained photo collections. Unlike prior methods that compress the per-image global illumination into a single latent vector, our approach enables simultaneously diverse shading manipulation and the generation of dynamic shadow effects. This is achieved through three key innovations: (1) a residual-based sun visibility extraction method to accurately separate direct sunlight effects, (2) a region-based supervision framework with a structural consistency loss for physically interpretable and coherent illumination decomposition, and (3) a ray-tracing-based technique for realistic shadow simulation. Extensive experiments demonstrate that our framework synthesizes novel views with competitive fidelity against state-of-the-art relighting solutions and produces more natural and multifaceted illumination and shadow effects.o

我们提出了一种基于三维高斯溅射的户外重光照框架，利用固有图像分解技术，从非受限的照片集合中精确融合阳光、天空辐射和间接光照。与以往将每张图像的全局光照压缩为单一潜向量的方法不同，我们的方法能够同时实现多样化的着色操作以及动态阴影效果的生成。这一能力得益于三项关键创新：（1）基于残差的太阳可见性提取方法，可精确分离直射阳光的影响；（2）基于区域的监督框架结合结构一致性损失，实现具有物理可解释性且一致的光照分解；（3）基于光线追踪的逼真阴影模拟技术。大量实验表明，我们的框架在新视角合成中能达到与最新重光照方法相当的保真度，并生成更自然、更多样化的光照与阴影效果。


---

## [70] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction

### Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction

We present a framework that enables fast reconstruction and real-time rendering of urban-scale scenes while maintaining robustness against appearance variations across multi-view captures. Our approach begins with scene partitioning for parallel training, employing a visibility-based image selection strategy to optimize training efficiency. A controllable level-of-detail (LOD) strategy explicitly regulates Gaussian density under a user-defined budget, enabling efficient training and rendering while maintaining high visual fidelity. The appearance transformation module mitigates the negative effects of appearance inconsistencies across images while enabling flexible adjustments. Additionally, we utilize enhancement modules, such as depth regularization, scale regularization, and antialiasing, to improve reconstruction fidelity. Experimental results demonstrate that our method effectively reconstructs urban-scale scenes and outperforms previous approaches in both efficiency and quality.

我们提出了一个能够实现城市级场景快速重建与实时渲染的框架，同时在多视图捕获中保持对外观变化的鲁棒性。我们的方法首先通过场景划分进行并行训练，并采用基于可见性的图像选择策略以优化训练效率。我们设计了一种可控的细节层次（LOD）策略，在用户设定的预算范围内显式调节高斯密度，从而在保持高视觉保真度的同时实现高效训练与渲染。外观变换模块用于减轻跨图像外观不一致带来的负面影响，并支持灵活调整。此外，我们还引入了深度正则化、尺度正则化和抗锯齿等增强模块，以提升重建的保真度。实验结果表明，我们的方法能够高效、准确地重建城市级场景，并在效率和质量上均优于以往方法。


---

## [71] NeRF Is a Valuable Assistant for 3D Gaussian Splatting

### NeRF Is a Valuable Assistant for 3D Gaussian Splatting

We introduce NeRF-GS, a novel framework that jointly optimizes Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). This framework leverages the inherent continuous spatial representation of NeRF to mitigate several limitations of 3DGS, including sensitivity to Gaussian initialization, limited spatial awareness, and weak inter-Gaussian correlations, thereby enhancing its performance. In NeRF-GS, we revisit the design of 3DGS and progressively align its spatial features with NeRF, enabling both representations to be optimized within the same scene through shared 3D spatial information. We further address the formal distinctions between the two approaches by optimizing residual vectors for both implicit features and Gaussian positions to enhance the personalized capabilities of 3DGS. Experimental results on benchmark datasets show that NeRF-GS surpasses existing methods and achieves state-of-the-art performance. This outcome confirms that NeRF and 3DGS are complementary rather than competing, offering new insights into hybrid approaches that combine 3DGS and NeRF for efficient 3D scene representation.

我们提出了 NeRF-GS，这是一种联合优化神经辐射场（NeRF）与三维高斯溅射（3DGS）的新型框架。该框架利用 NeRF 所固有的连续空间表示来缓解 3DGS 的多项局限性，包括对高斯初始化的敏感性、空间感知能力有限以及高斯之间相关性弱，从而提升其整体性能。在 NeRF-GS 中，我们重新审视了 3DGS 的设计，并逐步将其空间特征与 NeRF 对齐，使两种表示能够在同一场景中通过共享的三维空间信息共同优化。我们还针对两种方法在形式上的差异，优化了隐式特征和高斯位置的残差向量，以增强 3DGS 的个性化能力。基准数据集上的实验结果表明，NeRF-GS 超越了现有方法并达到了当前最优性能。这一结果表明，NeRF 与 3DGS 是互补而非竞争的关系，为结合 3DGS 与 NeRF 的混合高效三维场景表示方法提供了新的见解。


---

## [72] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction

### MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction

We present MoGA, a novel method to reconstruct high-fidelity 3D Gaussian avatars from a single-view image. The main challenge lies in inferring unseen appearance and geometric details while ensuring 3D consistency and realism. Most previous methods rely on 2D diffusion models to synthesize unseen views; however, these generated views are sparse and inconsistent, resulting in unrealistic 3D artifacts and blurred appearance. To address these limitations, we leverage a generative avatar model, that can generate diverse 3D avatars by sampling deformed Gaussians from a learned prior distribution. Due to limited 3D training data, such a 3D model alone cannot capture all image details of unseen identities. Consequently, we integrate it as a prior, ensuring 3D consistency by projecting input images into its latent space and enforcing additional 3D appearance and geometric constraints. Our novel approach formulates Gaussian avatar creation as model inversion by fitting the generative avatar to synthetic views from 2D diffusion models. The generative avatar provides an initialization for model fitting, enforces 3D regularization, and helps in refining pose. Experiments show that our method surpasses state-of-the-art techniques and generalizes well to real-world scenarios. Our Gaussian avatars are also inherently animatable.

我们提出了 MoGA，这是一种从单视图图像重建高保真三维高斯头像的新方法。其主要挑战在于在保证三维一致性与真实感的同时，推断未见的外观和几何细节。以往大多数方法依赖二维扩散模型来合成未见视角，但这些生成的视角往往稀疏且不一致，导致三维伪影和外观模糊等不真实现象。为克服这些局限，我们利用生成头像模型，通过从学习到的先验分布中采样变形高斯来生成多样化的三维头像。由于三维训练数据有限，这类三维模型单独使用时无法捕捉所有未见身份的图像细节。因此，我们将其作为先验引入，通过将输入图像投影到其潜空间并施加额外的三维外观与几何约束来保证三维一致性。我们的新方法将高斯头像的生成形式化为模型反演过程，即将生成头像拟合到由二维扩散模型生成的合成视图上。生成头像不仅为模型拟合提供初始化，还能施加三维正则化并辅助姿态优化。实验表明，我们的方法优于现有最先进技术，并在真实场景中具有良好的泛化能力。此外，我们的高斯头像天然具有可动画性。


---

## [73] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis

### Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis

In this paper, we present a novel framework for video-to-4D generation that creates high-quality dynamic 3D content from single video inputs. Direct 4D diffusion modeling is extremely challenging due to costly data construction and the high-dimensional nature of jointly representing 3D shape, appearance, and motion. We address these challenges by introducing a Direct 4DMesh-to-GS Variation Field VAE that directly encodes canonical Gaussian Splats (GS) and their temporal variations from 3D animation data without per-instance fitting, and compresses high-dimensional animations into a compact latent space. Building upon this efficient representation, we train a Gaussian Variation Field diffusion model with temporal-aware Diffusion Transformer conditioned on input videos and canonical GS. Trained on carefully-curated animatable 3D objects from the Objaverse dataset, our model demonstrates superior generation quality compared to existing methods. It also exhibits remarkable generalization to in-the-wild video inputs despite being trained exclusively on synthetic data, paving the way for generating high-quality animated 3D content.

本文提出了一种新颖的视频到 4D 生成框架，可从单个视频输入生成高质量的动态三维内容。直接进行 4D 扩散建模极具挑战性，因为这不仅需要高成本的数据构建，还要应对同时表示三维形状、外观和运动的高维特性。为解决这些问题，我们引入了一种直接的 4DMesh-to-GS 变动场 VAE，该方法可直接从三维动画数据中编码标准高斯泼溅（GS）及其时间变化，而无需针对每个实例进行拟合，并将高维动画压缩到紧凑的潜在空间。在这一高效表示的基础上，我们训练了一个高斯变动场扩散模型，该模型结合了时间感知的扩散 Transformer，并以输入视频和标准 GS 为条件。该模型在精心筛选的 Objaverse 数据集中可动画的三维物体上进行训练，与现有方法相比展现出更高的生成质量。同时，即便仅在合成数据上训练，它在真实视频输入上的泛化能力也十分出色，为生成高质量的动画三维内容开辟了新途径。


---

## [74] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation

### IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation

Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete space matching, which can be equivalent to efficient 3D convolution. When the agent is close to the goal, we finally solve the fine target pose with optimization via differentiable rendering. The proposed IGL-Nav outperforms existing state-of-the-art methods by a large margin across diverse experimental configurations. It can also handle the more challenging free-view image-goal setting and be deployed on real-world robotic platform using a cellphone to capture goal image at arbitrary pose.

以图像作为目标的视觉导航是一个基础且具有挑战性的问题。传统方法要么依赖端到端的强化学习，要么采用基于模块化的策略，将拓扑图或鸟瞰图（BEV）作为记忆，但这些方法无法充分建模已探索的三维环境与目标图像之间的几何关系。为了在三维空间中高效且准确地定位目标图像，我们将导航系统建立在可渲染的三维高斯（3DGS）表示之上。然而，由于 3DGS 优化的计算开销大以及 6 自由度相机位姿的搜索空间庞大，在智能体探索过程中直接利用 3DGS 进行图像定位效率极低。为此，我们提出了 IGL-Nav，这是一种用于高效且具备三维感知能力的图像目标导航的增量式三维高斯定位框架。具体而言，我们在新图像到来时，利用前向单目预测增量更新场景表示；然后利用几何信息进行离散空间匹配以粗略定位目标，这相当于高效的三维卷积；当智能体接近目标时，我们通过可微渲染优化求解精确的目标位姿。所提出的 IGL-Nav 在多种实验配置下均显著超越现有的最新方法，同时还能处理更具挑战性的自由视角图像目标设定，并可部署在真实的机器人平台上，仅需使用手机在任意姿态下拍摄目标图像即可。


---

## [75] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

### No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

We introduce SPFSplat, an efficient framework for 3D Gaussian splatting from sparse multi-view images, requiring no ground-truth poses during training or inference. It employs a shared feature extraction backbone, enabling simultaneous prediction of 3D Gaussian primitives and camera poses in a canonical space from unposed inputs within a single feed-forward step. Alongside the rendering loss based on estimated novel-view poses, a reprojection loss is integrated to enforce the learning of pixel-aligned Gaussian primitives for enhanced geometric constraints. This pose-free training paradigm and efficient one-step feed-forward design make SPFSplat well-suited for practical applications. Remarkably, despite the absence of pose supervision, SPFSplat achieves state-of-the-art performance in novel view synthesis even under significant viewpoint changes and limited image overlap. It also surpasses recent methods trained with geometry priors in relative pose estimation.

我们提出了 SPFSplat，这是一种高效的三维高斯泼溅框架，可从稀疏多视图图像中进行建模，训练和推理过程中均无需真实位姿。该方法采用共享特征提取骨干网络，使得在单次前向推理中即可从无位姿输入同时预测标准空间下的三维高斯基元和相机位姿。除了基于估计的新视角位姿的渲染损失外，还引入了重投影损失，以强化像素对齐的高斯基元学习，从而增强几何约束。这种无位姿监督的训练范式与高效的一步前向设计，使 SPFSplat 非常适用于实际应用。值得注意的是，即使在缺乏位姿监督的情况下，SPFSplat 在新视角合成中依然取得了最新的性能，即便在视角变化较大和图像重叠有限的条件下亦如此。此外，它在相对位姿估计中也优于利用几何先验进行训练的最新方法。


---

## [76] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

### Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

3D generation has made significant progress, however, it still largely remains at the object-level. Feedforward 3D scene-level generation has been rarely explored due to the lack of models capable of scaling-up latent representation learning on 3D scene-level data. Unlike object-level generative models, which are trained on well-labeled 3D data in a bounded canonical space, scene-level generations with 3D scenes represented by 3D Gaussian Splatting (3DGS) are unbounded and exhibit scale inconsistency across different scenes, making unified latent representation learning for generative purposes extremely challenging. In this paper, we introduce Can3Tok, the first 3D scene-level variational autoencoder (VAE) capable of encoding a large number of Gaussian primitives into a low-dimensional latent embedding, which effectively captures both semantic and spatial information of the inputs. Beyond model design, we propose a general pipeline for 3D scene data processing to address scale inconsistency issue. We validate our method on the recent scene-level 3D dataset DL3DV-10K, where we found that only Can3Tok successfully generalizes to novel 3D scenes, while compared methods fail to converge on even a few hundred scene inputs during training and exhibit zero generalization ability during inference. Finally, we demonstrate image-to-3DGS and text-to-3DGS generation as our applications to demonstrate its ability to facilitate downstream generation tasks.

三维生成技术已取得显著进展，但仍主要停留在物体级别。由于缺乏能够在三维场景级数据上扩展潜在表示学习的模型，前向式的三维场景级生成鲜有探索。与在有界标准空间中利用标注完善的三维数据训练的物体级生成模型不同，基于三维高斯泼溅（3DGS）表示的场景级生成是无界的，并且在不同场景间存在尺度不一致问题，这使得面向生成任务的统一潜在表示学习极具挑战性。本文提出了 Can3Tok，这是首个能够将大量高斯基元编码为低维潜在嵌入的三维场景级变分自编码器（VAE），能够有效捕获输入的语义和空间信息。除了模型设计，我们还提出了一套通用的三维场景数据处理流程，以解决尺度不一致的问题。我们在最新的场景级三维数据集 DL3DV-10K 上验证了该方法，结果发现，只有 Can3Tok 能够成功泛化到新的三维场景，而对比方法在训练中即使面对几百个场景输入也无法收敛，并且在推理时表现出零泛化能力。最后，我们展示了图像到 3DGS 和文本到 3DGS 的生成应用，以证明其在下游生成任务中的促进作用。


---

## [77] Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing

### Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing

We address the challenge of lifting 2D visual segmentation to 3D in Gaussian Splatting. Existing methods often suffer from inconsistent 2D masks across viewpoints and produce noisy segmentation boundaries as they neglect these semantic cues to refine the learned Gaussians. To overcome this, we introduce Gaussian Instance Tracing (GIT), which augments the standard Gaussian representation with an instance weight matrix across input views. Leveraging the inherent consistency of Gaussians in 3D, we use this matrix to identify and correct 2D segmentation inconsistencies. Furthermore, since each Gaussian ideally corresponds to a single object, we propose a GIT-guided adaptive density control mechanism to split and prune ambiguous Gaussians during training, resulting in sharper and more coherent 2D and 3D segmentation boundaries. Experimental results show that our method extracts clean 3D assets and consistently improves 3D segmentation in both online (e.g., self-prompting) and offline (e.g., contrastive lifting) settings, enabling applications such as hierarchical segmentation, object extraction, and scene editing.

我们研究了在高斯溅射（Gaussian Splatting）中将二维视觉分割提升到三维的挑战。现有方法常常存在跨视角二维掩码不一致的问题，并且由于忽视利用这些语义线索来优化已学习的高斯，导致分割边界噪声较大。为解决这一问题，我们提出了高斯实例追踪（Gaussian Instance Tracing，GIT）方法，在标准高斯表示中引入了跨输入视角的实例权重矩阵。利用三维高斯固有的一致性，我们使用该矩阵来识别并纠正二维分割中的不一致。此外，由于每个高斯理想情况下应对应于单一物体，我们提出了基于 GIT 引导的自适应密度控制机制，在训练过程中对存在歧义的高斯进行拆分与剪枝，从而获得更清晰、更一致的二维与三维分割边界。实验结果表明，我们的方法能够提取干净的三维资产，并在在线（如自提示）和离线（如对比提升）设置中持续提升三维分割性能，从而支持分层分割、物体提取和场景编辑等应用。


---

## [78] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework

### Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework

We propose 3D Super Resolution (3DSR), a novel 3D Gaussian-splatting-based super-resolution framework that leverages off-the-shelf diffusion-based 2D super-resolution models. 3DSR encourages 3D consistency across views via the use of an explicit 3D Gaussian-splatting-based scene representation. This makes the proposed 3DSR different from prior work, such as image upsampling or the use of video super-resolution, which either don't consider 3D consistency or aim to incorporate 3D consistency implicitly. Notably, our method enhances visual quality without additional fine-tuning, ensuring spatial coherence within the reconstructed scene. We evaluate 3DSR on MipNeRF360 and LLFF data, demonstrating that it produces high-resolution results that are visually compelling, while maintaining structural consistency in 3D reconstructions.

在三维高斯溅射（3D Gaussian Splatting，3DGS）中，超参数调优是一个耗时且依赖专家经验的过程，往往导致重建结果不一致和效果次优。我们提出了 RLGS，这是一种即插即用的强化学习框架，通过轻量级策略模块在 3DGS 中自适应调整超参数，如学习率和加密阈值。该框架与具体模型无关，可无缝集成到现有的 3DGS 管线中，无需修改架构。我们验证了其在多种最先进的 3DGS 变体（包括 Taming-3DGS 和 3DGS-MCMC）中的泛化能力，并在多种数据集上验证了其鲁棒性。RLGS 能够持续提升渲染质量，例如，在固定高斯预算下，它使 Taming-3DGS 在 Tanks and Temple（TNT）数据集上的 PSNR 提高了 0.7 dB，并且即使在基线性能饱和时仍能带来收益。结果表明，RLGS 为 3DGS 训练中的超参数调优提供了一种高效且通用的自动化解决方案，填补了强化学习在 3DGS 应用中的空白。


---

## [79] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction

### MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction

We present Multi-Baseline Gaussian Splatting (MuRF), a generalized feed-forward approach for novel view synthesis that effectively handles diverse baseline settings, including sparse input views with both small and large baselines. Specifically, we integrate features from Multi-View Stereo (MVS) and Monocular Depth Estimation (MDE) to enhance feature representations for generalizable reconstruction. Next, We propose a projection-and-sampling mechanism for deep depth fusion, which constructs a fine probability volume to guide the regression of the feature map. Furthermore, We introduce a reference-view loss to improve geometry and optimization efficiency. We leverage 3D Gaussian representations to accelerate training and inference time while enhancing rendering quality. MuRF achieves state-of-the-art performance across multiple baseline settings and diverse scenarios ranging from simple objects (DTU) to complex indoor and outdoor scenes (RealEstate10K). We also demonstrate promising zero-shot performance on the LLFF and Mip-NeRF 360 datasets.

我们提出了多基线高斯溅射（Multi-Baseline Gaussian Splatting，MuRF），这是一种通用的前向新视角合成方法，能够有效处理包括小基线和大基线稀疏输入视图在内的多种基线设置。具体而言，我们融合了多视图立体（Multi-View Stereo, MVS）和单目深度估计（Monocular Depth Estimation, MDE）的特征，以增强特征表示能力，从而实现更强的重建泛化性。随后，我们提出了一种投影与采样机制进行深度融合，构建精细的概率体以引导特征图的回归。此外，我们引入了参考视图损失，以提升几何质量和优化效率。我们利用三维高斯表示加速训练与推理的同时提升渲染质量。MuRF 在多种基线设置及从简单物体（DTU）到复杂室内外场景（RealEstate10K）的多样化场景中均实现了最先进的性能。我们还在 LLFF 和 Mip-NeRF 360 数据集上展示了具有潜力的零样本表现。


---

## [80] CF3: Compact and Fast 3D Feature Fields

### CF3: Compact and Fast 3D Feature Fields

3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.

三维高斯溅射（3D Gaussian Splatting, 3DGS）已开始引入来自二维基础模型的丰富信息。然而，大多数方法依赖于一种自下而上的优化过程，将原始二维特征视为真实值，这带来了较高的计算成本。我们提出了一种自上而下的管线，用于构建紧凑且高效的三维高斯特征场，称为 CF3。我们首先将多视图二维特征与预训练的高斯进行快速加权融合。这一方法使得可以直接在提升后的特征上训练每个高斯的自编码器，而非在二维域中训练自编码器，从而使自编码器与特征分布更好地对齐。更重要的是，我们引入了一种自适应稀疏化方法，在优化特征场中高斯属性的同时，对冗余高斯进行裁剪与合并，从而在保留几何细节的前提下构建高效表示。与 Feature-3DGS 相比，我们的方法仅使用 5% 的高斯即可实现具有竞争力的三维特征场。


---

## [81] GAP: Gaussianize Any Point Clouds with Text Guidance

### GAP: Gaussianize Any Point Clouds with Text Guidance

3D Gaussian Splatting (3DGS) has demonstrated its advantages in achieving fast and high-quality rendering. As point clouds serve as a widely-used and easily accessible form of 3D representation, bridging the gap between point clouds and Gaussians becomes increasingly important. Recent studies have explored how to convert the colored points into Gaussians, but directly generating Gaussians from colorless 3D point clouds remains an unsolved challenge. In this paper, we propose GAP, a novel approach that gaussianizes raw point clouds into high-fidelity 3D Gaussians with text guidance. Our key idea is to design a multi-view optimization framework that leverages a depth-aware image diffusion model to synthesize consistent appearances across different viewpoints. To ensure geometric accuracy, we introduce a surface-anchoring mechanism that effectively constrains Gaussians to lie on the surfaces of 3D shapes during optimization. Furthermore, GAP incorporates a diffuse-based inpainting strategy that specifically targets at completing hard-to-observe regions. We evaluate GAP on the Point-to-Gaussian generation task across varying complexity levels, from synthetic point clouds to challenging real-world scans, and even large-scale scenes.

三维高斯溅射（3D Gaussian Splatting, 3DGS）已展现出在实现快速且高质量渲染方面的优势。由于点云是一种广泛使用且易于获取的三维表示形式，弥合点云与高斯之间的差距变得愈发重要。近期研究已经探索了如何将带颜色的点转换为高斯，但直接从无颜色的三维点云生成高斯仍是一个未解决的挑战。本文提出了 GAP，这是一种能够在文本引导下将原始点云高斯化为高保真三维高斯的新方法。我们的核心思想是设计一个多视图优化框架，利用深度感知的图像扩散模型，在不同视角下合成一致的外观。为确保几何精度，我们引入了一种表面锚定机制，在优化过程中有效约束高斯位于三维形状的表面。此外，GAP 结合了一种基于漫反射的修补策略，专门用于完成难以观测区域的补全。我们在从合成点云到复杂的真实世界扫描，甚至大规模场景的多种复杂度下，对 GAP 在点云到高斯生成任务中的表现进行了评估。


---

## [82] ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors

### ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors

Recent advances in novel view synthesis (NVS) have enabled real-time rendering with 3D Gaussian Splatting (3DGS). However, existing methods struggle with artifacts and missing regions when rendering from viewpoints that deviate from the training trajectory, limiting seamless scene exploration. To address this, we propose a 3DGS-based pipeline that generates additional training views to enhance reconstruction. We introduce an information-gain-driven virtual camera placement strategy to maximize scene coverage, followed by video diffusion priors to refine rendered results. Fine-tuning 3D Gaussians with these enhanced views significantly improves reconstruction quality. To evaluate our method, we present Wild-Explore, a benchmark designed for challenging scene exploration. Experiments demonstrate that our approach outperforms existing 3DGS-based methods, enabling high-quality, artifact-free rendering from arbitrary viewpoints.

新视图合成（Novel View Synthesis, NVS）的最新进展使得基于三维高斯溅射（3D Gaussian Splatting, 3DGS）的实时渲染成为可能。然而，当从偏离训练轨迹的视点进行渲染时，现有方法容易出现伪影和区域缺失的问题，从而限制了无缝的场景探索。为此，我们提出了一种基于 3DGS 的管线，通过生成额外的训练视图来增强重建效果。我们引入了一种基于信息增益的虚拟相机布置策略，以最大化场景覆盖率，并结合视频扩散先验来优化渲染结果。利用这些增强视图对三维高斯进行微调，可以显著提升重建质量。为评估我们的方法，我们提出了 Wild-Explore 基准，用于应对具有挑战性的场景探索任务。实验结果表明，我们的方法优于现有的基于 3DGS 的方法，实现了任意视点下的高质量、无伪影渲染。


---

## [83] Learning an Implicit Physics Model for Image-based Fluid Simulation

### Learning an Implicit Physics Model for Image-based Fluid Simulation

Humans possess an exceptional ability to imagine 4D scenes, encompassing both motion and 3D geometry, from a single still image. This ability is rooted in our accumulated observations of similar scenes and an intuitive understanding of physics. In this paper, we aim to replicate this capacity in neural networks, specifically focusing on natural fluid imagery. Existing methods for this task typically employ simplistic 2D motion estimators to animate the image, leading to motion predictions that often defy physical principles, resulting in unrealistic animations. Our approach introduces a novel method for generating 4D scenes with physics-consistent animation from a single image. We propose the use of a physics-informed neural network that predicts motion for each surface point, guided by a loss term derived from fundamental physical principles, including the Navier-Stokes equations. To capture appearance, we predict feature-based 3D Gaussians from the input image and its estimated depth, which are then animated using the predicted motions and rendered from any desired camera perspective. Experimental results highlight the effectiveness of our method in producing physically plausible animations, showcasing significant performance improvements over existing methods.

人类具备一种非凡的能力，能够从单张静态图像中想象出包含运动与三维几何的四维场景。这种能力源于我们对类似场景的长期观察以及对物理规律的直觉理解。本文旨在在神经网络中复现这一能力，特别聚焦于自然流体图像。现有方法通常采用简化的二维运动估计器来对图像进行动画化，导致的运动预测往往违背物理规律，从而产生不真实的动画。为解决这一问题，我们提出了一种新颖的方法，可从单张图像生成具有物理一致性的四维场景动画。具体而言，我们提出了一种物理约束的神经网络，用于预测每个表面点的运动，其损失函数由包括纳维-斯托克斯方程在内的基本物理原理推导而来。为捕捉外观信息，我们从输入图像及其估计深度中预测基于特征的三维高斯表示，并利用预测的运动进行动画化，从任意相机视角进行渲染。实验结果表明，该方法能够生成物理合理的动画，在性能上显著优于现有方法。


---

## [84] GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments

### GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments

Novel view synthesis with neural models has advanced rapidly in recent years, yet adapting these models to scene changes remains an open problem. Existing methods are either labor-intensive, requiring extensive model retraining, or fail to capture detailed types of changes over time. In this paper, we present GaussianUpdate, a novel approach that combines 3D Gaussian representation with continual learning to address these challenges. Our method effectively updates the Gaussian radiance fields with current data while preserving information from past scenes. Unlike existing methods, GaussianUpdate explicitly models different types of changes through a novel multi-stage update strategy. Additionally, we introduce a visibility-aware continual learning approach with generative replay, enabling self-aware updating without the need to store images. The experiments on the benchmark dataset demonstrate our method achieves superior and real-time rendering with the capability of visualizing changes over different times

近年来，基于神经网络的新视角合成技术发展迅速，但如何使这些模型适应场景变化仍然是一个未解决的问题。现有方法要么劳动强度大，需要大量模型重新训练，要么无法捕捉场景随时间变化的细节。本文提出了一种名为 **GaussianUpdate** 的新方法，将三维高斯表示与持续学习相结合，以应对这些挑战。我们的方法能够在利用当前数据更新高斯辐射场的同时，保留历史场景信息。与现有方法不同，GaussianUpdate 通过一种新颖的多阶段更新策略，显式建模不同类型的场景变化。此外，我们提出了一种结合生成回放的可见性感知持续学习方法，使模型能够在无需存储图像的情况下实现自适应更新。基准数据集上的实验结果表明，我们的方法在实时渲染与跨时间可视化场景变化方面均实现了优越表现。


---

## [85] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing

### SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing

Creating high-fidelity and editable head avatars is a pivotal challenge in computer vision and graphics, boosting many AR/VR applications. While recent advancements have achieved photorealistic renderings and plausible animation, head editing, especially real-time appearance editing, remains challenging due to the implicit representation and entangled modeling of the geometry and global appearance. To address this, we propose Surface-Volumetric Gaussian Head Avatar (SVG-Head), a novel hybrid representation that explicitly models the geometry with 3D Gaussians bound on a FLAME mesh and leverages disentangled texture images to capture the global appearance. Technically, it contains two types of Gaussians, in which surface Gaussians explicitly model the appearance of head avatars using learnable texture images, facilitating real-time texture editing, while volumetric Gaussians enhance the reconstruction quality of non-Lambertian regions (e.g., lips and hair). To model the correspondence between 3D world and texture space, we provide a mesh-aware Gaussian UV mapping method, which leverages UV coordinates given by the FLAME mesh to obtain sharp texture images and real-time rendering speed. A hierarchical optimization strategy is further designed to pursue the optimal performance in both reconstruction quality and editing flexibility. Experiments on the NeRSemble dataset show that SVG-Head not only generates high-fidelity rendering results, but also is the first method to obtain explicit texture images for Gaussian head avatars and support real-time appearance editing.

高保真且可编辑的人头头像生成是计算机视觉与图形学中的一个关键挑战，对于增强现实（AR）和虚拟现实（VR）应用具有重要推动作用。尽管近期方法已实现了照片级逼真渲染与合理的动画效果，但头像编辑，尤其是实时外观编辑，仍然面临困难，主要源于隐式表示以及几何与整体外观的耦合建模。为解决这一问题，我们提出了 **表面-体积高斯头像（SVG-Head）**，这是一种新颖的混合表示方法：通过绑定在 FLAME 网格上的三维高斯显式建模几何，并利用解耦的纹理图像捕捉整体外观。在技术实现上，该方法包含两类高斯：表面高斯通过可学习的纹理图像显式建模头像外观，从而支持实时纹理编辑；体积高斯则提升了非朗伯区域（如嘴唇和头发）的重建质量。为建立三维世界与纹理空间的对应关系，我们提出了一种基于网格的高斯 UV 映射方法，利用 FLAME 网格提供的 UV 坐标获得清晰的纹理图像并实现实时渲染速度。我们进一步设计了一种分层优化策略，以同时追求重建质量与编辑灵活性的最优性能。在 NeRSemble 数据集上的实验表明，SVG-Head 不仅生成了高保真的渲染结果，而且是首个能够为高斯头像获得显式纹理图像并支持实时外观编辑的方法。


---

## [86] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos

### TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos

In this paper, we aim to model 3D scene geometry, appearance, and physical information just from dynamic multi-view videos in the absence of any human labels. By leveraging physics-informed losses as soft constraints or integrating simple physics models into neural nets, existing works often fail to learn complex motion physics, or doing so requires additional labels such as object types or masks. We propose a new framework named TRACE to model the motion physics of complex dynamic 3D scenes. The key novelty of our method is that, by formulating each 3D point as a rigid particle with size and orientation in space, we directly learn a translation rotation dynamics system for each particle, explicitly estimating a complete set of physical parameters to govern the particle's motion over time. Extensive experiments on three existing dynamic datasets and one newly created challenging synthetic datasets demonstrate the extraordinary performance of our method over baselines in the task of future frame extrapolation. A nice property of our framework is that multiple objects or parts can be easily segmented just by clustering the learned physical parameters.

本文旨在在没有任何人工标注的情况下，仅通过动态多视角视频来建模三维场景的几何、外观和物理信息。现有方法通常通过将物理约束损失作为软约束，或将简单物理模型集成到神经网络中，但往往无法有效学习复杂的运动物理，或者需要额外的标注（如物体类型或掩码）。我们提出了一个新的框架 TRACE，用于建模复杂动态三维场景的运动物理。该方法的关键创新在于：将每个三维点视为空间中具有大小和方向的刚体粒子，直接为每个粒子学习一个平移-旋转动力学系统，并显式估计一整套物理参数，以控制粒子随时间的运动。我们在三个现有的动态数据集和一个新构建的具有挑战性的合成数据集上进行了大量实验，结果表明该方法在未来帧外推任务中表现远超基线方法。该框架的一个良好特性是：只需对学习到的物理参数进行聚类，就能轻松地实现对多个物体或部分的分割。


---

## [87] WIPES: Wavelet-based Visual Primitives

### WIPES: Wavelet-based Visual Primitives

Pursuing a continuous visual representation that offers flexible frequency modulation and fast rendering speed has recently garnered increasing attention in the fields of 3D vision and graphics. However, existing representations often rely on frequency guidance or complex neural network decoding, leading to spectrum loss or slow rendering. To address these limitations, we propose WIPES, a universal Wavelet-based vIsual PrimitivES for representing multi-dimensional visual signals. Building on the spatial-frequency localization advantages of wavelets, WIPES effectively captures both the low-frequency "forest" and the high-frequency "trees." Additionally, we develop a wavelet-based differentiable rasterizer to achieve fast visual rendering. Experimental results on various visual tasks, including 2D image representation, 5D static and 6D dynamic novel view synthesis, demonstrate that WIPES, as a visual primitive, offers higher rendering quality and faster inference than INR-based methods, and outperforms Gaussian-based representations in rendering quality.

在三维视觉与图形学领域，追求既能灵活调控频率又能实现高速渲染的连续视觉表示正引起越来越多的关注。然而，现有表示往往依赖频率引导或复杂的神经网络解码，导致频谱丢失或渲染速度缓慢。为克服这些限制，我们提出了 WIPES，这是一种通用的基于小波的多维视觉信号表示方法。利用小波在空间-频率局部化上的优势，WIPES 能够有效捕捉低频的“整体”（forest）和高频的“细节”（trees）。此外，我们还开发了一个基于小波的可微光栅化器，实现快速的视觉渲染。在多种视觉任务上（包括二维图像表示、五维静态与六维动态新视角合成）的实验结果表明，作为一种视觉基元，WIPES 比基于 INR 的方法具有更高的渲染质量和更快的推理速度，并且在渲染质量上优于基于高斯的表示方法。


---

## [88] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos

### LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos

LongSplat addresses critical challenges in novel view synthesis (NVS) from casually captured long videos characterized by irregular camera motion, unknown camera poses, and expansive scenes. Current methods often suffer from pose drift, inaccurate geometry initialization, and severe memory limitations. To address these issues, we introduce LongSplat, a robust unposed 3D Gaussian Splatting framework featuring: (1) Incremental Joint Optimization that concurrently optimizes camera poses and 3D Gaussians to avoid local minima and ensure global consistency; (2) a robust Pose Estimation Module leveraging learned 3D priors; and (3) an efficient Octree Anchor Formation mechanism that converts dense point clouds into anchors based on spatial density. Extensive experiments on challenging benchmarks demonstrate that LongSplat achieves state-of-the-art results, substantially improving rendering quality, pose accuracy, and computational efficiency compared to prior approaches.

LongSplat 针对从随意捕获的长视频中进行新视角合成（NVS）时面临的关键挑战，包括不规则的相机运动、未知的相机位姿以及大范围场景。现有方法常常遭遇位姿漂移、几何初始化不准确以及严重的内存限制等问题。为解决这些问题，我们提出了 LongSplat，一种鲁棒的无位姿三维高斯喷溅框架，主要包括三项创新：(1) 增量式联合优化，同时优化相机位姿和三维高斯，以避免局部最优并保证全局一致性；(2) 利用学习到的三维先验的鲁棒位姿估计模块；(3) 高效的八叉树锚点生成机制，将稠密点云基于空间密度转换为锚点。在具有挑战性的基准上进行的大量实验表明，LongSplat 在渲染质量、位姿精度和计算效率方面均显著优于现有方法，达到了当前最先进水平。


---

## [89] GWM: Towards Scalable Gaussian World Models for Robotic Manipulation

### GWM: Towards Scalable Gaussian World Models for Robotic Manipulation

Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.

由于现实世界交互的低效性，在学习到的世界模型中训练机器人策略正逐渐成为趋势。已有的基于图像的世界模型和策略虽然取得了先前的成功，但缺乏稳健的几何信息，而几何信息对于三维世界的一致空间和物理理解至关重要，即便是在经过互联网规模视频预训练的情况下仍然如此。为此，我们提出了一种新型的世界模型分支——高斯世界模型（Gaussian World Model，GWM），用于机器人操作，其通过推理机器人动作作用下高斯基元的传播来重建未来状态。其核心是结合三维变分自编码器的潜变量扩散Transformer（Diffusion Transformer, DiT），从而能够利用高斯点绘（Gaussian Splatting）实现精细的场景级未来状态重建。GWM不仅可以通过自监督的未来预测训练增强模仿学习智能体的视觉表征，还可以作为神经模拟器，支持基于模型的强化学习。仿真和真实环境的实验结果均表明，GWM能够在多样化机器人动作条件下精确预测未来场景，并进一步用于训练在性能上显著超越现有最先进方法的策略，展示了三维世界模型在数据扩展方面的潜在能力。


---

## [90] GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene Representations

### GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene Representations

We introduce GSVisLoc, a visual localization method designed for 3D Gaussian Splatting (3DGS) scene representations. Given a 3DGS model of a scene and a query image, our goal is to estimate the camera's position and orientation. We accomplish this by robustly matching scene features to image features. Scene features are produced by downsampling and encoding the 3D Gaussians while image features are obtained by encoding image patches. Our algorithm proceeds in three steps, starting with coarse matching, then fine matching, and finally by applying pose refinement for an accurate final estimate. Importantly, our method leverages the explicit 3DGS scene representation for visual localization without requiring modifications, retraining, or additional reference images. We evaluate GSVisLoc on both indoor and outdoor scenes, demonstrating competitive localization performance on standard benchmarks while outperforming existing 3DGS-based baselines. Moreover, our approach generalizes effectively to novel scenes without additional training.

我们提出了 **GSVisLoc**，一种专为三维高斯点绘（3D Gaussian Splatting, 3DGS）场景表示设计的视觉定位方法。给定一个场景的 3DGS 模型和一张查询图像，我们的目标是估计相机的位置和朝向。我们通过将场景特征与图像特征进行稳健匹配来实现这一目标。场景特征通过对三维高斯进行下采样和编码获得，而图像特征则通过对图像块进行编码获得。我们的算法分为三个步骤：首先是粗匹配，其次是精匹配，最后通过位姿优化获得准确的最终估计。重要的是，我们的方法直接利用了显式的 3DGS 场景表示来进行视觉定位，无需修改、重新训练或额外的参考图像。我们在室内和室外场景中对 GSVisLoc 进行了评估，结果表明其在标准基准上实现了具有竞争力的定位性能，并优于现有基于 3DGS 的基线方法。此外，我们的方法能够在无需额外训练的情况下有效泛化到新的场景。


---

## [91] Seam360GS: Seamless 360° Gaussian Splatting from Real-World Omnidirectional Images

### Seam360GS: Seamless 360° Gaussian Splatting from Real-World Omnidirectional Images

360-degree visual content is widely shared on platforms such as YouTube and plays a central role in virtual reality, robotics, and autonomous navigation. However, consumer-grade dual-fisheye systems consistently yield imperfect panoramas due to inherent lens separation and angular distortions. In this work, we introduce a novel calibration framework that incorporates a dual-fisheye camera model into the 3D Gaussian splatting pipeline. Our approach not only simulates the realistic visual artifacts produced by dual-fisheye cameras but also enables the synthesis of seamlessly rendered 360-degree images. By jointly optimizing 3D Gaussian parameters alongside calibration variables that emulate lens gaps and angular distortions, our framework transforms imperfect omnidirectional inputs into flawless novel view synthesis. Extensive evaluations on real-world datasets confirm that our method produces seamless renderings-even from imperfect images-and outperforms existing 360-degree rendering models.

360 度视觉内容在 YouTube 等平台上被广泛分享，并在虚拟现实、机器人技术和自动驾驶中发挥着核心作用。然而，消费级双鱼眼系统由于固有的镜头分离和角度畸变，常常生成不完美的全景图。在这项工作中，我们提出了一种新颖的标定框架，将双鱼眼相机模型引入三维高斯点绘（3D Gaussian Splatting）管线。我们的方法不仅能够模拟双鱼眼相机产生的真实视觉伪影，还可以合成无缝渲染的 360 度图像。通过联合优化三维高斯参数与标定变量（用于模拟镜头间隙和角度畸变），我们的框架能够将不完美的全向输入转化为无瑕的新视角合成结果。在真实世界数据集上的大量评估结果表明，我们的方法即便在输入图像不完美的情况下也能生成无缝渲染，并优于现有的 360 度渲染模型。


---

## [92] Im2Haircut: Single-view Strand-based Hair Reconstruction for Human Avatars

### Im2Haircut: Single-view Strand-based Hair Reconstruction for Human Avatars

We present a novel approach for 3D hair reconstruction from single photographs based on a global hair prior combined with local optimization. Capturing strand-based hair geometry from single photographs is challenging due to the variety and geometric complexity of hairstyles and the lack of ground truth training data. Classical reconstruction methods like multi-view stereo only reconstruct the visible hair strands, missing the inner structure of hairstyles and hampering realistic hair simulation. To address this, existing methods leverage hairstyle priors trained on synthetic data. Such data, however, is limited in both quantity and quality since it requires manual work from skilled artists to model the 3D hairstyles and create near-photorealistic renderings. To address this, we propose a novel approach that uses both, real and synthetic data to learn an effective hairstyle prior. Specifically, we train a transformer-based prior model on synthetic data to obtain knowledge of the internal hairstyle geometry and introduce real data in the learning process to model the outer structure. This training scheme is able to model the visible hair strands depicted in an input image, while preserving the general 3D structure of hairstyles. We exploit this prior to create a Gaussian-splatting-based reconstruction method that creates hairstyles from one or more images. Qualitative and quantitative comparisons with existing reconstruction pipelines demonstrate the effectiveness and superior performance of our method for capturing detailed hair orientation, overall silhouette, and backside consistency.

我们提出了一种基于全局发型先验与局部优化相结合的从单张照片进行三维头发重建的新方法。从单张照片中捕捉基于发丝的头发几何结构极具挑战性，这源于发型的多样性与几何复杂性，以及缺乏真实的训练数据。传统的重建方法（如多视角立体重建）只能重建可见的发丝，忽略了发型的内部结构，从而阻碍了真实感头发模拟。为了解决这一问题，现有方法通常利用在合成数据上训练的发型先验。然而，这类数据在数量和质量上都存在局限性，因为其需要专业艺术家手工建模三维发型并制作近乎照片级的渲染。针对这一问题，我们提出了一种新方法，结合真实数据和合成数据共同学习有效的发型先验。具体而言，我们在合成数据上训练一个基于Transformer的先验模型，以获取发型内部几何结构知识，并在训练过程中引入真实数据来建模外部结构。该训练方案能够对输入图像中可见的发丝进行建模，同时保持发型整体的三维结构。我们进一步利用这一先验，提出了一种基于高斯溅射的重建方法，可从一张或多张图像生成发型。与现有重建流程的定性与定量比较表明，我们的方法在捕捉头发细节方向、整体轮廓以及背面一致性方面具有有效性和优越性能。


---

## [93] T2Bs: Text-to-Character Blendshapes via Video Generation

### T2Bs: Text-to-Character Blendshapes via Video Generation

We present T2Bs, a framework for generating high-quality, animatable character head morphable models from text by combining static text-to-3D generation with video diffusion. Text-to-3D models produce detailed static geometry but lack motion synthesis, while video diffusion models generate motion with temporal and multi-view geometric inconsistencies. T2Bs bridges this gap by leveraging deformable 3D Gaussian splatting to align static 3D assets with video outputs. By constraining motion with static geometry and employing a view-dependent deformation MLP, T2Bs (i) outperforms existing 4D generation methods in accuracy and expressiveness while reducing video artifacts and view inconsistencies, and (ii) reconstructs smooth, coherent, fully registered 3D geometries designed to scale for building morphable models with diverse, realistic facial motions. This enables synthesizing expressive, animatable character heads that surpass current 4D generation techniques.

我们提出了T2Bs框架，一种结合静态文本到三维（text-to-3D）生成与视频扩散（video diffusion）的方法，用于从文本生成高质量、可动画化的角色头部可变形模型（morphable models）。现有的text-to-3D模型能够生成细致的静态几何结构，但缺乏运动合成能力；而视频扩散模型虽然能够生成运动，但通常存在时间一致性差和多视图几何不一致等问题。T2Bs通过引入可变形三维高斯溅射（deformable 3D Gaussian splatting）来对齐静态三维资产与视频输出，从而弥合两者之间的差距。该方法通过静态几何约束运动，并引入视角相关的变形多层感知机（view-dependent deformation MLP），从而：（i）在准确性与表现力方面优于现有的四维生成（4D generation）方法，同时有效减少视频伪影与视角不一致问题；（ii）能够重建平滑、连贯且完全配准的三维几何，为构建具有多样化且逼真面部运动的可变形模型提供可扩展基础。这一框架实现了超越当前4D生成技术的高表现力可动画角色头部合成能力。


---

