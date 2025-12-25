## [1] An Efficient 3D Gaussian Representation for Monocular/Multi-view Dynamic Scenes

### An Efficient 3D Gaussian Representation for Monocular/Multi-view Dynamic Scenes

In novel view synthesis of scenes from multiple input views, 3D Gaussian splatting emerges as a viable alternative to existing radiance field approaches, delivering great visual quality and real-time rendering. While successful in static scenes, the present advancement of 3D Gaussian representation, however, faces challenges in dynamic scenes in terms of memory consumption and the need for numerous observations per time step, due to the onus of storing 3D Gaussian parameters per time step. In this study, we present an efficient 3D Gaussian representation tailored for dynamic scenes in which we define positions and rotations as functions of time while leaving other time-invariant properties of the static 3D Gaussian unchanged. Notably, our representation reduces memory usage, which is consistent regardless of the input sequence length. Additionally, it mitigates the risk of overfitting observed frames by accounting for temporal changes. The optimization of our Gaussian representation based on image and flow reconstruction results in a powerful framework for dynamic scene view synthesis in both monocular and multi-view cases. We obtain the highest rendering speed of 118 frames per second (FPS) at a resolution of 1352×1014 with a single GPU, showing the practical usability and effectiveness of our proposed method in dynamic scene rendering scenarios.

在从多个输入视图合成场景的新视角中，3D高斯溅射作为现有辐射场方法的一种可行替代方案出现，提供了出色的视觉质量和实时渲染。虽然在静态场景中取得了成功，但目前3D高斯表示在动态场景中面临着挑战，主要是由于需要存储每个时间步的3D高斯参数，导致内存消耗大和每个时间步需要大量观测。在这项研究中，我们提出了一种针对动态场景量身定制的高效3D高斯表示，我们将位置和旋转定义为时间的函数，同时保持静态3D高斯的其他时间不变属性不变。值得注意的是，我们的表示减少了内存使用，这与输入序列的长度无关。此外，它通过考虑时间变化，减轻了过度拟合观测帧的风险。我们基于图像和流重建优化的高斯表示形成了一个强大的框架，用于在单目和多视图情况下进行动态场景视图合成。我们在单个GPU上以1352×1014的分辨率实现了最高118帧每秒（FPS）的渲染速度，显示了我们提出的方法在动态场景渲染场景中的实用性和有效性。


---

## [2] Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing

### Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing

We present a novel differentiable point-based rendering framework for material and lighting decomposition from multi-view images, enabling editing, ray-tracing, and real-time relighting of the 3D point cloud. Specifically, a 3D scene is represented as a set of relightable 3D Gaussian points, where each point is additionally associated with a normal direction, BRDF parameters, and incident lights from different directions. To achieve robust lighting estimation, we further divide incident lights of each point into global and local components, as well as view-dependent visibilities. The 3D scene is optimized through the 3D Gaussian Splatting technique while BRDF and lighting are decomposed by physically-based differentiable rendering. Moreover, we introduce an innovative point-based ray-tracing approach based on the bounding volume hierarchy for efficient visibility baking, enabling real-time rendering and relighting of 3D Gaussian points with accurate shadow effects. Extensive experiments demonstrate improved BRDF estimation and novel view rendering results compared to state-of-the-art material estimation approaches. Our framework showcases the potential to revolutionize the mesh-based graphics pipeline with a relightable, traceable, and editable rendering pipeline solely based on point cloud.

我们提出了一种新颖的可微分点基渲染框架，用于从多视图图像中进行材质和光照分解，使得3D点云的编辑、光线追踪和实时重新照明成为可能。具体来说，一个3D场景被表示为一组可重新照明的3D高斯点，其中每个点额外关联有法线方向、BRDF参数和来自不同方向的入射光。为了实现稳健的光照估计，我们进一步将每个点的入射光分为全局和局部组成部分，以及视角依赖的可见性。3D场景通过3D高斯飞溅技术进行优化，而BRDF和光照通过基于物理的可微分渲染进行分解。此外，我们引入了一种基于边界体积层次结构的创新点基光线追踪方法，用于高效的可见性烘焙，使得3D高斯点的实时渲染和重新照明能够实现准确的阴影效果。广泛的实验展示了与最先进的材质估计方法相比，我们的框架在BRDF估计和新视角渲染结果方面的改进。我们的框架展示了用基于点云的可重新照明、可追踪和可编辑渲染管线革命性地替代基于网格的图形管线的潜力。


---

## [3] Compact3D: Smaller and Faster Gaussian Splatting with Vector Quantization

### Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization

3D Gaussian Splatting is a new method for modeling and rendering 3D radiance fields that achieves much faster learning and rendering time compared to SOTA NeRF methods. However, it comes with a drawback in the much larger storage demand compared to NeRF methods since it needs to store the parameters for several 3D Gaussians. We notice that many Gaussians may share similar parameters, so we introduce a simple vector quantization method based on \kmeans algorithm to quantize the Gaussian parameters. Then, we store the small codebook along with the index of the code for each Gaussian. Moreover, we compress the indices further by sorting them and using a method similar to run-length encoding. We do extensive experiments on standard benchmarks as well as a new benchmark which is an order of magnitude larger than the standard benchmarks. We show that our simple yet effective method can reduce the storage cost for the original 3D Gaussian Splatting method by a factor of almost 20× with a very small drop in the quality of rendered images.

3D高斯喷溅是一种新的建模和渲染3D辐射场的方法，与最新的NeRF方法相比，它实现了更快的学习和渲染时间。然而，与NeRF方法相比，它的一个缺点是需要更大的存储需求，因为它需要存储几个3D高斯的参数。我们注意到许多高斯可能具有相似的参数，因此我们引入了一种基于\kmeans算法的简单向量量化方法来量化高斯参数。然后，我们存储小型码本以及每个高斯的码索引。此外，我们通过排序索引并使用类似于游程编码的方法进一步压缩索引。我们在标准基准测试以及一个比标准基准测试大一个数量级的新基准测试上进行了广泛的实验。我们展示了我们这种简单而有效的方法可以将原始3D高斯喷溅方法的存储成本减少近20倍，同时渲染图像的质量只有非常小的下降。


---

## [4] DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting

### DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting

Accurately and efficiently modeling dynamic scenes and motions is considered so challenging a task due to temporal dynamics and motion complexity. To address these challenges, we propose DynMF, a compact and efficient representation that decomposes a dynamic scene into a few neural trajectories. We argue that the per-point motions of a dynamic scene can be decomposed into a small set of explicit or learned trajectories. Our carefully designed neural framework consisting of a tiny set of learned basis queried only in time allows for rendering speed similar to 3D Gaussian Splatting, surpassing 120 FPS, while at the same time, requiring only double the storage compared to static scenes. Our neural representation adequately constrains the inherently underconstrained motion field of a dynamic scene leading to effective and fast optimization. This is done by biding each point to motion coefficients that enforce the per-point sharing of basis trajectories. By carefully applying a sparsity loss to the motion coefficients, we are able to disentangle the motions that comprise the scene, independently control them, and generate novel motion combinations that have never been seen before. We can reach state-of-the-art render quality within just 5 minutes of training and in less than half an hour, we can synthesize novel views of dynamic scenes with superior photorealistic quality. Our representation is interpretable, efficient, and expressive enough to offer real-time view synthesis of complex dynamic scene motions, in monocular and multi-view scenarios.

准确高效地建模动态场景和运动被认为是一个极具挑战性的任务，因为时间动态和运动复杂性。为了应对这些挑战，我们提出了DynMF，一种紧凑高效的表示，将动态场景分解为少量神经轨迹。我们认为，动态场景的每个点的运动可以分解为一小组显式或学习的轨迹。我们精心设计的神经框架由一小组仅在时间上查询的学习基础组成，允许与3D高斯喷溅相似的渲染速度，超过120 FPS，同时仅需要与静态场景相比两倍的存储空间。我们的神经表示充分约束了动态场景本质上不受约束的运动场，从而实现了有效且快速的优化。这是通过将每个点绑定到运动系数上实现的，这些运动系数强制每个点共享基础轨迹。通过对运动系数仔细应用稀疏损失，我们能够分离构成场景的运动，独立控制它们，并生成之前从未见过的新的运动组合。我们可以在短短5分钟的训练内达到最新的渲染质量，并在不到半小时内，我们可以合成具有卓越真实感质量的动态场景的新视图。我们的表示是可解释的、高效的，并且足够表现力，以提供复杂动态场景运动的实时视图合成，无论是单眼还是多视图场景。


---

## [5] FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

### FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

Novel view synthesis from limited observations remains an important and persistent task. However, high efficiency in existing NeRF-based few-shot view synthesis is often compromised to obtain an accurate 3D representation. To address this challenge, we propose a few-shot view synthesis framework based on 3D Gaussian Splatting that enables real-time and photo-realistic view synthesis with as few as three training views. The proposed method, dubbed FSGS, handles the extremely sparse initialized SfM points with a thoughtfully designed Gaussian Unpooling process. Our method iteratively distributes new Gaussians around the most representative locations, subsequently infilling local details in vacant areas. We also integrate a large-scale pre-trained monocular depth estimator within the Gaussians optimization process, leveraging online augmented views to guide the geometric optimization towards an optimal solution. Starting from sparse points observed from limited input viewpoints, our FSGS can accurately grow into unseen regions, comprehensively covering the scene and boosting the rendering quality of novel views. Overall, FSGS achieves state-of-the-art performance in both accuracy and rendering efficiency across diverse datasets, including LLFF, Mip-NeRF360, and Blender.

从有限的观察中合成新视图仍然是一个重要且持续的任务。然而，在现有基于NeRF的少样本视图合成中，为了获得准确的3D表示，往往会牺牲高效性。为了应对这一挑战，我们提出了一种基于3D高斯喷溅的少样本视图合成框架，该框架能够仅使用三个训练视图实现实时和真实感视图合成。我们提出的方法，称为FSGS，通过精心设计的高斯Unpooling过程处理极其稀疏的初始化SfM点。我们的方法迭代地在最具代表性的位置周围分布新高斯，随后在空白区域填充局部细节。我们还在高斯优化过程中整合了一个大规模预训练的单目深度估计器，利用在线增强视图指导几何优化朝着最优解发展。从有限输入视点观察到的稀疏点开始，我们的FSGS能够准确地扩展到未见区域，全面覆盖场景并提升新视图的渲染质量。总体而言，FSGS在多种数据集上都实现了最新的性能，包括LLFF、Mip-NeRF360和Blender，无论是在准确性还是渲染效率方面。


---

## [6] Gaussian Grouping: Segment and Edit Anything in 3D Scenes

### Gaussian Grouping: Segment and Edit Anything in 3D Scenes

The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by SAM, along with introduced 3D spatial consistency regularization. Comparing to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization and scene recomposition.

最近的高斯喷溅技术实现了3D场景的高质量和实时新视图合成。然而，它仅专注于外观和几何建模，而缺乏细粒度的对象级场景理解。为了解决这个问题，我们提出了高斯分组，这是对高斯喷溅的扩展，用于同时重建和分割开放世界3D场景中的任何事物。我们为每个高斯增加了一个紧凑的身份编码，允许根据3D场景中的对象实例或材料成员将高斯进行分组。我们不是求助于昂贵的3D标签，而是在可微渲染过程中通过利用SAM的2D遮罩预测来监督身份编码，同时引入了3D空间一致性正则化。与隐式的NeRF表示相比，我们展示了离散且分组的3D高斯可以以高视觉质量、细粒度和效率在3D中重建、分割和编辑任何事物。基于高斯分组，我们进一步提出了一种局部高斯编辑方案，该方案在多种场景编辑应用中显示出有效性，包括3D对象移除、修复、上色和场景重组。


---

## [7] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting

### HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting

3D head animation has seen major quality and runtime improvements over the last few years, particularly empowered by the advances in differentiable rendering and neural radiance fields. Real-time rendering is a highly desirable goal for real-world applications. We propose HeadGaS, the first model to use 3D Gaussian Splats (3DGS) for 3D head reconstruction and animation. In this paper we introduce a hybrid model that extends the explicit representation from 3DGS with a base of learnable latent features, which can be linearly blended with low-dimensional parameters from parametric head models to obtain expression-dependent final color and opacity values. We demonstrate that HeadGaS delivers state-of-the-art results in real-time inference frame rates, which surpasses baselines by up to ~2dB, while accelerating rendering speed by over x10.

三维头部动画在过去几年里取得了重大的质量和运行时间改进，特别是受益于可微分渲染和神经辐射场的进步。实时渲染是现实世界应用中非常渴望达到的目标。我们提出了 HeadGaS，这是第一个使用三维高斯分散（3DGS）进行三维头部重建和动画的模型。在本文中，我们介绍了一种混合模型，该模型将来自3DGS的显式表示与可学习的潜在特征基底相结合，这些特征可以与参数头部模型中的低维参数线性混合，以获得表情依赖的最终颜色和不透明度值。我们展示了 HeadGaS 在实时推理帧率方面提供了最先进的结果，其性能超过基准线高达约2dB，同时加速渲染速度超过10倍。


---

## [8] EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS

### EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS

Recently, 3D Gaussian splatting (3D-GS) has gained popularity in novel-view scene synthesis. It addresses the challenges of lengthy training times and slow rendering speeds associated with Neural Radiance Fields (NeRFs). Through rapid, differentiable rasterization of 3D Gaussians, 3D-GS achieves real-time rendering and accelerated training. They, however, demand substantial memory resources for both training and storage, as they require millions of Gaussians in their point cloud representation for each scene. We present a technique utilizing quantized embeddings to significantly reduce memory storage requirements and a coarse-to-fine training strategy for a faster and more stable optimization of the Gaussian point clouds. Our approach results in scene representations with fewer Gaussians and quantized representations, leading to faster training times and rendering speeds for real-time rendering of high resolution scenes. We reduce memory by more than an order of magnitude all while maintaining the reconstruction quality. We validate the effectiveness of our approach on a variety of datasets and scenes preserving the visual quality while consuming 10-20x less memory and faster training/inference speed.

近来，三维高斯分散（3D-GS）在新视角场景合成中获得了人气。它解决了与神经辐射场（NeRFs）相关的漫长训练时间和缓慢的渲染速度的挑战。通过快速、可微的三维高斯光栅化，3D-GS实现了实时渲染和加速训练。然而，它们需要大量的内存资源用于训练和存储，因为每个场景的点云表示需要数百万个高斯点。我们提出了一种使用量化嵌入的技术，显著降低了内存存储需求，并采用了从粗到细的训练策略，以更快、更稳定地优化高斯点云。我们的方法导致使用更少的高斯点和量化表示的场景表示，从而实现了更快的训练时间和渲染速度，用于实时渲染高分辨率场景。我们在保持重建质量的同时，将内存需求减少了一个数量级以上。我们在各种数据集和场景上验证了我们方法的有效性，同时保持了视觉质量，同时消耗了比以往少10-20倍的内存，并且训练/推断速度更快。


---

## [9] Learn to Optimize Denoising Scores for 3D Generation: A Unified and Improved Diffusion Prior on NeRF and 3D Gaussian Splatting

### Learn to Optimize Denoising Scores for 3D Generation: A Unified and Improved Diffusion Prior on NeRF and 3D Gaussian Splatting

We propose a unified framework aimed at enhancing the diffusion priors for 3D generation tasks. Despite the critical importance of these tasks, existing methodologies often struggle to generate high-caliber results. We begin by examining the inherent limitations in previous diffusion priors. We identify a divergence between the diffusion priors and the training procedures of diffusion models that substantially impairs the quality of 3D generation. To address this issue, we propose a novel, unified framework that iteratively optimizes both the 3D model and the diffusion prior. Leveraging the different learnable parameters of the diffusion prior, our approach offers multiple configurations, affording various trade-offs between performance and implementation complexity. Notably, our experimental results demonstrate that our method markedly surpasses existing techniques, establishing new state-of-the-art in the realm of text-to-3D generation. Furthermore, our approach exhibits impressive performance on both NeRF and the newly introduced 3D Gaussian Splatting backbones. Additionally, our framework yields insightful contributions to the understanding of recent score distillation methods, such as the VSD and DDS loss.

我们提出了一个旨在增强三维生成任务扩散先验的统一框架。尽管这些任务极其重要，现有方法在生成高质量结果方面往往面临挑战。我们首先审视了以往扩散先验中的固有局限性。我们发现了扩散先验与扩散模型训练程序之间的偏差，这大大降低了三维生成的质量。为了解决这个问题，我们提出了一个新颖的统一框架，该框架迭代优化三维模型和扩散先验。利用扩散先验的不同可学习参数，我们的方法提供了多种配置，允许在性能和实施复杂性之间进行各种权衡。值得注意的是，我们的实验结果表明，我们的方法显著超越了现有技术，在文本到三维生成领域确立了新的最先进水平。此外，我们的方法在神经辐射场（NeRF）和新引入的三维高斯分散骨架上都表现出色。此外，我们的框架对最近的得分蒸馏方法，如 VSD 和 DDS 损失的理解，提供了有意义的贡献。


---

## [10] Repaint123: Fast and High-quality One Image to 3D Generation with Progressive Controllable 2D Repainting

### Repaint123: Fast and High-quality One Image to 3D Generation with Progressive Controllable 2D Repainting

Recent one image to 3D generation methods commonly adopt Score Distillation Sampling (SDS). Despite the impressive results, there are multiple deficiencies including multi-view inconsistency, over-saturated and over-smoothed textures, as well as the slow generation speed. To address these deficiencies, we present Repaint123 to alleviate multi-view bias as well as texture degradation and speed up the generation process. The core idea is to combine the powerful image generation capability of the 2D diffusion model and the texture alignment ability of the repainting strategy for generating high-quality multi-view images with consistency. We further propose visibility-aware adaptive repainting strength for overlap regions to enhance the generated image quality in the repainting process. The generated high-quality and multi-view consistent images enable the use of simple Mean Square Error (MSE) loss for fast 3D content generation. We conduct extensive experiments and show that our method has a superior ability to generate high-quality 3D content with multi-view consistency and fine textures in 2 minutes from scratch.

最近的一种从单张图片生成三维内容的方法通常采用分数蒸馏采样（SDS）。尽管结果令人印象深刻，但还存在多个缺陷，包括多视角不一致性、过饱和和过平滑的纹理，以及生成速度慢。为了解决这些缺陷，我们提出了 Repaint123，旨在减轻多视角偏见以及纹理退化，并加快生成过程。核心思想是结合二维扩散模型的强大图像生成能力和重绘策略的纹理对齐能力，生成具有一致性的高质量多视角图像。我们进一步提出了可见性感知的自适应重绘强度，以增强重绘过程中生成图像的质量。生成的高质量且多视角一致的图像使得使用简单的均方误差（MSE）损失快速生成三维内容成为可能。我们进行了广泛的实验，并展示了我们的方法在2分钟内从零开始生成具有多视角一致性和精细纹理的高质量三维内容的卓越能力。


---

## [11] Compact 3D Scene Representation via Self-Organizing Gaussian Grids

### Compact 3D Scene Representation via Self-Organizing Gaussian Grids

3D Gaussian Splatting has recently emerged as a highly promising technique for modeling of static 3D scenes. In contrast to Neural Radiance Fields, it utilizes efficient rasterization allowing for very fast rendering at high-quality. However, the storage size is significantly higher, which hinders practical deployment, e.g.~on resource constrained devices. In this paper, we introduce a compact scene representation organizing the parameters of 3D Gaussian Splatting (3DGS) into a 2D grid with local homogeneity, ensuring a drastic reduction in storage requirements without compromising visual quality during rendering. Central to our idea is the explicit exploitation of perceptual redundancies present in natural scenes. In essence, the inherent nature of a scene allows for numerous permutations of Gaussian parameters to equivalently represent it. To this end, we propose a novel highly parallel algorithm that regularly arranges the high-dimensional Gaussian parameters into a 2D grid while preserving their neighborhood structure. During training, we further enforce local smoothness between the sorted parameters in the grid. The uncompressed Gaussians use the same structure as 3DGS, ensuring a seamless integration with established renderers. Our method achieves a reduction factor of 8x to 26x in size for complex scenes with no increase in training time, marking a substantial leap forward in the domain of 3D scene distribution and consumption.

三维高斯飞溅(3D Gaussian Splatting)技术近来已成为静态三维场景建模的一种非常有前景的技术。与神经辐射场(Neural Radiance Fields)相比，它利用高效的光栅化实现了高质量的快速渲染。然而，它的存储大小显著增加，这限制了它在资源受限设备上的实际部署。在本文中，我们引入了一种紧凑的场景表示方法，将三维高斯飞溅的参数组织到具有局部同质性的二维网格中，从而大幅减少存储需求，同时在渲染过程中不影响视觉质量。我们的想法核心是明确利用自然场景中存在的感知冗余。本质上，场景的固有特性允许使用众多高斯参数的排列来等效地表示它。为此，我们提出了一种新颖的高度并行算法，它将高维高斯参数有规律地排列到二维网格中，同时保留它们的邻域结构。在训练过程中，我们进一步在网格中对排序的参数施加局部平滑性。未压缩的高斯使用与三维高斯飞溅相同的结构，确保与现有渲染器的无缝集成。我们的方法在复杂场景的大小上实现了8倍至26倍的减少，且不增加训练时间，标志着在三维场景分发和消费领域的一大飞跃。


---

## [12] Deblurring 3D Gaussian Splatting

### Deblurring 3D Gaussian Splatting

Recent studies in Radiance Fields have paved the robust way for novel view synthesis with their photorealistic rendering quality. Nevertheless, they usually employ neural networks and volumetric rendering, which are costly to train and impede their broad use in various real-time applications due to the lengthy rendering time. Lately 3D Gaussians splatting-based approach has been proposed to model the 3D scene, and it achieves remarkable visual quality while rendering the images in real-time. However, it suffers from severe degradation in the rendering quality if the training images are blurry. Blurriness commonly occurs due to the lens defocusing, object motion, and camera shake, and it inevitably intervenes in clean image acquisition. Several previous studies have attempted to render clean and sharp images from blurry input images using neural fields. The majority of those works, however, are designed only for volumetric rendering-based neural radiance fields and are not straightforwardly applicable to rasterization-based 3D Gaussian splatting methods. Thus, we propose a novel real-time deblurring framework, deblurring 3D Gaussian Splatting, using a small Multi-Layer Perceptron (MLP) that manipulates the covariance of each 3D Gaussian to model the scene blurriness. While deblurring 3D Gaussian Splatting can still enjoy real-time rendering, it can reconstruct fine and sharp details from blurry images. A variety of experiments have been conducted on the benchmark, and the results have revealed the effectiveness of our approach for deblurring.

最近在辐射场的研究为新视角合成铺平了一条坚实的道路，其逼真的渲染质量令人印象深刻。然而，它们通常采用神经网络和体积渲染，这在训练上成本高昂，并且由于渲染时间过长，阻碍了它们在各种实时应用中的广泛使用。最近，基于3D高斯涂抹的方法被提出来模拟3D场景，并在实时渲染图像时实现了显著的视觉质量。然而，如果训练图像模糊，它会严重降低渲染质量。由于镜头失焦、物体运动和相机抖动，模糊通常发生，并且不可避免地干扰了清晰图像的获取。一些先前的研究已经尝试使用神经场从模糊输入图像中渲染出清晰锐利的图像。然而，这些工作的大多数只设计用于基于体积渲染的神经辐射场，并不适用于基于光栅化的3D高斯涂抹方法。因此，我们提出了一种新的实时去模糊框架，使用一个小型的多层感知器（MLP）操作每个3D高斯的协方差来模拟场景模糊度，从而去模糊3D高斯涂抹。虽然去模糊3D高斯涂抹仍然可以实现实时渲染，但它可以从模糊图像中重建出细腻和锐利的细节。我们在基准上进行了多种实验，结果显示了我们方法的去模糊效果。


---

## [13] Street Gaussians for Modeling Dynamic Urban Scenes

### Street Gaussians for Modeling Dynamic Urban Scenes

This paper aims to tackle the problem of modeling dynamic urban street scenes from monocular videos. Recent methods extend NeRF by incorporating tracked vehicle poses to animate vehicles, enabling photo-realistic view synthesis of dynamic urban street scenes. However, significant limitations are their slow training and rendering speed, coupled with the critical need for high precision in tracked vehicle poses. We introduce Street Gaussians, a new explicit scene representation that tackles all these limitations. Specifically, the dynamic urban street is represented as a set of point clouds equipped with semantic logits and 3D Gaussians, each associated with either a foreground vehicle or the background. To model the dynamics of foreground object vehicles, each object point cloud is optimized with optimizable tracked poses, along with a dynamic spherical harmonics model for the dynamic appearance. The explicit representation allows easy composition of object vehicles and background, which in turn allows for scene editing operations and rendering at 133 FPS (1066×1600 resolution) within half an hour of training. The proposed method is evaluated on multiple challenging benchmarks, including KITTI and Waymo Open datasets. Experiments show that the proposed method consistently outperforms state-of-the-art methods across all datasets. Furthermore, the proposed representation delivers performance on par with that achieved using precise ground-truth poses, despite relying only on poses from an off-the-shelf tracker.

本文旨在解决从单目视频建模动态城市街景的问题。最近的方法通过结合跟踪的车辆姿态来扩展NeRF，以激活车辆，实现动态城市街景的逼真视角合成。然而，这些方法的显著局限性在于它们的训练和渲染速度缓慢，加上对跟踪车辆姿态高精度的关键需求。我们引入了Street Gaussians，这是一种新的显式场景表征，解决了所有这些限制。具体来说，动态城市街道被表示为一组点云，配备语义逻辑和3D高斯，每个高斯都与前景车辆或背景相关联。为了模拟前景物体车辆的动态，每个物体点云都通过可优化的跟踪姿态进行优化，同时还有一个动态球形谐波模型来表达动态外观。显式表征允许轻松组合物体车辆和背景，这反过来允许进行场景编辑操作，并在半小时的训练内以133 FPS（1066×1600分辨率）渲染。所提出的方法在包括KITTI和Waymo Open数据集在内的多个具有挑战性的基准上进行了评估。实验表明，提出的方法在所有数据集上始终优于最先进的方法。此外，尽管仅依赖于现成跟踪器的姿态，提出的表征在性能上与使用精确地面真实姿态达到的水平相当。


---

## [14] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy

### Optimal Projection for 3D Gaussian Splatting

3D Gaussian Splatting has garnered extensive attention and application in real-time neural rendering. Concurrently, concerns have been raised about the limitations of this technology in aspects such as point cloud storage, performance , and robustness in sparse viewpoints , leading to various improvements. However, there has been a notable lack of attention to the projection errors introduced by the local affine approximation inherent in the splatting itself, and the consequential impact of these errors on the quality of photo-realistic rendering. This paper addresses the projection error function of 3D Gaussian Splatting, commencing with the residual error from the first-order Taylor expansion of the projection function ϕ. The analysis establishes a correlation between the error and the Gaussian mean position. Subsequently, leveraging function optimization theory, this paper analyzes the function's minima to provide an optimal projection strategy for Gaussian Splatting referred to Optimal Gaussian Splatting. Experimental validation further confirms that this projection methodology reduces artifacts, resulting in a more convincingly realistic rendering.

3D高斯散射在实时神经渲染中获得了广泛的关注和应用。同时，也有人对这项技术在点云存储、性能以及在稀疏视点下的鲁棒性等方面的局限性提出了担忧，这导致了各种改进。然而，对于散射本身固有的局部仿射近似引入的投影错误及这些错误对于照片级真实渲染质量的影响，缺乏足够的关注。本文讨论了3D高斯散射的投影误差函数，从投影函数ϕ的一阶泰勒展开的残差误差开始。分析建立了误差与高斯平均位置之间的相关性。随后，利用函数优化理论，本文分析了函数的最小值，以提供一个称为最优高斯散射的高斯散射的最优投影策略。实验验证进一步确认了这种投影方法减少了伪影，结果是更加令人信服的真实渲染。


---

## [15] SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM

### SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM

Semantic understanding plays a crucial role in Dense Simultaneous Localization and Mapping (SLAM), facilitating comprehensive scene interpretation. Recent advancements that integrate Gaussian Splatting into SLAM systems have demonstrated its effectiveness in generating high-quality renderings through the use of explicit 3D Gaussian representations. Building on this progress, we propose SGS-SLAM, the first semantic dense visual SLAM system grounded in 3D Gaussians, which provides precise 3D semantic segmentation alongside high-fidelity reconstructions. Specifically, we propose to employ multi-channel optimization during the mapping process, integrating appearance, geometric, and semantic constraints with key-frame optimization to enhance reconstruction quality. Extensive experiments demonstrate that SGS-SLAM delivers state-of-the-art performance in camera pose estimation, map reconstruction, and semantic segmentation, outperforming existing methods meanwhile preserving real-time rendering ability.

语义理解在密集型同时定位与地图构建（SLAM）中扮演着至关重要的角色，它促进了对场景的全面解释。近期将高斯喷溅技术整合到SLAM系统中的进展证明了其在通过使用显式的3D高斯表示生成高质量渲染图像方面的有效性。基于这一进展，我们提出了SGS-SLAM，这是第一个基于3D高斯的语义密集视觉SLAM系统，它提供精确的3D语义分割与高保真重建。具体来说，我们提议在映射过程中采用多通道优化，整合外观、几何和语义约束与关键帧优化来提升重建质量。广泛的实验表明，SGS-SLAM在相机位姿估计、地图重建和语义分割方面提供了最先进的性能，同时保持了实时渲染能力，超越了现有方法。


---

## [16] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

### LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

3D content creation has achieved significant progress in terms of both quality and speed. Although current feed-forward models can produce 3D objects in seconds, their resolution is constrained by the intensive computation required during training. In this paper, we introduce Large Multi-View Gaussian Model (LGM), a novel framework designed to generate high-resolution 3D models from text prompts or single-view images. Our key insights are two-fold: 1) 3D Representation: We propose multi-view Gaussian features as an efficient yet powerful representation, which can then be fused together for differentiable rendering. 2) 3D Backbone: We present an asymmetric U-Net as a high-throughput backbone operating on multi-view images, which can be produced from text or single-view image input by leveraging multi-view diffusion models. Extensive experiments demonstrate the high fidelity and efficiency of our approach. Notably, we maintain the fast speed to generate 3D objects within 5 seconds while boosting the training resolution to 512, thereby achieving high-resolution 3D content generation.

3D内容创作在质量和速度方面都取得了显著进展。尽管当前的前馈模型可以在几秒钟内产生3D对象，但它们的分辨率受到训练期间所需密集计算的限制。在这篇论文中，我们介绍了大型多视图高斯模型（LGM），这是一个旨在从文本提示或单视图图像生成高分辨率3D模型的新颖框架。我们的关键洞察有两点：1) 3D表示：我们提出多视图高斯特征作为一种高效且强大的表示，然后可以将其融合用于可微渲染。2) 3D骨干网络：我们展示了一个不对称的U-Net作为高通量骨干网络，操作在多视图图像上，这些多视图图像可以通过利用多视图扩散模型从文本或单视图图像输入产生。广泛的实验展示了我们方法的高保真度和效率。值得注意的是，我们保持了在5秒内生成3D对象的快速速度，同时将训练分辨率提高到512，从而实现了高分辨率3D内容的生成。


---

## [17] HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting

### HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting

Creating digital avatars from textual prompts has long been a desirable yet challenging task. Despite the promising outcomes obtained through 2D diffusion priors in recent works, current methods face challenges in achieving high-quality and animated avatars effectively. In this paper, we present HeadStudio, a novel framework that utilizes 3D Gaussian splatting to generate realistic and animated avatars from text prompts. Our method drives 3D Gaussians semantically to create a flexible and achievable appearance through the intermediate FLAME representation. Specifically, we incorporate the FLAME into both 3D representation and score distillation: 1) FLAME-based 3D Gaussian splatting, driving 3D Gaussian points by rigging each point to a FLAME mesh. 2) FLAME-based score distillation sampling, utilizing FLAME-based fine-grained control signal to guide score distillation from the text prompt. Extensive experiments demonstrate the efficacy of HeadStudio in generating animatable avatars from textual prompts, exhibiting visually appealing appearances. The avatars are capable of rendering high-quality real-time (≥40 fps) novel views at a resolution of 1024. They can be smoothly controlled by real-world speech and video. We hope that HeadStudio can advance digital avatar creation and that the present method can widely be applied across various domains.

从文本提示创建数字化头像一直是一个令人期待但又充满挑战的任务。尽管通过在最近的研究中使用2D扩散先验获得了有希望的结果，当前方法在有效地实现高质量和动画化头像方面面临挑战。在本文中，我们介绍了HeadStudio，一个新颖的框架，它利用3D高斯喷溅技术从文本提示生成逼真和动画化的头像。我们的方法通过中间的FLAME表示，语义驱动3D高斯体，以创建灵活且可实现的外观。具体来说，我们将FLAME融入到3D表示和分数蒸馏中：1）基于FLAME的3D高斯喷溅，通过将每个点绑定到FLAME网格来驱动3D高斯点。2）基于FLAME的分数蒸馏采样，利用基于FLAME的细粒度控制信号来指导从文本提示中的分数蒸馏。广泛的实验展示了HeadStudio在从文本提示生成可动画化头像方面的有效性，展示了视觉上吸引人的外观。这些头像能够以1024的分辨率渲染高质量实时（≥40 fps）新视图。它们可以被真实世界的语音和视频平滑控制。我们希望HeadStudio能推进数字头像创建，而且当前方法能广泛应用于各个领域。


---

## [18] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis

### Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis

X-ray is widely applied for transmission imaging due to its stronger penetration than natural light. When rendering novel view X-ray projections, existing methods mainly based on NeRF suffer from long training time and slow inference speed. In this paper, we propose a 3D Gaussian splatting-based framework, namely X-Gaussian, for X-ray novel view synthesis. Firstly, we redesign a radiative Gaussian point cloud model inspired by the isotropic nature of X-ray imaging. Our model excludes the influence of view direction when learning to predict the radiation intensity of 3D points. Based on this model, we develop a Differentiable Radiative Rasterization (DRR) with CUDA implementation. Secondly, we customize an Angle-pose Cuboid Uniform Initialization (ACUI) strategy that directly uses the parameters of the X-ray scanner to compute the camera information and then uniformly samples point positions within a cuboid enclosing the scanned object. Experiments show that our X-Gaussian outperforms state-of-the-art methods by 6.5 dB while enjoying less than 15% training time and over 73x inference speed. The application on sparse-view CT reconstruction also reveals the practical values of our method.

X射线由于其比自然光更强的穿透能力，被广泛应用于透射成像。在渲染新视角X射线投影时，现有方法主要基于NeRF，遭受长时间训练和慢速推理的问题。在本文中，我们提出了一个基于3D高斯Splatting的框架，命名为X-Gaussian，用于X射线新视角合成。首先，我们重新设计了一个辐射高斯点云模型，灵感来自X射线成像的各向同性特性。我们的模型在学习预测3D点的辐射强度时排除了视角方向的影响。基于此模型，我们开发了一个具有CUDA实现的可微分辐射栅格化（DRR）。其次，我们定制了一个角度-姿态立方体均匀初始化（ACUI）策略，直接使用X射线扫描器的参数计算相机信息，然后在包围扫描对象的立方体内均匀采样点位置。实验表明，我们的X-Gaussian在性能上超越了最先进的方法6.5 dB，同时享受不到15%的训练时间和超过73倍的推理速度。在稀疏视图CT重建上的应用也揭示了我们方法的实际价值。


---

## [19] BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling

### BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling

Recent efforts in using 3D Gaussians for scene reconstruction and novel view synthesis can achieve impressive results on curated benchmarks; however, images captured in real life are often blurry. In this work, we analyze the robustness of Gaussian-Splatting-based methods against various image blur, such as motion blur, defocus blur, downscaling blur, \etc. Under these degradations, Gaussian-Splatting-based methods tend to overfit and produce worse results than Neural-Radiance-Field-based methods. To address this issue, we propose Blur Agnostic Gaussian Splatting (BAGS). BAGS introduces additional 2D modeling capacities such that a 3D-consistent and high quality scene can be reconstructed despite image-wise blur. Specifically, we model blur by estimating per-pixel convolution kernels from a Blur Proposal Network (BPN). BPN is designed to consider spatial, color, and depth variations of the scene to maximize modeling capacity. Additionally, BPN also proposes a quality-assessing mask, which indicates regions where blur occur. Finally, we introduce a coarse-to-fine kernel optimization scheme; this optimization scheme is fast and avoids sub-optimal solutions due to a sparse point cloud initialization, which often occurs when we apply Structure-from-Motion on blurry images. We demonstrate that BAGS achieves photorealistic renderings under various challenging blur conditions and imaging geometry, while significantly improving upon existing approaches.

近期在使用3D高斯进行场景重建和新视角合成的努力在精心策划的基准测试上可以取得令人印象深刻的结果；然而，实际生活中捕获的图像往往是模糊的。在这项工作中，我们分析了基于高斯Splatting方法对各种图像模糊（如运动模糊、散焦模糊、缩小模糊等）的鲁棒性。在这些退化条件下，基于高斯Splatting的方法往往会过拟合并产生比基于神经辐射场的方法更糟的结果。为了解决这个问题，我们提出了对模糊不敏感的高斯Splatting（BAGS）。BAGS引入了额外的2D建模能力，使得尽管存在图像级的模糊，也能重建出3D一致且高质量的场景。具体来说，我们通过估计每个像素的卷积核从模糊提议网络（BPN）来模拟模糊。BPN被设计为考虑场景的空间、颜色和深度变化，以最大化建模能力。此外，BPN还提出了一个质量评估掩码，指示出模糊发生的区域。最后，我们引入了一个从粗到细的核优化方案；这个优化方案快速且避免了由于在模糊图像上应用运动结构时经常发生的稀疏点云初始化而导致的次优解。我们证明BAGS在各种具有挑战性的模糊条件和成像几何下实现了逼真的渲染效果，同时显著改善了现有方法。



---

## [20] ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation

### ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation

Performing language-conditioned robotic manipulation tasks in unstructured environments is highly demanded for general intelligent robots. Conventional robotic manipulation methods usually learn semantic representation of the observation for action prediction, which ignores the scene-level spatiotemporal dynamics for human goal completion. In this paper, we propose a dynamic Gaussian Splatting method named ManiGaussian for multi-task robotic manipulation, which mines scene dynamics via future scene reconstruction. Specifically, we first formulate the dynamic Gaussian Splatting framework that infers the semantics propagation in the Gaussian embedding space, where the semantic representation is leveraged to predict the optimal robot action. Then, we build a Gaussian world model to parameterize the distribution in our dynamic Gaussian Splatting framework, which provides informative supervision in the interactive environment via future scene reconstruction. We evaluate our ManiGaussian on 10 RLBench tasks with 166 variations, and the results demonstrate our framework can outperform the state-of-the-art methods by 13.1\% in average success rate.

在非结构化环境中执行语言条件下的机器人操纵任务对于通用智能机器人来说需求极高。传统的机器人操纵方法通常学习观察的语义表示以预测动作，这忽略了完成人类目标的场景级时空动态。在本文中，我们提出了一种名为ManiGaussian的动态高斯溅射方法，用于多任务机器人操纵，该方法通过未来场景重建来挖掘场景动态。具体来说，我们首先构建了动态高斯溅射框架，该框架推断高斯嵌入空间中的语义传播，其中语义表示被利用来预测最优的机器人动作。然后，我们构建了一个高斯世界模型来参数化我们的动态高斯溅射框架中的分布，该模型通过未来场景重建在交互环境中提供信息丰富的监督。我们在10个RLBench任务上评估了我们的ManiGaussian，包含166种变化，结果表明我们的框架可以平均成功率比最先进方法高出13.1\%。


---

## [21] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing

### GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing

We propose GaussCtrl, a text-driven method to edit a 3D scene reconstructed by the 3D Gaussian Splatting (3DGS).
Our method first renders a collection of images by using the 3DGS and edits them by using a pre-trained 2D diffusion model (ControlNet) based on the input prompt, which is then used to optimise the 3D model.
Our key contribution is multi-view consistent editing, which enables editing all images together instead of iteratively editing one image while updating the 3D model as in previous works.
It leads to faster editing as well as higher visual quality.
This is achieved by the two terms:
(a) depth-conditioned editing that enforces geometric consistency across multi-view images by leveraging naturally consistent depth maps.
(b) attention-based latent code alignment that unifies the appearance of edited images by conditioning their editing to several reference views through self and cross-view attention between images' latent representations.
Experiments demonstrate that our method achieves faster editing and better visual results than previous state-of-the-art methods.

我们提出了GaussCtrl，这是一种基于文本的方法，用于编辑由3D高斯溅射（3DGS）重建的3D场景。我们的方法首先使用3DGS渲染一系列图像，并根据输入提示，使用预训练的2D扩散模型（ControlNet）编辑它们，然后用于优化3D模型。我们的关键贡献是多视图一致性编辑，它允许同时编辑所有图像，而不是像以前的工作那样，迭代编辑一个图像同时更新3D模型。这导致编辑速度更快以及更高的视觉质量。这通过两个条款实现：
(a)深度条件编辑，通过利用自然一致的深度图来强制多视图图像间的几何一致性。
(b)基于注意力的潜码对齐，通过将编辑的图像条件化到几个参考视图上，并通过图像潜在表示之间的自注意力和交叉视图注意力，统一编辑图像的外观。实验表明，我们的方法比以往的最先进方法实现了更快的编辑和更好的视觉结果。


---

## [22] Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians

### Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians

Reconstructing and simulating elastic objects from visual observations is crucial for applications in computer vision and robotics. Existing methods, such as 3D Gaussians, provide modeling for 3D appearance and geometry but lack the ability to simulate physical properties or optimize parameters for heterogeneous objects. We propose Spring-Gaus, a novel framework that integrates 3D Gaussians with physics-based simulation for reconstructing and simulating elastic objects from multi-view videos. Our method utilizes a 3D Spring-Mass model, enabling the optimization of physical parameters at the individual point level while decoupling the learning of physics and appearance. This approach achieves great sample efficiency, enhances generalization, and reduces sensitivity to the distribution of simulation particles. We evaluate Spring-Gaus on both synthetic and real-world datasets, demonstrating accurate reconstruction and simulation of elastic objects. This includes future prediction and simulation under varying initial states and environmental parameters.

从视觉观察重建和模拟弹性对象对于计算机视觉和机器人学的应用至关重要。现有方法，如3D高斯，为3D外观和几何提供了建模，但缺乏模拟物理属性或为异质对象优化参数的能力。我们提出了一种名为Spring-Gaus的新型框架，将3D高斯与基于物理的模拟整合起来，用于从多视角视频重建和模拟弹性对象。我们的方法利用了一个3D弹簧质量模型，使得在个别点水平上优化物理参数成为可能，同时解耦了物理和外观的学习。这种方法实现了极高的样本效率，增强了泛化能力，并减少了对模拟粒子分布的敏感性。我们在合成和现实世界数据集上评估了Spring-Gaus，展示了准确重建和模拟弹性对象的能力。这包括在不同初始状态和环境参数下的未来预测和模拟。


---

## [23] GGRt: Towards Pose-free Generalizable 3D Gaussian Splatting in Real-time

### GGRt: Towards Generalizable 3D Gaussians without Pose Priors in Real-Time

This paper presents GGRt, a novel approach to generalizable novel view synthesis that alleviates the need for real camera poses, complexity in processing high-resolution images, and lengthy optimization processes, thus facilitating stronger applicability of 3D Gaussian Splatting (3D-GS) in real-world scenarios. Specifically, we design a novel joint learning framework that consists of an Iterative Pose Optimization Network (IPO-Net) and a Generalizable 3D-Gaussians (G-3DG) model. With the joint learning mechanism, the proposed framework can inherently estimate robust relative pose information from the image observations and thus primarily alleviate the requirement of real camera poses. Moreover, we implement a deferred back-propagation mechanism that enables high-resolution training and inference, overcoming the resolution constraints of previous methods. To enhance the speed and efficiency, we further introduce a progressive Gaussian cache module that dynamically adjusts during training and inference. As the first pose-free generalizable 3D-GS framework, GGRt achieves inference at ≥ 5 FPS and real-time rendering at ≥ 100 FPS. Through extensive experimentation, we demonstrate that our method outperforms existing NeRF-based pose-free techniques in terms of inference speed and effectiveness. It can also approach the real pose-based 3D-GS methods. Our contributions provide a significant leap forward for the integration of computer vision and computer graphics into practical applications, offering state-of-the-art results on LLFF, KITTI, and Waymo Open datasets and enabling real-time rendering for immersive experiences.

本文提出了GGRt，一种新颖的可泛化新视角合成方法，该方法减轻了对真实相机姿态的需求、处理高分辨率图像的复杂性以及漫长的优化过程，从而加强了3D高斯溅射（3D-GS）在现实世界场景中的应用性。具体来说，我们设计了一个新颖的联合学习框架，该框架由迭代姿态优化网络（IPO-Net）和可泛化3D高斯模型（G-3DG）组成。借助联合学习机制，所提出的框架可以从图像观测中固有地估计出稳健的相对姿态信息，从而主要减轻了对真实相机姿态的需求。此外，我们实现了一种延迟反向传播机制，使得高分辨率训练和推断成为可能，克服了先前方法的分辨率限制。为了提高速度和效率，我们进一步引入了一个渐进式高斯缓存模块，该模块在训练和推断过程中动态调整。作为首个无姿态可泛化3D-GS框架，GGRt实现了≥5 FPS的推断速度和≥100 FPS的实时渲染速度。通过广泛的实验，我们证明了我们的方法在推断速度和有效性方面超越了现有的基于NeRF的无姿态技术。它还可以接近真实姿态基的3D-GS方法。我们的贡献为计算机视觉与计算机图形学融入实际应用提供了重大进步，在LLFF、KITTI和Waymo Open数据集上提供了最先进的结果，并实现了沉浸式体验的实时渲染。


---

## [24] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting

### BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting

While neural rendering has demonstrated impressive capabilities in 3D scene reconstruction and novel view synthesis, it heavily relies on high-quality sharp images and accurate camera poses. Numerous approaches have been proposed to train Neural Radiance Fields (NeRF) with motion-blurred images, commonly encountered in real-world scenarios such as low-light or long-exposure conditions. However, the implicit representation of NeRF struggles to accurately recover intricate details from severely motion-blurred images and cannot achieve real-time rendering. In contrast, recent advancements in 3D Gaussian Splatting achieve high-quality 3D scene reconstruction and real-time rendering by explicitly optimizing point clouds as Gaussian spheres.
In this paper, we introduce a novel approach, named BAD-Gaussians (Bundle Adjusted Deblur Gaussian Splatting), which leverages explicit Gaussian representation and handles severe motion-blurred images with inaccurate camera poses to achieve high-quality scene reconstruction. Our method models the physical image formation process of motion-blurred images and jointly learns the parameters of Gaussians while recovering camera motion trajectories during exposure time.
In our experiments, we demonstrate that BAD-Gaussians not only achieves superior rendering quality compared to previous state-of-the-art deblur neural rendering methods on both synthetic and real datasets but also enables real-time rendering capabilities.

虽然神经渲染在3D场景重建和新视角合成方面展示了令人印象深刻的能力，但它严重依赖于高质量清晰图像和准确的相机姿态。许多方法已被提出来用运动模糊图像训练神经辐射场（NeRF），这是在实际场景中常遇到的情况，比如低光照或长时间曝光条件。然而，NeRF的隐式表示难以从严重运动模糊的图像中准确恢复出复杂的细节，且无法实现实时渲染。相比之下，最近在3D高斯喷溅方面的进展通过显式优化点云为高斯球，实现了高质量的3D场景重建和实时渲染。
在本文中，我们介绍了一种新颖的方法，名为BAD-Gaussians（Bundle Adjusted Deblur Gaussian Splatting），它利用显式高斯表示，并能处理严重运动模糊图像及不准确的相机姿态，以实现高质量的场景重建。我们的方法模拟了运动模糊图像的物理成像过程，并在曝光时间内共同学习高斯参数，同时恢复相机运动轨迹。
在我们的实验中，我们展示了BAD-Gaussians不仅在合成和真实数据集上相比之前的最先进去模糊神经渲染方法实现了更优越的渲染质量，而且还启用了实时渲染能力。


---

## [25] View-Consistent 3D Editing with Gaussian Splatting

### View-Consistent 3D Editing with Gaussian Splatting

The advent of 3D Gaussian Splatting (3DGS) has revolutionized 3D editing, offering efficient, high-fidelity rendering and enabling precise local manipulations. Currently, diffusion-based 2D editing models are harnessed to modify multi-view rendered images, which then guide the editing of 3DGS models. However, this approach faces a critical issue of multi-view inconsistency, where the guidance images exhibit significant discrepancies across views, leading to mode collapse and visual artifacts of 3DGS. To this end, we introduce View-consistent Editing (VcEdit), a novel framework that seamlessly incorporates 3DGS into image editing processes, ensuring multi-view consistency in edited guidance images and effectively mitigating mode collapse issues. VcEdit employs two innovative consistency modules: the Cross-attention Consistency Module and the Editing Consistency Module, both designed to reduce inconsistencies in edited images. By incorporating these consistency modules into an iterative pattern, VcEdit proficiently resolves the issue of multi-view inconsistency, facilitating high-quality 3DGS editing across a diverse range of scenes.

3D高斯喷溅（3DGS）的出现彻底革新了3D编辑，提供了高效、高保真的渲染并实现了精确的局部操作。目前，扩散基础的2D编辑模型被用于修改多视图渲染图像，这些图像随后指导3DGS模型的编辑。然而，这种方法面临一个关键问题，即多视图不一致性，其中指导图像在不同视图中展现出显著差异，导致模式崩溃和3DGS的视觉缺陷。为此，我们引入了视图一致性编辑（VcEdit），一个将3DGS无缝整合到图像编辑过程中的新颖框架，确保编辑后的指导图像具有多视图一致性，并有效缓解模式崩溃问题。VcEdit采用了两个创新的一致性模块：交叉注意力一致性模块和编辑一致性模块，都旨在减少编辑图像中的不一致性。通过将这些一致性模块纳入迭代模式，VcEdit熟练地解决了多视图不一致性问题，促进了在多样化场景中进行高质量3DGS编辑。


---

## [26] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration

### Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration

The 3D Gaussian Splatting (3DGS) gained its popularity recently by combining the advantages of both primitive-based and volumetric 3D representations, resulting in improved quality and efficiency for 3D scene rendering. However, 3DGS is not alias-free, and its rendering at varying resolutions could produce severe blurring or jaggies. This is because 3DGS treats each pixel as an isolated, single point rather than as an area, causing insensitivity to changes in the footprints of pixels. Consequently, this discrete sampling scheme inevitably results in aliasing, owing to the restricted sampling bandwidth. In this paper, we derive an analytical solution to address this issue. More specifically, we use a conditioned logistic function as the analytic approximation of the cumulative distribution function (CDF) in a one-dimensional Gaussian signal and calculate the Gaussian integral by subtracting the CDFs. We then introduce this approximation in the two-dimensional pixel shading, and present Analytic-Splatting, which analytically approximates the Gaussian integral within the 2D-pixel window area to better capture the intensity response of each pixel. Moreover, we use the approximated response of the pixel window integral area to participate in the transmittance calculation of volume rendering, making Analytic-Splatting sensitive to the changes in pixel footprint at different resolutions. Experiments on various datasets validate that our approach has better anti-aliasing capability that gives more details and better fidelity.

3D高斯平滑（3DGS）最近因结合了基于原始和体积3D表示的优势，从而提高了3D场景渲染的质量和效率而受到欢迎。然而，3DGS并非无别名，其在不同分辨率下的渲染可能会产生严重的模糊或锯齿。这是因为3DGS将每个像素视为一个孤立的单点而不是一个区域，导致对像素足迹变化的不敏感。因此，这种离散的采样方案不可避免地导致了别名问题，这是由于受限的采样带宽。在本文中，我们推导出一种解决这一问题的分析解。更具体地说，我们使用条件逻辑函数作为一维高斯信号的累积分布函数（CDF）的解析近似，并通过减去CDF来计算高斯积分。然后，我们在二维像素着色中引入这种近似，并提出分析平滑，它在2D像素窗口区域内解析近似高斯积分，以更好地捕捉每个像素的强度响应。此外，我们使用像素窗口积分区域的近似响应参与体渲染的透射计算，使分析平滑对不同分辨率下像素足迹的变化敏感。在各种数据集上的实验验证了我们的方法具有更好的抗锯齿能力，提供了更多细节和更好的保真度。


---

## [27] GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering

### GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering

During the Gaussian Splatting optimization process, the scene's geometry can gradually deteriorate if its structure is not deliberately preserved, especially in non-textured regions such as walls, ceilings, and furniture surfaces. This degradation significantly affects the rendering quality of novel views that deviate significantly from the viewpoints in the training data. To mitigate this issue, we propose a novel approach called GeoGaussian. Based on the smoothly connected areas observed from point clouds, this method introduces a novel pipeline to initialize thin Gaussians aligned with the surfaces, where the characteristic can be transferred to new generations through a carefully designed densification strategy. Finally, the pipeline ensures that the scene's geometry and texture are maintained through constrained optimization processes with explicit geometry constraints. Benefiting from the proposed architecture, the generative ability of 3D Gaussians is enhanced, especially in structured regions. Our proposed pipeline achieves state-of-the-art performance in novel view synthesis and geometric reconstruction, as evaluated qualitatively and quantitatively on public datasets.

在高斯平滑优化过程中，如果不特意保持场景的结构，场景的几何形态会逐渐恶化，特别是在非纹理区域如墙壁、天花板和家具表面。这种退化显著影响了从训练数据中的视点大幅偏离的新视角的渲染质量。为了缓解这个问题，我们提出了一种名为GeoGaussian的新方法。基于从点云观察到的平滑连接区域，这种方法引入了一种新的管道来初始化与表面对齐的细高斯，其中的特性可以通过精心设计的密集化策略转移到新生成物上。最后，该管道确保通过具有显式几何约束的受限优化过程保持场景的几何形态和纹理。得益于所提出的架构，3D高斯的生成能力得到了增强，特别是在结构化区域。我们提出的管道在新视角合成和几何重建方面达到了最新技术水平，已通过公共数据集上的定性和定量评估证实。


---

## [28] RGBD GS-ICP SLAM

### RGBD GS-ICP SLAM

Simultaneous Localization and Mapping (SLAM) with dense representation plays a key role in robotics, Virtual Reality (VR), and Augmented Reality (AR) applications. Recent advancements in dense representation SLAM have highlighted the potential of leveraging neural scene representation and 3D Gaussian representation for high-fidelity spatial representation. In this paper, we propose a novel dense representation SLAM approach with a fusion of Generalized Iterative Closest Point (G-ICP) and 3D Gaussian Splatting (3DGS). In contrast to existing methods, we utilize a single Gaussian map for both tracking and mapping, resulting in mutual benefits. Through the exchange of covariances between tracking and mapping processes with scale alignment techniques, we minimize redundant computations and achieve an efficient system. Additionally, we enhance tracking accuracy and mapping quality through our keyframe selection methods. Experimental results demonstrate the effectiveness of our approach, showing an incredibly fast speed up to 107 FPS (for the entire system) and superior quality of the reconstructed map.

在机器人、虚拟现实（VR）和增强现实（AR）应用中，具有密集表示的同时定位与建图（SLAM）起着关键作用。最近在密集表示SLAM的进展突显了利用神经场景表示和3D高斯表示进行高保真空间表示的潜力。在本文中，我们提出了一种新颖的密集表示SLAM方法，该方法融合了广义迭代最近点（G-ICP）和3D高斯喷溅（3DGS）。与现有方法不同，我们利用单一的高斯地图同时进行跟踪和映射，从而获得相互利益。通过在跟踪和映射过程中交换协方差，并使用比例对齐技术，我们最小化了冗余计算并实现了一个高效的系统。此外，我们通过我们的关键帧选择方法提高了跟踪精度和映射质量。实验结果证明了我们方法的有效性，显示出高达107 FPS（整个系统）的令人难以置信的快速速度和重建地图的优越质量。


---

## [29] GVGEN: Text-to-3D Generation with Volumetric Representation

### GVGEN: Text-to-3D Generation with Volumetric Representation

In recent years, 3D Gaussian splatting has emerged as a powerful technique for 3D reconstruction and generation, known for its fast and high-quality rendering capabilities. To address these shortcomings, this paper introduces a novel diffusion-based framework, GVGEN, designed to efficiently generate 3D Gaussian representations from text input. We propose two innovative techniques:(1) Structured Volumetric Representation. We first arrange disorganized 3D Gaussian points as a structured form GaussianVolume. This transformation allows the capture of intricate texture details within a volume composed of a fixed number of Gaussians. To better optimize the representation of these details, we propose a unique pruning and densifying method named the Candidate Pool Strategy, enhancing detail fidelity through selective optimization. (2) Coarse-to-fine Generation Pipeline. To simplify the generation of GaussianVolume and empower the model to generate instances with detailed 3D geometry, we propose a coarse-to-fine pipeline. It initially constructs a basic geometric structure, followed by the prediction of complete Gaussian attributes. Our framework, GVGEN, demonstrates superior performance in qualitative and quantitative assessments compared to existing 3D generation methods. Simultaneously, it maintains a fast generation speed (∼7 seconds), effectively striking a balance between quality and efficiency.

近年来，3D高斯喷溅作为一种强大的3D重建和生成技术而崭露头角，以其快速和高质量的渲染能力而闻名。为了解决这些不足，本文介绍了一种新颖的基于扩散的框架，GVGEN，旨在高效地从文本输入生成3D高斯表示。我们提出了两种创新技术：（1）结构化体积表示。我们首先将无组织的3D高斯点作为一种结构化形式的GaussianVolume排列。这种转换允许捕捉由固定数量的高斯组成的体积内的复杂纹理细节。为了更好地优化这些细节的表示，我们提出了一种独特的修剪和密集化方法，名为候选池策略，通过选择性优化增强细节保真度。（2）由粗到细的生成管道。为了简化GaussianVolume的生成并使模型能够生成具有详细3D几何形状的实例，我们提出了一种由粗到细的管道。它最初构建一个基本的几何结构，随后预测完整的高斯属性。我们的框架，GVGEN，在定性和定量评估中相比现有的3D生成方法表现出优越的性能。同时，它保持了快速的生成速度（∼7秒），有效地在质量和效率之间找到了平衡。


---

## [30] Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion

### Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion

High-quality scene reconstruction and novel view synthesis based on Gaussian Splatting (3DGS) typically require steady, high-quality photographs, often impractical to capture with handheld cameras. We present a method that adapts to camera motion and allows high-quality scene reconstruction with handheld video data suffering from motion blur and rolling shutter distortion. Our approach is based on detailed modelling of the physical image formation process and utilizes velocities estimated using visual-inertial odometry (VIO). Camera poses are considered non-static during the exposure time of a single image frame and camera poses are further optimized in the reconstruction process. We formulate a differentiable rendering pipeline that leverages screen space approximation to efficiently incorporate rolling-shutter and motion blur effects into the 3DGS framework. Our results with both synthetic and real data demonstrate superior performance in mitigating camera motion over existing methods, thereby advancing 3DGS in naturalistic settings.

高质量场景重建和新视角合成基于高斯喷溅（3DGS）通常需要稳定、高质量的照片，这往往难以通过手持相机捕捉实现。我们提出了一种方法，该方法能够适应相机运动，并允许使用受运动模糊和卷帘快门畸变影响的手持视频数据进行高质量场景重建。我们的方法基于对物理图像形成过程的详细建模，并利用视觉-惯性测程（VIO）估计出的速度。考虑到单个图像帧的曝光时间内相机姿态是非静态的，并且在重建过程中进一步优化相机姿态。我们构建了一个可微渲染管线，该管线利用屏幕空间近似高效地将卷帘快门和运动模糊效果纳入到3DGS框架中。我们使用合成数据和真实数据的结果展示了在减轻相机运动方面相较于现有方法的优越性能，从而推进了3DGS在自然场景设置中的应用。


---

## [31] Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians

### Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians

In this study, we explore the challenge of efficiently representing scenes with a constrained number of Gaussians. Our analysis shifts from traditional graphics and 2D computer vision to the perspective of point clouds, highlighting the inefficient spatial distribution of Gaussian representation as a key limitation in model performance. To address this, we introduce strategies for densification including blur split and depth reinitialization, and simplification through Gaussian binarization and sampling. These techniques reorganize the spatial positions of the Gaussians, resulting in significant improvements across various datasets and benchmarks in terms of rendering quality, resource consumption, and storage compression. Our proposed Mini-Splatting method integrates seamlessly with the original rasterization pipeline, providing a strong baseline for future research in Gaussian-Splatting-based works.

在这项研究中，我们探讨了如何高效地用有限数量的高斯函数表示场景的挑战。我们的分析从传统图形学和二维计算机视觉转向点云的视角，强调高斯表示的低效空间分布是模型性能的一个关键限制。为了解决这个问题，我们引入了密集化策略，包括模糊分裂和深度重新初始化，以及通过高斯二值化和采样来简化。这些技术重新组织了高斯的空间位置，导致在渲染质量、资源消耗和存储压缩方面在各种数据集和基准测试中的显著改进。我们提出的Mini-Splatting方法与原始光栅化管线无缝集成，为未来基于高斯喷溅的研究提供了一个强大的基线。


---

## [32] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression

### HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression

3D Gaussian Splatting (3DGS) has emerged as a promising framework for novel view synthesis, boasting rapid rendering speed with high fidelity. However, the substantial Gaussians and their associated attributes necessitate effective compression techniques. Nevertheless, the sparse and unorganized nature of the point cloud of Gaussians (or anchors in our paper) presents challenges for compression. To address this, we make use of the relations between the unorganized anchors and the structured hash grid, leveraging their mutual information for context modeling, and propose a Hash-grid Assisted Context (HAC) framework for highly compact 3DGS representation. Our approach introduces a binary hash grid to establish continuous spatial consistencies, allowing us to unveil the inherent spatial relations of anchors through a carefully designed context model. To facilitate entropy coding, we utilize Gaussian distributions to accurately estimate the probability of each quantized attribute, where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. Additionally, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors. Importantly, our work is the pioneer to explore context-based compression for 3DGS representation, resulting in a remarkable size reduction of over 75× compared to vanilla 3DGS, while simultaneously improving fidelity, and achieving over 11× size reduction over SOTA 3DGS compression approach Scaffold-GS.

3D高斯喷溅（3DGS）已成为新颖视图合成的一个有前途的框架，以其快速渲染速度和高保真度而自豪。然而，大量的高斯及其相关属性需要有效的压缩技术。尽管如此，高斯点云（或在我们的论文中称为锚点）的稀疏和无组织性质给压缩带来了挑战。为了解决这一问题，我们利用了无组织锚点与结构化哈希网格之间的关系，利用它们的互信息进行上下文建模，并提出了一个哈希网格辅助上下文（HAC）框架，用于高度紧凑的3DGS表示。我们的方法引入了二进制哈希网格以建立连续的空间一致性，允许我们通过精心设计的上下文模型揭示锚点的固有空间关系。为了促进熵编码，我们使用高斯分布来准确估计每个量化属性的概率，其中提出了一个自适应量化模块，以实现这些属性的高精度量化，从而改善保真度恢复。此外，我们加入了一个自适应遮罩策略来消除无效的高斯和锚点。重要的是，我们的工作是首次探索基于上下文的压缩，用于3DGS表示，与普通的3DGS相比，实现了超过75倍的显著大小减少，同时提高了保真度，并且与SOTA 3DGS压缩方法Scaffold-GS相比，实现了超过11倍的大小减少。


---

## [33] Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering

### Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering

We propose Gaussian Frosting, a novel mesh-based representation for high-quality rendering and editing of complex 3D effects in real-time. Our approach builds on the recent 3D Gaussian Splatting framework, which optimizes a set of 3D Gaussians to approximate a radiance field from images. We propose first extracting a base mesh from Gaussians during optimization, then building and refining an adaptive layer of Gaussians with a variable thickness around the mesh to better capture the fine details and volumetric effects near the surface, such as hair or grass. We call this layer Gaussian Frosting, as it resembles a coating of frosting on a cake. The fuzzier the material, the thicker the frosting. We also introduce a parameterization of the Gaussians to enforce them to stay inside the frosting layer and automatically adjust their parameters when deforming, rescaling, editing or animating the mesh. Our representation allows for efficient rendering using Gaussian splatting, as well as editing and animation by modifying the base mesh. We demonstrate the effectiveness of our method on various synthetic and real scenes, and show that it outperforms existing surface-based approaches. We will release our code and a web-based viewer as additional contributions.

我们提出了高斯霜化，这是一种新颖的基于网格的表示法，用于实时高质量渲染和编辑复杂的3D效果。我们的方法基于最近的3D高斯喷溅框架，该框架优化了一组3D高斯以从图像中近似辐射场。我们提出首先在优化过程中从高斯中提取一个基础网格，然后在网格周围构建并细化一个具有可变厚度的自适应高斯层，以更好地捕捉表面附近的细节和体积效果，如头发或草。我们称这层为高斯霜化，因为它类似于蛋糕上的一层霜。材料越模糊，霜化层越厚。我们还引入了高斯的参数化，以强制它们保持在霜化层内，并在变形、重新缩放、编辑或动画化网格时自动调整它们的参数。我们的表示法允许使用高斯喷溅进行高效渲染，以及通过修改基础网格进行编辑和动画制作。我们在各种合成和真实场景上演示了我们方法的有效性，并展示了它超越了现有的基于表面的方法。我们将发布我们的代码和一个基于网络的查看器作为额外的贡献。


---

## [34] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

### MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

We propose MVSplat, an efficient feed-forward 3D Gaussian Splatting model learned from sparse multi-view images. To accurately localize the Gaussian centers, we propose to build a cost volume representation via plane sweeping in the 3D space, where the cross-view feature similarities stored in the cost volume can provide valuable geometry cues to the estimation of depth. We learn the Gaussian primitives' opacities, covariances, and spherical harmonics coefficients jointly with the Gaussian centers while only relying on photometric supervision. We demonstrate the importance of the cost volume representation in learning feed-forward Gaussian Splatting models via extensive experimental evaluations. On the large-scale RealEstate10K and ACID benchmarks, our model achieves state-of-the-art performance with the fastest feed-forward inference speed (22 fps). Compared to the latest state-of-the-art method pixelSplat, our model uses 10× fewer parameters and infers more than 2× faster while providing higher appearance and geometry quality as well as better cross-dataset generalization.

我们提出了MVSplat，这是一个从稀疏多视图图像学习得来的高效前馈3D高斯喷溅模型。为了准确地定位高斯中心，我们提出通过在3D空间中进行平面扫描来构建成本体积表示，其中存储在成本体积中的跨视图特征相似性可以为深度估计提供宝贵的几何线索。我们学习高斯原语的不透明度、协方差和球谐函数系数，同时仅依赖于光度监督与高斯中心共同进行。我们通过广泛的实验评估，展示了成本体积表示在学习前馈高斯喷溅模型中的重要性。在大规模的RealEstate10K和ACID基准测试上，我们的模型以最快的前馈推理速度（22fps）实现了最先进的性能。与最新的最先进方法pixelSplat相比，我们的模型使用了10倍更少的参数，并且推理速度快2倍以上，同时提供了更高的外观和几何质量以及更好的跨数据集泛化能力。


---

## [35] STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians

### STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians

Recent progress in pre-trained diffusion models and 3D generation have spurred interest in 4D content creation. However, achieving high-fidelity 4D generation with spatial-temporal consistency remains a challenge. In this work, we propose STAG4D, a novel framework that combines pre-trained diffusion models with dynamic 3D Gaussian splatting for high-fidelity 4D generation. Drawing inspiration from 3D generation techniques, we utilize a multi-view diffusion model to initialize multi-view images anchoring on the input video frames, where the video can be either real-world captured or generated by a video diffusion model. To ensure the temporal consistency of the multi-view sequence initialization, we introduce a simple yet effective fusion strategy to leverage the first frame as a temporal anchor in the self-attention computation. With the almost consistent multi-view sequences, we then apply the score distillation sampling to optimize the 4D Gaussian point cloud. The 4D Gaussian spatting is specially crafted for the generation task, where an adaptive densification strategy is proposed to mitigate the unstable Gaussian gradient for robust optimization. Notably, the proposed pipeline does not require any pre-training or fine-tuning of diffusion networks, offering a more accessible and practical solution for the 4D generation task. Extensive experiments demonstrate that our method outperforms prior 4D generation works in rendering quality, spatial-temporal consistency, and generation robustness, setting a new state-of-the-art for 4D generation from diverse inputs, including text, image, and video.

近期，预训练的扩散模型和3D生成技术的进步激发了对4D内容创作的兴趣。然而，实现具有空间-时间一致性的高保真4D生成仍然是一个挑战。在这项工作中，我们提出了STAG4D，一个新颖的框架，结合了预训练的扩散模型和动态3D高斯喷溅技术，用于高保真4D生成。借鉴3D生成技术的灵感，我们利用多视图扩散模型来初始化固定在输入视频帧上的多视图图像，其中视频可以是现实世界捕获的，也可以是通过视频扩散模型生成的。为了确保多视图序列初始化的时间一致性，我们引入了一个简单而有效的融合策略，利用第一帧作为自注意力计算中的时间锚。通过几乎一致的多视图序列，我们随后应用得分蒸馏采样来优化4D高斯点云。4D高斯喷溅特别为生成任务设计，其中提出了一种适应性增密策略，以缓解不稳定的高斯梯度，实现稳健的优化。值得注意的是，所提出的流程不需要任何预训练或微调扩散网络，为4D生成任务提供了一个更加可行和实用的解决方案。广泛的实验表明，我们的方法在渲染质量、空间-时间一致性和生成鲁棒性方面超越了以往的4D生成工作，为从多样化输入（包括文本、图像和视频）生成4D内容设定了新的行业标准。


---

## [36] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting

### Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis results while advancing real-time rendering performance. However, it relies heavily on the quality of the initial point cloud, resulting in blurring and needle-like artifacts in areas with insufficient initializing points. This is mainly attributed to the point cloud growth condition in 3DGS that only considers the average gradient magnitude of points from observable views, thereby failing to grow for large Gaussians that are observable for many viewpoints while many of them are only covered in the boundaries. To this end, we propose a novel method, named Pixel-GS, to take into account the number of pixels covered by the Gaussian in each view during the computation of the growth condition. We regard the covered pixel numbers as the weights to dynamically average the gradients from different views, such that the growth of large Gaussians can be prompted. As a result, points within the areas with insufficient initializing points can be grown more effectively, leading to a more accurate and detailed reconstruction. In addition, we propose a simple yet effective strategy to scale the gradient field according to the distance to the camera, to suppress the growth of floaters near the camera. Extensive experiments both qualitatively and quantitatively demonstrate that our method achieves state-of-the-art rendering quality while maintaining real-time rendering speed, on the challenging Mip-NeRF 360 and Tanks & Temples datasets.

3D高斯喷溅（3DGS）在推进实时渲染性能的同时，展示了令人印象深刻的新视角合成结果。然而，它严重依赖于初始点云的质量，导致在初始化点不足的区域出现模糊和针状伪影。这主要归因于3DGS中的点云生长条件只考虑了来自可观察视图的点的平均梯度大小，因此对于许多视点可观察但许多仅在边界覆盖的大高斯，它未能进行生长。为此，我们提出了一种新的方法，名为Pixel-GS，考虑在计算生长条件时，每个视图中高斯覆盖的像素数。我们将覆盖的像素数视为权重，以动态平均不同视图的梯度，使得可以促进大高斯的生长。结果，可以更有效地增长初始化点不足区域内的点，导致更准确和详细的重建。此外，我们提出了一种简单而有效的策略，根据到相机的距离缩放梯度场，以抑制靠近相机的浮点生长。大量的实验，无论是定性的还是定量的，都证明了我们的方法在挑战性的Mip-NeRF 360和Tanks & Temples数据集上，实现了最先进的渲染质量，同时保持了实时渲染速度。


---

## [37] Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections

### Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections

Novel view synthesis from unconstrained in-the-wild images remains a meaningful but challenging task. The photometric variation and transient occluders in those unconstrained images make it difficult to reconstruct the original scene accurately. Previous approaches tackle the problem by introducing a global appearance feature in Neural Radiance Fields (NeRF). However, in the real world, the unique appearance of each tiny point in a scene is determined by its independent intrinsic material attributes and the varying environmental impacts it receives. Inspired by this fact, we propose Gaussian in the wild (GS-W), a method that uses 3D Gaussian points to reconstruct the scene and introduces separated intrinsic and dynamic appearance feature for each point, capturing the unchanged scene appearance along with dynamic variation like illumination and weather. Additionally, an adaptive sampling strategy is presented to allow each Gaussian point to focus on the local and detailed information more effectively. We also reduce the impact of transient occluders using a 2D visibility map. More experiments have demonstrated better reconstruction quality and details of GS-W compared to previous methods, with a 1000× increase in rendering speed.

从不受约束的野外图像中合成新视角仍是一个有意义但充满挑战的任务。这些不受约束图像中的光度变化和瞬时遮挡物使得准确重建原始场景变得困难。以往的方法通过在神经辐射场（NeRF）中引入全局外观特征来解决这个问题。然而，在现实世界中，场景中每个微小点的独特外观是由其独立的内在材料属性和它接收的不同环境影响决定的。受此启发，我们提出了一种方法，名为野外中的高斯（GS-W），使用3D高斯点来重建场景，并为每个点引入分离的内在和动态外观特征，捕捉不变的场景外观以及光照和天气等动态变化。此外，我们提出了一种自适应采样策略，允许每个高斯点更有效地关注局部和详细信息。我们还使用2D可见性图减少了瞬时遮挡物的影响。更多实验已经证明，与以往方法相比，GS-W在重建质量和细节方面表现更好，渲染速度提高了1000倍。


---

## [38] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field

### CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field

Recently neural radiance fields (NeRF) have been widely exploited as 3D representations for dense simultaneous localization and mapping (SLAM). Despite their notable successes in surface modeling and novel view synthesis, existing NeRF-based methods are hindered by their computationally intensive and time-consuming volume rendering pipeline. This paper presents an efficient dense RGB-D SLAM system, i.e., CG-SLAM, based on a novel uncertainty-aware 3D Gaussian field with high consistency and geometric stability. Through an in-depth analysis of Gaussian Splatting, we propose several techniques to construct a consistent and stable 3D Gaussian field suitable for tracking and mapping. Additionally, a novel depth uncertainty model is proposed to ensure the selection of valuable Gaussian primitives during optimization, thereby improving tracking efficiency and accuracy. Experiments on various datasets demonstrate that CG-SLAM achieves superior tracking and mapping performance with a notable tracking speed of up to 15 Hz.

最近，神经辐射场（NeRF）作为3D表示，已被广泛用于密集的同时定位与地图构建（SLAM）。尽管在表面建模和新视角合成方面取得了显著成功，但现有基于NeRF的方法受到其计算密集和耗时的体积渲染流程的阻碍。本文提出了一个高效的密集RGB-D SLAM系统，即CG-SLAM，基于一个新颖的、具有高一致性和几何稳定性的不确定性感知3D高斯场。通过对高斯喷溅的深入分析，我们提出了几种技术，以构建一个适用于跟踪和映射的一致且稳定的3D高斯场。此外，我们提出了一个新颖的深度不确定性模型，以确保在优化过程中选择有价值的高斯原语，从而提高跟踪效率和准确性。在各种数据集上的实验表明，CG-SLAM实现了卓越的跟踪和映射性能，具有高达15 Hz的显著跟踪速度。


---

## [39] latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction

### latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction

We present latentSplat, a method to predict semantic Gaussians in a 3D latent space that can be splatted and decoded by a light-weight generative 2D architecture. Existing methods for generalizable 3D reconstruction either do not enable fast inference of high resolution novel views due to slow volume rendering, or are limited to interpolation of close input views, even in simpler settings with a single central object, where 360-degree generalization is possible. In this work, we combine a regression-based approach with a generative model, moving towards both of these capabilities within the same method, trained purely on readily available real video data. The core of our method are variational 3D Gaussians, a representation that efficiently encodes varying uncertainty within a latent space consisting of 3D feature Gaussians. From these Gaussians, specific instances can be sampled and rendered via efficient Gaussian splatting and a fast, generative decoder network. We show that latentSplat outperforms previous works in reconstruction quality and generalization, while being fast and scalable to high-resolution data.

我们提出了latentSplat，一种在3D潜空间中预测语义高斯的方法，这些高斯可以被轻量级生成性2D架构喷溅并解码。现有的通用3D重建方法要么由于体积渲染速度慢而无法快速推断高分辨率新视图，要么限于对接近输入视图的插值，即使在具有单一中心对象的更简单设置中，其中360度概括是可能的。在这项工作中，我们结合了基于回归的方法和生成模型，向在同一方法内同时拥有这两种能力迈进，该方法完全基于现成的真实视频数据进行训练。我们方法的核心是变分3D高斯，这是一种有效编码潜空间中不同不确定性的表示，该潜空间由3D特征高斯组成。从这些高斯中，可以采样特定实例并通过高效的高斯喷溅和快速的生成解码器网络渲染。我们展示了latentSplat在重建质量和概括性方面超越了之前的工作，同时在处理高分辨率数据方面快速且可扩展。


---

## [40] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians

### CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians

The field of 3D reconstruction from images has rapidly evolved in the past few years, first with the introduction of Neural Radiance Field (NeRF) and more recently with 3D Gaussian Splatting (3DGS). The latter provides a significant edge over NeRF in terms of the training and inference speed, as well as the reconstruction quality. Although 3DGS works well for dense input images, the unstructured point-cloud like representation quickly overfits to the more challenging setup of extremely sparse input images (e.g., 3 images), creating a representation that appears as a jumble of needles from novel views. To address this issue, we propose regularized optimization and depth-based initialization. Our key idea is to introduce a structured Gaussian representation that can be controlled in 2D image space. We then constraint the Gaussians, in particular their position, and prevent them from moving independently during optimization. Specifically, we introduce single and multiview constraints through an implicit convolutional decoder and a total variation loss, respectively. With the coherency introduced to the Gaussians, we further constrain the optimization through a flow-based loss function. To support our regularized optimization, we propose an approach to initialize the Gaussians using monocular depth estimates at each input view. We demonstrate significant improvements compared to the state-of-the-art sparse-view NeRF-based approaches on a variety of scenes.

近几年来，从图像到3D重建的领域迅速发展，首先是神经辐射场（NeRF）的引入，最近则是3D高斯喷溅（3DGS）。后者在训练和推理速度以及重建质量方面，相较于NeRF有显著的优势。尽管3DGS在密集输入图像中表现良好，但在极其稀疏输入图像（例如，3张图像）的更具挑战性的设置中，类似于无结构点云的表示很快就会过度拟合，从新的视角看上去像是一团乱麻。为了解决这个问题，我们提出了正则化优化和基于深度的初始化。我们的关键思想是引入一个可以在2D图像空间中控制的结构化高斯表示。然后，我们约束高斯，特别是它们的位置，并防止它们在优化过程中独立移动。具体来说，我们通过一个隐式卷积解码器和总变分损失分别引入单视图和多视图约束。通过对高斯引入连贯性，我们进一步通过基于流的损失函数约束优化。为了支持我们的正则化优化，我们提出了一种使用每个输入视图处的单目深度估计来初始化高斯的方法。我们在多种场景上与最新的稀疏视图基于NeRF的方法相比，展示了显著的改进。


---

## [41] CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians

### CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians

The advancement of real-time 3D scene reconstruction and novel view synthesis has been significantly propelled by 3D Gaussian Splatting (3DGS). However, effectively training large-scale 3DGS and rendering it in real-time across various scales remains challenging. This paper introduces CityGaussian (CityGS), which employs a novel divide-and-conquer training approach and Level-of-Detail (LoD) strategy for efficient large-scale 3DGS training and rendering. Specifically, the global scene prior and adaptive training data selection enables efficient training and seamless fusion. Based on fused Gaussian primitives, we generate different detail levels through compression, and realize fast rendering across various scales through the proposed block-wise detail levels selection and aggregation strategy. Extensive experimental results on large-scale scenes demonstrate that our approach attains state-of-theart rendering quality, enabling consistent real-time rendering of largescale scenes across vastly different scales.

三维高斯喷溅（3DGS）的进步显著推动了实时三维场景重建和新颖视角合成的发展。然而，有效地训练大规模3DGS并在各种规模上实时渲染仍然具有挑战性。本文介绍了CityGaussian（CityGS），它采用了一种新颖的分而治之训练方法和细节级别（LoD）策略，以高效训练和渲染大规模3DGS。具体来说，全局场景先验和自适应训练数据选择使得训练高效且能无缝融合。基于融合的高斯原始体，我们通过压缩生成不同的细节级别，并通过提出的块状细节级别选择和聚合策略，实现在各种规模上的快速渲染。广泛的实验结果在大规模场景上展示了我们的方法达到了最先进的渲染质量，使得能够在极其不同的规模上实现大规模场景的一致实时渲染。


---

## [42] Feature Splatting: Language-Driven Physics-Based Scene Synthesis and Editing

### Feature Splatting: Language-Driven Physics-Based Scene Synthesis and Editing

Scene representations using 3D Gaussian primitives have produced excellent results in modeling the appearance of static and dynamic 3D scenes. Many graphics applications, however, demand the ability to manipulate both the appearance and the physical properties of objects. We introduce Feature Splatting, an approach that unifies physics-based dynamic scene synthesis with rich semantics from vision language foundation models that are grounded by natural language. Our first contribution is a way to distill high-quality, object-centric vision-language features into 3D Gaussians, that enables semi-automatic scene decomposition using text queries. Our second contribution is a way to synthesize physics-based dynamics from an otherwise static scene using a particle-based simulator, in which material properties are assigned automatically via text queries. We ablate key techniques used in this pipeline, to illustrate the challenge and opportunities in using feature-carrying 3D Gaussians as a unified format for appearance, geometry, material properties and semantics grounded on natural language.

使用三维高斯原始体的场景表示在建模静态和动态三维场景的外观方面取得了优异的成果。然而，许多图形应用程序要求能够操纵对象的外观和物理属性。我们介绍了特征喷溅（Feature Splatting），这是一种将基于物理的动态场景合成与基于自然语言的视觉语言基础模型中的丰富语义统一起来的方法。我们的第一个贡献是一种方法，能够将高质量的、以对象为中心的视觉-语言特征提炼到三维高斯中，这使得使用文本查询进行半自动场景分解成为可能。我们的第二个贡献是一种方法，能够使用基于粒子的模拟器从本来静态的场景中合成基于物理的动力学，其中材料属性通过文本查询自动分配。我们对在此流程中使用的关键技术进行了剖析，以说明使用携带特征的三维高斯作为外观、几何、材料属性和基于自然语言的语义的统一格式所面临的挑战和机遇。


---

## [43] Surface Reconstruction from Gaussian Splatting via Novel Stereo Views

### Surface Reconstruction from Gaussian Splatting via Novel Stereo Views

The Gaussian splatting for radiance field rendering method has recently emerged as an efficient approach for accurate scene representation. It optimizes the location, size, color, and shape of a cloud of 3D Gaussian elements to visually match, after projection, or splatting, a set of given images taken from various viewing directions. And yet, despite the proximity of Gaussian elements to the shape boundaries, direct surface reconstruction of objects in the scene is a challenge.
We propose a novel approach for surface reconstruction from Gaussian splatting models. Rather than relying on the Gaussian elements' locations as a prior for surface reconstruction, we leverage the superior novel-view synthesis capabilities of 3DGS. To that end, we use the Gaussian splatting model to render pairs of stereo-calibrated novel views from which we extract depth profiles using a stereo matching method. We then combine the extracted RGB-D images into a geometrically consistent surface. The resulting reconstruction is more accurate and shows finer details when compared to other methods for surface reconstruction from Gaussian splatting models, while requiring significantly less compute time compared to other surface reconstruction methods.
We performed extensive testing of the proposed method on in-the-wild scenes, taken by a smartphone, showcasing its superior reconstruction abilities. Additionally, we tested the proposed method on the Tanks and Temples benchmark, and it has surpassed the current leading method for surface reconstruction from Gaussian splatting models.

高斯喷溅用于辐射场渲染方法最近已经作为一种高效的准确场景表示方法而出现。它优化了一团三维高斯元素的位置、大小、颜色和形状，以便在投影或喷溅后，从各个观察方向拍摄的一组给定图像视觉上匹配。然而，尽管高斯元素接近形状边界，直接重建场景中对象的表面仍是一项挑战。
我们提出了一种从高斯喷溅模型重建表面的新方法。我们不是依赖高斯元素的位置作为表面重建的先验，而是利用3DGS卓越的新视角合成能力。为此，我们使用高斯喷溅模型渲染一对经过立体校准的新视角，从中我们使用立体匹配方法提取深度轮廓。然后，我们将提取的RGB-D图像合并成一个几何上一致的表面。与其他从高斯喷溅模型进行表面重建的方法相比，结果重建更加准确，展示了更细致的细节，同时与其他表面重建方法相比，所需的计算时间显著减少。
我们对提出的方法进行了广泛的测试，这些测试在野外场景中进行，由智能手机拍摄，展示了其卓越的重建能力。此外，我们还在Tanks and Temples基准测试上测试了提出的方法，它已经超过了当前领先的从高斯喷溅模型进行表面重建的方法。


---

## [44] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting

### Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting

As 3D Gaussian Splatting (3DGS) provides fast and high-quality novel view synthesis, it is a natural extension to deform a canonical 3DGS to multiple frames. However, previous works fail to accurately reconstruct dynamic scenes, especially 1) static parts moving along nearby dynamic parts, and 2) some dynamic areas are blurry. We attribute the failure to the wrong design of the deformation field, which is built as a coordinate-based function. This approach is problematic because 3DGS is a mixture of multiple fields centered at the Gaussians, not just a single coordinate-based framework. To resolve this problem, we define the deformation as a function of per-Gaussian embeddings and temporal embeddings. Moreover, we decompose deformations as coarse and fine deformations to model slow and fast movements, respectively. Also, we introduce an efficient training strategy for faster convergence and higher quality.

由于三维高斯喷溅（3DGS）提供了快速且高质量的新视角合成，将标准的3DGS变形应用于多帧是一个自然的扩展。然而，以往的工作未能准确重建动态场景，特别是1) 静态部分沿着附近的动态部分移动，以及2) 一些动态区域模糊不清。我们将这一失败归因于变形场设计错误，该设计构建为基于坐标的函数。这种方法存在问题，因为3DGS是多个以高斯为中心的场的混合，而不仅仅是一个基于单一坐标的框架。为了解决这个问题，我们将变形定义为每个高斯嵌入和时间嵌入的函数。此外，我们将变形分解为粗变形和细变形，分别模拟慢速和快速运动。同时，我们引入了一种高效的训练策略，以实现更快的收敛和更高的质量。


---

## [45] PhysAvatar: Learning the Physics of Dressed 3D Avatars from Visual Observations

<!-- ❌ 无法加载摘要: https://raw.githubusercontent.com/Awesome3DGS/3D-Gaussian-Splatting-Papers/main/abs/2404.04421.md -->

---
Abstract. Modeling and rendering photorealistic avatars is of crucial
importance in many applications. Existing methods that build a 3D
avatar from visual observations, however, struggle to reconstruct clothed
humans. We introduce PhysAvatar, a novel framework that combines inverse rendering with inverse physics to automatically estimate the shape
and appearance of a human from multi-view video data along with the
physical parameters of the fabric of their clothes. For this purpose, we
adopt a mesh-aligned 4D Gaussian technique for spatio-temporal mesh
tracking as well as a physically based inverse renderer to estimate the intrinsic material properties. PhysAvatar integrates a physics simulator
to estimate the physical parameters of the garments using gradientbased optimization in a principled manner. These novel capabilities enable PhysAvatar to create high-quality novel-view renderings of avatars
dressed in loose-fitting clothes under motions and lighting conditions
not seen in the training data. This marks a significant advancement
towards modeling photorealistic digital humans using physically based
inverse rendering with physics in the loop. Our project website is at:
https://qingqing-zhao.github.io/PhysAvatar.

在许多应用中，对光真实感虚拟化身进行建模与渲染至关重要。然而，现有基于视觉观测构建三维化身的方法难以有效重建着装人体。我们提出PhysAvatar——一种将逆向渲染与逆向物理相结合的新型框架，能够从多视角视频数据中自动估算人体形状与外观，并同步估算衣物面料的物理参数。为实现该目标，我们采用网格对齐的四维高斯技术进行时空网格追踪，并基于物理原理的逆向渲染器来估算本征材质属性。PhysAvatar通过集成物理模拟器，采用基于梯度的优化方法以原理化方式估算衣物的物理参数。这些创新功能使PhysAvatar能够对穿着宽松衣物的虚拟化身进行高质量新视角渲染，即使在训练数据未出现的动作与光照条件下仍能保持真实感。这标志着通过引入物理循环的基于物理的逆向渲染技术，在光真实感数字人建模领域取得了重要进展。项目网站详见：https://qingqing-zhao.github.io/PhysAvatar。
## [46] Dual-Camera Smooth Zoom on Mobile Phones

### Dual-Camera Smooth Zoom on Mobile Phones

When zooming between dual cameras on a mobile, noticeable jumps in geometric content and image color occur in the preview, inevitably affecting the user's zoom experience. In this work, we introduce a new task, ie, dual-camera smooth zoom (DCSZ) to achieve a smooth zoom preview. The frame interpolation (FI) technique is a potential solution but struggles with ground-truth collection. To address the issue, we suggest a data factory solution where continuous virtual cameras are assembled to generate DCSZ data by rendering reconstructed 3D models of the scene. In particular, we propose a novel dual-camera smooth zoom Gaussian Splatting (ZoomGS), where a camera-specific encoding is introduced to construct a specific 3D model for each virtual camera. With the proposed data factory, we construct a synthetic dataset for DCSZ, and we utilize it to fine-tune FI models. In addition, we collect real-world dual-zoom images without ground-truth for evaluation. Extensive experiments are conducted with multiple FI methods. The results show that the fine-tuned FI models achieve a significant performance improvement over the original ones on DCSZ task. The datasets, codes, and pre-trained models will be publicly available.

在移动设备上双摄像头之间缩放时，几何内容和图像颜色在预览中会发生明显跳变，不可避免地影响用户的缩放体验。在这项工作中，我们引入了一个新任务，即双摄像头平滑缩放（DCSZ），以实现平滑的缩放预览。帧插值（FI）技术是一个潜在的解决方案，但在收集真实数据方面遇到困难。为了解决这个问题，我们建议一个数据工厂解决方案，其中连续的虚拟摄像头被组装起来，通过渲染场景的重建3D模型来生成DCSZ数据。特别地，我们提出了一个新颖的双摄像头平滑缩放高斯喷涂（ZoomGS），引入了一个特定于摄像头的编码，以构建每个虚拟摄像头的特定3D模型。借助提出的数据工厂，我们构建了一个用于DCSZ的合成数据集，并利用它来微调FI模型。此外，我们收集了没有真实数据的现实世界双重缩放图像进行评估。我们使用多种FI方法进行了广泛的实验。结果显示，微调后的FI模型在DCSZ任务上比原始模型取得了显著的性能提升。数据集、代码和预训练模型将公开可用。


---

## [47] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting

### DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting

The increasing demand for virtual reality applications has highlighted the significance of crafting immersive 3D assets. We present a text-to-3D 360∘ scene generation pipeline that facilitates the creation of comprehensive 360∘ scenes for in-the-wild environments in a matter of minutes. Our approach utilizes the generative power of a 2D diffusion model and prompt self-refinement to create a high-quality and globally coherent panoramic image. This image acts as a preliminary "flat" (2D) scene representation. Subsequently, it is lifted into 3D Gaussians, employing splatting techniques to enable real-time exploration. To produce consistent 3D geometry, our pipeline constructs a spatially coherent structure by aligning the 2D monocular depth into a globally optimized point cloud. This point cloud serves as the initial state for the centroids of 3D Gaussians. In order to address invisible issues inherent in single-view inputs, we impose semantic and geometric constraints on both synthesized and input camera views as regularizations. These guide the optimization of Gaussians, aiding in the reconstruction of unseen regions. In summary, our method offers a globally consistent 3D scene within a 360∘ perspective, providing an enhanced immersive experience over existing techniques.

随着虚拟现实应用需求的增加，制作沉浸式3D资产的重要性日益凸显。我们提出了一种文本到3D全景360度场景生成流程，该流程能在几分钟内为野外环境创建全面的360度场景。我们的方法利用了2D扩散模型的生成能力和自我完善提示来创建高质量且全局一致的全景图像。这个图像作为初步的“平面”（2D）场景表示。随后，它被提升到3D高斯体，使用喷涂技术以实现实时探索。为了产生一致的3D几何结构，我们的流程通过将2D单眼深度对齐到全局优化的点云中来构建空间连贯的结构。这个点云作为3D高斯体的质心的初始状态。为了解决单视角输入固有的不可见问题，我们在合成和输入相机视图上施加语义和几何约束作为规范。这些约束指导高斯体的优化，帮助重建未见区域。总之，我们的方法提供了一个全局一致的3D场景，具有360度的视角，相较现有技术提供了更优的沉浸式体验。


---

## [48] GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal

### GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal

This paper tackles the intricate challenge of object removal to update the radiance field using the 3D Gaussian Splatting. The main challenges of this task lie in the preservation of geometric consistency and the maintenance of texture coherence in the presence of the substantial discrete nature of Gaussian primitives. We introduce a robust framework specifically designed to overcome these obstacles. The key insight of our approach is the enhancement of information exchange among visible and invisible areas, facilitating content restoration in terms of both geometry and texture. Our methodology begins with optimizing the positioning of Gaussian primitives to improve geometric consistency across both removed and visible areas, guided by an online registration process informed by monocular depth estimation. Following this, we employ a novel feature propagation mechanism to bolster texture coherence, leveraging a cross-attention design that bridges sampling Gaussians from both uncertain and certain areas. This innovative approach significantly refines the texture coherence within the final radiance field. Extensive experiments validate that our method not only elevates the quality of novel view synthesis for scenes undergoing object removal but also showcases notable efficiency gains in training and rendering speeds.

本文解决了使用三维高斯涂抹更新辐射场中物体移除的复杂挑战。这项任务的主要难点在于保持几何一致性和在高斯原始图形显著的离散特性存在的情况下维护纹理一致性。我们引入了一个专门设计的强大框架来克服这些障碍。我们方法的核心见解是增强可见区域和不可见区域之间的信息交换，从而在几何和纹理两个方面促进内容恢复。我们的方法首先通过优化高斯原始图形的定位来提高被移除区域和可见区域的几何一致性，这一过程由单目深度估计信息的在线注册过程指导。接下来，我们采用一种新颖的特征传播机制来增强纹理一致性，利用跨注意力设计桥接不确定和确定区域的采样高斯。这种创新方法显著提高了最终辐射场内的纹理一致性。广泛的实验验证了我们的方法不仅提升了经历物体移除的场景新视角合成的质量，而且还在训练和渲染速度上展示了显著的效率提升。


---

## [49] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting

### TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting

Radiance fields have demonstrated impressive performance in synthesizing lifelike 3D talking heads. However, due to the difficulty in fitting steep appearance changes, the prevailing paradigm that presents facial motions by directly modifying point appearance may lead to distortions in dynamic regions. To tackle this challenge, we introduce TalkingGaussian, a deformation-based radiance fields framework for high-fidelity talking head synthesis. Leveraging the point-based Gaussian Splatting, facial motions can be represented in our method by applying smooth and continuous deformations to persistent Gaussian primitives, without requiring to learn the difficult appearance change like previous methods. Due to this simplification, precise facial motions can be synthesized while keeping a highly intact facial feature. Under such a deformation paradigm, we further identify a face-mouth motion inconsistency that would affect the learning of detailed speaking motions. To address this conflict, we decompose the model into two branches separately for the face and inside mouth areas, therefore simplifying the learning tasks to help reconstruct more accurate motion and structure of the mouth region. Extensive experiments demonstrate that our method renders high-quality lip-synchronized talking head videos, with better facial fidelity and higher efficiency compared with previous methods.

辐射场在合成逼真的3D说话头部方面表现出色。然而，由于适应剧烈的外观变化较为困难，当前通过直接修改点的外观来呈现面部动作的范式可能导致动态区域的扭曲。为了解决这一挑战，我们引入了TalkingGaussian，这是一个基于形变的辐射场框架，用于高保真的说话头部合成。通过利用基于点的高斯溅射，我们的方法可以通过对持久的高斯原始体应用平滑且连续的形变来表示面部动作，无需学习像以前的方法那样困难的外观变化。由于这种简化，可以合成精确的面部动作，同时保持高度完整的面部特征。在这种形变范式下，我们进一步发现了一个面部-口部动作不一致性，这会影响详细说话动作的学习。为了解决这一冲突，我们将模型分解为面部和口内区域的两个独立分支，从而简化学习任务，帮助重建更精确的口部动作和结构。广泛的实验表明，我们的方法渲染出的高质量唇同步说话头部视频，在面部保真度和效率上比以前的方法有更好的表现。


---

## [50] DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing

### DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing

We consider the problem of editing 3D objects and scenes based on open-ended language instructions. The established paradigm to solve this problem is to use a 2D image generator or editor to guide the 3D editing process. However, this is often slow as it requires do update a computationally expensive 3D representations such as a neural radiance field, and to do so by using contradictory guidance from a 2D model which is inherently not multi-view consistent. We thus introduce the Direct Gaussian Editor (DGE), a method that addresses these issues in two ways. First, we modify a given high-quality image editor like InstructPix2Pix to be multi-view consistent. We do so by utilizing a training-free approach which integrates cues from the underlying 3D geometry of the scene. Second, given a multi-view consistent edited sequence of images of the object, we directly and efficiently optimize the 3D object representation, which is based on 3D Gaussian Splatting. Because it does not require to apply edits incrementally and iteratively, DGE is significantly more efficient than existing approaches, and comes with other perks such as allowing selective editing of parts of the scene.

我们考虑了基于开放式语言指令编辑3D对象和场景的问题。解决这个问题的传统范式是使用2D图像生成器或编辑器来指导3D编辑过程。然而，这通常较慢，因为它需要更新如神经辐射场这样的计算成本高昂的3D表示，并且还需使用本质上不具备多视图一致性的2D模型提供的矛盾指导。因此，我们引入了直接高斯编辑器（DGE），这是一种以两种方式解决这些问题的方法。首先，我们修改了像InstructPix2Pix这样的高质量图像编辑器，使其具有多视图一致性。我们通过使用一种无需训练的方法实现，该方法整合了场景底层3D几何的线索。其次，给定一个多视图一致的编辑过的对象图像序列，我们直接且高效地优化基于3D高斯喷溅的3D对象表示。由于DGE不需要逐步和迭代地应用编辑，它比现有方法更加高效，并且还具有其他优点，如允许选择性编辑场景的部分。


---

## [51] SAGS: Structure-Aware 3D Gaussian Splatting

### SAGS: Structure-Aware 3D Gaussian Splatting

Following the advent of NeRFs, 3D Gaussian Splatting (3D-GS) has paved the way to real-time neural rendering overcoming the computational burden of volumetric methods. Following the pioneering work of 3D-GS, several methods have attempted to achieve compressible and high-fidelity performance alternatives. However, by employing a geometry-agnostic optimization scheme, these methods neglect the inherent 3D structure of the scene, thereby restricting the expressivity and the quality of the representation, resulting in various floating points and artifacts. In this work, we propose a structure-aware Gaussian Splatting method (SAGS) that implicitly encodes the geometry of the scene, which reflects to state-of-the-art rendering performance and reduced storage requirements on benchmark novel-view synthesis datasets. SAGS is founded on a local-global graph representation that facilitates the learning of complex scenes and enforces meaningful point displacements that preserve the scene's geometry. Additionally, we introduce a lightweight version of SAGS, using a simple yet effective mid-point interpolation scheme, which showcases a compact representation of the scene with up to 24× size reduction without the reliance on any compression strategies. Extensive experiments across multiple benchmark datasets demonstrate the superiority of SAGS compared to state-of-the-art 3D-GS methods under both rendering quality and model size. Besides, we demonstrate that our structure-aware method can effectively mitigate floating artifacts and irregular distortions of previous methods while obtaining precise depth maps.

自从NeRFs的出现之后，3D高斯喷溅（3D-GS）已经开辟了实时神经渲染的道路，克服了体积方法的计算负担。继3D-GS的开创性工作之后，几种方法试图实现可压缩和高保真度的性能替代方案。然而，这些方法采用了几何无关的优化方案，忽视了场景的固有3D结构，从而限制了表示的表现力和质量，导致了各种浮点和伪影。在这项工作中，我们提出了一种结构感知的高斯喷溅方法（SAGS），该方法隐式编码了场景的几何结构，反映出最先进的渲染性能和在基准新视角合成数据集上减少的存储需求。SAGS基于一个局部-全局图表示，便于学习复杂场景并强制执行有意义的点位移，以保持场景的几何结构。此外，我们引入了SAGS的轻量级版本，使用一种简单而有效的中点插值方案，展示了场景的紧凑表示，无需依赖任何压缩策略，可实现高达24倍的尺寸减少。在多个基准数据集上进行的广泛实验表明，与最先进的3D-GS方法相比，SAGS在渲染质量和模型大小方面具有优越性。此外，我们证明了我们的结构感知方法可以有效地减轻以前方法的浮动伪影和不规则扭曲，同时获得精确的深度图。


---

## [52] MirrorGaussian: Reflecting 3D Gaussians for Reconstructing Mirror Reflections

### MirrorGaussian: Reflecting 3D Gaussians for Reconstructing Mirror Reflections

3D Gaussian Splatting showcases notable advancements in photo-realistic and real-time novel view synthesis. However, it faces challenges in modeling mirror reflections, which exhibit substantial appearance variations from different viewpoints. To tackle this problem, we present MirrorGaussian, the first method for mirror scene reconstruction with real-time rendering based on 3D Gaussian Splatting. The key insight is grounded on the mirror symmetry between the real-world space and the virtual mirror space. We introduce an intuitive dual-rendering strategy that enables differentiable rasterization of both the real-world 3D Gaussians and the mirrored counterpart obtained by reflecting the former about the mirror plane. All 3D Gaussians are jointly optimized with the mirror plane in an end-to-end framework. MirrorGaussian achieves high-quality and real-time rendering in scenes with mirrors, empowering scene editing like adding new mirrors and objects. Comprehensive experiments on multiple datasets demonstrate that our approach significantly outperforms existing methods, achieving state-of-the-art results.

3D高斯喷溅在光度真实和实时新视角合成方面展示了显著的进步。然而，它在模拟镜面反射时面临挑战，这些反射从不同视点显示出显著的外观变化。为了解决这个问题，我们提出了MirrorGaussian，这是第一个基于3D高斯喷溅进行镜面场景重建并实现实时渲染的方法。关键见解基于真实世界空间与虚拟镜面空间之间的镜面对称性。我们引入了一种直观的双重渲染策略，使得真实世界的3D高斯和通过镜面反射得到的镜像部分都能进行可微栅格化。所有3D高斯与镜面一起在端到端框架中进行联合优化。MirrorGaussian在含镜子的场景中实现了高质量和实时渲染，增强了场景编辑能力，如添加新镜子和物体。在多个数据集上的全面实验表明，我们的方法显著超过现有方法，达到了最先进的结果。


---

## [53] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization

### CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization

Splatting (3DGS) creates a radiance field consisting of 3D Gaussians to represent a scene. With sparse training views, 3DGS easily suffers from overfitting, negatively impacting the reconstruction quality. This paper introduces a new co-regularization perspective for improving sparse-view 3DGS. When training two 3D Gaussian radiance fields with the same sparse views of a scene, we observe that the two radiance fields exhibit point disagreement and rendering disagreement that can unsupervisedly predict reconstruction quality, stemming from the sampling implementation in densification. We further quantify the point disagreement and rendering disagreement by evaluating the registration between Gaussians' point representations and calculating differences in their rendered pixels. The empirical study demonstrates the negative correlation between the two disagreements and accurate reconstruction, which allows us to identify inaccurate reconstruction without accessing ground-truth information. Based on the study, we propose CoR-GS, which identifies and suppresses inaccurate reconstruction based on the two disagreements: (1) Co-pruning considers Gaussians that exhibit high point disagreement in inaccurate positions and prunes them. (2) Pseudo-view co-regularization considers pixels that exhibit high rendering disagreement are inaccurately rendered and suppress the disagreement. Results on LLFF, Mip-NeRF360, DTU, and Blender demonstrate that CoR-GS effectively regularizes the scene geometry, reconstructs the compact representations, and achieves state-of-the-art novel view synthesis quality under sparse training views.

3D高斯喷溅（3DGS）创建一个由3D高斯组成的辐射场来表示场景。在稀疏训练视图的情况下，3DGS容易过拟合，这对重建质量产生负面影响。本文引入了一种新的共正则化视角，用于改善稀疏视图下的3DGS。当使用同一场景的相同稀疏视图训练两个3D高斯辐射场时，我们观察到两个辐射场表现出\textit{点不一致}和\textit{渲染不一致}，这两种不一致可以无监督地预测重建质量，源自在密集化中的采样实现。我们通过评估高斯点表示之间的配准以及计算它们渲染像素的差异来进一步量化点不一致和渲染不一致。实证研究表明两种不一致与精确重建之间的负相关性，这使我们能够在不访问真实信息的情况下识别不准确的重建。基于这项研究，我们提出了CoR-GS，该方法基于两种不一致来识别和抑制不准确的重建：（1）共修剪考虑显示高点不一致的高斯处于不准确的位置并将其修剪。（2）伪视图共正则化考虑显示高渲染不一致的像素是不准确渲染的，并抑制这种不一致。在LLFF、Mip-NeRF360、DTU和Blender数据集上的结果表明，CoR-GS有效地规范了场景几何结构，重建了紧凑的表示，并在稀疏训练视图下实现了最先进的新视角合成质量。


---

## [54] Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo

### Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo

We present MVSGaussian, a new generalizable 3D Gaussian representation approach derived from Multi-View Stereo (MVS) that can efficiently reconstruct unseen scenes. Specifically, 1) we leverage MVS to encode geometry-aware Gaussian representations and decode them into Gaussian parameters. 2) To further enhance performance, we propose a hybrid Gaussian rendering that integrates an efficient volume rendering design for novel view synthesis. 3) To support fast fine-tuning for specific scenes, we introduce a multi-view geometric consistent aggregation strategy to effectively aggregate the point clouds generated by the generalizable model, serving as the initialization for per-scene optimization. Compared with previous generalizable NeRF-based methods, which typically require minutes of fine-tuning and seconds of rendering per image, MVSGaussian achieves real-time rendering with better synthesis quality for each scene. Compared with the vanilla 3D-GS, MVSGaussian achieves better view synthesis with less training computational cost. Extensive experiments on DTU, Real Forward-facing, NeRF Synthetic, and Tanks and Temples datasets validate that MVSGaussian attains state-of-the-art performance with convincing generalizability, real-time rendering speed, and fast per-scene optimization.

我们提出了MVSGaussian，这是一种新的从多视图立体（MVS）衍生的通用3D高斯表示方法，能够高效地重建未见过的场景。具体来说，1）我们利用MVS来编码具有几何意识的高斯表示，并将其解码为高斯参数。2）为了进一步提高性能，我们提出了一种混合高斯渲染技术，该技术整合了一种高效的体积渲染设计，用于新视角合成。3）为了支持特定场景的快速微调，我们引入了一种多视图几何一致性聚合策略，有效地聚合由通用模型生成的点云，作为每个场景优化的初始化。与之前需要几分钟微调时间和每幅图像几秒钟渲染时间的通用NeRF-based方法相比，MVSGaussian实现了每个场景更好的合成质量的实时渲染。与原始的3D-GS相比，MVSGaussian在更低的训练计算成本下实现了更好的视图合成。在DTU、Real Forward-facing、NeRF Synthetic以及Tanks and Temples数据集上的广泛实验验证了MVSGaussian具有卓越的性能、令人信服的通用性、实时渲染速度和快速的场景特定优化能力。


---

## [55] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction

### GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction

3D semantic occupancy prediction aims to obtain 3D fine-grained geometry and semantics of the surrounding scene and is an important task for the robustness of vision-centric autonomous driving. Most existing methods employ dense grids such as voxels as scene representations, which ignore the sparsity of occupancy and the diversity of object scales and thus lead to unbalanced allocation of resources. To address this, we propose an object-centric representation to describe 3D scenes with sparse 3D semantic Gaussians where each Gaussian represents a flexible region of interest and its semantic features. We aggregate information from images through the attention mechanism and iteratively refine the properties of 3D Gaussians including position, covariance, and semantics. We then propose an efficient Gaussian-to-voxel splatting method to generate 3D occupancy predictions, which only aggregates the neighboring Gaussians for a certain position. We conduct extensive experiments on the widely adopted nuScenes and KITTI-360 datasets. Experimental results demonstrate that GaussianFormer achieves comparable performance with state-of-the-art methods with only 17.8% - 24.8% of their memory consumption.

三维语义占用预测旨在获取周围场景的三维细粒度几何形状和语义，这是视觉中心自动驾驶系统稳定性的重要任务。大多数现有方法使用密集网格（如体素）作为场景表示，这忽略了占用的稀疏性和对象尺度的多样性，从而导致资源分配不均衡。为了解决这一问题，我们提出了一种以对象为中心的表示方法来描述三维场景，使用稀疏的三维语义高斯表示，每个高斯代表一个灵活的兴趣区域及其语义特征。我们通过注意力机制从图像中聚合信息，并迭代细化三维高斯的属性，包括位置、协方差和语义。接着，我们提出了一种高效的高斯到体素的喷溅方法来生成三维占用预测，该方法只聚合特定位置附近的高斯。我们在广泛采用的nuScenes和KITTI-360数据集上进行了广泛的实验。实验结果表明，GaussianFormer在只有17.8%至24.8%的内存消耗下，达到了与最先进方法相当的性能。


---

## [56] DGD: Dynamic 3D Gaussians Distillation

### DGD: Dynamic 3D Gaussians Distillation

We tackle the task of learning dynamic 3D semantic radiance fields given a single monocular video as input. Our learned semantic radiance field captures per-point semantics as well as color and geometric properties for a dynamic 3D scene, enabling the generation of novel views and their corresponding semantics. This enables the segmentation and tracking of a diverse set of 3D semantic entities, specified using a simple and intuitive interface that includes a user click or a text prompt. To this end, we present DGD, a unified 3D representation for both the appearance and semantics of a dynamic 3D scene, building upon the recently proposed dynamic 3D Gaussians representation. Our representation is optimized over time with both color and semantic information. Key to our method is the joint optimization of the appearance and semantic attributes, which jointly affect the geometric properties of the scene. We evaluate our approach in its ability to enable dense semantic 3D object tracking and demonstrate high-quality results that are fast to render, for a diverse set of scenes.

我们处理的任务是基于单个单目视频学习动态三维语义辐射场。我们学习到的语义辐射场能够捕捉每个点的语义以及动态三维场景的颜色和几何特性，从而生成新视角及其对应的语义。这使得可以使用简单直观的界面（如用户点击或文本提示）对各种三维语义实体进行分割和跟踪。为此，我们提出了DGD，这是一种用于动态三维场景外观和语义的统一三维表示，基于最近提出的动态三维高斯表示构建。我们的表示随时间进行优化，结合了颜色和语义信息。我们方法的关键是外观和语义属性的联合优化，这共同影响场景的几何特性。我们在密集语义三维对象跟踪能力方面对我们的方法进行了评估，并展示了在各种场景中快速渲染的高质量结果。


---

## [57] Topo4D: Topology-Preserving Gaussian Splatting for High-Fidelity 4D Head Capture

### Topo4D: Topology-Preserving Gaussian Splatting for High-Fidelity 4D Head Capture

4D head capture aims to generate dynamic topological meshes and corresponding texture maps from videos, which is widely utilized in movies and games for its ability to simulate facial muscle movements and recover dynamic textures in pore-squeezing. The industry often adopts the method involving multi-view stereo and non-rigid alignment. However, this approach is prone to errors and heavily reliant on time-consuming manual processing by artists. To simplify this process, we propose Topo4D, a novel framework for automatic geometry and texture generation, which optimizes densely aligned 4D heads and 8K texture maps directly from calibrated multi-view time-series images. Specifically, we first represent the time-series faces as a set of dynamic 3D Gaussians with fixed topology in which the Gaussian centers are bound to the mesh vertices. Afterward, we perform alternative geometry and texture optimization frame-by-frame for high-quality geometry and texture learning while maintaining temporal topology stability. Finally, we can extract dynamic facial meshes in regular wiring arrangement and high-fidelity textures with pore-level details from the learned Gaussians. Extensive experiments show that our method achieves superior results than the current SOTA face reconstruction methods both in the quality of meshes and textures.

4D 头部捕捉旨在从视频中生成动态拓扑网格和相应的纹理贴图，这在电影和游戏中得到了广泛使用，因为它能够模拟面部肌肉运动并恢复毛孔挤压中的动态纹理。该行业通常采用多视图立体和非刚性对齐方法。然而，这种方法容易出错，并且严重依赖于艺术家耗时的手动处理。为了简化这个过程，我们提出了 Topo4D，这是一个用于自动几何和纹理生成的新框架，它可以直接从校准的多视图时间序列图像中优化密集对齐的 4D 头部和 8K 纹理贴图。具体来说，我们首先将时间序列面部表示为一组具有固定拓扑的动态 3D 高斯分布，其中高斯中心绑定到网格顶点。之后，我们逐帧进行几何和纹理的交替优化，以实现高质量的几何和纹理学习，同时保持时间拓扑稳定性。最后，我们可以从学习到的高斯分布中提取具有规则布线排列和具有毛孔级细节的高保真纹理的动态面部网格。广泛的实验表明，我们的方法在网格和纹理的质量上均优于当前最先进的面部重建方法。


---

## [58] SuperGaussian: Repurposing Video Models for 3D Super Resolution

### SuperGaussian: Repurposing Video Models for 3D Super Resolution

We present a simple, modular, and generic method that upsamples coarse 3D models by adding geometric and appearance details. While generative 3D models now exist, they do not yet match the quality of their counterparts in image and video domains. We demonstrate that it is possible to directly repurpose existing (pretrained) video models for 3D super-resolution and thus sidestep the problem of the shortage of large repositories of high-quality 3D training models. We describe how to repurpose video upsampling models, which are not 3D consistent, and combine them with 3D consolidation to produce 3D-consistent results. As output, we produce high quality Gaussian Splat models, which are object centric and effective. Our method is category agnostic and can be easily incorporated into existing 3D workflows. We evaluate our proposed SuperGaussian on a variety of 3D inputs, which are diverse both in terms of complexity and representation (e.g., Gaussian Splats or NeRFs), and demonstrate that our simple method significantly improves the fidelity of the final 3D models.

我们提出了一种简单、模块化且通用的方法，该方法通过增加几何和外观细节来上采样粗糙的 3D 模型。尽管现在存在生成式 3D 模型，但它们的质量尚未达到图像和视频领域同类产品的质量。我们证明了可以直接将现有的（预训练的）视频模型用于 3D 超分辨率，从而绕过高质量 3D 训练模型库短缺的问题。我们描述了如何重新利用视频上采样模型，这些模型在 3D 上不一致，并将它们与 3D 整合结合起来，以产生 3D 一致的结果。作为输出，我们生成了高质量的高斯喷溅模型，这些模型以对象为中心且效果显著。我们的方法不受类别限制，可以轻松整合到现有的 3D 工作流程中。我们在多种 3D 输入上评估了我们提出的 SuperGaussian，这些输入在复杂性和表现形式（例如，高斯喷溅或 NeRFs）方面都具有多样性，并证明我们的简单方法显著提高了最终 3D 模型的保真度。



---

## [59] End-to-End Rate-Distortion Optimized 3D Gaussian Representation

### End-to-End Rate-Distortion Optimized 3D Gaussian Representation

3D Gaussian Splatting (3DGS) has become an emerging technique with remarkable potential in 3D representation and image rendering. However, the substantial storage overhead of 3DGS significantly impedes its practical applications. In this work, we formulate the compact 3D Gaussian learning as an end-to-end Rate-Distortion Optimization (RDO) problem and propose RDO-Gaussian that can achieve flexible and continuous rate control. RDO-Gaussian addresses two main issues that exist in current schemes: 1) Different from prior endeavors that minimize the rate under the fixed distortion, we introduce dynamic pruning and entropy-constrained vector quantization (ECVQ) that optimize the rate and distortion at the same time. 2) Previous works treat the colors of each Gaussian equally, while we model the colors of different regions and materials with learnable numbers of parameters. We verify our method on both real and synthetic scenes, showcasing that RDO-Gaussian greatly reduces the size of 3D Gaussian over 40x, and surpasses existing methods in rate-distortion performance.

3D 高斯涂抹（3DGS）已成为3D表征和图像渲染中一种具有显著潜力的新兴技术。然而，3DGS的巨大存储开销显著阻碍了其实际应用。在这项工作中，我们将紧凑的3D高斯学习表述为端到端的速率失真优化（RDO）问题，并提出了RDO-Gaussian，能够实现灵活和连续的速率控制。RDO-Gaussian解决了当前方案中存在的两个主要问题：1）与之前只是在固定失真下最小化速率的尝试不同，我们引入了动态剪枝和熵约束向量量化（ECVQ），同时优化速率和失真。2）以往的工作对每个高斯的颜色处理相同，而我们对不同区域和材料的颜色采用可学习的参数数量进行建模。我们在真实和合成场景中验证了我们的方法，显示出RDO-Gaussian在减少3D高斯大小方面超过40倍，并且在速率失真表现上超越了现有方法。


---

## [60] VEGS: View Extrapolation of Urban Scenes in 3D Gaussian Splatting using Learned Priors

### VEGS: View Extrapolation of Urban Scenes in 3D Gaussian Splatting using Learned Priors

Neural rendering-based urban scene reconstruction methods commonly rely on images collected from driving vehicles with cameras facing and moving forward. Although these methods can successfully synthesize from views similar to training camera trajectory, directing the novel view outside the training camera distribution does not guarantee on-par performance. In this paper, we tackle the Extrapolated View Synthesis (EVS) problem by evaluating the reconstructions on views such as looking left, right or downwards with respect to training camera distributions. To improve rendering quality for EVS, we initialize our model by constructing dense LiDAR map, and propose to leverage prior scene knowledge such as surface normal estimator and large-scale diffusion model. Qualitative and quantitative comparisons demonstrate the effectiveness of our methods on EVS. To the best of our knowledge, we are the first to address the EVS problem in urban scene reconstruction.

基于神经渲染的城市场景重建方法通常依赖于从驾驶车辆上采集的图像，摄像头面向前方移动。虽然这些方法能够成功地合成与训练相似视角的图像，但是指向训练摄像头分布之外的新视角，并不能保证同等水平的性能。在本文中，我们解决了“外推视角合成（EVS）”问题，通过评估在不同于训练摄像头分布的视角下的重建效果，例如向左、向右或向下查看。为了改善EVS的渲染质量，我们通过构建密集的激光雷达地图来初始化模型，并提出利用场景先验知识，如表面法线估计器和大规模扩散模型。定性和定量比较显示了我们方法在EVS上的有效性。据我们所知，我们是首个解决城市场景重建中EVS问题的研究工作。



---

## [61] GSD: View-Guided Gaussian Splatting Diffusion for 3D Reconstruction

### GSD: View-Guided Gaussian Splatting Diffusion for 3D Reconstruction

We present GSD, a diffusion model approach based on Gaussian Splatting (GS) representation for 3D object reconstruction from a single view. Prior works suffer from inconsistent 3D geometry or mediocre rendering quality due to improper representations. We take a step towards resolving these shortcomings by utilizing the recent state-of-the-art 3D explicit representation, Gaussian Splatting, and an unconditional diffusion model. This model learns to generate 3D objects represented by sets of GS ellipsoids. With these strong generative 3D priors, though learning unconditionally, the diffusion model is ready for view-guided reconstruction without further model fine-tuning. This is achieved by propagating fine-grained 2D features through the efficient yet flexible splatting function and the guided denoising sampling process. In addition, a 2D diffusion model is further employed to enhance rendering fidelity, and improve reconstructed GS quality by polishing and re-using the rendered images. The final reconstructed objects explicitly come with high-quality 3D structure and texture, and can be efficiently rendered in arbitrary views. Experiments on the challenging real-world CO3D dataset demonstrate the superiority of our approach.

我们提出了GSD，一种基于高斯喷溅（Gaussian Splatting，GS）表示的扩散模型方法，用于从单个视角进行3D物体重建。先前的工作由于不恰当的表示方法而导致3D几何不一致或渲染质量中等。我们通过利用最近的最先进3D显式表示方法——高斯喷溅，以及无条件的扩散模型，试图解决这些问题。这个模型学习生成由一组GS椭球体表示的3D物体。凭借这些强大的生成3D先验，虽然是无条件学习的，扩散模型已经可以进行视图引导的重建，无需进一步的模型微调。这是通过通过高效而灵活的光滑函数和引导去噪采样过程传播细粒度的2D特征实现的。此外，还进一步使用了2D扩散模型来增强渲染保真度，并通过优化和重复使用渲染图像来改善重建的GS质量。最终重建的物体具有高质量的3D结构和纹理，能够在任意视角高效渲染。在具有挑战性的真实世界CO3D数据集上的实验证明了我们方法的优越性。


---

## [62] LaRa: Efficient Large-Baseline Radiance Fields

### LaRa: Efficient Large-Baseline Radiance Fields

Radiance field methods have achieved photorealistic novel view synthesis and geometry reconstruction. But they are mostly applied in per-scene optimization or small-baseline settings. While several recent works investigate feed-forward reconstruction with large baselines by utilizing transformers, they all operate with a standard global attention mechanism and hence ignore the local nature of 3D reconstruction. We propose a method that unifies local and global reasoning in transformer layers, resulting in improved quality and faster convergence. Our model represents scenes as Gaussian Volumes and combines this with an image encoder and Group Attention Layers for efficient feed-forward reconstruction. Experimental results demonstrate that our model, trained for two days on four GPUs, demonstrates high fidelity in reconstructing 360&deg radiance fields, and robustness to zero-shot and out-of-domain testing.

辐射场方法已经实现了逼真的新视角合成和几何重建。但它们大多应用于每个场景的优化或小基线设置。尽管最近有几项研究探讨了利用变压器进行大基线的前向重建，但它们都使用了标准的全局注意力机制，因此忽略了3D重建的局部特性。我们提出了一种方法，在变压器层中统一了局部和全局推理，从而提高了质量并加快了收敛速度。我们的模型将场景表示为高斯体，并结合图像编码器和群组注意力层进行高效的前向重建。实验结果表明，我们的模型在四个GPU上训练两天后，展示了在重建360度辐射场方面的高保真度，并对零样本和域外测试具有稳健性。


---

## [63] GaussReg: Fast 3D Registration with Gaussian Splatting

### GaussReg: Fast 3D Registration with Gaussian Splatting

Point cloud registration is a fundamental problem for large-scale 3D scene scanning and reconstruction. With the help of deep learning, registration methods have evolved significantly, reaching a nearly-mature stage. As the introduction of Neural Radiance Fields (NeRF), it has become the most popular 3D scene representation as its powerful view synthesis capabilities. Regarding NeRF representation, its registration is also required for large-scale scene reconstruction. However, this topic extremly lacks exploration. This is due to the inherent challenge to model the geometric relationship among two scenes with implicit representations. The existing methods usually convert the implicit representation to explicit representation for further registration. Most recently, Gaussian Splatting (GS) is introduced, employing explicit 3D Gaussian. This method significantly enhances rendering speed while maintaining high rendering quality. Given two scenes with explicit GS representations, in this work, we explore the 3D registration task between them. To this end, we propose GaussReg, a novel coarse-to-fine framework, both fast and accurate. The coarse stage follows existing point cloud registration methods and estimates a rough alignment for point clouds from GS. We further newly present an image-guided fine registration approach, which renders images from GS to provide more detailed geometric information for precise alignment. To support comprehensive evaluation, we carefully build a scene-level dataset called ScanNet-GSReg with 1379 scenes obtained from the ScanNet dataset and collect an in-the-wild dataset called GSReg. Experimental results demonstrate our method achieves state-of-the-art performance on multiple datasets. Our GaussReg is 44 times faster than HLoc (SuperPoint as the feature extractor and SuperGlue as the matcher) with comparable accuracy.

点云配准是大规模 3D 场景扫描和重建的一个基本问题。在深度学习的帮助下，配准方法已经显著发展，达到了接近成熟的阶段。随着神经辐射场(NeRF)的引入，由于其强大的视图合成能力，它已成为最流行的 3D 场景表示方法。对于 NeRF 表示，其配准也是大规模场景重建所需的。然而，这个主题极度缺乏探索。这是由于在具有隐式表示的两个场景之间建模几何关系的固有挑战。现有方法通常将隐式表示转换为显式表示以进行进一步配准。最近，高斯散射(GS)被引入，采用显式 3D 高斯。这种方法显著提高了渲染速度，同时保持了高渲染质量。
给定两个具有显式 GS 表示的场景，在本工作中，我们探索了它们之间的 3D 配准任务。为此，我们提出了 GaussReg，一种新颖的粗到细框架，既快速又准确。粗配准阶段遵循现有的点云配准方法，并对来自 GS 的点云进行粗略对齐估计。我们进一步提出了一种新的图像引导精细配准方法，该方法从 GS 渲染图像以提供更详细的几何信息，用于精确对齐。
为支持全面评估，我们精心构建了一个名为 ScanNet-GSReg 的场景级数据集，其中包含从 ScanNet 数据集获得的 1379 个场景，并收集了一个名为 GSReg 的实际应用数据集。实验结果表明，我们的方法在多个数据集上达到了最先进的性能。我们的 GaussReg 比 HLoc（使用 SuperPoint 作为特征提取器和 SuperGlue 作为匹配器）快 44 倍，同时保持了可比的准确性。


---

## [64] MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition

### MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition

We introduce MIGS (Multi-Identity Gaussian Splatting), a novel method that learns a single neural representation for multiple identities, using only monocular videos. Recent 3D Gaussian Splatting (3DGS) approaches for human avatars require per-identity optimization. However, learning a multi-identity representation presents advantages in robustly animating humans under arbitrary poses. We propose to construct a high-order tensor that combines all the learnable 3DGS parameters for all the training identities. By assuming a low-rank structure and factorizing the tensor, we model the complex rigid and non-rigid deformations of multiple subjects in a unified network, significantly reducing the total number of parameters. Our proposed approach leverages information from all the training identities, enabling robust animation under challenging unseen poses, outperforming existing approaches. We also demonstrate how it can be extended to learn unseen identities.

我们介绍了 MIGS（多身份高斯散射），这是一种新型方法，仅使用单目视频就能学习多个身份的单一神经表示。最近的用于人体头像的 3D 高斯散射（3DGS）方法需要针对每个身份进行优化。然而，学习多身份表示在任意姿势下稳健地为人体制作动画方面具有优势。
我们提出构建一个高阶张量，该张量结合了所有训练身份的所有可学习 3DGS 参数。通过假设低秩结构并对张量进行因子分解，我们在一个统一的网络中建模多个主体的复杂刚性和非刚性变形，显著减少了总参数数量。
我们提出的方法利用了所有训练身份的信息，使其能够在具有挑战性的未见姿势下进行稳健的动画制作，优于现有方法。我们还展示了如何将其扩展到学习未见身份。


---

## [65] 3DEgo: 3D Editing on the Go!

### 3DEgo: 3D Editing on the Go!

We introduce 3DEgo to address a novel problem of directly synthesizing photorealistic 3D scenes from monocular videos guided by textual prompts. Conventional methods construct a text-conditioned 3D scene through a three-stage process, involving pose estimation using Structure-from-Motion (SfM) libraries like COLMAP, initializing the 3D model with unedited images, and iteratively updating the dataset with edited images to achieve a 3D scene with text fidelity. Our framework streamlines the conventional multi-stage 3D editing process into a single-stage workflow by overcoming the reliance on COLMAP and eliminating the cost of model initialization. We apply a diffusion model to edit video frames prior to 3D scene creation by incorporating our designed noise blender module for enhancing multi-view editing consistency, a step that does not require additional training or fine-tuning of T2I diffusion models. 3DEgo utilizes 3D Gaussian Splatting to create 3D scenes from the multi-view consistent edited frames, capitalizing on the inherent temporal continuity and explicit point cloud data. 3DEgo demonstrates remarkable editing precision, speed, and adaptability across a variety of video sources, as validated by extensive evaluations on six datasets, including our own prepared GS25 dataset.

我们引入3DEgo来解决一个新问题：直接从单眼视频中，通过文本提示引导合成逼真的3D场景。传统方法通过三阶段过程构建一个文本条件的3D场景，包括使用Structure-from-Motion（SfM）库如COLMAP进行姿态估计，用未编辑的图像初始化3D模型，并通过迭代更新编辑过的图像数据集以实现具有文本保真度的3D场景。我们的框架通过克服对COLMAP的依赖并消除模型初始化的成本，将传统的多阶段3D编辑过程简化为单阶段工作流。我们在3D场景创建之前应用扩散模型编辑视频帧，整合我们设计的噪声混合模块以增强多视图编辑的一致性，这一步骤不需要额外的训练或微调T2I扩散模型。3DEgo利用3D高斯喷溅从多视图一致的编辑帧中创建3D场景，利用固有的时间连续性和显式点云数据。3DEgo在各种视频源上展示了卓越的编辑精度、速度和适应性，通过在六个数据集上的广泛评估进行了验证，包括我们自己准备的GS25数据集。


---

## [66] iHuman: Instant Animatable Digital Humans From Monocular Videos

### iHuman: Instant Animatable Digital Humans From Monocular Videos

Personalized 3D avatars require an animatable representation of digital humans. Doing so instantly from monocular videos offers scalability to broad class of users and wide-scale applications. In this paper, we present a fast, simple, yet effective method for creating animatable 3D digital humans from monocular videos. Our method utilizes the efficiency of Gaussian splatting to model both 3D geometry and appearance. However, we observed that naively optimizing Gaussian splats results in inaccurate geometry, thereby leading to poor animations. This work achieves and illustrates the need of accurate 3D mesh-type modelling of the human body for animatable digitization through Gaussian splats. This is achieved by developing a novel pipeline that benefits from three key aspects: (a) implicit modelling of surface's displacements and the color's spherical harmonics; (b) binding of 3D Gaussians to the respective triangular faces of the body template; (c) a novel technique to render normals followed by their auxiliary supervision. Our exhaustive experiments on three different benchmark datasets demonstrates the state-of-the-art results of our method, in limited time settings. In fact, our method is faster by an order of magnitude (in terms of training time) than its closest competitor. At the same time, we achieve superior rendering and 3D reconstruction performance under the change of poses.

个性化3D头像需要一种可动画化的数字人类表现形式。通过单目视频即时实现这一点，可以扩展到广泛的用户和大规模应用。在本文中，我们介绍了一种快速、简单且有效的方法，用于从单目视频创建可动画化的3D数字人类。我们的方法利用高斯喷溅的效率来同时建模3D几何形状和外观。然而，我们观察到，简单地优化高斯喷溅会导致几何形状不准确，从而导致动画效果差。本工作通过高斯喷溅实现并展示了精确的3D网格类型人体建模对可动画数字化的需求。这是通过开发一个从三个关键方面受益的新颖流程实现的：(a) 表面位移和颜色球谐的隐式建模；(b) 将3D高斯绑定到身体模板的相应三角形面；(c) 一种渲染法线的新技术，随后进行辅助监督。我们在三个不同的基准数据集上进行的详尽实验展示了我们方法在有限时间设置中的最新成果。事实上，就训练时间而言，我们的方法比最接近的竞争者快一个数量级。同时，在姿势变化下，我们实现了更优越的渲染和3D重建性能。



---

## [67] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians

### Click-Gaussian: Interactive Segmentation to Any 3D Gaussians

Interactive segmentation of 3D Gaussians opens a great opportunity for real-time manipulation of 3D scenes thanks to the real-time rendering capability of 3D Gaussian Splatting. However, the current methods suffer from time-consuming post-processing to deal with noisy segmentation output. Also, they struggle to provide detailed segmentation, which is important for fine-grained manipulation of 3D scenes. In this study, we propose Click-Gaussian, which learns distinguishable feature fields of two-level granularity, facilitating segmentation without time-consuming post-processing. We delve into challenges stemming from inconsistently learned feature fields resulting from 2D segmentation obtained independently from a 3D scene. 3D segmentation accuracy deteriorates when 2D segmentation results across the views, primary cues for 3D segmentation, are in conflict. To overcome these issues, we propose Global Feature-guided Learning (GFL). GFL constructs the clusters of global feature candidates from noisy 2D segments across the views, which smooths out noises when training the features of 3D Gaussians. Our method runs in 10 ms per click, 15 to 130 times as fast as the previous methods, while also significantly improving segmentation accuracy.

交互式3D高斯分割为实时操作3D场景提供了巨大的机会，这得益于3D高斯喷溅的实时渲染能力。然而，当前方法因为要处理噪声分割输出而遭受耗时的后处理问题。同时，它们难以提供详细的分割，这对于精细操作3D场景非常重要。在本研究中，我们提出了Click-Gaussian，该方法学习两级粒度的可区分特征场，促进了无需耗时后处理的分割。我们深入探讨了由独立于3D场景获得的2D分割产生的特征场学习不一致性带来的挑战。当2D分割结果（3D分割的主要线索）在不同视图中存在冲突时，3D分割精度会恶化。为克服这些问题，我们提出了全局特征引导学习（GFL）。GFL从各视图的噪声2D分割中构建全局特征候选者群，这在训练3D高斯的特征时平滑了噪声。我们的方法每次点击运行时间为10毫秒，比以前的方法快15到130倍，同时显著提高了分割精度。



---

## [68] Generalizable Human Gaussians for Sparse View Synthesis

### Generalizable Human Gaussians for Sparse View Synthesis

Recent progress in neural rendering has brought forth pioneering methods, such as NeRF and Gaussian Splatting, which revolutionize view rendering across various domains like AR/VR, gaming, and content creation. While these methods excel at interpolating {\em within the training data}, the challenge of generalizing to new scenes and objects from very sparse views persists. Specifically, modeling 3D humans from sparse views presents formidable hurdles due to the inherent complexity of human geometry, resulting in inaccurate reconstructions of geometry and textures. To tackle this challenge, this paper leverages recent advancements in Gaussian Splatting and introduces a new method to learn generalizable human Gaussians that allows photorealistic and accurate view-rendering of a new human subject from a limited set of sparse views in a feed-forward manner. A pivotal innovation of our approach involves reformulating the learning of 3D Gaussian parameters into a regression process defined on the 2D UV space of a human template, which allows leveraging the strong geometry prior and the advantages of 2D convolutions. In addition, a multi-scaffold is proposed to effectively represent the offset details. Our method outperforms recent methods on both within-dataset generalization as well as cross-dataset generalization settings.

近期在神经渲染领域的进展催生了一些开创性的方法，例如神经辐射场（NeRF）和高斯投影，这些方法在增强现实/虚拟现实、游戏以及内容创作等多个领域革新了视图渲染。虽然这些方法在插值训练数据方面表现出色，但在从非常稀疏的视角推广到新场景和对象的挑战仍然存在。特别是，从稀疏视角对三维人类进行建模面临巨大障碍，因为人类几何形态的固有复杂性，导致几何和纹理重建的不准确。为了应对这一挑战，本文利用高斯投影的最新进展，并引入了一种新方法学习可推广的人类高斯，这种方法能够以前馈方式从有限的稀疏视角实现新人类对象的逼真和精确视图渲染。我们方法的一个关键创新在于将学习三维高斯参数的过程重新定义为在人类模板的二维UV空间上进行的回归过程，这使得我们能够利用强大的几何先验和二维卷积的优势。此外，还提出了一个多脚手架模型来有效地表示偏移细节。我们的方法在数据集内外的泛化能力上均优于最近的方法。


---

## [69] Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation

### Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation

Although recent advancements in text-to-3D generation have significantly improved generation quality, issues like limited level of detail and low fidelity still persist, which requires further improvement. To understand the essence of those issues, we thoroughly analyze current score distillation methods by connecting theories of consistency distillation to score distillation. Based on the insights acquired through analysis, we propose an optimization framework, Guided Consistency Sampling (GCS), integrated with 3D Gaussian Splatting (3DGS) to alleviate those issues. Additionally, we have observed the persistent oversaturation in the rendered views of generated 3D assets. From experiments, we find that it is caused by unwanted accumulated brightness in 3DGS during optimization. To mitigate this issue, we introduce a Brightness-Equalized Generation (BEG) scheme in 3DGS rendering. Experimental results demonstrate that our approach generates 3D assets with more details and higher fidelity than state-of-the-art methods.

尽管最近在文本到三维生成的技术进展显著提高了生成质量，但诸如细节水平有限和保真度低等问题仍然存在，这需要进一步改进。为了理解这些问题的本质，我们通过将一致性蒸馏理论与得分蒸馏相结合，对当前的得分蒸馏方法进行了深入分析。基于通过分析获得的洞察，我们提出了一个优化框架——引导一致性采样（GCS），并将其与三维高斯投影（3DGS）整合，以缓解这些问题。此外，我们还观察到生成的三维资产的渲染视图中持续存在过饱和现象。通过实验，我们发现这是由于在优化过程中3DGS中不希望的亮度累积造成的。为了缓解这一问题，我们在3DGS渲染中引入了一个亮度均衡生成（BEG）方案。实验结果表明，我们的方法生成的三维资产比现有最先进方法具有更多细节和更高的保真度。


---

## [70] 3D Gaussian Parametric Head Model

### 3D Gaussian Parametric Head Model

Creating high-fidelity 3D human head avatars is crucial for applications in VR/AR, telepresence, digital human interfaces, and film production. Recent advances have leveraged morphable face models to generate animated head avatars from easily accessible data, representing varying identities and expressions within a low-dimensional parametric space. However, existing methods often struggle with modeling complex appearance details, e.g., hairstyles and accessories, and suffer from low rendering quality and efficiency. This paper introduces a novel approach, 3D Gaussian Parametric Head Model, which employs 3D Gaussians to accurately represent the complexities of the human head, allowing precise control over both identity and expression. Additionally, it enables seamless face portrait interpolation and the reconstruction of detailed head avatars from a single image. Unlike previous methods, the Gaussian model can handle intricate details, enabling realistic representations of varying appearances and complex expressions. Furthermore, this paper presents a well-designed training framework to ensure smooth convergence, providing a guarantee for learning the rich content. Our method achieves high-quality, photo-realistic rendering with real-time efficiency, making it a valuable contribution to the field of parametric head models.

创建高保真度的3D人头头像对于虚拟现实/增强现实、远程存在、数字人界面和电影制作等应用至关重要。近期的进展利用了可变形面部模型,从易于获取的数据生成动画头像,在低维参数空间内表示不同的身份和表情。然而,现有方法往往难以模拟复杂的外观细节,如发型和配饰,并且存在渲染质量低和效率低的问题。本文介绍了一种新颖的方法,3D高斯参数化头部模型,该模型采用3D高斯分布精确表示人头的复杂性,允许对身份和表情进行精确控制。此外,它还能实现无缝的面部肖像插值和从单一图像重建详细的头像。与之前的方法不同,高斯模型可以处理复杂的细节,能够逼真地表现各种外观和复杂的表情。此外,本文提出了一个精心设计的训练框架,以确保平稳收敛,为学习丰富内容提供保证。我们的方法实现了高质量、真实感的渲染,同时具有实时效率,为参数化头部模型领域做出了宝贵贡献。


---

## [71] 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model

### 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model

We propose 6DGS to estimate the camera pose of a target RGB image given a 3D Gaussian Splatting (3DGS) model representing the scene. 6DGS avoids the iterative process typical of analysis-by-synthesis methods (e.g. iNeRF) that also require an initialization of the camera pose in order to converge. Instead, our method estimates a 6DoF pose by inverting the 3DGS rendering process. Starting from the object surface, we define a radiant Ellicell that uniformly generates rays departing from each ellipsoid that parameterize the 3DGS model. Each Ellicell ray is associated with the rendering parameters of each ellipsoid, which in turn is used to obtain the best bindings between the target image pixels and the cast rays. These pixel-ray bindings are then ranked to select the best scoring bundle of rays, which their intersection provides the camera center and, in turn, the camera rotation. The proposed solution obviates the necessity of an "a priori" pose for initialization, and it solves 6DoF pose estimation in closed form, without the need for iterations. Moreover, compared to the existing Novel View Synthesis (NVS) baselines for pose estimation, 6DGS can improve the overall average rotational accuracy by 12% and translation accuracy by 22% on real scenes, despite not requiring any initialization pose. At the same time, our method operates near real-time, reaching 15fps on consumer hardware.

我们提出了6DGS,用于在给定表示场景的3D高斯喷溅(3DGS)模型的情况下估计目标RGB图像的相机姿态。6DGS避免了分析合成方法(如iNeRF)典型的迭代过程,这些方法还需要对相机姿态进行初始化才能收敛。相反,我们的方法通过反转3DGS渲染过程来估计6自由度姿态。从物体表面开始,我们定义了一个辐射椭胞体(Ellicell),它均匀地生成从参数化3DGS模型的每个椭球体出发的射线。每个椭胞体射线与每个椭球体的渲染参数相关联,这反过来用于获得目标图像像素和投射射线之间的最佳绑定。然后对这些像素-射线绑定进行排序,以选择得分最高的射线束,其交点提供相机中心,进而确定相机旋转。提出的解决方案避免了需要"先验"姿态进行初始化,并以闭式形式解决6自由度姿态估计,无需迭代。此外,与用于姿态估计的现有新视角合成(NVS)基准相比,6DGS在真实场景中可以将整体平均旋转精度提高12%,平移精度提高22%,尽管不需要任何初始化姿态。同时,我们的方法接近实时运行,在消费级硬件上达到15fps。


---

## [72] Expressive Whole-Body 3D Gaussian Avatar

### Expressive Whole-Body 3D Gaussian Avatar

Facial expression and hand motions are necessary to express our emotions and interact with the world. Nevertheless, most of the 3D human avatars modeled from a casually captured video only support body motions without facial expressions and hand motions.In this work, we present ExAvatar, an expressive whole-body 3D human avatar learned from a short monocular video. We design ExAvatar as a combination of the whole-body parametric mesh model (SMPL-X) and 3D Gaussian Splatting (3DGS). The main challenges are 1) a limited diversity of facial expressions and poses in the video and 2) the absence of 3D observations, such as 3D scans and RGBD images. The limited diversity in the video makes animations with novel facial expressions and poses non-trivial. In addition, the absence of 3D observations could cause significant ambiguity in human parts that are not observed in the video, which can result in noticeable artifacts under novel motions. To address them, we introduce our hybrid representation of the mesh and 3D Gaussians. Our hybrid representation treats each 3D Gaussian as a vertex on the surface with pre-defined connectivity information (i.e., triangle faces) between them following the mesh topology of SMPL-X. It makes our ExAvatar animatable with novel facial expressions by driven by the facial expression space of SMPL-X. In addition, by using connectivity-based regularizers, we significantly reduce artifacts in novel facial expressions and poses.

面部表情和手部动作对表达情感和与世界互动至关重要。然而，大多数从随意捕获的视频中建模的3D人类头像仅支持身体动作而不支持面部表情和手部动作。在这项工作中，我们提出了ExAvatar，这是一种从短单目视频中学习到的具有表现力的全身3D人类头像。我们将ExAvatar设计为全身参数化网格模型（SMPL-X）和3D高斯点云（3DGS）的组合。主要挑战包括：1）视频中的面部表情和姿势多样性有限，2）缺乏3D观测，如3D扫描和RGBD图像。视频中的有限多样性使得带有新面部表情和姿势的动画变得复杂。此外，缺乏3D观测可能导致视频中未观察到的人体部分出现显著模糊，这可能在新动作下产生明显的伪影。为解决这些问题，我们引入了网格和3D高斯点云的混合表示。我们的混合表示将每个3D高斯点视为表面上的一个顶点，并根据SMPL-X的网格拓扑定义其连接信息（即三角形面）。这使得我们的ExAvatar能够通过SMPL-X的面部表情空间驱动，从而实现新面部表情的动画。此外，通过使用基于连接的正则化器，我们显著减少了新面部表情和姿势下的伪影。


---

## [73] EmoTalk3D: High-Fidelity Free-View Synthesis of Emotional 3D Talking Head

### EmoTalk3D: High-Fidelity Free-View Synthesis of Emotional 3D Talking Head

We present a novel approach for synthesizing 3D talking heads with controllable emotion, featuring enhanced lip synchronization and rendering quality. Despite significant progress in the field, prior methods still suffer from multi-view consistency and a lack of emotional expressiveness. To address these issues, we collect EmoTalk3D dataset with calibrated multi-view videos, emotional annotations, and per-frame 3D geometry. By training on the EmoTalk3D dataset, we propose a Speech-to-Geometry-to-Appearance mapping framework that first predicts faithful 3D geometry sequence from the audio features, then the appearance of a 3D talking head represented by 4D Gaussians is synthesized from the predicted geometry. The appearance is further disentangled into canonical and dynamic Gaussians, learned from multi-view videos, and fused to render free-view talking head animation. Moreover, our model enables controllable emotion in the generated talking heads and can be rendered in wide-range views. Our method exhibits improved rendering quality and stability in lip motion generation while capturing dynamic facial details such as wrinkles and subtle expressions. Experiments demonstrate the effectiveness of our approach in generating high-fidelity and emotion-controllable 3D talking heads.

我们提出了一种新颖的方法，用于合成可控情感的三维说话头部模型，具有增强的唇部同步性和渲染质量。尽管该领域取得了显著进展，但现有方法仍存在多视角一致性差和情感表现不足的问题。为解决这些问题，我们收集了带有校准多视角视频、情感标注和每帧三维几何数据的EmoTalk3D数据集。通过在EmoTalk3D数据集上训练，我们提出了一个从语音到几何再到外观的映射框架，该框架首先根据音频特征预测忠实的三维几何序列，然后从预测的几何中合成由4D高斯表示的三维说话头部的外观。外观进一步被解构为从多视角视频中学习到的标准和动态高斯，并融合以渲染自由视角的说话头部动画。此外，我们的模型能够在生成的说话头部中实现可控的情感，并能在广泛的视角中渲染。我们的方法在唇部运动生成的渲染质量和稳定性方面表现出色，同时捕捉到动态的面部细节，如皱纹和细微表情。实验表明，我们的方法在生成高保真度和可控情感的三维说话头部方面有效。


---

## [74] 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting

### 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting

The use of 3D Gaussians as representation of radiance fields has enabled high quality novel view synthesis at real-time rendering speed. However, the choice of optimising the outgoing radiance of each Gaussian independently as spherical harmonics results in unsatisfactory view dependent effects. In response to these limitations, our work, Factorised Tensorial Illumination for 3D Gaussian Splatting, or 3iGS, improves upon 3D Gaussian Splatting (3DGS) rendering quality. Instead of optimising a single outgoing radiance parameter, 3iGS enhances 3DGS view-dependent effects by expressing the outgoing radiance as a function of a local illumination field and Bidirectional Reflectance Distribution Function (BRDF) features. We optimise a continuous incident illumination field through a Tensorial Factorisation representation, while separately fine-tuning the BRDF features of each 3D Gaussian relative to this illumination field. Our methodology significantly enhances the rendering quality of specular view-dependent effects of 3DGS, while maintaining rapid training and rendering speeds.

将 3D 高斯点光源用于辐射场表示已实现了高质量的新视角合成，并具备实时渲染速度。然而，选择独立优化每个高斯点光源的输出辐射作为球面谐波的方法，导致了不令人满意的视角依赖效果。为了应对这些局限性，我们的研究提出了分解张量光照用于 3D 高斯点光源（3iGS），旨在提升 3D 高斯点光源（3DGS）的渲染质量。与优化单一的输出辐射参数不同，3iGS 通过将输出辐射表示为局部光照场和双向反射分布函数（BRDF）特征的函数，增强了 3DGS 的视角依赖效果。我们通过张量分解表示优化连续的入射光照场，同时相对于该光照场单独微调每个 3D 高斯点光源的 BRDF 特征。我们的方法显著提高了 3DGS 的高光视角依赖效果，同时保持了快速的训练和渲染速度。


---

## [75] Fisheye-GS: Lightweight and Extensible Gaussian Splatting Module for Fisheye Cameras

### Fisheye-GS: Lightweight and Extensible Gaussian Splatting Module for Fisheye Cameras

Recently, 3D Gaussian Splatting (3DGS) has garnered attention for its high fidelity and real-time rendering. However, adapting 3DGS to different camera models, particularly fisheye lenses, poses challenges due to the unique 3D to 2D projection calculation. Additionally, there are inefficiencies in the tile-based splatting, especially for the extreme curvature and wide field of view of fisheye lenses, which are crucial for its broader real-life applications. To tackle these challenges, we introduce Fisheye-GS.This innovative method recalculates the projection transformation and its gradients for fisheye cameras. Our approach can be seamlessly integrated as a module into other efficient 3D rendering methods, emphasizing its extensibility, lightweight nature, and modular design. Since we only modified the projection component, it can also be easily adapted for use with different camera models. Compared to methods that train after undistortion, our approach demonstrates a clear improvement in visual quality.

近期，3D高斯分裂（3DGS）因其高保真度和实时渲染性能备受关注。然而，将3DGS应用于不同的相机模型，尤其是鱼眼镜头，面临挑战，主要是由于独特的3D到2D投影计算。此外，基于平铺的高斯分裂在鱼眼镜头的极端曲率和广视角条件下效率不高，这对于其更广泛的实际应用至关重要。为了解决这些问题，我们提出了Fisheye-GS⋆。该创新方法重新计算了鱼眼相机的投影变换及其梯度。我们的方法可以作为一个模块无缝集成到其他高效的3D渲染方法中，强调其可扩展性、轻量化和模块化设计。由于我们只修改了投影部分，该方法也可以轻松适配不同的相机模型。与通过图像去畸变后进行训练的方法相比，我们的方法在视觉质量上有明显提升。


---

## [76] Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis

### Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis

Novel-view synthesis based on visible light has been extensively studied. In comparison to visible light imaging, thermal infrared imaging offers the advantage of all-weather imaging and strong penetration, providing increased possibilities for reconstruction in nighttime and adverse weather scenarios. However, thermal infrared imaging is influenced by physical characteristics such as atmospheric transmission effects and thermal conduction, hindering the precise reconstruction of intricate details in thermal infrared scenes, manifesting as issues of floaters and indistinct edge features in synthesized images. To address these limitations, this paper introduces a physics-induced 3D Gaussian splatting method named Thermal3D-GS. Thermal3D-GS begins by modeling atmospheric transmission effects and thermal conduction in three-dimensional media using neural networks. Additionally, a temperature consistency constraint is incorporated into the optimization objective to enhance the reconstruction accuracy of thermal infrared images. Furthermore, to validate the effectiveness of our method, the first large-scale benchmark dataset for this field named Thermal Infrared Novel-view Synthesis Dataset (TI-NSD) is created. This dataset comprises 20 authentic thermal infrared video scenes, covering indoor, outdoor, and UAV(Unmanned Aerial Vehicle) scenarios, totaling 6,664 frames of thermal infrared image data. Based on this dataset, this paper experimentally verifies the effectiveness of Thermal3D-GS. The results indicate that our method outperforms the baseline method with a 3.03 dB improvement in PSNR and significantly addresses the issues of floaters and indistinct edge features present in the baseline method.

基于可见光的新视图合成已被广泛研究。相比于可见光成像，热红外成像具备全天候成像和强穿透力的优势，能够在夜间和恶劣天气条件下提供更多的重建可能性。然而，热红外成像受到大气传输效应和热传导等物理特性的影响，难以精准重建热红外场景中的细节，表现为合成图像中的浮动伪影和边缘特征模糊等问题。为了解决这些局限性，本文提出了一种名为 Thermal3D-GS 的物理驱动3D高斯散点方法。Thermal3D-GS 首先通过神经网络对三维介质中的大气传输效应和热传导进行建模。此外，还将温度一致性约束引入到优化目标中，以提高热红外图像的重建精度。
为了验证该方法的有效性，本文创建了该领域首个大规模基准数据集，名为 Thermal Infrared Novel-view Synthesis Dataset (TI-NSD)。该数据集包含20个真实的热红外视频场景，涵盖室内、室外以及无人机（UAV）场景，总计包含6,664帧热红外图像数据。基于该数据集，本文通过实验验证了 Thermal3D-GS 的有效性。结果表明，我们的方法在PSNR上较基线方法提升了3.03 dB，并显著解决了基线方法中存在的浮动伪影和边缘模糊问题。


---

## [77] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally

### FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally

This study addresses the challenge of accurately segmenting 3D Gaussian Splatting from 2D masks. Conventional methods often rely on iterative gradient descent to assign each Gaussian a unique label, leading to lengthy optimization and sub-optimal solutions. Instead, we propose a straightforward yet globally optimal solver for 3D-GS segmentation. The core insight of our method is that, with a reconstructed 3D-GS scene, the rendering of the 2D masks is essentially a linear function with respect to the labels of each Gaussian. As such, the optimal label assignment can be solved via linear programming in closed form. This solution capitalizes on the alpha blending characteristic of the splatting process for single step optimization. By incorporating the background bias in our objective function, our method shows superior robustness in 3D segmentation against noises. Remarkably, our optimization completes within 30 seconds, about 50× faster than the best existing methods. Extensive experiments demonstrate the efficiency and robustness of our method in segmenting various scenes, and its superior performance in downstream tasks such as object removal and inpainting.

本研究解决了从2D掩码中准确分割3D Gaussian Splatting (3D-GS) 的挑战。传统方法通常依赖迭代的梯度下降算法为每个高斯分配唯一的标签，这导致了冗长的优化过程和次优解。相较之下，我们提出了一种简单且全局最优的3D-GS分割求解器。我们方法的核心洞见在于，对于已重建的3D-GS场景，2D掩码的渲染本质上是与每个高斯的标签相关的线性函数。因此，最优的标签分配可以通过线性规划以闭式形式解决。该方案利用了散点渲染过程中alpha混合的特性，实现了单步优化。通过在目标函数中引入背景偏置，我们的方法在面对噪声时展现出更强的鲁棒性。值得注意的是，我们的优化在30秒内完成，比现有最佳方法快约50倍。广泛的实验表明，我们的方法在分割各种场景中的高效性和鲁棒性，并且在物体移除和修补等下游任务中表现出优越的性能。


---

## [78] MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation

### MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation

3D Gaussian Splatting demonstrates excellent quality and speed in novel view synthesis. Nevertheless, the huge file size of the 3D Gaussians presents challenges for transmission and storage. Current works design compact models to replace the substantial volume and attributes of 3D Gaussians, along with intensive training to distill information. These endeavors demand considerable training time, presenting formidable hurdles for practical deployment. To this end, we propose MesonGS, a codec for post-training compression of 3D Gaussians. Initially, we introduce a measurement criterion that considers both view-dependent and view-independent factors to assess the impact of each Gaussian point on the rendering output, enabling the removal of insignificant points. Subsequently, we decrease the entropy of attributes through two transformations that complement subsequent entropy coding techniques to enhance the file compression rate. More specifically, we first replace rotation quaternions with Euler angles; then, we apply region adaptive hierarchical transform to key attributes to reduce entropy. Lastly, we adopt finer-grained quantization to avoid excessive information loss. Moreover, a well-crafted finetune scheme is devised to restore quality. Extensive experiments demonstrate that MesonGS significantly reduces the size of 3D Gaussians while preserving competitive quality.

3D 高斯投影（3D Gaussian Splatting）在新视角合成中展现了出色的质量和速度。然而，3D 高斯文件的巨大体积在传输和存储方面带来了挑战。当前的工作通过设计紧凑的模型来取代3D高斯的庞大体积和属性，并通过密集的训练来提取信息。这些工作需要大量的训练时间，给实际部署带来了巨大的困难。为此，我们提出了 MesonGS，这是一种用于 3D 高斯后训练压缩的编解码器。首先，我们引入了一种度量标准，考虑视角相关和视角无关因素，以评估每个高斯点对渲染输出的影响，从而删除不重要的点。接下来，我们通过两种变换减少属性的熵，以配合后续的熵编码技术，从而提高文件的压缩率。具体来说，我们首先用欧拉角替代旋转四元数；然后，我们对关键属性应用区域自适应分层变换以减少熵。最后，我们采用更精细的量化方法，避免过度的信息丢失。此外，我们设计了一种精心构造的微调方案，以恢复质量。大量实验表明，MesonGS 在显著减少 3D 高斯文件体积的同时，保持了有竞争力的质量。


---

## [79] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction

### SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction

Digitizing 3D static scenes and 4D dynamic events from multi-view images has long been a challenge in computer vision and graphics. Recently, 3D Gaussian Splatting (3DGS) has emerged as a practical and scalable reconstruction method, gaining popularity due to its impressive reconstruction quality, real-time rendering capabilities, and compatibility with widely used visualization tools. However, the method requires a substantial number of input views to achieve high-quality scene reconstruction, introducing a significant practical bottleneck. This challenge is especially severe in capturing dynamic scenes, where deploying an extensive camera array can be prohibitively costly. In this work, we identify the lack of spatial autocorrelation of splat features as one of the factors contributing to the suboptimal performance of the 3DGS technique in sparse reconstruction settings. To address the issue, we propose an optimization strategy that effectively regularizes splat features by modeling them as the outputs of a corresponding implicit neural field. This results in a consistent enhancement of reconstruction quality across various scenarios. Our approach effectively handles static and dynamic cases, as demonstrated by extensive testing across different setups and scene complexities.

从多视角图像中数字化3D静态场景和4D动态事件一直是计算机视觉和图形学中的一大挑战。近年来，3D高斯投影（3D Gaussian Splatting, 3DGS）作为一种实用且可扩展的重建方法，凭借其卓越的重建质量、实时渲染能力以及与广泛使用的可视化工具兼容性，逐渐受到关注。然而，该方法需要大量的输入视角才能实现高质量的场景重建，这成为一个显著的实际瓶颈。这个挑战在捕捉动态场景时尤为严重，因为部署大规模的摄像机阵列成本高昂。
在本研究中，我们确定了高斯点特征缺乏空间自相关性是3DGS技术在稀疏重建场景中表现不佳的原因之一。为了解决这一问题，我们提出了一种优化策略，通过将高斯点特征建模为相应隐式神经场的输出，有效地对其进行正则化处理。这种方法在各种场景下持续提升了重建质量。通过广泛的测试，我们的方法在处理静态和动态场景方面都表现出色，并在不同的设置和场景复杂度下取得了显著的效果。


---

## [80] Vista3D: Unravel the 3D Darkside of a Single Image

### Vista3D: Unravel the 3D Darkside of a Single Image

We embark on the age-old quest: unveiling the hidden dimensions of objects from mere glimpses of their visible parts. To address this, we present Vista3D, a framework that realizes swift and consistent 3D generation within a mere 5 minutes. At the heart of Vista3D lies a two-phase approach: the coarse phase and the fine phase. In the coarse phase, we rapidly generate initial geometry with Gaussian Splatting from a single image. In the fine phase, we extract a Signed Distance Function (SDF) directly from learned Gaussian Splatting, optimizing it with a differentiable isosurface representation. Furthermore, it elevates the quality of generation by using a disentangled representation with two independent implicit functions to capture both visible and obscured aspects of objects. Additionally, it harmonizes gradients from 2D diffusion prior with 3D-aware diffusion priors by angular diffusion prior composition. Through extensive evaluation, we demonstrate that Vista3D effectively sustains a balance between the consistency and diversity of the generated 3D objects.

我们着手解决一个古老的难题：通过仅能看到物体的一部分来揭示其隐藏的维度。为此，我们提出了Vista3D，一个能够在短短5分钟内实现快速且一致的3D生成框架。Vista3D的核心采用了两阶段方法：粗略阶段和精细阶段。在粗略阶段，我们通过单张图像快速生成初步几何形态，使用高斯散点技术。在精细阶段，我们从学习到的高斯散点中直接提取符号距离函数（SDF），并通过可微等值面表示进行优化。此外，Vista3D通过使用两组独立的隐式函数对可见和隐藏部分进行解耦表示，进一步提升了生成质量。同时，它通过角度扩散先验组合，将来自2D扩散模型的梯度与3D感知的扩散先验相结合。通过广泛的评估，我们证明了Vista3D能够在生成的3D物体的一致性和多样性之间有效地保持平衡。


---

## [81] MVPGS: Excavating Multi-view Priors for Gaussian Splatting from Sparse Input Views

### MVPGS: Excavating Multi-view Priors for Gaussian Splatting from Sparse Input Views

Recently, the Neural Radiance Field (NeRF) advancement has facilitated few-shot Novel View Synthesis (NVS), which is a significant challenge in 3D vision applications. Despite numerous attempts to reduce the dense input requirement in NeRF, it still suffers from time-consumed training and rendering processes. More recently, 3D Gaussian Splatting (3DGS) achieves real-time high-quality rendering with an explicit point-based representation. However, similar to NeRF, it tends to overfit the train views for lack of constraints. In this paper, we propose MVPGS, a few-shot NVS method that excavates the multi-view priors based on 3D Gaussian Splatting. We leverage the recent learning-based Multi-view Stereo (MVS) to enhance the quality of geometric initialization for 3DGS. To mitigate overfitting, we propose a forward-warping method for additional appearance constraints conforming to scenes based on the computed geometry. Furthermore, we introduce a view-consistent geometry constraint for Gaussian parameters to facilitate proper optimization convergence and utilize a monocular depth regularization as compensation. Experiments show that the proposed method achieves state-of-the-art performance with real-time rendering speed.

最近，神经辐射场（NeRF）的进步推动了少样本新视角合成（NVS）的发展，这是3D视觉应用中的一个重要挑战。尽管有许多尝试减少NeRF对密集输入的需求，它仍然面临耗时的训练和渲染过程。近期，3D Gaussian Splatting（3DGS）通过显式的点基表示实现了实时高质量渲染。然而，和NeRF类似，它也容易由于缺乏约束而过拟合训练视图。在本文中，我们提出了MVPGS，一种基于3D Gaussian Splatting的少样本NVS方法，挖掘多视角先验。我们利用最近的基于学习的多视角立体（MVS）方法，提升3DGS几何初始化的质量。为减轻过拟合，我们提出了一种前向变形方法，基于计算出的几何为场景提供额外的外观约束。此外，我们引入了一个视角一致的几何约束，以促进高斯参数的优化收敛，并使用单目深度正则化作为补充。实验表明，所提出的方法在实时渲染速度下实现了最先进的性能。


---

## [82] Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation

### Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation

The creation of digital replicas of physical objects has valuable applications for the preservation and dissemination of tangible cultural heritage. However, existing methods are often slow, expensive, and require expert knowledge. We propose a pipeline to generate a 3D replica of a scene using only RGB images (e.g. photos of a museum) and then extract a model for each item of interest (e.g. pieces in the exhibit). We do this by leveraging the advancements in novel view synthesis and Gaussian Splatting, modified to enable efficient 3D segmentation. This approach does not need manual annotation, and the visual inputs can be captured using a standard smartphone, making it both affordable and easy to deploy. We provide an overview of the method and baseline evaluation of the accuracy of object segmentation.

创建物理对象的数字复制品在保护和传播有形文化遗产方面具有重要的应用。然而，现有的方法通常速度慢、成本高，并且需要专业知识。我们提出了一种管道，仅使用RGB图像（如博物馆的照片）生成场景的三维复制品，并随后提取每个感兴趣物体（如展览中的展品）的模型。我们通过利用新视角合成和高斯分布（Gaussian Splatting）的进展，修改这些技术以实现高效的三维分割。此方法无需手动注释，视觉输入可以通过普通智能手机捕捉，使其既经济又易于部署。我们提供了该方法的概述以及物体分割精度的基准评估。


---

## [83] Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats

### Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats

We introduce a simple yet effective approach for separating transmitted and reflected light. Our key insight is that the powerful novel view synthesis capabilities provided by modern inverse rendering methods (e.g.,~3D Gaussian splatting) allow one to perform flash/no-flash reflection separation using unpaired measurements -- this relaxation dramatically simplifies image acquisition over conventional paired flash/no-flash reflection separation methods. Through extensive real-world experiments, we demonstrate our method, Flash-Splat, accurately reconstructs both transmitted and reflected scenes in 3D. Our method outperforms existing 3D reflection separation methods, which do not leverage illumination control, by a large margin.

我们提出了一种简单且有效的传输光与反射光分离方法。我们的关键见解在于，现代逆向渲染方法（如3D高斯散射）所提供的强大新视图合成功能，使得可以利用非成对的测量进行闪光灯/无闪光灯反射分离——这一放宽条件极大简化了图像获取过程，优于传统的成对闪光灯/无闪光灯反射分离方法。通过大量真实场景实验，我们展示了我们的方法——Flash-Splat，能够准确重建传输场景和反射场景的3D结构。我们的方法大幅超越了现有不依赖光照控制的3D反射分离方法。


---

## [84] Few-shot Novel View Synthesis using Depth Aware 3D Gaussian Splatting

### Few-shot Novel View Synthesis using Depth Aware 3D Gaussian Splatting

3D Gaussian splatting has surpassed neural radiance field methods in novel view synthesis by achieving lower computational costs and real-time high-quality rendering. Although it produces a high-quality rendering with a lot of input views, its performance drops significantly when only a few views are available. In this work, we address this by proposing a depth-aware Gaussian splatting method for few-shot novel view synthesis. We use monocular depth prediction as a prior, along with a scale-invariant depth loss, to constrain the 3D shape under just a few input views. We also model color using lower-order spherical harmonics to avoid overfitting. Further, we observe that removing splats with lower opacity periodically, as performed in the original work, leads to a very sparse point cloud and, hence, a lower-quality rendering. To mitigate this, we retain all the splats, leading to a better reconstruction in a few view settings. Experimental results show that our method outperforms the traditional 3D Gaussian splatting methods by achieving improvements of 10.5% in peak signal-to-noise ratio, 6% in structural similarity index, and 14.1% in perceptual similarity, thereby validating the effectiveness of our approach.

3D高斯散射在新视角合成中已超越神经辐射场方法，实现了更低的计算成本和实时高质量渲染。尽管在大量输入视角下能生成高质量的渲染，但当仅有少量视角时，其性能会显著下降。在本工作中，我们提出了一种深度感知的高斯散射方法，专门用于少样本的新视角合成。我们使用单目深度预测作为先验，并结合尺度不变的深度损失来约束在少量输入视角下的3D形状。此外，我们采用低阶球谐函数来建模颜色，以避免过拟合。此外，我们观察到在原始方法中定期移除低不透明度的散点会导致点云过于稀疏，从而降低渲染质量。为了解决这一问题，我们保留了所有的散点，从而在少视角设置下实现了更好的重建。实验结果表明，我们的方法在峰值信噪比（PSNR）上提高了10.5%，结构相似性指数（SSIM）上提高了6%，感知相似性上提高了14.1%，验证了我们方法的有效性。


---

## [85] Scalable Indoor Novel-View Synthesis using Drone-Captured 360 Imagery with 3D Gaussian Splatting

### Scalable Indoor Novel-View Synthesis using Drone-Captured 360 Imagery with 3D Gaussian Splatting

Scene reconstruction and novel-view synthesis for large, complex, multi-story, indoor scenes is a challenging and time-consuming task. Prior methods have utilized drones for data capture and radiance fields for scene reconstruction, both of which present certain challenges. First, in order to capture diverse viewpoints with the drone's front-facing camera, some approaches fly the drone in an unstable zig-zag fashion, which hinders drone-piloting and generates motion blur in the captured data. Secondly, most radiance field methods do not easily scale to arbitrarily large number of images. This paper proposes an efficient and scalable pipeline for indoor novel-view synthesis from drone-captured 360 videos using 3D Gaussian Splatting. 360 cameras capture a wide set of viewpoints, allowing for comprehensive scene capture under a simple straightforward drone trajectory. To scale our method to large scenes, we devise a divide-and-conquer strategy to automatically split the scene into smaller blocks that can be reconstructed individually and in parallel. We also propose a coarse-to-fine alignment strategy to seamlessly match these blocks together to compose the entire scene. Our experiments demonstrate marked improvement in both reconstruction quality, i.e. PSNR and SSIM, and computation time compared to prior approaches.

对于大规模、复杂的多层室内场景，场景重建和新视角合成是一项充满挑战且耗时的任务。以往的方法使用无人机进行数据捕捉和辐射场进行场景重建，但面临一些挑战。首先，为了使用无人机的前置摄像头捕捉多样化的视角，一些方法采用不稳定的Z字形飞行模式，这不仅影响无人机的操作，还会导致捕捉数据时出现运动模糊。其次，大多数辐射场方法难以轻松扩展至任意大量的图像。本论文提出了一种高效且可扩展的管线，利用3D高斯散射从无人机捕捉的360度视频中进行室内新视角合成。360度相机捕捉到了广泛的视角范围，允许在简单直线飞行轨迹下全面捕捉场景。为了使我们的方法能够扩展到大场景，我们设计了一种分而治之的策略，自动将场景划分为可独立并行重建的小块。我们还提出了一种由粗到精的对齐策略，能够无缝匹配这些小块，从而构建整个场景。实验表明，与以往方法相比，我们的方法在重建质量（如PSNR和SSIM）和计算时间上都有显著提升。


---

## [86] ArCSEM: Artistic Colorization of SEM Images via Gaussian Splatting

### ArCSEM: Artistic Colorization of SEM Images via Gaussian Splatting

Scanning Electron Microscopes (SEMs) are widely renowned for their ability to analyze the surface structures of microscopic objects, offering the capability to capture highly detailed, yet only grayscale, images. To create more expressive and realistic illustrations, these images are typically manually colorized by an artist with the support of image editing software. This task becomes highly laborious when multiple images of a scanned object require colorization. We propose facilitating this process by using the underlying 3D structure of the microscopic scene to propagate the color information to all the captured images, from as little as one colorized view. We explore several scene representation techniques and achieve high-quality colorized novel view synthesis of a SEM scene. In contrast to prior work, there is no manual intervention or labelling involved in obtaining the 3D representation. This enables an artist to color a single or few views of a sequence and automatically retrieve a fully colored scene or video.

扫描电子显微镜（SEM）因其分析微观物体表面结构的能力而广受认可，能够捕捉高度精细的图像，但仅限于灰度显示。为了创造更具表现力和真实感的图像，这些图像通常由艺术家借助图像编辑软件手动上色。当需要对同一扫描物体的多张图像进行上色时，此任务显得尤为繁重。我们提出利用微观场景的三维结构，将色彩信息传播到所有捕获的图像中，从而简化这一过程，仅需一个上色视图即可实现。我们探讨了几种场景表示技术，并实现了SEM场景的高质量彩色新视角合成。与以往的工作不同，我们无需手动干预或标签来获得三维表示，从而使艺术家只需对序列中的一个或少数视角进行上色，即可自动获得完整上色的场景或视频。


---

