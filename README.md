
# Low-rank Adaptation for Foundation Models: Foundations and Frontiers

## 1. Foundations of LoRA

### a. Parameter Efficiency

**(i) Parameter Decomposition**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY)
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arxiv 2403](https://arxiv.org/abs/2403.13037)
- LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models | [arxiv 2402](https://arxiv.org/pdf/2402.11417.pdf)| [Code](https://github.com/yifanycc/loretta) | NAACL 2024 Oral
- LoTR: Low Tensor Rank Weight Adaptation | [arxiv 2402](https://arxiv.org/pdf/2402.01376.pdf)| [Code](https://github.com/daskol/lotr)
- Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs | [arxiv 2408](https://arxiv.org/pdf/2408.01008)
- DoRA: Weight-Decomposed Low-Rank Adaptation | [arxiv 2402](https://arxiv.org/pdf/2402.09353.pdf) | [Code](https://github.com/NVlabs/DoRA) | ICML 2024

**(ii) Parameter Selection**

- SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters | [arxiv 2210](https://arxiv.org/abs/2210.04284) | [Code](https://github.com/Shwai-He/SparseAdapter) | Findings of EMNLP 2022
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arxiv 2311](https://arxiv.org/pdf/2311.11696.pdf)| [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023
- Asymmetry in Low-Rank Adapters of Foundation Models | [arxiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA?utm_source=catalyzex.com) [Code](https://github.com/NVIDIA/NeMo/tree/adithyare/vera) | 
- LoRA-FA: Memory-efficient low-rank adaptation for large language models fine-tuning | [arxiv 2308](https://arxiv.org/pdf/2308.03303.pdf)
- LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation | [arxiv 2402](https://arxiv.org/pdf/2402.07721.pdf)

**(iii) Parameter Sharing**

- VeRA: Vector-based Random Matrix Adaptation | [ICLR 2024](https://openreview.net/forum?id=NjNfLdxr3A)
- Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying ｜[arxiv 2311](https://arxiv.org/pdf/2311.09578)
- NOLA: Networks as linear combination of low rank random basis | [arxiv 2310](https://arxiv.org/pdf/2310.02556.pdf)| [Code](https://github.com/UCDvision/NOLA) | [Code](https://github.com/UCDvision/NOLA) | ICLR 2024
- Delta-LoRA: Fine-tuning high-rank parameters with the delta of low-rank matrices | [arxiv 2309](https://arxiv.org/pdf/2309.02411.pdf)

**(iv) Parameter Quantization**

- QLoRA: Efficient finetuning of quantized llms | [arxiv 2305](https://arxiv.org/pdf/2305.14314.pdf)| [Code](https://github.com/artidoro/qLoRA) | NeurIPS 2023
- Qa-LoRA: Quantization-aware low-rank adaptation of large language models | [NeurIPS 2023 (oral)](https://arxiv.org/pdf/2309.14717.pdf)| [Code](https://github.com/yuhuixu1993/qa-LoRA)
- QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning | [arxiv 2402](https://arxiv.org/pdf/2402.10462.pdf)
- Loftq: LoRA-fine-tuning-aware quantization for large language models | [arxiv 2310](https://arxiv.org/pdf/2310.08659.pdf)| [Code](https://github.com/yxli2123/LoftQ)
- Lq-LoRA: Low-rank plus quantized matrix decomposition for efficient language model finetuning | [arxiv 2311](https://arxiv.org/pdf/2311.12023.pdf)| [Code](https://github.com/HanGuo97/lq-LoRA)
- LQER: Low-Rank Quantization Error Reconstruction for LLMs | [arxiv 2402](https://arxiv.org/pdf/2402.02446.pdf) | [Code](https://github.com/OpenGVLab/OmniQuant?utm_source=catalyzex.com) | ICLR 2024

### b. Ranking Adaptation

**(i) Ranking Refinement**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY)
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arxiv](https://arxiv.org/pdf/2403.13037v1)
- DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation | [EACL](https://arxiv.org/abs/2210.07558) | [Code](https://github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA?utm_source=catalyzex.com)
- PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation | [arxiv 2401](https://arxiv.org/pdf/2401.11316.pdf)
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arxiv 2311](https://arxiv.org/pdf/2311.11696.pdf) | [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023
  
**(ii) Ranking Augmentation**

- FLoRA: Low-Rank Adapters Are Secretly Gradient Compressors | [arxiv 2402](https://arxiv.org/pdf/2402.03293.pdf)| [Code](https://github.com/MANGA-UOFA/FLoRA) | ICML 2024
- Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | [arxiv 2401](https://arxiv.org/pdf/2401.04151.pdf) | ICML 2024
- ReLoRA: High-Rank Training Through Low-Rank Updates | [arxiv 2307](https://arxiv.org/pdf/2307.05695.pdf)| [Code](https://github.com/guitaricet/reLoRA)
- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA | [arxiv 2402](https://arxiv.org/abs/2402.16902) | [Code](https://github.com/sahil280114/codealpaca?utm_source=catalyzex.com) 
- Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | [arxiv 2402](https://arxiv.org/abs/2402.17263) | ACL 2024
- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | [arxiv 2403](https://arxiv.org/abs/2403.03507) | [Code](https://github.com/jiaweizzhao/GaLore) | Oral ICML 2024
  
### c. Learning Process

#### **(i) Learning Rate**

- LoRA+: Efficient Low-Rank Adaptation of Large Models | [arxiv 2402](https://arxiv.org/pdf/2402.12354.pdf)| [Code](https://github.com/nikhil-ghosh-berkeley/LoRAplus) | ICML 2024

#### **(ii) Dropout**

- LoRA Meets Dropout under a Unified Framework [arxiv 2403](https://arxiv.org/pdf/2403.00812)

#### **(iii) Scaling Factor**

- A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | [arxiv 2312](https://arxiv.org/pdf/2312.03732.pdf) | [Code](https://github.com/kingoflolz/mesh-transformer-jax)

#### **(iv) Learning Methods**

- AMAL: Meta Knowledge-Driven Few-Shot Adapter Learning | [ACL 2022](https://aclanthology.org/2022.emnlp-main.709.pdf)

#### **(v) Post-hoc Processing**

- Bayesian Low-rank Adaptation for Large Language Models | [arxiv 2308](https://arxiv.org/abs/2308.13111) | [Code](https://github.com/adamxyang/laplace-lora) | ICLR 2024

### d. Theoretical Foundations

- The Expressive Power of Low-Rank Adaptation | [arxiv 2310](https://arxiv.org/pdf/2310.17513.pdf) | [Code](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) | ICLR 2024
- LoRA Training in the NTK Regime has No Spurious Local Minima | [arxiv 2402](https://arxiv.org/pdf/2402.11867.pdf)| [Code](https://github.com/UijeongJang/LoRA-NTK) | ICML 2024
- ROSA: Random Orthogonal Subspace Adaptation | [ICML 2023](https://openreview.net/pdf?id=4P9vOFpb63) | [Code](https://github.com/marawangamal/rosa)
- Asymmetry in Low-Rank Adapters of Foundation Models | [arxiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA?utm_source=catalyzex.com)

## 2. Frontiers of LoRA

### a. Advanced Structures

**LoRA Composition**
- Adaptersoup: Weight averaging to improve generalization of pretrained language models | [arxiv2302](https://arxiv.org/pdf/2302.07027) | [Code](https://github.com/UKPLab/sentence-transformers)
- LoRAhub: Efficient cross-task generalization via dynamic LoRA composition | [arxiv 2307](https://arxiv.org/pdf/2307.13269.pdf) | [Code](https://github.com/sail-sg/LoRAhub) | COLM 2024
- LoRARetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild | [arxiv 2402](https://arxiv.org/pdf/2402.09997.pdf) | [Code](https://github.com/tatsu-lab/stanford_alpaca) 
- Batched Low-Rank Adaptation of Foundation Models | [arxiv 2312](https://arxiv.org/pdf/2312.05677.pdf) | [Code](https://github.com/huggingface/peft/tree/main)
- Hydra: Multi-head low-rank adaptation for parameter efficient fine-tuning | [arxiv 2309](https://arxiv.org/pdf/2309.06922.pdf)| [Code](https://github.com/extremebird/Hydra)
- One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning | [arxiv 2306](https://arxiv.org/pdf/2306.07967.pdf)| [Code](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA)
- LoRA ensembles for large language model fine-tuning | [arxiv 2310](https://arxiv.org/pdf/2310.00035.pdf) | [Code](https://github.com/huggingface/peft?utm_source)  
- MultiLoRA: Democratizing LoRA for Better Multi-Task Learning | [arxiv 2311](https://arxiv.org/pdf/2311.11501.pdf)

**LoRA MoE**
- MoeLoRA: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models | [arxiv 2402](https://arxiv.org/pdf/2402.12851.pdf)
- Higher Layers Need More LoRA Experts | [arxiv 2402](https://arxiv.org/pdf/2402.08562.pdf)| [Code](https://github.com/GCYZSL/MoLA)
- Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning.| [arxiv 2309](https://arxiv.org/abs/2309.05444) | [Code](https://github.com/for-ai/parameter-efficient-moe) 
- MOELoRA: An moe-based parameter efficient fine-tuning method for multi-task medical applications | [arxiv 2310](https://arxiv.org/pdf/2310.18339.pdf)| [Code](https://github.com/liuqidong07/MOELoRA-peft) | SIGIR 24
- LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs | [arxiv 2401](https://arxiv.org/pdf/2401.16160.pdf)
- Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models | [arxiv 2403](https://arxiv.org/pdf/2403.03432)
- Mixture of Cluster-Conditional LoRA Experts for Vision-Language Instruction Tuning | [arxiv 2312](https://arxiv.org/pdf/2312.12379) | [Code](https://github.com/gyhdog99/mocle) 
- MIXLORA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts | [arxiv2404](https://arxiv.org/pdf/2404.15159) | [Code](https://github.com/TUDB-Labs/MixLoRA)
- LoRAMOE: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment | [arxiv 2312](https://arxiv.org/abs/2312.09979) | [Code](https://github.com/Ablustrund/LoRAMoE) |
- MoRAL: MoE Augmented LoRA for LLMs' Lifelong Learning | [arxiv 2402](https://arxiv.org/pdf/2402.11260)
- Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts [arxiv 2405](https://arxiv.org/abs/2405.11273) | [Code](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs) | JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8
- AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts | [arxiv 2405](https://arxiv.org/abs/2405.00361) | [Code](https://github.com/zefang-liu/AdaMoLE) | COLM 2024
- Mixture of LoRA Experts | [arxiv 2404](https://arxiv.org/abs/2404.13628) |  [Code](https://github.com/yushuiwx/MoLE) | ICLR 2024

### b. LoRA for Long Sequence Modeling

- LongLoRA: Efficient fine-tuning of long-context large language models | [arxiv 2309](https://arxiv.org/pdf/2309.12307.pdf)| [Code](https://github.com/dvlab-research/LongLoRA) | ICLR 2024 Oral
- LongqLoRA: Efficient and effective method to extend context length of large language models | [arxiv 2311](https://arxiv.org/pdf/2311.04879.pdf)| [Code](https://github.com/yangjianxin1/LongQLoRA)
- With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation | [arxiv 2401](https://arxiv.org/abs/2401.11504) | [Code](https://github.com/TemporaryLoRA/Temp-LoRA/tree/main) | COLM 2024
- RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization | [arxiv 2405](https://arxiv.org/abs/2405.00657)

### c. LoRA for Continue Learning
- Orthogonal Subspace Learning for Language Model Continual Learning
- Continual Learning with Low Rank Adaptation
- Task Arithmetic with LoRA for Continual Learning
- A Unified Continual Learning Framework with General Parameter-Efficient Tuning
  
### d. LoRA for Federated Learning

- SLoRA: Federated parameter efficient fine-tuning of language models | [arxiv 2308](https://arxiv.org/pdf/2308.06522.pdf) | 
- pFedLoRA: Model-heterogeneous personalized federated learning with LoRA tuning | [arxiv 2310](https://arxiv.org/pdf/2310.13283.pdf)
- Improving LoRA in Privacy-preserving Federated Learning [OpenReview](https://openreview.net/pdf?id=NLPzL6HWNl) | ICLR 2024
- Heterogeneous Low-Rank Approximation for Federated Fine-tuning of On-Device Foundation Models | [arxiv 2401](https://arxiv.org/pdf/2401.06432.pdf) 
- OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning ｜ [arxiv 2402](https://arxiv.org/pdf/2402.06954.pdf)| [Code](https://github.com/rui-ye/OpenFedLLM) 
- Federatedscope-llm: A comprehensive package for fine-tuning large language models in federated learning | [arxiv 2309](https://arxiv.org/abs/2309.00363) | [Code](https://github.com/alibaba/FederatedScope/tree/llm) 
- FedHLT: Efficient Federated Low-Rank Adaption with Hierarchical Language Tree for Multilingual Modeling | [acm](https://dl.acm.org/doi/pdf/10.1145/3589335.3651933)
- FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning [arxiv 2404](https://arxiv.org/abs/2404.15182)
- FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering | [arxiv 2404](https://arxiv.org/abs/2404.15384) | ICLR 2024
- DP-DyLoRA: Fine-Tuning Transformer-Based Models On-Device under Differentially Private Federated Learning using Dynamic Low-Rank Adaptation | [arxiv 2405](https://arxiv.org/abs/2405.06368)
- FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning | [arxiv 2406](https://arxiv.org/pdf/2406.07925)
- FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations | [arxiv 2409](https://arxiv.org/pdf/2409.05976) [Code](https://github.com/ATP-1010/FederatedLLM) 

## 3. Applications

### LoRA in Natural Language Processing

- Machine Translation with Large Language Models: Prompting, Few-shot Learning, and Fine-tuning with QLoRA [ACL](https://aclanthology.org/2023.wmt-1.43.pdf)
- Task-Agnostic Low-Rank Adapters for Unseen English Dialects | [ACL](https://aclanthology.org/2023.emnlp-main.487.pdf)| [Code](https://github.com/zedian/hyperLoRA)
- LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training | [arxiv 2401](https://arxiv.org/pdf/2401.04348.pdf)| [Code](https://github.com/VinAIResearch/LAMPAT) | AAAI 2024
- Task Arithmetic with LoRA for Continual Learning | [arxiv 2311](https://arxiv.org/pdf/2311.02428.pdf) | Neurips 2023 Workshop

### LoRA in Computer Vision

**a. Visual Understanding**

(1) Domain Adaptation and Transfer Learning
- Motion style transfer: Modular low-rank adaptation for deep motion forecasting | [arxiv 2211](https://arxiv.org/pdf/2211.03165.pdf)| [Code](https://github.com/vita-epfl/motion-style-transfer)
- Efficient low-rank backpropagation for vision transformer adaptation | [arxiv 2309](https://arxiv.org/pdf/2309.15275.pdf) | NeurIPS 20223
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | [arxiv 2402](https://arxiv.org/pdf/2402.04964.pdf) | [Code](https://github.com/aleemsidra/ConvLoRA)
- ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | [arxiv 2406](https://arxiv.org/abs/2406.10973)
- Melo: Low-rank adaptation is better than fine-tuning for medical image diagnosis | [arxiv 2311](https://arxiv.org/pdf/2311.08236.pdf)| [Code](https://github.com/JamesQFreeman/LoRA-ViT)
- Enhancing General Face Forgery Detection via Vision Transformer with Low-Rank Adaptation | [arxiv 2303](https://arxiv.org/pdf/2303.00917.pdf)

(2) Semantic Segmentation
- Customized Segment Anything Model for Medical Image Segmentation | [arxiv 2304](https://arxiv.org/abs/2304.13785) | [Code](https://github.com/hitachinsk/SAMed) 
- SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation | [MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_23)
- Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model | [arxiv 2401](https://arxiv.org/abs/2401.17868)


(3) Others
- FullLoRA-AT: Efficiently Boosting the Robustness of Pretrained Vision Transformers | [arxiv 2401](https://arxiv.org/pdf/2401.01752.pdf)
- Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach | [arxiv 2403](https://arxiv.org/abs/2403.19067)
- LORTSAR: Low-Rank Transformer for Skeleton-based Action Recognition | [arxiv 2407](https://arxiv.org/abs/2407.14655)
- Parameter-efficient Model Adaptation for Vision Transformers | [arxiv 2203](https://arxiv.org/pdf/2203.16329.pdf)| [Code](https://github.com/eric-ai-lab/PEViT) | AAAI 2023

  
**b. Visual Generation**
- Cones: Concept Neurons in Diffusion Models for Customized Generation | [arxiv 2303](https://arxiv.org/abs/2303.05125)
- Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models | [arxiv 2305](https://arxiv.org/abs/2305.18292)
- Generating coherent comic with rich story using ChatGPT and Stable Diffusion | [arxiv 2305](https://arxiv.org/abs/2305.11067)
- Cones 2: Customizable Image Synthesis with Multiple Subjects | [arxiv 2305](https://arxiv.org/abs/2305.19327)
- StyleAdapter: A Single-Pass LoRA-Free Model for Stylized Image Generation | [arxiv 2309](https://arxiv.org/abs/2309.01770)
- ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs | [arxiv 2311](https://arxiv.org/abs/2311.13600)
- Intrinsic LoRA: A Generalist Approach for Discovering Knowledge in Generative Models | [arxiv 2311](https://arxiv.org/abs/2311.17137)
- Lcm-LoRA: A universal stable-diffusion acceleration module | [arxiv 2311](https://arxiv.org/pdf/2311.05556.pdf)| [Code](https://github.com/luosiallen/latent-consistency-model)
- Continual Diffusion with STAMINA: STack-And-Mask INcremental Adapters | [arxiv 2311](https://arxiv.org/abs/2311.18763) 
- Orthogonal Adaptation for Modular Customization of Diffusion Models | [arxiv 2312](https://arxiv.org/abs/2312.02432)
- Style Transfer to Calvin and Hobbes comics using Stable Diffusion | [arxiv 2312](https://arxiv.org/abs/2312.03993)
- Lora-enhanced distillation on guided diffusion models | [arxiv 2312](https://arxiv.org/pdf/2312.06899)
- Multi-LoRA Composition for Image Generation | [arxiv 2402](https://arxiv.org/abs/2402.16843)
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | [arxiv 2402](https://arxiv.org/pdf/2402.04964.pdf)| [Code](https://github.com/aleemsidra/ConvLoRA) | IEEE ISBI 2024
- LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models | [arxiv 2403](https://arxiv.org/abs/2403.11627)
- Resadapter: Domain consistent resolution adapter for diffusion models | [arxiv 2403](https://arxiv.org/abs/2403.02084)
- Implicit Style-Content Separation using B-LoRA | [arxiv 2403](https://arxiv.org/abs/2403.14572)
- Mixture of Low-rank Experts for Transferable AI-Generated Image Detection | [arxiv 2404](https://arxiv.org/abs/2404.04883)
- MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection | [arxiv 2404](https://arxiv.org/abs/2404.08452)
- Low-Rank Few-Shot Adaptation of Vision-Language Models | [arxiv 2405](https://arxiv.org/abs/2405.18541)
- FouRA: Fourier Low Rank Adaptation | [arxiv 2406](https://arxiv.org/abs/2406.08798)




### LoRA in Multimodal Learning
- Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks | [arxiv 2112](https://arxiv.org/pdf/2112.06825.pdf)| [Code](https://github.com/ylsung/VL_adapter) | CVPR 2022
- DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback | [arxiv 2311](https://arxiv.org/abs/2311.17946)
- Block-wise LoRA: Revisiting Fine-grained LoRA for Effective Personalization and Stylization in Text-to-Image Generation | [arxiv 2304](https://arxiv.org/pdf/2403.07500) | AAAI 2024
- AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning | [arxiv 2307](https://arxiv.org/pdf/2307.04725) | [Code](https://github.com/guoyww/AnimateDiff) | ICLR 2024
- Multi-Concept Customization of Text-to-Image Diffusion
- SELMA: Learning and Merging Skill-Specific Text-to-Image Experts with Auto-Generated Data
- MACE: Mass Concept Erasure in Diffusion Models
- AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models
- MoVA: Adapting Mixture of Vision Experts to Multimodal Context
- LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation
- Customizing 360-degree panoramas through text-to-image diffusion models. 
- Space narrative: Generating images and 3d scenes of chinese garden from text using deep learning.


### LoRA in Speech Processing

- Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition | [arxiv 2309](https://arxiv.org/pdf/2309.15223.pdf)
- Low-rank Adaptation Method for Wav2vec2-based Fake Audio Detection | [arxiv 2306](https://arxiv.org/pdf/2306.05617.pdf) | CEUR Workshop
- Sparsely Shared LoRA on Whisper for Child Speech Recognition | [arxiv 2309](https://arxiv.org/pdf/2309.11756.pdf) | [Code](https://github.com/huggingface/peft)

### LoRA in Code Engineering

- LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning | [arxiv 2308](https://arxiv.org/pdf/2308.11148.pdf)
- RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair | [arxiv 2312](https://arxiv.org/abs/2312.15698) | [Code](https://repairllama.github.io)
- MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair [arxiv 2408](https://arxiv.org/pdf/2408.09568)


### LoRA in Scientific Discovery

- X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design | [APL Machine Learning](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581)
- ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Post Translational Modification | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.abstract)
- Fine-tuning protein language models boosts predictions across diverse tasks | [Nature Communication](https://www.nature.com/articles/s41467-024-51844-2)
- Parameter-efficient fine-tuning on large protein language models improves signal peptide prediction | [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.04.565642v1)
- Prollama: A protein large language model for multi-task protein language processing | [arxiv 2402](https://arxiv.org/pdf/2402.16445)

### LoRA in Time Series
- Low-rank Adaptation for Spatio-Temporal Forecasting | [arxiv](https://arxiv.org/abs/2404.07919) [Code](https://github.com/RWLinno/ST-LoRA) | 
- Channel-Aware Low-Rank Adaptation in Time Series Forecasting | [arxiv](https://arxiv.org/pdf/2407.17246) | [Code](https://github.com/tongnie/C-LoRA)
- Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting | [arxiv](https://arxiv.org/abs/2405.10216)

### LoRA in Graph Learning
- GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning [arxiv 2409](https://arxiv.org/pdf/2409.16670) 
- Fast and Continual Knowledge Graph Embedding via Incremental LoRA | [arxiv 2407](https://arxiv.org/pdf/2407.05705) | [Code](https://github.com/seukgcode/FastKGE) | IJCAI2024
  
### LoRA in Recommender System

- Customizing Language Models with Instance-wise LoRA for Sequential Recommendation | [arxiv 2408](https://arxiv.org/pdf/2408.10159) 
- Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation | [arxiv 2408](https://arxiv.org/pdf/2408.03533) 
- MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction | [arxiv 2408](https://arxiv.org/pdf/2408.08913) | [Code](https://github.com/gaohaining/MLoRA) | [arxiv 2409](https://arxiv.org/pdf/2409.08543)
- ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Model | [mdpi](https://www.mdpi.com/2227-7390/11/16/3577) 
- LoRA-NCL: Neighborhood-Enriched Contrastive Learning with Low-Rank Dimensionality Reduction for Graph Collaborative Filtering | [arxiv 2403](https://arxiv.org/pdf/2403.13325) | [Code](https://github.com/zhengzhi-1997/LLM-TRSR) | WWW2024
- LoRA for Sequential Recommendation Harnessing large language models for text-rich sequential recommendation

### LoRA in Anomaly Detection

- Parameter-Efficient Log Anomaly Detection based on Pre-training model and LoRA [Zenodo](https://zenodo.org/records/8270065)

### LoRA in PDE

- PIHLoRA: Physics-informed hypernetworks for low-ranked adaptation [NeurIPS 2023](https://openreview.net/pdf?id=kupYlLLGdf)

### LoRA in RL

- Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent | [arxiv 2402](https://arxiv.org/pdf/2402.13717.pdf)| [Code](https://github.com/weiyifan1023/Neeko)
- Handling coexistence of LoRA with other networks through embedded reinforcement learning [ACM](https://dl.acm.org/doi/abs/10.1145/3576842.3582383)

### LoRA for Pretraining

- Training Neural Networks from Scratch with Parallel Low-Rank Adapters | [arxiv 2402](https://arxiv.org/pdf/2402.16828.pdf)| [Code](https://github.com/minyoungg/LTE)

### Deployment of LoRA Adapters

- S-LoRA: Serving thousands of concurrent LoRA adapters | [arxiv 2311](https://arxiv.org/pdf/2311.03285.pdf)| [Code](https://github.com/S-LoRA/S-LoRA) | MLSys Conference 2024
- CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference | [arxiv 2401](https://arxiv.org/pdf/2401.11240.pdf)
- Local LoRA: Memory-Efficient Fine-Tuning of Large Language Models | [OpenReview](https://openreview.net/pdf?i d=LHKmzWP7RN#:~:text=Our%20approach%20aims%20to%20decouple,LoRA%20on%20math%20reasoning%20tasks.) | Neurips 2023 Workshop
- Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | [arxiv 2401](https://arxiv.org/pdf/2401.04151.pdf) | ICML 2024
  
## 4. Resource

- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models | [arxiv 2304](https://arxiv.org/pdf/2304.01933.pdf) | [Code](https://github.com/AGI-Edgerunners/LLM-Adapters) | 
- Run LoRA Run: Faster and Lighter LoRA Implementations | [arxiv 2312](https://arxiv.org/pdf/2312.03415.pdf)
- Large language model LoRA specifically fine-tuned for medical domain tasks | [Code](https://huggingface.co/nmitchko/medfalcon-40b-LoRA)
