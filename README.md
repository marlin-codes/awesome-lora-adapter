
# Low-rank Adaptation for Foundation Models: Foundations and Frontiers

## 1. Foundations of LoRA

### a. Parameter Efficiency

**(i) Parameter Decomposition**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY)
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arxiv 2403](https://arxiv.org/abs/2403.13037)
- LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models | [arxiv 2402](https://arxiv.org/pdf/2402.11417.pdf)| [Code](https://github.com/yifanycc/loretta)
- LoTR: Low Tensor Rank Weight Adaptation | [arxiv 2402](https://arxiv.org/pdf/2402.01376.pdf)| [Code](github.com/skolai/lotr)
- Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs | [arxiv 2408](https://arxiv.org/pdf/2408.01008)
- DoRA: Weight-Decomposed Low-Rank Adaptation | [arxiv 2402](https://arxiv.org/pdf/2402.09353.pdf)

**(ii) Parameter Selection**

- SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters | [arxiv 2210](https://arxiv.org/abs/2210.04284)
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arxiv 2311](https://arxiv.org/pdf/2311.11696.pdf)| [Code](https://github.com/TsinghuaC3I/SoRA)
- Asymmetry in Low-Rank Adapters of Foundation Models | [arxiv 2402](https://arxiv.org/abs/2402.16842)
- LoRA-FA: Memory-efficient low-rank adaptation for large language models fine-tuning | [arxiv 2308](https://arxiv.org/pdf/2308.03303.pdf)
- LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation | [arxiv 2402](https://arxiv.org/pdf/2402.07721.pdf)

**(iii) Parameter Sharing**

- VeRA: Vector-based Random Matrix Adaptation | [ICLR 2024](https://openreview.net/forum?id=NjNfLdxr3A)
- Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying ｜[arxiv 2311](https://arxiv.org/pdf/2311.09578)
- NOLA: Networks as linear combination of low rank random basis | [arxiv 2310](https://arxiv.org/pdf/2310.02556.pdf)| [Code](https://github.com/UCDvision/NOLA)
- Delta-LoRA: Fine-tuning high-rank parameters with the delta of low-rank matrices | [arxiv 2309](https://arxiv.org/pdf/2309.02411.pdf)

**(iv) Parameter Quantization**

- QLoRA: Efficient finetuning of quantized llms | [arxiv 2305](https://arxiv.org/pdf/2305.14314.pdf)| [Code](https://github.com/artidoro/qLoRA)
- Qa-LoRA: Quantization-aware low-rank adaptation of large language models | [NeurIPS 2023 (oral)](https://arxiv.org/pdf/2309.14717.pdf)| [Code](https://github.com/yuhuixu1993/qa-LoRA)
- QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning | [arxiv 2402](https://arxiv.org/pdf/2402.10462.pdf)
- Loftq: LoRA-fine-tuning-aware quantization for large language models | [arxiv 2310](https://arxiv.org/pdf/2310.08659.pdf)| [Code](https://github.com/yxli2123/LoftQ)
- Lq-LoRA: Low-rank plus quantized matrix decomposition for efficient language model finetuning | [arxiv 2311](https://arxiv.org/pdf/2311.12023.pdf)| [Code](https://github.com/HanGuo97/lq-LoRA)
- LQER: Low-Rank Quantization Error Reconstruction for LLMs | [arxiv 2402](https://arxiv.org/pdf/2402.02446.pdf)

### b. Ranking Adaptation

**(i) Ranking Refinement**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY)
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arxiv](https://arxiv.org/pdf/2403.13037v1)
- DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation | [EACL](https://arxiv.org/abs/2210.07558)
- PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation | [arxiv 2401](https://arxiv.org/pdf/2401.11316.pdf)
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arxiv 2311](https://arxiv.org/pdf/2311.11696.pdf)| [Code](https://github.com/TsinghuaC3I/SoRA)
  
**(ii) Ranking Augmentation**

- FLoRA: Low-Rank Adapters Are Secretly Gradient Compressors | [arxiv 2402](https://arxiv.org/pdf/2402.03293.pdf)| [Code](https://github.com/MANGA-UOFA/FLoRA)
- Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | [arxiv 2401](https://arxiv.org/pdf/2401.04151.pdf)
- ReLoRA: High-Rank Training Through Low-Rank Updates | [arxiv 2307](https://arxiv.org/pdf/2307.05695.pdf)| [Code](https://github.com/guitaricet/reLoRA)
- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA | [arxiv 2402](https://arxiv.org/abs/2402.16902)
- Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | [arxiv 2402](https://arxiv.org/abs/2402.17263)
- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | [arxiv 2403](https://arxiv.org/abs/2403.03507)
  
### c. Learning Process

#### **(i) Learning Rate**

- LoRA+: Efficient Low-Rank Adaptation of Large Models | [arxiv 2402](https://arxiv.org/pdf/2402.12354.pdf)| [Code](https://github.com/nikhil-ghosh-berkeley/LoRAplus)

#### **(ii) Dropout**

- LoRA Meets Dropout under a Unified Framework

#### **(iii) Scaling Factor**

- A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | [arxiv 2312](https://arxiv.org/pdf/2312.03732.pdf)

#### **(iv) Learning Methods**

- AMAL: Meta Knowledge-Driven Few-Shot Adapter Learning | [ACL 2022](https://aclanthology.org/2022.emnlp-main.709.pdf)

#### **(v) Post-hoc Processing**

- Bayesian Low-rank Adaptation for Large Language Models | [arxiv 2308](https://arxiv.org/abs/2308.13111)

### d. Theoretical Studies

- The Expressive Power of Low-Rank Adaptation | [arxiv 2310](https://arxiv.org/pdf/2310.17513.pdf)| [Code](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA)
- LoRA Training in the NTK Regime has No Spurious Local Minima | [arxiv 2402](https://arxiv.org/pdf/2402.11867.pdf)| [Code](https://github.com/UijeongJang/LoRA-NTK)
- ROSA: Random Orthogonal Subspace Adaptation | [ICML](https://openreview.net/pdf?id=4P9vOFpb63)| [Code](https://github.com/marawangamal/rosa)
- Asymmetry in Low-Rank Adapters of Foundation Models | [arxiv 2402](https://arxiv.org/abs/2402.16842)

## 2. Frontiers of LoRA

### a. Advanced Structures
- Adaptersoup: Weight averaging to improve generalization of pretrained language models | [arxiv2302](https://arxiv.org/pdf/2302.07027)
- LoRAhub: Efficient cross-task generalization via dynamic LoRA composition | [arxiv 2307](https://arxiv.org/pdf/2307.13269.pdf) | [Code](https://github.com/sail-sg/LoRAhub)
- LoRARetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild | [arxiv 2402](https://arxiv.org/pdf/2402.09997.pdf)
- Batched Low-Rank Adaptation of Foundation Models | [arxiv 2312](https://arxiv.org/pdf/2312.05677.pdf)
- Hydra: Multi-head low-rank adaptation for parameter efficient fine-tuning | [arxiv 2309](https://arxiv.org/pdf/2309.06922.pdf)| [Code](https://github.com/extremebird/Hydra)
- One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning | [arxiv 2306](https://arxiv.org/pdf/2306.07967.pdf)| [Code](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA)
- LoRA ensembles for large language model fine-tuning | [arxiv 2310](https://arxiv.org/pdf/2310.00035.pdf)

### b. LoRA MoE

- MoeLoRA: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models | [arxiv 2402](https://arxiv.org/pdf/2402.12851.pdf)
- Higher Layers Need More LoRA Experts | [arxiv 2402](https://arxiv.org/pdf/2402.08562.pdf)| [Code](https://github.com/GCYZSL/MoLA)
- Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning.| [arxiv 2309](https://arxiv.org/abs/2309.05444)
- MOELoRA: An moe-based parameter efficient fine-tuning method for multi-task medical applications | [arxiv 2310](https://arxiv.org/pdf/2310.18339.pdf)| [Code](https://github.com/liuqidong07/MOELoRA-peft)
- LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs | [arxiv 2401](https://arxiv.org/pdf/2401.16160.pdf)
- Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models | [arxiv 2403](https://arxiv.org/pdf/2403.03432)
- Mixture of Cluster-Conditional LoRA Experts for Vision-Language Instruction Tuning | [arxiv 2312](https://arxiv.org/pdf/2312.12379)
- MIXLORA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts | [arxiv2404](https://arxiv.org/pdf/2404.15159)
- LoRAMOE: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment | [arxiv 2312](https://arxiv.org/abs/2312.09979)
- MoRAL: MoE Augmented LoRA for LLMs' Lifelong Learning | [arxiv 2402](https://arxiv.org/pdf/2402.11260)
- Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts [arxiv 2405](https://arxiv.org/abs/2405.11273)
- AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts | [arxiv 2405](https://arxiv.org/abs/2405.00361)
- Mixture of LoRA Experts | [arxiv 2404](https://arxiv.org/abs/2404.13628)

## 3. Applications

### LoRA in NLP

- Machine Translation with Large Language Models: Prompting, Few-shot Learning, and Fine-tuning with QLoRA [ACL](https://aclanthology.org/2023.wmt-1.43.pdf)
- Task-Agnostic Low-Rank Adapters for Unseen English Dialects | [ACL](https://aclanthology.org/2023.emnlp-main.487.pdf)| [Code](https://github.com/zedian/hyperLoRA)
- LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training | [arxiv 2401](https://arxiv.org/pdf/2401.04348.pdf)| [Code](https://github.com/VinAIResearch/LAMPAT)

### LoRA in CV

- Efficient low-rank backpropagation for vision transformer adaptation | [arxiv 2309](https://arxiv.org/pdf/2309.15275.pdf)
- LORTSAR: Low-Rank Transformer for Skeleton-based Action Recognition
- Melo: Low-rank adaptation is better than fine-tuning for medical image diagnosis | [arxiv 2311](https://arxiv.org/pdf/2311.08236.pdf)| [Code](https://github.com/JamesQFreeman/LoRA-ViT)
- FullLoRA-AT: Efficiently Boosting the Robustness of Pretrained Vision Transformers | [arxiv 2401](https://arxiv.org/pdf/2401.01752.pdf)
- Parameter-efficient Model Adaptation for Vision Transformers | [arxiv 2203](https://arxiv.org/pdf/2203.16329.pdf)| [Code](https://github.com/eric-ai-lab/PEViT)
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | [arxiv 2402](https://arxiv.org/pdf/2402.04964.pdf)| [Code](https://github.com/aleemsidra/ConvLoRA)
- Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks | [arxiv 2112](https://arxiv.org/pdf/2112.06825.pdf)| [Code](https://github.com/ylsung/VL_adapter)
- Motion style transfer: Modular low-rank adaptation for deep motion forecasting | [arxiv 2211](https://arxiv.org/pdf/2211.03165.pdf)| [Code](https://github.com/vita-epfl/motion-style-transfer)
- Enhancing General Face Forgery Detection via Vision Transformer with Low-Rank Adaptation | [arxiv 2303](https://arxiv.org/pdf/2303.00917.pdf)
- Customized Segment Anything Model for Medical Image Segmentation | \[arxiv 2]
- Block-wise LoRA: Revisiting Fine-grained LoRA for Effective Personalization and Stylization in Text-to-Image Generation
- AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning
- Lcm-LoRA: A universal stable-diffusion acceleration module | [arxiv 2311](https://arxiv.org/pdf/2311.05556.pdf)| [Code](https://github.com/luosiallen/latent-consistency-model)

### LoRA for Speech

- Low-rank adaptation of large language model rescoring for parameter-efficient speech recognition | [arxiv 2309](https://arxiv.org/pdf/2309.15223.pdf)
- Low-rank Adaptation Method for Wav2vec2-based Fake Audio Detection | [arxiv 2306](https://arxiv.org/pdf/2306.05617.pdf)
- Sparsely Shared LoRA on Whisper for Child Speech Recognition | [arxiv 2309](https://arxiv.org/pdf/2309.11756.pdf)

### LoRA in Code

- LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning | [arxiv 2308](https://arxiv.org/pdf/2308.11148.pdf)
- Task Arithmetic with LoRA for Continual Learning | [arxiv 2311](https://arxiv.org/pdf/2311.02428.pdf)

### LoRA in SCI

- X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design | [APL Machine Learning](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581)
- ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Post Translational Modification | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.abstract)
- Fine-tuning protein language models boosts predictions across diverse tasks | [Nature Communication](https://www.nature.com/articles/s41467-024-51844-2)
- Parameter-efficient fine-tuning on large protein language models improves signal peptide prediction | 

### LoRA in Time series
- Low-rank Adaptation for Spatio-Temporal Forecasting | [arxiv](https://arxiv.org/abs/2404.07919)
- Channel-Aware Low-Rank Adaptation in Time Series Forecasting | [arxiv](https://arxiv.org/pdf/2407.17246)
- Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting | [arxiv](https://arxiv.org/abs/2405.10216)
  
### LoRA in Recommender System

- Customizing Language Models with Instance-wise LoRA for Sequential Recommendation
- Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation
- MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction
- LoRA-NCL: Neighborhood-Enriched Contrastive Learning with Low-Rank Dimensionality Reduction for Graph Collaborative Filtering

### LoRA in Anomaly Detection

- Parameter-Efficient Log Anomaly Detection based on Pre-training model and LoRA [Zenodo](https://zenodo.org/records/8270065)

### LoRA in PDE

- PIHLoRA: Physics-informed hypernetworks for low-ranked adaptation [NeurIPS 2023](https://openreview.net/pdf?id=kupYlLLGdf)

### LoRA in RL

- Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent | [arxiv 2402](https://arxiv.org/pdf/2402.13717.pdf)| [Code](https://github.com/weiyifan1023/Neeko)

### LoRA for Federated Learning

- SLoRA: Federated parameter efficient fine-tuning of language models | [arxiv 2308](https://arxiv.org/pdf/2308.06522.pdf)
- pFedLoRA: Model-heterogeneous personalized federated learning with LoRA tuning | [arxiv 2310](https://arxiv.org/pdf/2310.13283.pdf)
- Improving LoRA in Privacy-preserving Federated Learning [OpenReview](https://openreview.net/pdf?id=NLPzL6HWNl)
- Heterogeneous Low-Rank Approximation for Federated Fine-tuning of On-Device Foundation Models | [arxiv 2401](https://arxiv.org/pdf/2401.06432.pdf)
- OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning ｜ [arxiv 2402](https://arxiv.org/pdf/2402.06954.pdf)| [Code](https://github.com/rui-ye/OpenFedLLM)
- Federatedscope-llm: A comprehensive package for fine-tuning large language models in federated learning | [arxiv 2309](https://arxiv.org/abs/2309.00363)
- FedHLT: Efficient Federated Low-Rank Adaption with Hierarchical Language Tree for Multilingual Modeling | 
- FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning [arxiv 2404](https://arxiv.org/abs/2404.15182)
- FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering | [arxiv 2404](https://arxiv.org/abs/2404.15384)
- DP-DyLoRA: Fine-Tuning Transformer-Based Models On-Device under Differentially Private Federated Learning using Dynamic Low-Rank Adaptation | [arxiv 2405](https://arxiv.org/abs/2405.06368)

### LoRA for Multi-Task Learning

- MultiLoRA: Democratizing LoRA for Better Multi-Task Learning | [arxiv 2311](https://arxiv.org/pdf/2311.11501.pdf)
  

### LoRA for Long sequence learning

- LongLoRA: Efficient fine-tuning of long-context large language models | [arxiv 2309](https://arxiv.org/pdf/2309.12307.pdf)| [Code](https://github.com/dvlab-research/LongLoRA)
- LongqLoRA: Efficient and effective method to extend context length of large language models | [arxiv 2311](https://arxiv.org/pdf/2311.04879.pdf)| [Code](https://github.com/yangjianxin1/LongQLoRA)
- With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation | [arxiv 2401](https://arxiv.org/abs/2401.11504)
- RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization | [arxiv 2405](https://arxiv.org/abs/2405.00657)

### LoRA for Pretraining

- Training Neural Networks from Scratch with Parallel Low-Rank Adapters | [arxiv 2402](https://arxiv.org/pdf/2402.16828.pdf)| [Code](https://github.com/minyoungg/LTE)

### Deployment of LoRA Adapters

- S-LoRA: Serving thousands of concurrent LoRA adapters | [arxiv 2311](https://arxiv.org/pdf/2311.03285.pdf)| [Code](https://github.com/S-LoRA/S-LoRA)
- CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference | [arxiv 2401](https://arxiv.org/pdf/2401.11240.pdf)
- Local LoRA: Memory-Efficient Fine-Tuning of Large Language Models | [OpenReview](https://openreview.net/pdf?id=LHKmzWP7RN#:~:text=Our%20approach%20aims%20to%20decouple,LoRA%20on%20math%20reasoning%20tasks.)
- Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | [arxiv 2401](https://arxiv.org/pdf/2401.04151.pdf)
  
## 4. Resource

- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models | [arxiv 2304](https://arxiv.org/pdf/2304.01933.pdf)
- Run LoRA Run: Faster and Lighter LoRA Implementations | [arxiv 2312](https://arxiv.org/pdf/2312.03415.pdf)
- Large language model LoRA specifically fine-tuned for medical domain tasks | [Code](https://huggingface.co/nmitchko/medfalcon-40b-LoRA)
