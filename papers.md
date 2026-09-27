# LoRA Adapter Papers

This document contains detailed paper listings organized by mechanism, adaptation setting, and application domain. The list focuses on low-rank adaptation methods and adapter methods that directly compose with or substitute for LoRA. Prompt or prefix tuning is included only when low-rank structure is central to the contribution. For the overview, curated recent venue updates, and repository entry point, see [README.md](README.md).

If a paper spans multiple themes, it appears once under its primary contribution: mechanism for a new parameterization or training method, setting for a deployment or adaptation regime, and domain for a primarily application-specific contribution.

## Table of Contents

1. [Foundations and Core Mechanisms](#1-foundations-and-core-mechanisms)

   - [a. Parameter Efficiency and Structural Design](#a-parameter-efficiency-and-structural-design)
   - [b. Rank Design and Capacity Scaling](#b-rank-design-and-capacity-scaling)
   - [c. Optimization, Initialization, and Training Dynamics](#c-optimization-initialization-and-training-dynamics)
   - [d. Theory and Analysis](#d-theory-and-analysis)
2. [Adaptation Settings and Systems](#2-adaptation-settings-and-systems)

   - [a. Composition, Routing, and Structural Extensions](#a-composition-routing-and-structural-extensions)
   - [b. Long-Context and Sequence Modeling](#b-long-context-and-sequence-modeling)
   - [c. Continual and Lifelong Adaptation](#c-continual-and-lifelong-adaptation)
   - [d. Federated and Distributed Adaptation](#d-federated-and-distributed-adaptation)
   - [e. Pretraining and Full Training](#e-pretraining-and-full-training)
   - [f. Serving and Systems](#f-serving-and-systems)
   - [g. Privacy, Security, and Attacks](#g-privacy-security-and-attacks)
3. [Domains and Modalities](#3-domains-and-modalities)

   - [a. Language and NLP](#a-language-and-nlp)
   - [b. Vision and Generative Vision](#b-vision-and-generative-vision)
   - [c. Multimodal and Vision-Language](#c-multimodal-and-vision-language)
   - [d. Speech and Audio](#d-speech-and-audio)
   - [e. Code and Software Engineering](#e-code-and-software-engineering)
   - [f. Scientific, Biomedical, and Physics](#f-scientific-biomedical-and-physics)
   - [g. Structured Data: Graphs and Recommendation](#g-structured-data-graphs-and-recommendation)
   - [h. Time Series and Forecasting](#h-time-series-and-forecasting)
   - [i. Emerging Applications](#i-emerging-applications)
4. [Resource](#4-resource)

## 1. Foundations and Core Mechanisms

### a. Parameter Efficiency and Structural Design

**(i) Parameter Decomposition**

- LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models | [arXiv 2402](https://arxiv.org/pdf/2402.11417.pdf) | [Code](https://github.com/yifanycc/loretta) | NAACL 2024 \
  Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang
- LoTR: Low Tensor Rank Weight Adaptation | [arXiv 2402](https://arxiv.org/pdf/2402.01376.pdf) | [Code](https://github.com/daskol/lotr) \
  Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, Ivan Oseledets
- Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs | [arXiv 2408](https://arxiv.org/pdf/2408.01008) \
  Afia Anjum, Maksim E. Eren, Ismael Boureima, Boian Alexandrov, Manish Bhattarai
- DoRA: Weight-Decomposed Low-Rank Adaptation | [arXiv 2402](https://arxiv.org/pdf/2402.09353.pdf) | [Code](https://github.com/NVlabs/DoRA) | ICML 2024 \
  Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen
- LORA-CRAFT: Cross-layer Rank Adaptation via Frozen Tucker Decomposition of Pre-trained Attention Weights | [COLM 2026](https://colmweb.org/AcceptedPapers.html) \
  Kasun Dewage, Marianna Pensky, Suranadi De Silva, Shankhadeep Mondal

**(ii) Parameter Selection**

- SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters | [Findings of EMNLP 2022](https://arxiv.org/abs/2210.04284) | [Code](https://github.com/Shwai-He/SparseAdapter) \
  Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arXiv 2311](https://arxiv.org/pdf/2311.11696.pdf) | [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023 \
  Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun
- LoRA-FA: Memory-efficient low-rank adaptation for large language models fine-tuning | [arXiv 2308](https://arxiv.org/pdf/2308.03303.pdf) \
  Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, Bo Li
- LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation | [arXiv 2402](https://arxiv.org/pdf/2402.07721.pdf) \
  Hongyun Zhou, Xiangyu Lu, Wang Xu, Conghui Zhu, Tiejun Zhao, Muyun Yang
- WeightLoRA: Keep Only Necessary Adapters | [ACL 2026](https://aclanthology.org/2026.acl-long.566/) \
  Andrey Veprikov, Vladimir Solodkin, Zyl Alexander, Andrey Savchenko, Aleksandr Beznosikov
- Localized Low-Rank Adaptation within Clustered Parameter Subspaces | [ACL 2026](https://aclanthology.org/2026.acl-long.1223/) \
  Jiahao Xiong, Yihe Liu, Xianming Hu, Hongbo Zhao, Nuoyi Chen, Jie Zhang, Kai Zhang

**(iii) Parameter Sharing**

- VeRA: Vector-based Random Matrix Adaptation | [ICLR 2024](https://openreview.net/forum?id=NjNfLdxr3A) \
  Dawid J. Kopiczko, Tijmen Blankevoort, Yuki M. Asano
- Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying | [arXiv 2311](https://arxiv.org/pdf/2311.09578) \
  Adithya Renduchintala, Tugrul Konuk, Oleksii Kuchaiev
- NOLA: Networks as linear combination of low rank random basis | [arXiv 2310](https://arxiv.org/pdf/2310.02556.pdf) | [Code](https://github.com/UCDvision/NOLA) | ICLR 2024 \
  Soroush Abbasi Koohpayegani, KL Navaneet, Parsa Nooralinejad, Soheil Kolouri, Hamed Pirsiavash
- Delta-LoRA: Fine-tuning high-rank parameters with the delta of low-rank matrices | [arXiv 2309](https://arxiv.org/pdf/2309.02411.pdf) \
  Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, Lei Zhang
- Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA ｜ [ICLR 2025](https://openreview.net/forum?id=WwpYSOkkCt) \
  Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, Tal Schuster
- RaSA: Rank-Sharing Low-Rank Adaptation ｜[ICLR 2025](https://openreview.net/forum?id=GdXI5zCoAt) | [Code](https://github.com/zwhe99/RaSA) \
  Zhiwei He, Zhaopeng Tu, Xing Wang, Xingyu Chen, Zhijie Wang, Jiahao Xu, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Rui Wang
- E²LoRA: Efficient and Effective Low-Rank Adaptation with Entropy-Guided Adaptive Sharing | [OpenReview](https://openreview.net/forum?id=IQttyo0460) | ICLR 2026 \
  Minglei Li, Peng Ye, Jingqi Ye, Haonan He, Tao Chen

**(iv) Parameter Quantization**

- QLoRA: Efficient finetuning of quantized llms | [arXiv 2305](https://arxiv.org/pdf/2305.14314.pdf) | [Code](https://github.com/artidoro/qLoRA) | NeurIPS 2023 \
  Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- Qa-LoRA: Quantization-aware low-rank adaptation of large language models | [NeurIPS 2023](https://arxiv.org/pdf/2309.14717.pdf) | [Code](https://github.com/yuhuixu1993/qa-LoRA) \
  Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian
- QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning | [arXiv 2402](https://arxiv.org/pdf/2402.10462.pdf) \
  Hossein Rajabzadeh, Mojtaba Valipour, Tianshu Zhu, Marzieh Tahaei, Hyock Ju Kwon, Ali Ghodsi, Boxing Chen, Mehdi Rezagholizadeh
- Loftq: LoRA-fine-tuning-aware quantization for large language models | [arXiv 2310](https://arxiv.org/pdf/2310.08659.pdf) | [Code](https://github.com/yxli2123/LoftQ) \
  Hossein Rajabzadeh, Mojtaba Valipour, Tianshu Zhu, Marzieh Tahaei, Hyock Ju Kwon, Ali Ghodsi, Boxing Chen, Mehdi Rezagholizadeh
- Lq-LoRA: Low-rank plus quantized matrix decomposition for efficient language model finetuning | [arXiv 2311](https://arxiv.org/pdf/2311.12023.pdf) | [Code](https://github.com/HanGuo97/lq-LoRA) \
  Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim
- LQER: Low-Rank Quantization Error Reconstruction for LLMs | [arXiv 2402](https://arxiv.org/pdf/2402.02446.pdf) | [Code](https://github.com/OpenGVLab/OmniQuant) | ICLR 2024 \
  Cheng Zhang, Jianyi Cheng, George A. Constantinides, Yiren Zhao
- L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models | [arXiv 2402](https://arxiv.org/abs/2402.04902) | ACL 2025 \
  Hyesung Jeon, Yulhwa Kim, Jae-joon Kim
- LoQT: Low-Rank Adapters for Quantized Pretraining| [arXiv 2405](https://arxiv.org/abs/2405.16528)| NeurIPS2024 \
  Sebastian Loeschcke, Mads Toftrup, Michael J. Kastoryano, Serge Belongie, Vésteinn Snæbjarnarson
- LowRA: Accurate and Efficient LoRA Fine-Tuning of LLMs under 2 Bits | [ICML 2025](https://openreview.net/forum?id=Fm0nDMKBwC&noteId=X2TeKe8AkH) \
  Zikai Zhou, Qizheng Zhang, Hermann Kumbong, Kunle Olukotun
- IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models| [ICML 2025](https://openreview.net/forum?id=f4mQ2SU5tp) \
  Hang Guo, Yawei Li, Tao Dai, Shu-Tao Xia, Luca Benini
- Budget-Aware LLM Quantization and Low-Rank Correction via Information-Guided Subspace Matrices | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/475) \
  Sinuo Fan, Yingjie Lao
- Low-Rank Ternary Adaptation for Fine-Tuning Transformers | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5924) \
  Alexandru-Dragos Manolache, Yunqiang Li, Jan van Gemert

**(v) Structured and Nonlinear Extensions**

- Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adaptation | [ACL 2026](https://aclanthology.org/2026.acl-long.97/) \
  Xi Xiao, Chenrui Ma, Yunbei Zhang, Chen Liu, Zhuxuanzi Wang, Yanshu Li, Lin Zhao, Guosheng Hu, Tianyang Wang, Hao Xu
- SOS-LoRA: Static Orthogonal-Subspace Low-Rank Adaptation with Fixed Multi-Scale Scaling | [ACL 2026](https://aclanthology.org/2026.acl-long.184/) \
  Yupeng Chang, Yuan Wu, Yi Chang
- Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.650/) \
  Wenhao Zhang, Lin Mu, Li Ni, Peiquan Jin, Yiwen Zhang
- RanLoRA: Residual-aware Nonlinear Low-Rank Adaptation | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.852/) \
  Xu Luo, Yongbin Liu, Chunping Ouyang, Ying Yu
- G-LoRA: Global-Local Decoupled Low-Rank Adaptation | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1005/) \
  Jiahao Xiong, Yihong Huang, Yihe Liu, Xianming Hu, Hongbo Zhao, Kai Zhang
- MaskLoRA: Low-Rank Subspace-Induced Token Masking for Efficient and Faithful Language Models | [Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.300/) \
  Rifat Rafiuddin
- Dynamic Positional Attention Modulation for Parameter-Efficient Fine-Tuning of Large Language Models | [KDD 2026](https://doi.org/10.1145/3770855.3817911) \
  Dayan Pan, Jingyuan Wang, Xie Yu
- PEANuT: Parameter-Efficient Adaptation with Weight-aware Neural Tweakers | [KDD 2026](https://doi.org/10.1145/3770854.3780230) \
  Yibo Zhong, Haoxiang Jiang, Lincan Li, Ryumei Nakada, Tianci Liu, Linjun Zhang, Huaxiu Yao, Haoyu Wang
- Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning | [SIGIR 2026](https://doi.org/10.1145/3805712.3809719) \
  Jujia Zhao, Zihan Wang, Shuaiqun Pan, Suzan Verberne, Zhaochun Ren
- CeRA: Extreme Parameter Efficiency in Low-Rank Adaptation via Non-linear Expansion | [COLM 2026](https://colmweb.org/AcceptedPapers.html) \
  Hung-Hsuan Chen
- LoCo: Low-Rank Compositional Rotation Fine-Tuning | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/521) \
  An Nguyen, Jaesik Choi, Anh Tong
- SeMi-LoRA: Enhancing Low-Rank Adaptation via Separation and Mixing | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/671) \
  Zhenfei Yang, Beiming Yu, Peiqin Lin, Yongkang Liu, Deyi Xiong

### b. Rank Design and Capacity Scaling

**(i) Ranking Refinement**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY) \
  Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arXiv 2403](https://arxiv.org/pdf/2403.13037v1) \
  Rushi Qiang, Ruiyi Zhang, Pengtao Xie
- DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation | [EACL](https://arxiv.org/abs/2210.07558) | [Code](https://github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA) \
  Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, Ali Ghodsi
- PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation | [arXiv 2401](https://arxiv.org/pdf/2401.11316.pdf) \
  Nadav Benedek, Lior Wolf
- IGU-LoRA: Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Aware Scoring | [arXiv 2603](https://arxiv.org/abs/2603.13792) | ICLR 2026 \
  Xuan Cui, Huiyue Li, Run Zeng, Yunfei Zhao, Jinrui Qian, Wei Duan, Bo Liu, Zhanpeng Zhou

**(ii) Ranking Augmentation**

- FLoRA: Low-Rank Adapters Are Secretly Gradient Compressors | [arXiv 2402](https://arxiv.org/pdf/2402.03293.pdf) | [Code](https://github.com/MANGA-UOFA/FLoRA) | ICML 2024 \
  Yongchang Hao, Yanshuai Cao, Lili Mou
- Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | [arXiv 2401](https://arxiv.org/pdf/2401.04151.pdf) | ICML 2024 \
  Wenhan Xia, Chengwei Qin, Elad Hazan
- ReLoRA: High-Rank Training Through Low-Rank Updates | [arXiv 2307](https://arxiv.org/pdf/2307.05695.pdf) | [Code](https://github.com/guitaricet/reLoRA) \
  Vladislav Lialin, Namrata Shivagunde, Sherin Muckatira, Anna Rumshisky
- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA | [arXiv 2402](https://arxiv.org/abs/2402.16902) | [Code](https://github.com/sahil280114/codealpaca) \
  Sheng Wang, Boyang Xue, Jiacheng Ye, Jiyue Jiang, Liheng Chen, Lingpeng Kong, Chuan Wu
- Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | [arXiv 2402](https://arxiv.org/abs/2402.17263) | ACL 2024 \
  Pengjie Ren, Chengshun Shi, Shiguang Wu, Mengqi Zhang, Zhaochun Ren, Maarten de Rijke, Zhumin Chen, Jiahuan Pei
- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | [arXiv 2403](https://arxiv.org/abs/2403.03507) | [Code](https://github.com/jiaweizzhao/GaLore) | ICML 2024 \
  Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian
- Mora: High-rank updating for parameter-efficient fine-tuning | [arXiv 2405](https://arxiv.org/abs/2405.12130) | [Code](https://github.com/kongds/MoRA) \
  Ting Jiang, Shaohan Huang, Shengyue Luo, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, Fuzhen Zhuang
- On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models | [ICLR 2025](http://openreview.net/forum?id=pxclAomHat) \
  Xu-Hui Liu, Yali Du, Jun Wang, Yang Yu
- Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering ｜[ICLR 2025](https://openreview.net/forum?id=j6fsbpAllN) \
  Ziyu Zhao, Tao Shen, Didi Zhu, Zexi Li, Jing Su, Xuwu Wang, Fei Wu
- HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models Lora rank augmentation ｜[ICLR 2025](https://openreview.net/forum?id=TwJrTz9cRS) | [Code](https://github.com/hqsiswiliam/hira.) \
  Qiushi Huang, Tom Ko, Zhan Zhuang, Lilian Tang, Yu Zhang
- RandLoRA: Full rank parameter-efficient fine-tuning of large models | [ICLR 2025](https://openreview.net/forum?id=Hn5eoTunHN) | [Code](https://github.com/PaulAlbert31/RandLoRA) \
  Paul Albert, Frederic Z. Zhang, Hemanth Saratchandran, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad
- BoRA: Towards More Expressive Low-Rank Adaptation with Block Diversity | [arXiv 2508](https://arxiv.org/abs/2508.06953) | ICLR 2026 \
  Shiwei Li, Xiandi Luo, Haozhao Wang, Xing Tang, Ziqiang Cui, Dugang Liu, Yuhua Li, Xiuqiang He, Ruixuan Li

**(iii) Adaptive Rank Allocation and Routing**

- TLoRA: Task-aware Low Rank Adaptation of Large Language Models | [ACL 2026](https://aclanthology.org/2026.acl-long.1348/) \
  Weicheng Lin, Yi Zhang, Jiawei Dang, Liang-Jie Zhang
- FARSS: Fisher-Optimized Adaptive Low-Rank and Singular-Vector Selection for Knowledge-Preserving Fine-Tuning | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.883/) \
  Renxing Chen, Ziwei Xiang, Peisong Wang, Hongjian Fang, Meng Li, Fanhu Zeng, Yanan Zhu, Peipei Yang, Xu-Yao Zhang, Jian Cheng
- Context-Conditioned Masked LoRA: Dynamic Rank Routing for Compute-Efficient Parameter-Efficient Fine-Tuning | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1329/) \
  Rifat Rafiuddin, Rafae Abdullah
- ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning | [ICML 2026](https://icml.cc/virtual/2026/poster/63892) \
  Yilang Zhang, Xiaodong Yang, Yiwei Cai, Georgios B. Giannakis
- Low Kruskal-Rank Adaptation | [ICML 2026](https://icml.cc/virtual/2026/poster/63686) \
  Yixing Xu, Guanchen Li, Chao Li, Xuanwu Yin, Dong Li, Spandan Tiwari, Ashish Sirasao, Emad Barsoum
- DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models | [COLM 2026](https://colmweb.org/AcceptedPapers.html) \
  Guanzhi Deng, Bo Li, Ronghao Chen, Xiujin Liu, Huacan Wang, Zhuo Han, Lijie Wen, Linqi Song

### c. Optimization, Initialization, and Training Dynamics

#### **(i) Learning Rate**

- LoRA+: Efficient Low-Rank Adaptation of Large Models | [arXiv 2402](https://arxiv.org/pdf/2402.12354.pdf) | [Code](https://github.com/nikhil-ghosh-berkeley/LoRAplus) | ICML 2024 \
  Soufiane Hayou, Nikhil Ghosh, Bin Yu

#### **(ii) Dropout**

- LoRA Meets Dropout under a Unified Framework | [arXiv 2403](https://arxiv.org/pdf/2403.00812) \
  Sheng Wang, Liheng Chen, Jiyue Jiang, Boyang Xue, Lingpeng Kong, Chuan Wu

#### **(iii) Scaling Factor**

- A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | [arXiv 2312](https://arxiv.org/pdf/2312.03732.pdf) | [Code](https://github.com/kingoflolz/mesh-transformer-jax) \
  Damjan Kalajdzievski

#### **(iv) Learning Methods**

- AMAL: Meta Knowledge-Driven Few-Shot Adapter Learning | [ACL 2022](https://aclanthology.org/2022.emnlp-main.709.pdf) \
  S. K. Hong, Tae Young Jang
- LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning | [arXiv 2505](https://arxiv.org/abs/2505.21289) | ICLR 2026 \
  Nurbek Tastan, Stefanos Laskaridis, Martin Takac, Karthik Nandakumar, Samuel Horvath
- LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters | [arXiv 2507](https://arxiv.org/abs/2507.12142) | ICLR 2026 \
  Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, Denis Bobkov, Vera Soboleva, Aibek Alanov, Maxim Rakhuba
- Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models | [arXiv 2508](https://arxiv.org/abs/2508.19564) | ICLR 2026 \
  Yuhang Liu, Tao Li, Zhehao Huang, Zuopeng Yang, Xiaolin Huang
- BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models | [OpenReview](https://openreview.net/forum?id=q0X9SiXiRO) | ICLR 2026 \
  Yupeng Chang, Yi Chang, Yuan Wu
- LoRA-E^2: Effective and Efficient Low-rank Adaptation | [WWW 2026](https://doi.org/10.1145/3774904.3792500) \
  Shengkun Zhu, Jinshan Zeng, Yiming Wang, Sheng Wang, Yuan Sun, Shangfeng Chen, Yuan Yao, Qiang Yang
- Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEFT | [arXiv 2511](https://arxiv.org/abs/2511.00051) | AAAI 2026 \
  Chang Da, Peng Xue, Yu Li, Yongxiang Liu, Pengxiang Xu, Shixun Zhang
- RoZO: Geometry-Aware Zeroth-Order Fine-Tuning on Low-Rank Adapters for Black-Box Large Language Models | [EACL 2026](https://aclanthology.org/2026.eacl-long.80/) \
  Zichen Song, Weijia Li
- SADA: Bridging In-Context Learning and Fine-Tuning via State-Aligned Distillation Adapters | [ACL 2026](https://aclanthology.org/2026.acl-long.1046/) \
  Wenhao Gao, Tianlong Wang, Wei Jia, Linhao Zhang, Aiwei Liu, Miao Fan, Zhou Xiao
- GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR | [ACL 2026](https://aclanthology.org/2026.acl-long.1110/) \
  Jiaying Zhang, Lei Shi, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He
- Can Spectral-Clipping Enable Better Learning While Forgetting Less for Low-Rank Adaptation? | [ACL 2026](https://aclanthology.org/2026.acl-long.1179/) \
  Hyowon Wi, Noseong Park
- Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1593/) \
  Kainan Liu, Yong Zhang, Ning Cheng, Yun Zhu, Yanmeng Wang, Shaojun Wang, Jing Xiao
- Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence | [ICML 2026](https://icml.cc/virtual/2026/poster/62055) \
  Valerie Castin, Kimia Nadjahi, Pierre Ablin, Gabriel Peyre
- Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning | [ICML 2026](https://icml.cc/virtual/2026/poster/64288) \
  Zhi-Quan Feng, Ying-Jia Lin, Hung-Yu Kao
- LoRA-DA: Data-Aware Initialization for Low-Rank Adaptation via Asymptotic Analysis | [ICML 2026](https://icml.cc/virtual/2026/poster/64590) \
  Qingyue Zhang, Chang Chu, Tianren Peng, Qi Li, Xiangyang Luo, Zhihao Jiang, Shao-Lun Huang
- Modality-Agnostic Zeroth-Order LoRA Fine-Tuning for Black-Box Prompt Optimization | [KDD 2026](https://doi.org/10.1145/3770855.3817738) \
  Xingchen Li, Jia Zhang, Tianxing Man, Wenkang Wang, Bin Gu
- FLINT: Influence-Guided Active Learning Framework for LoRA via Curvature-Aware Data Selection and Fine-Tuning | [COLM 2026](https://colmweb.org/AcceptedPapers.html) \
  Zixuan Li, Shenglan Guo, Zhigen Li, Yanmeng Wang, Ning Cheng, Shaojun Wang, Deyi Xiong
- Unlocking More Granular Control of Memory-Efficient LLM Finetuning | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/555) \
  Yezhen Wang, Zhouhao Yang, Fanyi Pu, Kenji Kawaguchi
- TaRA: Training-Aware Low-Rank Adaptation Initialization | [OpenReview](https://openreview.net/forum?id=T3GF7rAVhx) | EMNLP 2026 \
  Taehyeon Kim, Eunhyeok Park

#### **(v) Pre-processing**

- PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models | [arXiv 2506](https://arxiv.org/abs/2506.20629) | [Code](https://github.com/soufiane001/plop) \
  Soufiane Hayou, Nikhil Ghosh, Bin Yu
- Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics | [ICML 2025](https://openreview.net/forum?id=8V6MEtSnlR) | [Code](https://github.com/Leopold1423/non_zero_lora-icml25) \
  Shiwei Li, Xiandi Luo, Xing Tang, Haozhao Wang, Hao Chen, weihongluo, Yuhua Li, xiuqiang He, Ruixuan Li

#### **(vi) Post-hoc Processing**

- Bayesian Low-rank Adaptation for Large Language Models | [arXiv 2308](https://arxiv.org/abs/2308.13111) | [Code](https://github.com/adamxyang/laplace-lora) | ICLR 2024 \
  Adam X. Yang, Maxime Robeyns, Xi Wang, Laurence Aitchison

### d. Theory and Analysis

- The Expressive Power of Low-Rank Adaptation | [arXiv 2310](https://arxiv.org/pdf/2310.17513.pdf) | [Code](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) | ICLR 2024 \
  Yuchen Zeng, Kangwook Lee
- LoRA Training in the NTK Regime has No Spurious Local Minima | [arXiv 2402](https://arxiv.org/pdf/2402.11867.pdf) | [Code](https://github.com/UijeongJang/LoRA-NTK) | ICML 2024 \
  Uijeong Jang, Jason D. Lee, Ernest K. Ryu
- ROSA: Random Orthogonal Subspace Adaptation | [ICML 2023](https://openreview.net/pdf?id=4P9vOFpb63) | [Code](https://github.com/marawangamal/rosa) \
  Marawan Gamal, Guillaume Rabusseau
- Asymmetry in Low-Rank Adapters of Foundation Models | [arXiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA) \
  Jiacheng Zhu, Kristjan Greenewald, Kimia Nadjahi, Haitz Sáez de Ocáriz Borde, Rickard Brüel Gabrielsson, Leshem Choshen, Marzyeh Ghassemi, Mikhail Yurochkin, Justin Solomon
- LoRA Training Provably Converges to a Low-Rank Global Minimum Or It Fails Loudly (But it Probably Won’t Fail) | [ICML 2025](https://openreview.net/forum?id=o9zDYV4Ism) \
  Junsu Kim, Jaeyeon Kim, Ernest K. Ryu
- Computational Limits of Low-Rank Adaptation (LoRA) Fine-Tuning for Transformer Models | [ICLR 2025](https://openreview.net/forum?id=Lf5znhZmFu)\
  Jerry Yao-Chieh Hu, Maojiang Su, En-jui kuo, Zhao Song, Han Liu
- LoRA Learns Less and Forgets Less | [TMLR](https://openreview.net/forum?id=aloEru2qCG) \
  Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John Patrick Cunningham
- LoRA-Pro: Are Low-Rank Adapters Properly Optimized? | [ICLR 2025](https://openreview.net/forum?id=gTwRMU3lJ5) | [Code](https://github.com/mrflogs/LoRA-Pro) \
  Zhengbo Wang, Jian Liang, Ran He, Zilei Wang, Tieniu Tan
- Look Within or Beyond? A Theoretical Comparison Between Parameter-Efficient and Full Fine-Tuning | [ACL 2026](https://aclanthology.org/2026.acl-long.2208/) \
  YongKang Liu, Xingle Xu, Ercong Nie, Zijing Wang, Shi Feng, Daling Wang, Qian Li, Hinrich Schuetze
- On the Representation Geometry of LoRA Model Merging | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.261/) \
  Chenyang Lu, Jiaru Li, Jinman Zhao, Xinran Chen, Yining Wang, Renyi Cai, Yuchen Li, Chao He
- Towards Understanding the Dynamics of Low-Rank Adaptation | [ICML 2026](https://icml.cc/virtual/2026/poster/62781) \
  Shu Ding, Yang Peng, Hangan Zhou, Xinyu Lu, Shangwei Chen, Junhua Huang, Mingxuan Yuan, Wei Wang
- On the Convergence Rate of LoRA Gradient Descent | [ICML 2026](https://icml.cc/virtual/2026/poster/65870) \
  Siqiao Mu, Diego Klabjan
- Dive into the implicit biases of low-rank vision-language alignment | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4041) \
  Mingjia Shi, Shuo Wang, Xiaobo Wang, Sifan Zhou, Kai Wang, Tianyu Fu, Chenxu Zhao, Anyang Su, Ping Jiang, Minghui Wu
- What Did My Adapter Break? A Generation-Level Evaluation of Adaptation and Retention in PEFT-Adapted LLMs | [INLG 2026](https://2026.inlgmeeting.org/accepted-papers.html) \
  Guy Bilitski, Kfir Bar, Shai Fine

## 2. Adaptation Settings and Systems

### a. Composition, Routing, and Structural Extensions

**Composition and Merging**

- Adaptersoup: Weight averaging to improve generalization of pretrained language models | [arXiv 2302](https://arxiv.org/pdf/2302.07027) | [Code](https://github.com/UKPLab/sentence-transformers) \
  Alexandra Chronopoulou, Matthew E. Peters, Alexander Fraser, Jesse Dodge
- LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks ｜ [COLING 2025](https://arxiv.org/abs/2410.13025) | [Code](https://github.com/aksh555/LoRA-Soups) \
  Akshara Prabhakar, Yuanzhi Li, Karthik Narasimhan, Sham Kakade, Eran Malach, Samy Jelassi
- LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition | [arXiv 2307](https://arxiv.org/pdf/2307.13269.pdf) | [Code](https://github.com/sail-sg/LoRAhub) | COLM 2024 \
  Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin
- LoRARetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild | [arXiv 2402](https://arxiv.org/pdf/2402.09997.pdf) | [Code](https://github.com/tatsu-lab/stanford_alpaca) \
  Ziyu Zhao, Leilei Gan, Guoyin Wang, Wangchunshu Zhou, Hongxia Yang, Kun Kuang, Fei Wu
- Batched Low-Rank Adaptation of Foundation Models | [arXiv 2312](https://arxiv.org/pdf/2312.05677.pdf) | [Code](https://github.com/huggingface/peft/tree/main) \
  Yeming Wen, Swarat Chaudhuri
- Hydra: Multi-head low-rank adaptation for parameter efficient fine-tuning | [arXiv 2309](https://arxiv.org/pdf/2309.06922.pdf) | [Code](https://github.com/extremebird/Hydra) \
  Sanghyeon Kim, Hyunmo Yang, Younghyun Kim, Youngjoon Hong, Eunbyung Park
- One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning | [arXiv 2306](https://arxiv.org/pdf/2306.07967.pdf) | [Code](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA) \
  Arnav Chavan, Zhuang Liu, Deepak Gupta, Eric Xing, Zhiqiang Shen
- LoRA ensembles for large language model fine-tuning | [arXiv 2310](https://arxiv.org/pdf/2310.00035.pdf) | [Code](https://github.com/huggingface/peft) \
  Xi Wang, Laurence Aitchison, Maja Rudolph
- MultiLoRA: Democratizing LoRA for Better Multi-Task Learning | [arXiv 2311](https://arxiv.org/pdf/2311.11501.pdf) \
  Yiming Wang, Yu Lin, Xiaodong Zeng, Guannan Zhang
- ComLoRA: A Competitive Learning Approach for Enhancing LoRA ｜ [ICLR 2025](https://openreview.net/forum?id=jFcNXJGPGh) | [Code](https://github.com/hqsiswiliam/comlora) \
  Qiushi Huang, Tom Ko, Lilian Tang, Yu Zhang
- SeedLoRA: A Fusion Approach to Efficient LLM Fine-Tuning | [ICML 2025](https://proceedings.mlr.press/v267/liu25o.html) \
  Yong Liu, Di Fu, Shenggan Cheng, Zirui Zhu, Yang Luo, Minhao Cheng, Cho-Jui Hsieh, Yang You
- Completely Modular Fine-tuning for Dynamic Language Adaptation | [Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.252/) \
  Zhe Cao, Yusuke Oda, Qianying Liu, Akiko Aizawa, Taro Watanabe
- TIPA: Typologically Informed Parameter Aggregation | [Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.119/) \
  Stef Accou, Wessel Poelman
- Evolutionary Negative Module Pruning for Better LoRA Merging | [ACL 2026](https://aclanthology.org/2026.acl-long.1730/) \
  Anda Cao, Zhuo Gou, Yi Wang, Kaixuan Chen, Yu Wang, Can Wang, Mingli Song, Jie Song
- LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging | [ACL 2026](https://aclanthology.org/2026.acl-long.1837/) \
  Seungeon Lee, Soumi Das, Manish Gupta, Krishna P. Gummadi
- Two-Stage Parameter Alignment for Multi-LoRA Merging in Large Language Models | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1504/) \
  Zijian Li, Xiachong Feng, Weitao Ma, Yichong Huang, Xiaocheng Feng, Bing Qin
- Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter | [ICML 2026](https://icml.cc/virtual/2026/poster/61546) \
  Zhengbao He, Ruiqi Ding, Zhehao Huang, Ruikai Yang, Tao Li, Xiaolin Huang
- Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jeong_Preference-Aligned_LoRA_Merging_Preserving_Subspace_Coverage_and_Addressing_Directional_Anisotropy_CVPR_2026_paper.html) \
  Wooseong Jeong, Wonyoung Lee, Kuk-Jin Yoon
- DA-MergeLoRA: Hypernetwork-Based LoRA Merging for Few-Shot Test-Time Domain Adaptation | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4515) \
  Siobhan Reid, Zhixiang Chi, Li Gu, Omid Heidari, Ziqiang Wang, Yang Wang
- Towards Plug-and-Play Attribute Control Modules in Text Generation: An Exploratory Study of LoRA Portability Across Model Families | [INLG 2026](https://2026.inlgmeeting.org/accepted-papers.html) \
  Michela Lorandi, Anya Belz

**MoE and Expert Routing**

- MoELoRA: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models | [arXiv 2402](https://arxiv.org/pdf/2402.12851.pdf) \
  Tongxu Luo, Jiahe Lei, Fangyu Lei, Weihao Liu, Shizhu He, Jun Zhao, Kang Liu
- Higher Layers Need More LoRA Experts | [arXiv 2402](https://arxiv.org/pdf/2402.08562.pdf) | [Code](https://github.com/GCYZSL/MoLA) \
  Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, VS Subrahmanian
- Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning | [arXiv 2309](https://arxiv.org/abs/2309.05444) | [Code](https://github.com/for-ai/parameter-efficient-moe) \
  Ted Zadouri, Ahmet Üstün, Arash Ahmadian, Beyza Ermiş, Acyr Locatelli, Sara Hooker
- MOELoRA: An moe-based parameter efficient fine-tuning method for multi-task medical applications | [arXiv 2310](https://arxiv.org/pdf/2310.18339.pdf) | [Code](https://github.com/liuqidong07/MOELoRA-peft) | SIGIR 24 \
  Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, Yefeng Zheng
- LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs | [arXiv 2401](https://arxiv.org/pdf/2401.16160.pdf) \
  Shaoxiang Chen, Zequn Jie, Lin Ma
- Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models | [arXiv 2403](https://arxiv.org/pdf/2403.03432) \
  Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, Hao Wang
- Mixture of Cluster-Conditional LoRA Experts for Vision-Language Instruction Tuning | [arXiv 2312](https://arxiv.org/pdf/2312.12379) | [Code](https://github.com/gyhdog99/mocle) \
  Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang
- MIXLORA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts | [arXiv 2404](https://arxiv.org/pdf/2404.15159) | [Code](https://github.com/TUDB-Labs/MixLoRA) \
  Dengchun Li, Yingzi Ma, Naizheng Wang, Zhengmao Ye, Zhiyuan Cheng, Yinghao Tang, Yan Zhang, Lei Duan, Jie Zuo, Cal Yang, Mingjie Tang
- LoRAMOE: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment | [arXiv 2312](https://arxiv.org/abs/2312.09979) | [Code](https://github.com/Ablustrund/LoRAMoE) \
  Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang
- MoRAL: MoE Augmented LoRA for LLMs' Lifelong Learning | [arXiv 2402](https://arxiv.org/pdf/2402.11260) \
  Shu Yang, Muhammad Asif Ali, Cheng-Long Wang, Lijie Hu, Di Wang
- Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts | [arXiv 2405](https://arxiv.org/abs/2405.11273) | [Code](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs) \
  Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, Min Zhang
- AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts | [arXiv 2405](https://arxiv.org/abs/2405.00361) | [Code](https://github.com/zefang-liu/AdaMoLE) | COLM 2024 \
  Zefang Liu, Jiahua Luo
- Mixture of LoRA Experts | [arXiv 2404](https://arxiv.org/abs/2404.13628) | [Code](https://github.com/yushuiwx/MoLE) | ICLR 2024 \
  Xun Wu, Shaohan Huang, Furu Wei
- HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning | [NeurIPS 2024](https://openreview.net/forum?id=qEpi8uWX3N&referrer=%5Bthe%20profile%20of%20Zhijiang%20Guo%5D(%2Fprofile%3Fid%3D~Zhijiang_Guo2)) | [Code](https://github.com/Clin0212/HydraLoRA) \
  Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, Chengzhong Xu
- AlphaLoRA: Assigning LoRA Experts Based on Layer Training Quality | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1141/) | [Code](https://github.com/morelife2017/alphalora) \
  Peijun Qing, Chongyang Gao, Yefan Zhou, Xingjian Diao, Yaoqing Yang, Soroush Vosoughi
- TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition | [ACL 2025](https://aclanthology.org/2025.acl-long.669/) | [Code](https://github.com/DCDmllm/TeamLoRA) \
  Tianwei Lin, Jiang Liu, Wenqiao Zhang, Yang Dai, Haoyuan Li, Zhelun Yu, Wanggui He, Juncheng Li, Jiannan Guo, Hao Jiang, Siliang Tang, Yueting Zhuang
- RepLoRA: Reparameterizing Low-rank Adaptation via the Perspective of Mixture of Experts | [ICML 2025](https://openreview.net/forum?id=Sg8ZqQ9J6W) \
  Tuan Truong, Chau Nguyen, Huy Nguyen, Minh Le, Trung Le, Nhat Ho
- Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment | [ICML 2025](https://openreview.net/forum?id=SUxq4HeIAd&noteId=cDt4PKFUE0) | [Code](https://github.com/Facico/GOAT-PEFT) \
  Chenghao Fan, Zhenyi Lu, Sichen Liu, Chengfeng Gu, Xiaoye Qu, Wei Wei, Yu Cheng
- MoKA: Parameter Efficiency Fine-Tuning via Mixture of Kronecker Product Adaption | [COLING 2025](https://aclanthology.org/2025.coling-main.679/) \
  Beiming Yu, Zhenfei Yang, Xiushuang Yi
- Adapters Selector: Cross-domains and Multi-tasks LoRA Modules Integration Usage Method | [COLING 2025](https://aclanthology.org/2025.coling-main.40/) \
  Yimin Tian, Bolin Zhang, Zhiying Tu, Dianhui Chu
- Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules | [Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.232/) \
  Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, Volker Tresp
- MoSLD: An Extremely Parameter-Efficient Mixture-of-Shared LoRAs for Multi-Task Learning | [COLING 2025](https://aclanthology.org/2025.coling-main.111/) \
  Lulu Zhao, Weihao Zeng, Shi Xiaofeng, Hua Zhou
- HMoRA: Making LLMs More Effective with Hierarchical Mixture of LoRA Experts | [ICLR 2025](https://openreview.net/forum?id=lTkHiXeuDl) | [Code](https://github.com/LiaoMengqi/HMoRA.) \
  Mengqi Liao, Wei Chen, Junfeng Shen, Shengnan Guo, Huaiyu Wan
- TalkLoRA: Communication-Aware Mixture of Low-Rank Adaptation for Large Language Models | [ACL 2026](https://aclanthology.org/2026.acl-long.840/) \
  Lin Mu, Haiyang Wang, Li Ni, Lei Sang, Zhize Wu, Peiquan Jin, Yiwen Zhang
- MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models | [ACL 2026](https://aclanthology.org/2026.acl-long.965/) \
  Jie Cao, Tianwei Lin, Bo Yuan, Rolan Yan, Hongyang He, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, Yueting Zhuang
- CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.811/) \
  Jie Cao, Zhenxuan Fan, Zhuonan Wang, Tianwei Lin, Ziyuan Zhao, Rolan Yan, Wenqiao Zhang, Feifei Shao, Hongwei Wang, Jun Xiao, Siliang Tang
- SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1404/) \
  Boyan Shi, Wei Chen, Shuyuan Zhao, Junfeng Shen, Shengnan Guo, Shaojiang Wang, Huaiyu Wan
- Adaptive Utilization of Low-Rank Adaptation via Conditioned Gating | [ICML 2026](https://icml.cc/virtual/2026/poster/62892) \
  Guang Yang, Changhao Guan, Chao Huang, Yufeng Chen, Kaiyu Huang
- MoLoRA: Composable Specialization via Per-Token Adapter Routing | [ICML 2026](https://icml.cc/virtual/2026/poster/66486) \
  Shrey Shah, Justin Wagle
- Reinforcement Routing for Mixtures of LoRAs in Parameter-Efficient LLM Finetuning | [COLM 2026](https://colmweb.org/AcceptedPapers.html) \
  Ruizhong Qiu, Hanqing Zeng, Yinglong Xia, Yiwen Meng, Ren Chen, Jiarui Feng, Dongqi Fu, Qifan Wang, Jiayi Liu, Jun Xiao, Xiangjun Fan, Benyu Zhang, Hong Li, Zhining Liu, Hyunsik Yoo, Zhichen Zeng, Tianxin Wei, Hanghang Tong
- Each Rank Could be an Expert: Single-Ranked Mixture of Experts LoRA for Multi-task Learning | [KDD 2026](https://doi.org/10.1145/3770854.3780222) \
  Ziyu Zhao, Yixiao Zhou, Xin Yu, Zhi Zhang, Didi Zhu, Tao Shen, Zexi Li, Jinluan Yang, Xuwu Wang, Jing Su, Kun Kuang, Zhongyu Wei, Fei Wu, Yu Cheng
- When Gradient Boosting Meets Adaption: Exploring Weak Learner Principle for Parameter Efficient Fine-tuning of LLMs | [KDD 2026](https://doi.org/10.1145/3770855.3817971) \
  Yifei Zhang, Hao Zhu, Haoran Shi, Junhao Dong, Lingyun Song, Xiaolin Han, Yanyu Chen, Wenxuan Wang, Han Yu, Xuequn Shang, Piotr Koniusz
- TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jeon_TAS-LoRA_Transformer_Architecture_Search_with_Mixture-of-LoRA_Experts_CVPR_2026_paper.html) \
  Jeimin Jeon, Hyunju Lee, Bumsub Ham
- RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5012) \
  Woo Jae Kim, Kyle Min, Suhyeon Ha, Joonsung Jeon, Sung-eui Yoon


**Other Structural Extensions**
- From Weight-Based to State-Based Fine-Tuning: Further Memory Reduction on LoRA with Parallel Control | [ICML 2025](https://openreview.net/forum?id=x4qvBVuzzu&noteId=D3Jn9eOmNx) \
  Chi Zhang, REN Lianhai, Jingpu Cheng, Qianxiao Li
- LoRA-One: One-Step Full Gradient Could Suffice for Fine-Tuning Large Language Models, Provably and Efficiently | [ICML 2025](https://openreview.net/forum?id=KwIlvmLDLm&noteId=sxMON0AT0E) | [Code](https://github.com/YuanheZ/LoRA-One) \
  Yuanhe Zhang, Fanghui Liu, Yudong Chen
- Text-to-LoRA: Instant Transformer Adaption | [ICML 2025](https://openreview.net/forum?id=zWskCdu3QA) | [Code](https://github.com/SakanaAI/text-to-lora) \
  Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange
- BSLoRA: Enhancing the Parameter Efficiency of LoRA with Intra-Layer and Inter-Layer Sharing | [ICML 2025](https://openreview.net/forum?id=IXYBuwCOMl&noteId=CDQZjHPfax) | [Code](https://github.com/yuhua-zhou/BSLoRA.git) \
  Yuhua Zhou, Ruifeng Li, Changhai Zhou, Fei Yang, Aimin PAN
- SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity | [ICML 2025](https://openreview.net/forum?id=z83rodY0Pw) \
  Samir Khaki, Xiuyu Li, Junxian Guo, Ligeng Zhu, Konstantinos N. Plataniotis, Amir Yazdanbakhsh, Kurt Keutzer, Song Han, Zhijian Liu
- GeoLoRA: Geometric integration for parameter efficient fine-tuning | [ICLR 2025](https://openreview.net/pdf?id=bsFWJ0Kget) \
  Steffen Schotthöfer, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco, Jonas Kusch
- LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization | [ICLR 2025](https://openreview.net/forum?id=VpWki1v2P8) \
  Jui-Nan Yen, Si Si, Zhao Meng, Felix Yu, Sai Surya Duvvuri, Inderjit S Dhillon, Cho-Jui Hsieh, Sanjiv Kumar
- SG-LoRA: Semantic-guided LoRA Parameters Generation | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_SG-LoRA_Semantic-guided_LoRA_Parameters_Generation_CVPR_2026_paper.html) \
  Miaoge Li, Yang Chen, Zhijie Rao, Can Jiang, Kang Wei, Jingcai Guo

### b. Long-Context and Sequence Modeling

- LongLoRA: Efficient fine-tuning of long-context large language models | [arXiv 2309](https://arxiv.org/pdf/2309.12307.pdf) | [Code](https://github.com/dvlab-research/LongLoRA) | ICLR 2024 \
  Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia
- LongqLoRA: Efficient and effective method to extend context length of large language models | [arXiv 2311](https://arxiv.org/pdf/2311.04879.pdf) | [Code](https://github.com/yangjianxin1/LongQLoRA) \
  Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia
- With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation | [arXiv 2401](https://arxiv.org/abs/2401.11504) | [Code](https://github.com/TemporaryLoRA/Temp-LoRA/tree/main) | COLM 2024 \
  Y. Wang, D. Ma, D. Cai
- RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization | [arXiv 2405](https://arxiv.org/abs/2405.00657) \
  Dongqi Pu, Vera Demberg
- Doc-to-LoRA: Learning to Instantly Internalize Contexts | [ICML 2026](https://icml.cc/virtual/2026/poster/62227) \
  Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, Robert Lange

### c. Continual and Lifelong Adaptation

- Orthogonal Subspace Learning for Language Model Continual Learning | [EMNLP 2023 findings](https://arxiv.org/pdf/2310.14152) | [Code](https://github.com/cmnfriend/O-LoRA) \
  Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, Xuanjing Huang
- Continual Learning with Low Rank Adaptation | [NeurIPS 2023 Workshop](https://arxiv.org/pdf/2311.17601) \
  Martin Wistuba, Prabhu Teja Sivaprasad, Lukas Balles, Giovanni Zappella
- Task Arithmetic with LoRA for Continual Learning | [NeurIPS 2023 Workshop](https://arxiv.org/pdf/2311.02428) \
  Rajas Chitale, Ankit Vaidya, Aditya Kane, Archana Ghotkar
- A Unified Continual Learning Framework with General Parameter-Efficient Tuning | [ICCV 2023](https://arxiv.org/pdf/2303.10070) | [Code](https://github.com/gqk/LAE) \
  Qiankun Gao, Chen Zhao, Yifan Sun, Teng Xi, Gang Zhang, Bernard Ghanem, Jian Zhang
- TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree | [ICML 2025](https://openreview.net/forum?id=f6ibJCQfH4) | [Code](https://github.com/ZinYY/TreeLoRA) \
  Yu-Yang Qian, Yuan-Ze Xu, Zhen-Yu Zhang, Peng Zhao, Zhi-Hua Zhou
- SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning | [ICLR 2025](https://openreview.net/forum?id=5U1rlpX68A) | [Code](https://github.com/WuYichen-97/SD-Lora-CL) \
  Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma, Ying Wei
- Sparse Adapter Fusion for Continual Learning in NLP | [EACL 2026](https://aclanthology.org/2026.eacl-long.37/) \
  Min Zeng, Xi Chen, Haiqin Yang, Yike Guo
- Continual Low-Rank Adapters for LLM-based Generative Recommender Systems | [arXiv 2510](https://arxiv.org/abs/2510.25093) | ICLR 2026 \
  Hyunsik Yoo, Ting-Wei Li, SeongKu Kang, Zhining Liu, Charlie Xu, Qilin Qi, Hanghang Tong
- Soft Orthogonal Low-Rank Adaptation for Knowledge Sharing in Large Language Model Continual Learning | [ACL 2026](https://aclanthology.org/2026.acl-long.842/) \
  Yitong Wang, Xue Han, WenChun Gao, Qian Hu, Jiahui Wang, Ziqing Wang, Lijun Mei, Junlan Feng
- SDC-LoRA: Singular-Subspace Drift Controlled LoRA to Mitigate Knowledge Forgetting | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1207/) \
  Geyuan Zhang, Xiaofei Zhou, Shihao Liu, Jingyuan Tian, Jizheng Ma
- GR-LoRA: Gradient-Recycling Low-Rank Adaptation for Class-Incremental Learning | [ICML 2026](https://icml.cc/virtual/2026/poster/64527) \
  Yipeng Lin, Fengqiang Wan, Yang Yang
- JANUS-LORA: A Balanced Low-Rank Adaptation for Continual Learning | [ICML 2026](https://icml.cc/virtual/2026/poster/63423) \
  Cheng Chen, Pengpeng Zeng, Yuyu Guo, Lianli Gao, Heng Tao Shen, Jingkuan Song
- G$^2$LoRA: Gradient Orthogonal Low-Rank Adaptation Framework for Graph Continual Learning on Text-Attributed Graphs | [KDD 2026](https://doi.org/10.1145/3770855.3817966) \
  Yuhan Wang, Yibo Ding, Yutong Ye, Mufan Zhao, Wenbo Zhang, Ruijie Wang, Jianxin Li
- ELLA: Efficient Lifelong Learning for Adapters in Large Language Models | [EACL 2026](https://aclanthology.org/2026.eacl-long.84/) \
  Shristi Das Biswas, Yue Zhang, Anwesan Pal, Radhika Bhargava, Kaushik Roy
- HyLoVQA: Dynamic Hypernetwork-Generated Low-Rank Adaptation for Continual Visual Question Answering | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/200) \
  Yiran Wang, Chenyi Xiong, Ziyue Qin, Miao Zhang, Kui Xiao, Zhifei Li
- Shared LoRA Subspaces for almost Strict Continual Learning | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/3502) \
  Prakhar Kaushik, Ankit Vaidya, Shravan Sunil Chaudhari, Rama Chellappa, Alan Yuille
- VD-LoRA: Adaptive Reuse of Low-Rank Directions for Continual Learning | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5333) \
  Luqiong Ding, Jiayao Tan, Chenggong Ni, Fuyuan Hu, Fan Lyu
- COLA: Continual Orthogonal Low-Rank Adaptation for Class-Incremental Learning | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5464) \
  Monu Nagar, Debasis Das

### d. Federated and Distributed Adaptation

- SLoRA: Federated parameter efficient fine-tuning of language models | [arxiv 2308](https://arxiv.org/pdf/2308.06522.pdf) \
  Sara Babakniya, Ahmed Roushdy Elkordy, Yahya H. Ezzeldin, Qingfeng Liu, Kee-Bong Song, Mostafa El-Khamy, Salman Avestimehr
- pFedLoRA: Model-heterogeneous personalized federated learning with LoRA tuning | [arxiv 2310](https://arxiv.org/pdf/2310.13283.pdf) \
  Liping Yi, Han Yu, Gang Wang, Xiaoguang Liu, Xiaoxiao Li
- Heterogeneous Low-Rank Approximation for Federated Fine-tuning of On-Device Foundation Models | [arxiv 2401](https://arxiv.org/pdf/2401.06432.pdf) \
  Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, Gauri Joshi
- OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning | [arxiv 2402](https://arxiv.org/pdf/2402.06954.pdf)| [Code](https://github.com/rui-ye/OpenFedLLM) \
  Rui Ye, Wenhao Wang, Jingyi Chai, Dihan Li, Zexi Li, Yinda Xu, Yaxin Du, Yanfeng Wang, Siheng Chen
- Federatedscope-llm: A comprehensive package for fine-tuning large language models in federated learning | [arxiv 2309](https://arxiv.org/abs/2309.00363) | [Code](https://github.com/alibaba/FederatedScope/tree/llm) \
  Weirui Kuang, Bingchen Qian, Zitao Li, Daoyuan Chen, Dawei Gao, Xuchen Pan, Yuexiang Xie, Yaliang Li, Bolin Ding, Jingren Zhou
- FedHLT: Efficient Federated Low-Rank Adaption with Hierarchical Language Tree for Multilingual Modeling | [acm](https://dl.acm.org/doi/pdf/10.1145/3589335.3651933) \
  Zhihan Guo, Yifei Zhang, Zhuo Zhang, Zenglin Xu, Irwin King
- FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning | [arxiv 2404](https://arxiv.org/abs/2404.15182) \
  Duy Phuong Nguyen, J. Pablo Munoz, Ali Jannesari
- FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering | [arxiv 2404](https://arxiv.org/abs/2404.15384) | ICLR 2024 \
  Siqi Ping, Yuzhu Mao, Yang Liu, Xiao-Ping Zhang, Wenbo Ding
- FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning | [arxiv 2406](https://arxiv.org/pdf/2406.07925) \
  Jiaxing QI, Zhongzhi Luan, Shaohan Huang, Carol Fung, Hailong Yang, Depei Qian
- FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations | [arxiv 2409](https://arxiv.org/pdf/2409.05976) [Code](https://github.com/ATP-1010/FederatedLLM) \
  Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, Ang Li
- Automated Federated Pipeline for Parameter-Efficient Fine-Tuning of Large Language Models | [arxiv 2404](https://arxiv.org/pdf/2404.06448) \
  Zihan Fang, Zheng Lin, Zhe Chen, Xianhao Chen, Yue Gao, Yuguang Fang
- Towards Federated Low-Rank Adaptation of Language Models with Rank Heterogeneity | [NAACL 2025](https://aclanthology.org/2025.naacl-short.30/) \
  Yuji Byun, Jaeho Lee
- Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients | [ACL 2025](https://aclanthology.org/2025.acl-long.19/) \
  Jabin Koo, Minwoo Jang, Jungseul Ok
- FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models | [ACL 2025](https://aclanthology.org/2025.acl-long.67/) \
  Raghav Singhal, Kaustubh Ponkshe, Praneeth Vepakomma
- DoFIT: Domain-aware Federated Instruction Tuning with Alleviated Catastrophic Forgetting | [NeurIPS 2024](https://openreview.net/forum?id=FDfrPugkGU) \
  Binqian Xu, Xiangbo Shu, Haiyang Mei, Zechen Bai, Basura Fernando, Mike Zheng Shou, Jinhui Tang
- RB-LoRA: Rank-Balanced Aggregation for Low-Rank Adaptation with Federated Fine-Tuning | [Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.88/) \
  Sihyeon Ha, Yongjeong Oh, Yo-Seb Jeon
- FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA | [arXiv 2503](https://arxiv.org/abs/2503.11880) | AAAI 2026 \
  Jieming Bian, Lei Wang, Letian Zhang, Jie Xu
- WinFLoRA: Incentivizing Client-Adaptive Aggregation in Federated LoRA under Privacy Heterogeneity | [arXiv 2602](https://arxiv.org/abs/2602.01126) | WWW 2026 \
  Mengsha Kou, Xiaoyu Xia, Ziqi Wang, Ibrahim Khalil, Runkun Luo, Jingwen Zhou, Minhui Xue
- Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients | [OpenReview](https://openreview.net/forum?id=0g5Dk4Qfh0) | ICLR 2026 \
  Minhyuk Seo, Taeheon Kim, Hankook Lee, Jonghyun Choi, Tinne Tuytelaars
- Federated LoRA Fine-Tuning with Pipelined Error-Mitigated Aggregation and Matrix-Wise Freezing | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.284/) \
  Haoran Wang, Xiong Wang, Yuqing Li, Jing Chen, Junyi Zhang, Nan Yan, Kun He, Wei Wang
- FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA | [ICML 2026](https://icml.cc/virtual/2026/poster/66566) \
  Haoran Zhang, Dongjun Kim, Seohyeon Cha, Haris Vikalo
- HeteroFL-LoRA: Federated LoRA Fine-Tuning Across Heterogeneous LFMs via Singular Value Collaboration | [KDD 2026](https://doi.org/10.1145/3770855.3817794) \
  Zhuojia Wu, Qi Zhang, Xuerong Zhao, Duoqian Miao, Kun Yi, Liang Hu
- HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_HiLoRA_Hierarchical_Low-Rank_Adaptation_for_Personalized_Federated_Learning_CVPR_2026_paper.html) \
  Zihao Peng, Nan Zou, Jiandian Zeng, Guo Li, Ke Chen, Boyuan Li, Tian Wang
- CA-PFL: Client-adaptive Parameter-efficient Fine-tuning for Personalized Federated Learning | [WWW 2026](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Daixin Song, Hui Cai, Haojie Zhang, Biyun Sheng, Jian Zhou, Mang Ye, Fu Xiao
- FedGLoRA: Grassmann-Manifold Federated Learning via Dual LoRA for Large EEG Models | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/398) \
  Qianyu Chen, Yihao Zhong, Runxuan Tang, Tianyi Zhang, Jing Liu, Ziyu Jia, Chenyu Liu
- FediLoRA: Practical Federated Fine-Tuning of Foundation Models Under Missing-Modality Constraints | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/775) \
  Lishan Yang, Wei Emma Zhang, Nam Kha Nguyen, Po Hu, Yanjun Shu, Weitong Chen, Sim Mong Yuan

### e. Pretraining and Full Training

- Training Neural Networks from Scratch with Parallel Low-Rank Adapters | [arXiv 2402](https://arxiv.org/pdf/2402.16828.pdf) | [Code](https://github.com/minyoungg/LTE) \
  Minyoung Huh, Brian Cheung, Jeremy Bernstein, Phillip Isola, Pulkit Agrawal

### f. Serving and Systems

- Peft: State-of-the-art parameter-efficient fine-tuning methods | [Huggingface](https://github.com/huggingface/peft) \
  Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, Fu Lee Wang
- S-LoRA: Serving thousands of concurrent LoRA adapters | [arXiv 2311](https://arxiv.org/pdf/2311.03285.pdf) | [Code](https://github.com/S-LoRA/S-LoRA) | MLSys Conference 2024 \
  Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica
- CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference | [arXiv 2401](https://arxiv.org/pdf/2401.11240.pdf) \
  Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Qizhen Weng, Xusheng Chen, Yizhou Shan, Binhang Yuan, Wei Wang
- Local LoRA: Memory-Efficient Fine-Tuning of Large Language Models | [OpenReview](https://openreview.net/pdf?id=LHKmzWP7RN) | WANT@NeurIPS 2023 \
  Oscar Key, Jean Kaddour, Pasquale Minervini
- LoRA-Gen: Specializing Large Language Model via Online LoRA Generation | [ICML 2025](https://openreview.net/forum?id=oZM5g4IvmS) \
  Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Yixiao Ge, Xiu Li, Ying Shan
- Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead | [ICML 2025](https://openreview.net/forum?id=3XMA8RDJu2) \
  Rickard Brüel Gabrielsson, Jiacheng Zhu, Onkar Bhardwaj, Leshem Choshen, Kristjan Greenewald, Mikhail Yurochkin, Justin Solomon
- Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models | [ICLR 2025](https://openreview.net/forum?id=s7DkcgpRxL) | [Code](https://github.com/junzhang-zj/LoRAM) \
  Jun Zhang, Jue WANG, Huan Li, Lidan Shou, Ke Chen, Yang You, Guiming Xie, Xuejian Gong, Kunlong Zhou
- K-Merge: Online Continual Merging of Adapters for On-device Large Language Models | [ACL 2026](https://aclanthology.org/2026.acl-long.137/) \
  Donald Shenaj, Ondrej Bohdal, Taha Ceritli, Mete Ozay, Pietro Zanuttigh, Umberto Michieli
- PLoRA: Efficient Concurrent LoRA Training for Large Language Models | [ICML 2026](https://icml.cc/virtual/2026/poster/62013) \
  Minghao Yan, Zhuang Wang, Zhen Jia, Shivaram Venkataraman, Yida Wang
- CLIMB: Taming the LoRA Residency Cliff in Multi-LoRA Serving | [ICML 2026](https://icml.cc/virtual/2026/poster/66332) \
  Haoran Zhang, Zhiyu Liang, Decheng Zuo, Hongzhi Wang
- DyMerge-LoRA: On-GPU Post-Merge Fusion for High-Throughput Multi-Tenant Composite LoRA Serving | [KDD 2026](https://doi.org/10.1145/3770854.3780270) \
  Rui Xu, Long Chen, Huazheng Lao, Jinquan Zhang, Xia Zhu
- Task-Aware Cloud-End Offloading for Vision-Language Model Serving via Dynamic Modality-Specific Adapter Scheduling | [WWW 2026](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Zian Wang, Ziyi Wang, Jie Xing, Yaya Wei, Ziyan Zhong, Lanshan Zhang
- M-LoRA: Efficient Serving for Concurrent LoRA Adapters with Memory-Aware Speculative Scheduler on Single GPU | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/502) \
  Shaolong Li, Xiang Yang, Qi Qi, Haifeng Sun, Zirui Zhuang, Bo He, Wanyi Ning, Jingyu Wang
- SwarmLoRA: Serverless Shared-Computation Disaggregation for Isolated, High-Throughput Multi-LoRA Serving | [SC26](https://sc26.conference-program.com/presentation/?id=pap979&sess=sess359) \
  Mausam Basnet, Tong Shu

### g. Privacy, Security, and Attacks

- Improving LoRA in Privacy-preserving Federated Learning | [OpenReview](https://openreview.net/pdf?id=NLPzL6HWNl) | ICLR 2024 \
  Youbang Sun, Zitao Li, Yaliang Li, Bolin Ding
- DP-DyLoRA: Fine-Tuning Transformer-Based Models On-Device under Differentially Private Federated Learning using Dynamic Low-Rank Adaptation | [arXiv 2405](https://arxiv.org/abs/2405.06368) \
  Jie Xu, Karthikeyan Saravanan, Rogier van Dalen, Haaris Mehmood, David Tuckey, Mete Ozay
- MineGrad: Gradient Inversion Attacks on LoRA Fine-Tuning | [AISTATS 2026](https://virtual.aistats.org/virtual/2026/poster/13520) \
  Hasin Us Sami, Swapneel Sen, Basak Guler
- PrivSplit: A Lossless Method for Prompt Privacy in Distributed Parameter-Efficient Fine-Tuning | [WWW 2026](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Wujia Niu, Lan Zhang, Haoran Cheng, Shen Li
- Reconstructing Training Data from Adapter-based Federated Large Language Models | [WWW 2026](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Silong Chen, Yuchuan Luo, Guilin Deng, Yi Liu, Ming Xu, Shaojing Fu, Xiaohua Jia
- CoLOR-DP: Conjugate Low-Rank Differential Privacy for Structure-Aware LoRA Fine-Tuning | [WWW 2026](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Kai Zhang, Yuxuan Xu, Wenxiang Lin, Chaoqun Hong, Pei-Wei Tsai, Xin Yuan, Minhui Xue
- LoRAShield: Data-Free Editing Alignment for Secure Personalized LoRA Sharing | [KDD 2026](https://doi.org/10.1145/3770855.3817625) \
  Jiahao Chen, Junhao Li, Yiming Wang, Yong Yang, Yi Jiang, Chunyi Zhou, Qingming Li, Tianyu Du, Shouling Ji
- SDFLoRA: Selective Decoupled Federated LoRA for Privacy-preserving Fine-tuning with Heterogeneous Clients | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/533) \
  Zhikang Shen, Jianrong Lu, Haiyuan Wan, Jianhai Chen
- Toward LoRA Copyright Protection with an Authorized Dual-Watermarking Framework | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/96) \
  Zhipeng Yin, Zichong Wang, Ruijun Chen, Xin Ning, Xingyu Zhang, Wenbin Zhang

## 3. Domains and Modalities

### a. Language and NLP

- Machine Translation with Large Language Models: Prompting, Few-shot Learning, and Fine-tuning with QLoRA | [ACL 2023](https://aclanthology.org/2023.wmt-1.43.pdf) \
  Xuan Zhang, Navid Rajabi, Kevin Duh, Philipp Koehn
- Task-Agnostic Low-Rank Adapters for Unseen English Dialects | [ACL 2023](https://aclanthology.org/2023.emnlp-main.487.pdf) | [Code](https://github.com/zedian/hyperLoRA) \
  Zedian Xiao, William Held, Yanchen Liu, Diyi Yang
- LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training | [arXiv 2401](https://arxiv.org/pdf/2401.04348.pdf) | [Code](https://github.com/VinAIResearch/LAMPAT) | AAAI 2024 \
  Khoi M.Le, Trinh Pham, Tho Quan, Anh Tuan Luu
- MLAS-LoRA: Language-Aware Parameters Detection and LoRA-Based Knowledge Transfer for Multilingual Machine Translation | [ACL 2025](https://aclanthology.org/2025.acl-long.762/) \
  Tianyu Dong, Bo Li, Jinsong Liu, Shaolin Zhu, Deyi Xiong
- MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models | [ICLR 2025](https://openreview.net/forum?id=yOOJwR15xg) \
  Jingwei Xu, Junyu Lai, Yunpeng Huang
- NaRA: Noise-Aware LoRA for Parameter-Efficient Fine-Tuning of Diffusion LLMs | [ICML 2026](https://icml.cc/virtual/2026/poster/61250) \
  Shuaidi Wang, Zhan Zhuang, Ruping Huang, Yu Zhang
- Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs | [SIGIR 2026](https://doi.org/10.1145/3805712.3809713) \
  Lukas Gienapp, Martin Potthast, Andrew Yates, Harrisen Scells, Eugene Yang

### b. Vision and Generative Vision

**Vision Understanding and Adaptation**

**(1) Domain Adaptation and Transfer Learning**

- Motion style transfer: Modular low-rank adaptation for deep motion forecasting | [arXiv 2211](https://arxiv.org/pdf/2211.03165.pdf) | [Code](https://github.com/vita-epfl/motion-style-transfer) \
  Parth Kothari, Danya Li, Yuejiang Liu, Alexandre Alahi
- Efficient low-rank backpropagation for vision transformer adaptation | [arXiv 2309](https://arxiv.org/pdf/2309.15275.pdf) | NeurIPS 2023 \
  Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | [arXiv 2402](https://arxiv.org/pdf/2402.04964.pdf) | [Code](https://github.com/aleemsidra/ConvLoRA) \
  Sidra Aleem, Julia Dietlmeier, Eric Arazo, Suzanne Little
- ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | [arXiv 2406](https://arxiv.org/abs/2406.10973) ｜ ICML 2025 \
  Samar Khanna, Medhanie Irgau, David B. Lobell, Stefano Ermon
- Melo: Low-rank adaptation is better than fine-tuning for medical image diagnosis | [arXiv 2311](https://arxiv.org/pdf/2311.08236.pdf) | [Code](https://github.com/JamesQFreeman/LoRA-ViT) \
  Yitao Zhu, Zhenrong Shen, Zihao Zhao, Sheng Wang, Xin Wang, Xiangyu Zhao, Dinggang Shen, Qian Wang
- Enhancing General Face Forgery Detection via Vision Transformer with Low-Rank Adaptation | [arXiv 2303](https://arxiv.org/pdf/2303.00917.pdf) \
  Yitao Zhu, Zhenrong Shen, Zihao Zhao, Sheng Wang, Xin Wang, Xiangyu Zhao, Dinggang Shen, Qian Wang
- PILO: Principal Component-based Implicit Regularization with Low-rank Optimization for Robust Transfer Learning | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/160) \
  Shuaihe Liu, Qiugang Zhan, Guisong Liu, Tai-Xiang Jiang


**(2) Semantic Segmentation**

- Customized Segment Anything Model for Medical Image Segmentation | [arXiv 2304](https://arxiv.org/abs/2304.13785) | [Code](https://github.com/hitachinsk/SAMed) \
  Kaidong Zhang, Dong Liu
- SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation | [MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_23) \
  An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, Hongliang Ren
- Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model | [Code](https://github.com/autogluon/autogluon/tree/master/examples/automm/Conv-LoRA) \
  An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, Hongliang Ren
- SAM+D: Parameter-Efficient Dimensional Lifting of SAM-Family Models via Depth-Routed LoRA and Depth Shifting | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/3465) \
  Yu Song, Hao Sun, Shiyu Teng, Ikuko Nishikawa, Yen-Wei Chen

**(3) General Vision Adaptation**

- FullLoRA-AT: Efficiently Boosting the Robustness of Pretrained Vision Transformers | [arXiv 2401](https://arxiv.org/pdf/2401.01752.pdf) \
  Zheng Yuan, Jie Zhang, Shiguang Shan
- Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach | [arXiv 2403](https://arxiv.org/abs/2403.19067) | [Code](https://github.com/zstarN70/RLRR) \
  Wei Dong, Xing Zhang, Bihui Chen, Dawei Yan, Zhijun Lin, Qingsen Yan, Peng Wang, Yang Yang
- LORTSAR: Low-Rank Transformer for Skeleton-based Action Recognition | [arXiv 2407](https://arxiv.org/abs/2407.14655) \
  Soroush Oraki, Harry Zhuang, Jie Liang
- Parameter-efficient Model Adaptation for Vision Transformers | [arXiv 2203](https://arxiv.org/pdf/2203.16329.pdf) | [Code](https://github.com/eric-ai-lab/PEViT) | AAAI 2023 \
  Xuehai He, Chunyuan Li, Pengchuan Zhang, Jianwei Yang, Xin Eric Wang
- Canonical Rank Adaptation: An Efficient Fine-Tuning Strategy for Vision Transformers | [ICML 2025](https://openreview.net/forum?id=vexHifrbJg) \
  Lokesh Veeramacheneni, Moritz Wolter, Hilde Kuehne, Juergen Gall
- LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation models | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6db7c49b14da8006892fda7350d76b6a-Paper-Conference.pdf) \
  Ziqi Lu, Heng Yang, Danfei Xu, Boyi Li, Boris Ivanovic, Marco Pavone, Yue Wang
- MixTTA: Low-Rank Cross-Channel Mixing for Reliable Test-Time Adaptation | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4612) \
  Mansoo Jung, Youngwook Kim, Jungwoo Lee
- REAL-OW: Rehearsal-free Open World Object Detection with Low-Rank Adaptation and Dual-Stage Objectness Modeling | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5205) \
  Huazhong Zhang, Xiaowen Fu, Yang Zhang, Linlin Shen, Jinbao Wang
- LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5809) \
  Sojung An, Junha Lee, Sujeong You, Nam Ik Cho, Donghyun Kim


**Vision Generation and Personalization**

- Cones: Concept Neurons in Diffusion Models for Customized Generation | [arXiv 2303](https://arxiv.org/abs/2303.05125) | [Code](https://github.com/Johanan528/Cones) \
  Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao
- Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models | [arXiv 2305](https://arxiv.org/abs/2305.18292) | [Code](https://github.com/TencentARC/Mix-of-Show) \
  Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou
- Generating coherent comic with rich story using ChatGPT and Stable Diffusion | [arXiv 2305](https://arxiv.org/abs/2305.11067) \
  Ze Jin, Zorina Song
- Cones 2: Customizable Image Synthesis with Multiple Subjects | [arXiv 2305](https://arxiv.org/abs/2305.19327) | [Code](https://github.com/ali-vilab/Cones-V2) \
  Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao
- StyleAdapter: A Single-Pass LoRA-Free Model for Stylized Image Generation | [arXiv 2309](https://arxiv.org/abs/2309.01770) \
  Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, Ping Luo
- ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs | [arXiv 2311](https://arxiv.org/abs/2311.13600) \
  Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, Varun Jampani
- Intrinsic LoRA: A Generalist Approach for Discovering Knowledge in Generative Models | [arXiv 2311](https://arxiv.org/abs/2311.17137) | [Code](https://github.com/duxiaodan/intrinsic-lora) \
  Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, Anand Bhattad
- Lcm-LoRA: A universal stable-diffusion acceleration module | [arXiv 2311](https://arxiv.org/pdf/2311.05556.pdf) | [Code](https://github.com/luosiallen/latent-consistency-model) \
  Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, Hang Zhao
- Continual Diffusion with STAMINA: STack-And-Mask INcremental Adapters | [arXiv 2311](https://arxiv.org/abs/2311.18763) \
  James Seale Smith, Yen-Chang Hsu, Zsolt Kira, Yilin Shen, Hongxia Jin
- Orthogonal Adaptation for Modular Customization of Diffusion Models | [arXiv 2312](https://arxiv.org/abs/2312.02432) \
  Ryan Po, Guandao Yang, Kfir Aberman, Gordon Wetzstein
- Style Transfer to Calvin and Hobbes comics using Stable Diffusion | [arXiv 2312](https://arxiv.org/abs/2312.03993) \
  Sloke Shrestha, Sundar Sripada V. S., Asvin Venkataramanan
- Lora-enhanced distillation on guided diffusion models | [arXiv 2312](https://arxiv.org/pdf/2312.06899) \
  Pareesa Ameneh Golnari
- Multi-LoRA Composition for Image Generation | [arXiv 2402](https://arxiv.org/abs/2402.16843) | [Code](https://github.com/maszhongming/Multi-LoRA-Composition) \
  Ming Zhong, Yelong Shen, Shuohang Wang, Yadong Lu, Yizhu Jiao, Siru Ouyang, Donghan Yu, Jiawei Han, Weizhu Chen
- LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models | [arXiv 2403](https://arxiv.org/abs/2403.11627) | [Code](https://github.com/Young98CN/LoRA_Composer) \
  Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, Wei Liu
- Resadapter: Domain consistent resolution adapter for diffusion models | [arXiv 2403](https://arxiv.org/abs/2403.02084) | [Code](https://github.com/bytedance/res-adapter) \
  Jiaxiang Cheng, Pan Xie, Xin Xia, Jiashi Li, Jie Wu, Yuxi Ren, Huixia Li, Xuefeng Xiao, Min Zheng, Lean Fu
- Implicit Style-Content Separation using B-LoRA | [arXiv 2403](https://arxiv.org/abs/2403.14572) | [Code](https://github.com/yardenfren1996/B-LoRA) \
  Yarden Frenkel, Yael Vinker, Ariel Shamir, Daniel Cohen-Or
- Mixture of Low-rank Experts for Transferable AI-Generated Image Detection | [arXiv 2404](https://arxiv.org/abs/2404.04883) | [Code](https://github.com/zhliuworks/CLIPMoLE) \
  Zihan Liu, Hanyi Wang, Yaoyu Kang, Shilin Wang
- MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection | [arXiv 2404](https://arxiv.org/abs/2404.08452) \
  Chenqi Kong, Anwei Luo, Peijun Bao, Yi Yu, Haoliang Li, Zengwei Zheng, Shiqi Wang, Alex C. Kot
- FouRA: Fourier Low Rank Adaptation | [arXiv 2406](https://arxiv.org/abs/2406.08798) \
  Shubhankar Borse, Shreya Kadambi, Nilesh Prasad Pandey, Kartikeya Bhardwaj, Viswanath Ganapathy, Sweta Priyadarshi, Risheek Garrepalli, Rafael Esteves, Munawar Hayat, Fatih Porikli
- LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation | [ICLR 2025](https://openreview.net/forum?id=6cQ6cBqzV3) \
  Farzad Farhadzadeh, Debasmit Das, Shubhankar Borse, Fatih Porikli
- TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts for Versatile and Efficient Diffusion Models in Vision ｜[ICML 2025](https://arxiv.org/abs/2503.07416) \
  Shaobin Zhuang, Yiwei Guo, Yanbo Ding, Kunchang Li, Xinyuan Chen, Yaohui Wang, Fangyikang Wang, Ying Zhang, Chen Li, Yali Wang
- CtrLoRA: An Extensible and Efficient Framework for Controllable Image Generation | [ICLR 2025](https://openreview.net/forum?id=3Gga05Jdmj) | [Code](https://github.com/xyfJASON/ctrlora.) \
  Yifeng Xu, Zhenliang He, Shiguang Shan, Xilin Chen
- CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion | [arXiv 2602](https://arxiv.org/abs/2602.18936) | CVPR 2026 \
  Yu Li, Yujun Cai, Chi Zhang
- Prompt2Effect: Training-Free LoRA Synthesis for Controllable Video Effects | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/3453) \
  Xiaomeng Yang, Yanyu Li, Gordon Qian, Ivan Skorokhodov, Viacheslav Ivanov, Avalon Vinella, Xuan Zhang, Yanzhi Wang, Sergey Tulyakov, Anil Kag
- CollectionLoRA: Collecting 50 Effects in 1 LoRA for Deployment | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4163) \
  Fangtai Wu, Hailong Guo, Shijie Huang, Jiayi Song, Yubo Huang, Mushui Liu, Zhao Wang, Yunlong Yu, Jiaming Liu, Ruihua Huang
- Spanning the Visual Analogy Space with a Weight Basis of LoRAs | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4237) \
  Hila Manor, Rinon Gal, Haggai Maron, Tomer Michaeli, Gal Chechik
- One4D: Unified 4D Generation and Reconstruction via Decoupled LoRA Control | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4290) \
  Zhenxing Mi, Yuxin Wang, Dan Xu
- In-Context Sync-LoRA for Portrait Video Editing | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4473) \
  Sagi Polaczek, Or Patashnik, Ali Mahdavi-Amiri, Danny Cohen-Or
- AnyStyle: A Single LoRA is Sufficient for Image-Guided Style Transfer | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4851) \
  Yongwen Lai, Chaoqun Wang
- EraseLoRA: MLLM-Driven Foreground Exclusion and Background Subtype Aggregation for Dataset-Free Object Removal | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5267) \
  Sanghyun Jo, Donghwan Lee, Eunji Jung, Seong Je Oh, Kyungsu Kim
- UnGuide: Learning to Forget with LoRA-Guided Diffusion Models | [UAI 2026](https://proceedings.mlr.press/v337/polowczyk26a.html) \
  Alicja Polowczyk, Agnieszka Polowczyk, Dawid Malarz, Artur Kasymov, Jacek Tabor, Marcin Mazur, Przemysław Spurek

### c. Multimodal and Vision-Language

- Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks | [arXiv 2112](https://arxiv.org/pdf/2112.06825.pdf) | [Code](https://github.com/ylsung/VL_adapter) | CVPR 2022 \
  Yi-Lin Sung, Jaemin Cho, Mohit Bansal
- DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback | [arXiv 2311](https://arxiv.org/abs/2311.17946) \
  Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, Cyrus Rashtchian
- Block-wise LoRA: Revisiting Fine-grained LoRA for Effective Personalization and Stylization in Text-to-Image Generation | [arXiv 2304](https://arxiv.org/pdf/2403.07500) | AAAI 2024 \
  Likun Li, Haoqi Zeng, Changpeng Yang, Haozhe Jia, Di Xu
- AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning | [arXiv 2307](https://arxiv.org/pdf/2307.04725) | [Code](https://github.com/guoyww/AnimateDiff) | ICLR 2024 \
  Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai
- Multi-Concept Customization of Text-to-Image Diffusion | [arXiv 2212](https://arxiv.org/pdf/2212.04488) | [Code](https://github.com/adobe-research/custom-diffusion) \
  Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu
- SELMA: Learning and Merging Skill-Specific Text-to-Image Experts with Auto-Generated Data | [arXiv 2403](https://arxiv.org/pdf/2403.06952) | [Code](https://github.com/jialuli-luka/SELMA) \
  Jialu Li, Jaemin Cho, Yi-Lin Sung, Jaehong Yoon, Mohit Bansal
- MACE: Mass Concept Erasure in Diffusion Models | [arXiv 2403](https://arxiv.org/pdf/2403.06135) | [Code](https://github.com/Shilin-LU/MACE) \
  Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, Adams Wai-Kin Kong
- AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models | [arXiv 2404](https://arxiv.org/pdf/2404.13425) \
  Yuheng Ji, Yue Liu, Zhicheng Zhang, Zhao Zhang, Yuting Zhao, Gang Zhou, Xingwei Zhang, Xinwang Liu, Xiaolong Zheng
- Low-Rank Few-Shot Adaptation of Vision-Language Models | [arXiv 2405](https://arxiv.org/abs/2405.18541) | [Code](https://github.com/MaxZanella/CLIP-LoRA) \
  Maxime Zanella, Ismail Ben Ayed
- MoVA: Adapting Mixture of Vision Experts to Multimodal Context | [arXiv 2404](https://arxiv.org/pdf/2404.13046) | [Code](https://github.com/TempleX98/MoVA) \
  Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, Yu Liu
- Customizing 360-degree panoramas through text-to-image diffusion models | [WACV 2024](https://arxiv.org/pdf/2310.18840) | [Code](https://github.com/littlewhitesea/StitchDiffusion) \
  Hai Wang, Xiaoyu Xiang, Yuchen Fan, Jing-Hao Xue
- Space narrative: Generating images and 3d scenes of chinese garden from text using deep learning | [arXiv 2311](https://arxiv.org/pdf/2311.00339) \
  Jiaxi Shi, Hao Hua
- Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning | [ICML 2025](https://openreview.net/forum?id=zpGK1bOlHt) \
  Chendi Ge, Xin Wang, Zeyang Zhang, Hong Chen, Jiapei Fan, Longtao Huang, Hui Xue, Wenwu Zhu
- SV-RAG: LoRA-Contextualizing Adaptation of MLLMs for Long Document Understanding | [ICLR 2025](https://openreview.net/forum?id=FDaHjwInXO) \
  Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A. Rossi, Changyou Chen, Tong Sun
- MMLoP: Multi-Modal Low-Rank Prompting for Efficient Vision-Language Adaptation | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4334) \
  Sajjad Ghiasvand, Haniyeh Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani
- ID-LoRA: Identity-Driven Audio-Video Personalization with In-Context LoRA | [ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/3277) \
  Aviad Dahan, Moran Yanuka, Noa Kraicer, Lior Wolf, Raja Giryes

### d. Speech and Audio

- Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition | [arXiv 2309](https://arxiv.org/pdf/2309.15223.pdf) \
  Yu Yu, Chao-Han Huck Yang, Jari Kolehmainen, Prashanth G. Shivakumar, Yile Gu, Sungho Ryu, Roger Ren, Qi Luo, Aditya Gourav, I-Fan Chen, Yi-Chieh Liu, Tuan Dinh, Ankur Gandhe, Denis Filimonov, Shalini Ghosh, Andreas Stolcke, Ariya Rastow, Ivan Bulyko
- Low-rank Adaptation Method for Wav2vec2-based Fake Audio Detection | [arXiv 2306](https://arxiv.org/pdf/2306.05617.pdf) | CEUR Workshop \
  Chenglong Wang, Jiangyan Yi, Xiaohui Zhang, Jianhua Tao, Le Xu, Ruibo Fu
- Sparsely Shared LoRA on Whisper for Child Speech Recognition | [arXiv 2309](https://arxiv.org/pdf/2309.11756.pdf) | [Code](https://github.com/huggingface/peft) \
  Wei Liu, Ying Qin, Zhiyuan Peng, Tan Lee

### e. Code and Software Engineering

- LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning | [arXiv 2308](https://arxiv.org/pdf/2308.11148.pdf) \
  Junyi Lu, Lei Yu, Xiaojia Li, Li Yang, Chun Zuo
- RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair | [arXiv 2312](https://arxiv.org/abs/2312.15698) | [Code](https://repairllama.github.io) \
  André Silva, Sen Fang, Martin Monperrus
- MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair | [arXiv 2408](https://arxiv.org/pdf/2408.09568) \
  Meghdad Dehghan, Jie JW Wu, Fatemeh H. Fard, Ali Ouni

### f. Scientific, Biomedical, and Physics

**Scientific Discovery and Biomedicine**

- X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design | [APL Machine Learning](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581) \
  Eric L. Buehler, Markus J. Buehler
- ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Post Translational Modification | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.abstract) \
  Amelie Schreiber
- Fine-tuning protein language models boosts predictions across diverse tasks | [Nature Communications](https://www.nature.com/articles/s41467-024-51844-2) \
  Robert Schmirler, Michael Heinzinger, Burkhard Rost
- Parameter-efficient fine-tuning on large protein language models improves signal peptide prediction | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.04.565642v1) \
  Shuai Zeng, Duolin Wang, Dong Xu
- Prollama: A protein large language model for multi-task protein language processing | [arXiv 2402](https://arxiv.org/pdf/2402.16445) \
  Liuzhenghao Lv, Zongying Lin, Hao Li, Yuyang Liu, Jiaxi Cui, Calvin Yu-Chian Chen, Li Yuan, Yonghong Tian
- Structured information extraction from scientific text with large language models | [Nature Communications](https://www.nature.com/articles/s41467-024-45563-x) \
  John Dagdelen, Alexander Dunn, Sanghoon Lee, Nicholas Walker, Andrew S. Rosen, Gerbrand Ceder, Kristin A. Persson, Anubhav Jain
- GeoSFLoRA: Geometry-Conditioned Spectral Flow Low-Rank Adaptation for 2D-to-3D Transfer in Medical Image Segmentation | [IJCAI 2026](https://www.ijcai.org/proceedings/2026/748) \
  Qin Hao, Bonian Chen, Shengwei Tian, Long Yu

**Scientific Computing and PDEs**

- PIHLoRA: Physics-informed hypernetworks for low-ranked adaptation | [NeurIPS 2023](https://openreview.net/pdf?id=kupYlLLGdf) \
  Ritam Majumdar, Vishal Sudam Jadhav, Anirudh Deodhar, Shirish Karande, Lovekesh Vig, Venkataramana Runkana

### g. Structured Data: Graphs and Recommendation

**Graph Learning**

- GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning | [arXiv 2409](https://arxiv.org/pdf/2409.16670) \
  Zhe-Rui Yang, Jindong Han, Chang-Dong Wang, Hao Liu
- Fast and Continual Knowledge Graph Embedding via Incremental LoRA | [arXiv 2407](https://arxiv.org/pdf/2407.05705) | [Code](https://github.com/seukgcode/FastKGE) | IJCAI 2024 \
  Jiajun Liu, Wenjun Ke, Peng Wang, Jiahao Wang, Jinhua Gao, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji, Yining Li
- ELoRA: Low-Rank Adaptation for Equivariant GNNs |[ICML 2025](https://openreview.net/forum?id=hcoxm3Vwgy) | [Code](https://github.com/hyjwpk/ELoRA)\
  Chen Wang, Siyu Hu, Guangming Tan, Weile Jia
- Graph Cross-Domain Continual Fine-Tuning via Orthogonal LoRA Routing with Contrastive Expert Specialization | [WWW 2026 Accepted Papers](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Qianyi Cai, Ziyue Qiao, Minghao Yang, Xiao Luo, Hui Xiong
- GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation | [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.645/) \
  Lin Mu, Guoji Wang, Li Ni, Lei Sang, Zhize Wu, Peiquan Jin, Yiwen Zhang

**Recommendation**

- Customizing Language Models with Instance-wise LoRA for Sequential Recommendation | [arXiv 2408](https://arxiv.org/pdf/2408.10159) \
  Xiaoyu Kong, Jiancan Wu, An Zhang, Leheng Sheng, Hui Lin, Xiang Wang, Xiangnan He
- Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation | [arXiv 2408](https://arxiv.org/pdf/2408.03533) \
  Jiachen Zhu, Jianghao Lin, Xinyi Dai, Bo Chen, Rong Shan, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang
- MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction | [arXiv 2408](https://arxiv.org/pdf/2408.08913) | [Code](https://github.com/gaohaining/MLoRA) \
  Zhiming Yang, Haining Gao, Dehong Gao, Luwei Yang, Libin Yang, Xiaoyan Cai, Wei Ning, Guannan Zhang
- ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Model | [arXiv 2409](https://arxiv.org/pdf/2409.08543) | [MDPI](https://www.mdpi.com/2227-7390/11/16/3577) \
  Zezheng Qin
- LoRA-NCL: Neighborhood-Enriched Contrastive Learning with Low-Rank Dimensionality Reduction for Graph Collaborative Filtering | [Mathematics 2023](https://doi.org/10.3390/math11163577) \
  Tianruo Cao, Honghui Chen, Zepeng Hao, Tao Hu
- LoRA for Sequential Recommendation Harnessing large language models for text-rich sequential recommendation | [arXiv 2403](https://arxiv.org/pdf/2403.13325) | [Code](https://github.com/zhengzhi-1997/LLM-TRSR) | WWW 2024 \
  Zhi Zheng, Wenshuo Chao, Zhaopeng Qiu, Hengshu Zhu, Hui Xiong
- Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation | [arXiv 2602](https://arxiv.org/abs/2602.09445) | WWW 2026 \
  Sunwoo Kim, Hyunjin Hwang, Kijung Shin
- RAIE: Region-Aware Incremental Preference Editing with LoRA for LLM-based Recommendation | [arXiv 2603](https://arxiv.org/abs/2603.00638) | WWW 2026 \
  Jin Zeng, Yupeng Qi, Hui Li, Chengming Li, Ziyu Lyu, Lixin Cui, Lu Bai
- PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering | [WWW 2026](https://www2026.thewebconf.org/accepted/research-tracks.html) \
  Doyun Choi, Cheonwoo Lee, Biniyam Aschalew Tolera, Taewook Ham, Chanyoung Park, Jaemin Yoo

### h. Time Series and Forecasting

- Low-rank Adaptation for Spatio-Temporal Forecasting | [arXiv 2404](https://arxiv.org/abs/2404.07919) | [Code](https://github.com/RWLinno/ST-LoRA) \
  Weilin Ruan, Wei Chen, Xilin Dang, Jianxiang Zhou, Weichuang Li, Xu Liu, Yuxuan Liang
- Channel-Aware Low-Rank Adaptation in Time Series Forecasting | [arXiv 2407](https://arxiv.org/pdf/2407.17246) | [Code](https://github.com/tongnie/C-LoRA) \
  Tong Nie, Yuewen Mei, Guoyang Qin, Jian Sun, Wei Ma
- Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting | [arXiv 2405](https://arxiv.org/abs/2405.10216) \
  Divij Gupta, Anubhav Bhatti, Suraj Parmar, Chen Dan, Yuwei Liu, Bingjie Shen, San Lee
- Mixture of Low Rank Adaptation with Partial Parameter Sharing for Time Series Forecasting | [arXiv 2505](https://arxiv.org/abs/2505.17872) \
  Licheng Pan, Zhichao Chen, Haoxuan Li, Guangyi Liu, Zhijian Xu, Zhaoran Liu, Hao Wang, Ying Wei

### i. Emerging Applications

**Anomaly Detection**

- Parameter-Efficient Log Anomaly Detection based on Pre-training model and LoRA | [Zenodo](https://zenodo.org/records/8270065) \
  Shiming He, Ying Lei, Ying Zhang, Kun Xie, Pradip Kumar Sharma

**Reinforcement Learning and Agents**

- Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent | [arXiv 2402](https://arxiv.org/pdf/2402.13717.pdf) | [Code](https://github.com/weiyifan1023/Neeko) \
  Xiaoyan Yu, Tongxu Luo, Yifan Wei, Fangyu Lei, Yiming Huang, Hao Peng, Liehuang Zhu
- Handling coexistence of LoRA with other networks through embedded reinforcement learning | [ACM](https://dl.acm.org/doi/abs/10.1145/3576842.3582383) \
  Sezana Fahmida, Venkata Prashant Modekurthy, Mahbubur Rahman, Abusayeed Saifullah

## 4. Resource

- Bayesian Adaptation Gym: A Benchmark for the Bayesian Low-Rank Adaptation of Multi-Modal Language Models | [UAI 2026](https://proceedings.mlr.press/v337/samplawski26a.html) \
  Colin Samplawski, Ramneet Kaur, Manoj Acharya, Anirban Roy, Adam D. Cobb
- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models | [arXiv 2304](https://arxiv.org/pdf/2304.01933.pdf) | [Code](https://github.com/AGI-Edgerunners/LLM-Adapters) \
  Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, Roy Ka-Wei Lee
- Run LoRA Run: Faster and Lighter LoRA Implementations | [arXiv 2312](https://arxiv.org/pdf/2312.03415.pdf) \
  Daria Cherniuk, Aleksandr Mikhalev, Ivan Oseledets
- Large language model LoRA specifically fine-tuned for medical domain tasks | [Code](https://huggingface.co/nmitchko/medfalcon-40b-LoRA)
