# 🚀 Awesome LoRA Adapter

# Low-rank Adaptation for Foundation Models: Foundations and Frontiers

> This repository is based on the survey paper: [Low-Rank Adaptation for Foundation Models: A Comprehensive Review](https://arxiv.org/abs/2501.00365) by Menglin Yang, Jialin Chen, Yifei Zhang, Jiahong Liu, Jiasheng Zhang, Qiyao Ma, Harshit Verma, Qianru Zhang, Min Zhou, Irwin King, Rex Ying.

## Table of Contents

1. [Foundations of LoRA](#1-foundations-of-lora)
   - [a. Parameter Efficiency](#a-parameter-efficiency)
   - [b. Ranking Adaptation](#b-ranking-adaptation)
   - [c. Learning Process](#c-learning-process)
   - [d. Theoretical Foundations](#d-theoretical-foundations)

2. [Frontiers of LoRA](#2-frontiers-of-lora)
   - [a. Advanced Structures](#a-advanced-structures)
   - [b. LoRA for Long Sequence Modeling](#b-lora-for-long-sequence-modeling)
   - [c. LoRA for Continue Learning](#c-lora-for-continue-learning)
   - [d. LoRA for Federated Learning](#d-lora-for-federated-learning)

3. [Applications](#3-applications)
   - [LoRA in Natural Language Processing](#lora-in-natural-language-processing)
   - [LoRA in Computer Vision](#lora-in-computer-vision)
   - [LoRA in Multimodal Learning](#lora-in-multimodal-learning)
   - [LoRA in Speech Processing](#lora-in-speech-processing)
   - [LoRA in Code Engineering](#lora-in-code-engineering)
   - [LoRA in Scientific Discovery](#lora-in-scientific-discovery)
   - [LoRA in Time Series](#lora-in-time-series)
   - [LoRA in Graph Learning](#lora-in-graph-learning)
   - [LoRA in Recommender System](#lora-in-recommender-system)
   - [LoRA in Anomaly Detection](#lora-in-anomaly-detection)
   - [LoRA in PDE](#lora-in-pde)
   - [LoRA in RL](#lora-in-rl)
   - [LoRA for Pretraining](#lora-for-pretraining)
   - [LoRA Serving System](#lora-serving-system)

4. [Resource](#4-resource)

## 1. Foundations of LoRA

### a. Parameter Efficiency

**(i) Parameter Decomposition**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY) \
Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen Tuo Zhao 
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arXiv 2403](https://arxiv.org/abs/2403.13037) \
Rushi Qiang, Ruiyi Zhang, Pengtao Xie 
- LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models | [arXiv 2402](https://arxiv.org/pdf/2402.11417.pdf) | [Code](https://github.com/yifanycc/loretta) | NAACL 2024 \
Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang 
- LoTR: Low Tensor Rank Weight Adaptation | [arXiv 2402](https://arxiv.org/pdf/2402.01376.pdf) | [Code](https://github.com/daskol/lotr) \
Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, Ivan Oseledets
- Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs | [arXiv 2408](https://arxiv.org/pdf/2408.01008) \
Afia Anjum, Maksim E. Eren, Ismael Boureima, Boian Alexandrov, Manish Bhattarai
- DoRA: Weight-Decomposed Low-Rank Adaptation | [arXiv 2402](https://arxiv.org/pdf/2402.09353.pdf) | [Code](https://github.com/NVlabs/DoRA) | ICML 2024 \
Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen

**(ii) Parameter Selection**

- SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters | [Findings of EMNLP 2022](https://arxiv.org/abs/2210.04284) | [Code](https://github.com/Shwai-He/SparseAdapter) \
Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao 
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arXiv 2311](https://arxiv.org/pdf/2311.11696.pdf) | [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023 \
Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun
- Asymmetry in Low-Rank Adapters of Foundation Models | [arXiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA) | [Code](https://github.com/NVIDIA/NeMo/tree/adithyare/vera) \
Jiacheng Zhu, Kristjan Greenewald, Kimia Nadjahi, Haitz Sáez de Ocáriz Borde, Rickard Brüel Gabrielsson, Leshem Choshen, Marzyeh Ghassemi, Mikhail Yurochkin, Justin Solomon
- LoRA-FA: Memory-efficient low-rank adaptation for large language models fine-tuning | [arXiv 2308](https://arxiv.org/pdf/2308.03303.pdf) \
Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, Bo Li
- LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation | [arXiv 2402](https://arxiv.org/pdf/2402.07721.pdf) \
Hongyun Zhou, Xiangyu Lu, Wang Xu, Conghui Zhu, Tiejun Zhao, Muyun Yang

**(iii) Parameter Sharing**

- VeRA: Vector-based Random Matrix Adaptation | [ICLR 2024](https://openreview.net/forum?id=NjNfLdxr3A) \
Dawid J. Kopiczko, Tijmen Blankevoort, Yuki M. Asano
- Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying | [arXiv 2311](https://arxiv.org/pdf/2311.09578) \
Adithya Renduchintala, Tugrul Konuk, Oleksii Kuchaiev
- NOLA: Networks as linear combination of low rank random basis | [arXiv 2310](https://arxiv.org/pdf/2310.02556.pdf) | [Code](https://github.com/UCDvision/NOLA) | ICLR 2024 \
Soroush Abbasi Koohpayegani, KL Navaneet, Parsa Nooralinejad, Soheil Kolouri, Hamed Pirsiavash
- Delta-LoRA: Fine-tuning high-rank parameters with the delta of low-rank matrices | [arXiv 2309](https://arxiv.org/pdf/2309.02411.pdf) \
Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, Lei Zhang

**(iv) Parameter Quantization**

- QLoRA: Efficient finetuning of quantized llms | [arXiv 2305](https://arxiv.org/pdf/2305.14314.pdf) | [Code](https://github.com/artidoro/qLoRA) | NeurIPS 2023 \
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- Qa-LoRA: Quantization-aware low-rank adaptation of large language models | [NeurIPS 2023](https://arxiv.org/pdf/2309.14717.pdf) | [Code](https://github.com/yuhuixu1993/qa-LoRA) \
Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian
- QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning | [arXiv 2402](https://arxiv.org/pdf/2402.10462.pdf) \
- Loftq: LoRA-fine-tuning-aware quantization for large language models | [arXiv 2310](https://arxiv.org/pdf/2310.08659.pdf) | [Code](https://github.com/yxli2123/LoftQ) \
Hossein Rajabzadeh, Mojtaba Valipour, Tianshu Zhu, Marzieh Tahaei, Hyock Ju Kwon, Ali Ghodsi, Boxing Chen, Mehdi Rezagholizadeh
- Lq-LoRA: Low-rank plus quantized matrix decomposition for efficient language model finetuning | [arXiv 2311](https://arxiv.org/pdf/2311.12023.pdf) | [Code](https://github.com/HanGuo97/lq-LoRA) \
Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim
- LQER: Low-Rank Quantization Error Reconstruction for LLMs | [arXiv 2402](https://arxiv.org/pdf/2402.02446.pdf) | [Code](https://github.com/OpenGVLab/OmniQuant) | ICLR 2024 \
Cheng Zhang, Jianyi Cheng, George A. Constantinides, Yiren Zhao

### b. Ranking Adaptation

**(i) Ranking Refinement**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY) \
Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | [arXiv 2403](https://arxiv.org/pdf/2403.13037v1) \
Rushi Qiang, Ruiyi Zhang, Pengtao Xie
- DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation | [EACL](https://arxiv.org/abs/2210.07558) | [Code](https://github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA) \
Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, Ali Ghodsi
- PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation | [arXiv 2401](https://arxiv.org/pdf/2401.11316.pdf) \
Nadav Benedek, Lior Wolf
- Sparse Low-rank Adaptation of Pre-trained Language Models | [arXiv 2311](https://arxiv.org/pdf/2311.11696.pdf) | [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023 \
Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun
  
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
  
### c. Learning Process

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

#### **(v) Post-hoc Processing**

- Bayesian Low-rank Adaptation for Large Language Models | [arXiv 2308](https://arxiv.org/abs/2308.13111) | [Code](https://github.com/adamxyang/laplace-lora) | ICLR 2024 \
Adam X. Yang, Maxime Robeyns, Xi Wang, Laurence Aitchison

### d. Theoretical Foundations

- The Expressive Power of Low-Rank Adaptation | [arXiv 2310](https://arxiv.org/pdf/2310.17513.pdf) | [Code](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) | ICLR 2024 \
Yuchen Zeng, Kangwook Lee
- LoRA Training in the NTK Regime has No Spurious Local Minima | [arXiv 2402](https://arxiv.org/pdf/2402.11867.pdf) | [Code](https://github.com/UijeongJang/LoRA-NTK) | ICML 2024 \
Uijeong Jang, Jason D. Lee, Ernest K. Ryu
- ROSA: Random Orthogonal Subspace Adaptation | [ICML 2023](https://openreview.net/pdf?id=4P9vOFpb63) | [Code](https://github.com/marawangamal/rosa) \
Marawan Gamal, Guillaume Rabusseau
- Asymmetry in Low-Rank Adapters of Foundation Models | [arXiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA) \
Jiacheng Zhu, Kristjan Greenewald, Kimia Nadjahi, Haitz Sáez de Ocáriz Borde, Rickard Brüel Gabrielsson, Leshem Choshen, Marzyeh Ghassemi, Mikhail Yurochkin, Justin Solomon

## 2. Frontiers of LoRA

### a. Advanced Structures

**LoRA Composition**
- Adaptersoup: Weight averaging to improve generalization of pretrained language models | [arXiv 2302](https://arxiv.org/pdf/2302.07027) | [Code](https://github.com/UKPLab/sentence-transformers) \
Alexandra Chronopoulou, Matthew E. Peters, Alexander Fraser, Jesse Dodge
- LoRAhub: Efficient cross-task generalization via dynamic LoRA composition | [arXiv 2307](https://arxiv.org/pdf/2307.13269.pdf) | [Code](https://github.com/sail-sg/LoRAhub) | COLM 2024 \
Alexandra Chronopoulou, Matthew E. Peters, Alexander Fraser, Jesse Dodge
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

**LoRA MoE**

- MoeLoRA: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models | [arXiv 2402](https://arxiv.org/pdf/2402.12851.pdf) \
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

### b. LoRA for Long Sequence Modeling

- LongLoRA: Efficient fine-tuning of long-context large language models | [arXiv 2309](https://arxiv.org/pdf/2309.12307.pdf) | [Code](https://github.com/dvlab-research/LongLoRA) | ICLR 2024 \
Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia
- LongqLoRA: Efficient and effective method to extend context length of large language models | [arXiv 2311](https://arxiv.org/pdf/2311.04879.pdf) | [Code](https://github.com/yangjianxin1/LongQLoRA) \
Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia
- With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation | [arXiv 2401](https://arxiv.org/abs/2401.11504) | [Code](https://github.com/TemporaryLoRA/Temp-LoRA/tree/main) | COLM 2024 \
Y. Wang, D. Ma, D. Cai
- RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization | [arXiv 2405](https://arxiv.org/abs/2405.00657) \
Dongqi Pu, Vera Demberg

### c. LoRA for Continue Learning
- Orthogonal Subspace Learning for Language Model Continual Learning | [EMNLP 2023 findings](https://arxiv.org/pdf/2310.14152) | [Code](https://github.com/cmnfriend/O-LoRA) \
Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, Xuanjing Huang
- Continual Learning with Low Rank Adaptation | [NeurIPS 2023 Workshop](https://arxiv.org/pdf/2311.17601) \
Martin Wistuba, Prabhu Teja Sivaprasad, Lukas Balles, Giovanni Zappella 
- Task Arithmetic with LoRA for Continual Learning | [NeurIPS 2023 Workshop](https://arxiv.org/pdf/2311.02428) \
Rajas Chitale, Ankit Vaidya, Aditya Kane, Archana Ghotkar
- A Unified Continual Learning Framework with General Parameter-Efficient Tuning | [ICCV 2023](https://arxiv.org/pdf/2303.10070) | [Code](https://github.com/gqk/LAE) \
Qiankun Gao, Chen Zhao, Yifan Sun, Teng Xi, Gang Zhang, Bernard Ghanem, Jian Zhang 
  
### d. LoRA for Federated Learning

- SLoRA: Federated parameter efficient fine-tuning of language models | [arxiv 2308](https://arxiv.org/pdf/2308.06522.pdf) \
Sara Babakniya, Ahmed Roushdy Elkordy, Yahya H. Ezzeldin, Qingfeng Liu, Kee-Bong Song, Mostafa El-Khamy, Salman Avestimehr
- pFedLoRA: Model-heterogeneous personalized federated learning with LoRA tuning | [arxiv 2310](https://arxiv.org/pdf/2310.13283.pdf) \
Liping Yi, Han Yu, Gang Wang, Xiaoguang Liu, Xiaoxiao Li 
- Improving LoRA in Privacy-preserving Federated Learning | [OpenReview](https://openreview.net/pdf?id=NLPzL6HWNl) | ICLR 2024 \
Youbang Sun, Zitao Li, Yaliang Li, Bolin Ding 
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
- DP-DyLoRA: Fine-Tuning Transformer-Based Models On-Device under Differentially Private Federated Learning using Dynamic Low-Rank Adaptation | [arxiv 2405](https://arxiv.org/abs/2405.06368) \
Jie Xu, Karthikeyan Saravanan, Rogier van Dalen, Haaris Mehmood, David Tuckey, Mete Ozay 
- FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning | [arxiv 2406](https://arxiv.org/pdf/2406.07925) \
Jiaxing QI, Zhongzhi Luan, Shaohan Huang, Carol Fung, Hailong Yang, Depei Qian 
- FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations | [arxiv 2409](https://arxiv.org/pdf/2409.05976) [Code](https://github.com/ATP-1010/FederatedLLM) \
Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, Ang Li 
- Automated Federated Pipeline for Parameter-Efficient Fine-Tuning of Large Language Models | [arxiv 2404](https://arxiv.org/pdf/2404.06448) \
Zihan Fang, Zheng Lin, Zhe Chen, Xianhao Chen, Yue Gao, Yuguang Fang 
  
## 3. Applications

### LoRA in Natural Language Processing

- Machine Translation with Large Language Models: Prompting, Few-shot Learning, and Fine-tuning with QLoRA | [ACL 2023](https://aclanthology.org/2023.wmt-1.43.pdf) \
Xuan Zhang, Navid Rajabi, Kevin Duh, Philipp Koehn
- Task-Agnostic Low-Rank Adapters for Unseen English Dialects | [ACL 2023](https://aclanthology.org/2023.emnlp-main.487.pdf) | [Code](https://github.com/zedian/hyperLoRA) \
Zedian Xiao, William Held, Yanchen Liu, Diyi Yang
- LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training | [arXiv 2401](https://arxiv.org/pdf/2401.04348.pdf) | [Code](https://github.com/VinAIResearch/LAMPAT) | AAAI 2024 \
Khoi M.Le, Trinh Pham, Tho Quan, Anh Tuan Luu
- Task Arithmetic with LoRA for Continual Learning | [arXiv 2311](https://arxiv.org/pdf/2311.02428.pdf) | NeurIPS 2023 Workshop \
Rajas Chitale, Ankit Vaidya, Aditya Kane, Archana Ghotkar

### LoRA in Computer Vision

**Visual Understanding**

**(1) Domain Adaptation and Transfer Learning**

- Motion style transfer: Modular low-rank adaptation for deep motion forecasting | [arXiv 2211](https://arxiv.org/pdf/2211.03165.pdf) | [Code](https://github.com/vita-epfl/motion-style-transfer) \
Parth Kothari, Danya Li, Yuejiang Liu, Alexandre Alahi
- Efficient low-rank backpropagation for vision transformer adaptation | [arXiv 2309](https://arxiv.org/pdf/2309.15275.pdf) | NeurIPS 2023 \
Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | [arXiv 2402](https://arxiv.org/pdf/2402.04964.pdf) | [Code](https://github.com/aleemsidra/ConvLoRA) \
Sidra Aleem, Julia Dietlmeier, Eric Arazo, Suzanne Little
- ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | [arXiv 2406](https://arxiv.org/abs/2406.10973) \
Samar Khanna, Medhanie Irgau, David B. Lobell, Stefano Ermon
- Melo: Low-rank adaptation is better than fine-tuning for medical image diagnosis | [arXiv 2311](https://arxiv.org/pdf/2311.08236.pdf) | [Code](https://github.com/JamesQFreeman/LoRA-ViT) \
Yitao Zhu, Zhenrong Shen, Zihao Zhao, Sheng Wang, Xin Wang, Xiangyu Zhao, Dinggang Shen, Qian Wang
- Enhancing General Face Forgery Detection via Vision Transformer with Low-Rank Adaptation | [arXiv 2303](https://arxiv.org/pdf/2303.00917.pdf) \
Yitao Zhu, Zhenrong Shen, Zihao Zhao, Sheng Wang, Xin Wang, Xiangyu Zhao, Dinggang Shen, Qian Wang

**(2) Semantic Segmentation**

- Customized Segment Anything Model for Medical Image Segmentation | [arXiv 2304](https://arxiv.org/abs/2304.13785) | [Code](https://github.com/hitachinsk/SAMed) \
Kaidong Zhang, Dong Liu
- SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation | [MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_23) \
An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, Hongliang Ren
- Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model | [Code](https://github.com/autogluon/autogluon/tree/master/examples/automm/Conv-LoRA) \
An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, Hongliang Ren

**(3) Others**

- FullLoRA-AT: Efficiently Boosting the Robustness of Pretrained Vision Transformers | [arXiv 2401](https://arxiv.org/pdf/2401.01752.pdf) \
Zheng Yuan, Jie Zhang, Shiguang Shan
- Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach | [arXiv 2403](https://arxiv.org/abs/2403.19067) | [Code](https://github.com/zstarN70/RLRR) \
Wei Dong, Xing Zhang, Bihui Chen, Dawei Yan, Zhijun Lin, Qingsen Yan, Peng Wang, Yang Yang
- LORTSAR: Low-Rank Transformer for Skeleton-based Action Recognition | [arXiv 2407](https://arxiv.org/abs/2407.14655) \
Soroush Oraki, Harry Zhuang, Jie Liang
- Parameter-efficient Model Adaptation for Vision Transformers | [arXiv 2203](https://arxiv.org/pdf/2203.16329.pdf) | [Code](https://github.com/eric-ai-lab/PEViT) | AAAI 2023 \
Xuehai He, Chunyuan Li, Pengchuan Zhang, Jianwei Yang, Xin Eric Wang

**Visual Generation**

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
- Low-Rank Few-Shot Adaptation of Vision-Language Models | [arXiv 2405](https://arxiv.org/abs/2405.18541) | [Code](https://github.com/MaxZanella/CLIP-LoRA) \
Maxime Zanella, Ismail Ben Ayed
- FouRA: Fourier Low Rank Adaptation | [arXiv 2406](https://arxiv.org/abs/2406.08798) \
Shubhankar Borse, Shreya Kadambi, Nilesh Prasad Pandey, Kartikeya Bhardwaj, Viswanath Ganapathy, Sweta Priyadarshi, Risheek Garrepalli, Rafael Esteves, Munawar Hayat, Fatih Porikli

### LoRA in Multimodal Learning

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
- MoVA: Adapting Mixture of Vision Experts to Multimodal Context | [arXiv 2404](https://arxiv.org/pdf/2404.13046) | [Code](https://github.com/TempleX98/MoVA) \
Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, Yu Liu
- Customizing 360-degree panoramas through text-to-image diffusion models | [WACV 2024](https://arxiv.org/pdf/2310.18840) | [Code](https://github.com/littlewhitesea/StitchDiffusion) \
Hai Wang, Xiaoyu Xiang, Yuchen Fan, Jing-Hao Xue
- Space narrative: Generating images and 3d scenes of chinese garden from text using deep learning | [arXiv 2311](https://arxiv.org/pdf/2311.00339) \
Jiaxi Shi, Hao Hua

### LoRA in Speech Processing

- Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition | [arXiv 2309](https://arxiv.org/pdf/2309.15223.pdf) \
Yu Yu, Chao-Han Huck Yang, Jari Kolehmainen, Prashanth G. Shivakumar, Yile Gu, Sungho Ryu, Roger Ren, Qi Luo, Aditya Gourav, I-Fan Chen, Yi-Chieh Liu, Tuan Dinh, Ankur Gandhe, Denis Filimonov, Shalini Ghosh, Andreas Stolcke, Ariya Rastow, Ivan Bulyko
- Low-rank Adaptation Method for Wav2vec2-based Fake Audio Detection | [arXiv 2306](https://arxiv.org/pdf/2306.05617.pdf) | CEUR Workshop \
Chenglong Wang, Jiangyan Yi, Xiaohui Zhang, Jianhua Tao, Le Xu, Ruibo Fu
- Sparsely Shared LoRA on Whisper for Child Speech Recognition | [arXiv 2309](https://arxiv.org/pdf/2309.11756.pdf) | [Code](https://github.com/huggingface/peft) \
Wei Liu, Ying Qin, Zhiyuan Peng, Tan Lee

### LoRA in Code Engineering

- LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning | [arXiv 2308](https://arxiv.org/pdf/2308.11148.pdf) \
Junyi Lu, Lei Yu, Xiaojia Li, Li Yang, Chun Zuo
- RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair | [arXiv 2312](https://arxiv.org/abs/2312.15698) | [Code](https://repairllama.github.io) \
André Silva, Sen Fang, Martin Monperrus
- MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair | [arXiv 2408](https://arxiv.org/pdf/2408.09568) \
Meghdad Dehghan, Jie JW Wu, Fatemeh H. Fard, Ali Ouni

### LoRA in Scientific Discovery

- X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design | [APL Machine Learning](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581) \
Eric L. Buehler, Markus J. Buehler
- ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Post Translational Modification | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.abstract) \
Amelie Schreiber
- Fine-tuning protein language models boosts predictions across diverse tasks | [Nature Communication](https://www.nature.com/articles/s41467-024-51844-2) \
Robert Schmirler, Michael Heinzinger, Burkhard Rost
- Parameter-efficient fine-tuning on large protein language models improves signal peptide prediction | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.04.565642v1) \
Shuai Zeng, Duolin Wang, Dong Xu
- Prollama: A protein large language model for multi-task protein language processing | [arXiv 2402](https://arxiv.org/pdf/2402.16445) \
Liuzhenghao Lv, Zongying Lin, Hao Li, Yuyang Liu, Jiaxi Cui, Calvin Yu-Chian Chen, Li Yuan, Yonghong Tian

### LoRA in Time Series

- Low-rank Adaptation for Spatio-Temporal Forecasting | [arXiv 2404](https://arxiv.org/abs/2404.07919) | [Code](https://github.com/RWLinno/ST-LoRA) \
Weilin Ruan, Wei Chen, Xilin Dang, Jianxiang Zhou, Weichuang Li, Xu Liu, Yuxuan Liang
- Channel-Aware Low-Rank Adaptation in Time Series Forecasting | [arXiv 2407](https://arxiv.org/pdf/2407.17246) | [Code](https://github.com/tongnie/C-LoRA) \
Tong Nie, Yuewen Mei, Guoyang Qin, Jian Sun, Wei Ma
- Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting | [arXiv 2405](https://arxiv.org/abs/2405.10216) \
Divij Gupta, Anubhav Bhatti, Suraj Parmar, Chen Dan, Yuwei Liu, Bingjie Shen, San Lee

### LoRA in Graph Learning

- GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning | [arXiv 2409](https://arxiv.org/pdf/2409.16670) \
Zhe-Rui Yang, Jindong Han, Chang-Dong Wang, Hao Liu
- Fast and Continual Knowledge Graph Embedding via Incremental LoRA | [arXiv 2407](https://arxiv.org/pdf/2407.05705) | [Code](https://github.com/seukgcode/FastKGE) | IJCAI 2024 \
Jiajun Liu, Wenjun Ke, Peng Wang, Jiahao Wang, Jinhua Gao, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji, Yining Li

### LoRA in Recommender System

- Customizing Language Models with Instance-wise LoRA for Sequential Recommendation | [arXiv 2408](https://arxiv.org/pdf/2408.10159) \
Xiaoyu Kong, Jiancan Wu, An Zhang, Leheng Sheng, Hui Lin, Xiang Wang, Xiangnan He
- Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation | [arXiv 2408](https://arxiv.org/pdf/2408.03533) \
Jiachen Zhu, Jianghao Lin, Xinyi Dai, Bo Chen, Rong Shan, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang
- MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction | [arXiv 2408](https://arxiv.org/pdf/2408.08913) | [Code](https://github.com/gaohaining/MLoRA) \
Zhiming Yang, Haining Gao, Dehong Gao, Luwei Yang, Libin Yang, Xiaoyan Cai, Wei Ning, Guannan Zhang
- ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Model | [arXiv 2409](https://arxiv.org/pdf/2409.08543) | [MDPI](https://www.mdpi.com/2227-7390/11/16/3577) \
Zezheng Qin
- LoRA-NCL: Neighborhood-Enriched Contrastive Learning with Low-Rank Dimensionality Reduction for Graph Collaborative Filtering | [arXiv 2403](https://arxiv.org/pdf/2403.13325) | [Code](https://github.com/zhengzhi-1997/LLM-TRSR) | WWW 2024 \
Tianruo Cao, Honghui Chen, Zepeng Hao
- LoRA for Sequential Recommendation Harnessing large language models for text-rich sequential recommendation | [arXiv 2403](https://arxiv.org/pdf/2403.13325) | [Code](https://github.com/zhengzhi-1997/LLM-TRSR) | WWW 2024 \
Zhi Zheng, Wenshuo Chao, Zhaopeng Qiu, Hengshu Zhu, Hui Xiong

### LoRA in Anomaly Detection

- Parameter-Efficient Log Anomaly Detection based on Pre-training model and LoRA | [Zenodo](https://zenodo.org/records/8270065) \
Shiming He, Ying Lei, Ying Zhang, Kun Xie, Pradip Kumar Sharma

### LoRA in PDE

- PIHLoRA: Physics-informed hypernetworks for low-ranked adaptation | [NeurIPS 2023](https://openreview.net/pdf?id=kupYlLLGdf) \
Ritam Majumdar, Vishal Sudam Jadhav, Anirudh Deodhar, Shirish Karande, Lovekesh Vig, Venkataramana Runkana

### LoRA in RL

- Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent | [arXiv 2402](https://arxiv.org/pdf/2402.13717.pdf) | [Code](https://github.com/weiyifan1023/Neeko) \
Xiaoyan Yu, Tongxu Luo, Yifan Wei, Fangyu Lei, Yiming Huang, Hao Peng, Liehuang Zhu
- Handling coexistence of LoRA with other networks through embedded reinforcement learning | [ACM](https://dl.acm.org/doi/abs/10.1145/3576842.3582383) \
Sezana Fahmida, Venkata Prashant Modekurthy, Mahbubur Rahman, Abusayeed Saifullah

### LoRA for Pretraining

- Training Neural Networks from Scratch with Parallel Low-Rank Adapters | [arXiv 2402](https://arxiv.org/pdf/2402.16828.pdf) | [Code](https://github.com/minyoungg/LTE) \
Minyoung Huh, Brian Cheung, Jeremy Bernstein, Phillip Isola, Pulkit Agrawal

### LoRA Serving System

- Peft: State-of-the-art parameter-efficient fine-tuning methods | [Huggingface](https://github.com/huggingface/peft) \
Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, Fu Lee Wang
- S-LoRA: Serving thousands of concurrent LoRA adapters | [arXiv 2311](https://arxiv.org/pdf/2311.03285.pdf) | [Code](https://github.com/S-LoRA/S-LoRA) | MLSys Conference 2024 \
Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica
- CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference | [arXiv 2401](https://arxiv.org/pdf/2401.11240.pdf) \
Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Qizhen Weng, Xusheng Chen, Yizhou Shan, Binhang Yuan, Wei Wang
- Local LoRA: Memory-Efficient Fine-Tuning of Large Language Models | [OpenReview](https://openreview.net/pdf?id=LHKmzWP7RN) | WANT@NeurIPS 2023 \
Oscar Key, Jean Kaddour, Pasquale Minervini

## 4. Resource

- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models | [arXiv 2304](https://arxiv.org/pdf/2304.01933.pdf) | [Code](https://github.com/AGI-Edgerunners/LLM-Adapters) \
Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, Roy Ka-Wei Lee
- Run LoRA Run: Faster and Lighter LoRA Implementations | [arXiv 2312](https://arxiv.org/pdf/2312.03415.pdf) \
Daria Cherniuk, Aleksandr Mikhalev, Ivan Oseledets
- Large language model LoRA specifically fine-tuned for medical domain tasks | [Code](https://huggingface.co/nmitchko/medfalcon-40b-LoRA)

## Citation

If you find this repository useful, please cite our survey paper:

```bibtex
@article{yang2024low,
  title={Low-Rank Adaptation for Foundation Models: A Comprehensive Review},
  author={Yang, Menglin and Chen, Jialin and Zhang, Yifei and Liu, Jiahong and Zhang, Jiasheng and Ma, Qiyao and Verma, Harshit and Zhang, Qianru and Zhou, Min and King, Irwin and Ying, Rex},
  journal={arXiv preprint arXiv:2501.00365},
  year={2024}
}
```

## Contributing

If you find any LoRA-related papers that are not included in this repository, we welcome your contributions! You can:
1. Open an issue to report the missing paper
2. Submit a pull request to add the paper to the appropriate section

Your contributions will help keep this repository comprehensive and up-to-date.
