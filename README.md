
# Low-rank Adaptation for Foundation Models: Foundations and Frontiers

## 1. Foundations of LoRA

### a. Parameter Efficiency

**(i) Parameter Decomposition**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning|  [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY) \\ Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen Tuo Zhao 
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | Rushi Qiang, Ruiyi Zhang, Pengtao Xie | [arxiv 2403](https://arxiv.org/abs/2403.13037)
- LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models | Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang |[arxiv 2402](https://arxiv.org/pdf/2402.11417.pdf)| [Code](https://github.com/yifanycc/loretta) | NAACL 2024 Oral
- LoTR: Low Tensor Rank Weight Adaptation | Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, Ivan Oseledets |[arxiv 2402](https://arxiv.org/pdf/2402.01376.pdf)| [Code](https://github.com/daskol/lotr)
- Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs| Afia Anjum, Maksim E. Eren, Ismael Boureima, Boian Alexandrov, Manish Bhattarai | [arxiv 2408](https://arxiv.org/pdf/2408.01008)
- DoRA: Weight-Decomposed Low-Rank Adaptation | Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen |[arxiv 2402](https://arxiv.org/pdf/2402.09353.pdf) | [Code](https://github.com/NVlabs/DoRA) | ICML 2024

**(ii) Parameter Selection**

- SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters | Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao | [arxiv 2210](https://arxiv.org/abs/2210.04284) | [Code](https://github.com/Shwai-He/SparseAdapter) | Findings of EMNLP 2022
- Sparse Low-rank Adaptation of Pre-trained Language Models | Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun | [arxiv 2311](https://arxiv.org/pdf/2311.11696.pdf)| [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023
- Asymmetry in Low-Rank Adapters of Foundation Models | Jiacheng Zhu, Kristjan Greenewald, Kimia Nadjahi, Haitz Sáez de Ocáriz Borde, Rickard Brüel Gabrielsson, Leshem Choshen, Marzyeh Ghassemi, Mikhail Yurochkin, Justin Solomon | [arxiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA?utm_source=catalyzex.com) [Code](https://github.com/NVIDIA/NeMo/tree/adithyare/vera) | 
- LoRA-FA: Memory-efficient low-rank adaptation for large language models fine-tuning | Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, Bo Li | [arxiv 2308](https://arxiv.org/pdf/2308.03303.pdf)
- LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation | Hongyun Zhou, Xiangyu Lu, Wang Xu, Conghui Zhu, Tiejun Zhao, Muyun Yang | [arxiv 2402](https://arxiv.org/pdf/2402.07721.pdf)

**(iii) Parameter Sharing**

- VeRA: Vector-based Random Matrix Adaptation | Dawid J. Kopiczko, Tijmen Blankevoort, Yuki M. Asano | [ICLR 2024](https://openreview.net/forum?id=NjNfLdxr3A)
- Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying | Adithya Renduchintala, Tugrul Konuk, Oleksii Kuchaiev | [arxiv 2311](https://arxiv.org/pdf/2311.09578)
- NOLA: Networks as linear combination of low rank random basis | Soroush Abbasi Koohpayegani, KL Navaneet, Parsa Nooralinejad, Soheil Kolouri, Hamed Pirsiavash | [arxiv 2310](https://arxiv.org/pdf/2310.02556.pdf)| [Code](https://github.com/UCDvision/NOLA) | [Code](https://github.com/UCDvision/NOLA) | ICLR 2024
- Delta-LoRA: Fine-tuning high-rank parameters with the delta of low-rank matrices | Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, Lei Zhang | [arxiv 2309](https://arxiv.org/pdf/2309.02411.pdf)

**(iv) Parameter Quantization**

- QLoRA: Efficient finetuning of quantized llms | Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer | [arxiv 2305](https://arxiv.org/pdf/2305.14314.pdf)| [Code](https://github.com/artidoro/qLoRA) | NeurIPS 2023
- Qa-LoRA: Quantization-aware low-rank adaptation of large language models | Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian | Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian | [NeurIPS 2023 (oral)](https://arxiv.org/pdf/2309.14717.pdf)| [Code](https://github.com/yuhuixu1993/qa-LoRA)
- QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning | [arxiv 2402](https://arxiv.org/pdf/2402.10462.pdf)
- Loftq: LoRA-fine-tuning-aware quantization for large language models | Hossein Rajabzadeh, Mojtaba Valipour, Tianshu Zhu, Marzieh Tahaei, Hyock Ju Kwon, Ali Ghodsi, Boxing Chen, Mehdi Rezagholizadeh | [arxiv 2310](https://arxiv.org/pdf/2310.08659.pdf)| [Code](https://github.com/yxli2123/LoftQ)
- Lq-LoRA: Low-rank plus quantized matrix decomposition for efficient language model finetuning | Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim | [arxiv 2311](https://arxiv.org/pdf/2311.12023.pdf)| [Code](https://github.com/HanGuo97/lq-LoRA)
- LQER: Low-Rank Quantization Error Reconstruction for LLMs | Cheng Zhang, Jianyi Cheng, George A. Constantinides, Yiren Zhao | [arxiv 2402](https://arxiv.org/pdf/2402.02446.pdf) | [Code](https://github.com/OpenGVLab/OmniQuant?utm_source=catalyzex.com) | ICLR 2024

### b. Ranking Adaptation

**(i) Ranking Refinement**

- Adaptive Budget Allocation for Parameter Efficient Fine-Tuning | Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao | [ICLR 2023](https://openreview.net/pdf?id=lq62uWRJjiY)
- BiLoRA: A Bi-level Optimization Framework for Low-rank Adapters | Rushi Qiang, Ruiyi Zhang, Pengtao Xie | [arxiv](https://arxiv.org/pdf/2403.13037v1)
- DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation | Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, Ali Ghodsi | [EACL](https://arxiv.org/abs/2210.07558) | [Code](https://github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA?utm_source=catalyzex.com)
- PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation | Nadav Benedek, Lior Wolf | [arxiv 2401](https://arxiv.org/pdf/2401.11316.pdf)
- Sparse Low-rank Adaptation of Pre-trained Language Models | Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun | [arxiv 2311](https://arxiv.org/pdf/2311.11696.pdf) | [Code](https://github.com/TsinghuaC3I/SoRA) | EMNLP 2023
  
**(ii) Ranking Augmentation**

- FLoRA: Low-Rank Adapters Are Secretly Gradient Compressors | Yongchang Hao, Yanshuai Cao, Lili Mou | [arxiv 2402](https://arxiv.org/pdf/2402.03293.pdf)| [Code](https://github.com/MANGA-UOFA/FLoRA) | ICML 2024
- Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | Wenhan Xia, Chengwei Qin, Elad Hazan | [arxiv 2401](https://arxiv.org/pdf/2401.04151.pdf) | ICML 2024
- ReLoRA: High-Rank Training Through Low-Rank Updates | Vladislav Lialin, Namrata Shivagunde, Sherin Muckatira, Anna Rumshisky | [arxiv 2307](https://arxiv.org/pdf/2307.05695.pdf)| [Code](https://github.com/guitaricet/reLoRA)
- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA | Sheng Wang, Boyang Xue, Jiacheng Ye, Jiyue Jiang, Liheng Chen, Lingpeng Kong, Chuan Wu | [arxiv 2402](https://arxiv.org/abs/2402.16902) | [Code](https://github.com/sahil280114/codealpaca?utm_source=catalyzex.com) 
- Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | Pengjie Ren, Chengshun Shi, Shiguang Wu, Mengqi Zhang, Zhaochun Ren, Maarten de Rijke, Zhumin Chen, Jiahuan Pei | [arxiv 2402](https://arxiv.org/abs/2402.17263) | ACL 2024
- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian | [arxiv 2403](https://arxiv.org/abs/2403.03507) | [Code](https://github.com/jiaweizzhao/GaLore) | Oral ICML 2024
  
### c. Learning Process

#### **(i) Learning Rate**

- LoRA+: Efficient Low-Rank Adaptation of Large Models | Soufiane Hayou, Nikhil Ghosh, Bin Yu | [arxiv 2402](https://arxiv.org/pdf/2402.12354.pdf)| [Code](https://github.com/nikhil-ghosh-berkeley/LoRAplus) | ICML 2024

#### **(ii) Dropout**

- LoRA Meets Dropout under a Unified Framework | Sheng Wang, Liheng Chen, Jiyue Jiang, Boyang Xue, Lingpeng Kong, Chuan Wu |[arxiv 2403](https://arxiv.org/pdf/2403.00812)

#### **(iii) Scaling Factor**

- A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | Damjan Kalajdzievski | [arxiv 2312](https://arxiv.org/pdf/2312.03732.pdf) | [Code](https://github.com/kingoflolz/mesh-transformer-jax)

#### **(iv) Learning Methods**

- AMAL: Meta Knowledge-Driven Few-Shot Adapter Learning | S. K. Hong, Tae Young Jang | [ACL 2022](https://aclanthology.org/2022.emnlp-main.709.pdf)

#### **(v) Post-hoc Processing**

- Bayesian Low-rank Adaptation for Large Language Models | Adam X. Yang, Maxime Robeyns, Xi Wang, Laurence Aitchison | [arxiv 2308](https://arxiv.org/abs/2308.13111) | [Code](https://github.com/adamxyang/laplace-lora) | ICLR 2024

### d. Theoretical Foundations

- The Expressive Power of Low-Rank Adaptation | Yuchen Zeng, Kangwook Lee | [arxiv 2310](https://arxiv.org/pdf/2310.17513.pdf) | [Code](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) | ICLR 2024
- LoRA Training in the NTK Regime has No Spurious Local Minima | Uijeong Jang, Jason D. Lee, Ernest K. Ryu | [arxiv 2402](https://arxiv.org/pdf/2402.11867.pdf)| [Code](https://github.com/UijeongJang/LoRA-NTK) | ICML 2024
- ROSA: Random Orthogonal Subspace Adaptation | Marawan Gamal, Guillaume Rabusseau | [ICML 2023](https://openreview.net/pdf?id=4P9vOFpb63) | [Code](https://github.com/marawangamal/rosa)
- Asymmetry in Low-Rank Adapters of Foundation Models | Jiacheng Zhu, Kristjan Greenewald, Kimia Nadjahi, Haitz Sáez de Ocáriz Borde, Rickard Brüel Gabrielsson, Leshem Choshen, Marzyeh Ghassemi, Mikhail Yurochkin, Justin Solomon | [arxiv 2402](https://arxiv.org/abs/2402.16842) | [Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA?utm_source=catalyzex.com)

## 2. Frontiers of LoRA

### a. Advanced Structures

**LoRA Composition**
- Adaptersoup: Weight averaging to improve generalization of pretrained language models | Alexandra Chronopoulou, Matthew E. Peters, Alexander Fraser, Jesse Dodge | [arxiv2302](https://arxiv.org/pdf/2302.07027) | [Code](https://github.com/UKPLab/sentence-transformers)
- LoRAhub: Efficient cross-task generalization via dynamic LoRA composition | Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin | [arxiv 2307](https://arxiv.org/pdf/2307.13269.pdf) | [Code](https://github.com/sail-sg/LoRAhub) | COLM 2024
- LoRARetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild | Ziyu Zhao, Leilei Gan, Guoyin Wang, Wangchunshu Zhou, Hongxia Yang, Kun Kuang, Fei Wu | [arxiv 2402](https://arxiv.org/pdf/2402.09997.pdf) | [Code](https://github.com/tatsu-lab/stanford_alpaca) 
- Batched Low-Rank Adaptation of Foundation Models | Yeming Wen, Swarat Chaudhuri | [arxiv 2312](https://arxiv.org/pdf/2312.05677.pdf) | [Code](https://github.com/huggingface/peft/tree/main)
- Hydra: Multi-head low-rank adaptation for parameter efficient fine-tuning | Sanghyeon Kim, Hyunmo Yang, Younghyun Kim, Youngjoon Hong, Eunbyung Park | [arxiv 2309](https://arxiv.org/pdf/2309.06922.pdf)| [Code](https://github.com/extremebird/Hydra)
- One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning | Arnav Chavan, Zhuang Liu, Deepak Gupta, Eric Xing, Zhiqiang Shen | [arxiv 2306](https://arxiv.org/pdf/2306.07967.pdf)| [Code](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA)
- LoRA ensembles for large language model fine-tuning | Xi Wang, Laurence Aitchison, Maja Rudolph | [arxiv 2310](https://arxiv.org/pdf/2310.00035.pdf) | [Code](https://github.com/huggingface/peft?utm_source)  
- MultiLoRA: Democratizing LoRA for Better Multi-Task Learning | Yiming Wang, Yu Lin, Xiaodong Zeng, Guannan Zhang | [arxiv 2311](https://arxiv.org/pdf/2311.11501.pdf)

**LoRA MoE**
- MoeLoRA: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models | Tongxu Luo, Jiahe Lei, Fangyu Lei, Weihao Liu, Shizhu He, Jun Zhao, Kang Liu | [arxiv 2402](https://arxiv.org/pdf/2402.12851.pdf)
- Higher Layers Need More LoRA Experts | Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, VS Subrahmanian | [arxiv 2402](https://arxiv.org/pdf/2402.08562.pdf)| [Code](https://github.com/GCYZSL/MoLA)
- Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning.| Ted Zadouri, Ahmet Üstün, Arash Ahmadian, Beyza Ermiş, Acyr Locatelli, Sara Hooker | [arxiv 2309](https://arxiv.org/abs/2309.05444) | [Code](https://github.com/for-ai/parameter-efficient-moe) 
- MOELoRA: An moe-based parameter efficient fine-tuning method for multi-task medical applications | Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, Yefeng Zheng | [arxiv 2310](https://arxiv.org/pdf/2310.18339.pdf)| [Code](https://github.com/liuqidong07/MOELoRA-peft) | SIGIR 24
- LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs | Shaoxiang Chen, Zequn Jie, Lin Ma | [arxiv 2401](https://arxiv.org/pdf/2401.16160.pdf)
- Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models | Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, Hao Wang | [arxiv 2403](https://arxiv.org/pdf/2403.03432)
- Mixture of Cluster-Conditional LoRA Experts for Vision-Language Instruction Tuning | Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang | [arxiv 2312](https://arxiv.org/pdf/2312.12379) | [Code](https://github.com/gyhdog99/mocle) 
- MIXLORA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts | Dengchun Li, Yingzi Ma, Naizheng Wang, Zhengmao Ye, Zhiyuan Cheng, Yinghao Tang, Yan Zhang, Lei Duan, Jie Zuo, Cal Yang, Mingjie Tang | [arxiv2404](https://arxiv.org/pdf/2404.15159) | [Code](https://github.com/TUDB-Labs/MixLoRA)
- LoRAMOE: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment | Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang | [arxiv 2312](https://arxiv.org/abs/2312.09979) | [Code](https://github.com/Ablustrund/LoRAMoE) |
- MoRAL: MoE Augmented LoRA for LLMs' Lifelong Learning | Shu Yang, Muhammad Asif Ali, Cheng-Long Wang, Lijie Hu, Di Wang | [arxiv 2402](https://arxiv.org/pdf/2402.11260)
- Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts [arxiv 2405](https://arxiv.org/abs/2405.11273) | Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, Min Zhang | [Code](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs)
- AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts | Zefang Liu, Jiahua Luo | [arxiv 2405](https://arxiv.org/abs/2405.00361) | [Code](https://github.com/zefang-liu/AdaMoLE) | COLM 2024
- Mixture of LoRA Experts | Xun Wu, Shaohan Huang, Furu Wei | [arxiv 2404](https://arxiv.org/abs/2404.13628) |  [Code](https://github.com/yushuiwx/MoLE) | ICLR 2024

### b. LoRA for Long Sequence Modeling

- LongLoRA: Efficient fine-tuning of long-context large language models| Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia | [arxiv 2309](https://arxiv.org/pdf/2309.12307.pdf)| [Code](https://github.com/dvlab-research/LongLoRA) | ICLR 2024 Oral
- LongqLoRA: Efficient and effective method to extend context length of large language models | Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia | [arxiv 2311](https://arxiv.org/pdf/2311.04879.pdf)| [Code](https://github.com/yangjianxin1/LongQLoRA)
- With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation | Y. Wang, D. Ma, D. Cai |  [arxiv 2401](https://arxiv.org/abs/2401.11504) | [Code](https://github.com/TemporaryLoRA/Temp-LoRA/tree/main) | COLM 2024
- RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization | Dongqi Pu, Vera Demberg | [arxiv 2405](https://arxiv.org/abs/2405.00657)

### c. LoRA for Continue Learning
- Orthogonal Subspace Learning for Language Model Continual Learning | Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, Xuanjing Huang | [EMNLP 2023 findings](https://arxiv.org/pdf/2310.14152) | [Code](https://github.com/cmnfriend/O-LoRA) 
- Continual Learning with Low Rank Adaptation | Martin Wistuba, Prabhu Teja Sivaprasad, Lukas Balles, Giovanni Zappella | [NeurIPS 2023 Workshop](https://arxiv.org/pdf/2311.17601) 
- Task Arithmetic with LoRA for Continual Learning | Rajas Chitale, Ankit Vaidya, Aditya Kane, Archana Ghotkar | [NeurIPS 2023 Workshop](https://arxiv.org/pdf/2311.02428)
- A Unified Continual Learning Framework with General Parameter-Efficient Tuning | Qiankun Gao, Chen Zhao, Yifan Sun, Teng Xi, Gang Zhang, Bernard Ghanem, Jian Zhang | [ICCV 2023](https://arxiv.org/pdf/2303.10070) | [Code](https://github.com/gqk/LAE)
  
### d. LoRA for Federated Learning

- SLoRA: Federated parameter efficient fine-tuning of language models | Sara Babakniya, Ahmed Roushdy Elkordy, Yahya H. Ezzeldin, Qingfeng Liu, Kee-Bong Song, Mostafa El-Khamy, Salman Avestimehr | [arxiv 2308](https://arxiv.org/pdf/2308.06522.pdf) 
- pFedLoRA: Model-heterogeneous personalized federated learning with LoRA tuning | Liping Yi, Han Yu, Gang Wang, Xiaoguang Liu, Xiaoxiao Li | [arxiv 2310](https://arxiv.org/pdf/2310.13283.pdf)
- Improving LoRA in Privacy-preserving Federated Learning | Youbang Sun, Zitao Li, Yaliang Li, Bolin Ding | [OpenReview](https://openreview.net/pdf?id=NLPzL6HWNl) | ICLR 2024
- Heterogeneous Low-Rank Approximation for Federated Fine-tuning of On-Device Foundation Models | Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, Gauri Joshi | [arxiv 2401](https://arxiv.org/pdf/2401.06432.pdf) 
- OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning | Rui Ye, Wenhao Wang, Jingyi Chai, Dihan Li, Zexi Li, Yinda Xu, Yaxin Du, Yanfeng Wang, Siheng Chen ｜ [arxiv 2402](https://arxiv.org/pdf/2402.06954.pdf)| [Code](https://github.com/rui-ye/OpenFedLLM) 
- Federatedscope-llm: A comprehensive package for fine-tuning large language models in federated learning | Weirui Kuang, Bingchen Qian, Zitao Li, Daoyuan Chen, Dawei Gao, Xuchen Pan, Yuexiang Xie, Yaliang Li, Bolin Ding, Jingren Zhou | [arxiv 2309](https://arxiv.org/abs/2309.00363) | [Code](https://github.com/alibaba/FederatedScope/tree/llm) 
- FedHLT: Efficient Federated Low-Rank Adaption with Hierarchical Language Tree for Multilingual Modeling | Zhihan Guo, Yifei Zhang, Zhuo Zhang, Zenglin Xu, Irwin King | [acm](https://dl.acm.org/doi/pdf/10.1145/3589335.3651933)
- FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning | Duy Phuong Nguyen, J. Pablo Munoz, Ali Jannesari |[arxiv 2404](https://arxiv.org/abs/2404.15182)
- FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering | Siqi Ping, Yuzhu Mao, Yang Liu, Xiao-Ping Zhang, Wenbo Ding | [arxiv 2404](https://arxiv.org/abs/2404.15384) | ICLR 2024
- DP-DyLoRA: Fine-Tuning Transformer-Based Models On-Device under Differentially Private Federated Learning using Dynamic Low-Rank Adaptation | Jie Xu, Karthikeyan Saravanan, Rogier van Dalen, Haaris Mehmood, David Tuckey, Mete Ozay | [arxiv 2405](https://arxiv.org/abs/2405.06368)
- FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning | Jiaxing QI, Zhongzhi Luan, Shaohan Huang, Carol Fung, Hailong Yang, Depei Qian | [arxiv 2406](https://arxiv.org/pdf/2406.07925)
- FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations | Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, Ang Li | [arxiv 2409](https://arxiv.org/pdf/2409.05976) [Code](https://github.com/ATP-1010/FederatedLLM) 
- Automated Federated Pipeline for Parameter-Efficient Fine-Tuning of Large Language Models | Zihan Fang, Zheng Lin, Zhe Chen, Xianhao Chen, Yue Gao, Yuguang Fang | [arxiv 2404](https://arxiv.org/pdf/2404.06448)
  
## 3. Applications

### LoRA in Natural Language Processing

- Machine Translation with Large Language Models: Prompting, Few-shot Learning, and Fine-tuning with QLoRA | Xuan Zhang, Navid Rajabi, Kevin Duh, Philipp Koehn |[ACL](https://aclanthology.org/2023.wmt-1.43.pdf)
- Task-Agnostic Low-Rank Adapters for Unseen English Dialects | Zedian Xiao, William Held, Yanchen Liu, Diyi Yang | [ACL](https://aclanthology.org/2023.emnlp-main.487.pdf)| [Code](https://github.com/zedian/hyperLoRA)
- LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training | Khoi M.Le, Trinh Pham, Tho Quan, Anh Tuan Luu | [arxiv 2401](https://arxiv.org/pdf/2401.04348.pdf)| [Code](https://github.com/VinAIResearch/LAMPAT) | AAAI 2024
- Task Arithmetic with LoRA for Continual Learning | Rajas Chitale, Ankit Vaidya, Aditya Kane, Archana Ghotkar | [arxiv 2311](https://arxiv.org/pdf/2311.02428.pdf) | Neurips 2023 Workshop

### LoRA in Computer Vision

**a. Visual Understanding**

(1) Domain Adaptation and Transfer Learning
- Motion style transfer: Modular low-rank adaptation for deep motion forecasting | Parth Kothari, Danya Li, Yuejiang Liu, Alexandre Alahi | [arxiv 2211](https://arxiv.org/pdf/2211.03165.pdf)| [Code](https://github.com/vita-epfl/motion-style-transfer)
- Efficient low-rank backpropagation for vision transformer adaptation | Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu | [arxiv 2309](https://arxiv.org/pdf/2309.15275.pdf) | NeurIPS 20223
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | Sidra Aleem, Julia Dietlmeier, Eric Arazo, Suzanne Little | [arxiv 2402](https://arxiv.org/pdf/2402.04964.pdf) | [Code](https://github.com/aleemsidra/ConvLoRA)
- ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | Samar Khanna, Medhanie Irgau, David B. Lobell, Stefano Ermon | [arxiv 2406](https://arxiv.org/abs/2406.10973)
- Melo: Low-rank adaptation is better than fine-tuning for medical image diagnosis | Yitao Zhu, Zhenrong Shen, Zihao Zhao, Sheng Wang, Xin Wang, Xiangyu Zhao, Dinggang Shen, Qian Wang | [arxiv 2311](https://arxiv.org/pdf/2311.08236.pdf)| [Code](https://github.com/JamesQFreeman/LoRA-ViT)
- Enhancing General Face Forgery Detection via Vision Transformer with Low-Rank Adaptation | Chenqi Kong, Haoliang Li, Shiqi Wang | [arxiv 2303](https://arxiv.org/pdf/2303.00917.pdf)

(2) Semantic Segmentation
- Customized Segment Anything Model for Medical Image Segmentation | Kaidong Zhang, Dong Liu | [arxiv 2304](https://arxiv.org/abs/2304.13785) | [Code](https://github.com/hitachinsk/SAMed) 
- SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation | An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, Hongliang Ren | [MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_23)
- Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model | Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan | [arxiv 2401](https://arxiv.org/abs/2401.17868) | [Code](https://github.com/autogluon/autogluon/tree/master/examples/automm/Conv-LoRA?utm_source=catalyzex.com)


(3) Others
- FullLoRA-AT: Efficiently Boosting the Robustness of Pretrained Vision Transformers | Zheng Yuan, Jie Zhang, Shiguang Shan | [arxiv 2401](https://arxiv.org/pdf/2401.01752.pdf)
- Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach | Wei Dong, Xing Zhang, Bihui Chen, Dawei Yan, Zhijun Lin, Qingsen Yan, Peng Wang, Yang Yang | [arxiv 2403](https://arxiv.org/abs/2403.19067) | [Code](https://github.com/zstarN70/RLRR?utm_source=catalyzex.com)
- LORTSAR: Low-Rank Transformer for Skeleton-based Action Recognition | Soroush Oraki, Harry Zhuang, Jie Liang | [arxiv 2407](https://arxiv.org/abs/2407.14655)
- Parameter-efficient Model Adaptation for Vision Transformers | Xuehai He, Chunyuan Li, Pengchuan Zhang, Jianwei Yang, Xin Eric Wang | [arxiv 2203](https://arxiv.org/pdf/2203.16329.pdf)| [Code](https://github.com/eric-ai-lab/PEViT) | AAAI 2023

  
**b. Visual Generation**
- Cones: Concept Neurons in Diffusion Models for Customized Generation | Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao | [arxiv 2303](https://arxiv.org/abs/2303.05125) | [Code](https://github.com/Johanan528/Cones)
- Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models | Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou | [arxiv 2305](https://arxiv.org/abs/2305.18292) | [Code](https://github.com/TencentARC/Mix-of-Show)
- Generating coherent comic with rich story using ChatGPT and Stable Diffusion | Ze Jin, Zorina Song | Generating coherent comic with rich story using ChatGPT and Stable Diffusion | [arxiv 2305](https://arxiv.org/abs/2305.11067)
- Cones 2: Customizable Image Synthesis with Multiple Subjects | Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao | [arxiv 2305](https://arxiv.org/abs/2305.19327) | [Code](https://github.com/ali-vilab/Cones-V2)
- StyleAdapter: A Single-Pass LoRA-Free Model for Stylized Image Generation | Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, Ping Luo | [arxiv 2309](https://arxiv.org/abs/2309.01770)
- ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs | Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, Varun Jampani | [arxiv 2311](https://arxiv.org/abs/2311.13600)
- Intrinsic LoRA: A Generalist Approach for Discovering Knowledge in Generative Models | Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, Anand Bhattad | [arxiv 2311](https://arxiv.org/abs/2311.17137) | [Code](https://github.com/duxiaodan/intrinsic-lora)
- Lcm-LoRA: A universal stable-diffusion acceleration module | Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, Hang Zhao | [arxiv 2311](https://arxiv.org/pdf/2311.05556.pdf)| [Code](https://github.com/luosiallen/latent-consistency-model)
- Continual Diffusion with STAMINA: STack-And-Mask INcremental Adapters | James Seale Smith, Yen-Chang Hsu, Zsolt Kira, Yilin Shen, Hongxia Jin | [arxiv 2311](https://arxiv.org/abs/2311.18763) 
- Orthogonal Adaptation for Modular Customization of Diffusion Models | Ryan Po, Guandao Yang, Kfir Aberman, Gordon Wetzstein | [arxiv 2312](https://arxiv.org/abs/2312.02432)
- Style Transfer to Calvin and Hobbes comics using Stable Diffusion | Sloke Shrestha, Sundar Sripada V. S., Asvin Venkataramanan | [arxiv 2312](https://arxiv.org/abs/2312.03993)
- Lora-enhanced distillation on guided diffusion models | Pareesa Ameneh Golnari | [arxiv 2312](https://arxiv.org/pdf/2312.06899)
- Multi-LoRA Composition for Image Generation | Ming Zhong, Yelong Shen, Shuohang Wang, Yadong Lu, Yizhu Jiao, Siru Ouyang, Donghan Yu, Jiawei Han, Weizhu Chen | [arxiv 2402](https://arxiv.org/abs/2402.16843) | [Code](https://github.com/maszhongming/Multi-LoRA-Composition)
- ConvLoRA and AdaBN based Domain Adaptation via Self-Training | Sidra Aleem, Julia Dietlmeier, Eric Arazo, Suzanne Little | [arxiv 2402](https://arxiv.org/pdf/2402.04964.pdf)| [Code](https://github.com/aleemsidra/ConvLoRA) | IEEE ISBI 2024
- LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models | Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, Wei Liu | [arxiv 2403](https://arxiv.org/abs/2403.11627) | [Code](https://github.com/Young98CN/LoRA_Composer?utm_source=catalyzex.com)
- Resadapter: Domain consistent resolution adapter for diffusion models | Jiaxiang Cheng, Pan Xie, Xin Xia, Jiashi Li, Jie Wu, Yuxi Ren, Huixia Li, Xuefeng Xiao, Min Zheng, Lean Fu | [arxiv 2403](https://arxiv.org/abs/2403.02084) | [Code](https://github.com/bytedance/res-adapter)
- Implicit Style-Content Separation using B-LoRA | Yarden Frenkel, Yael Vinker, Ariel Shamir, Daniel Cohen-Or | [arxiv 2403](https://arxiv.org/abs/2403.14572) | [Code](https://github.com/yardenfren1996/B-LoRA) | 
- Mixture of Low-rank Experts for Transferable AI-Generated Image Detection | Zihan Liu, Hanyi Wang, Yaoyu Kang, Shilin Wang | [arxiv 2404](https://arxiv.org/abs/2404.04883) | [Code](https://github.com/zhliuworks/CLIPMoLE) 
- MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection | Chenqi Kong, Anwei Luo, Peijun Bao, Yi Yu, Haoliang Li, Zengwei Zheng, Shiqi Wang, Alex C. Kot | [arxiv 2404](https://arxiv.org/abs/2404.08452)
- Low-Rank Few-Shot Adaptation of Vision-Language Models | Maxime Zanella, Ismail Ben Ayed | [arxiv 2405](https://arxiv.org/abs/2405.18541) | [Code](https://github.com/MaxZanella/CLIP-LoRA)
- FouRA: Fourier Low Rank Adaptation | Shubhankar Borse, Shreya Kadambi, Nilesh Prasad Pandey, Kartikeya Bhardwaj, Viswanath Ganapathy, Sweta Priyadarshi, Risheek Garrepalli, Rafael Esteves, Munawar Hayat, Fatih Porikli | [arxiv 2406](https://arxiv.org/abs/2406.08798)




### LoRA in Multimodal Learning
- Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks | Yi-Lin Sung, Jaemin Cho, Mohit Bansal | [arxiv 2112](https://arxiv.org/pdf/2112.06825.pdf)| [Code](https://github.com/ylsung/VL_adapter) | CVPR 2022
- DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback | Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, Cyrus Rashtchian | [arxiv 2311](https://arxiv.org/abs/2311.17946)
- Block-wise LoRA: Revisiting Fine-grained LoRA for Effective Personalization and Stylization in Text-to-Image Generation | Likun Li, Haoqi Zeng, Changpeng Yang, Haozhe Jia, Di Xu | [arxiv 2304](https://arxiv.org/pdf/2403.07500) | AAAI 2024
- AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning | Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai | [arxiv 2307](https://arxiv.org/pdf/2307.04725) | [Code](https://github.com/guoyww/AnimateDiff) | ICLR 2024
- Multi-Concept Customization of Text-to-Image Diffusion | Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu | [arxiv 2212](https://arxiv.org/pdf/2212.04488) | [Code](https://github.com/adobe-research/custom-diffusion)
- SELMA: Learning and Merging Skill-Specific Text-to-Image Experts with Auto-Generated Data | Jialu Li, Jaemin Cho, Yi-Lin Sung, Jaehong Yoon, Mohit Bansal | [arxiv 2403](https://arxiv.org/pdf/2403.06952) | [Code](https://github.com/jialuli-luka/SELMA)
- MACE: Mass Concept Erasure in Diffusion Models | Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, Adams Wai-Kin Kong | [arxiv 2403](https://arxiv.org/pdf/2403.06135)| [Code](https://github.com/Shilin-LU/MACE)
- AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models | Yuheng Ji, Yue Liu, Zhicheng Zhang, Zhao Zhang, Yuting Zhao, Gang Zhou, Xingwei Zhang, Xinwang Liu, Xiaolong Zheng |[arxiv 2404](https://arxiv.org/pdf/2404.13425)
- MoVA: Adapting Mixture of Vision Experts to Multimodal Context | Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, Yu Liu | [arxiv 2404](https://arxiv.org/pdf/2404.13046) | [Code](https://github.com/TempleX98/MoVA)
- Customizing 360-degree panoramas through text-to-image diffusion models. | Hai Wang, Xiaoyu Xiang, Yuchen Fan, Jing-Hao Xue | [WACV 2024](https://arxiv.org/pdf/2310.18840) | [Code](https://github.com/littlewhitesea/StitchDiffusion?utm_source=catalyzex.com)
- Space narrative: Generating images and 3d scenes of chinese garden from text using deep learning | Jiaxi Shi1, Hao Hua1 | [arxiv 2311](https://arxiv.org/pdf/2311.00339)


### LoRA in Speech Processing

- Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition | Yu Yu, Chao-Han Huck Yang, Jari Kolehmainen, Prashanth G. Shivakumar, Yile Gu, Sungho Ryu, Roger Ren, Qi Luo, Aditya Gourav, I-Fan Chen, Yi-Chieh Liu, Tuan Dinh, Ankur Gandhe, Denis Filimonov, Shalini Ghosh, Andreas Stolcke, Ariya Rastow, Ivan Bulyko | [arxiv 2309](https://arxiv.org/pdf/2309.15223.pdf)
- Low-rank Adaptation Method for Wav2vec2-based Fake Audio Detection | Chenglong Wang, Jiangyan Yi, Xiaohui Zhang, Jianhua Tao, Le Xu, Ruibo Fu | [arxiv 2306](https://arxiv.org/pdf/2306.05617.pdf) | CEUR Workshop
- Sparsely Shared LoRA on Whisper for Child Speech Recognition | Wei Liu, Ying Qin, Zhiyuan Peng, Tan Lee | [arxiv 2309](https://arxiv.org/pdf/2309.11756.pdf) | [Code](https://github.com/huggingface/peft)

### LoRA in Code Engineering

- LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning | Junyi Lu, Lei Yu, Xiaojia Li, Li Yang, Chun Zuo | [arxiv 2308](https://arxiv.org/pdf/2308.11148.pdf)
- RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair | RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair | André Silva, Sen Fang, Martin Monperrus | [arxiv 2312](https://arxiv.org/abs/2312.15698) | [Code](https://repairllama.github.io)
- MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair | Meghdad Dehghan, Jie JW Wu, Fatemeh H. Fard, Ali Ouni |[arxiv 2408](https://arxiv.org/pdf/2408.09568)


### LoRA in Scientific Discovery

- X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design | Eric L. Buehler, Markus J. Buehler | [APL Machine Learning](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581)
- ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Post Translational Modification | Amelie Schreiber | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.abstract)
- Fine-tuning protein language models boosts predictions across diverse tasks | Robert Schmirler, Michael Heinzinger & Burkhard Rost | [Nature Communication](https://www.nature.com/articles/s41467-024-51844-2)
- Parameter-efficient fine-tuning on large protein language models improves signal peptide prediction | Shuai Zeng, Duolin Wang, Dong Xu | [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.04.565642v1)
- Prollama: A protein large language model for multi-task protein language processing | Liuzhenghao Lv, Zongying Lin, Hao Li, Yuyang Liu, Jiaxi Cui, Calvin Yu-Chian Chen, Li Yuan, Yonghong Tian | [arxiv 2402](https://arxiv.org/pdf/2402.16445)

### LoRA in Time Series
- Low-rank Adaptation for Spatio-Temporal Forecasting | Weilin Ruan, Wei Chen, Xilin Dang, Jianxiang Zhou, Weichuang Li, Xu Liu, Yuxuan Liang | [arxiv](https://arxiv.org/abs/2404.07919) | [Code](https://github.com/RWLinno/ST-LoRA) 
- Channel-Aware Low-Rank Adaptation in Time Series Forecasting | Tong Nie, Yuewen Mei, Guoyang Qin, Jian Sun, Wei Ma | [arxiv](https://arxiv.org/pdf/2407.17246) | [Code](https://github.com/tongnie/C-LoRA)
- Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting | Divij Gupta, Anubhav Bhatti, Suraj Parmar, Chen Dan, Yuwei Liu, Bingjie Shen, San Lee | [arxiv](https://arxiv.org/abs/2405.10216)

### LoRA in Graph Learning
- GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning | Zhe-Rui Yang, Jindong Han, Chang-Dong Wang, Hao Liu | [arxiv 2409](https://arxiv.org/pdf/2409.16670) 
- Fast and Continual Knowledge Graph Embedding via Incremental LoRA | Jiajun Liu, Wenjun Ke, Peng Wang, Jiahao Wang, Jinhua Gao, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji, Yining Li | [arxiv 2407](https://arxiv.org/pdf/2407.05705) | [Code](https://github.com/seukgcode/FastKGE) | IJCAI2024
  
### LoRA in Recommender System

- Customizing Language Models with Instance-wise LoRA for Sequential Recommendation | Xiaoyu Kong, Jiancan Wu, An Zhang, Leheng Sheng, Hui Lin, Xiang Wang, Xiangnan He | [arxiv 2408](https://arxiv.org/pdf/2408.10159) 
- Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation | Jiachen Zhu, Jianghao Lin, Xinyi Dai, Bo Chen, Rong Shan, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang | [arxiv 2408](https://arxiv.org/pdf/2408.03533) 
- MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction | Zhiming Yang, Haining Gao, Dehong Gao, Luwei Yang, Libin Yang, Xiaoyan Cai, Wei Ning, Guannan Zhang | [arxiv 2408](https://arxiv.org/pdf/2408.08913) | [Code](https://github.com/gaohaining/MLoRA) | [arxiv 2409](https://arxiv.org/pdf/2409.08543)
- ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Model | Zezheng Qin | [mdpi](https://www.mdpi.com/2227-7390/11/16/3577) 
- LoRA-NCL: Neighborhood-Enriched Contrastive Learning with Low-Rank Dimensionality Reduction for Graph Collaborative Filtering | Tianruo Cao, Honghui Chen, Zepeng Hao | [arxiv 2403](https://arxiv.org/pdf/2403.13325) | [Code](https://github.com/zhengzhi-1997/LLM-TRSR) | WWW2024
- LoRA for Sequential Recommendation Harnessing large language models for text-rich sequential recommendation | Zhi Zheng, Wenshuo Chao, Zhaopeng Qiu, Hengshu Zhu, Hui Xiong | [arxiv 2403](https://arxiv.org/pdf/2403.13325) | [Code](https://github.com/zhengzhi-1997/LLM-TRSR) | WWW2024

### LoRA in Anomaly Detection

- Parameter-Efficient Log Anomaly Detection based on Pre-training model and LoRA | Shiming He, Ying Lei, Ying Zhang, Kun Xie, Pradip Kumar Sharma | [Zenodo](https://zenodo.org/records/8270065)

### LoRA in PDE

- PIHLoRA: Physics-informed hypernetworks for low-ranked adaptation | Ritam Majumdar, Vishal Sudam Jadhav, Anirudh Deodhar, Shirish Karande, Lovekesh Vig, Venkataramana Runkana | [NeurIPS 2023](https://openreview.net/pdf?id=kupYlLLGdf)

### LoRA in RL

- Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent | Xiaoyan Yu, Tongxu Luo, Yifan Wei, Fangyu Lei, Yiming Huang, Hao Peng, Liehuang Zhu | [arxiv 2402](https://arxiv.org/pdf/2402.13717.pdf)| [Code](https://github.com/weiyifan1023/Neeko)
- Handling coexistence of LoRA with other networks through embedded reinforcement learning | Sezana Fahmida, Venkata Prashant Modekurthy, Mahbubur Rahman, Abusayeed Saifullah |[ACM](https://dl.acm.org/doi/abs/10.1145/3576842.3582383)

### LoRA for Pretraining

- Training Neural Networks from Scratch with Parallel Low-Rank Adapters | Minyoung Huh, Brian Cheung, Jeremy Bernstein, Phillip Isola, Pulkit Agrawal | [arxiv 2402](https://arxiv.org/pdf/2402.16828.pdf)| [Code](https://github.com/minyoungg/LTE)

### LoRA Serving System

- Peft: State-of-the-art parameter-efficient fine-tuning methods | Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, Fu Lee Wang | [Huggingface](https://github.com/huggingface/peft)
- S-LoRA: Serving thousands of concurrent LoRA adapters | Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica | [arxiv 2311](https://arxiv.org/pdf/2311.03285.pdf)| [Code](https://github.com/S-LoRA/S-LoRA) | MLSys Conference 2024
- CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference | Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Qizhen Weng, Xusheng Chen, Yizhou Shan, Binhang Yuan, Wei Wang | [arxiv 2401](https://arxiv.org/pdf/2401.11240.pdf)
- Local LoRA: Memory-Efficient Fine-Tuning of Large Language Models | Oscar Key, Jean Kaddour1, Pasquale Minervini | [OpenReview](https://openreview.net/pdf?id=LHKmzWP7RN) | WANT@NeurIPS 2023
  
## 4. Resource

- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models | Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, Roy Ka-Wei Lee | [arxiv 2304](https://arxiv.org/pdf/2304.01933.pdf) | [Code](https://github.com/AGI-Edgerunners/LLM-Adapters) | 
- Run LoRA Run: Faster and Lighter LoRA Implementations | Daria Cherniuk, Aleksandr Mikhalev, Ivan Oseledets | [arxiv 2312](https://arxiv.org/pdf/2312.03415.pdf)
- Large language model LoRA specifically fine-tuned for medical domain tasks | [Code](https://huggingface.co/nmitchko/medfalcon-40b-LoRA)
