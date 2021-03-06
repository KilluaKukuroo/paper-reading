# This repo is for my brief notes when reading papers.

# 论文笔记和原文
* [smart city & mobile computing & spatial-temporal mining](#Mobile-&&-smart-city-&&-spatial-temporal-mining)
* [privacy](#privacy)
* [Graph Neural Network](#GNN)
* [联邦学习](#联邦学习)
* [deep learning and neural network](#deep-learning)
* [transfer learning](#transfer-learning)
* [Interpretable ML](#Interpretable-ML)
* [小样本学习 && 类别不均衡](#小样本学习-&&-类别不均衡)
* [NLP and web, knowledge graph](NLP-and-web,-knowledge-graph)
* [CV](#CV)
* [集成学习](#集成学习)
* [智慧电网 smart grid](#智慧电网-smart-grid)
* [优化方法](#优化方法)
* [多模态](#多模态-multi-modal)
* [data processing](#data-processing)



## Before reading a paper

[1.How to read a Technical paper -- Jason Eisner from JHU](file/How to Read a Technical Paper-Jason Eisner 2009.pdf)

2009年的一个文章，建议了读什么，怎么读，什么时候读，为什么要做笔记等，写的很好；

[2.Lessons from My First Two Years of AI Research](file/Lessons from My First Two Years of AI Research.pdf)

by Tom Silver, Ph.D. student at MIT. 介绍了自己刚开始读博的两年总结的经验教训：

- 包括寻找几个可以随意问傻问题的人；
- 在不同的地方思考研究问题：
  - 和不同领域的学者交流，发现他们领域的痛点；
  - 思考问题的时候，先用代码实现一个baseline，写的过程会发现很多新的问题，更深的理解；
  - 实现或者扩展别人论文中你感兴趣的部分；
- 如何读论文：**Conversations >> videos > papers > conference talks**
- Write！多写技术博客、总结、论文，并且write down your idea throughout the day;
- 保持健康：mental health and physical health are prerequists for research;


## Mobile && smart city && spatial-temporal mining
### scholars
- [Alex 'Sandy' Pentland - MIT media lab](https://scholar.google.com/citations?hl=zh-CN&user=P4nfoKYAAAAJ&view_op=list_works&sortby=pubdate)<br>
- [Yu Zheng - JD.COM](https://scholar.google.com/citations?hl=zh-CN&user=juUcdgYAAAAJ&view_op=list_works&sortby=pubdate)<br>
- [Jie Feng](https://vonfeng.github.io/publications/) ** Ph.D candidate in THU **,  [Yong Li](http://fi.ee.tsinghua.edu.cn/~liyong/) ** Associate Prof in THU, mobile computing + ML ** <br>
- [Lijun Sun](https://lijunsun.github.io/) AP at McGill Univ, machine learning + smart city <br>
- [Xue (Steve) Liu](https://www.cs.mcgill.ca/~xueliu/site/intro.html) FIEEE, McGill Univ, *IoT,CPS,ML,smart energy system* <br>
- [Xiaohui Yu](http://www.yorku.ca/xhyu/) Associate Prof, York University, *data mining, transportation,location-based services,social network*<br>
- [Mi Zhang](https://www.egr.msu.edu/~mizhang/index.html) Associate Professor, in MSU, On-device ML for mobile, IoT, FL, ML for systems, AI for health;

### review

1.[Mobile Cyber-Physical Systems for Smart Cities](https://dl.acm.org/doi/abs/10.1145/3366424.3382121)<br>
2020年WWW workshop，research overview, 作者Desheng Zhang来自Rutgers CS. <br>
**problem**：城市化面临的问题：congestion，energy consumption需要CPS（IoT）来解决；现有的CPS方法大都是针对single infrastructure and single domain data, 限制了对mobility建模的能力。所以，
针对现在城市的发展可以获得的多种数据，本文介绍作者已经做的*cross-domain interaction*的研究，和潜在的研究方向。
**特点**: 文章非常简短；主要总结了作者自己在CPS领域的工作;  <br>
**future work**：1)Measuring and predicting mobility by mobility models; 2)Intervening and altering mobility by city services; <br>

2.[Data Sets, Modeling and Decision Making in Smart Cities: A Survey](https://www.cs.virginia.edu/~stankovic/psfiles/Smart_City_Survey%20(002).pdf)<br>
2019 《ACM Transactions on Cyber-Physical Systems》,作者MEIYI MA, SARAH M. PREUM, and MOHSIN Y. AHMED, ABDELTAWAB HENDAWI and JOHN A. STANKOVIC(**University of Virginia**),WILLIAM TÄRNEBERG (**Lund University, Sweden**).<br>
**summary**: 综述了*data sets*和*modeling*，*decision-making*相关的文章，并且从这两个角度介绍了相应的研究方向和面临解决的问题；介绍了智慧城市框架下细分的研究方向，**现有的工作和待解决的问题**；<br>
**content** : <br>
1) key data issue: （**很多待解决的问题**）heterogeneity, interdisciplinary,integrity, completeness, real timeliness, interdependencies; <br> 
2) key decision-making issue: safety and serivce conflict, security, uncertainty, humans in the loop, and privacy; <br>

3.[Urban Computing: Concepts, Methodologies, and Applications](https://dl.acm.org/doi/pdf/10.1145/2629592)<br>
2014《ACM Transactions on Intelligent Systems and Technology》，citation=971,   <br>

4.[Urban Computing: Building Intelligent Cities Using Big Data and AI](http://urban-computing.com/pdf/urban%20computing-AAAI%202019.pdf)<br>
2019 AAAI keynote, PDF tutorial, Yu Zheng from **JD.com**, citation = 8; <br>

5.[Mobility Prediction: A Survey on State-of-the-Art Schemes and Future Applications](https://ieeexplore.ieee.org/document/8570749)<br>
2018, IEEE Access, Hongtao Zhang and Lingcheng Dai from **BUPT**, citation =  16; <br>
**summary**: 本文侧重mobility prediction 的三个部分：解释movement predictability， 归类预测的输出类型，评价指标performance metrics; 介绍了在5G场景下，prediction的相关问题；<br>
**method**： prediction output: *moving direction*, *transition probability*, *future location*, *user trajectory*, *next cell ID*;  <br>


6.[Thinking about smart cities](https://dusp.mit.edu/sites/dusp.mit.edu/files/attachments/publications/Smart%20Cities%20CJRES%20021415.pdf)<br>
2015,《Cambridge Journal of Regions, Economy and Society》,Amy Glasmeier and Susan Christopherson from **MIT and Cornell U**, citation  = 218; <br>
**summary**: 从**非技术**细节的角度，介绍了智慧城市的goals, ethics, potential and limitations; 给出了智慧城市未来的发展规模，从不同角度对智慧城市的定义； <br>

7.[Human Mobility from theory to practice: Data, Models and Applications](https://dl.acm.org/doi/10.1145/3308560.3320099)<br>
[PDF tutorial](https://dl.acm.org/doi/10.1145/3308560.3320099)<br>
2019 WWW keynote, **Filippo Simini **(University of Bristol), Gianni Barlacchi (Uni of Trento, Italy), Roberto Pellungrini (Uni Pisa), Luca Pappalardo(ISTI-CNR, Pisa);<br>
**summary**: 
**content**: <br>

- 介绍了CDR, GPS，location-based social network (LSN)等数据的特点，预处理方式，优点和缺点，公开的数据集，相关的研究文献，可以应用的场景;
- **privacy**：
	- why mobility data privacy is a concern? -- mobility data可以反映个人的很多信息，比如habit,health condition, religious preference; mobility data is abundant and readily 
	avaiable; 
**contribution**: <br>
- preprocessin, privacy risk assessment, mobility measures and simulation,generative models
- 开源了scikit mobility: predictive model, visualization methods, map matching, anonymization techniques;

8.[Trajectory Data Mining: An Overview](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/TrajectoryDataMining-tist-yuzheng_published.pdf)<br>
2015, **Yu Zheng** from Microsoft Research, citaion = 1046; <br>

**summary**: A very solid review. explores the connections, correlations, and differences among existing techniques. 同时，本文介绍了将tajectory data表示成 tensor，graph，matrix的方法，从而可以更方便的利用机器学习的技术来做轨迹数据挖掘。

**content**:

- 1.trajectory 数据的产生来源：（四个来源）：mobility of people (**cell tower ID from phone**), transportation (GPS data), animal(保护研究动物), natural phonomena; 
- 2.trajectory data preprocessing： 噪声过滤；stay point detection (restaurant, tourist point...); trajectory compression; trajectory segmentation; map matching;
- 3.
- 4.uncertainty and privacy
- 5.

**future direction**: 

- data management: effitient retrieval and mining multi-modal data; 
- cross-domain machine learning : 
- visualization techniques that can suggest insights across different sources;

4.[DeepTransport: Prediction and Simulation of Human Mobility and Transportation Mode at a City wide Level](https://www.ijcai.org/Proceedings/16/Papers/372.pdf)<br>
2016 IJCAI, Xuan Song, Hiroshi Kanasugi and Ryosuke Shibasaki (**The university of Tokyo**). citation = 109; <br>

**contribution**: 第一个将深度学习用来对city-wide 的 human mobility and transportation mode建模，做prediction and simulation; <br>
**problem**: traffic congestion 是一个越来越严重的问题，造成了经济等方面的大量损失，所以为了减小损失需要研究traffic congestion的预测，从而需要理解human mobility and transportation mode; <br>
**SoA**: The existing studies mainly rely on **simulation techniques** or **complex network theory** to model trafﬁc congestion dynamics in a small-scale network; <br>
**Opportunity**: 深度学习在语音、图像等多个领域取得了很好的成绩；可以收集大量的除了GPS之外的异构数据；<br>
**method**: 基于LSTM的深层multi-task learning network, 输出是 mobility output + transportation mode output; <br>

5.[Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16069/15978)<br>
2018 AAAI, Huaxiu Yao,Fei Wu(PSU), Jintao Ke(HUST), Xianfeng Tang(PSU),Yitian Jia, Siyu Lu, Pinghua Gong, Jieping Ye(Didi Chuxing), Zhenhui Li(PSU);
citation = 216;<br>
**summary**:                    <br>
**future work**: 1) 提高模型的可解释性; 2)增加可利用信息,比如POI information;<br>


### mobility embedding

1.[Identifying Human Mobility via Trajectory Embeddings](https://www.ijcai.org/Proceedings/2017/0234.pdf)<br>
2017,IJCAI. Qiang Gao, Fan Zhou, Kunpeng Zhang, Goce Trajcevski, Xucheng Luo, Fengli Zhang from **UESTC, U Maryland, Northwesten U**, citation = 42; <br>

**summary**: 传统工作是将轨迹数据分类为不同行为，而本文提出解决trajectory user linking (TUL),将轨迹与产生轨迹的人结合起来，面对的挑战是类别数太多(人多) + 数据稀疏；为了解决这些问题，提出RNN-based semi-supervised 
model; <br>
**method**: 1)trajectory segmentation 2)check-in embedding 3)RNN-based semi-supervised model; <br>
**contribution**： 1)第一个解决TUL问题的工作； 2)提出了RNN-based 半监督模型(**获取embedding的过程是半监督**)，在两个公开数据集取得了SoA成绩； <br>

2.[MPE: A Mobility Pattern Embedding Model for Predicting Next Locations](https://arxiv.org/pdf/2003.07782.pdf)<br>
2020 WWW, Meng Chen · Xiaohui Yu (**York University**) · Yang Liu. citation = 9;   <br>
**keywork**: human mobility modeling, embedding learning, next location prediction, **traffic trajectory data**;   <br>
**summary**: 使用embedding（没有使用神经网络）对交通轨迹数据进行分析，可以运用到next location prediction, visualization; <br>
**contribution**: <br>

- 第一个使用embedding method to model mobility pattern from **traffic trajectory data**;    
- ocnsidering sequential, temporal and personal information when embedding data into vector; 
- VPR data and taxi trajectory data 数据集上取得了很好的成绩； 

3.[Personalized Ranking Metric Embedding for Next New POI Recommendation](https://www.ijcai.org/Proceedings/15/Papers/293.pdf)<br>
2015 IJCAI, Shanshan Feng,1 Xutao Li,2 Yifeng Zeng,3 Gao Cong,2 Yeow Meng Chee,4 Quan Yuan from **NTU**;  <br>

[4.A General Multi-Context Embedding Model for Mining Human Trajectory Data](https://ieeexplore.ieee.org/document/7447767)<br>
2016, TKDE.     <br>



[5.General-Purpose User Embeddings based on Mobile App Usage](https://arxiv.org/pdf/2005.13303.pdf)

[short video intro](https://www.kdd.org/kdd2020/accepted-papers/view/general-purpose-user-embeddings-based-on-mobile-app-usage)

2020 KDD, Junqi Zhang: Tencent; Bing Bai: Tencent; Ye Lin: Tencent; Jian Liang: Tencent; Kun Bai: Tencent; Fei Wang: Cornell University;

**summary**: This paper is proposed by Tencent Group to address the general purpose user embedding via mobile APP usage. 

**problem**: Model user long term and short-term interest is important for many downstream task, e.g., reconmendation, advertising.

**SoA and limitation**: 

- traditional method relies on human-crafted features, therefore, ---> it takes huge amount of human effort for different tasks; 

**chellenge**:

- Retention, installation and uninstallation for different APPs should be modeled at the same time;
- Actions of (un)installing apps are low-frequency and unevenly distributed over time
- Long-tailed apps suffer from sparsity. 很多小众app安装人数非常少;

**Method**: propose a  tailored AutoEncoder-coupled Transformer Network (AETN)  to analyze usr behavior based on mobile app usage.

[6.Incremental Mobile User Profiling: Reinforcement Learning with Spatial Knowledge Graph for Modeling Event Streams](https://dl.acm.org/doi/pdf/10.1145/3394486.3403128)

2020 KDD, Pengyang Wang, Kunpeng Liu (U of central Florida), Lu Jiang (Northeast Normal U), Xiaolin Li (Nanjin U), Yanjie Fu (U of Central Florida). 

**summary**: 本文是结合了强化学习和时空图网络来做移动用户数据（轨迹数据、check-in数据）的embedding，从而得到未来location 的预测。we integrated spatial KG to reinforcement learning to incrementally learn user representations and generate the next-visit prediction。**state**： 用户和时空网络的结合；**policy**：模仿用户来generate next-visit location. 

**Mobile user profiling**, 移动用户分析， is to extract a user’s interest and behavioral patterns from mobile behavioral data； 常见应用场景：customer segmentation, fraud detection, recommendation, user identification; 

[7.Characterizing and Learning Representation on Customer Contact Journeys in Cellular Services](https://dl.acm.org/doi/pdf/10.1145/3394486.3403377)

2020 KDD, Shuai Zhao from (New Jersey Institute of Technology), Wen-Ling Hsu, George Ma, Tan Xu, Guy Jacobson, Raif Rustamov from **AT&T**; 

**summary**: 

**Problem**:   **什么是customer contact journey**:Customer journey refers to the complete path of experiences that customers go through when interacting with a company—a record of how customers interact with multiple touch points via different channels ;  **目的**： provide better customer service and improve customer satisfaction；减少公司的开支；提供定制化的服务；

**contribution**：

- (1) We **define a new problem** of learning customer contact journey representations
- (2) We cast this problem into an attributed sequence embedding problem, and propose an effective sequence-to-sequence model solution accordingly。
- 给模型加入了Wasserstein divergence regularization to learn a disentangled representation of the data
- 

**data**: 一个. Communication service providers (CSPs)公司连续6个月几百万用户的三个类型的chennel数据：call (c), chat (t), and store visits (s)；每个信道数据：s **anonymized customer id, contact channel, contact reason as well as date and time**； 

9.[Route prediction for instant delivery](https://dl.acm.org/doi/pdf/10.1145/3351282) <br>
2019 Ubicomp, Yan Zhang, Yunhuai Liu, Genjian Li, Yi Ding, Ning Chen, Hao Zhang, Tian He, and Desheng Zhang.    <br>

**summary**: 通过对骑手路径选择的预测，制定不同的订单分发系统，从而减少平均的外卖派送时间，减少延误率； <br>

10.[Deep Multimodal Embedding: Manipulating Novel Objects with Point-clouds, Language and Trajectories](https://cs.stanford.edu/people/asaxena/papers/robobarista_deepmultimodalembedding.pdf)

2017 **ICRA, finalist for ICRA Best cognitive robotics paper award**, [Jaeyong Sung](https://arxiv.org/search/cs?searchtype=author&query=Sung%2C+J), [Ian Lenz](https://arxiv.org/search/cs?searchtype=author&query=Lenz%2C+I), [Ashutosh Saxena](https://arxiv.org/search/cs?searchtype=author&query=Saxena%2C+A) from Cornell University. citation = 18;

**summary**: 机器人与现实世界交互需要处理视觉，语言，还有轨迹数据。本文提出一个算法，将这三种模态的数据embed 到同一个shared embedding space. 在机器人的任务上取得了很好的结果，acc 和inference time。


### ETA estimation of time of arrival
10.[Learning to Estimate the Travel Time](https://dl.acm.org/doi/pdf/10.1145/3219819.3219900)<br>
2018 KDD, Zheng Wang, Kun Fu, Jieping Ye from **DiDi Chuxing**; <br>
**summary**: 将车辆从A到B的estimated time of arrival(ETA) 转化为机器学习的回归问题，利用大量的历史数据，设计 Wide-Deep-Recurrent (WDR) learning model预测旅行时间，
并且在DIDI上验证了这个方法； <br>
**method**: <br>

- feature extraction: spatial info, temporal info, traffic info, personalized info (driver profile, etc.), augmented info (whether condition, traffic restriction, etc.); <br>
- transform ETA as a machine learning problem, design loss function; <br>
- wide-deep-recurrent learning model design; <br>

11.[Travel Time Estimation without Road Networks: An Urban Morphological Layout Representation Approach](https://lanwuwei.github.io/IJCAI19_Travel_Time_Estimation.pdf)<br>
2019 IJCAI, Wuwei Lan (Ohio State U), Yanyan Xu (UC Berkerly), Binzhao(Wisense AI, Jinan, China). citation = 3. <br>
**keywork**: ETA, deep learning, multi task, image; <br>
**summary**：以前的ETA估计方法分为两类，一：将路径划分成为多个子路径，计算子路径时间进行叠加；二：利用深度学习进行端到端学习；本文假设：local traffic condition is closely related with the landuse and 
built environment, i.e., metro station, intersections, etc. 并且交通情况和这些局部环境的关系是时变的，太过复杂；所以，本文提出一个多任务深度学习框架，从**built environment images**学习travel time,不适用road network信息；
并且在两个城市的数据集取得SoA；*与下面2018 AAAI 的论文框架类似，不过本文是直接利用图片数据*<br>

**method**: 1) image representation: learning feature patterns from morphological layout images using CNN; 2)multitask prediction: **哪些task？**   <br>

12.[WhenWill You Arrive? Estimating Travel Time Based on Deep Neural Networks](http://urban-computing.com/pdf/travel%20time%20estimation-AAAI%202018-Zheng.pdf)<br>
2018 AAAI, Dong Wang (Duke U), Junbo Zhang (Microsoft Research), Wei Cao and Jian Li (THU), **Yu Zheng** (Microsoft Research, XiDian, Chinese Academy of Science) ; <br>
**keywords**: ETA    <br>
**summary**: 以前的ETA估计方法是：将路径划分成为多个子路径，计算子路径时间进行叠加。但是这样不准确，因为没有考虑road intersection/traffic light等情况。所以本文提出
一种end-to-end模型，直接预测到达时间; <br>
**datasets**: chengdu dataset (1.4 billion GPS records) + beijing dataset (0.45 billion GPS records); 没说明是公开的还是自己的数据,应该是自己采集的数据; <br>
**methods**:  1) transform GPS sequence to feature maps, to capture local spatial correlations;   <br>


13.[TEMP-A Simple Baseline for Travel Time Estimation using Large-Scale Trip Data](https://arxiv.org/pdf/1512.08580.pdf)<br>
[short version](https://dl.acm.org/doi/pdf/10.1145/2996913.2996943)<br>
**keywork**: ETA;    <br>
2016 SIGSPATIAL,    citation=51; <br>

14.[Doing in One Go: Delivery Time Inference Based on Couriers’ Trajectories]<br>
2020 KDD, Sijie Ruan(XD), Zi Xiong(Wuhan U), Cheng Long(NTU), Yiheng Chen(JD), Jie Bao(JD), Tianfu He(HIT), **Ruiyuan Li(XD)**, 
Shengnan Wu(JD), Zhongyuan Jiang(XD), **Yu Zheng(XD)**;<br>





### location and trajectory prediction

#### useful materials

1.[Human Mobility from theory to practice:Data, Models and Applications](https://www.researchgate.net/publication/333075305_Human_Mobility_from_theory_to_practiceData_Models_and_Applications?enrichId=rgreq-76f6194e2f133d556b554f57dad3accb-XXX&enrichSource=Y292ZXJQYWdlOzMzMzA3NTMwNTtBUzo3NTk2ODA0NTIyODQ0MTZAMTU1ODEzMzM0MjEwMQ%3D%3D&el=1_x_3&_esc=publicationCoverPdf)

2019 tutorial, Luca Pappalardo and Gianni Barlacchi (Università degli Studi di Trento), ;

**summary**: 对mobility的dataset, models, applications 做了很详细的介绍，写的很好。比如：CDR 数据的好处坏处，预处理方法等等；

**content**:

- 介绍了privacy risk 评估的内容；
  - Why privacy for mobility data is a concern?   
    - **Mobility is a sensitive type of information**：Depending on the location visited, one could infer religious preferences, daily habits, health problems
    - Mobility data is abundant and readily available ： Location based services, social media access etc
  - Privacy actors
    • Data respondent: individuals
    • Data holder: businesses, enterprises etc.
    • Adversary or attacker: malicious third party
  - 

[NLPMM: a Next Location Predictor with Markov Modeling](https://arxiv.org/pdf/2003.07037.pdf)<br>
2020,    citation=64; <br>
**keywords**: moving pattern, next location prediction, time factor;  <br>

[Where will you go? Mobile Data Mining for Next Place Prediction](https://www.idiap.ch/project/mdc/publications/files/nov13_08_nextPlace-dawak2013.pdf)<br>
2013, Jo˜ao B´artolo Gomes, Clifton Phua, Shonali Krishnaswamy from **A Star**, citation=47; <br>
**summary**: 基于手机GPS和其他信息(e.g., accelerometer,bluetooth and call/sms logs),在单个用户的数据上预测该用户下一个位置(不关心semantic location,i.e., tagged as home,etc.)；<br>
**data**: Nokia Mobile Data Challenge (MDC):70 users for one year smartphone data; <br>
**contribution**: <br>

- privacy preserving: 对每个用户只使用他们自己的数据在本地（手机）训练模型，不需要将数据分享出去；<br>
- rich *context information* are exploited for personalization <br>
- 做了特征选择(based on information gain, and cross-validated best feature subset)，发现几乎所有特征都对预测有用，使用92%的特征取得了最好结果并且与使用所有特征类似；<br>
- 历史数据可能无法获取，或者稀疏，所以只使用当前数据进行训练预测模型； 

**Question**:  历史数据可能无法获取，或者稀疏，所以只使用当前数据进行训练预测模型???只使用当前数据是不是太少？使用当前多少数据？

[Mining moving patterns for predicting next location](http://www.yorku.ca/xhyu/papers/infosys2015.pdf)<br>
2015, Information Systems,  citation=44; <br>

[A Survey of Location Prediction on Twitter](https://arxiv.org/pdf/1705.03172.pdf)<br>
2018 TKDE,  Xin Zheng(NTU), Jialong Han(Tencent), and Aixin Sun(NTU), citation=93; <br>
**keywords**: tweet, location prediction, home location, mentioned location, tweet location; <br>
**summary**:本文的核心是基于tweet信息，预测三种位置信息(i.e., home tweet mentioned location)，不是预测mobility里面的下一个位置；并且简单综述了两个研究方向：
semantic location, point-of-interest recommendation; <br>



[Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11900/11583)<br>
2016, AAAI, citation = 315; <br>

[Introduction to Trajectory Data Mining](https://mycourses.aalto.fi/pluginfile.php/461972/mod_folder/content/0/Lecture%204%20slides.pdf?forcedownload=1)<br>
2017 lecture notes, Kirsi Virrantaus from **Aalto University **; <br>

[Hybrid Spatio-Temporal Graph Convolutional Network: Improving Traffic Prediction with Navigation Data](https://arxiv.org/pdf/2006.12715.pdf)<br>
2020 KDD, Rui Dai, Shenkun Xu, Qian Gu, Chenguang Ji, Kaikui Liu from **Alibaba Gaode**; <br>

[Collaborative Motion Prediction via Neural Motion Message Passing](https://arxiv.org/pdf/2003.06594.pdf)<br>
2020 CVPR, Yue Hu (SJTU), Siheng Chen (**Mitsubishi Electric Research Laboratories**) and Xiao Gu (**SJTU**) ; <br> 
**summary**:轨迹预测;

[MotionNet: Joint Perception and Motion Prediction for Autonomous Driving
Based on Bird’s Eye View Maps](https://arxiv.org/pdf/2003.06754.pdf)<br>
2020 CVPR,Pengxiang Wu (**Rutgers**), Siheng Chen (**Mitsubishi Electric Research Laboratories**) and Dimitris Metaxas (**Rutgers**); <br>
**summary**: 轨迹预测；自动驾驶领域利用3D点云数据，提出一个时空网络进行perception和motion prediction； 

[Social Bridges in Urban Purchase Behavior](https://dl.acm.org/doi/pdf/10.1145/3149409)<br>
2017 TIST, Xiaowen DONG and Yoshihiko Suhara (MIT), BURÇIN BOZKAYA (Sabancı University), VIVEK K. SINGH (Rutgers University), BRUNO LEPRI (Fondazione Bruno Kessler),
**Alex Pentland** from **MIT**, citation=17; <br>
**summary**: 利用social bridge的概念，对城市居民的购买力进行建模； <br>



[Predicting Origin-Destination Flow via Multi-Perspective Graph Convolutional Network](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b818/290300b818.pdf)

ICDE 2020, Hongzhi Shi (Tsinghui U), Quanming Yao (4Paradigm Inc.), Yaguang Li (Google Inc.), Lingyu Zhang and Jieping Ye (DiDi Chuxing), Yong Li (Tsinghua) and **Yan Liu (USC).**

**summary**: 利用GNN预测origin-destination flow. 



### network (cellular network, wifi, etc.)

1.[CellPred: A Behavior-aware Scheme for Cellular Data Usage Prediction](https://dl.acm.org/doi/pdf/10.1145/3380982)

2020 Ubicomp, Zhou Qin etc. from **Rutgers**;

**summary**: 本文研究了individual-level的1）cellular data 使用情况预测；2）mobility pattern；设计了一个预测网络，输入是历史的cellular data得到的user trace和user tag数据，输出是prediction future location, future data usage。潜在的应用：location service, network optimization, cellular services;

**contribution**：

- the first work to study cellular data usage prediction from individual-level with user behavior tag data; 
- 通过实验验证：考虑tag data，会提高“mobility 和 data usage prediction的效果；
- 提出prediction framework， cellPred: encoder：两个模块，分别embed 历史的轨迹信息（从cellular data得到）和历史的user tag信息；decoder：输出mobility and data usage prediction; 

![avatar](pic/cellpred-model.png)

**data**:

- cellular data： 合肥一家运营商的数据；300多万用户；2万多基站；
- tag data: 来源于一种新的商业模式，IT公司和运营商合作，使用特定的网络应用可以获得更低的网络价格。所以可以获取反应用户行为的tag信息，比如car, stocks(finantial), ...

![avatar](pic/cellpred-data.png)

**problem**：

- Signalling Type: the related signaling protocol this record belongs to
- 有 tag  和没 tag 有对比效果差别吗？cellpred-wo是不带tag和mobility、usage feature吗？

**实验**:

- evaluation: 对每个人预测data usage，然后把同一个tower的用户使用预测加起来，得到整个tower的预测，和真实值做对比；
- mobility prediction：根据grid index from a hash table; 计算MAPE；

[2.CellRep:Usage Representativeness Modeling and Correction Based on Multiple City-Scale Cellular Networks](https://dl.acm.org/doi/abs/10.1145/3366423.3380141)

2020 WWW oral, Zhihan Fang etc. from **Rutgers, Peking U, Southeast U, iFlytek**;

**summary**: 这篇文章是第一个研究城市中所有cellular（所有的通信运营商数据） 网络的文章，研究切入点是分析单个网络的representativeness（代表性），从而说明单个cellular network存在bias。进而提出一种基于贪心算法和diversity-driven contextual information的数据选择算法，在单个cellular network中选择diversity更大的数据来代表整个network，提高了40%以上的representativeness。

**Pros**: 1)This paper is well written and has very clear logic. It is a very good measurement paper in the field of cellular network. 2)The study of multiple cellular network and the representativeness of single network, the factors which could influence network representativeness could have possitive impact for both research community and cellular cervice providers.

**contribution**:

- diversity-driven data sampling: 有点类似于机器学习中的hard sample mining, 比如在imagenet中通过寻找有代表性的数据，只需要50%的数据，训练的模型不会损失精度和召回率(*ICML 2020, SELECTION VIA PROXY: EFFICIENT DATA SELECTION FOR DEEP LEARNING*); 
- 本文是第一个研究cellular usage representative on multiple cellular networks in the same city 的工作；

**concept**:

- **cellular network usage representativeness**, which is defined as the degree that a single network can be a representative of operational patterns of all cellular users in a region.
- contextual data: POI, mobility data (e.g., subway, bus, personal car), population



### others

[1.Data-Driven Model Predictive Control of Autonomous Mobility-on-Demand Systems](https://arxiv.org/pdf/1709.07032.pdf)

2017 **Jure Leskovec** et al., citation=41;

**summary**: This paper proposes a data-driven end-to-end  Autonomous Mobility-on Demand systems (共享经济分享系统，比如共享单车和DIDI)(AMoD, i.e. fleets of self-driving vehicles) control framework, to reduce average customer waiting time via LSTM to predict customer demand.

**Problem**： 共享单车很流行，但是核心问题是“imbalance problem”;

**contribution**:

- We propose an optimal dispatching policy if the trip demand is known ahead of time. (this provides the upper bound of the system)
- We propose a method to run the system in real time by predicting short-term customer demand.
- We validate our algorithm on DIDI dataset.

[2.Friendship and Mobility: User Movement In Location-Based Social Networks](https://cs.stanford.edu/people/jure/pubs/mobile-kdd11.pdf)

2011 KDD, Eunjoon Cho, Seth A. Myers and Jure Leskovec from **Stanford**, citation = 2522;

**summary**: This paper studied the pattern of human mobility in terms of regular mobility and social relationship-related mobility.

**lesson learned**:

- human mobility = periodic movement (constrined by geographic) + random jumps related with their social networks
- social relationship accounts for 10%~30% of human moement; regular behavior explains 50~70%;
- Short-ranged travel is periodic both spatially and temporally and not effected by the social network structure, while long-distance travel is more influenced by social network ties
- 虽然cellular location data和社交网络的数据很不同，但是他们却表现出了想同的mobility pattern；
- 模型：基于day-to-day movement pattern, 和来自朋友网络的social movement effect --》 得到更好的mobility prediction 模型；（什么粒度？）

**dataset**: cell phone location data + 2 online-based social networks;

**Q**:

- Cell phone location data, social network data 怎么结合起来？ ---》 好像没有结合，只是单独的验证？


## privacy
### scholars

- [Cynthia Dwork](https://scholar.google.com/citations?user=y2H5xmkAAAAJ&hl=zh-CN), distinguished scientist in **Microsoft Research**, citation>30,000;
- [Aaron Roth](https://scholar.google.com/citations?user=kLUQrrYAAAAJ&hl=zh-CN), Associate professor in **U Pennsylvania**, citation > 7,000;
- [Jie Feng](https://vonfeng.github.io/publications/) Ph.D candidate in THU<br>
- [Yang Cao](https://www.db.soc.i.kyoto-u.ac.jp/~cao/research.html) AP at **Kyoto U**, privacy-preserving data release

### privacy-preserving data release

1.[P3GM: Private High-Dimensional Data Release via Privacy Preserving Phased Generative Model](https://arxiv.org/pdf/2006.12101.pdf)

2020, 2021 ICDE, Shun Takagi, Tsubasa Takahashi, **Yang Cao** and Masatoshi Yoshikawa from **kyoto U**;

**summary**: 基于VAE，用生成模型的方法，来做data-release;




### trajectory privacy


1.[Trajectory Recovery From Ash: User Privacy Is NOT Preserved in Aggregated Mobility Data](https://arxiv.org/pdf/1702.06270.pdf)<br>
2017 WWW.

2.[Unique in the Crowd: The privacy bounds of human mobility](https://www.nature.com/articles/srep01376)<br>
2013 **Nature**, citation = 1252; <br>

3.[De-anonymization of Mobility Trajectories: Dissecting the Gaps between Theory and Practice](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_06B-3_Wang_paper.pdf)<br>
2018 NDSS. 

4.[Trajectory Privacy in Location-based Services and Data Publication](https://www.cs.cityu.edu.hk/~chiychow/papers/Explorations_2011.pdf)<br>
2011 KDD, Chi-Yin Chow from **cityU**, Mohamed F. Mokbel from **U of Minnesota**, citation = 197; <br>

1.[Plausible Deniability for Privacy Preserving Data Synthesis](http://www.vldb.org/pvldb/vol10/p481-bindschaedler.pdf)<br>
2017 VLDB, 作者来自UIUC(Vincent Bindschaedler, Carl A.Gunter), Cornell Tech(Reza Shokri). 目前引用次数61. <br>
**problem**: 一方面很多场景都涉及到数据的分享，另一方面数据的隐私往往得不到很好的保护；   <br>
**SoA**:传统的两种保护数据隐私的方法：data de-identification需要对对手有背景知识的了解，exponential mechanism for differential privacy面对高维数据计算量太大；<br>
**Opportunity**: 本文提出一种privacy-preserving 的方法来生成synthetic data; <br>
**Challenges**:          <br>
**Contributions**:          <br>

2.[Data Synthesis based on Generative Adversarial Networks](http://www.vldb.org/pvldb/vol11/p1071-park.pdf)<br>
2018年VLDB, 作者来自UNCC, Georgy Mason Unviersity, ETRI(South Korea), 目前引用=24. 提出tableGAN生成假的table数据，在保证数据隐私性的同时满足数据分享质量的要求。<br>
**problem**:      <br>
**SoA**:            <br>
**opportunity**:       <br>
**challenges**:           <br>
**contribution**:1)设计了table-GAN,生成数据质量高并且隐私性强的数据；2）生成的数据没有一对一的关系；3)多种实验验证了生成数据的隐私性和质量；<br>
**methods**：1)*table-GAN*改编自DCGAN，比传统的GAN多了一个模块，包括generator,discriminator,classifier(提高生成数据的semantic integrity，让生成的数据更加合理)；2)在原始的loss基础上，
添加information loss(保证生成数据在统计意义上和原始数据一致), classification loss(保证生成数据的semantic integrity);3）从data utility（包括statistical analysis,model compatibility(classification and regression)）, 
privacy两个角度验证生成的数据；    <br>




5.[Enhancing Gradient-based Attacks with Symbolic Intervals](https://arxiv.org/pdf/1906.02282.pdf)<br>
2019 **ICML workshop** on *Security and privacy of Machine learning*, Shiqi Wang, Yizheng Chen , Ahmed Abdou, Suman Jana from **Columbia U and PSU**, citation = 4; <br>

6.[A Data Mining Approach to Assess Privacy Risk in Human Mobility Data]() <br>
[paper slides](https://pdfs.semanticscholar.org/5711/dc77a4fa972e3c4781397fa51e9689a03bd2.pdf) <br>
2017 ACM Transactions on Intelligent Systems and Technology, Roberto Pellungrini, Luca  Pappalardo ,Francesca  Pratesi, Anna  Monreale from ***University of Pisa, Italy***, citation=13;<br>
**summary**: 通过攻击模型构建车辆GPS数据集individual privacy risk level 标签，基于random forest对**隐私风险进行分类**；   <br>
**datasets**:    <br>

- GPS data covering two Italian cities, i.e., Florence (9715 individuals) and Pisa (2280 individuals) ; <br>
- 利用多种攻击方法，得出individual的离散隐私风险，进一步将离散风险归类到6个区间，形成六分类问题的标签；<br>
**problem**: 没有读懂文章用的是什么data mining 的方法来做分类 ---> 更正：用的random forest classifier； <br>
**contribution** : <br>
- 对比传统判断隐私风险的文章，本文客服了传统方法计算开销大的问题，提出利用data mining分类方法，基于individual数据分类该用户的隐私风险； <br>
- 探讨了用户的mobility pattern 和隐私风险之间的关系； <br>
- 基于随机森林，对mobility feature进行了重要性分析，探讨不同feature对于隐私风险的贡献程度，发现the most visited place是最高风险的特征； <br> 


7.[Private and Byzantine-Proof Cooperative Decision-Making](https://dl.acm.org/doi/pdf/10.5555/3398761.3398807)<br>
2020, AAMAS, Abhimanyu Dubey and **Alex Pentland** from **MIT**, citation=1; <br>



8.[Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning](https://www.usenix.org/system/files/sec20summer_salem_prepub.pdf)<br>
2019, citation=13; 
**summary**: online learning场景下，每次查询model得到的output都不一样；是否可以根据两个结果的差值来infer data set and reconstruction? <br>

10. [Deep Models Under the GAN: Information Leakage from
Collaborative Deep Learning](https://acmccs.github.io/papers/p603-hitajA.pdf)<br>
2017 CCS, citation=292; <br>

11.[Beyond Inferring Class Representatives: User-Level
Privacy Leakage From Federated Learning](https://arxiv.org/pdf/1812.00535.pdf)(br)
2019 INFOCOM, citaion = 57; <br>

12.[Deep Leakage from Gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)<br>
2019 NIPS, **Song Han** from MIT , citaion = 32; 

13.[AttriGuard: A Practical Defense Against Attribute Inference Attacks via
Adversarial Machine Learning](https://arxiv.org/pdf/1805.04810.pdf)<br>
2018 **Usenix Security Symposium**, citation = 36; <br>


14.[The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural
Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.pdf)<br>
2020 CVPR, <br>

15.[Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning](https://arxiv.org/pdf/1812.00535.pdf)<br>
2019 INFOCOM, citation = 64;<br>
**summary**: 利用multi-task GAN让D同时预测real or fake, class label, and identity of user; 在联邦学习只更新参数的场景下，不仅实现recondtruction攻击，
也可以实现对特定用户隐私信息的重现；<br>

**problem**: **如果都可以重构用户信息图片了，为什么还要单独推测用户的隐私信息？**<br>

**[16.Anonymization of Location Data Does Not Work: A Large-Scale Measurement Study](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.651.44&rep=rep1&type=pdf)**

2011 Mobicom,Hui Zang (from Sprint), Jean Bolot from Technocolor, citation = 346;



17.[A Predictive Model for User Motivation and Utility Implications of Privacy-Protection Mechanisms in Location Check-Ins](https://www.computer.org/csdl/journal/tm/2018/04/08013715/13rRUxASu1o)

2018, TMC, Kevin Huguenin et al. , University of Lausanne, NUS, Google; citation = 18;

**data**: 拿到用户24个月的Foursquare数据，通过在线问卷并且给与金钱奖励的方式，**让用户对自己的历史check-in数据标记自己的意图**，比如：share food, wish people to join me, ...; 

18.[]()

**summary**:

**future challenges and directions**:

- 自适应的动态保护：很少有工作基于semantics of visited locations 进行动态的保护；
- dataset：This lack of large datasets strongly limit the ability of researchers to test their solutions under real condition.

- 

18.[The Long Road to Computational Location Privacy: A Survey](https://arxiv.org/abs/1810.03568)

2018 IEEE Communications Surveys & Tutorials. citation = 47;

**summary**: This survey focuses on computational location privacy, i.e., ignore privacy issues brought by human brain, only focuses on privacy issue brought by algorithms. This survey divides the data lifecycle into two phases: data collection and data publication. Also, it partitions the scenario into 3 categories, online protection, batch protection, and offline protection (data publication).  Firstly, this survey introduces the possible threats to location privacy; Secondly, evaluation metrics for Location Privacy Protection Mechenism (LPPM) are discussed; Thirdly, different LPPM are introduced (6 categories, each consists of online and offline scenario).  

**threats and attack**:

- adversarial attack
- POI inference
- social relationship 
- **re-identification： associate an identity to each trace, re-identify physical users**
  - 包括了membership inference attack
- Future mobility prediction

**evaluation metrics**:

- privacy metric
  - formal theoratical guarantee
  - data distortion: entrophy of protected data; evaluating whether POIs can be retrieved after protection;
  - attack correctness; (各种攻击方法的成功程度；)
- utility metric
- performance metric

**LPPM**:(保护隐私的方法)

- Mixed-zones
- Generalization-based mechanisms
- Dummies-based mechanisms: generate fake users
- perturbation-based mechanisms
- protocal-based mechanism
- rule-based mechanism

**SoA and limitation**:

- Most existing surveys have not covered the evaluation of privacy protection mechenism, and when they cover privacy metric, only one of privacy, performance, utility is taken into consideration.
- Previous work often focus on either online or offline scenario. But we cover both.

[PATE-GAN: GENERATING SYNTHETIC DATA WITH DIFFERENTIAL PRIVACY GUARANTEES](https://openreview.net/pdf?id=S1zk9iRqF7)<br>
2019 ICLR, citation=22; <br>
**summary**: 首先将*private aggregation of teacher ensembles (PATE)*引入到GANs，得到可以生成很强隐私性的GAN；接下来用一种新的角度评估生成的数据：在生成数据上训练测试算法
应该和在原始数据上得到同样的效果；


[Privacy Adversarial Network: Representation Learning for Mobile
Data Privacy](https://arxiv.org/pdf/2006.06535.pdf)<br>
2019 Ubicomp, **SICONG LIU**, Junzhao Du (**XD U**), ANSHUMALI SHRIVASTAVA and **Lin Zhong** (**Rice U**), citation = 1; <br>


[Unique in the shopping mall: On the reidentifiability of credit card metadata](https://dspace.mit.edu/handle/1721.1/96321)<br>
2015, **science**, citation = 408; <br>



[Rethinking Privacy Preserving Deep Learning: How to Evaluate and Thwart Privacy Attacks](https://arxiv.org/pdf/2006.11601.pdf)<br>
2020 under review, **Qiang Yang group** from HKUST and Webank, University of Malaya; <br>
**summary**: **理论上证明，可以在不影响模型效果的同时，完全防御深度泄露攻击**；


## 联邦学习
### scholars

- [Jakub Konečný](https://scholar.google.co.uk/citations?hl=en&user=4vq7eXQAAAAJ&view_op=list_works) P.h.D from Edinburge, research scientist in Google, proposed *federated learning*<br>

**blog** <br>
[Federated learning: Is it really better for your privacy and security?](https://www.comparitech.com/blog/information-security/federated-learning/)<br>
2019, 介绍了联邦学习的概念,**联邦学习的工作流程**,在GBoard, healthcare,无人驾驶车等领域的应用,隐私和安全问题,联邦学习的局限性; <br>


1.[Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885.pdf)
2019, 《ACM Trans. Intell. Syst. Technol》, **Qiang Yang (HUST)**, Yang Liu, Tianjian Chen (Webank), Yongxin Tong(Beihang U), citation = 202; <br>

**summary**: federated learning是一种概念、一种框架，类似于privacy-preserving distributed machine learning, 但是强调的是解决两个痛点：<br>
- data isolated,数据分散在不同的企业、部门，很难将这些数据融合来学习更好的模型; <br>
- data privacy and security. 传统的分享数据方法风险太大; <br>

**contribution**:  <br>
1)根据sample and feature的分布，将联邦学习概念扩充，分为三类：horisontal,vertical and transfer federated learning; <br>
2）compare federated learning with edge computing, distributed machine learning, federated database systems; <br>
3) 介绍了联邦学习的应用场景,e.g., smart retail, multi-party database querying, smart healthcare. <br>
4) 分析了联邦学习中的几种解决隐私问题的方法, secure multi-party computation (SMC), differential privacy, Homomorphic Encryption; <br>

**limitations**: 没有介绍联邦学习在communication等方面的问题和分析; <br>

5.[PMF: A Privacy-preserving Human Mobility Prediction Framework via Federated Learning](https://dl.acm.org/doi/pdf/10.1145/3381006)<br>
2020 Ubicomp, Jie Feng(THU), Can Rong(PKU), Funing Sun and Diansheng Guo(Tencent), Yongli(THU); citation = NAN; <br>
**summary**: 面对modelling human mobility中的隐私安全问题,本文提出基于federated learning的方法,在不明显降低acc的条件下解决数据隐私问题; 解决individual human mobility prediction;<br>

**contribution**: 1) 在mobility prediction场景下,第一个提出privacy-preserving mobility prediction framework; 2)给出了mobility prediction场景中一个真实的攻击案例,并且提出了
group optimization algorithm加速和安全训练; 3)在两个真实数据集进行了实验验证; <br>
**method**
**problem**:
**SoA**:
**Opportunity**:
**Challenges**:


6.[Human activity recognition using Federated learning](https://people.kth.se/~sarunasg/Papers/Sozinov2018FederatedLearning.pdf)<br>
2018 2018 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Ubiquitous Computing & Communications, Big Data & Cloud Computing, Social Computing & Networking, 
Sustainable Computing & Communications (ISPA/IUCC/BDCloud/SocialCom/SustainCom). <br>
citation = NAN, Konstantin Sozinov, Vladimir Vlassov, Sarunas Girdzijauskas (**KTH**). <br>
**summary**: 将federated learning引入到动作识别（running, walking, biking, etc.）（但不是第一个做这个工作的文章）,与传统的centrelized learning对比；没有评估**privacy**; <br>
**contribution**: 1)构建了softmax regression model and deep NN,分别用联邦学习（分布式训练）和集中训练的方法训练； 2）评估了communication cost and performance; 3)检测了corrupted data; <br>
**method**: 异常数据检测：在local端对数据测试，将acc低的数据通过阈值去除; 通过对比加入异常数据和没有异常数据，以及利用算法检测去除异常数据三种方法的实验结果，来说明检测异常数据点很重要；<br>
**problem**:  Human activity recognition 在 healthcare等领域用途很多； <br>
**SoA**: 已经有工作将联邦学习结合进来； <br>
**Opportunity**: --
**Challenges**:

7.[Reliable Federated learning for Mobile networks](https://arxiv.org/abs/1910.06837)<br>
2020 IEEE Wireless Communications , Jiawen Kang, Zehui Xiong, Dusit Niyato, Fellow, IEEE, Yuze Zou, Yang Zhang, Mohsen Guizani, Fellow, IEEE (NTU, HUST, Qatar University), citation = 10;<br>

**summary**: 通过设计得multi-weight subjective logic model计算workers' reputation, 从而实现对federated learning中可信任worker的选择；实验使用MNIST数据集，貌似与标题中的mobile network无关；
**介绍了联邦学习的很多应用场景**<br>
**problem**: federated learning经常遇到部分worker有意或者无意的上传假数据、脏数据，影响学习的结果，所以对worker进行筛选很重要；<br>
**contribution**: 1）引入reputation概念来(不是本文最先提出)select worker；2）提出subjective logic model计算reputation； 3)引入blockchain 进行reputation管理；<br>


8.[Collaborative Learning between Cloud and End Devices: An Empirical Study on Location Prediction](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/sec19colla.pdf)<br>
2019, SEC '19: Proceedings of the 4th ACM/IEEE Symposium on Edge Computing. <br>

**summary**: *不涉及隐私的描述*; 通过手机和云端协同训练模型RNN-based model,实现更加精准的location prediction; 首先通过云端一部分数据训练一个初始模型,然后在一个新的device加入之后自动压缩并下载模型到
手机端,结合本地数据训练更新模型,接下来上传参数到云端,云端收集一定量的更新之后进行云端更新,新模型被分发到手机端.迭代进行;<br>


9.[Differentially Private Federated Learning: A Client Level Perspective](https://arxiv.org/pdf/1712.07557.pdf)<br>
2017, NIPS. Robin C. Geyer, Tassilo Klein, Moin Nabi (SAP SE, ETH Zurich). citation = 136; <br>

**summary**:首先说明federated learning is vulnerable to differential attack,即识别出分布式训练中某个客户端的信息；本文的目标就是基于联邦学习，结合差分隐私，保护隐藏client所有数据的信息；通过实验表明，当client足够多，
本文方法可以实现client-level privacy并且几乎不损失模型性能；**introduce DP into FL to protect client-side data** by hiding clients' contribution during training.<br>

**method**:主要分为两部：random sub-sampling + distorting; 即随机选取一部分client节点，利用Gaussian mechanism打乱节点的梯度更新，从而实现防止节点信息的泄露；<br>
**experience**: 利用cross validation grid search搜索最优的超参数；在MINIST做实验，每个节点只分到两个类别的数字，从而使得任何节点如果只使用本地数据就不可能高准确率的分类十个类别； <br>
**contribution**: 1)not single data level, but client-level privacy could be achieved, i.e., a client's participation can be hiden during training federated model; 
2)propose to dynamiclly adapt the differential-privacy mechanism during decentralized training,并且实验表明这样可以取得更好的效果，并给出了可能的原因;



10.[Practical Secure Aggregation for Federated Learning on User-Held Data](https://arxiv.org/pdf/1611.04482.pdf)<br>
2016 NIPS workshop, Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth (**Google, Cornell University**). citation = 59<br>

**summary**:首先定义secure aggregation: a class of secure multi-party computation algorithms. secure aggregation protocol是为了在联邦学习中保护各个节点的模型梯度隐私(因为根据梯度仍然有推测出原始数据的风险);<br>



**联邦学习的安全问题**
1.[Can You Really Backdoor Federated Learning?](https://arxiv.org/pdf/1911.07963.pdf)<br>
2019,     <br>
研究了联邦学习的对抗攻击，backdoor attack， 以及攻击的次数和任务的复杂程度会影响攻击的效果，并且differential privacy对防御攻击有好的效果，代码开源在Tensorflow Federated(TFF)；<br>
backdoor attack: keep the model a good performance on overall task but let the model fail on specific task, e.g., text classification failed on specific input "what is my favorite restaurant"; <br>



## GNN

图神经网络；graph representation learning; node classiffication; graph classification;

### tutorial

[A gentle introduction to graph neural networks]([https://aifrenz.github.io/present_file/A%20gentle%20introduction%20to%20graph%20neural%20networks.pdf](https://aifrenz.github.io/present_file/A gentle introduction to graph neural networks.pdf))



[Graph Representation Learning](https://web.stanford.edu/class/cs246/slides/12-graphs2.pdf)

[course video for this tutorial](https://www.youtube.com/watch?v=uEPPnR22fxg&list=PL-Y8zK4dwCrQyASidb2mjj_itW2-YYx6-)

2019 by **Jure Leskovec**, from Stanford University. 



[BOOK-Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)

2020, Book, by William Hamilton from McGill University. 



### papers

[1.Representation Learning on Graphs: Methods and Applications](https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf)

2017, e IEEE Computer Society Technical Committee on Data Engineering, by William L. Hamilton, Rex Ying and **Jure Leskovec**, citation = 687;



[2.Fast Graph Construction Using Auction Algorithm](https://arxiv.org/ftp/arxiv/papers/1210/1210.4917.pdf)

2012, UAI , [Jun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Yinglong Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+Y), from IBM Watson Reasearch. citation=12;



[3.Learning Effective Road Network Representation with Hierarchical Graph Neural Networks](https://www.kdd.org/kdd2020/accepted-papers/view/learning-effective-road-network-representation-with-hierarchical-graph-neur)

2020 KDD, Ning Wu: Beihang University; Xin Zhao: Renmin University of China; Jingyuan Wang: Beihang University; Dayan Pan: Beihang University;

**summary**: This paper proposes a hierarchical representation GNN model （“functional zones” → “structural regions” → “road segments”. ） to model road network to generate road network representation for downstream tasks, e.g., 

**Input and output**: 输入是将轨迹数据转为road network，通过map matching；输出是根据任务不同而不同，比如label classification: 将road segmen分类为桥梁、道路，等；

![avatar](pic/kdd2020-road-network.png)

**experiments**: destination prediction;next location prediction;label classification;route planning

**future work**:

- **静态转为动态表征**：As future work, we will consider extending our model to learn time-varying representations
- 直接利用trajectory data来学习表征： Currently, we utilize trajectory data as supervision signal for network reconstruction. We will investigate how to explicitly incorporate trajectory data in the representation model

[4.Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049)

KDD 2020, Wei Jin, Jiliang Tang, etc. from **MSU and PSU**. 

**summary**: This paper proposes a method to defend graph adversarial attacks. The idea is based on the observation that the attacked graph violates some intrinsic properties of original true graph, e.g., sparse and low rank. Based on this idea, the authors design a method which could achieve very good representation learning even with strong perturbed graph.

[5.GPT-GNN: Generative Pre-Training of Graph Neural Networks](https://arxiv.org/pdf/2006.15437.pdf)

2020 KDD, Ziniu Hu (UCLA), Yuxiao Dong, Kuansan Wang (Microsoft), Kai-Wei Chang and Yizhou Sun (UCLA); 

**summary** : This paper presents a GNN pretraining method : GTP-GNN, GPT-GNN introduces a self-supervised attributed graph generation task to pre-train a GNN . 类似于BERT把某个单词遮盖然后来预测，本文将节点和边部分遮盖，然后求最大似然，来学习强有力的GNN。

![avatar](pic/kdd2020-gpt-gnn.png)



[6.Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/pdf/2005.11650.pdf)

2020 KDD, Zonghan Wu (UTS), Shirui Pan (Monash Univerisity), Guodong Long, Jing Jiang, Xiaojun Chang(Monash), Chengqi Zhang from (UTS)

**summary**:  本文利用图网络做multi-variate时间序列的预测；Our approach automatically extracts the uni-directed relations among variables through a graph learning module, into which external knowledge like variable attributes can be easily integrated.

**challenge**:

- Unknown Graph structure, 时间序列的关系需要从从数据中发掘，而不是提前定义好；
- Graph learning and GNN learning: 图网络的结构应该要随时间变化；

**data**:

![avatar](pic/kdd2020-time-series-gnn.png)

## deep learning

1.[ResNeSt: Split-Attention Networks](https://hangzhang.org/files/resnest.pdf)<br>
发表于2020年 arxiv，作者来自 Amason, UC Davis, 包括 Hang Zhang, Mu Li. 网上传言史上最强resnet魔改版。 <br>
**Problem**：目前大部分视觉的任务, e.g., obeject detection and semantic segmentation 还是使用ResNet的变体作为backbone，因为网络结构的简单和结构化。但是ResNet是为了image classification设计，在CV的其他下游任务性能不是很好，可能因为limited receptive-field and lack of cross-channel interaction. 并且resnet的各种变体往往只能在特定的任务上取得较好性能。而且，最近的cross-channel information在下游任务被证明很有效，而image classification的模型大都缺乏cross-channel interation，所以本文提出一种带有cross-channel representation的网络模型，目标是*打造一个versatile backbone with universally improved feature representation*, 从而同时提高多个任务的性能。 <br>
**SoA**：AlexNet -> NIN(1*1 convolution) -> VGG-Net(modular network design) --> Highway network(highway connection) --> ResNet(identity skip connection); NAS;
GoogleNet(muiti-path representation) --> ResNeXt(group convolution) --> SE-Net(channel-attention) --> SK-Net(feature map attention across two network branches);<br>
**contribution**：1）研究了带有feature map split attention 的resnet网络结构；
2) 在image classification 和其他transfer learning的应用场景种提供了一个大规模的benchmark, 刷新了SoA，在不同任务上分别提高1~3个点;<br>
**future work**: 通过神经网络结构搜索寻找不同硬件上对应得低延时low latency model; 不同的超参数(radix, cardinality, width)组合调优，可能会在不同的具体任务上取得更好的结果；
增加图片的size 可以提高acc；

2.[Backpropagation and the Brain](https://www.nature.com/articles/s41583-020-0277-3.pdf)<br>
发表在2020年《nature reviews|neuroscience》, 作者包括Geoffrey Hinton. <br>
大脑皮层如何修改突触，从而实现学习是一个很神秘的问题。几十年前，backpropagation 被认为是可以用来解释大脑学习机制的一个可能方法，但是由于反向传播机制没有带来很好的网络学习效果，并且反向传播缺乏生物学上的可解释性，反向传播的意义被忽视。但是，随着近年算力的提升，NN在多个领域以反向传播为学习方法的基础上取得了很好的成绩，我们认为 backpropagation offers a conceptual framework for understanding how the cortex learns. <br>
**contribution**：
**特点**：通过比较大脑神经元和人工神经网络，介绍了很多关于 learning algorithm的基础概念，e.g., backpropagation, supervised learning, error encoding, auto encoder.


3.[宽度学习-Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture](https://ieeexplore.ieee.org/document/7987745)<br>
2017年发表在《IEEE Transactions on Neural Networks and Learning Systems》，作者来自澳门科技大学；



4.**[Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning]**(https://arxiv.org/pdf/1811.12808.pdf)<br>
2018,    <br>

**summary**: 总结了机器学习中的模型评估、模型选择、算法选择的问题、挑战和各种方法；介绍了k-fold cross-validation的几种用法，不同数据量场景下的选择；<br>

5.[DeepXplore: Automated Whitebox Testing of Deep Learning Systems](https://arxiv.org/pdf/1705.06640.pdf)<br>
2017 "Communications of the ACM", Kexin Pei, Yinzhi Cao, Junfeng Yang, Suman Jana from Columbia University, †Lehigh University.  citation=398; <br>
**summary**: 深度学习系统第一个白盒测试框架；本方法基于differential testing (borrowed from software analysis)在不需要标签的情况下，自动的检测neuron behavior；本
方法发现了SoA的深度学习框架在现实世界数据库上的很多*错误行为*(找出能够出发DNN错误的test case,e.g., a special image)；<br>
**methods**: 本文主要解决两个挑战 <br>
- How to generate test cases that trigger erronous behavior of DNN? -- neuron covarge <br>
- ‘如何在没有标签的情况下发现DNN系统的错误？‘ -- 利用multiple DNNs with similar functionality; <br>
**limitations**: <br>
- differential testing要求至少两个有相同功能的DNN系统；而且，如果两个相同功能的DNN只有很小的区别（few neurons difference），系统需要很长时间寻找differential-inputs; <br>
- differential testing只能在至少有一个DNN做出不一样的结果的时候检测出错误，如果所有DNN都犯同样的错，则检测不出来对应的test case；

6.[ResNet](https://arxiv.org/abs/1512.03385)<br>
[分析residual block博文](https://shuzhanfan.github.io/2018/11/ResNet/)<br>
2016 CVPR best paper, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun from **Microsoft Research**, citation=48900; <br>
**summary**: 本文针对深度神经网络难以训练网络退化的问题，提出了resnet来简化深层网络的训练，并在图片分类目标检测等任务取得了SoA；<br>
**problem**: <br>
1)为什么假设residual mapping(f(x)) is easier to optimize than H(x)?

- before adding shortcut connection, we learn underlying mapping H(x) --> adding shortcut connection we learn F(x), H(x)=F(x)+x;
- 当网络acc饱和或者深层导致acc下降，学习的任务变成让后面深层复制浅层网络，即恒等映射，H(x)=x; 但是经过卷积和激活函数等非线性函数学习H(x)=x不容易，转变为学习F(x)=0,H(x)=F(x)+x;
i.e., x -> weight1 -> ReLU -> weight2 -> ReLU ... -> 0，因为直接将所有参数设为0就可以实现； <br>
2)本文解决的是什么问题？ --> 本文解决网络退化问题(degradation problem) <br>
- 最近很多论文比如VGG都加深了网络层数，所以很自然有一个问题：“Is learning better networks as easy as stacking more layers?”但是当网络层数增多，会出现**梯度消失和梯度爆炸**的问题。
正则化**normalization**解决了这个问题；
- 当可以训练深层网络的时候，出现了新的问题(**退化degradation(of training acc)**)：随着层数的增加，acc稳定不增加了，然后剧烈下降。这是因为过拟合嘛?不是，过拟合会在training set上
增加acc，但是现在training acc也下降。所以本文提出resnet解决这个网络退化问题，使得能够训练深层网络，提高acc；
3)what is identity mapping 恒等映射，shortcut connection 跨层连接？
- 恒等映射：在模型很深的时候发现了网络退化acc下降的问题。现有的方法：现在假设首先学习浅层网络已经达到了饱和的acc，然后在浅层网络上添加额外的几层恒等映射网络(y=x),这样网络的acc
理论上起码不会下降。但是实验表明，添加了恒等映射的网络没有提高acc，有时候还会下降；所以本文要来学习F(x)而不是H(x)来解决这个问题；
- 跨层连接：skipping one or more layers,是从highway network借鉴的思想，但是将连接中的权重去掉了，减少了参数; (为啥highway network达不到resnet的效果？理论上参数多应该模型容量更大啊？)
- 本文通过跨层连接来实验恒等映射；
**contribution**: <br>
- 假设residual mapping更容易学习，并且用实验说明的确residual network更容易优化而且可以提高模型精度；
- 在COCO目标检测上也取得很好结果，验证了深层网络对于特征提取的重要性和resnet的通用性；


7.[Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/pdf/1911.00068.pdf)<br>
2020 ICML, Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang from **MIT and Google**.<br>
**summary**: 本文generilize confidence learning (以前可能被提出过), 提出一种框架[cleanlab](https://github.com/cgnorthcutt/cleanlab/) 来发现错误标签，表
征标签噪声并且应用于带噪学习;虽然用CV做例子，但是可以扩展到其他领域; <br>

8.[A Beginner's Guide to the Mathematics of Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.3556&rep=rep1&type=pdf)<br>
1998 , A.C.C. Coolen from **Department of Mathematics, KCL**,citation=15; <br>
**summary**: **理论**;从生物角度，神经元角度阐述了神经网络的数学机制，长文； <br>


9.[Fundamentals of Recurrent Neural Network (RNN)
and Long Short-Term Memory (LSTM) Network] (https://arxiv.org/pdf/1808.03314.pdf)<br>
2020, Alex Sherstinsky from **MIT**; <br>


10.[SKIP CONNECTIONS ELIMINATE SINGULARITIES](https://arxiv.org/pdf/1701.09175.pdf)<br>
2018, A. Emin Orhan and Xaq Pitkow from **Rice U**; citation=78;<br>
**summary**: 从singularity的角度解释了为什么Resnet跨层连接效果会这么好；<br>

11.[Highway Networks](https://arxiv.org/pdf/1505.00387.pdf)<br>
2015, Rupesh Kumar Srivastava, Klaus Greff, **Jurgen Schmidhuber** from **The Swiss AI Lab IDSIA**, citation=1215; <br>
**summary**: resnet的前身，都是通过跨层链接来实现网络更容易训练，网络容量比resnet更大，但是效果却没有resnet好；<br>

12.[HyperNetworks](https://arxiv.org/pdf/1609.09106.pdf)

2017 ICLR,David Ha∗ , Andrew Dai, Quoc V. Le from **Google Brain**, citation = 470;

**summary**: 使用一个小的网络（HyperNet）为其他的大型网络（deep CNN and long RNN）生成权重，得到很好的效果。

![avatar](pic/hypernet.png)

## transfer learning


## Interpretable ML
#### scholars
- [Synthia Rudin](https://users.cs.duke.edu/~cynthia/home.html)可解释机器学习,**Duke U**<br>

1.[Do Simpler Models Exist and How Can We Find Them?](/file/Do-simpler-model-exists.pdf)<br>
[paper link](https://dl.acm.org/doi/10.1145/3292500.3330823)<br>
2019 KDD, **Cynthia Rudin** from Duke University, citation = ; <br>
**summary**: 本文要解决两个问题：1）**是否存在简单的模型能代替复杂的黑盒子模型，并且可以达到类似的acc；**2）什么情况下这些模型存在；本文从对犯罪率预测的例子出发
，表示有些情况下简单模型反而可以取得更好的效果，接着引入Rashomon set, **Rashomon Ratio（RR）**,得出在RR大的时候，很有坑会存在简单并且效果好的模型。但是RR不好求解也
不需要求解，本文给出了一种判断RR达到大的条件：选择你熟悉的几个ML模型(Random forest,SVM,etc.)-->找几个几十个数据集跑一遍-->如果各个模型表现差不多，则说明RR大，
存在简单模型，不用使用复杂模型；如果多模型差别大，则继续去探索复杂模型，因为这时候RR小，复杂模型效果更好；<br>
**few words**:

- claim: if Rashomon Set is large, --> a large-yet-accurate model is likely to exist;就像海里水越多，越有可能存在大鱼；
- Rashomon set allows us to use a simpler model w/o lossing acc;
**method**:
- 如何近似的表示Rashomon Ratio？ -- 用7层随机森林，complex but not too complex, 不同层数RF表示不同复杂度的模型；
**Q-A**：
- 什么是RR？
-- The true Rashomon set is the set of models with low true loss<br>
- RR can benefit what? -- 如果RR大，你就知道可以使用一个简单模型代替当前的复杂模型；
- 为什么复杂模型流行？ 
-- it is profitable;
-- it is easier to construct a complex model compared to the simpler one;
- what is your intuition?
-- if the datasets' feature is meaningful. the RR is tend to be large; e.g., pixel in image is not meaninful, therefore the RR is small;
- 可否用RR提高dataaset的质量？
-- **没想过这个问题，但是很有趣。比如挑选根据RR数据集；**

2.[Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use
Interpretable Models Instead](https://arxiv.org/pdf/1811.10154.pdf)<br>
2019 Nature Machine Intelligence, **Cynthia Rudin** from Duke University; citataion = 355; <br>

## 机器学习
[on discriminative vs. generative classifiers a comparison of logistic regression and naive bayes](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)<br>
2002, **Andrew Y.Ng and Michael Jordan**, citation = 2000; <br>


### co-training
1.[Co-teaching: Robust Training of Deep Neural
Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)<br>
2018 NIPS, Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor W. Tsang and **Masashi Sugiyama **from **U of Tokyo, **, citation = 187;<br>
**summary**: 本文基于co-training提出了co-teaching训练范式，解决**强噪声数据集中学习的问题**；

**method**:
- 1.总共两个子网络，两个网络分别前向传播所有的samples，然后挑选small loss samples作为可能干净的数据；
- 2.两个子网络通信，将自己选择的干净数据传递给对方；
- 3.**根据对方传递的数据，更新自己网络的参数；**
- 4. batch size从大到小，迭代123步骤；

**Problem**：
- 每次两个网络前向传播的数据相同吗？否则如何用对方挑选的数据更新自己的网络；

2.[Combining labeled and unlabeled data with co-training](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)<br>
1998, Avrim Blum and Tom Mitchell from **CMU**, citation = 5831; <br>
1998 "Computational Learning Theory", citaion = 
**summary**: **co-train最早的论文**;在一个label对应两个描述角度（x1,x2）的场景下，本文提供一种学习方法，用labeled data分别利用x1,x2训练两个模型，对unlabeled data做预测，预测
的标签用来扩充labelled data数据集；<br>



3.[Learning from Noisy Labels with Distillation](https://arxiv.org/abs/1703.02391)

2017, Yuncheng Li, Jianchao Yang, Yale Song, Liangliang Cao, **Jiebo Luo**, Li-jia Li, citation = 210;

**summary**: 受到 Hinton distillation的启发，本文提出 use large amount of noisy data to augment small clean dataset to learn a better visual representation and classifier. The key contributions are as follows:

- under some conditions, we have theoretical analysis of the distillation process;
- 通过维基百科构建texual knowledge graph (labels are related by their definitions, **encode label space**), 并且通过构建的图谱指导蒸馏学习；(**好像就是利用了一个将图片的tag链接到维基百科的entity的一个工具**)
- 收集了包含480K图片、几百个类别的多领域数据集，as benchmakr datasets for realworld labeling noises;

**key**：

- 知识图谱的作用：将在small clean dataset 上训练得到的模型产生的输出 soft label --> 变成new soft label;
- 获取noisy label：收集 YFCC100M 数据集（photo,title,tag）(没有clean annotation，有noisy label (**soft label**)) --》 利用DBpedia Spotlight 工具，link a photo's title and tags with a wikipedia entity  --> noisy labels （**new soft labels**）;
- 获取clean label: 众包；或者拿imageNet的部分数据；

**future work**:

- 除了knowledge graph，用其他来源来指导蒸馏学习；
- 将learning from noisy labels 应用到其他场景（比如网络的图片搜索）；

### training technique

[1.Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data](https://dl.acm.org/doi/pdf/10.1145/3394486.3403228)

2020 KDD, Garrett Wilson, Janardhan Rao Doppa, Diane J. Cook from Washington state U; 

**summary**: This paper studies the semi-supervised domain adaption in time series data. **虽然图像领域domain adaption做的很多，但是时序数据很少**；

**domain adaption**:  方便复用数据，Domain adaptation (DA) offers a valuable means to reuse data and models for new problem domains；



## 小样本学习 && 类别不均衡
1.[decoupling representation AND classifier FOR LONG-TAILED RECOGNITION](https://arxiv.org/pdf/1910.09217.pdf)<br>
2020 ICLR, Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis 来自Facebook AI and NUS; <br>



2.[Generalizing from a Few Examples: A Survey on Few-Shot Learning](https://arxiv.org/pdf/1904.05046.pdf)<br>
2020 , Yaqing Wang (HKUST and Baidu Research), Quanming Yao (4Paradigm), JAMES T. KWOK and LIONEL M. NI (HKUST);   <BR>



## semi-supervised &unsupervised
1.[DEEP SEMI-SUPERVISED ANOMALY DETECTION](https://arxiv.org/abs/1906.02694)<br>
[openReview link](https://openreview.net/forum?id=HkgH0TEYwH)<br>
2020 ICLR, citation = 20; <br>
**summary**: This paper proposes a deep semi-supervised framework for general anomaly detection (AD). Existing methods focus on unsupervised manner, they 

**SoA and limitation**:(写的很精彩)<br>
- **Shallow AD**requires feature engineering and limited scalability on large dataset; --> **deep AD**;  (unsupervised)
- Most existing work focus on **unsupervised anomaly detection**; however --> there are often a small set of labelled data by expert annotation;
So, we propose to utilize labelled data with semi-supervised manner;
- most **existing semi-supervised shallow and deep models**, only focus on labeled normal data but not abnormal data; 
- some work investigated general semi-supervised model and **utilized labeled anomalies**, however --> are domain or data-type specific;
- **semi-supervised learning **mostly focus on classification between normal and anomaly, they assume similar data are the same type,--> this is invalid
for anomalies because anomalies don't have to be similar to each other;
- Therefore, we introduce the general semi-supervised, considering labeled anomalies, Deep SAD; 

**Deep SADD**: 本文的前身工作，无监督做AD，原理是让神经网络来学习一个超平面C，loss：让数据集的点到C的距离最小化；用最开始forward得到的网络输出平均值作为C
的初始值；<br>

## NLP and web, knowledge graph
1.[Correcting Knowledge Base Assertions](https://arxiv.org/pdf/2001.06917.pdf)<br>
本文发表在2020WWW的oral，作者来自Oxford, Tencent, University of Oslo. <br>
**summary**: 检测并且更正knowledge base中的错误；<br>
**Problem**: knowledge bases(KB) 在搜索引擎，问答系统，common sense reasoning, machine learning等领域起到了重要的作用，but KB is suffering from quality issues, e.g.,
constraint violations and erroneous assertions.  <br>
SoA: 现有工作在KB quality上主要包含：error detection and assessment, quality improvement via completion, canonicalization.<br>
**Opportunity**: 现有的工作大都是检测出 KB中的错误之后就将错误去除，而不是去更正这些错误。所以更正检测出的erroneous assertions is a new opportunity; 关于correction，现有工作在忽视了assertion的上下文语义信息；
关于quality improvement，现有工作没有提出一个general correction method；<br>
**Challenges**: 
**Contributions**: 1-提出了可以用于更正错误的entity assertions and literal assertions的通用框架；2-利用了semantic embedding and observed features捕捉局部的上下文信息；3-soft property constraint;
4-在meidcal KB验证entity assertions correction, 在DPpedia验证literal assertions correction;  <br>




2.[GPT3-Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)<br>
2020, Tom B. Brown et al. from **OpenAI, JHU**, citation=None; <br>
**summary**: 现在的NLP模型通过大规模预训练已经取得了很好的效果，但是在即使是很相似的任务上还是需要大量数据进行fine-tuning，但是人类可以轻易的进行知识迁移；
为了解决这个问题，OpenAI用10000张GPU训练了**1700亿参数,45TB training data**的GPT-3模型，不需要在特定任务上进行fine-tuning，并且在多个数据集多个任务
（translation,QA,etc.）进行zero shot, one shot,
few-shot learning测试，在一些任务上甚至超越了微调的SOA；同时，本文指明了在一些数据集上GPT-S效果并不好；<br>

**limitations**:<br>
- text synthesis上还有不足，比如在document level text synthesis; 
- several structural and algorithmic limitations;
- poor sample effitiency during pre-training, 虽然看了很多文本，但是没有人类的拓展性分析能力；
- 模型太大了，在推理阶段both expensive and inconvenient; 
- 和其他深度学习模型类似，模型的决策不具有**可解释性**；
**history**：<br>
- GPT-1: **5GB** training data;
- GPT-2: **15亿**参数；**40GB** training data;


3.[Embedding-based Retrieval in Facebook Search](https://arxiv.org/pdf/2006.11632.pdf)<br>
2020 KDD, Jui-Ting Huang et al. from **Facebook**, citation = 1; <br>
**summary**: 本文介绍了facebook基于embedding的检索技术，偏工业的一篇文章。Facebook以前都是基于对text的布尔匹配做检索，本文将embedding based method和
社交网络的social graph 结和，做特性化的搜索，是Facebook最新的工作。
**method**:<br>
- 提出了 unified embedding framework；相对于传统的搜索，model 会结合搜索的time,location,social connection, etc. 信息做embedding；
- feature engineering： 
-- textual feature, location features, social graph;
- **embedding ensembles**: 不同阶段用不同的目的训练模型，会使得模型有不同的特性[1]，比如first stage focuses on recall, second stage focuses on distinguishing
similarity, 所以本文探讨了不同的ensemble embedding 集成：
-- weighted concatenation; cascade model; 
- Hard mining： 因为数据过于多样，为了学习到更好的embedding，设计了hard mining（以前的都是CV领域的） ；
- 提出了很多trick，to optimize retrival system end-to-end; <br>
**citation**:<br>
[Hard-Aware Deeply Cascaded Embedding]做了cascaded embedding training (https://arxiv.org/abs/1611.05720)<br>
**future**:<br>
- go deep: 使用更强的model, such as BERT， 来解决特定的任务；
- go universal: 打造一个通用的embedding model来对付不同的任务；


4.[](https://arxiv.org/pdf/2005.04118.pdf)<br>
2020 **ACL Best paper **, Marco Tulio Ribeiro (**Microsoft Research**), **Tongshuang Wu**, Carlos Guestrin (**UW**) and Sameer Singh (**UC Irvine**),
citation=0; <br>


**Discussion**: <br>
- 模型缺少泛化能力，是一个设计上的错误，还是算法本身的缺点？


- 以后可能不需要刷SOA，只需要在测试指标中提高一个部分，就可以写论文了；

**experiment**:<br>
- case study: Microsoft Sentiment analysis, 用了5个小时，发现已经strong test的系统里面新的很多bug；
- case study: MFT, test BERT on QQP (2h); 

**QA**：<br>
**为什么做这个工作**:<br>
1.现在很多人都在做**模型的分析，evaluation**,追论文很累，影响做分析的研究；我们做分析是想理解具体的任务，希望有一个工具来简化我们分析的工作；
2.可解释性只是理解模型的一个方法，我们觉得还需要从不同的角度来理解模型；就像工程领域的软件测试;

**困难**：关于性能的定义；
如何获best paper：看看大家都在做啥，就可以知道啥比较重要；<br>
CHECKLIST可以用到CV或者其他领域的任务嘛？：你只需要一个capability列表，针对你的任务去修改capability list就行；CHECKLIST 作为一个框架是可以用到其他领域的；CV或者其他领域和NLP有不同的任务，比如pixel
属于哪个物体等，所以可能需要一些不同的capability list; <br>

首先给你一个足够小、但是有意思的方向，可操作性很强的课题，通过这个课题让你入门科研，建立自信心。然后慢慢接触更加有挑战性的问题。不要一开始就改变世界。

**ultimate goal**:<br>
- providing a shared test suite for NLP task ;

## CV
1.[A segregated cortical stream for retinal direction selectivity](https://www.nature.com/articles/s41467-020-14643-z.pdf) <br>
[Nature子刊研究颠覆常识：视网膜计算使眼睛先于大脑产生视觉信息](https://mp.weixin.qq.com/s/rzsS203NWKOhLqlenIGeDg)
2020年发表在"nature communication", 作者来自丹麦的Aarhus University. <br>
**problem**: 眼睛视网膜中包含能够可以感知运动的神经元细胞，但是这些神经元细胞如何感知的信息如何贡献给大脑皮层中的神经元细胞仍然
是一个没有被解决的问题；视网膜中有多个spatial-temporal channel感知运动方向、速度等信息，老鼠大脑皮层包含16个higher
 visual area (HVA)分别倾向于处理不同channel的信息比如颜色、方向。但是视网膜中每个通道的信息如何影响HVA，并且多个HVA
 多个通道的信息如何融合还是未解决的难题。<br>
**SoA**:
**Opportunity**:
**Challenges**:
**Contributions**: 描述了一种从眼睛到大脑皮层细胞特殊的神经回路，该回路传输运动视觉信息；可能潜在有助于治疗大脑感官失调
相关的疾病，比如痴呆、精神分裂；研究发现眼睛中一组细胞可以很好的感知运动信息；
Methods: 用小鼠作为实验动物，将小鼠的部分眼睛里面部分神经细胞感染，来研究小鼠眼睛的神经细胞如何对大脑皮层内部神经细胞感知
运动(visual movement)产生影响；<br>
**future work**: 研究在老鼠不同的运动场景下，视网膜中感应运动的细胞如何和何时发挥作用；


2.[Learning to Extract a Video Sequence from a Single Motion-Blurred Image](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jin_Learning_to_Extract_CVPR_2018_paper.pdf) <br>
2018 CVPR, Meiguang Jin, Givi Meishvili, Paolo Favaro from **University of Bern, Switzerland**. citation = 24. <br>
**key words**: image deblur, motion deblur, blind deblur; <br>
**summary**: 从一张模糊的图片，提取出一个带有时序关系的视频序列；**难点**：一张模糊的图片是多张图片的叠加，叠加的过程破坏了时许关系，想要逆向恢复出时序关系很难；     <br>
**significance**：recovering texture and motion from motion blurred images can be used to understand
the **dynamics of a scene **(e.g., in entertainment with sports or in surveillance when monitoring the traffic). <br>

3.[HRNet-Deep High-Resolution Representation Learning
for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)<br>
2019 PAMI, Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu,
Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao from **Microsoft Research Beijing**, citation=47; <br>
**summary**: HRNet (High Resolution Net), 传统的方法都是把high resolution --> low resolution --> high resolution(ResNet, VGG, etc.),本文觉得这样
会对position-sensitive 的应用不友好；所以本文提出HRNet，把**高低resolution并行连接+不断改变高低resolution特征图的信息**，并且做了一系列下游实验比如:
**human pose estimation, semantic segmentation, object detection, facial landmark detection**;<br>


## 集成学习
### [review]
[Ensemble methods in Machine Learning](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf)<br>
本文是Oregon State University的Thomas G.Dietterich 2000年的工作，引用数已经 > 6000，是比较早的模型集成的综述工作。
本文综述了现有的集成方法，并且解释了为什么集成方法往往比单一的分类器效果好，同时用实验解释了为什么adaboost不容易过拟合。

[Ensemble Learning: A survey](https://onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1249)<br>
本文是2018年的工作，引用量已经到100。


[Hard-Aware Deeply Cascaded Embedding](https://arxiv.org/pdf/1611.05720.pdf)<br>
2017 ICCV, Yuhui Yuan, Kuiyuan Yang, Chao Zhang from **PKU**, citation = 149; <br>
**summary**: Deep metric learning因为样本空间太大，所以要寻找hard example；传统的hard example mining在hard数据上花费大量计算；因为不同模型对hard的
判断不一样，所以本文提出了cascaded model，to ensemble models with different complexity to mine hard examples at different level;

**deep metric learning**:<br>
- 输入两个或者以上成对数据，学习一个度量距离的函数，在embedding space使得相似目标距离近，不相似距离远；应用：face recognition, RE-id, 
- **孪生网络**，可以用来做one shot learning;

## 多模态 multi-modal

1. [Multi-modal Approach for Affective Computing](https://arxiv.org/pdf/1804.09452.pdf)<br>
   [code](https://github.com/zhanghang1989/ResNeSt)<br>
   本文发表在 IEEE 40th International Engineering in Medicine and Biology Conference (EMBC) 2018， 作者来自 UC san diego。<br>

## 优化方法

1. [Fast Exact Multiplication by the Hessian](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)<br>
   本文是Siemens Corporate Research的一位作者在1993年发表在《*Neural Computation*》上面的文章。文章关于神经网络的二阶优化的研究，也是最近重新焕发青春的一个方向。
   由于计算和存储Hessian矩阵（黑塞矩阵）巨大的开销，使得利用二阶梯度优化神经网络变得困难。本文提出一种巧妙的方法，可以方便的计算Hessian矩阵的很多性质，而不用计算
   完整的Hessian矩阵。并且在backpropagation, recurrent backpropagation, Boltzmann Machines上做了实验验证本文方法的有效性。

2. [Practical Gauss-Newton Optimization for Deep Learning](http://proceedings.mlr.press/v70/botev17a/botev17a.pdf)<br>
   本文是2017年的工作，引用量目前34。

3.[Do We Need Zero Training Loss
After Achieving Zero Training Error?](https://arxiv.org/pdf/2002.08709.pdf)<br>
2020 ICLR, **Masashi Sugiyama** et al. from **The U of Tokyo, RIKEN, NEC corporation**, citation = 1; <br>
**summary**: 本文提出一种叫做flooding的方法，在training error为0的时候，阻止training loss -->0, 从而提高了模型的泛化效果和性能;并且，实验表明flodding带来了
test loss double descent curve (测试loss会下降两次)。
**method**：<br>

- **只用改动一行代码**，在loss>flooding level使用梯度下降，loss<flooding level使用梯度上升，使得training loss在flooding level附近浮动，不至于接近0；
- J(theta)' = |J(theta) - b| + b; when J(theta) > b(flooding level), then J(theta)' is the same direction of J(theta), otherwise, opposite;<br>

**problem**: <br>

- 本文防止training loss成为0，但是training loss --> 0是否是对于训练比较好，还是一个open issue;

## data processing

1.Training with streaming annotation  
[原文](https://arxiv.org/abs/2002.04165)<br>
[笔记](https://github.com/KilluaKukuroo/paper-reading/blob/master/Training%20with%20streaming%20annotation.pdf)<br>
本文是Siemens Corporate Technology， UIUC， 剑桥， Information Sciences Institute的四位作者2020年2.13放在Arxiv上的文章。主要解决的问题是， 
在带标注的数据分批次到来，并且新来的数据比以前的数据标注质量好的情况下（streaming），如何更好的利用不同质量的数据进行模型训练。实验是
基于预训练的transformer在NLP里的event extraction task上面做的，但是思想可以很容易的扩展到其他通用领域。

2.[SELECTION VIA PROXY: EFFICIENT DATA SELECTION FOR DEEP LEARNING](https://arxiv.org/pdf/1906.11829.pdf)

2020 ICML splotlight, Cody Coleman∗ , Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, Matei Zaharia from **stanford**. 

**summary**: Data selection is an important task for downstream deep learning tasks to reduce training samples. This paper proposes an efficient data selection method via a small proxy model. Compared to traditional data selection method (**active learning, core-set selection**), traditional method depends on feature representation learning using deep neural network therefore introduces huge computation. In this paper, the authors propose a small proxy network to select data for downstream task. With minor accuracy decrease, this method is 10 times faster than other active learning methods on ImageNet, CIFAR10, etc.  **On CIFAR10, this method could remove 50% of data without harming the accuracy**.

## 智慧电网 smart grid

1. [Application of Big Data and Machine Learning in Smart Grid, and Associated Security Concerns: A Review](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8625421)<br>
本文2018年发表在 IEEE Access上，作者来自 Oregon Tech 等多个高校，综述了大数据机器学习在智慧电网中的应用研究，主要介绍智慧电网的安全隐患。智慧电网(smart grid)是将传统电网和
信息通信技术相结合，实现通信和电能流动的双向性，从而增强电力系统的可靠性安全性和效率。In other words, SG is the
integration of technologies, systems and processes to make power grid intelligent and automated。<br>
本文特点：1.介绍了传统电网和智慧电网各自的特点，以及前者到后者的转换原因，遇到的问题和可能的解决办法；2.介绍了电网系统中的物联网组件和特点，介绍了智慧电网的数据采集，存储，分析，
等模块，大数据和机器学习在智慧电网中的应用，比如需求预测，攻击检测，通过采集的数据预测未来一段时间的风能、太阳能产能； 3.介绍了智慧电网中的网络攻击风险，以及可能的解决办法；

2. [Review on the Research and Practice of Deep Learning and Reinforcement Learning in Smart Grids](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8468674)<br>
本文2018年发表在CSEE，作者来自太原理工和中国电力科学研究院，是关于深度学习和强化学习在智慧电网中应用研究的综述。介绍了深度学习和强化学习在智慧电网中的以下几个方面的研究：
These application fields cover load/power consumption forecasting, microgrids, demand response, defect/fault detection, cyber security, stability analysis, and
nearly all the technical fields of smart grids.



## others

[CICLAD: A Fast and Memory-efficient Closed Itemset Miner for Streams]

2020 KDD, 	Tomas Martin, Guy Francoeur, Petko Valtchev from Centre de recherche en intelligence artificielle (CRIA), UQÀM, Canada;

**summary**: This paper proposes a new intersection-based sliding-window (frequent closed itemset )FCI miner, to mine the association rules. 

**看不懂**



[Parameterized Correlation Clustering in Hypergraphs and Bipartite Graphs]

2020 KDD, Nate Veldt (Cornell University), Anthony Wirth (U of Melbourne), David F. Gleich (Purdue); 

**summary**: This paper solves clustering problem in Hypergraphs and Bipartite Graphs.



[A Non-Iterative Quantile Change Detection Method in Mixture Model with Heavy-Tailed Components]

2020 KDD,Yuantong Li from Purdue, Qi Ma and Sujit K. Ghosh from North Carolina State University;

**summary**: This is a proof-of-concept paper. This paper proposes a data-driven method for parameter estimation in mixture models with heavy-tailed component. While traditionally, most litereatures use iterative Expectation Maximization (EM) method, this paper proposes Non-Iterative Quantile Change Detection (NIQCD) by using change-point detection methods. 



[Learning based distributed tracking]

2020 KDD, Hao Wu, Junhao Gan, Rui Zhang from **U of Melbourne**; 

**summary**: This paper studies the fundamental problem called Distributed Tracking. WIth the popularity of machine learning, people starts to explore the theory of ML via data distribution. This paper follows this line of research and proposes two methods, i.e., w and w/o known of data distribution in advance, to minimize the communication cost in coordinator and players.



[Stable Learning via Differentiated Variable Decorrelation]

2020 KDD, Zheyean Shen: Tsinghua University; Peng Cui: Tsinghua University; Jiashuo Liu: Tsinghua University; Tong Zhang: Hong Kong University of Science and Technology; Bo Li: Tsinghua University; Zhitang Chen: Huawei Noah's Ark Lab

**summary**: This paper studies model robustness, i.e., the model could achieve similar performance in the chaning wild environment. 

**Method**: This paper incorporates the **unlabelled data** from multiple environment in the variable decorrelation framework.