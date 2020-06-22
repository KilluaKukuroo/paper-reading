# This repo is for my brief notes when reading papers.

# 论文笔记和原文
* [优化方法](#优化方法)

* [多模态](#多模态-multi-modal)

* [smart city](#Mobile-&&-smart-city)

* [privacy](#privacy)

* [联邦学习](#联邦学习)

* [deep learning and neural network](#deep-learning)

* [transfer learning](#transfer-learning)

* [小样本学习 && 类别不均衡](#小样本学习-&&-类别不均衡)
* [NLP and web, knowledge graph](NLP-and-web,-knowledge-graph)

* [CV](#CV)

* [集成学习](#集成学习)

* [智慧电网 smart grid](#智慧电网-smart-grid)

1. Training with streaming annotation  
[原文](https://arxiv.org/abs/2002.04165)<br>
[笔记](https://github.com/KilluaKukuroo/paper-reading/blob/master/Training%20with%20streaming%20annotation.pdf)<br>
本文是Siemens Corporate Technology， UIUC， 剑桥， Information Sciences Institute的四位作者2020年2.13放在Arxiv上的文章。主要解决的问题是， 
在带标注的数据分批次到来，并且新来的数据比以前的数据标注质量好的情况下（streaming），如何更好的利用不同质量的数据进行模型训练。实验是
基于预训练的transformer在NLP里的event extraction task上面做的，但是思想可以很容易的扩展到其他通用领域。
 


## 优化方法
1. [Fast Exact Multiplication by the Hessian](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)<br>
本文是Siemens Corporate Research的一位作者在1993年发表在《*Neural Computation*》上面的文章。文章关于神经网络的二阶优化的研究，也是最近重新焕发青春的一个方向。
由于计算和存储Hessian矩阵（黑塞矩阵）巨大的开销，使得利用二阶梯度优化神经网络变得困难。本文提出一种巧妙的方法，可以方便的计算Hessian矩阵的很多性质，而不用计算
完整的Hessian矩阵。并且在backpropagation, recurrent backpropagation, Boltzmann Machines上做了实验验证本文方法的有效性。

2. [Practical Gauss-Newton Optimization for Deep Learning](http://proceedings.mlr.press/v70/botev17a/botev17a.pdf)<br>
本文是2017年的工作，引用量目前34。

## 多模态 multi-modal
1. [Multi-modal Approach for Affective Computing](https://arxiv.org/pdf/1804.09452.pdf)<br>
[code](https://github.com/zhanghang1989/ResNeSt)<br>
本文发表在 IEEE 40th International Engineering in Medicine and Biology Conference (EMBC) 2018， 作者来自 UC san diego。<br>


## Mobile && smart city
### scholars
- [Alex 'Sandy' Pentland - MIT media lab](https://scholar.google.com/citations?hl=zh-CN&user=P4nfoKYAAAAJ&view_op=list_works&sortby=pubdate)<br>
- [Yu Zheng - JD.COM](https://scholar.google.com/citations?hl=zh-CN&user=juUcdgYAAAAJ&view_op=list_works&sortby=pubdate)<br>
- [Jie Feng](https://vonfeng.github.io/publications/) ** Ph.D candidate in THU **,  [Yong Li](http://fi.ee.tsinghua.edu.cn/~liyong/) ** Associate Prof in THU, mobile computing + ML ** <br>
- [Lijun Sun](https://lijunsun.github.io/) AP at McGill Univ, machine learning + smart city <br>
- [Xue (Steve) Liu](https://www.cs.mcgill.ca/~xueliu/site/intro.html) FIEEE, McGill Univ, *IoT,CPS,ML,smart energy system* <br>
- [Xiaohui Yu](http://www.yorku.ca/xhyu/) Associate Prof, York University, *data mining, transportation,location-based services,social network*<br>

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

6.[Identifying Human Mobility via Trajectory Embeddings](https://www.ijcai.org/Proceedings/2017/0234.pdf)<br>
2017,IJCAI. Qiang Gao, Fan Zhou, Kunpeng Zhang, Goce Trajcevski, Xucheng Luo, Fengli Zhang from **UESTC, U Maryland, Northwesten U**, citation = 42; <br>

**summary**: 传统工作是将轨迹数据分类为不同行为，而本文提出解决trajectory user linking (TUL),将轨迹与产生轨迹的人结合起来，面对的挑战是类别数太多(人多) + 数据稀疏；为了解决这些问题，提出RNN-based semi-supervised model; <br>

**method**: 1)trajectory segmentation 2)check-in embedding 3)RNN-based semi-supervised model; <br>

**contribution**： 1)第一个解决TUL问题的工作； 2)提出了RNN-based 半监督模型(**获取embedding的过程是半监督**)，在两个公开数据集取得了SoA成绩； <br>

[MPE: A Mobility Pattern Embedding Model for Predicting Next Locations](https://arxiv.org/pdf/2003.07782.pdf)<br>
2020 WWW, Meng Chen · Xiaohui Yu (**York University**) · Yang Liu. citation = 9;   <br>
**keywork**: human mobility modeling, embedding learning, next location prediction, **traffic trajectory data**;   <br>
**summary**: 使用embedding（没有使用神经网络）对交通轨迹数据进行分析，可以运用到next location prediction, visualization; <br>
**contribution**: <br>
- 第一个使用embedding method to model mobility pattern from **traffic trajectory data**;    
- ocnsidering sequential, temporal and personal information when embedding data into vector; 
- VPR data and taxi trajectory data 数据集上取得了很好的成绩； 


[Personalized Ranking Metric Embedding for Next New POI Recommendation](https://www.ijcai.org/Proceedings/15/Papers/293.pdf)<br>
2015 IJCAI, Shanshan Feng,1 Xutao Li,2 Yifeng Zeng,3 Gao Cong,2 Yeow Meng Chee,4 Quan Yuan from **NTU**;  <br>


[A General Multi-Context Embedding Model for Mining Human Trajectory Data](https://ieeexplore.ieee.org/document/7447767)<br>
2016, TKDE.     <br>






9.[Route prediction for instant delivery](https://dl.acm.org/doi/pdf/10.1145/3351282) <br>
2019 Ubicomp, Yan Zhang, Yunhuai Liu, Genjian Li, Yi Ding, Ning Chen, Hao Zhang, Tian He, and Desheng Zhang.    <br>

**summary**: 通过对骑手路径选择的预测，制定不同的订单分发系统，从而减少平均的外卖派送时间，减少延误率； <br>



10.[Learning to Estimate the Travel Time](https://dl.acm.org/doi/pdf/10.1145/3219819.3219900)<br>
2018 KDD, Zheng Wang, Kun Fu, Jieping Ye from **DiDi Chuxing**; <br>
**summary**: 将车辆从A到B的estimated time of arrival(ETA) 转化为机器学习的回归问题，利用大量的历史数据，设计 Wide-Deep-Recurrent (WDR) learning model预测旅行时间，并且在DIDI上验证了这个方法； <br>

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
**summary**: 以前的ETA估计方法是：将路径划分成为多个子路径，计算子路径时间进行叠加。但是这样不准确，因为没有考虑road intersection/traffic light等情况。所以本文提出一种end-to-end模型，直接预测到达时间; <br>

**datasets**: chengdu dataset (1.4 billion GPS records) + beijing dataset (0.45 billion GPS records); 没说明是公开的还是自己的数据,应该是自己采集的数据; <br>

**methods**:  1) transform GPS sequence to feature maps, to capture local spatial correlations;   <br>


13.[TEMP-A Simple Baseline for Travel Time Estimation using Large-Scale Trip Data](https://arxiv.org/pdf/1512.08580.pdf)<br>
[short version](https://dl.acm.org/doi/pdf/10.1145/2996913.2996943)<br>
**keywork**: ETA;    <br>
2016 SIGSPATIAL,    citation=51; <br>








### location and trajectory prediction
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
**summary**:本文的核心是基于tweet信息，预测三种位置信息(i.e., home tweet mentioned location)，不是预测mobility里面的下一个位置；并且简单综述了两个研究方向：semantic location,
point-of-interest recommendation; <br>




[Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11900/11583)<br>
2016, AAAI, citation = 315; <br>




[Social Bridges in Urban Purchase Behavior](https://dl.acm.org/doi/pdf/10.1145/3149409)<br>
2017 TIST, Xiaowen DONG and Yoshihiko Suhara (MIT), BURÇIN BOZKAYA (Sabancı University), VIVEK K. SINGH (Rutgers University), BRUNO LEPRI (Fondazione Bruno Kessler),
**Alex Pentland** from MIT, citation=17; <br>

**summary**: 利用social bridge的概念，对城市居民的购买力进行建模； <br>


## privacy
## scholars
- [Cynthia Dwork](https://scholar.google.com/citations?user=y2H5xmkAAAAJ&hl=zh-CN), distinguished scientist in **Microsoft Research**, citation>30,000;
- [Aaron Roth](https://scholar.google.com/citations?user=kLUQrrYAAAAJ&hl=zh-CN), Associate professor in **U Pennsylvania**, citation > 7,000;
- [Jie Feng](https://vonfeng.github.io/publications/) Ph.D candidate in THU<br>

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

3.[Trajectory Recovery From Ash: User Privacy Is NOT Preserved in Aggregated Mobility Data](https://arxiv.org/pdf/1702.06270.pdf)<br>
2017 WWW.

4.[De-anonymization of Mobility Trajectories: Dissecting the Gaps between Theory and Practice](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_06B-3_Wang_paper.pdf)<br>
2018 NDSS. 



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




## 联邦学习
## scholars
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




## deep learning
1.[ResNeSt: Split-Attention Networks](https://hangzhang.org/files/resnest.pdf)<br>
发表于2020年 arxiv，作者来自 Amason, UC Davis, 包括 Hang Zhang, Mu Li. 网上传言史上最强resnet魔改版。 <br>
**Problem**：目前大部分视觉的任务, e.g., obeject detection and semantic segmentation 还是使用ResNet的变体作为backbone，因为网络结构的简单和结构化。但是ResNet是为了image classification设计，
在CV的其他下游任务性能不是很好，可能因为limited receptive-field and lack of cross-channel interaction. 并且resnet的各种变体往往只能在特定的任务上取得较好性能。而且，最近的cross-channel information
在下游任务被证明很有效，而image classification的模型大都缺乏cross-channel interation，所以本文提出一种带有cross-channel representation的网络模型，目标是*打造一个versatile backbone with universally 
improved feature representation*, 从而同时提高多个任务的性能。 <br>
**SoA**：AlexNet -> NIN(1*1 convolution) -> VGG-Net(modular network design) --> Highway network(highway connection) --> ResNet(identity skip connection); NAS;
GoogleNet(muiti-path representation) --> ResNeXt(group convolution) --> SE-Net(channel-attention) --> SK-Net(feature map attention across two network branches);<br>
**contribution**：1）研究了带有feature map split attention 的resnet网络结构；
2) 在image classification 和其他transfer learning的应用场景种提供了一个大规模的benchmark, 刷新了SoA，在不同任务上分别提高1~3个点;<br>
**future work**: 通过神经网络结构搜索寻找不同硬件上对应得低延时low latency model; 不同的超参数(radix, cardinality, width)组合调优，可能会在不同的具体任务上取得更好的结果；
增加图片的size 可以提高acc；

2.[Backpropagation and the Brain](https://www.nature.com/articles/s41583-020-0277-3.pdf)<br>
发表在2020年《nature reviews|neuroscience》, 作者包括Geoffrey Hinton. <br>
大脑皮层如何修改突触，从而实现学习是一个很神秘的问题。几十年前，backpropagation 被认为是可以用来解释大脑学习机制的一个可能方法，但是由于反向传播机制没有带来很好的网络学习效果，
并且反向传播缺乏生物学上的可解释性，反向传播的意义被忽视。但是，随着近年算力的提升，NN在多个领域以反向传播为学习方法的基础上取得了很好的成绩，我们认为 backpropagation offers
a conceptualframework for understanding how the cortex learns. <br>
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
- ‘如何在没有标签的情况下发现DNN系统的错误'？ -- 利用multiple DNNs with similar functionality; <br>
**limitations**: <br>
- differential testing要求至少两个有相同功能的DNN系统；而且，如果两个相同功能的DNN只有很小的区别（few neurons difference），系统需要很长时间寻找differential-inputs; <br>
- differential testing只能在至少有一个DNN做出不一样的结果的时候检测出错误，如果所有DNN都犯同样的错，则检测不出来对应的test case；


## transfer learning




## 小样本学习 && 类别不均衡
1.[decoupling representation AND classifier FOR LONG-TAILED RECOGNITION](https://arxiv.org/pdf/1910.09217.pdf)<br>
2020 ICLR, Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis 来自Facebook AI and NUS; <br>



2.[Generalizing from a Few Examples: A Survey on Few-Shot Learning](https://arxiv.org/pdf/1904.05046.pdf)<br>
2020 , Yaqing Wang (HKUST and Baidu Research), Quanming Yao (4Paradigm), JAMES T. KWOK and LIONEL M. NI (HKUST);   <BR>


## NLP and web, knowledge graph
1.[Correcting Knowledge Base Assertions](https://arxiv.org/pdf/2001.06917.pdf)<br>
本文发表在2020WWW的oral，作者来自Oxford, Tencent, University of Oslo. <br>
**Problem**: knowledge bases(KB) 在搜索引擎，问答系统，common sense reasoning, machine learning等领域起到了重要的作用，but KB is suffering from quality issues, e.g.,
constraint violations and erroneous assertions.  <br>
SoA: 现有工作在KB quality上主要包含：error detection and assessment, quality improvement via completion, canonicalization.<br>
**Opportunity**: 现有的工作大都是检测出 KB中的错误之后就将错误去除，而不是去更正这些错误。所以更正检测出的erroneous assertions is a new opportunity; 关于correction，现有工作在忽视了assertion的上下文语义信息；
关于quality improvement，现有工作没有提出一个general correction method；<br>
**Challenges**: 
**Contributions**: 1-提出了可以用于更正错误的entity assertions and literal assertions的通用框架；2-利用了semantic embedding and observed features捕捉局部的上下文信息；3-soft property constraint;
4-在meidcal KB验证entity assertions correction, 在DPpedia验证literal assertions correction;  <br>


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

## 集成学习
### [review]
[Ensemble methods in Machine Learning](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf)<br>
本文是Oregon State University的Thomas G.Dietterich 2000年的工作，引用数已经 > 6000，是比较早的模型集成的综述工作。
本文综述了现有的集成方法，并且解释了为什么集成方法往往比单一的分类器效果好，同时用实验解释了为什么adaboost不容易过拟合。

[Ensemble Learning: A survey](https://onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1249)<br>
本文是2018年的工作，引用量已经到100。


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



