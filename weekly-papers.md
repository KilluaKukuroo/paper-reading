
# 每周2篇精读论文

## 2020-8-10 to 8-16

1.[Multi-scale representation learning for spatial
feature distributions using grid cells](https://arxiv.org/abs/2003.00824)<br>
[openreview link](https://openreview.net/forum?id=rJljdh4KDH)<br>

2020 ICLR spotlight, Gengchen Mai, Krzysztof Janowicz, Rui Zhu, Ling Cai from **UCSB**, Bo Yan **LinkedIn**, Ni Lao from **SayMosaic Inc.**. 

**summary**：Inspired by unsupervised text encoding model in NLP, this paper propose Space2Vec to learn representation for space, i.e., model spatial position and spatial contexts of geographic objects such as POIs. **As in NLP word is embedded based on location in the sentence and nearby context, and GIS point can also be embeded to a vector based on spatial position and nearby samples**. Inspired by Nobel-winner neurosience research on grid cells, this paper applys multi-scale representation learning based on grid-cell. Experiments are conducted on predicting types of POIs, and image classification leveraging geo-locations. Futhermore, firing patterns are visualized to capture how encoding layers handle spatial structures at different scales.

**opportunity**:

- NLP unsupervised embedding method
- Nobel prize winning research on multi-scale grid cell representation;

**challenge (Why non-trivial)**:

- different POI types have very different distribution charateristics. How to jointly discribe these distributions and patterns? 

**contributions**:

- 1) propose a encoder-decoder to model absolute spatial positions and spatial context.
-  2) conduct two experiements to validate the model: a.POI types classification based on positions and context; b. fine-grained image classification leveraging geo-locations;
- 3) firing patterns visualization to show how different layers in encoder hander spatial structures. 

**method**:

- **TODO**: unsupervise learning??

**experiments**:

- on Yelp Data Challenge dataset: POI type prediction; location modeling and spatial context modeling;
- on BirdSnap and NABirds datasets: fine-grained image classification, baselines are also works who consider geo-locations from paper *Presence-Only Geographical Priors for Fine-Grained Image Classification*; the assumption is utilizing geo-location info could help distinguish between similar classes;

**keys**:

- unsupervised learning
- multi-scale representation
- firing patterns visualization

## 2020-8-17 to 8-23

1.[SIGIR 2020 Hinton speech](https://www.jiqizhixin.com/articles/2020-07-29-3)<br>

[ABSTRACT-The next generation of Neural networks](https://sigir.org/sigir2020/assets/files/SIGIR20.pdf)

**summary**：类比人脑中的神经元连接和生命的长度，Hinton指出无法单纯依靠有监督学习来训练所有的神经元，所以无监督学习才是方向。人工神经网络一直以来面临的一个问题就是：如何像人类大脑一样来高效的训练神经元。目前有两种无监督学习的方法：

- 1）like BERT and VAE, a neutwork is trained to reconstruct its input; 但是对于图片很难，因为要求深层网络重构图片的细节，很难实现；（因为计算量大？）
- 2）the second method is proposed by Becker and Hinton in 1992, 训练两个同样的网络，给两个网络输入图片的不同部分而网络的输出有很高的mutual information；这种方法可以使得网络学习representation并且忽视输入数据的无关细节；这种方法有一个缺点：to learn pairs of vector representation, 需要从2^N候选向量中挑选类似的。所以接下来hinton提出一种有效的方法来解决这个问题。

autoencoder： “A way to implement unsupervised learning by using supervised learning”.

现在无监督的一个问题是过于关注数据的重构损失，忽略了数据之间关系的捕捉。因此需要“contrastive loss”捕捉数据的局部关系。

**QA**：

- 我们可以在模型中加入人工的constraint吗？比如会议论文推荐系统，我们想要尽可能的推荐学生的文章。
  - 可以。方法就是把我们的先验限制放在objective function内部。
- Can we learn word embedding using vision?
  - Yes. 我们实验室有学生做：首先将word在谷歌搜索相对应的图片，然后对word和image做embedding，然后做结合。类似的论文: "Learning Multilingual Word Embeddings Using Image-Text Data";

2.[Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)<br>
[Blog for Transfomer](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)<br>

2017 NIPS, 

[Blog-Illustrated Guide to Transformers- Step by Step Explanatio](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)<br>







