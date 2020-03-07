# This repo is for my brief notes when reading papers.

# 论文笔记和原文
1. Training with streaming annotation
[原文](https://arxiv.org/abs/2002.04165)<br>
[笔记](https://github.com/KilluaKukuroo/paper-reading/blob/master/Training%20with%20streaming%20annotation.pdf)<br>
本文是Siemens Corporate Technology， UIUC， 剑桥， Information Sciences Institute的四位作者2020年2.13放在Arxiv上的文章。主要解决的问题是，
在带标注的数据分批次到来，并且新来的数据比以前的数据标注质量好的情况下（streaming），如何更好的利用不同质量的数据进行模型训练。实验是
基于预训练的transformer在NLP里的event extraction task上面做的，但是思想可以很容易的扩展到其他通用领域。

2. 



## 优化方法
1. Fast Exact Multiplication by the Hessian
[原文](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)<br>
本文是Siemens Corporate Research的一位作者在1993年发表在*《Neural Computation》*上面的文章。文章关于神经网络的二阶优化的研究，也是最近重新焕发青春的一个方向。
由于计算和存储Hessian矩阵（黑塞矩阵）巨大的开销，使得利用二阶梯度优化神经网络变得困难。本文提出一种巧妙的方法，可以方便的计算Hessian矩阵的很多性质。
并且在backpropagation, 