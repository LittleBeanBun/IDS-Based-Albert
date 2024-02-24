# 基于ALBERT的车辆入侵检测算法

## 1.项目简介

在该项目中，我们提出了一种基于ALBERT模型的CAN总线入侵检测算法，通过将报文ID转换成图像进行分类，成功解决了CAN总线入侵检测算法前期**人工特征提取步骤繁琐**和**无法给出具体攻击类型**等问题。在**Car-hacking**数据集上，算法取得了99.7%的分类精度。项目中的挑战包括数据集处理复杂和ALBERT模型过大、训练时间长等问题，我们通过模型压缩和轻量化克服了这些困难，并取得了显著的成果。该算法具有广泛的应用前景，可用于汽车网络安全领域的入侵检测和其他相关领域。

## 2.项目背景

与传统的互联网入侵检测系统不同，车辆网络中已知的攻击特征很少。此外，车辆的IDS要求高精度，因为任何假阳性错误都可能严重影响驾驶员的安全。且由于车载网络缺乏认证和加密机制，入侵检测系统（IDS）是保护现代车载系统免受网络攻击的重要手段。
![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/67d198c2-1d1d-457b-a057-810981325c48)



## 3.数据集

<p>Car-hacking数据集 项目详情可见：https://download.csdn.net/download/m0_70420861/88872592</p>


![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/d5e27e5e-af77-4342-bdd3-6b91d66a347f)


![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/26c7c860-2d48-4d32-aedd-980cfbb5da12)





## 4.实验环境

相关技术：Albert One-Hot编码

计算机GPU 型号为 NVidia GTX1080 12G，内存16G，操作系统为Windows10，编程语言为 Python，采用谷歌深度学习框架TensorFlow2.3搭建网络，采用的ALBERT-base模型为预训练模型“ALBERT-base-V2”模型


## 5.方法和原理

<h4>数据处理：使用One-Hot编码将ID转换为图片</h4>

![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/28379836-ee3d-4130-ac59-a65f7592b1ec)
![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/6e6777ae-35d7-4613-8f39-039cbe436bac)




![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/3c1c39d4-9943-4f2e-b17b-46a1481e097b)

<h3>Transformer与BERT结构</h3>

<h4>Transformer的本质上是一个Encoder-Decoder的结构，和经典的 seq2seq 模型一样，Transformer 模型中也采用了 Encoder-decoder 的架构。
左半边用代表一个encoder单元，Nx表示有N个encoder单元，右半边则代表一个decoder单元。</h4>

![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/72738a92-fbc8-420d-bd31-c6cf0210894e)


![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/6db5d9a4-2bca-4127-801a-32240654df87)
<p><h4>Q,K,V三个向量分别代表词向量的三个属性。Q(Query)可以理解为词向量E在当前训练语料下的注意力权重，它保存了剩下每个词与E之间的关系；K(key)可以理解为权重索引，通过用别的词的注意力索引K(key)与E的注意力权重(Query)相乘，就可以得到其他词对E的注意力加权；V(value)可以理解为在当前训练语料下的词向量。Multi-Head Attention机制相当于h个不同的self-attention的集成，每次self-attention中得到的向量Q,K,V进行线性变换，每次采用的变换矩阵不同，最后把得到的不同的输出结果拼接起来，再进行一次线性映射后得到最终结果。与卷积神经网络中设置多个卷积核的目的类似，可以允许模型在不同的表示子空间里学习到相关的信息。</h4></p>

![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/26be55e8-9c42-4d86-bd6d-e05baae2222a)



![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/a3d5f814-9f0e-4cb7-ba6f-46d8be24c59b)


## 6. 实验结果和结论

![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/f5cda651-faef-4321-802a-d2b6214b677e)

![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/76229e31-fd90-4e29-bd1b-1e1e4cba54a9)

![image](https://github.com/LittleBeanBun/IDS-Based-Albert/assets/152876743/5a1e2e9d-55c2-44cd-9e04-587ee2421c1f)
<p><h4>该算法将CAN总线发出的报文ID作为文本输入给模型，不用对报文数据场内容进行特征提取，便可对不同种类的攻击报文进行分类。经过ALBERT-base模型(Transformer Encoder的数量为12)后，将输出经过Softmax分类层输出预测的攻击类别，将真实类别与预测类别输入交叉熵损失函数中，通过background机制更新参数，经过一定数量的迭代过程即可完成模型的微调训练。该模型不仅可以对报文的正常或异常做出判断，还能给出异常报文的种类，方便驾驶员做出相应的措施。</h4></p>

## 7.项目总结


<p>1.将提取的报文特征按时序以块为单位进行数据转换，可以在保证攻击特征的有效性的同时保证时间序列性。</p>
<p>2.将经数据转换后得到的图像作为模型输入，对车内外网络中的入侵信号进行检测和分类。经过实验验证，模型的检测和分类性能表现很好。
  不仅可以检测出未知攻击还能对已知攻击进行检测和分类，以便按攻击类型进行及时响应。</p>
<p>3.模型采用迁移学习的方法，模型训练稳定，泛化能力也得到增强。且由于是黑盒的预训练模型，不会被攻击者操纵而使得算法失效。</p>

具体可见ppt: <a>https://download.csdn.net/download/m0_70420861/88872601?spm=1001.2014.3001.5503</a>



## 参考文献

<p>[1] T. Hoppe, S. Kiltz, and J. Dittmann, “Security threats to automotive can networks–practical examples and selected short-term countermeasures,” Computer Safety, Reliability, and Security, pp. 235–248, 2008.</p>
<p>[2] S. Abbott-McCune and L. A. Shay, “Intrusion prevention system of automotive network can bus,” in Security Technology (ICCST), 2016 IEEE International Carnahan Conference on. IEEE, 2016, pp. 1–8.</p>
<p>[3] H. M. Song, H. R. Kim, and H. K. Kim, “Intrusion detection system based on the analysis of time intervals of can messages for in-vehicle network,” in Information Networking (ICOIN), 2016 International Conference on. IEEE, 2016, pp. 63–68.</p>
<p>[4] H. Lee, S. H. Jeong, and H. K. Kim, “Otids: A novel intrusion detection system for in-vehicle network by using remote frame.”</p>
<p>[5] T. Matsumoto, M. Hata, M. Tanabe, K. Yoshioka, and K. Oishi, “A method of preventing unauthorized data transmission in controller area network,” in Vehicular Technology Conference (VTC Spring), 2012 IEEE 75th. IEEE, 2012, pp. 1–5.</p>
<p>[6] M. M¨uter and N. Asaj, “Entropy-based anomaly detection for in-vehicle networks,” in Intelligent Vehicles Symposium (IV), 2011 IEEE. IEEE, 2011, pp. 1110–1115.</p>
<p>[7] M. Marchetti and D. Stabili, “Anomaly detection of can bus messages through analysis of id sequences,” in Intelligent Vehicles Symposium (IV), 2017 IEEE. IEEE, 2017, pp. 1577–1583.</p>
<p>[8] M. Markovitz and A. Wool“, Field classification, modeling and anomaly detection in unknown can bus networks,” Vehicular Communications, vol. 9, pp. 43–52, 2017.</p>
<p>[9] N. SALMAN and M. BRESCH, “Design and implementation of an intrusion detection system (ids) for in-vehicle networks.”</p>
<p>[10] Kang M. J. and Kang J. W., “Intrusion detection system using deep neural network for in-vehicle network security”, PLoS ONE, 11(6),pp. 35-41, 2016</p>
<p>[11] Kang M. J. and Kang J. W., “A novel intrusion detection method using deep neural network  for in-vehicle network security,” IEEE 83rd Veh. Technol. Conf.(VTC Spring), 2016, pp. 1-5. </p>
<p>[12] Taylor A., Leblanc S. and Japkowicz N. “Anomaly detection in auto-mobile control network  data  with  long  short-term  memory  networks,” IEEE Int. Conf. Data Sci. Adv. Anal(DSAA), 2016, pp. 130-139.</p>
<p>[13] Narayanan S. N., Mittal S. and Joshi A, “OBD SecureAlert: An anormaly detection system for vehicles,” IEEE Int. Conf. Smart Comput(SMARTCOMP), 2016, pp.1-6. </p>
<p>[14] Studnia I., Alata E., Nicomette  V. and Kaâniche M, “A language-based intrusion detection approach for automotive embedded networks,” Int. J. Embedded Syst., 10(1), pp. 1-12, 2018</p>
<p>[15] 杨宏, “基于智能网联汽车的 CAN 总线攻击与防御检测技术研究,” 天津理工大学, 2017.</p>
<p>[16] 曾润, “车载 CAN 总线网络异常数据检测技术研究与实现,” 北京邮电大学, 2018.</p>
<p>[17] M. Zhang, B. Xu, S. Bai, S. Lu, and Z. Lin, “A deep learning method to detect web attacks using a specially designed cnn,” in International Conference on Neural Information Processing. Springer, 2017, pp. 828–836.</p>
<p>[18] O. Gorokhov, M. Petrovskiy, and I. Mashechkin, “Convolutional neural networks for unsupervised anomaly detection in text data,” in International Conference on Intelligent Data Engineering and Automated Learning. Springer, 2017, pp. 500–507.</p>
<p>[19] T. Schlegl, P. Seeb¨ock, S. M. Waldstein, U. Schmidt-Erfurth, and G. Langs, “Unsupervised anomaly detection with generative adversarial networks to guide marker discovery,” in International Conference on Information Processing in Medical Imaging. Springer, 2017, pp. 146–157.</p>
<p>[20] Li D, Chen D, Goh J, et al. Anomaly detection with generative adversarial networks for multivariate time series[J]. arXiv preprint arXiv:1809.04758, 2018.</p>
<p>[21] Seo E, Song H M, Kim H K. Gids: Gan based intrusion detection system for in-vehicle network[C]. 2018 16th Annual Conference on Privacy, Security and Trust (PST). IEEE, 2018: 1-6.</p>
<p>[22] Nicholas L, Ooi S Y, Pang Y H, et al. Study of long short-term memory in flow-based network intrusion detection system[J]. Journal of Intelligent & Fuzzy Systems, 2018, 35(6): 5947-5957.</p>
<p>[23] Agarap A F M. A neural network architecture combining gated recurrent unit (GRU) and support vector machine (SVM) for intrusion detection in network traffic data[C]//Proceedings of the 2018 10th International Conference on Machine Learning and Computing. 2018: 26-30.</p>
<p>[24] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need [C]. Advances in Neural Information Processing Systems. 2017: 5998-6008.</p>
<p>[25] Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.</p>
<p>[26] Lan Z, Chen M, Goodman S, et al. Albert: A lite bert for self-supervised learning of language representations[J]. arXiv preprint arXiv:1909.11942, 2019.</p>


