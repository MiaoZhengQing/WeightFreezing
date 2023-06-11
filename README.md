# WeightFreezing

# Description
Source code for the paper: Weight-Freezing: A Regularization Approach for Fully Connected Layers with an Application in EEG Classification
Submitted to Neural Networks

# Requirements
- Python == 3.6 or higher
- Pytorch == 1.10 or higher
- GPU is required. 

# Contributions
- To the best of our knowledge, this paper is the first to study the impact of the classifier in ANNs on EEG decoding performance. For this purpose, Weight-Freezing is proposed, which suppresses the influence of some input neurons on certain decision results by freezing some parameters in the fully connected layer, thereby achieving higher classification accuracy.
- Weight-Freezing is also a novel regularization method, which can achieve sparse connections in the fully connected network.
- Weight-Freezing is thoroughly validated and analyzed in three classic decoding networks and three highly cited public EEG datasets. The experimental results confirm the superiority of Weight-Freezing in classification and have also achieved state-of-the-art classification performance (averaged across all participants) for all the three highly cited datasets.

This study's primary contribution lies in its potent facilitation of the application and implementation of Artificial Neural Network (ANN) models within Brain-Computer Interface (BCI) systems. Simultaneously, it sets a new performance benchmark for future EEG signal decoding efforts using more sizable models, such as transformers.
Emerging research is increasingly adopting transformer networks for EEG signal decoding. These approaches can be viewed as enrichments to existing ANN models, as they elevate EEG classification accuracy via more sophisticated feature extraction networks. However, these enhancements have inadvertently complicated the deployment of these ANN models in real-world BCI systems.
In a stark contrast, our study introduces Weight-Freezing as an innovative, subtractive strategy that refines existing ANN models. Empowered by Weight-Freezing, some lightweight and shallow decoding networks surpass all current transformer-based methods in terms of classification performance on identical public datasets.
The incorporation of Weight-Freezing not only simplifies the deployment of ANN models within BCI systems but also sets a new performance standard for the deployment of larger models, such as transformers, in the future. Moreover, it provokes an intriguing question in the realm of EEG decoding: Is the deployment of large models like transformers for EEG feature extraction truly indispensable?

# Results
![33f3428681103234abb0acb07c6a6ca](https://github.com/MiaoZhengQing/WeightFreezing/assets/116713490/abb617bd-f3ae-418f-9dd5-5ffb24cbbb4f)
![6b598f8a5dfeff920c909b9f93f4a09](https://github.com/MiaoZhengQing/WeightFreezing/assets/116713490/5a86123d-852c-405d-b98b-539e039243a6)

# Models Implemented
- [LMDA-Net](https://doi.org/10.1016/j.neuroimage.2023.120209)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [ShallowConvNet](https://github.com/TNTLFreiburg/braindecode)

# Related works
- This paper is a follow-up version of [SDDA](https://arxiv.org/pdf/2202.09559.pdf) and [LMDA-Net](https://doi.org/10.1016/j.neuroimage.2023.120209), the preprocessing method is inherited from SDDA.


# Paper Citation
If you use this code in a scientific publication, please cite us as:  
% TSFF-Net  
Miao Z, Zhao M. Time-space-frequency feature Fusion for 3-channel motor imagery classification[J]. arXiv preprint arXiv:2304.01461, 2023.

% LMDA-Net  
Miao Z, Zhang X, Zhao M, et al. LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interface paradigms and interpretability[J]. arXiv preprint arXiv:2303.16407, 2023.

% SDDA  
Miao Z, Zhang X, Menon C, et al. Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework[J]. arXiv preprint arXiv:2202.09559, 2022.

```
% TSFF-Net
@article{miao2023time,
  title={Time-space-frequency feature Fusion for 3-channel motor imagery classification},
  author={Miao, Zhengqing and Zhao, Meirong},
  journal={arXiv preprint arXiv:2304.01461},
  year={2023}
}

% LMDA
@article{miao2023lmda,
title = {LMDA-Net:A lightweight multi-dimensional attention network for general EEG-based brain-computer interfaces and interpretability},
journal = {NeuroImage},
volume = {276},
pages = {120209},
year = {2023},
issn = {1053-8119},
doi = {https://doi.org/10.1016/j.neuroimage.2023.120209},
url = {https://www.sciencedirect.com/science/article/pii/S1053811923003609},
author = {Zhengqing Miao and Meirong Zhao and Xin Zhang and Dong Ming},
keywords = {Attention, Brain-computer interface (BCI), Electroencephalography (EEG), Model interpretability, Neural networks},
abstract = {Electroencephalography (EEG)-based brain-computer interfaces (BCIs) pose a challenge for decoding due to their low spatial resolution and signal-to-noise ratio. Typically, EEG-based recognition of activities and states involves the use of prior neuroscience knowledge to generate quantitative EEG features, which may limit BCI performance. Although neural network-based methods can effectively extract features, they often encounter issues such as poor generalization across datasets, high predicting volatility, and low model interpretability. To address these limitations, we propose a novel lightweight multi-dimensional attention network, called LMDA-Net. By incorporating two novel attention modules designed specifically for EEG signals, the channel attention module and the depth attention module, LMDA-Net is able to effectively integrate features from multiple dimensions, resulting in improved classification performance across various BCI tasks. LMDA-Net was evaluated on four high-impact public datasets, including motor imagery (MI) and P300-Speller, and was compared with other representative models. The experimental results demonstrate that LMDA-Net outperforms other representative methods in terms of classification accuracy and predicting volatility, achieving the highest accuracy in all datasets within 300 training epochs. Ablation experiments further confirm the effectiveness of the channel attention module and the depth attention module. To facilitate an in-depth understanding of the features extracted by LMDA-Net, we propose class-specific neural network feature interpretability algorithms that are suitable for evoked responses and endogenous activities. By mapping the output of the specific layer of LMDA-Net to the time or spatial domain through class activation maps, the resulting feature visualizations can provide interpretable analysis and establish connections with EEG time-spatial analysis in neuroscience. In summary, LMDA-Net shows great potential as a general decoding model for various EEG tasks.}
}

% SDDA
@article{miao2022priming,
  title={Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework},
  author={Miao, Zhengqing and Zhang, Xin and Menon, Carlo and Zheng, Yelong and Zhao, Meirong and Ming, Dong},
  journal={arXiv preprint arXiv:2202.09559},
  year={2022}
}
```

# Contact
Email: mzq@tju.edu.cn
