---
layout: post
title: "Day39. Quantization, Distilation"
subtitle: "모델 압축을 위한 양자화, 지식 증류"
date: 2021-03-18 23:09:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> Quantization 기법과 Distilation 기법에 대해 배웠다.  
      
이 글은 아래와 같은 내용으로 구성된다.  
- [Quantization](#quantization)
- [Knowledge Distilation](#knowledge-distilation)
- [Logit, Sigmoid, Softmax](#logit-sigmoid-softmax)
- [Reference](#reference)
  
<br />
  
## Quantization  
![quantization](/img/posts/39-1.png){: width="90%" height="90%"}{: .center}  
Quantization은 float16, float32 등의 FP number를 int8 등의 정수 보다 적은 bit의 자료형으로 mapping하여 정보를 잃는 대신 필요한 메모리를 줄이고 컴퓨팅 속도를 빠르게 하기 위한 compression 기법이다.   
  
예를 들어 절댓값 1 이하의 FP number를 int 8로 quantization한다고 하면, -128부터 127까지의 수에 quantization unit $\delta$를 $1/128$로 설정하여 mapping할 수 있다. 

<center>

$$
-1 \; \sim \; 1
$$
$$
\rightarrow \delta \times -128(-1.0) \; \sim \; \delta \times 127 (0.992) 
$$

</center>  
  
앞서 quantization의 효과에 메모리 절약 뿐만 아니라 **연산 속도 가속화**도 있다고 언급하였다. 
이는 FPU(Floating Point Unit)가 하던 연산을 ALU(Arithmetic and Logic Unit)가 대체할 수 있게 되기 때문이다.   
   
한편, 지금까지 보고 떠오르는 생각은 여러 layer들의 weight 값들을 quantization하면 되겠다라는 것인데, 이를 보다 효율적으로 활용하려면 **activation에도 quantization이 적용되어야한다.**  
  
잘 생각해보면 결국 어떤 layer에 들어오는 input부터가 quantization되어있는 상태라면 computational cost를 크게 줄일 수 있다.
그래서 보통 아래와 같이 weight과 activation(ReLU) 모두에 quantization을 적용한다.  

![quantization_activation](/img/posts/39-2.png){: width="90%" height="90%"}{: .center}    
k-bit quantization + ReLU는 우측 상단에 나와있듯이 해당 bit수로 양자화된 ReLU function을 말한다.   
  
문제는 이렇게 하면 backward pass에서 미분이 안된다는 점인데, 이것은 forward pass에서 quantize 되기 이전의 미분 값을 그대로 사용하는 트릭을 사용하여 해결할 수 있다.  
  
<center>

$$
y = Quantize(x)
$$
$$
E = f(y)
$$
<br />
$$
\frac{dy }{dx } = \frac{d}{dx} Quantize(x) ^{\prime} \rightarrow 1
$$
$$
\begin{aligned}
\frac{\partial E}{\partial x} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial x} \\
&= \frac{\partial E}{\partial y} \cdot 1
\end{aligned}
$$

</center>

  
![relu_smoothing](/img/posts/39-3.png){: width="90%" height="90%"}{: .center}    
혹은 위와 같이 ReLU function을 아래 함수로 smoothing하여 quantization을 적용하는 동시에 미분 가능하도록 만들어 활용하는 방법도 있다.   
  
<center>

$$
\sigma(Tx) = \frac{1}{1 + \exp(-Tx)}
$$

</center>
  
극단적으로는 BNNs(Binarized Neural Networks)와 같이 activation을 -1과 1만 나오게 해버리는 신경망 구조도 존재한다. 
정확도가 엄청 떨어질 것 같지만 생각보다는 덜 떨어지는 한편, 메모리 및 속도는 크게 향상시킨 구조다.  
  
  
![kind_of_quantization](/img/posts/39-4.png){: width="90%" height="90%"}{: .center}    
Quantization 기법도 여러 종류가 존재한다. 어떤 것을 양자화할 것인지(weight, activaiton), 어떻게 할 건지(Dynamic, Static), 얼마나 할건지(mixed-precise, 16bit, 8bit, ...),
언제 할 건지(Post-training, Quantization-aware training) 등으로 나눠볼 수 있다.  
  
![kind_of_quantization2](/img/posts/39-5.png){: width="90%" height="90%"}{: .center}    
**Dynamic quantization**은 weight은 미리 quantize해두고 inference time때 activation을 dynamic하게 quantize하는 기법이다. 
inference때만 양자화되기 때문에, 결국 activation 부분에 의한 메모리 절약은 기대할 수 없다.  
  
동적 양자화 기법은 보통 모델의 inference time 대부분을 메모리 로딩 시간이 잡아먹는 경우 많이 활용한다. 
실제로 LSTM이나 Transformer 등의 모델은 matrix multiplication시간보다 저장해둔 weight 값들을 불러오는 데에 시간을 많이 소모한다.   
  
**Static quantization**은 inference 때(즉, training 이후) weight과 activation을 모두 양자화하는 기법이다.
post-training(PTQ)과 static quantization은 사실상 같은 의미를 지닌다.   
  
**Quantization-aware training**은 학습 도중 **fake node를 두어** 추후 quantize 되었을 때의 영향을 미리 simulation(modeling)한다.
즉, training 단계의 precision과 추후 quantization된 모델의 inference에서의 precision 차이를 최소화하도록 학습된다. 
대략적인 방법을 보자면 (1) training path를 inference path와 최대한 비슷하게 만들어 사용하며 (2) training의 loss와 inference의 loss가 비슷해지도록 학습한다.  
  
![quantization_aware_training](/img/posts/39-6.png){: width="90%" height="90%"}{: .center}    
위와 같이 fake-quant node를 두어 simultation을 돌려볼 수 있다. 자세한 방법은 reference에 논문을 달아놨으니 참조하도록 하자. 
정확도 하락폭은 저 이 기법이 보통 제일 작은 편이다.  
  
Quantization을 수행할 때는 **사용하는 Hardware가 해당 연산을 지원하는지**도 살펴보아야 한다. float16, int8 연산이 당연히 어디서든 된다고 생각할 수 있지만 지원하지 않는 하드웨어가 존재할 수 있다.    
  
추가적으로, quantization도 fix된 bit을 계속 활용하는 것보다 레이어별로 다른 bit를 두고 quantization(**flexible-bit quantization**)을 하는 것이 보편적으로 성능이 더 좋았다는 연구결과도 있다.  
  
<br />

## Knowledge Distilation  
사실 knowledge distilation에 관해서는 이전에 이미 여러번 다룬 적이 있다. 
어떻게 보면 그만큼 경량화 기법 중에서는 가장 접근성이 좋다고도 이해할 수 있을 것 같다. 
   
![distilation](/img/posts/39-7.png){: width="90%" height="90%"}{: .center}   
지금은 좀 익숙한 그림이다. 가장 기본적인 형태의 distilation으로, 위쪽에서는 soft prediction으로, 아래쪽에서는 hard prediction으로 각각 loss를 만들어 학습한다.  
  
위쪽에서는 soft prediction과 soft label에 대한 **KL Divergence**, 아래 쪽에서는 hard prediction과 hard label에 대한 **softmax**를 적용한다. 
물론 soft label은 teacher model이 만들어낸 라벨을 뜻한다.   
  
이번에는 한발 더 나아가 loss function을 제대로 설계해보고 다시 한 번 살펴보도록 해보자.  
  
<center>

$$
\mathcal{L}(x;W) = \alpha * \mathcal{H}(y, \sigma(z_s; T=1)) + \beta * \mathcal{H}(\sigma(z_t; T=\tau), \sigma(z_s, T=\tau))
$$

</center>
  
$x$는 input, $W$는 student model의 parameter을 의미한다. $\mathcal{H}$는 cross-entropy이고 $\sigma$는 softmax function으로, $\mathcal{H}$의 인자로 두 softmax가 들어간 경우는 **KL Divergence**를 의미한다. 그 외 $\alpha$, $\beta$는 상수 값으로 보통 $\beta = 1 - \alpha$(혹은 그 반대)이다. 마지막으로 $z\_t$, $z\_t$는 각각 teacher과 studnet의 logit을 의미한다. $T$는 softmax에 들어가는 temperature 값이다.  
   
여기까지 보면 이전과 별반 다를 바가 없지만, 실제 구현에서는 우변의 두 번째 항에 $T^2$도 곱해준다. 즉, 실제 loss의 구현은 아래와 같다.   
  
``` python
#distilation_loss.py
import torch.nn as nn
import torch.nn.functional as F

# hyperparameter
temperature = 20
alpha = 0.2

KLDiv = nn.KLDivLoss(reduction='batchmean')

distilation_loss = KLDiv(F.log_softmax(outputs/temperature, dim=1), 
                    F.softmax(teacher_outputs/temperature, dim=1)) * \
                        (self.alpha * self.temperature * self.temperature)

cross_entroy_loss = F.cross_entropy(outputs, labels) * (1. - self.alpha)

total_loss = distilation_loss + cross_entropy_loss
```

이는 distilation을 제안한 원 논문(Distilling the Knowledge in a Neural Network, reference 참조)에서 제안한 식으로, hard loss와 soft loss를 둘 다 사용하려면 
hard loss 쪽에 가중치를 적게 주어야 한다는 이유에서 기인한다. 
그런데 가중치를 적게 주는 대신, 다른 쪽에 가중치를 많이 주는 방법을 택했다.  
  
한편 애초에 softmax 자체가 $1/T^2$만큼 scale이 되었기 때문에 이걸 다시 곱해줘야 의도한 결과가 나올 것이라는 설명도 타당하다.  
  
여담으로, 위 코드와 같이 <code>nn.KLDivLoss</code> 클래스의 **기대 input은 student에 대한 log_softmax와 teacher에 대한 softmax이니 실제 구현할 때도 주의해야 할 것 같다.**
  
한편, 이번 기회에 temperature $T$에 따른 softmax 함수의 변화는 얼마나 dramatic하게 변하는지 살펴보도록 하자.  
  
![softmax_with_temperature](/img/posts/39-8.png){: width="90%" height="90%"}{: .center}   
MNIST dataset에서 학습된 모델이 정답 클래스를 7으로 예측할 때의 softmax 값을 temperature의 변화에 따라 나타낸 것이다.
위에서 보이듯이 $T$ 값이 50즈음 되면 사실상 정답 클래스와 주변 다른 클래스들간의 확률 차이가 거의 보이지도 않는다.  
  
실제 적용에서는 당연히 50정도 되는 극단적인 값을 사용하지 않는다. 이것을 어느정도로 설정해야할지도 연구에 따라 중요하게 고려해봐야할 부분인 것 같다.
   
<br />

## Logit, Sigmoid, Softmax
logit은 log odds의 약자이다. class가 $C\_1$, $C\_2$ 2개이고 $X$가 주어졌을 때 $X$가 $C\_1$일 확률 $P(C\_1 \vert X)$를 $y$라고 하자.
이 때 odds 값은 다음과 같다.  
  
<center>

$$
odds = \frac{y}{1-y} = \frac{P(C_1 \vert X)}{1 - P(C_1 \vert X)}
$$

</center>

즉, odds가 1보다 크면 $X$를 $C\_1$, 1보다 작으면 $X$를 $C\_2$로 예측하게 된다. 
여기에 log를 씌우면 앞에서 말했던 logit이 된다.   

**logit은 어떤 실수값이든 가질 수 있다.** 즉, $-\infty < logits < +\infty$이다. 
그래서 이는 **확률값을 실수로 mapping하는 함수**라고도 볼 수 있다.  
  
그런데 여기까지 놓고 보니 **실수를 확률값으로 mapping하는 함수**도 있지 않은가?
바로 sigmoid와 softmax가 그것이다. 특히, 지금과 같이 binomial classification에서는 sigmoid가 그 역할을 톡톡히 해낼 수 있다. 
그리고 실제로 logit와 sigmoid는 서로 역함수 관계이다.  
  
<center>

$$
z = \log (\frac{y}{1-y}), \; e^z = \frac{y}{1-y}
$$
$$
\begin{aligned}
y &= \frac{e^z}{1+e^z} \\
&= \frac{1}{1 + e^{-z}}
\end{aligned}
$$

</center>
  
logit과 sigmoid의 관계를 놓고 보니 softmax는 2개 이상의 클래스, 즉 multinomial classification을 위해 sigmoid가 응용된 형태가 아닐까라는 생각을 해볼 수 있다.  
  
본래 odds의 값은 $\frac{P(C\_1 \vert X)}{P(C\_2 \vert X)} = e^z$이다. 
클래스 개수를 2개에서 $K$개로 확장했을 때 $K$번째 클래스에 대한 $i$번째 클래스의 odds를 $\frac{P(C\_i \vert X)}{P(C\_K \vert X)} = e^{z\_i}$와 같이 나타낼 수 있다.   
  
이제 여기서 양변을 $i=1$부터 $K-1$까지 더해보면 아래와 같다.  
  
<center>

$$
\sum ^{K-1} _{i=1} \frac{P(C_i \vert X)}{P(C_K \vert X)} = \sum ^{K-1} _{i=1} e^{z_i}
$$
$$
\frac{P(C_1 \vert X) + P(C_2 \vert X) + \cdots + P(C_{K-1} \vert X)}{P(C_K \vert X)} = \sum ^{K-1} _{i=1} e^{z_i}
$$

</center>
  
확률의 총 합은 1이므로,  
  
<center>

$$
\frac{1 -P(C_K \vert X)}{P(C_K \vert X)} = \sum ^{K-1} _{i=1} e^{z_i}
$$

</center>
  
위 식을 $P(C\_K \vert X)$에 대한 식으로 정리하면,  
  
<center>

$$
P(C_K \vert X) = \frac{1}{1 + \sum ^{K-1} _ {i=1} e^{z_i}}
$$

</center>
  
$\frac{P(C\_i \vert X)}{P(C\_K \vert X)} = e^{z\_i}$을 좌변의 분자에 대하여 정리하고 이를 대입하면,  
  
<center>

$$
P(C_i \vert X) = e^{z_i} P(C_K \vert X)
$$
$$
P(C_i \vert X) = \frac{e^{z_i}}{1 + \sum ^{K-1} _ {i=1} e^{z_i}}
$$

</center>

그런데 $\frac{P(C\_i \vert X)}{P(C\_K \vert X)} = e^{z\_i}$에서 $i=K$이면 $e^{z\_K} = 1$이다. 따라서

<center>

$$
\begin{aligned}
P(C_i \vert X) 
&= \frac{e^{z_i}}{1 + \sum ^{K-1} _ {i=1} e^{z_i}} \\ 
&= \frac{e^{z_i}}{\sum ^{K} _ {i=1} e^{z_i}}
\end{aligned}
$$

</center>
  
우리가 익히 알던 softmax 함수가 도출되었다.  

<br />
  
## Reference   
[Introduction to Deep Learning : Downsizing Neural Networks by Quantization](https://www.youtube.com/watch?v=DDelqfkYCuo&ab_channel=NeuralNetworkConsole)  
[Quantization in Deep Learning](https://medium.com/@joel_34050/quantization-in-deep-learning-478417eab72b)  
[Benoit, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)    
[Hinton, et al. "Distilling the Knowledge in a Neural Network"](https://arxiv.org/pdf/1503.02531.pdf)  
[Github - kmsravindra / ML-AI-experiments](https://github.com/kmsravindra/ML-AI-experiments/blob/master/AI/knowledge_distillation/Knowledge%20distillation.ipynb)  
[https://bit.ly/3vwD9jw](logit, sigmoid, softmax의 관계)  