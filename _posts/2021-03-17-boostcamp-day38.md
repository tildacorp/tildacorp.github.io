---
layout: post
title: "Day38. Acceleration, Pruning"
subtitle: "가속화, 가지치기"
date: 2021-03-17 18:13:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 오늘은 가속화와 pruning에 관한 내용을 배웠다. 그 외에 가속화와 어찌보면 관련이 있는 DL compiler에 대한 내용도 적어보았다.  
      
이 글은 아래와 같은 내용으로 구성된다.  
- [Acceleration](#acceleration)
- [Deeplearning compiler](#deeplearning-compiler)
- [Pruning](#pruning)
    - [Lottery Ticket Hyphothesis](#lottery-ticket-hyphothesis)
- [Other](#other)
- [Reference](#reference)
  
<br />
  
## Acceleration
**병렬처리**란 수행해야하는 하나의 task를 여러개로 분리하여 그들을 한번에 처리하는 것을 뜻한다. 
GPU를 사용하는 이유, 병렬 처리가 가능한 라이브러리를 사용하는 이유, 모두 **가속화**와 밀접한 관련이 있다.  
  
일례로 흔히 행렬 연산에 사용하던 numpy 라이브러리도 행렬 연산을 위한 병렬처리에 최적화된 라이브러리이다. 
행렬 연산 외 용도를 위해 multiprocessing, ray 등 다양한 병렬처리를 돕는 라이브러리들도 존재한다. 
   
![GPU](/img/posts/38-1.png){: width="60%" height="60%"}{: .center}  
우선 Hardware acceleration이라는 용어가 있다. 어떤 특정 연산을 CPU보다 빠르게 처리할 수 있는 하드웨어를 활용하여 task를 가속화하는 것을 뜻한다. 
물론 이 연산은 CPU도 처리할 수 있지만 특정 하드웨어를 활용하면 더 효율적으로 처리할 수 있다는 말이다.  
  
대표적인 하드웨어 가속기로 GPU가 존재한다. GPU는 말 그대로 그래픽 작업 즉 행렬연산을 빠르게 처리해주는 하드웨어이다. 
CPU도 물론 그래픽 작업을 할 수 있지만, GPU에 비할 바는 못된다.  

그 이유를 잠깐 짚고 넘어가자면, CPU는 고성능의 소수 코어로 이루어져있는 반면 GPU는 저성능의 다수 코어로 이루어져있기 때문이다.
행렬 연산 자체가 해보면 알겠지만 그리 복잡한 계산이 필요하지는 않다. **다만 그 계산이 너무 많은 것이 문제이다.** 
GPU는 코어 하나하나의 성능은 우수하지 않더라도 그 물량이 엄청나기 때문에 동시에 간단한 계산을 수없이 많이 할 수 있어 이를 손쉽게 해결한다.  
  
GPU와 비슷한 작업을 하지만 여러 task-specific한 processing unit들이 존재한다. TPU, IPU, VPU 등이 그 예이다.
그 외에도 FPGA라는 특이한 형태의 칩도 존재한다. FPGA는 설계할 때 task가 미리 정해져있지 않지만 사용자가 task-specific하게 **칩을 조립하여** 활용할 수 있다.  
  
마지막으로 병렬처리를 위해서는 **병렬처리를 할 수 있는 코드를 짜야한다.**  
   
![parallel_processing](/img/posts/38-3.png){: width="90%" height="90%"}{: .center}   
  
위 그림을 보면 알 수 있듯이 병렬처리를 할 수 없는 코드를 짜면 컴파일러가 이를 최적화해주거나 하지 않는다. 
병렬처리에 최적화된 코딩 또한 병렬처리에 있어 중요한 대목이다.  
  
한편 가속화라는게 꼭 병렬처리로만 이루어지는 것은 아니다. numpy는 C언어로 짜여져있는데 C언어는 컴파일 언어로 인터프리터 언어인 Python보다 빠르게 동작할 수 있다. 
이와 같이 가속화를 위해 언어를 바꾸어볼 수도 있다. 그 외에도 코딩시 cache memory를 효율적으로 활용하기 위해 temporal locality, spatial locality 등을 고려해볼 수 있다. 
(물론 최근 컴파일러들은 이러한 작업도 자동화되어있는 경우가 많다)

<br />

## Deeplearning compiler
세상엔 많은 딥러닝 프레임워크가 존재한다. TensorFlow, PyTorch 말고도 많은 프레임워크가 있다. 
그런데 이들이 여러 hardware-dependent한 환경에서 코드를 짤 때 의도되었던 작업을 아무 문제 없이 수행할 수 있을까?  
  
딥러닝 모델을 특정 디바이스에서 효율적으로 동작시키기 위해서는 해당 디바이스에 최적화된 코드가 필요하다.
하지만 이것을 우리가 매번 수작업으로 할 수는 없다. 이러한 작업을 자동으로 지원해주는 도구가 바로 **DL compiler**들이다.  
  
딥러닝 프레임워크는 대개 각 프레임워크마다 고유의 컴파일러가 존재한다. TensorFlow는 XLA, PyTorch는 GLOW를 지원한다. 
이렇게 컴파일된 코드는 다시 LLVM이라는 컴파일러와 비슷한 구조에 들어가 요구하는 hardware에 맞춰진 형태로 변환된다.  
  
![DL_compiler](/img/posts/38-2.png){: width="90%" height="90%"}{: .center}  
LLVM을 활용하여 우리는 위와 같이 곱연산 만큼 필요했던 compiler의 개수를 합연산 정도로 줄일 수 있다. 
  
여담으로, LLVM은 꽤 오래전부터 진행되었던 프로젝트로 그 사용처가 비단 DL compiler에만 국한된 것이 아니라 다른 여러 언어에도 적용되어왔다.

<br />

## Pruning  
사람의 시냅스(뉴런과 뉴런의 연결 지점)는 태어난 직후 가장 적고, 영유아기에 가장 많았다가 이후 **필요 없는 시냅스들이 없어지고** 성인이 되서는 시냅스의 수가 어느정도 안정된 상태로 유지된다.  
  
딥러닝에 쓰이는 여러 기법들이 그러하듯, pruning 기법도 실제 사람의 신경망과 같은 맥락에서 동작한다. 
여러 가중치(parameter)들 중 그 중요도가 낮은 것들을 잘라내어 정확도는 비슷하게 유지하는 한편(혹은 더 빨라지기도.), 속도 및 메모리를 최적화하는 기법이다.  
  
1989년에 처음 그 concept이 제시된 이후, 2015년 **Learning both weights and connections for efficient neural network**라는 논문에서 딥러닝에서의 pruning의 포문을 열었다.  

![Pruning](/img/posts/38-4.png){: width="90%" height="90%"}{: .center}  
그 형태는 앞서 설명했듯이 중요도가 낮은 뉴런/시냅스들을 죽이는 방식으로 이루어진다. 
드롭아웃과 어떻게 보면 비슷하지만, 차이점을 보자면 (1) 드롭아웃은 매번 없앤 뉴런을 다시 복원시키지만 pruning은 그렇지 않으며 (2) pruning은 inference도 뉴런을 없앤 상태로 진행하지만 dropout은 역시 그렇지 않다. 
보통 이렇게 해서 없애면 몇 개의 뉴런을 없애면 parameter수를 급격히 떨어뜨릴 수 있다. 그런데 없어진 parameter의 분포를 보면 그 형태가 꽤 주목할 만하다.  
  
![Pruning_2](/img/posts/38-5.png){: width="90%" height="90%"}{: .center}  
위와 같이 pruning을 적용하면 대부분 양상이 비슷한데 **0 주변의 파라미터들은 급격히 감소하게 된다(감소시킨다).** 
즉, 앞서 말한 중요도를 측정할 때 0 주변의 파라미터를 우선적으로 없애는 것이 정확도를 살리면서 parameter 수를 감소시키는 데에 큰 도움이 될 것이라는 점을 알 수 있다. 
그리고 그냥 직관적으로 생각해봐도 값이 0 주변인 파라미터는 모델의 결정에 큰 영향을 안주기 때문에 이를 제거해도 큰 문제가 없을 것이다. 
그래서 보통은 **magnitude(absolute value of parameters)를 파라미터의 중요도를 측정하는 데에 많이 활용**한다.   
  
Pruning 기법도 지금까지 여러가지가 제시되어왔다. 앞서 말한 2015년에 제시된 가장 기초적인 형태는 아래와 같다.   
  
![Pruning_algorithm](/img/posts/38-6.png){: width="60%" height="60%"}{: .center}  
(1) original 네트워크를 학습시키고 (2) 중요도에 의거하여 마스크를 씌운 모델을 만들어 fine-tuning을 한다. (3) 그리고 (1)과 (2)를 반복한다. 
여담으로 원 논문에서는 여기에 L1 norm/L2 norm을 적용했을 때 더 성능이 좋았다고 한다.  
  
아무튼 지금에 와서도 많은 pruning 기법들이 결국 위 방법에서 조금의 변형을 거쳐서 개발되고 있다. 
다만 pruning이라는 기법의 특성상 당연히 많이 가지치기를 할 수록 정확도는 감소하는 **tradeoff의 경향성**이 있어 정확도 감소는 최소화하되 parameter는 줄이는 것이 여러 연구의 목표가 될 것이다.  
  
pruning은 앞서 말했듯이 일일이 보자면 이미 많은 연구가 나와있다.   
![Pruning_kind](/img/posts/38-7.png){: width="90%" height="90%"}{: .center}  
여러 맥락이 존재하지만 위와 같이 어떻게 할건지(structured, unstructured), 중요도를 어떻게 측정할건지(magnitude, ..), 얼마나 자주 할건지(one shot, iterative), 언제 할건지(before/during/after training)에 따라 방법들을 나눠볼 수 있다.  
    
![structure_unstructure](/img/posts/38-8.png){: width="90%" height="90%"}{: .center}   
structured pruning과 unstructured pruning의 차이를 살펴보면, **unstructured**는 시냅스 중 아무거나 잘라내는 것이고, **structured**는 기준 및 규격을 가지고(커널 단위, 레이어 단위 등) 잘라내는 것이다. 위 그림을 보면 unstructured는 시냅스 몇 개가 잘려나간 반면, structured는 뉴런 단위로 통째로 잘려나간 것을 볼 수 있다.  
   
![iterative](/img/posts/38-9.png){: width="90%" height="90%"}{: .center}   
**iterative pruning**의 의미는 위 그림에서 찾을 수 있다. pruning을 할 때 한 번에 가지를 많이 잘라내버리면(one shot) 성능이 확 감소하고 이를 다시 회복시키기가 어렵다.
적은 폭의 성능을 다시 recover하는게 쉬우므로 iterative하게 가지를 조금씩 없애나가는 방법이 많이 택해지고 있다.  
   
![iterative_2](/img/posts/38-10.png){: width="100%" height="100%"}{: .center}   
iterative한 방법도 여러 전략을 세울 수 있다. 매 iteration마다 pruning된 network의 각 parameter를 다시 reset하여 retraining할 건지, 아니면 매 학습마다 발생한 parameter를 그대로 가져가서 fine-tuning을 할 건지, 그 외에도 여러 방법들이 있다. 다만 당장 논문에서는 strategy 1이 더 높은 정확도 및 빠른 학습 속도를 보였다고 제시되어있다.  
  
<br />

#### Lottery Ticket Hyphothesis
또다른 Pruning 기법 중 하나로 **lottery ticket hyphothesis**로 제시된 방법을 이용해볼 수 있다.   
  
![lottery_ticket_hyphothesis](/img/posts/38-11.png){: width="80%" height="80%"}{: .center}   
여기서는 **학습을 통해 얻은 pruning된 모델(즉, subgraph)에 initial weight을 그대로 적용하고 다시 학습시키면 성능이 원래의 것과 비슷하다는 의견을 내놓는다.**  
> A randomly-initialized, dense neural network contains a subnetwork that is initialized such that --when trained in isolation-- it can match the test accuracy of the original network after training for at most the same number of iterations.   
  
여기서 제시한 가설을 정리하면, **기존보다 학습에 필요한 iteration이 적고 test accuracy는 높으며 모델의 parameter 수는 적은 subnetwork 모델이 반드시 존재한다라는 것**이다.  
  
여기서는 iterative pruning 방법을 이용하되, pruning이 끝난 subnetwork의 모든 parameter를 initial parameter로 다시 되돌리는 ($\theta \_0 = \theta$) 작업을 수행한다. 
다만 그 방법이 비교적 비효율적일 수 있어서 아래와 같이 아예 처음 상태로 되돌리기보다는 그 **중간 단계의 상태($k$-th), 즉 어느정도 학습이 된 parameter로 초기화(weight rewinding)해주고 다시 iterating을 도는 방법이 고안된다.**    
  
![lottery_ticket_hyphothesis2](/img/posts/38-12.png){: width="90%" height="90%"}{: .center}    
  
알고리즘을 써보면 아래와 같다. 
![IMP_with_rewinding](/img/posts/38-13.png){: width="70%" height="70%"}{: .center}    
rewinding을 적용한 IMP(iterative magnitude pruning with rewinding)을 적용함으로써 
어느정도 pruning된 subgraph는 다시 처음 parameter로 돌아가 iterative하게 pruning된다. 
이 방법은 앞의 naive한 방법보다 학습 횟수는 줄이면서 정확도는 더욱 향상시킬 수 있었다.  
   
<br />

## Other
pruning 관련 기법들은 당장 한글로 된 자료들을 구하기가 좀 어려운 감이 있다. 
논문에도 자세히 설명이 되어있으니 논문을 위주로 학습하도록 하자. 
특히 reference에 달아둔 **What is the state of neural network pruning?** 논문에서 신경망 구조에서의 pruning 기법이 발전해온 큰 맥락을 읽을 수 있으니 이 논문은 추후 다시 읽어보면 좋을 것 같다.  

<br />
  
## Reference   
[Concept of Deep Learning for Autonomous Driving (2)](https://bit.ly/3lmzXT3)  
[The Lottery Ticket Hypothesis](https://seing.tistory.com/47)  
[Blalock, Davis, et al. "What is the state of neural network pruning?" (2020)](https://arxiv.org/pdf/1803.03635.pdf)  
[Frankle, Jonathan, et al. "Linear Mode Connectivity and the Lottery Ticket Hypothesis" (2020)](https://arxiv.org/pdf/1912.05671.pdf)  