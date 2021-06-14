---
layout: post
title: "Day40. Low-rank approximation"
subtitle: "행렬 분해, 텐서 분해"
date: 2021-03-19 11:57:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 이전에 배웠던 depthwise separable convolution과 같은 기술은 하나의 연산을 여러 개로 쪼갬으로써 연산량이나 parameter를 줄이는 기술이다. 오늘은 이런 기술의 기반이 되는 행렬 분해/텐서 분해에 대하여 공부하였다.   
      
이 글은 아래와 같은 내용으로 구성된다.  
- [Kernel method, Low-rank approximation](#kernel-method-low-rank-approximation)
- [Matrix decomposition/Tensor decomposition](#matrix-decompositiontensor-decomposition)
- [Reference](#reference)
  
<br />
  
## Kernel method, Low-rank approximation
**kernel method**는 저차원 데이터를 어떠한 필요성에 의해 고차원으로 바꿨을 때 필요한 연산에 trick을 적용하여 computational cost를 줄이는 기법이다.  

![kernel_method](/img/posts/40-1.png){: width="90%" height="90%"}{: .center}    
예를 들어 우리가 위와 같이 서로 다른 두 class의 데이터를 분류해야한다고 해보자. 만약 SVM과 같은 모델을 이용하여 boundary를 linear하게 그어야 하는 경우, 좌측 그림과 같은 데이터는 꽤나 난감하다. 물론 다른 방법도 있겠지만, 이 때 우측과 같이 차원을 늘린 상태에서 boundary를 긋는 것이 더 효과적일 수 있다.  
  
하지만 문제는 이렇게 되면 연산량이 꽤 된다는 것이다. 지금은 classification task이므로 아마 데이터들간 similarity를 계산하게 될텐데, 이 때 내적을 계산하게 된다. 
만약 아래와 같은 식으로 차원 확장을 하게된다면 그 내적을 계산하기 위해 필요한 연산량은 $O(n^2)$이다.
  
<center>

$$
\mathrm{x} = (x_1, x_2), \; \mathrm{y} = (y_1, y_2) \\
\longrightarrow \;\;\;f(\mathrm{x}) = (x_1, \; x_2, \; x_1 ^2 + x_2 ^2), \; f(\mathrm{y}) = (y_1, \; y_2, \; y_1 ^2 + y_2 ^2)
$$

<br />

$$
<f(\mathrm{x}), f(\mathrm{y})> = ?
$$

</center>
  
$O(n^2)$이 크게 와닿지는 않을 수 있지만, 점이 한두개가 아닐 뿐더러 보통 차원을 확장한다고 하면 3차원이 아니라 더욱 고차원으로 확장하게 된다. 
모든 점들의 쌍으로 이를 계산한다고 하면 연산량이 꽤 된다.  
  
따라서 kernel method를 도입헤볼 수 있다. 일종의 trick인데 $<f(\mathrm{x}), f(\mathrm{y})>$을 아래와 같이 계산할 수 있다.   
  
<center>

$$
K(\mathrm{x}, \mathrm{y}) = (x_1, x_2) \cdot (y_1, y_2) + \Vert (x_1, x_2) \Vert ^2 \Vert (y_1, y_2) \Vert ^2
$$

</center>
  
값은 결국 같지만 **차원 확장 이전의 값들을 이용한 계산을 하게 되므로** 연산량을 기존에 비해 크게 줄일 수 있다.   
  
kernel method를 갑자기 왜 이야기하였냐면, 우리가 이전에 배웠던 depthwise separable convolution이 생긴 배경도 결국 이것과 같기 때문이다.    
  
![filter decomposition](/img/posts/40-2.png){: width="90%" height="90%"}{: .center}    
depthwise separable convolution은 **filter decomposition**이라는 기법의 일종이다. 위에서는 단순히 벡터 연산을 분해하는 기법이었지만, 이를 응용하면 위와 같이 convolution 연산도 분해하여 그 연산량을 줄일 수 있다. 
  
<br />

## Matrix decomposition/Tensor decomposition
이전 강의에서 UV decomposition을 통해 추천 시스템을 구축하는 방법에 대해 배운 적이 있다. 이와 같이 행렬 분해는 그 자체로 무언가 의미있는 값을 뽑아낼 수 있다. 
SVD(Singular Value Decomposition), PCA(Principal Component Analysis)도 마찬가지이다. 이전에 다뤄봤기 때문에 여기서 자세히 설명을 하지는 않지만,
모두 **데이터를 특징짓는데에 중요하게 관여하는 벡터만을 남기고 없애도 문제 없는 벡터들은 삭제하는데에** 이용한다.    
  
공통적으로 matrix decomposition과 filter decomposition은 **정보를 효과적으로 저장 및 계산하고 혹은 중요한 정보를 골라내는데에 크게 기여할 수 있다.** 
지금까지 살펴본 바, 분해라는게 연산량에 큰 도움을 줄 수 있다는 것을 알게되었는데 특히 이것이 머신러닝에 도움이 되는 이유는 **training-free, 즉 학습이 필요 없기 때문**이다. 
만약 이 분해 작업에 학습이 동반되었다면 오히려 연산량이 늘어날 수도 있었겠지만, 지금까지 본 것들은 수학 기반 연역적 방식이므로 학습이 필요하지 않다.  
  
한편, Matrix decomposition의 반대편에는 **Tensor decomposition**도 존재한다. 
텐서 분해는 행렬 분해보다 당연히 더 복잡하다.  

![tensor_decomposition](/img/posts/40-3.png){: width="90%" height="90%"}{: .center}   
지금은 그 방법에 대해 자세히 다루지는 않겠지만, 위 그림을 보고 대충 어떤걸 하는거구나 정도만 알고 넘어가도록 하자.
텐서 분해에는 크게 두 가지 방법이 있다. 첫 번째는 **CP(Canonical Polyadic) decomposition**으로, rank 1의 vector 여러개로 분해하는 방법을 말한다. 
사실 이건 SVD에 가까운 방법이다. SVD도 $U$, $\Sigma$, $V^T$에서 같은 위치에 대응되는 벡터/스칼라를 각각 뽑아와서 위와 같이 나열할 수 있다. 이를테면,  
  
<center>

$$
A = U\Sigma V^T
$$
$$
= \begin{pmatrix} |  & | & {} & | \\\\
 \vec u_1 & \vec u_2 &\cdots &\vec u_m \\\\
 |  & | & {} &| \end{pmatrix} 
 
\begin{pmatrix} 
\sigma_1 &  &  &  & 0\\\\
 & \sigma_2 &  &  & 0\\\\
 & & \ddots &     & 0\\\\
 & & & \sigma_m   & 0
\end{pmatrix}

\begin{pmatrix}  - & \vec v^T_1 & - \\\\
- & \vec v^T_2 & - \\\\
  &\vdots& \\\\
- & \vec v^T_n & -
\end{pmatrix}
$$
$$
A = \sigma_1 \vec u_1 \vec v_1^T + \sigma_2 \vec u_2 \vec v_2^T +\cdots+ \sigma_m \vec u_m \vec v_m^T
$$

</center>
  
와 같이 말이다. CP decomposition을 나타내는 그림을 보면 형태가 똑같지만 그저 곱해지는 rank 1의 벡터 하나가 각 항에 하나씩 더 붙었을 뿐이다. 
  
두번째 방법인 **Tucker decomposition**은 CP decomposition에서 보다 일반화된 형태이다. 
여기서는 그림에서 보이듯이 가운데에 core tensor를 하나 찾고 각각의 면에 대하여 subspace(행렬)를 찾는다.
여기서 subspace의 rank는 물론 1보다 크지만, 원래의 텐서보다는 당연히 작다.    
  
그런데 텐서 분해가 과연 행렬 분해처럼 딱 떨어질까? 그렇게 되면 좋겠지만 안타깝게도 그렇지 않다. 
위 그림에서 보이듯이 우리는 반복 횟수 R을 정하고 해당 번째까지만 decomposition을 한다. 즉, 원래의 텐서에 근사된 값을 얻어낸다. 
물론 SVD에서 보았듯, 분해가 만약에 딱 떨어졌더라도 어차피 앞쪽의 중요한 성분만을 뽑아내어 사용했을 것이기는 하다.  
  
우리는 아래의 optimization 문제를 풀어야하며, 역시 iterative search 과정이다. 

<center>

$$

\underset{\widehat{X}}{\min}\lVert X - \widehat{X} \rVert \;\; \mathrm{with} \;\;  \widehat{X} = \sum_{r = 1} ^R \lambda_r u_r \circ v_r \circ w_r
$$

</center>
  
$\lVert \cdot \rVert$은 Frobenius norm으로 tensor 구조에서 쓰이는 norm이다.   
    
더 자세히 알아보고 싶지만, 일단 당장 이것에 대해 더 자세히 알아보는 것은 과한 것 같기도 하여 여기까지만 쓰려고 한다.
(사실 다 컴퓨터가 해주기 때문에.. :sweat:) 텐서 분해 관하여 부족한 부분에 대해선 추후 다시 공부하도록 하자.  

<br />
   
## Reference   
[What is the kernel trick? Why is it important?](https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d)  
[특이값 분해(SVD)](https://angeloyeo.github.io/2019/08/01/SVD.html)   
[Understanding the CANDECOMP/PARAFAC Tensor Decomposition, aka CP; with R code](https://www.alexejgossmann.com/tensor_decomposition_CP/)  