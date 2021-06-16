---
layout: post
background: '/img/backgrounds/transformers.jpg'
title:  "Attention and Transformer - part2"
date:   2021-06-15 23:59:59
categories: Transformer
tags: attention_mechanism, transformers
excerpt: 어텐션과 트랜스포머
use_math: true
---

지난 [포스트](https://tildacorp.github.io/2021/06/14/transformers/)에 attention mechanism이 다양한 task에 사용될 수 있다는 것을 보았습니다.
많은 연구자들의 관심이 쏠리자 결국 attention mechanism을 보다 더 다양한 task에 쉽게 사용하기 위한 generalization이 이루어지게 되고, 이렇게 일반화된 attention structure는 후에 다양한 network에 attention layer로 쉽게 넣어서 사용할 수 있게 됩니다. 이를 위해 지금까지 사용했던 notation을 재정의해보겠습니다. Attention layer는 말 그대로 '어느 input에 얼만큼씩 attend 해야 하는지'를 말해주는 layer이기 때문에, 지난 포스트에서의 seq-to-seq w/ attention 모델로 보면 encoder 부분만 구현한 것이라고 보시면 될 것 같습니다:


<h4>Attention Layer</h4>
<p>Inputs:</p>
<ul>
  <li> Query vector: $q$
    <ul><li>shape: $D_Q$ (dimensionality), 지난 설명에서는 decoder hidden vector $S_i$</li></ul>
  </li>
  <li>Input vectors: $X$
    <ul><li>shape: $N_X\times D_Q$, 지난 설명에서의 encoder hidden vector $H$</li></ul>
  </li>
  <li>Similarity function:
    <ul><li>$f_{att}$</li></ul>
  </li>
</ul>

<p>Computation:</p>
<ul>
  <li> Similarities: $e$
    <ul>
      <li>shape: $N_X$</li>
      <li>$e_i = f_{att}(q, X_i)$</li>
    </ul>
</li>
  <li>Attention weights: $a = softmax(e)$
    <ul><li>shape: $N_X$</li></ul>
  </li>
  <li>Output vector: $y = \sum_{i}a_i X_i$
    <ul><li>shape: $D_X$</li></ul>
  </li>
</ul>

가장 먼저 similarity function이 generalize 됩니다. 초기 attention mechanism에서는 $f_{att}$를 network layer로 학습하였는데, 이를 학습 없이 훨씬 효율적인 dot product로 계산해도 비슷한 성능을 보인다는 점이 밝혀졌습니다. 이 dot product high dimensional input에 대해 너무 큰 값이 나온다는 단점, dot product 결과의 kurtosis가 너무 높은 경우 대부분의 softmax probability가 0이 되어버려서 gradient 또한 0이 되어버려서 vanishing gradient problem이 발생하여 학습이 매우 어려워진다는 단점이 있기 때문에, 이 dot product를 $sqrt(dim(q)=D_Q)$로 나눈 scaled dot product로 추가 개선됩니다.<br />

![Fig1](https://tildacorp.github.io/img/attention_generalization1.PNG "Generalization of Attention Mechanism"){: width="70%"}{: .aligncenter}


다음 단계의 generalization은 multiple query vector를 사용하는 것입니다. 이전까지는 decoder time step마다 하나의 query vector ($q$, 예전의 $s_i$)를 사용하여 input state ($X$, 예전의 $H$)에 대한 하나씩의 similarity score를 구했는데요, 이제 모든 decoder state에 대한 모든 input vector들의 가중치를 한번에 계산하자는 것입니다. 앞서 similarity function을 scaled dot product로 generalize를 해 두었으니, 이제 similarity score를 구하는 것은 $Q$ (set of $q$)와 $X$의 matrix multiplication (MatMul)으로 간단히 구할 수 있습니다. 이렇게 구해진 similarity score에 한 방향으로 softmax를 적용하면 attention weight를 구할 수 있고, 이것과 input vector와 linear combination을 통해 encoder의 output vector를, 역시 한 번에 구할 수 있습니다.<br />

![Fig2](https://tildacorp.github.io/img/attention_generalization2.PNG "Further Generalization of Attention Mechanism"){: width="80%"}{: .aligncenter}


지금까지 input vector를 보면 두 가지 목적으로 사용된 것을 알 수 있는데요, 하나는 query vector와 비교하여 attention weight를 계산하기 위한 것이고 다른 하나는 이 attention weight와 결합하여 output vector를 만들기 위한 것입니다. 두 기능을 분리시켜 별도의 set of weights를 학습하게 한다면 모델 (layer)의 flexibility가 좋아질 것이라는 아이디어로, $key$와 $value$의 개념이 등장합니다. Similarity (attention weight)를 계산하기 위해 input vector에 key matrix $W_k$를 곱하여 key vector $K$를 구하고, 이걸 이용해서 similarity 계산을 합니다. $Softmax$를 거쳐 normalized probability distribution인 attention weight를 구한 다음, output vector를 계산할 때에는 input vector에 value matrix $W_v$를 굽한 value vector $V$를 구해서 attention weight와 곱해 output을 만들어냅니다.<br />

![Fig3](https://tildacorp.github.io/img/attention_generalization3.PNG "Even Further Generalization of Attention Mechanism"){: width="100%"}{: .aligncenter}


여기까지 결과를 그림으로 보면 다음과 같습니다:

<p><b>Initial state (input vector와 query vector만 있는 상태)</b></p>

![Fig4](https://tildacorp.github.io/img/attention_layer1.PNG "Attention Layer - step 1"){: width="70%"}{: .aligncenter}

<p><b>Key vector를 이용한 similarity 계산</b></p>

![Fig5](https://tildacorp.github.io/img/attention_layer2.PNG "Attention Layer - step 2"){: width="100%"}{: .aligncenter}

<p><b>Value vector를 이용한 output 계산</b></p>

![Fig6](https://tildacorp.github.io/img/attention_layer3.PNG "Attention Layer - step 3"){: width="100%"}{: .aligncenter}



