---
layout: post
background: '/img/backgrounds/transformers.jpg'
title:  "Attention and Transformer"
date:   2021-06-13 23:59:59
categories: Transformer
tags: attention_mechanism, transformers
excerpt: 어텐션과 트랜스포머
use_math: true
---
오늘은 transformer와 그 근간이 되는 attention mechanism에 대해 알아보겠습니다.

알고보면 그렇게 어렵지 않은데 알기까지가 어려운 개념들이 자주 있는데요, transformer도 그런 것들 중 하나였습니다.
논문, 튜토리얼, 블로그 등등 너무 많은 reference들이 같은 개념을 각각 다른 focus에서 접근하는 것 같았는데요, 개인적으로는 [Michigan 대학의 온라인 강의](https://youtu.be/YAgjfMR9R_M)가 가장 이해하기 쉽게 설명했던 것 같습니다. 링크 남긴 강의 내용을 기반으로 설명해보겠습니다.<br/>


심리학이나 computational neuroscience에서는 attention mechanism을 '제한된 뇌의 computational power 때문에 task 수행 시 생기는 malfunction' 이라는 측면에서 인간의 뇌(주로 시각 영역)를 이해하는 데 있어서 중요한 현상으로 연구해 오기도 하였습니다. [The Invisible Gorilla](https://youtu.be/vJG698U2Mvo)라는 유명한 테스트에서는 task에 attention을 집중하다보면 커다란 visual input을 처리하지 못하게 되는 것을 보이기도 했고 (제가 스포일을 했으니 여러분들은 고릴라가 보이실겁니다), 이와 비슷하게 selective attention을 이용한 테스트 들은 여럿 있습니다 [change blindness test](https://youtu.be/_bnnmWYI0lM). 아래 그림은 Stroop effect라는걸 보여주는 테스트인데요, 색깔을 나타내는 단어들을 검은색으로 써 놓았을 때 혹은 색-단어가 매칭되도록 써 놓았을 때보다 색-단어의 mismatch가 생기도록 써 놓았을 때 사람들이 이 단어들을 읽는 속도가 더 느려진다는 실험입니다. 이것 역시 제한적인 뇌의 capacity로 인해 색 정보에 attention이 쏠려 text 정보를 처리하는데 방해가 된다는 selective attention으로 설명되기도 합니다 (다른 설명도 있긴 합니다).

![Fig1](https://tildacorp.github.io/img/stroop_test.jpg "Stroop Test"){: width="70%"}{: .aligncenter}


이야기가 좀 샜는데요, 뇌에서 한계를 극복하기 위해 진화된 (visual) attentional mechanism이 딥러닝에서는 비슷하지만 조금 다른 context로 사용되게 됩니다. 이를 위해 attention이 사용되기 이전의 RNN의 sequence-to-sequence 동작 방식을 살펴보겠습니다. Sequence-to-sequence의 가장 이해하기 쉬운 예 중 하나인 번역 task로 설명합니다.
<p>
입력: $x_1, x_2, ..., x_T$의 sequence<br />
출력: $y_1, y_2, ..., y_{T'}$의 sequence
</p>

RNN에서는 encoder가 input sequence를 순서대로 처리한 후, encoding된 input 정보가 두 가지 형태로 decoder에 전달됩니다. 하나는 decoder의 initial (hidden) state인 $s_0$이고, 다른 하나는 context vector $c$ 입니다 (주로 $c = h_T$). 이 상태를 그림으로 나타내면 다음과 같습니다:
<p>
Encoder: $h_t = f_w(x_t, h_{t-1})$
</p>
![Fig2](https://tildacorp.github.io/img/seq2seq_rnn_step1.png "Seq-to-seq with RNN (after encoding)"){: width="70%"}{: .aligncenter}