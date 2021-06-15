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

RNN에서는 encoder가 input sequence를 순서대로 처리한 후, encoding된 input 정보가 두 가지 형태로 decoder에 전달됩니다. 하나는 decoder의 initial (hidden) state인 $s_0$이고, 다른 하나는 context vector $c$ 입니다 (주로 $c = h_T$). 이 상태를 그림으로 나타내면 다음과 같습니다:<br />
<!--<p>Encoder: $h_t = f_w(x_t, h_{t-1})$</p>-->
![Fig2](https://tildacorp.github.io/img/seq2seq_rnn_step1.PNG "Seq-to-seq with RNN (after encoding)"){: width="70%"}{: .aligncenter}

Decoder는 $s_0$과 context vector $c$로 전달된 encoder의 정보에 decoder input을 조합하여 decoder의 next hidden state를 만들고, 이로부터 번역 sequence의 첫 단어를 생성하게 됩니다. Decoder의 input은 정답을 밀어넣는 teacher forcing 방식을 많이 사용하지만, 이번 예에서는 $current\ decoder\ input = previous\ decoder\ output$ 방식을 택하였습니다 ($s_t = g_u(y_{t-1}, h_{t-1}, c)$). Decoder의 첫 input은 decoding 시작을 알리는 $start\ token$을 넣습니다. 여기까지를 그림으로 나타내면 다음과 같습니다:<br />

![Fig3](https://tildacorp.github.io/img/seq2seq_rnn_step2.PNG "Seq-to-seq with RNN (after decoding the first word)"){: width="90%"}{: .aligncenter}

이제 이런 방식으로 번역이 완료될 때까지의 그림은 아래와 같습니다. 번역의 마지막 word는 $stop\ token$이 됩니다:<br />

![Fig4](https://tildacorp.github.io/img/seq2seq_rnn_step3.PNG "Seq-to-seq with RNN (translation completion)"){: width="100%"}{: .aligncenter}

위의 그림에서 encoded sequence의 정보를 decoder 쪽으로 전달하는 메인 창구는 context vector $c$ 입니다. Initial decoder hidden state $s_0$도 있긴 하지만, 이건 decoder의 첫 번째 step에만 직접적으로 관계하기 때문에 '이제 encoding이 끝났다' 정도의 정보를 전달한다고볼 수 있겠네요. 하나의 벡터로 decoder가 번역을 하기 위한 모든 encoded sequence 정보를 전달하는 것은 sequence가 짧을 때는 문제가 되지 않을 수 있지만, 문단 전체 혹은 책 전체를 번역해야 하는 task의 경우에는 $c$가 encoder와 decoder 사이에서 정보의 병목현상을 일으킬 수 있습니다. 책 한 권의 정보를 하나의 vector로 표현하기 쉽지 않겠죠. 그래서 'decoder의 매 time step마다 별도의 context vector를 사용하자'는 방법이 고안되었습니다. Context vector $c$가 없다는 것만 빼면 시작은 동일합니다:<br />

![Fig5](https://tildacorp.github.io/img/seq2seq_rnn_attention_step1.PNG "Seq-to-seq with RNN and Attention (after encoding)"){: width="70%"}{: .aligncenter}


이제부터 다른데요, encoded sequence와 initial decoder state를 이용하여 $alignment\ score$라는 scalar 값들을 구합니다. $Alignment\ score$는 각 encoder input마다 하나씩을 구하며, 이는 current decoder state에서 어떤 encoder input에 얼마만큼씩의 '가중치'를 두어야 하는지를 의미합니다. 예를 들면 "I"는 "나는"에, "bread"는 "빵을"에 가장 높은 가중치를 두어야 할 것이고, "was"는 "나는"과 "먹고"에 약간씩을, "있었다"에 가장 높은 가중치를 두어야 할 것입니다.<br />

![Fig6](https://tildacorp.github.io/img/seq2seq_rnn_attention_step2.PNG "Seq-to-seq with RNN and Attention (alignment scores))"){: width="70%"}{: .aligncenter}


이제 계산된 alignment score에 softmax를 붙여 '가중치의 확률분포'를 구한 후, 이것과 encoded sequence의 linear combination (weighted sum)을 계산하여 context vector $c_1$을 만들어냅니다. Context vector $c_1$의 의미는 'decoder의 state가 $s_0$일 때 encoded input sequence에 적절한 가중치를 매겨 정보를 context vector $c_1$으로 가져오면, 이후 decoder input $y_0$이 들어왔을 때 가장 적절한 $s_1$과 $y_1$을 만들어 낼 수 있다'는 정도라고 말할 수 있습니다. 글보다는 그림이 이해하기 쉽습니다:<br />
<p><b>Initial context vector 계산</b></p>

![Fig7](https://tildacorp.github.io/img/seq2seq_rnn_attention_step3.PNG "Seq-to-seq with RNN and Attention (context vector))"){: width="80%"}{: .aligncenter}

<p><b>Initial decoder output 계산</b></p>

![Fig8](https://tildacorp.github.io/img/seq2seq_rnn_attention_step4.PNG "Seq-to-seq with RNN and Attention (first decoder output))"){: width="90%"}{: .aligncenter}

<p><b>번역 완료</b></p>

![Fig9](https://tildacorp.github.io/img/seq2seq_rnn_attention_step5.PNG "Seq-to-seq with RNN and Attention (translation completion))"){: width="100%"}{: .aligncenter}

앞서 설명한대로 decoder의 time step마다 context vector를 별도로 계산해서 사용한 것을 보실 수 있습니다. Encoded input sequence의 정보가 decoder까지 오는 데 bottleneck이 없어진 셈이지요. 이러한 decoder state-dependent한 context vector는 각 decoder 단계에서 어떤 input에 focus를 두어야 하는지를 알려주어서 매우 긴 input sequence의 번역도 가능해졌습니다. 이렇게 입력에 가중치를 두는 것이 서두에 언급하였던 attentional mechanism과 유사하여, 이를 딥러닝에서도 attentional mechanism이라고 부릅니다.<br />

Seq-to-seq RNN에 attention mechanism을 구현하면서 추가된 softmax나 weighted sum 등은 모두 미분 가능한 연산들이어서, 기존의 학습 방식에 크게 지장을 주지 않고 end-to-end로 backpropagation을 통한 동일한 방식의 네트워크 학습만으로 attention weight까지 학습할 수 있습니다. 즉, decoder state마다 어떤 input part에 focus를 두어야 하는지를 end-to-end 방식으로 네트워크가 스스로 학습한다는 말입니다.<br />

한 가지 흥미로운 점은, attention을 사용한 decoder는 encoded input ($h_i$)이 ordered sequence라는 정보를 전혀 사용하지 않고 있다는 것입니다. 매번 $c_t$를 구할 때마다 $s_{t-1}$을 모든 $h_i$와 weighted sum을 구하였을 뿐입니다. Input이 ordered sequence일 필요가 없다는 점은 attention mechanism을 image input에 사용할 수 있게 해서, 이미지를 넣으면 caption을 출력하는 image-to-sequence와 같은 응용이 가능해졌습니다. 설명은 동일하기 때문에 상세한 그림들로 대신합니다:<br />

<p><b>Image encoded with CNN</b></p>

![Fig9](https://tildacorp.github.io/img/img2sec_attention_step1.PNG "Img-2-seq with CNN and Attention (after encoding))"){: width="70%"}{: .aligncenter}

<p><b>Initial decoder output 계산</b></p>

![Fig10](https://tildacorp.github.io/img/img2sec_attention_step2.PNG "Img-2-seq with CNN and Attention (first decoder output))"){: width="80%"}{: .aligncenter}

<p><b>Second decoder output 계산</b></p>

![Fig11](https://tildacorp.github.io/img/img2sec_attention_step3.PNG "Img-2-seq with CNN and Attention (second decoder output))"){: width="90%"}{: .aligncenter}

<p><b>Image caption 생성 완료</b></p>

![Fig12](https://tildacorp.github.io/img/img2sec_attention_step4.PNG "Img-2-seq with CNN and Attention (caption translation completion))"){: width="100%"}{: .aligncenter}

Image caption generation의 논문 결과를 몇 개 보여드립니다:<br />
![Fig12](https://tildacorp.github.io/img/image_caption_generation.PNG "Image Caption Generation Results"){: width="70%"}{: .aligncenter}
