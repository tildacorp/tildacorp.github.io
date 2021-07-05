---
layout: post
background: '/img/backgrounds/deepfake2.jpg'
title:  "FaceNet, One-Shot Learning, Triplet Loss"
date:   2020-05-22 22:59:59
categories: Deepfake
tags: facenet one-shot_learning triplet_loss
excerpt: FaceNet의 Triplelet One-Shot Learning
use_math: true
---

> 총 n명의 국제 테러리스트 명단이 공개되었습니다. 공항 검색대에 설치된 CCTV로 N명 (N>>n) 입국자들의 얼굴을 찍어서 테러리스트인지 판정하는 모델을 만드십시오.

이런 태스크가 떨어지면 어떤 모델을 만들까요? 그냥 단순한 classifier를 붙이면 될까요? 안될까요?
Classifier로 만들면 예상되는 문제들은 어떤게 있을까요?

먼저 테러리스트 숫자인 n보다 훨씬 큰 N명의 사람들은 어느 클래스로 보내죠? 클래스 갯수를 n+1로 만들어서 non-terrorist를 하나로 몰아넣는다? 이 클래스에만 데이터가 넘쳐나겠죠? 뭐 그럼에도 불구하고 학습데이터를 실제 비율이랑 비슷하게 만들어서 열심히 학습을 시켜볼 수는 있겠습니다만 문제가 또 발생합니다. 테러리스트 데이터베이스가 업데이트되어서 명단이 추가되었거든요.

이젠 어떻게 해야할까요? FC layer에서 class 숫자를 하나 더 늘리고, 예전 데이터 일부에 추가된 테러리스트 얼굴을 더해서 추가학습을 시켜볼까요? 어느정도는 성능이 나올 수는 있지만 보장이 가능할까요? 데이터베이스가 계속 업데이트가 되면 이 때마다 매번 추가학습을 할 수 있을까요? 학습이 진행되는 동안 테러리스트가 공항에 도착하면 어떻게 하나요?

FaceNet은 이런 문제들을 해결하기 위한 방법론을 제시합니다. 비단 face domain에서 뿐만 아니라, use-case만 맞는다면 다양한 classification-based 솔루션에 적용할 수 있는 솔루션을요. 간단히 말씀드리면 big dataset의 low-dimensional embedding을, 동일한 class를 가지는 data들끼리는 distance가 가깝도록, 다른 class의 data들끼리는 distance가 멀도록 학습해서, 두 data가 같은 클래스인지 아닌지를 이 embedding 안에서의 distance로 구별해내는 generalizable 모델을 만든다는 것입니다. 테러리스트의 예로 이야기하자면, FaceNet은 일반적으로 두 사진의 얼굴이 동일 인물인지 아닌지를 학습하기 때문에 새로운 테러리스트가 추가되더라도 CCTV에 찍힌 인물이 테러리스트 명단에 있는지 없는지는 추가학습 없이 one-shot으로 $n+\alpha$개의 이미지랑만 비교해보면 된다는 것이죠. 그것도 low-dimensional embedding을 사용하기 때문에 빠르게 가능합니다.

Low-dimensional embedding은 일반 DNN classifier도 사용하는데요? 심지어 PCA를 사용한 eigenface도 low-dimensional embedding을 쓰는건데 무슨 차이가 있을까요? 저자는 Triplet Loss라는 개념을 고안해서 이 문제를 해결하였습니다.

![Fig1](https://tildacorp.github.io/img/triplet_loss.PNG "Triplet Loss"){: width="70%"}{: .aligncenter}


Triplet은 위의 그림과 같이 anchor - positive - negative의 dataset으로 구성됩니다. Anchor는 기준이 되는 class의 data이고, positive는 anchor와 동일한 class의 또다른 data instance, negative는 anchor와 다른 class의 data를 말합니다. 이렇게 구성된 dataset을 네트워크에 넣어주면서 low-dimensional embedding 내에서 동일 class의 data instance 사이의 distance가 다른 class의 data instance 사이의 distance보다 가깝도록 학습을 시켜주는 것이지요. 이렇게 하게되면 class가 같은 인물들은 가깝게, class가 다른 얼굴은 멀게끔 표현하는 embedding이 학습될 것입니다. 아래 그림과 같이 학습 전에는 이 embedding이 A-N이 A-P보다 더 가깝다고 나타냈던 것이, 학습 후에는 A-P 거리가 A-N 거리보다 최대 $\alpha$ 미만으로 가까와지게 되는거죠.

![Fig2](https://tildacorp.github.io/img/triplet_loss_learning.PNG "Learning w/ Triplet Loss"){: width="70%"}{: .aligncenter}


Distance(A, P)가 항상 distance(A, N)보다 일정 margin값 이상의 차이를 유지하도록 다음과 같은 수식을 이용합니다.

${\Vert f(x^a_i)-f(x^p_i) \Vert}^2_2+\alpha<{\Vert f(x^a_i)-f(x^n_i) \Vert}^2_2$

결국 모든 triplet의 $$net_{loss}$$는 다음과 같은 식이 될 것이고, 네트워크는 이 값을 최소화하도록 학습하면 됩니다.

$\sum_i^N[{\Vert f(x^a_i)-f(x^p_i) \Vert}^2_2-{\Vert f(x^a_i)-f(x^n_i) \Vert}^2_2+\alpha]$ 


이 Triple Loss를 구하기 위해서는 A, P, N 데이터의 low-dimensional embedding값을 구해야 하는데요, 이를 위해 동일한 구조에 weight까지 공유하는 3개의 동일한 네트워크를 사용하여-이런걸 샴 네트워크(Siamese network)라고 하죠-병렬적으로 빠른 계산이 가능합니다. 그런데 triplet으로 학습을 하는 데에는 두 가지 큰 문제가 생깁니다. 


첫째는 triplet의 갯수가 너무나 많다는 것입니다. 예를 들어 학습 데이터에 10,000명의 face ID가 있고 각 ID마다 30장 씩의 사진이 들어있다고 하면, 조합 가능한 triplet의 경우의 수가 <p>$(30\times10,000)_{anchor}\times29_{positive}\times(30\times9,999)_{negative}$</p>나 될 것입니다 (계산 맞나요 ㅠㅠ, 암튼 엄청 커집니다). 저자는 이 문제를 해결하기 위해, 가능한 모든 조합의 triplet을 만드는 대신 manageable한 숫자의 triplet만을 만들어서 학습하는 방식을 택합니다. 전체 데이터의 mini-batch를 만들어서, 그 안에서만 triplet을 조합하는 방식으로요. 본 논문에서는 ID (class)를 일부만 포함하되, 각 ID마다 40장 씩의 얼굴 사진을 포함하도록 mini-batch를 구성하였습니다. 뒤에 batch size 1,800으로 실험했다고 하니, 한 mini-batch마다 ID가 45명씩 들어가게 구성한 것 같습니다 (총 face data 숫자는 100M~200M).

두 번째 문제는 꼭 triplet 학습에 국한된 것이 아니라 전반적인 네트워크 학습의 효율성에 관한 것입니다. 머신러닝에서 학습을 할 때 초반부터 hard example로 학습을 하면 local minima에 빠질 가능성이 높다(또는 mode collapse가 생긴다)고 알려져 있습니다. 사람의 학습 방법이 그러하듯이, 학습 데이터의 난이도를 쉬운 것부터 점차적으로 어렵게 가져가야 보다 나은 generalizability와 빠른 수렴이 가능하다는 것이 curriculum learning의 개념인데요 (high-quality 얼굴 이미지를 생성한 PGGAN도 이 *start small* 컨셉을 이용한 것이죠), 저자는 triplet의 구성에 이 아이디어를 채용합니다. 그럼 triplet이 쉽고 어렵다는 구분은 어떻게 할 수 있을까요?

어떤 이미지들은 학습 초반부터 이미 $d(A, P)+\alpha < d(A, N)$을 쉽게 만족하는 것들도 있을 것입니다. 인종이 다르다거나 성별이 다르다거나 raw feature 자체에서부터 워낙 큰 차이를 보여서 dimension을 줄이고 나서도 차이가 남아있는 것들도 있겠죠. 이런 것들을 *easy triplet*이라고 할 수 있겠습니다. *Easy triplet*은 loss가 이미 0이어서 학습에 별로 기여할 것이 없습니다. 반대로 $d(A, N) < d(A, P)$와 같이 embedding에서의 distance가 역전되어 있는 것들을 *hard triplet*이라고 합니다. 그리고 이 둘 사이에 *semi-hard triplet*이 있는데요, 이건 $d(A, P) < d(A, N)$은 만족하지만, 아까 얘기했던 margin값 이하의 차이를 보이는, 그러니깐 $d(A, P) < d(A, N) < d(A, P)+\alpha$ 의 값을 가지는 triplet들을 말합니다. 논문에는 나와있지 않지만 [한 블로그](https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352)를 보니 총 triplet의 절반은 *hard*로, 나머지 절반은 random으로 선택해서 학습을 시작했다고 합니다. Curriculum learning을 따랐다고 하니 학습이 진행되면서 *hard*의 portion이 높아지도록 rate 변경을 해주었을 것 같습니다.


그림은 별로 없이 글로 설명이 길었는데요, 이같은 방식으로 FaceNet은 *Labeled Faces in the Wild* 데이터셋에서 99.63%의 정확도, *YouTube Faces DB*에서 95.12%를 기록했다고 하네요. 2015년 구글에서 발표한 꽤 오래된 논문임에도 불구하고 개연성 높은 사고의 전개방식과 놀라운 성능 덕분에 재미있게 읽었습니다. 말씀드렸다시피 face image 뿐만 아니라 다양한 데이터셋을 쓰는 real-world 태스크에서 활용해봄직한 논문이었습니다.

논문링크: [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)