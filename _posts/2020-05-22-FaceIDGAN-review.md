---
layout: post
background: '/img/backgrounds/deepfake1.jpg'
title:  "FaceID-GAN: Learning a Symmetry Three-Player GAN for Identity-Preserving Face Synthesis"
date:   2020-05-22 10:59:59
categories: Deepfake
tags: faceidgan id-preservation
excerpt: FaceID-GAN 논문 리뷰
mathjax: true
---

지난 포스트에서 [FSGAN](https://jiryang.github.io/2020/05/14/FSGAN-review/)에 대해 살펴보았었습니다.

'Deepfake'에 사용되는 face-swap이 source 인물의 얼굴을 target 영상에 잘라붙이는 방식으로 동작한다는건 많이들 아실텐데요, 간단히 설명하고 넘어가겠습니다. 'Deepfake'에는 1.5개의 AutoEncoder가 필요합니다 (Encoder 부분은 share되고, decoder만 별도로 학습하면 되니깐 1.5개라고...). Single encoder에 a set of source 및 a set of target 얼굴 이미지를 입력하고 차원을 축소하여 latent face를 만든 다음, source/target 별도의 decoder로 입력 이미지를 복원하도록 학습을 시킵니다 (결과물이 동영상인 경우라면 target 이미지들은 동영상을 ffmpeg 같은거로 frame으로 뜯어내어서 얼굴들을 가져오면 될꺼고요, source 쪽도 마찬가지로 영상에서 가져온거든 별개의 이미지들을 가져다넣은 것이든 상관없지만, 가능하면 target에서 나타나는 표정과 얼굴 각도의 variation을 포함하는 super-set을 넣어주면 합성 quality가 좀 더 좋습니다). 

이제 encoder는 source와 target얼굴 모두의 (공통의) latent face를 만들 수 있게 되었고 decoder는 여기서 각각의 원래 얼굴을 복원할 수 있게 되었기 때문에, target decoder를 떼어내고 학습된 encoder - source decoder로만 조합을 해서 target 얼굴 이미지를 입력하면 이 attribute(표정, 각도 등)를 따르는 source의 얼굴을 합성하게 됩니다. 이 합성된 얼굴을 잘라서 원래의 target 이미지에 티 안나게 잘 blending시키면 deepfake가 완성됩니다.

Source랑 target이 좀 헷갈릴 수 있는데, 잘 읽어보시면 이해가 될꺼예요. Face A가 target이고 Face B가 source겠죠?

![Fig1](https://jiryang.github.io/img/faceswap_autoencoder.png "How Deepfake Works"){: width="70%"}{: .aligncenter}

(Deepfake에 대해 궁금하시다면 이런 [reference](http://news.seoulbar.or.kr/news/articleView.html?idxno=1817)도 있습니다 :) )


이 방식은 비교적 간단한 네트워크를 사용하고, 데이터도 그렇게 많이 들어가지 않으며, source-target 간의 1-to-1 학습이기 때문에 학습이 쉽고 합성 quality도 좋은 편입니다. 하지만 source나 target 인물이 바뀔때마다 학습을 새로 해야한다는 점, source의 데이터가 많아야 한다는 점이 이 모델의 약점입니다.

이후 얼굴 합성 연구는 pose-guided 방식의 face reenactment 알고리즘들이 메인스트림인 듯 합니다. 이 방식은 MSCeleb이나 CelebA와 같은 얼굴 빅데이터로 general한 합성 모델을 학습시켜둔 다음, 어떤 source/target 얼굴이 입력되어도 그거에 맞춰서 reenact를 시킵니다. 학습이 오래걸리고 수렴이 어렵다는 단점은 있겠지만, 학습 후에는 합성 결과를 쭉쭉 뽑아낼 수 있다는 장점 때문에 use case에 따라 선호될 수 있겠지요? 게다가 이 모델은 특정 source나 target의 데이터를 많이 필요로 하지 않는다는 점도 큰 장점입니다.*

하지만, 아직까지 pose-guided 방식의 가장 큰 단점 중 하나가 바로 '얼굴이 애매하게 닮는다'는 점입니다. 아무래도 big data로 general하게 만들다 보니 합성 모델의 튜닝값이 개개인의 얼굴에 최적화될 수 없다는 점 때문일텐데요, pose-guided 방식 합성 모델을 상용화하게 되면 이 ID preservation을 얼마냐 잘 하느냐가 성패를 좌우하게 될 것 같습니다.

![Fig2](https://jiryang.github.io/img/fsgan_results.PNG "FSGAN Results"){: width="70%"}{:.aligncenter}

(FSGAN의 샘플 결과물입니다. Target의 attribute를 따라하는 source 얼굴을 합성한 결과가 source 얼굴같아 보이시나요? 약간씩 성능차이가 있긴 하지만 모든 pose-guided 방식 모델들의 결과 이미지를 저렇게 늘여놓고 보면 좀 비슷해 보이긴 하지만, 막상 result만 따로 떼어놓고 보면 과연 이게 source ID 인물인지 아닌지 아리까리할 때가 많습니다. Source와 target의 중간 얼굴같은 느낌도 들고요. 특히 내 얼굴이나 얼굴을 잘 아는 유명인일 경우 별로 안닮은것 처럼 느껴집니다.)

별도 포스팅 없이 배경설명을 하다보니 너무 서론이 길었네요. 오늘 간단히 살펴볼 FaceID-GAN 논문에서는 ID preserving을 위해 classifier를 독특한 방식으로 사용합니다. 특정 class에 속하는 GAN 합성 결과를 유도하기 위해서 discriminator와 classifier를 함께 뒷단에 배치해서 classification 결과에 따라서도 generator를 업데이트하는 방식의 네트워크는 이미 여러 개 있었는데요, 얘들도 역시 애매하게 닮는 한계를 극복하지 못하고 있었습니다. 저자들은 pre-trained 혹은 online training classifier를 단순히 붙여넣는 것에서 그치지 않고, (1) 이 classifier를 real id=n과 synth id=n 이미지를 구분할 수 있도록 하고 (2) 이 classifier가 discriminator와 같이 generator와 경쟁적으로 학습하도록 함으로써 generator의 결과물을 real source에 더 가깝게 push하겠다는 아이디어를 구현하였습니다. 무슨 소린지 좀 더 보시죠.

![Fig3](https://jiryang.github.io/img/GDC_network.PNG "G-D-C Network"){: width="100%"}{:.aligncenter}

Generator-{Discriminator+Classifier} 로 구성된 GAN 모델의 한 예입니다.


우선 저자는 FaceID-GAN의 classifier에 input face와 동일한 N개의 class 대신 각 id의 class를 $$f^{real}$$ vs $$f^{synth}$$로 한차례 더 나눈 2N개의 class를 부여해서 classifier가 동일한 id를 가진 얼굴이라도 real vs synth 구분을 할 수 있도록 학습시킵니다. 그리고 classifier를 사용한 여타 논문들에선 이 classifier가 단순히 generator의 결과물인 합성 이미지의 id가 올바른 class에 속하는지를 판단해 주었던 반면, 본 논문의 classifier는 discriminator와 유사하게 generator와 minimax 게임을 벌여 generator로 하여금 classifier를 '속이는' 이미지를 합성하도록 유도합니다. Discriminator는 일반 GAN과 마찬가지로 real vs synth를 구분하는 역할을 합니다. 이럼으로써 generator는 discriminator를 속이기 위해 '진짜같은 얼굴' 이미지를 만들어 내게 되고, 동시에 classifier를 속이기 위해 '특정 인물 내에서 좀 더 진짜같은 얼굴'을 만들어 내게 되는 것입니다. 아직도 헷갈리니깐 좀 더 보겠습니다.

![Fig4](https://jiryang.github.io/img/faceidgan_fig2_01.PNG "FaceID-GAN Fig2 Redrawn"){: width="70%"}{:.aligncenter}


논문의 Figure 2를 다시 그려보았습니다. 파란 큰 원은 id=1번의 real face($$f^r_{id1}$$)의 class boundary를 나타냅니다. 그 안에 있는 작은 파란 원들은 id=1번 얼굴의 각각 instance들입니다. 다양한 표정, 다양한 각도의 사진들이 있겠지만 그 facial feature는 어느정도 정규분포를 따른다고 봐도 무리가 없을 것 같습니다. 그래서 파란 원의 중심에 가까운 instance일 수록 id=1 인물의 특징을 잘 나타내는 사진이라고 할 수 있겠죠. 녹색 원은 마찬가지로 id=2 얼굴의 class boundary입니다. Fig3과 같은 pose-guided GAN 결과물이 얼굴을 애매하게 닮았다는 것은 id=1을 가진 synthesized face($$f^s_{id1}$$)가 $$class_{id1}$$의 중심에서 멀지만 boundary 안에는 들어있는, 그러니깐 Fig4의 파란 네모와 같은 instance를 생성했다는 의미라고 이해할 수 있습니다. 또한 $$f^s_{id1}$$가 id=2와도 어느정도 닮았다는 의미는, 이 instance가 $$class_{id2}$$의 boundary에서 벗어나 있기는 하지만 꽤나 가까운 위치에 있다고 이해할 수 있습니다. 이러한 classifier의 class 개수를 2N개로 세분화하면 $$f^r_{id1}$$과 $$f^s_{id1}$$ 사이의 영역을 구분하는 효과를 내게 되고, generator와 classifier를 경쟁하게 하여 수렴시키면 생성된 이미지 $$f^{s'}_{id1}$$은 $$f^r_{id1}$$과 $$f^s_{id1}$$ 사이 중간쯤에 생성될 것이고, 결과적으로 $$f^s_{id1}$$보다 $$f^r_{id1}$$에 '가까운' 이미지를 만들게 되어 ID preserving이 더 잘 될 것이라는 충분히 타당한 intuition을 가지고 구현을 한 것입니다.

![Fig5](https://jiryang.github.io/img/faceidgan_fig2_02.PNG "Pulling Effect of G-C Compatition"){: width="70%"}{:.aligncenter}


나머지 부분은 대동소이하니 FaceID-GAN을 face frontalization에 적용한 결과를 보고 마무리하겠습니다.

![Fig5](https://jiryang.github.io/img/faceidgan_results.PNG "FaceID-GAN Face Frontalization Results"){: width="100%"}{:.aligncenter}


Odd row는 DR-GAN, even row는 FaceID-GAN의 결과입니다. 수치는 ID similarity(아마 cosine distance인 듯)를 나타냅니다. 보시다시피 DR-GAN 대비 FaceID-GAN이 ID preserving을 더 잘 하는 이미지를 합성해냈네요.


GAN이 항상 결과물의 quality를 보장하지 못하기 때문에 논문들에서 발표하는 결과물을 cherry-picking하는 경향이 강합니다. 그래서 오픈소스 돌려보기 전에는 논문만 보고 성능을 믿을 수가 없는데요, 부디 FaceID-GAN은 좀 stable한 결과를 내주기를 바라면서 다음 포스팅을 기약하겠습니다.

논문링크: [FaceID-GAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_FaceID-GAN_Learning_a_CVPR_2018_paper.pdf)


* _이거에 대해서는 따로 한 번 얘기를 하고 싶은데, 어떤 문제(예를 들면 얼굴 합성)든 난이도가 정해져 있을텐데요, 이 난이도를 학습데이터, 모델 복잡도, generalizability 등의 요소로 분할할 수 있는 것 같습니다. 어느 요소가 희생을 하면 다른 요소에서 득을 보는 my loss is your gain 같은..._
