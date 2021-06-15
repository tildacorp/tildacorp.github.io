---
layout: post
background: '/img/backgrounds/fsgan.png'
title:  "FSGAN: Subject Agnostic Face Swapping and Reenactment"
lang: kr
lang-ref: fsgan
date:   2020-05-14 13:59:59
categories: Deepfake
tags: deepfake fsgan faceswap 
excerpt: FSGAN 논문 리뷰
mathjax: true
---

오늘은 작년 코엑스에서 열렸던 ICCV'19에서 소개된 `face swap` 알고리즘인 FSGAN에 대해 소개해보겠습니다.


헷갈리시는 분들을 위해 `face swap`과 `face reenactment`의 차이를 그림으로 보여주면서 SIGGRAPH 스타일로 논문을 시작합니다.

![Fig1](https://jiryang.github.io/img/faceswap_vs_facereenactment.JPG "Face Swap vs. Face Reenactment")


이 논문은 왼쪽의 `face swap`에 관한 내용입니다.
(Deepfake에 악용되었을 경우 `face reenactment`가 더 파장이 클 수 있겠으나 아직은 dummy actor를 놓고 swapping 하는 방식이 quality나 throughput 측면에서 더 낫습니다. 하지만 양쪽 모두 기술이 발전하고 있으니 계속 지켜봐야죠.)


Training data의 분포를 따르는 새로운 instance를 합성하는 `GAN (Generative Adversarial Network)`이 발명되고 수많은 분야에 적용 및 개선이 되어왔습니다. 이후 one-hot vector로 가이드를 줘서 원하는 방향으로 합성 결과를 뽑아내는 [cGAN](https://arxiv.org/pdf/1411.1784.pdf) 방식이 고안되었으며, 이어서 conditional vector의 dimension을 확장하여 한 이미지로 다른 이미지의 스타일을 가이드하여 변경/합성시키는 [pix2pix style transfer](https://arxiv.org/pdf/1611.07004.pdf) 방식이 개발되었습니다. 여기까지가 'innovation' 이라고 하면, 이 이후로는 성능을 최적화한다거나 scale을 높인다거나, 특정 도메인에 특화한다거나 하는 수많은 minor improvement 연구 결과물들이 쏟아져 나오게 되었죠.

| ![Fig2](https://jiryang.github.io/img/tech_s_curve.png "Innovation S-Curve"){: width="50%"}{: .center} |
|:--:|
|*(연구도, 진화도, 비지니스도 innovation S curve를 따르는 것 같습니다)*|


FSGAN은 아래와 같이 face reenactment & segmentation, inpainting, blending의 세 모듈을 통합한 GAN-based 모델을 구성하였습니다.

![Fig3](https://jiryang.github.io/img/fsgan_model.PNG "FSGAN Model Pipeline")


1. Face Reenactment & Segmentation (그림의 Gr & Gs)
  >> ID face를 Attribute face로 transform하게되면 interpolation에 의한 face feature의 변형이 불가피합니다. 두 얼굴 간의 distance(표정, 피부색, 각도 등)가 크면 클수록 필요한 transform magnitude도 커지게 되고, GAN을 수렴시키기가 힘들어집니다. Attribute face와 distance가 가까운 ID face가 있으면 좋겠지만, 이러면 one-shot이나 few-shot 학습이 불가능해지고 필요한 source face data의 양이 많아진다는 단점이 생깁니다.
  >> 저자는 이 문제를 최대한 해결해보기 위해 ID와 Attribute 얼굴들의 facial keypoints를 한 방에 transform하지 않고, 그 차이를 세분화하여 여러 개의 intermediate target facial keypoints를 만들어서 단계적으로 transform을 수행하였습니다. ID face(source)를 intermediate face(target)로 변환은, 2D Euler space(roll은 제외) 상에서 target과 가장 distance가 가까운 source를 선택하여 interpolate를 시킴으로써 one-shot도 가능하되, source data가 많아질수록 ID preserving 측면에서 손실이 줄어드는 방식을 꾀하였습니다.

2. Inpainting (그림의 Gc)
  >> 저자의 예전 논문에서 사용한 inpainting network를 붙여넣어 occlusion augmentation 기능을 구현하였습니다.

3. Blending (그림의 Gb)
  >> Poisson blending loss를 reconstruction loss에 추가하여 구현하였습니다. Blending 부분은 OpenCV에서 Poisson blending을 구현한 seamlessClone() 함수를 썼네요.

앞서 언급한대로 FSGAN은 이론적으로는 one-shot도 가능하지만, 결과 영상(이미지)의 성능을 좋게 하기 위해서는 multiple ID 이미지(혹은 영상)를 필요로 합니다.

![Fig4](https://jiryang.github.io/img/abe2conan.gif "Face Swapping (Abe Shinjo to Conan O'brien)")


위 결과는 저자가 'best practice'라고 말하는 옵션으로 아베신조 총리의 ID를 코난 오브라이언의 attribute에 집어넣은 결과입니다.
ID의 특징이 살아있기는 하지만 ID preserving이 조금 약한 듯 합니다. 아베와 코난의 중간 얼굴이 출력되는 듯한 인상이네요.

최근 FSGAN과 같이 ID 얼굴사진(1)과 Attribute 얼굴사진(2)을 입력하여, (2)의 표정을 따라하는 (1)의 얼굴을 만들어내는 모델들이 많이 개발되고 있는데요, fewer-shot으로 하면서 ID preserving을 얼마나 잘 하는지가 이 분야의 가장 큰 과제인 것 같습니다. Demo 영상에서의 결과물이 썩 괜찮았던것 같아서 FSGAN에 많은 기대를 했었는데요, 안타깝게도 아직 ID preserving 성능이 썩 좋지는 않은 것 같습니다.

[![FSGAN Demo](https://jiryang.github.io/img/fsgan_demo.PNG)](https://www.youtube.com/watch?v=BsITEVX6hkE)




