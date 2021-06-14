---
layout: post
title: "P Stage - 1. Image classification"
subtitle: "이미지 분류 문제"
date: 2021-04-01 23:00:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 이미지 분류 문제 풀이
  
일자별로 기록하였음  
- [Day 1 (3/29)](#day-1-329)
    - [강의 정리](#강의-정리)
    - [시도](#시도)
    - [놓쳤던 점](#놓쳤던-점)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것)
    - [느낀 점](#느낀-점)
- [Day 2 (3/30)](#day-2-330)
    - [강의 정리](#강의-정리-1)
    - [시도](#시도-1)
    - [놓쳤던 점](#놓쳤던-점-1)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-1)
    - [느낀 점](#느낀-점-1)
- [Day 3 (3/31)](#day-3-331)
    - [강의 정리](#강의-정리-2)
    - [시도](#시도-2)
    - [놓쳤던 점](#놓쳤던-점-2)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-2)
    - [느낀 점](#느낀-점-2)
- [Day 4 (4/1)](#day-4-41)
    - [강의 정리](#강의-정리-3)
    - [시도](#시도-3)
    - [놓쳤던 점](#놓쳤던-점-3)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-3)
    - [느낀 점](#느낀-점-3)
- [Day 5 (4/2)](#day-5-42)
    - [강의 정리](#강의-정리-4)
    - [시도](#시도-4)
    - [놓쳤던 점](#놓쳤던-점-4)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-4)
    - [느낀 점](#느낀-점-4)
- [Day 6 (4/5)](#day-6-45)
    - [강의 정리](#강의-정리-5)
    - [시도](#시도-5)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-5)
    - [느낀 점](#느낀-점-5)
- [Day 7 (4/6)](#day-7-46)
    - [시도](#시도-6)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-6)
    - [느낀 점](#느낀-점-6)
- [Day 8 (4/7)](#day-8-47)
    - [시도](#시도-7)
    - [놓쳤던 점](#놓쳤던-점-5)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-7)
    - [느낀 점](#느낀-점-7)
- [Day 9 (4/8)](#day-9-48)
    - [시도](#시도-8)
    - [놓쳤던 점](#놓쳤던-점-6)
    - [부족한 점 & 해볼 것](#부족한-점--해볼-것-8)
    - [느낀 점](#느낀-점-8)
- [총평](#총평)
- [Reference](#reference)

<br />
  
## Day 1 (3/29)
첫째 날은 문제에 대해 이해하고 간단하게 학습 모델을 설계하여 돌려보았다.  

#### 강의 정리
- Overview를 잊고 지나가는 경우가 많은데, overview에 힌트가 있는 경우도 더러 있다. overview를 보고 풀고자 하는 문제가 무엇인지 고민해보자.  
- 한계에 부딪히면 방향성을 생각해보면서 데이터 전처리 방식을 다시 고민해보자. 방향성을 잃지 말자.  
- **EDA(Exploratory Data Analysis)**는 중요하지만 하는 방식이 거창할 필요 없다.  
  + 그냥 엑셀로 보거나 손으로 직접 찍어봐도 된다. 꼭 이걸 파이썬으로 처리할 필요가 없다. 수단이 어떻든 주요한 특징 등을 파악하는 것이 중요하다.  
    
<br />

#### 시도
- Dataset을 직접 만들어본건 처음인데 생각보다 만만치 않았다..ㅋㅋㅋㅋㅋㅋ 파일 불러와서 사용할때는 <code>glob</code> 라이브러리 앞으로도 잘 활용해야할 것 같고 <code>os</code>도 제대로 써본적이 없었는데 오늘 Dataset 구성하면서 많이 배운 것 같다.  
- 모델은 ResNet, MNASNet, EfficientNet E0 시도해보았고, 당연히 pretrained EfficientNet의 성능이 제일 우월했다.
- 간단한 data augmentation으로써 RandomCrop, RandomHorizontalFlip을 적용해보았다.
- image dataset 만들 때 normalize도 적용하였다. 
- stratified된 train_test_split 적용하였는데, 추후 stratified kfold로 더 깊은(?) 학습이 필요하다.
- 현재 타입별로 라벨을 구분지어 모델 3개를 돌리는 학습 진행 중이며 내일 inference 예정. 근데 에폭을 좀 많이 둬서 오버피팅될수도 있을거같은데 일단 지켜봐야 할 것 같다.

<br />

#### 놓쳤던 점
- Inference 단계에서는 transform 적용이 달라져야한다. Random 같은거 다 빼야할 것.
- Inference 단계에서 Dataset과 submission file의 파일 순서가 같아야한다. 똑같은 순서로 정렬한 상태여야한다는 뜻. 다른 방법도 있겠지만 정렬하고 하는게 제일 편한거 같다.

<br />

#### 부족한 점 & 해볼 것
- training할 때 accuracy, loss, progress bar 표시하는거 너무 더럽게 해놨는데 이거 잘 된 사람꺼 참조해서 바꿀 필요가 있어보인다.
- Markdown 중간중간 넣어서 좀 보기 좋게 만들 필요가 있다.  
- 데이터가 불균형하므로 불균형 문제를 어떻게 해결할지 생각해보자. 적은 데이터만 증강할 것인지? 등 ~
- Image classification의 SOTA로 EfficientNet B8에 Meta Pseudo Labels을 적용한 모델이 있었다. 내일은 해당 논문을 읽어 볼 예정이다.
- 현재 data augmentation을 별로 적용하지 않았으니 증강을 어떤 방식으로 적용하는게 좋을지 알아보아야 한다.
- 현재 hyperparameter tuning이 전혀 되지 않았으니 이것도 해볼 필요가 있다. (GridSearch 등)
- cross validation도 안되었다.  
  
첫날이라 아직 해보지 않은게 수도없이 많지만 일단 기억나는건 여기까지.. :smile:    
  
<br />

#### 느낀 점
Competition에 이렇게 제대로 참여하는건 처음인데 밑바닥부터 짜는 것이다보니 지난번에 프라이빗 챌린지했을 때보다 훨씬 재밌는 것 같다. :smile:  

<br />

## Day 2 (3/30)

#### 강의 정리

- Data augmentation도 적재적소에 활용해야한다. 도메인에 맞지 않는 증강 기법을 활용하면 오히려 성능이 하락할 수 있다. 데이터셋을 살펴보고 어떤 것을 적용하면 좋을지 생각해보자. 
- <code>torchvision.transforms</code>도 좋지만 외부 라이브러리인 <code>Albumentations</code>의 
  **속도가 더 빠르므로** 범용적으로 많이 활용되는 편이다.
- Data augmentation 자체도 성능에 많은 영향을 주므로 어떤 것을 적용할지, 혹은 그 순서를 결정할때도 어느정도 근거가 있어야한다. 
    
<br />

#### 시도
- 여러 번의 epoch에서 best acc를 보인 모델을 가져와서 돌리는 시도를 하였다.
- training단의 코드를 좀 더 깔끔하게 정리하였다.
- 헤드를 다 한 모델에 몰아넣는 것은 확실히 성능이 그리 잘 나오지 않았다. 내부에서 여러 분기를 나누더라도 결국 마지막에 FC layer를 통해 18 클래스로 분류하는것은 별로인 것 같다.
다시 어제 짰던 v2로 돌아가서 생각해야할 것 같다.  
- data augmentation을 좀 더 적용해보았다.  
- 이상치 데이터를 삭제하였다. 좀 소수라서 실질적으로 성능에 얼마나 영향을 미쳤는지는 미지수이다.  
- 이미지 정규화시 실제 이미지의 수치를 적용하였다. 근데 생각해보니 테스트데이터는 분포가 다를텐데 이 수치를 그대로 적용해도되는건지 잘 모르겠다. 

<br />

#### 놓쳤던 점
- 데이터 분포에 대한 생각을 좀 더 많이 해봐야할 것 같다. 데이터를 어떻게 나누어 학습시킬 것인지도 매우 중요할 듯하다. 내일은 이것에 초점을 맞추어 좀 더 발전시켜보자.
- 외부 데이터셋을 활용해도 된다고 피어세션에서 들었다. 외부 데이터셋 활용도 고려해보아야겠다.

<br />

#### 부족한 점 & 해볼 것
데이터 불균형문제를 어떻게 해결할지 내일 최우선적으로 생각해보아야겠다. 
그 다음에 hyperparameter tuning도 해보아야겠다.
사실 하이퍼파라미터는 튜닝 한 번 하려고하면 시간이 너무 오래걸릴 것 같아 엄두가 잘 나지 않는다 :cry:  
  
어제 읽으려고 생각했던 pseudo labeling 논문을 아직 읽지 못했는데 내일은 이걸 토대로 데이터 불균형 문제를 어떻게 해결할지 고민해보자.  
   
데이터의 분포, 그리고 틀린 데이터가 주로 어디서 틀리는지 등을 중점적으로 생각해보아야겠다.
지금 몇개 봤을때 의외로 mask label은 잘 맞히는데, gender와 age(특히 60세 이상인지 미만인지)에서 많이 애매한 것 같다.   
    
<br />

#### 느낀 점
오늘은 생각했던 방법들을 여러가지 시도해보았지만 성능이 잘 오르지 않았다. 
계속해서 competition에 정진하는 것은 좋지만, 너무 모델 학습만 돌리지 말고 subtask들을 잘 구성하여 각각을 테스트해보는 것도 좋을 것 같다.
오늘은 뭔가 성급해진 느낌이 있었고 제대로 이해하지 못한 상태에서 여러가지 방법들을 적용해보았다. 시행착오라고 생각해야할 것 같다.
어차피 지금은 배우는 과정이니 너무 모델 성능만 신경쓸 것이 아니라 내일부터는 다양한 시도를 해보고, 그 시도 자체에 의의를 두자~ ~ ~  
  
한편, 뭔가 깊은 고민 없이 모든 과정을 진행하고있는 느낌이 든다. **근거 있는 판단을 하는 연습**을 하자. 
내일은 일단 데이터 전처리에서부터 깊은 생각이라는걸 해보면 좋을 것 같다.  

<br />

## Day 3 (3/31)

#### 강의 정리
- 가지고있는 데이터의 양, 그리고 pretrained model이 사전 학습할 때 사용한 데이터가 지금 적용하고자하는 task와 비슷한지, 이 두가지를 확인하여 pretrained model을 어떻게 활용할 것인지 결정해야한다.
  + 만약 데이터가 적고 pretrained model이 비슷한 task에서 적용되지 않았으면 오히려 scratch에서 학습을 진행하는 것이 더 나을지도 모를 것 같다.
    
<br />

#### 시도
- KFold를 적용하였다.
- weight이 적용된 CrossEntropyLoss를 활용하였다.
- SGD를 적용해보았다.

<br />

#### 놓쳤던 점
- CenterCrop을 train에서만 적용하고, test에서는 적용하지 않아 정확도 하락이 꽤 컸다. 이 부분을 놓쳤는데 피어세션에서 한 캠퍼분께서 말씀해주셔서 고칠 수 있었다.
- KFold를 했는데 model parameter 저장을 각 fold별로 별도로 두지 않고 계속해서 이어서 저장하는 어처구니없는 실수를 했다.
  + 이렇게 할 경우, 가장 치명적인 문제는 이전 fold에서 train쪽이었던 데이터를 다음 fold에서 valid로 마주하게 된다는 점이다. 당연히 그때부터 정확도가 100%에 육박하였다.
- Adam만을 활용하는 것이 능사인줄 알았는데 아니었다. SGD, AdamW등 다양한 optimizer를 고려해야할 것 같다. 

<br />

#### 부족한 점 & 해볼 것
- Data augmentation을 아직까지 제대로 하지 못했다. 내일은 data augmentation에 초점을 두어야할 것 같다.  
- Data augmentation을 어느 범위까지 적용해야할지, 어느 것이 도움이 되고 어느것이 도움이 안되는지 판별해야한다.  
    
<br />

#### 느낀 점
뭔가 고정관념에 사로잡혀있는 부분은 쉽게 고쳐지지 않는 것 같다. 
여러 코드들을 보며 이런 점들을 수정해나가야 할 것 같다.  

<br />

## Day 4 (4/1)

#### 강의 정리
- 오늘은 PyTorch의 파이프라인이 어떻게 구성되는지, 각각의 동작은 어떤 방식으로 이루어지는지 배웠다. 
- Competition에서는 metric을 먼저 제시하지만, 현업에서는 metric을 우리가 직접 골라야하므로 어떤 task에 어떤 metric을 활용하는 것이 합당한지 판별할 수 있는 능력을 기르는 것 역시 중요하다.
- Learning scheduler에 대해서 배웠다. 대표적으로 일정 step 걸으면 learning rate를 조정하는 StepLR, 코사인 함수의 형태로 learning rate가 계속해서 변화하는 CosineAnnealingLR, 성능 향상의 정도가 지지부진할 때 자동으로 learning rate를 더 낮춰주는 ReduceLROnPlateau 등이 있다.
- training pipeline을 자동으로 구성해주는 PyTorch Lightning 모듈도 있다. 

<br />

#### 시도
- KFold에서 데이터를 잘못나누었던 오류를 수정하였다.
  + 이전처럼 사람을 기준으로 데이터를 나누었다.
- 다양한 data augmentation을 적용하였고 이를 albumentation으로 고쳐 적용하였다.  
  + 마스크 라벨은 잘 구분하는 모습을 보여 마스크 라벨은 고려하지않고 성별/나이만 고려하여 데이터를 일정수준까지 증강시켰다.
  + 다만, valid set은 따로 증강을 하지 않았는데 생각해보니 이렇게 하면 valid set의 불균형때문에 정확도가 높게 잡힐 우려가 있어보인다. 내일은 이 부분을 수정해야한다.
  + augmentation에서 사진의 색상을 크게 왜곡시키는 트랜스포머들(GrayScale, ChannelShuffle)은 제외하였다. 성능이 안좋았다는 캠퍼분들의 의견이 있었다.
  + Normalization도 일단 제외하였다. 그런데 솔직히 이 픽셀정규화가 학습에 얼마나 도움이 될지 잘 모르겠다. 모델 내에서 BatchNorm이 있으면 영향이 덜하지 않을까?  
- 헤드 3개에서 마지막 FC layer를 하나 더 뒀던 파이프라인을 고쳐서 다시 모델 3개로 두었다. 모델 3개로 두는 것이 시간이 오래걸려도 성능이 더 좋긴 했다.

<br />

#### 놓쳤던 점
- 데이터 분포에 대한 생각을 좀 더 많이 해봐야할 것 같다. 데이터를 어떻게 나누어 학습시킬 것인지도 매우 중요할 듯하다. 내일은 이것에 초점을 맞추어 좀 더 발전시켜보자.
- 외부 데이터셋을 활용해도 된다고 피어세션에서 들었다. 외부 데이터셋 활용도 고려해보아야겠다.

<br />

#### 부족한 점 & 해볼 것
내일은 앞서 말했듯이 valid set에서도 합당한 validation이 이루어질 수 있도록 valid data의 추가적인 처리가 필요할 것 같다. (특히 나이 데이터셋에서)
더불어 이러한 수치까지 잘 반영할 수 있도록 f1-score도 validation시에 적용할 필요가 있어보인다.
실제 데이터셋에는 우리에게 부족한 노년층 데이터가 좀 더 많지 않을까 싶다. 
그리고 사람의 얼굴 부분만 잘라낼 수 있는 모듈도 존재하는 것 같은데 CenterCrop 대신 이러한 모듈들도 활용해보면 좋을 것 같다.        
    
내일은 터미널 환경 구축을 좀 해보도록하고, accuracy/f1-score/loss 등을 저장 및 시각화하는 코드도 추가적으로 짜야한다. 

<br />

#### 느낀 점
이전에도 느꼈지만 생각없이 하고 있는 것 같은 느낌이 들기도 한다. ㅋㅋ;   
새로운 정보나 논문을 알게되었을때 그것을 받아들이는 것을 꺼리지 말고 하루에 하나만 보더라도 그것만큼은 제대로 이해하고 넘어가자는 자세가 필요하다.  
  
또, 오늘 피어세션을 하면서 많은 것을 느꼈는데 나는 하고있는 모든 것을 기록으로 남기지 않고 있다. 
그래서 어쩌면 같은 것을 여러번 하고 있는? 비효율적인 파이프라인을 활용하고 있는 것 같기도 하다. 
사실 생각하기도 귀찮고 뭔가를 더 짜기도 귀찮아서 이러고 있는데 결국 모델 학습은 튜닝의 싸움이다. 
차라리 코드 하나를 제대로 짜고 비교분석을 제대로 할 수 있게 만드는 것이 더 좋을 것 같다.  
  
모델 구성을 고정해두고 data augmentation에 변화를 준다든지하여 뭔가 domain 별로 최적의 구성을 하나하나 찾아나가는 것이 필요할 것 같다.
그리고 이 과정에서 보다 가벼운 모델을 활용하여 하나하나를 찾아가는 과정이 길지 않도록 코드를 짜야할 것 같다. 
더불어 실질적으로 바뀐 변인이 얼마나 성능에 영향을 줬는지에 대해서도 시각화를 통해 확실히 파악해야한다.   

<br />

## Day 5 (4/2)

#### 강의 정리
- 하이퍼 파라미터 튜닝도 중요하지만 너무 오래걸리는 작업이고 그 시간에 비해 성능 향상이 도드라지는 영역이 아니므로 너무 여기에 몰입하지 말자. 이를 위한 Optuna 라는 라이브러리도 존재한다.  
- 대신 ensemble, (stratified) K-fold를 이용한 cross validation, TTA(Test Time Augmentation) 등의 방법을 고려해볼 수 있다.
- 파이프라인을 훨씬 효과적으로 돌리려면 아무래도 Jupyter Notebook(ipynb) 환경보다는 Python IDLE(py)를 통해 작성된 모듈화된 파일이 더욱 좋다. 
- 그 외 시각화툴(텐서보드, wandb)을 이용한 시각화가 효율성에 도움을 줄 수 있다.

<br />

#### 시도
- minor data에 대한 augmentation이 성능에 얼마나 영향을 줄지 궁금하여 실험을 진행해보았는데, minor data만 증강시키는 것이 정확도에는 좋은 영향을 주었지만 F1 score는 오히려 하락시켰다. 그런데 사실 큰 영향은 아니라서 영향 자체가 미미하다고 볼 수 있다.
- 입 주변으로 cutout을 시도해보았지만 성능 하락이 컸다. 아무래도 정밀한 cutout이 아니다보니 성능에 안좋은 영향을 준 것 같다.  
- Cross entropy에 weight을 주는게 좋을지, augmentation을 하는게 좋을지, augmentation을 준다면 얼마나 줘야하는지 등을 결정하는 것이 중요한 요소가 될 것 같다.
- learning scheduler를 ReduceLROnPlateau로 변경하였다.


<br />

#### 놓쳤던 점
- 데이터 augmentation에서 보다 정밀한 처리가 필요하다. cutout을 대충 적용하면 확실히 학습에 도움이 되지 않는다.  

<br />

#### 부족한 점 & 해볼 것
- 주말동안 좀 쉬어주기도하고, 지금까지 주피터 노트북 환경에서만 진행했던 학습 프로세스를 IDLE 환경으로 변경, 모듈화하여 새로운 학습 환경을 구축할 예정이다.
- 시각화 툴을 활용할 수 있도록 세팅을 변경하여야 한다.
- 컴퓨터가 꺼진 상태에서도 서버는 돌아갈 수 있도록 tmux 등의 모듈을 사용해볼 예정이다.
- Focal loss를 활용하면 좋을 것 같다. 다만 확실히 gamma 값을 너무 높게 주면 학습이 잘 되지 않는 것 같다. 

<br />

#### 느낀 점
이번 컴피티션은 성능 향상보다도 좀 새로운 시도나 효율성을 높일 수 있는 다양한 방법들을 시도해보는 시간으로 활용해도 좋을 것 같다. 
아직 컴피티션이라는 것 자체에 익숙하지 않다보니 계속 비효율적인 시도를 하고 있다는 생각이 든다. 

<br />

## Day 6 (4/5)

#### 강의 정리
- 시각화 툴의 필요성과 간단한 시각화 방법에 대해 배워보았다.  

<br />

#### 시도
- 모듈화된 학습 환경에서 SSH/IDLE를 활용한 학습을 시도하고 있다. 성능 자체에는 영향이 없으나 보다 효율적인 학습 프로세스를 구축할 수 있을 것으로 예상된다.

<br />

#### 부족한 점 & 해볼 것
- Data augmentation에 있어 정답이랄건 없겠지만, 하면 할수록 어떻게 해야할지 전혀 감이 잡히지 않는다..
- 슬슬 막바지에 다다르고 있어, KFold나 ensemble을 적용해야할 것 같다. ResNext50, NFNet, DeiT 등 이전에 SOTA로 거론되었던 모델들을 활용해보자.  
- 아직 시각화툴 세팅을 해보지 않았는데 시각화툴도 이번 기회에 세팅을 해보아야한다.  

<br />

#### 느낀 점
짧은 2주라는 시간에도 벌써 지친다는 느낌이 들기도 하는데 끝까지 지치지 않고 시도해보는 것이 중요할 것 같다.
그렇다고 아무렇게나 하지 말고, 항상 근거 있는 시도를 할 수 있도록 노력하자.
  
<br />

## Day 7 (4/6)
오늘부터는 강의가 없어 강의 정리는 제외하였다. :smile:  
  
#### 시도
- 뒤늦은 감이 있지만(...) 이제 와서 시드 고정 및 실험환경 세팅 후 augmentation 관련 여러 실험을 하고 있다. 
  사실 이전에도 focal loss/cross entropy의 차이, augmentation별 성능 등을 실험환경을 분리하여 했어야했는데.. 다음부터는 이러한 점을 명심하자.
- 이전에 제출했던 submission 파일들 및 이전 weight 값들로 ensemble을 시도 중이다. 의외로 내 환경에서는 hard voting이 잘 먹혔다. hard voting도 그냥 편해서 우연히 하게된건데, 더 성능이 좋아서 좀 놀랐다. 
- 지금까지는 EfficientNet B0만 사용했는데 ResNeXt도 테스트해보고 있다.

<br />

#### 부족한 점 & 해볼 것
- 앙상블 과정과 이전까지 해본 것들에서 느낀점은 gender/mask 모델도 재학습시켜야겠다는 점이다. 현재 계속해서 v11에서 훈련된 것을 사용하고있는데 여기서 더 score를 높일 수 있지 않을까?
- 버전 관리를 철저하게 하고 hard voting을 한다면 어느 모델까지 적용할 것인지 생각해보아야 한다.

<br />

#### 느낀 점  
정답이라는게 없는걸 느낀다. 문제마다 적재적소에 활용해야하는 기법들이 제각기 다르다. 꼭 가장 좋다고 통용되는 방법을 사용하는 것이 내가 보고 있는 문제에서도
잘 통하는 것은 아니라는 것을 느꼈다. 

<br />

## Day 8 (4/7)

#### 시도
- 현재 ResNeXt 50/101, EfficientNet B0/B4에 대하여 loss 함수별 성능/augmentation 여부별 성능을 재확인하고 있으며 val ratio를 0.0으로 낮추어 학습하는 것도 계획에 있다. 다만 모델 여러개를 학습시키다보니 시간이 오래걸려 KFold는 적용하지 못할 것 같다. ㅠㅠ KFold가 성능에 큰 영향을 줄까..?
- 토론 게시판에 올라온 multi-head 모델을 새로 돌려볼 계획이다. multi-head는 처음에 생각해보다가 성능이 생각보다 안나와서 폐기했는데 토론 게시판에 올라온 모델은 그 모습이 내가 생각했던 것과는 사뭇 다르다. 확실히 좋은 성능을 기대할 수 있을 것 같아 밤에 돌려놓으려고 한다.  

<br />

#### 놓쳤던 점
- 버전 관리 및 환경 세팅을 아직까지도 대충 해놔서 여러모로 빠뜨리는 경우가 많은데 이 점을 고쳐야한다.

<br />

#### 부족한 점 & 해볼 것
- age/gender 모델 재학습시 성능이 얼마나 오를지 테스트가 필요하다. submission 기회가 하루 10번이라 잘 활용해야할 것 같다.
- 실험을 슬슬 마무리하고 성능이 잘나오는 것들 위주로 빠르게 ensemble할 예정이다. ensemble은 hard voting으로 한다.  

<br />

#### 느낀 점  
감이 잡힌게 며칠전부터인데 여러가지를 테스트해볼만큼의 시간이 있지 않아 쉽지 않은 것 같다 .. -.- ;;      
  
근데 감이 잡혔다고해도 테스트해보았을때 성능이 무조건 잘 나오는 것은 또 아니라서 참 어렵다.   

그리고 다음 competition부터는 여기다가 기록을 중간중간 계속 해야할 것 같다. 
뭔가 하루가 다 끝난 후에 기록하려고 하니 그때그때 떠올랐던 생각들이 다시 안떠오르는 것 같다.  

<br />

## Day 9 (4/8)

#### 시도
- MultiHead 모델을 설계하여 ensemble을 적용하였다. 의외로 multihead model에서는 또 data augmentation 적용결과 성능이 더 좋게 나왔다. (!!) 그런데 이게 과연 augmentation 때문인지, 아니면 데이터 비율 자체가 증가해서 그런건지는 아직 확실하지 않다. 아무튼 무엇보다도 multihead model의 성능이 매우 좋았다는 점!    
(피어세션때 여기저기 augmentation 결과 성능이 하락했다는 것을 말하고 다녔는데 어떡하나요 :sob: :sob: 내 말을 듣고 augmentation을 고려대상에서 제외하신 분은 없길 기도한다 :cry: ) 
- hard voting부터 soft voting까지 다양한 기법들을 적용해보았다. 워낙 만들어본 모델이나 학습시킨 모델이 많다보니 앙상블을 적용할 수 있는 범위가 넓어졌다. 수많은 삽질의 보상인걸까..? 아무튼 앙상블 결과 성능이 잘 나와서 놀라웠다.  
- val_ratio를 0.001(사실상 없는거나 마찬가지)로 설정하고 모델을 돌렸고 예상대로 성능이 잘 나왔다. epochs는 대충 때려맞췄는데 12로 하였고 그럭저럭 잘 나온 듯하다.  
- 사실 이전부터 적용했던건데 이번에 만든 submission 파일과 이전 최고기록 submission 파일 간의 labeling 차이로 대충 성능이 잘 나올지 안 나올지 예측해볼 수 있는 것 같다. :smile:
 
<br />

#### 놓쳤던 점
- 앙상블을 적용할 수 있는 범위는 무궁무진하다.. 이번에도 역시 처음 적용하는 기법에 성능이 잘나오는 것을 보고 흥분하여 계획을 세우지 않고 마구잡이로 제출하다보니 제출기회 10번을 다 날렸다. :expressionless: 그래도.. 이 또한 교훈이 되리라 생각한다.  

<br />

#### 부족한 점 & 해볼 것
- 이제 대회 마감인데 코드 정리를 좀 해보고 README 작성, Wrap-up 리포트 작성을 좀 **깔끔하게** 하자.. 지난 2주간 작성해온 코드나 내가 보려고 만든 문서들은 너무 더럽고도 또 더럽다.  

<br />

#### 느낀 점  
토론 게시판을 많이 보자 했었지만 생각보다 많이 못봤는데 마지막에 놓고 보니 역시 쏠쏠한 글들이 많았다.
다음 대회부터는 올라오는 글들을 열심히 읽고 나도 많이 작성해야겠다는 생각이 든다. 
그리고 앞에서 말했듯이 처음 쓸 때부터 코드를 깔끔하게 써야할 것 같다. :sweat_smile: 한번 지나고보니 잘 돌아가는 부분을 고치고싶다는 생각이 잘 안들게 되는 것 같다.  
  
사실 힘이 빠지는건 결국 가장 좋은 성능이 CrossEntropy, Adam 등 기본적인 criterion 및 optimizer에서 나왔다는 점이다. 
많은 것을 적용해보았지만 이것들이 좋았다는 것은 좀 눈여겨볼만 한 것 같기도..   
  
하지만 결과론적으로 data augmentation이 성능 향상에 도움이 되었으니.. 역시 task마다 적합한 도구들을 찾는 게 중요한 것 같다.  

<br />

## 총평  
뭔가 아쉬운 것도 많고 내가 제대로 했나 싶기도 한데.... ㅋㅋㅋㅋ 그래도 순위가 높게 나와 기분이 좋다 :)    
  
이번에 순위 맛을 달콤하게 봤으니 다음부터는 성능에 너무 눈 멀지 말고, 꼭 플랜을 철저히 세우고 세운 플랜을 성실히 이행하는 것에 더욱 집중하도록 하자!  
  
더 자세한 총평은 랩업리포트에~  
    
<br />

## Reference  
[Hieu Pham, et al. "Meta Pseudo Labels"](https://arxiv.org/pdf/2003.10580.pdf)