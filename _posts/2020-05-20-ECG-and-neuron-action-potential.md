---
layout: post
title:  "ECG, Neuron's Action Potential, and ANN"
date:   2020-05-20 10:00:00
categories: Health
tags: ecg neuron actionpotential ann neuralnet
excerpt: ECG와 뉴럴 네트워크의 관계?
mathjax: true
---

오늘은 ECG 파형에 대해 간단히 이해를 해보고 (ECG 데이터를 가지고 리서치를 하려면 배경 지식이 좀 있으면 좋겠죠), 이 이야기를 뉴럴 네트워크까지 이어가보도록 하곘습니다.

심근세포의 action potential은 polarization-depolarization에 의해 발생합니다. 
뉴런의 action potential과 굉장히 비슷하기 때문에 뉴런의 예로 설명해보겠습니다.

지질로 구성된 생물의 세포막은 안쪽과 바깥쪽 voltage의 차이를 유지합니다 (막 안쪽이 -70mV 정도로 negative임). 어떤 세포들은 이 막전위를 constant하게 유지하지만 일부는 voltage가 변하기도 하며, 특히 신경세포와 근세포들은 매우 빠르고 연속적으로 이러한 voltage 변화를 일으켜서 세포내에서 신호를 전달하는 기능(dendrite-to-axon terminal)을 수행합니다.

![Fig1](https://jiryang.github.io/img/action_potential.png "Neuron's Action Potential"){: width="50%"}{: .center}


Negative로 idle 상태를 이루는 막전위는 sensory input (감각뉴런의 경우), neurotransmitter (inter-neuron activation) 또는 ion channel의 기능을 통해 action potential이 발생하게 되는데 이 메카니즘에 대해 알아보겠습니다.

![Fig2](https://jiryang.github.io/img/ion_pump.png "Ion Channeling in Action Potential"){: width="100%"}{: .center}


지질로 이루어진 세포막은 나트륨이나 칼륨과 같은 이온이 통과하지 못합니다. 뉴런의 세포막에는 이온들이 선택적으로 통과할 수 있는 channel들과 (e)와 같은 sodium-potassium pump가 있어서 나트륨은 내보내고 칼륨은 들여보내면서 막전위를 -70mV 정도로 유지하게 되는데요, 감각신호/이전 뉴런/주변의 action potential에 의해 자극을 받게되면 먼저 (c)의 sodium channel이 열리게 되고, 이를 통해 세포막 외부의 나트륨 이온들이 세포내로 들어오게 되면서 막 내 전위가 올라가게 되며, 이를 depolarization이라고 부릅니다 (idle state가 inner-negative, outer-positive로 polarized 된 상태였죠). 

(2) Depolarization으로 인한 전하량 변화가 threshold voltage (약 -55mV)를 넘지 않으면 neuronal firing은 일어나지 않은 채 sodium-potassium pump가 동작하여 다시 idle state로 polarization이 이루어지는데요, threhold를 넘게 되면 acton potential이 약 +40mV까지 급속도로 증가하게 되면서 피크를 치고 다시 repolarization 과정에 들어가게 됩니다. 

(3) Repolarization 때에는 sodium channel이 닫히고 potassium channel이 열리면서, 세포막 내의 칼륨 이온이 세포 밖으로 나가게끔 유도하여 전위를 떨어뜨리는데요, 급하게 빼느라 칼륨 이온을 좀 과도하게 배출하게 되어서 action potential이 아랫쪽으로 조그만 bump를 만들게 됩니다.

(4) Refractory period에서는 potassium channel마저 닫히고, sodium-potassium pump가 동작하여 세포 안에 들어온 나트륨은 다시 밖으로 배출하고, 세포 밖으로 나간 칼륨은 다시 안으로 들여와서 idle 상태로 복귀하게 됩니다.


이같은 action potential이 국지적으로 발생하게 되면 주변 세포막 주위의 이온 분포를 변화시키게 되고, 이 자극으로 인해 action potential이 연속적으로 발생하게 되는 효과를 낳습니다.

![Fig3](https://jiryang.github.io/img/action_potential_propagation.png "Action Potential Propagation"){: width="50%"}{: .center}



그 결과, 전기 신호가 뉴런을 타고 전달되는 효과를 낳는거죠.

![Fig4](https://jiryang.github.io/img/action_potential.gif "Inter-neuron Signal Transfer"){: width="60%"}{: .center}



뉴런의 axon은 myelin이라는 지방으로 코팅이 되어있는데요, myelin의 역할은 위의 action potential이 전달되는 속도를 배가시켜줍니다. 단위시간에 더 많은 impulse를 전달하게 되어 결국은 더 '강한' 신호가 전달되도록 하는 효과가 납니다.

![Fig5](https://jiryang.github.io/img/neuron.PNG "Neuron"){: width="50%"}{: .center}


뇌에서 A-to-B의 전달 신호의 세기는 두 지점을 연결하는 뉴런의 갯수, 그 연결을 구성하는 뉴런들의 synapse의 수, 구성 뉴런들 사이의 neurotransmitter의 양 뿐만 아니라 저 myelin 또한 영향을 미칩니다. 이를 굉장히 단순화하여 모델링 한 것이 ANN이고, 여기선 신호 전달의 세기가 weight의 magnitude로 표현이 되지요.

![Fig6](https://jiryang.github.io/img/ann1.jpg "Biological vs. Artificial Neuron"){: width="100%"}{: .center}


이제 biological neuron과 ANN의 neuron 사이의 analogy가 좀 더 명확해졌나요? ECG 잠깐 볼까요?

![Fig6](https://jiryang.github.io/img/ecg_pqrst.png "ECG of Normal Sinus Rythm"){: width="50%"}{: .center}


Neoron의 그것과 상당히 유사한 ECG로 측정한 심근세포의 action potential은 한 step이 PQRST 단계로 이루어져 있습니다. 심장의 구조, 동작방식 등의 차이로 인해 조금 모양이 다르긴 하지만, 세포막의 이온 채널에 의한 idle-depolarization-repolarization이라는 일련의 과정은 뉴런의 경우와 대동소이합니다. 6개 정도의 lead를 사용해서 가슴 여러 부위에서 이러한 파형을 측정하고, wave 간의 거리, peak의 높이, 모양 등등을 통해 심박 수 측정 뿐만 아니라 심방세동, 빈맥, 심방조동, 고칼륨혈증, 심실비대증, 심근허혈증, 동맥류, 협심증 등의 질병 가능성을 유추할 수 있다고 합니다. 

나이, 체중 등에 따른 개인별 차이, 질병의 경중에 따른 차이, 기왕증 유무에 따른 차이 등등 수많은 변수들이 prediction의 성능에 영향을 미칠 것이기 때문에, 이 부분에 ML을 적용하면 진단의 정확도와 정밀도를 높일 가능성이 있겠지요?