---
layout: post
title:  "Deep Neuroevolution - part1"
date:   2020-06-21 12:59:59
categories: Neuroevolution
tags: neuroevolution genetic_algorithm reinforcement_learning uber_ai
excerpt: Deep Neuroevolution - Genetic Algorithm 기초 다시보기
use_math: true
---


코로나 바이러스로 많은 비지니스들이 타격을 입었지만, 사람들의 이동이 줄어들어서 여행산업 쪽이 특히 피해가 컸습니다. 전통적인 항공사나 여행사들 뿐만 아니라 Uber, Airbnb, Lyft와 같은 tech 기업들도 적잖은 타격을 받아서 수천명씩 layoff를 하기에 이르렀죠. 특히 Uber의 경우는 전 세계 고용인원 중 14% 정도에 달하는 3700명 가량을 해고하면서 (Airbnb는 25%인 1900명, Lyft도 20%가 넘는 1200명 이상을 해고했네요), Uber AI의 프로젝트들을 줄여나가겠다고 발표하였습니다 ([링크](https://analyticsindiamag.com/uber-ai-labs-layoffs/)).


Uber AI에서는 메인 비지니스 영역에 필요한 Computer Vision, 자율주행, 음성인식 등과 같은 주제 외에 조금 색다른 포인트의 연구도 해왔는데요, 바로 neuroevolution입니다. Neuroevolution은 Genetic algorithm (GA)과 같은 진화 알고리즘을 사용하여 agent (agent를 구성하는 neural network의 weight나 topology) 또는 rule 같은걸 학습하는 머신러닝 방법으로, artificial life simulation, game playing, evolutionary robotics와 같은 분야에 사용되던 'old school' 기법이라고도 할 수 있습니다. 요새 유명해진 AutoML과 개념적으로 비슷한 neural network의 topological learning 방법인 NEAT ([Neuroevolution of Augmenting Topologies](https://www.cs.ucf.edu/~kstanley/neat.html))를 개발하였던 UCF의 Kenneth O. Stanley 교수님이 Uber AI의 Neuroevolution 분야를 리딩하셨는데요, NEAT는 2000년대 초반에 간단한 방법으로 네트워크의 topology를 효과적으로 진화시키는 결과를 보여주어서 굉장히 인상깊었던 논문입니다.


앞 단락에서 설명드린 neuroevolution의 사용처를 보시면 눈치채셨겠지만 주로 reinforcement learning의 영역이지요? 강화학습 쪽 market requirement도 있고 해서 언젠가 한 번 다루어야겠다 생각을 하고 있었는데요, 이번 기회에 Uber AI에서 진행되었던 neuroevolution 관련 리서치 결과들을 여러 개의 포스팅으로 나누어 훑어보도록 하겠습니다. GA부터 NEAT를 거쳐 Deep Neuroevolution까지 긴 시리즈가 될 것 같습니다. 중간중간 강화학습 자체에 대한 내용이 적잖이 들어가야 할 것 같은데, 이미 잘 설명된 다른 블로그들이 있는 것 같아서 상당 부분은 인용하도록 하겠습니다.

![Fig1](https://jiryang.github.io/img/neuroevolution.png "Neuroevolution"){: width="80%"}{: .aligncenter}


**Neuroevolution**<br><br>
Neuroevolution은 이름 그대로 neuron을 evolve하는 방식으로 학습을 한다는 의미입니다. Machine learning context에서 이야기를 하는 것일테니 여기서 neuron이란 Artificial Neural Network (ANN)를 뜻합니다. Evolution이란 진화의 방식을 모방한 학습을 이용했다는 의미로, population-based optimization 방식인 GA가 neuroevolution 계열의 가장 대표적인 알고리즘이라 할 수 있습니다.<br><br>
_Genetic Algorithm_<br>
GA는 여러 biological evolution theory를 조합해서 만든 머신러닝 알고리즘입니다. Charles Darwin의 'survival of the fittest', Jean-Baptiste Larmark의 'use and disuse inheritance', 그리고 Gregor Mendel의 'crossover & mutation'을 응용하였죠. 동작하는 방식은 다음과 같습니다:
1. Randomly initialized pool of agents로 (stochastic이라 할 만큼) 충분히 큰 population (size N>>)을 생성 (initial generation)
2. 당 generation의 각 agent에게 task를 수행시킴
3. 각 agent의 task에 대한 fitness 값 계산
4. 성능 좋은 n개 (n<<N) agent를 breeding 용으로 선택 (survival of the fittest, natural selection, elitism)
5. 선택된 n개를 mating 시켜 다음 generation의 population을 생성 (crossover & mutation)
6. Stopping criteria (성능, generation 수 등)에 이를 때까지 2$\sim$5를 반복

위의 프로세스를 도식화하면 다음 그림과 같습니다:

![Fig2](https://jiryang.github.io/img/ga_process2.png "Process of Genetic Algorithm"){: width="80%"}{: .aligncenter}


GA와 Neuroevolution을 이해하는데 알아두면 좋은 용어들을 몇 가지 소개합니다:<br>
* Chromosome and Gene<br>
Chromosome(염색체)은 하나의 agent를 표현하며, ANN으로 만들어진 agent인 경우라면 weight의 vector입니다. Given task에 대한 하나의 solution이라고 볼 수도 있습니다.<br>
Gene은 chromosome을 이루는 각 weight(parameter, variable)를 말합니다.
* Population<br>
Population은 pool of agents로 이루어진 solution의 집합입니다. 일반적으로 GA에서는 population의 크기는 유지되면서 generation을 거칠 수록 individual chromosome은 '진화하게' (업데이트) 됩니다.

![Fig3](https://jiryang.github.io/img/population_chromosome_gene.png "Units in GA"){: width="80%"}{: .aligncenter}


* Fitness<br>
Fitness는 agent가 task를 얼마나 잘 수행하는지, solution이 얼마나 잘 들어맞는지(fit)를 측정하는 함수입니다 (딥러닝에서 loss function과 비슷하다 할 수 있을까요?) Population 내의 모든 agent가 task를 수행하고, fitness function으로 측정된 각자의 fitness value를 가지고 다음 generation에 살아남을 지 도태될 지를 정하는 역할을 합니다. Task performance를 정량적으로 나타낼 수 있는 함수여야 하며, 매 generation의 매 agent들에 대해 function 값을 구해야 하기 때문에 지나치게 복잡한 fitness function은 성능 저하의 원인이 될 수도 있으므로 주의해야 합니다.
* Elitism<br>
Elitism은 natural selection을 모방하여 fitness 값이 더 높은 agent들이 도태되지 않고 다음 generation으로 살아남아 '진화'가 이루어질 수 있도록 하는 '엘리트들을 뽑는' 프로세스를 말합니다. 특정 proportion을 pre-define 해놓고 쭉 사용할 수도 있고, 학습(진화)에 따라 variable로 가져갈 수도 있습니다. 순서대로 정해진 숫자만큼을 뽑는 deterministic 방식을 사용할 수도 있고, exploration을 높이기 위해 randomness가 가미된 roulette-wheel과 같은 probabilistic한 방식이나 rank selection, tournament selection과 같은 다양한 방식이 있습니다.
* Offspring<br>
이제 elitism으로 선택된 일부의 better fitted solution들이 있습니다. Next generation 수행을 위해선 다시 정해진 숫자의 population을 채워야 하는데요, 선택된 elite들을 '교배'시켜서 자손을 만들어내는 방식을 사용하고 이 자손을 offspring이라고 부릅니다.
* Crossover<br>
앞서 언급한 '교배'에 crossover와 mutation을 사용합니다. Crossover는 선택된 elite들 중에서 parent chromosome(s)을 뽑아서 offspring 시키는 방법인데요, one-point, multi-point, uniform, permutation-maintaining crossover 등 종류가 다양합니다. 아래 그림에서 몇가지 대표적인 crossover 방식의 예를 보실 수 있습니다. Crossover는 필요에 따라서 생략할 수도 있는 process 입니다.

![Fig4](https://jiryang.github.io/img/one_point_crossover.jpg "One-Point Crossover"){: width="80%"}{: .aligncenter}<br>
_One-Point Crossover_<br><br>
![Fig5](https://jiryang.github.io/img/multi_point_crossover.jpg "Multi-Point Crossover"){: width="80%"}{: .aligncenter}<br>
_Multi-Point Crossover_<br><br>
![Fig6](https://jiryang.github.io/img/uniform_crossover.jpg "Uniform Crossover"){: width="80%"}{: .aligncenter}<br>
_Uniform Crossover_<br><br>
![Fig7](https://jiryang.github.io/img/david_order_crossover.jpg "OX1 (Permutation-Maintaining) Crossover"){: width="80%"}{: .aligncenter}<br>
_OX1 (Permutation-maintaining) Crossover_<br>


* Mutation<br>
Mutation은 offspring에 randomness를 더해 exploration power를 키워주는 또 하나의 방법입니다. 딥러닝에서 mini-batch의 backpropagation을 통한 stochastic gradient descent (SGD)를 구해 조금씩 weight를 업데이트했던 것과 비슷하게, GA에서는 mutation을 통해 offspring의 weight를 조금씩 변화시켜 global optimum으로 향하는 솔루션이 나오는지 탐색합니다. Mutation rate는 constant 또는 variable로 설정할 수 있으며, 방식도 bit flip, random resetting, swap, scamble 등 다양합니다.

![Fig8](https://jiryang.github.io/img/bit_flip_mutation.jpg "Bit Flip Mutation"){: width="80%"}{: .aligncenter}<br>
_Bit Flip Mutation_<br><br>
![Fig9](https://jiryang.github.io/img/swap_mutation.jpg "Swap Mutation"){: width="80%"}{: .aligncenter}<br>
_Swap Mutation_<br><br>
![Fig10](https://jiryang.github.io/img/scramble_mutation.jpg "Scramble Mutation"){: width="80%"}{: .aligncenter}<br>
_Scramble Mutation_<br>

* Genotype & Phenotype<br>
Genotype은 '유전자형'이나 '인자형', phenotype은 '표현형' 또는 '형질형'으로 번역됩니다. GA의 context에서는 genetic encoding, 즉 agent(또는 solution)의 representation을 말하는 것입니다. 그러니깐 GA에서 genotype이란 chrmomosome의 형태를 말하는 것이라고 할 수 있고, phenotype이란 chromosome에 structure를 씌운, 그러니깐 chromosome의 weight로 구성된 ANN 형태를 나타내는 말입니다. 일반적인 GA에서는 network structure가 고정된 채로 weight만 학습하는 방식이라 phenotype이 큰 의미가 없지만 (대부분의 딥러닝도 마찬가지로 weight만 업데이트하지요), topological evolution 알고리즘인 NEAT에서는 phenotype이 동시에 학습된다는 특징이 있습니다. 좀 헷갈리실 수 있는데, 이 부분은 NEAT를 설명할 때 좀 더 명확해질테니 일단은 이 정도 설명으로 넘어가도록 하겠습니다.

![Fig11](https://jiryang.github.io/img/genotype_and_phenotype.png "Genotype and Phenotype"){: width="80%"}{: .aligncenter}


GA를 사용하면 다양한 문제들을 풀 수 있습니다. Generation을 거듭하면서 elitism, crossover, mutation을 이용한 exploration을 통해 current best solution들의 주변을 탐색하면서 조금씩 optimal에 가까운 쪽으로 접근하게 되는 개념입니다. Magic black box 처럼 생각될 수도 있지만 실은 딥러닝처럼 수렴을 유도하기 위해 retention rate, mutation rate 등 다양한 parameter 조정도 필요하며, 문제에 대해 충분히 이해하고 GA task를 디자인해야 좋은 결과를 얻을 수 있습니다. GA로 해결한 몇몇 재밌는 샘플들을 첨부하고 일단 마무리합니다:

![Fig12](https://jiryang.github.io/img/evolutionary_algorithm-1.gif "GA Sample #1"){: width="80%"}{: .aligncenter}


![Fig13](https://jiryang.github.io/img/ga_sample-2.gif "GA Sample #2"){: width="80%"}{: .aligncenter}


![Fig14](https://jiryang.github.io/img/ga_sample-3.gif "GA Sample #3"){: width="80%"}{: .aligncenter}



다음 포스트에서는 Neuroevolution of Augmenting Topologies에 대해 이야기하고, 이후 Uber AI의 neuroevolution research에 대해 계속 다루도록 하겠습니다.
