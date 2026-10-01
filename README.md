# Minimising Edge Additions for Joint $k$-Core Activation of Query Nodes

본 저장소는 여러 query node가 최종 $k$-core에 포함되도록 그래프에 간선을 추가하는 방법의 구현 및 실험 코드를 제공한다.

목표는 가능한 적은 수의 간선을 추가하여 모든 query node $Q$가 최종 $k$-core $C_k(G_A)$에 포함되도록 하는 것이다.

## 파일 구성

```text id="ql9h7c"
.
├── code/
│   ├── query_kcore_greedy_anchored.py
│   └── run_edgelist_experiments.py
│
├── dataset/
│   ├── facebook.txt.gz
│   ├── brightkite.txt.gz
│   ├── twitter.txt.gz
│   └── ...
│
└── experiment_results/
```

- `code/query_kcore_greedy_anchored.py`: 제안 방법과 비교 방법을 구현한다.
- `code/run_edgelist_experiments.py`: 데이터셋을 이용하여 실험을 수행하고 결과를 저장한다.
- `dataset/`: 실험에 사용하는 그래프 데이터셋을 저장한다. facebook, brightkite, twitter 등의 데이터셋을 포함한다.

## 실행 환경

Python과 NetworkX를 사용한다.

NetworkX는 다음 명령어로 설치한다.

```bash id="81m0tp"
pip install networkx
```

## 데이터셋 형식

입력 그래프는 무방향 edge-list 형식을 사용한다.

예시는 다음과 같다.

```text id="27oyrt"
0 1
0 2
1 2
1 5
```

일반 텍스트 파일과 `.gz` 형식을 지원한다.

Self-loop는 제거하며, 중복 간선은 하나의 간선으로 처리한다.

## 비교 방법

실험에서는 다음 세 가지 방법을 비교한다.

- `proposed`: 제안하는 GAIN 방법
- `query-query-only`: query node 사이에만 간선을 추가하는 방법
- `query-kcore-only`: query node와 초기 $k$-core의 node 사이에만 간선을 추가하는 방법

## 제안 방법

각 query node $q\in Q$에 대해 현재 부족한 support의 수 $R(q)$를 관리한다.

제안 방법은

$C_{k-1}(G_A)\setminus C_k(G_A)$

영역에서 연결된 node끼리 candidate component를 구성하고 그 내부에서 onion layer

$L_1,L_2,\ldots,L_h$

를 구성한다.

각 layer $L_j$에 대해 candidate를 다음과 같이 정의한다.

$S_i^j=\bigcup_{r=j}^{h}L_r$

각 candidate는 다음 score를 이용하여 평가한다.

$\mathrm{Score}_Q(S_i^j)=G_Q(S_i^j)/N_S(S_i^j)$

여기서 $G_Q(S_i^j)$는 candidate 활성화를 통해 얻는 query gain이며, $NS(S_i^j)$는 $S_i^j$를 활성화하기 위해 필요한 지지의 수를 의미한다.

Score의 분모는 실제로 추가되는 간선의 수가 아니라 필요한 지지의 수를 사용한다.

## Query 선택

두 가지 query 선택 방법을 사용한다.

- `boundary-biased`: $k$-core 경계에 가까운 node를 우선적으로 선택한다.
- `random`: $V(G)\setminus C_k(G)$에서 무작위로 선택한다.

기본값은 `boundary-biased`이다.

Boundary-biased 방식에서는

$C_{k-1}(G)\setminus C_k(G)$

주변에 위치한 node를 우선적으로 query candidate로 사용한다.

## 실험 실행

기본 실행 형식은 다음과 같다.

이때 data를 제외한 항목은 default값이 정해져 있어 반드시 있어야 하는 것은 아니다.

```bash id="81u9fw"
python code/run_edgelist_experiments.py \
    --data dataset/facebook.txt.gz \
    --k-values 10,15 \
    --query-sizes 20,30,40 \
    --repeats 3 \
    --algorithm all \
    --query-selection boundary-biased \
    --boundary-candidate-fraction 0.5 \
    --seed 42
```

## 주요 옵션

```text id="6pjxjt"
--data                         입력 edge-list 파일
--k-values                     k 값
--query-sizes                  query set 크기
--repeats                      반복 횟수
--algorithm                    실행할 알고리즘
--query-selection              query 선택 방법
--boundary-candidate-fraction  candidate 비율
--seed                         random seed
```

사용 가능한 알고리즘은 다음과 같다.

```text id="g5dbpw"
all
proposed
query-query-only
query-kcore-only
```

## 결과

실험 결과는 기본적으로 다음 디렉터리에 저장한다.

```text id="34bv3u"
experiment_results/
```

각 실험마다 별도의 디렉터리를 생성하며, 주요 결과는 다음 파일에 저장한다.

```text id="41a7mm"
summary.csv
```

## 재현성

동일한 $(k,|Q|)$ 조건에서는 모든 비교 방법이 동일한 query set $Q$를 사용한다.

다음과 같이 seed를 지정하면 동일한 query sampling을 재현할 수 있다.

```bash id="wjvls3"
--seed 42
```

## 결과 검증

각 실험이 종료된 후 ordinary $k$-core를 다시 계산하여

$Q\subseteq C_k(G_A)$

를 만족하는지 확인한다.

모든 query node가 최종 $k$-core에 포함된 경우에만 성공한 실험으로 처리한다.
