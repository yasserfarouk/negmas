# ANL/ANAC パラメータ履歴 (2010-2025)

このファイルでは、NegMAS の `genius/ginfo.py` に記載されている ANAC/ANL 各年度の環境パラメータを Markdown テーブルでまとめています。

| Year | linear | learning | multilateral | bilateral | reservation | discounting | uncertainty | elicitation | geniusweb | java  | multideal | known_opponent_ufun | known_opponent_reserved_value |
|------|--------|----------|--------------|-----------|-------------|-------------|-------------|-------------|-----------|-------|-----------|---------------------|------------------------------|
| 2010 | True   | False    | False        | True      | False       | True        | False       | False       | False     | True  | False     | False               | False                        |
| 2011 | True   | False    | False        | True      | False       | True        | False       | False       | False     | True  | False     | False               | False                        |
| 2012 | True   | False    | False        | True      | True        | True        | False       | False       | False     | True  | False     | False               | False                        |
| 2013 | True   | True     | False        | True      | True        | True        | False       | False       | False     | True  | False     | False               | False                        |
| 2014 | False  | False    | False        | True      | True        | False       | False       | False       | False     | True  | False     | False               | False                        |
| 2015 | True   | False    | True         | False     | True        | False       | False       | False       | False     | True  | False     | False               | False                        |
| 2016 | True   | False    | True         | False     | True        | False       | False       | False       | False     | True  | False     | False               | False                        |
| 2017 | True   | True     | True         | False     | True        | False       | False       | False       | False     | True  | False     | False               | False                        |
| 2018 | True   | True     | True         | False     | True        | True        | False       | False       | False     | True  | False     | False               | False                        |
| 2019 | True   | False    | False        | True      | True        | False       | True        | False       | False     | True  | False     | False               | False                        |
| 2020 | True   | False    | False        | False     | True        | False       | True        | True        | True      | True  | False     | False               | False                        |
| 2021 | True   | True     | False        | False     | True        | False       | True        | True        | True      | True  | False     | False               | False                        |
| 2022 | True   | True     | False        | False     | True        | False       | False       | True        | True      | False | False     | False               | False                        |
| 2023 | True   | True     | False        | False     | True        | False       | False       | True        | True      | False | False     | False               | False                        |
| 2024 | True   | False    | False        | True      | True        | False       | False       | False       | False     | False | False     | True                | False                        |
| 2025 | True   | False    | False        | True      | True        | False       | False       | False       | False     | False | True      | False               | False                        |

## 統計サマリー

- `linear` : 15/16 年で True (93.8%)
- `learning` : 6/16 年で True (37.5%)
- `multilateral` : 4/16 年で True (25.0%)
- `bilateral` : 8/16 年で True (50.0%)
- `reservation` : 14/16 年で True (87.5%)
- `discounting` : 6/16 年で True (37.5%)
- `uncertainty` : 3/16 年で True (18.8%)
- `elicitation` : 4/16 年で True (25.0%)
- `geniusweb` : 4/16 年で True (25.0%)
- `java` : 10/16 年で True (62.5%)
- `multideal` : 1/16 年で True (6.3%)
- `known_opponent_ufun` : 1/16 年で True (6.3%)
- `known_opponent_reserved_value` : 0/16 年で True (0.0%)

## 直近の 2025 年時点の値

- `linear`: True
- `learning`: False
- `multilateral`: False
- `bilateral`: True
- `reservation`: True
- `discounting`: False
- `uncertainty`: False
- `elicitation`: False
- `geniusweb`: False
- `java`: False
- `multideal`: True
- `known_opponent_ufun`: False
- `known_opponent_reserved_value`: False

## パラメータの提供方法

これらのパラメータは **session内で与えられるものではありません**。ANLのルールとして各年度ごとに固定された環境制約です。session内で与えられるのは：

- 効用関数（utility function）の具体的な値
- 相手の情報（相手の効用関数が既知の場合）
- 交渉のタイムアウトやステップ数などの実行パラメータ

ANLパラメータは、交渉環境の「ルール」を定義し、negotiatorの設計思想を決定づけるものです。

## 各パラメータの具体的な意味と影響

各パラメータが `True` の場合の具体的な意味と、negotiator設計への影響を説明します。これらは単なるTrue/Falseの設定で、環境の制約や機能を表します：

### `linear: True`
- **意味**: 効用関数が線形（LinearAdditiveUtilityFunction）固定
- **具体的な影響**: 効用関数が `u(x) = w1*x1 + w2*x2 + ...` の形で表現される。非線形関数（例: 指数関数や対数関数）は使用不可
- **negotiatorへの影響**: 線形計画法で最適解を計算可能。効用空間が凸集合になるため、予測可能性が高い

### `learning: True`
- **意味**: 相手の戦略や効用関数を学習しながら交渉可能な環境である
- **具体的な影響**: 過去の交渉履歴や相手応答を使って戦略を適応的に変えることが想定される
- **negotiatorへの影響**: 学習型アルゴリズム（Q-learningや推定モデルなど）を利用すると有利になる可能性がある
- **注意**: `learning: False` は「学習型戦略がその年の問題仕様で重視されない」ことを示すだけです。エージェント実装で学習機能を完全に禁止するわけではありませんが、単発の交渉や履歴のない環境では意味が薄い、という設定です。

### `reservation: True`
- **意味**: 予約値（reserved value）が存在し、最低受容効用が設定されている
- **具体的な影響**: 効用が予約値未満の合意は拒否される。交渉が決裂する可能性が高くなる
- **negotiatorへの影響**: 予約値を前提とした戦略を設計し、譲歩の下限と合意拒否のリスクを考慮する必要がある
- **注意**: `reservation` はエージェント側が作るものではなく、環境設定として与えられる値の有無を示すフラグです。True/Falseはその年のANL条件で決まります。

### `discounting: True`
- **意味**: 時間割引が適用され、時間が経過するほど効用が減少する構成要素がある
- **具体的な影響**: 交渉が進むほど得られる効用が下がるので、遅延すると損失になる
- **negotiatorへの影響**: たとえ高効用の提案を後回しにする場合でも、時間経過で価値が下がることを考慮する必要がある
- **注意**: Trueの場合、時間推移（割引係数）の影響がある環境であることを示します。Falseなら時間経過による効用低下はありません。

### `uncertainty: True`
- **意味**: 相手の効用関数や行動に不確実性がある
- **具体的な影響**: 相手の真の効用関数が不明で、推定する必要がある
- **negotiatorへの影響**: 確率モデルを使用し、リスク評価を行う。ベイズ更新などで相手の情報を学習

### `elicitation: True`
- **意味**: 相手から効用関数に関する情報を質問して引き出せる
- **具体的な影響**: 交渉中に相手に提案を送り、その提案に対する相手の効用値を尋ねることができる
- **negotiatorへの影響**: 質問戦略を設計し、得られた情報で戦略を修正。情報収集コストを考慮
- **注意**: 相手のufunに直接アクセスするのではなく、提案に対する反応を通じて情報を得る

### `multilateral: True`
- **意味**: 3者以上の多者間交渉
- **具体的な影響**: 2者ではなく複数者との同時交渉。合意には全員の同意が必要
- **negotiatorへの影響**: 連立形成や投票を考慮した戦略。個別交渉だけでなくグループ調整が必要
- **注意**: ANLのほとんどの年でFalse。negotiatorは二者間を想定して設計されることが多い

### `known_opponent_ufun: True`
- **意味**: 相手の効用関数が完全に既知
- **具体的な影響**: 相手の効用関数パラメータが事前に与えられ、推定が不要になる
- **negotiatorへの影響**: 最適戦略を直接計算しやすくなる。相手の評価基準を利用した提案が可能
- **注意**: `known_opponent_ufun: True` は `uncertainty: False` と一致しやすいが、`uncertainty: False` だけでは相手の効用関数が既知であるとは限りません。
### `known_opponent_reserved_value: True`
- **意味**: 相手の予約値が既知
- **具体的な影響**: 相手の最低受容効用が分かる
- **negotiatorへの影響**: 相手の譲歩限界を把握し、ギリギリの提案が可能

### `multideal: True`
- **意味**: 複数のディールを同時に扱う
- **具体的な影響**: 単一ディールではなく、複数案件をパッケージで交渉
- **negotiatorへの影響**: 複数案件のトレードオフを計算。全体効用を最大化する提案が必要
- **注意**: ANLでTrueになったのは2025年のみ。negotiatorは単一ディールを想定して設計されることが多い

これらのパラメータはANLの各年度で異なる組み合わせで設定され、Trueのものがその年の「ルール」として有効になります。negotiatorはこれらの制約下で動作するよう設計されます。例えば2025年は `linear: True` なので、全員が線形効用関数を使用します。`multilateral` や `multideal` はほとんどの年でFalseなので、negotiatorは二者間・単一ディールを想定して設計されることが一般的です。
