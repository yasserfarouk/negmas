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

## 用語説明

- `linear`: 効用関数が線形かどうか
- `learning`: 相手の戦略/効用を学習することが許可されているか
- `multilateral`: 多者間交渉かどうか
- `bilateral`: 二者交渉かどうか
- `reservation`: 予約価（reserved value）があるかどうか
- `discounting`: 時間割引があるかどうか
- `uncertainty`: 不確実性があるかどうか
- `elicitation`: 効用関数のエリシテーション（相手から情報取得）があるかどうか
- `geniusweb`: GeniusWeb による実行環境かどうか
- `java`: Java 実装を想定しているかどうか
- `multideal`: 複数ディールを同時に扱うかどうか
- `known_opponent_ufun`: 相手の効用関数が既知かどうか
- `known_opponent_reserved_value`: 相手の留保価格が既知かどうか
