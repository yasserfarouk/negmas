# AlmostEqualAgent の関数一覧と処理フロー

対象コード: [agent.py](/Users/fujimotokou/dev/ANL/tutorial/negmas/coding_agents/scml_oneshot_winners/2025/team_284/agent.py)

## 概要

`AlmostEqualAgent` は `OneShotSyncAgent` を継承した SCML OneShot 用エージェントです。

基本方針は次の 3 つです。

- 必要数量にできるだけ近い契約組み合わせを選ぶ
- 価格も見るが、数量一致をかなり強く重視する
- 後半は、過去に多く契約できた相手へ配分を集中する

名前どおり、通常時は相手へ数量をほぼ均等に配分します。ただし、後半になると `total_agreed_quantity` を使い、契約実績のある相手へ寄せる動きが入ります。

## ファイル内の関数一覧

### 外部補助関数

| 関数 | 種類 | 役割 |
| --- | --- | --- |
| `distribute()` | ファイル内の補助関数 | 合計数量 `q` を `n` 個の交渉相手へ分配する |
| `powerset()` | ファイル内の補助関数 | 相手集合の全部分集合を列挙する |

### `AlmostEqualAgent` のインスタンス関数

| 関数 | 呼ばれるタイミング | 役割 |
| --- | --- | --- |
| `__init__()` | エージェント生成時 | 戦略パラメータを保存する |
| `init()` | AWI 設定後の初期化時 | 契約実績メモリを初期化する |
| `distribute_needs()` | 初手提案・反対提案の作成時 | 現在の必要数量を相手ごとに配分する |
| `on_negotiation_success()` | 契約成立時 | 相手ごとの成立数量を記録する |
| `first_proposals()` | 各交渉の最初 | 全相手への初手提案を作る |
| `counter_all()` | 相手オファー受信時 | 全交渉をまとめて評価し、受諾・拒否・反対提案を決める |
| `_step_and_price()` | 提案作成時 | 提案に使うステップと価格を決める |

## 外部補助関数のロジック

### `distribute(q, n, mx=None, equal=True, concentrated=False, allow_zero=False, concentrated_idx=[])`

合計数量 `q` を `n` 人の交渉相手に分ける関数です。このエージェントの数量配分の中心です。

引数の意味:

- `q`: 分配したい合計数量
- `n`: 分配先の数
- `mx`: 1 相手あたりの最大数量。`None` なら上限なし
- `equal`: `True` ならできるだけ均等に分ける
- `concentrated`: `True` なら特定順序の相手に寄せて分配する
- `allow_zero`: `True` なら 0 個の相手を許す
- `concentrated_idx`: 集中配分したい相手の index リスト

通常分配の流れ:

1. `q` と `n` を整数化する。
2. `mx` があり、`q > mx * n` なら、配り切れないので `q = mx * n` に切り詰める。
3. `q < n` なら、`q` 人に 1 個ずつ、残りは 0 個にしてシャッフルする。
4. `q == n` なら、全員に 1 個ずつ配る。
5. `allow_zero=False` の場合、`equal=True` ならまず `q // n` 個ずつ配る。`equal=False` なら最低 1 個ずつ配る。
6. 余りを `numpy.random.choice()` でランダムな相手に追加する。

集中配分 `concentrated=True` の流れ:

1. `mx` が必須。`assert mx is not None` で確認する。
2. まず全 `n` 人に 1 個ずつ配る。
3. 残り数量を、各相手が `mx` に達するまで順番に 1 個ずつ足す。
4. できた数量リスト `lst` を、`concentrated_idx` の順番で `result[target_idx]` に割り当てる。
5. `concentrated_idx` に含まれない相手には、この分岐では基本的に 0 が残る。

このコードでの使われ方:

- 通常時は `equal=True`, `mx=3`, `allow_zero=False` で、ほぼ均等配分する。
- 後半は `concentrated=True` にして、過去の契約量が多い相手へ優先的に数量を寄せる。

注意点:

- `concentrated=True` のとき、`concentrated_idx` に入っていない相手は `result` 上で 0 のままになりやすいです。
- `q < n` の場合は、`allow_zero=False` でも 0 が入ります。数量が相手数より少ないためです。

### `powerset(iterable)`

相手集合の全部分集合を作る関数です。

実装:

```python
def powerset(iterable):
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
```

例:

```python
powerset(["A", "B", "C"])
```

は概念的には次を返します。

```text
()
("A",)
("B",)
("C",)
("A", "B")
("A", "C")
("B", "C")
("A", "B", "C")
```

このコードでの使われ方:

- `counter_all()` で、どの相手のオファーを同時に受け入れるかを全探索する。
- 各部分集合について、合計数量・合計価格・過不足ペナルティ・最終 `QP_score` を計算する。

注意点:

- 相手が `n` 人なら `2^n` 通りになります。
- 相手が多い環境では計算量が急に重くなります。

## `AlmostEqualAgent` の関数ごとのロジック

### `__init__(...)`

役割:

- エージェントの行動パラメータを保存する。
- 最後に `super().__init__(*args, **kwargs)` を呼び、`OneShotSyncAgent` 側の初期化を行う。

保存する主なパラメータ:

- `self.equal_distribution`
  - 均等配分するかどうか。
  - デフォルトは `True`。

- `self.over_buying`
  - 買い手側のとき、必要量よりどれだけ多めに買おうとするか。
  - デフォルトは `0.2` なので、買い手では `needs * 1.2` を目安にする。

- `self.quantity_price_balance`
  - 数量評価と価格評価の重み。
  - デフォルトは `0.85`。
  - `QP_score` の 85% が数量一致、15% が価格評価になる。

- `self.buyer_QP_score_threshold`
  - 買い手側で受諾するための基準値。

- `self.seller_QP_score_threshold`
  - 売り手側で受諾するための基準値。

ロジック上の意味:

- このエージェントは、価格最適化よりも数量一致を重視します。
- 買い手では少し多めに買うことで、不足リスクを避けようとします。

### `init()`

役割:

- シミュレーション環境 `awi` が使えるようになった後に呼ばれる初期化処理です。
- 相手ごとの累積契約数量を記録する辞書を作ります。

処理:

```python
self.total_agreed_quantity = {
    k: 0
    for k in (
        self.awi.my_consumers
        if self.awi.my_suppliers == ["SELLER"]
        else self.awi.my_suppliers
    )
}
self.is_seller = self.awi.my_suppliers == ["SELLER"]
return super().init()
```

ロジック:

1. 自分が最初の生産レベルにいる場合、`my_suppliers == ["SELLER"]` になる。
2. その場合は、自分の取引相手として `my_consumers` を記録対象にする。
3. それ以外の場合は、`my_suppliers` を記録対象にする。
4. 各相手の累積契約数量を 0 で初期化する。
5. `self.is_seller` に、自分が売り手かどうかの判定を保存する。

この状態の使い道:

- `first_proposals()` と `counter_all()` の後半集中配分で、契約実績の多い相手を優先するために使います。

### `distribute_needs(t, mx=None, equal=None, allow_zero=None, concentrated=False, concentrated_ids=[])`

役割:

- 現在の必要数量を、現在交渉中の相手へ分配します。
- `distribute()` のラッパーであり、AWI から必要量と相手リストを取得する部分を担当します。

処理対象:

- 仕入れ側:
  - `needs = self.awi.needed_supplies`
  - `all_partners = self.awi.my_suppliers`

- 販売側:
  - `needs = self.awi.needed_sales`
  - `all_partners = self.awi.my_consumers`

ロジック:

1. `equal` が `None` なら `True` にする。
2. `allow_zero` が `None` なら `self.awi.allow_zero_quantity` を使う。
3. 仕入れ側と販売側を順に処理する。
4. `all_partners` のうち、現在まだ交渉中の相手だけを `partners` に入れる。
5. `concentrated_ids` に含まれる相手について、対応する index を `concentrated_idx` に入れる。
6. `needs <= 0` なら、その側の全相手への数量を 0 にする。
7. 買い手側なら `needs * (1 + self.over_buying)` に増やす。
8. `distribute()` で数量を配る。
9. `dict[partner_id, quantity]` として返す。

買い手判定:

```python
is_buyer = all_partners == self.awi.my_suppliers
```

つまり、仕入れ先に対して交渉しているときは買い手です。

重要な点:

- 買い手だけ `over_buying` を適用します。
- 売り手側では必要販売量をそのまま使います。
- `concentrated=True` のときは、契約実績が多い相手を前に並べて配分を寄せます。

### `on_negotiation_success(contract, mechanism)`

役割:

- 契約成立時に呼ばれます。
- 契約相手と成立数量を記録します。

処理:

```python
super().on_negotiation_success(contract, mechanism)
partner_id = [p for p in contract.partners if p != self.id][0]
self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]
```

ロジック:

1. 親クラスの成功処理を呼ぶ。
2. `contract.partners` から自分以外の相手 ID を取り出す。
3. `contract.agreement["quantity"]` を、その相手の累積成立数量に足す。

この関数の意味:

- 相手ごとの「契約できた実績」を学習します。
- 後半の集中配分では、この値が大きい相手が優先されます。

注意点:

- `self.total_agreed_quantity` に存在しない相手 ID が来ると `KeyError` になります。
- 通常は `init()` で対象相手を初期化しているため問題になりにくいですが、相手集合が途中で想定外に変わる環境では注意が必要です。

### `first_proposals()`

役割:

- 交渉開始時に、各相手へ送る最初の提案を作ります。

返す形式:

```python
{
    partner_id: (quantity, time, unit_price) | None
}
```

処理の流れ:

1. `_step_and_price(best_price=True)` で、現在ステップと自分に最良の価格を取得する。
2. 自分が売り手側なら `my_consumers`、そうでなければ `my_suppliers` から、現在交渉中の相手を取り出す。
3. 後半条件に入っていれば、契約実績順に `concentrated_ids` を作る。
4. 後半条件に入っていなければ、通常の均等配分を使う。
5. 分配結果を `(q, s, p)` の提案に変換する。
6. `q == 0` かつ 0 数量が許されない場合は `None` を返す。

後半集中の条件:

- `level == 0` の場合:

```python
self.awi.current_step > max(50, self.awi.n_steps * 0.5)
```

- `level == 1` の場合:

```python
self.awi.current_step > max(100, self.awi.n_steps * 0.75)
```

集中配分時の処理:

1. `total_agreed_quantity` が大きい相手順に並べる。
2. `distribute_needs(..., concentrated=True, concentrated_ids=concentrated_ids)` を呼ぶ。
3. `mx=3` で 1 相手あたり最大 3 個に制限する。

ロジック上の意味:

- 序盤は広く均等に交渉します。
- 後半は「これまで契約できた相手」に数量を寄せます。
- 初手価格は `best_price=True` なので、自分に有利です。

### `counter_all(offers, states)`

役割:

- このエージェントの中核です。
- 複数相手から来たオファーをまとめて見て、受諾・終了・反対提案を一括で決めます。

入力:

- `offers`: `dict[partner_id, offer]`
  - 相手ごとの現在オファー。
  - offer は `(quantity, time, unit_price)`。

- `states`: `dict[partner_id, SAOState]`
  - 相手ごとの交渉状態。
  - `relative_time` などを使います。

返す形式:

```python
{
    partner_id: SAOResponse(...)
}
```

全体の処理フロー:

1. 現在ステップ以外のオファーを `future_partners` に分ける。
2. `offers` から現在ステップ以外のオファーを除外する。
3. 仕入れ側と販売側を別々に処理する。
4. 現在オファーを出している相手集合 `partners` を作る。
5. 受け入れ不要な相手への応答 `unneeded_response` を作る。
6. `powerset(partners)` で相手の全組み合わせを列挙する。
7. 各組み合わせについて `QP_score` を計算する。
8. 最も `QP_score` が高い組み合わせを選ぶ。
9. 時間で緩和した閾値 `adjusted_threshold` と比較する。
10. 閾値以上なら、その組み合わせを受諾する。
11. 閾値未満なら、全相手に反対提案を返す。

#### 現在ステップ以外のオファー除外

```python
future_partners = {
    k for k, v in offers.items() if v[TIME] != self.awi.current_step
}
offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
```

SCML OneShot では現在ステップの取引が重要なので、違う納期のオファーは受諾候補から外します。

#### 仕入れ側・販売側の分離

次の 2 セットを順番に処理します。

```python
(
    self.awi.needed_supplies,
    self.awi.my_suppliers,
    self.awi.current_input_issues,
)
(
    self.awi.needed_sales,
    self.awi.my_consumers,
    self.awi.current_output_issues,
)
```

これにより、買う交渉と売る交渉を別々に評価します。

#### `unneeded_response`

相手のオファーを受けないときの応答です。

- 0 数量が許されない場合:
  - `END_NEGOTIATION`

- 0 数量が許される場合:
  - `(0, current_step, price)` を反対提案

```python
unneeded_response = (
    SAOResponse(ResponseType.END_NEGOTIATION, None)
    if not self.awi.allow_zero_quantity
    else SAOResponse(
        ResponseType.REJECT_OFFER, (0, self.awi.current_step, price)
    )
)
```

#### 受諾組み合わせの全探索

```python
plist = list(powerset(partners))[::-1]
```

`partners` の全部分集合を作り、逆順にします。

各 `partner_ids` について、以下を計算します。

- `offered_quantity`
  - その組み合わせを全部受けた場合の合計数量

- `quantity_diff`
  - `offered_quantity - needs`
  - 正なら過剰、負なら不足

- `total_price`
  - `unit_price * quantity` の合計

- `penalty`
  - 過不足による損失の近似

- `P_score`
  - 価格面の評価

- `normalized_P_score`
  - 価格評価を 0 から 1 に丸めた値

- `normalized_quantity`
  - 数量一致度

- `QP_score`
  - 数量と価格の合成評価

#### ペナルティ計算

売り手の場合:

```python
if offered_quantity > needs:
    penalty += (offered_quantity - needs) * self.awi.current_shortfall_penalty
if offered_quantity < needs:
    penalty += (needs - offered_quantity) * self.awi.current_disposal_cost
```

買い手の場合:

```python
if offered_quantity < needs:
    penalty += (needs - offered_quantity) * self.awi.current_shortfall_penalty
if offered_quantity > needs:
    penalty += (offered_quantity - needs) * self.awi.current_disposal_cost
```

意味:

- 売り手で過剰に売ると、作れない分の不足ペナルティが発生する。
- 売り手で売れ残ると、廃棄コストが発生する。
- 買い手で不足すると、生産できない分の不足ペナルティが発生する。
- 買い手で買いすぎると、余剰分の廃棄コストが発生する。

#### 価格スコア `P_score`

```python
P_score = (
    (total_price - penalty) if is_selling else -(total_price - penalty)
)
```

意味:

- 売り手は `total_price` が高いほど良い。
- 買い手は `total_price` が低いほど良いので、符号を反転する。
- どちらも `penalty` は悪い要素として扱う。

#### 正規化価格スコア `normalized_P_score`

```python
normalized_P_score = (
    0.5 + (P_score / 1000)
    if P_score > 0
    else 0.5 - (abs(P_score) / 1000)
)
normalized_P_score = max(0, min(1, normalized_P_score))
```

意味:

- `P_score == 0` なら 0.5 付近。
- 良い価格なら 0.5 より上。
- 悪い価格なら 0.5 より下。
- 最後に 0 から 1 にクリップする。

注意点:

- `1000` は固定値なので、価格スケールに依存します。
- 正確な正規化というより、簡易的なスコア化です。

#### 数量スコア `normalized_quantity`

```python
normalized_quantity = (
    1.0 - abs(quantity_diff) / needs if needs != 0 else 0
)
```

意味:

- 必要量にぴったりなら 1.0。
- 差が大きいほど下がる。
- `needs == 0` の場合は 0。

注意点:

- `abs(quantity_diff) > needs` の場合、負の値になる可能性があります。
- 価格スコアと違い、0 から 1 へのクリップはしていません。

#### 合成スコア `QP_score`

```python
QP_score = (
    self.quantity_price_balance * normalized_quantity
    + (1 - self.quantity_price_balance) * normalized_P_score
)
```

デフォルトでは:

- 数量: 85%
- 価格: 15%

つまり、多少価格が悪くても、数量が合う組み合わせを選びやすい設計です。

#### ベスト組み合わせの更新

```python
if QP_score > best_QP_score:
    best_QP_score = QP_score
    best_quantity = quantity_diff
    best_indx = i
elif QP_score == best_QP_score:
    if (is_selling and quantity_diff < best_quantity) or (
        not is_selling
        and quantity_diff > best_quantity
        and quantity_diff <= 0
    ):
        best_quantity = quantity_diff
        best_indx = i
```

同点時の方針:

- 売り手では、より過剰を避ける方向を選ぶ。
- 買い手では、不足側の中で、より不足が少ない方向を選ぶ。

#### 受諾閾値

売り手と買い手で別の閾値を使います。

```python
QP_score_threshold = (
    self.seller_QP_score_threshold
    if is_selling
    else self.buyer_QP_score_threshold
)
```

さらに、時間が進むほど閾値を下げます。

```python
relative_time = min(state.relative_time for state in states.values())
adjusted_threshold = QP_score_threshold * (1 - 0.5 * relative_time)
```

意味:

- 序盤は厳しめに判断する。
- 終盤は妥協して受け入れやすくなる。

#### 閾値以上なら受諾

```python
if best_indx >= 0 and best_QP_score >= adjusted_threshold:
    partner_ids = plist[best_indx]
    others = list(partners.difference(partner_ids).union(future_partners))
```

処理:

1. ベスト組み合わせの相手には `ACCEPT_OFFER` を返す。
2. それ以外の相手には `unneeded_response` を返す。
3. ただし不足数量がある場合は、他の相手に追加提案を出す。

不足がある場合:

```python
if best_quantity < 0 and len(others) > 0:
    shortage = -best_quantity
```

この `shortage` を `others` に再分配して、追加の反対提案を返します。

後半なら、この追加提案も契約実績の多い相手へ集中配分します。

#### 閾値未満なら反対提案

ベスト組み合わせが閾値に届かない場合は、現在のオファーを受けず、改めて必要数量を相手に配分して反対提案します。

処理:

1. `partners = partners.union(future_partners)` で未来納期の相手も戻す。
2. 買い手側なら `needs * (1 + over_buying)` に増やす。
3. 後半条件に入っていれば集中配分する。
4. 通常時は均等配分する。
5. 各相手に `REJECT_OFFER` と反対提案 `(q, current_step, price)` を返す。

### `_step_and_price(best_price=False)`

役割:

- 提案に使う `time` と `unit_price` を決めます。

処理:

1. `s = self.awi.current_step` を取得する。
2. `self.awi.is_first_level` から売り手かどうかを判定する。
3. 売り手なら `current_output_issues`、買い手なら `current_input_issues` を使う。
4. 価格 issue から `pmin`, `pmax` を取得する。
5. `best_price=True` なら、自分に最良の価格を返す。
6. それ以外は、基本的に `pmin` または `pmax` の二択をランダムに選ぶ。

`best_price=True` の価格:

- 売り手: `pmax`
- 買い手: `pmin`

`best_price=False` の価格:

- `level == 0` でも `level == 1` でも、実質的には `pmax` と `pmin` の 50% ランダムです。
- `level` がそれ以外なら `random.randint(pmin, pmax)` を使います。

注意点:

- 現在のコードでは、`level == 0` と `level == 1` の時間条件分岐はありますが、分岐の中身は同じです。
- 価格戦略は細かい譲歩カーブではなく、極端価格のランダム選択です。

## 全体フロー

### 1. エージェント生成

`__init__()` が呼ばれ、以下の戦略パラメータが保存されます。

- 均等配分するか
- 買い手でどれだけ多めに買うか
- 数量と価格の重み
- 買い手・売り手それぞれの受諾閾値

### 2. シミュレーション初期化

`init()` が呼ばれます。

ここで、相手ごとの累積成立数量 `total_agreed_quantity` を 0 で初期化します。

### 3. 交渉開始時の初手提案

`first_proposals()` が呼ばれます。

通常時:

1. 自分に最良の価格を選ぶ。
2. 必要数量を交渉中の相手へ均等配分する。
3. 各相手へ `(quantity, current_step, best_price)` を提案する。

後半:

1. 過去に多く契約できた相手を `total_agreed_quantity` で並べる。
2. その相手順に集中配分する。
3. 各相手へ提案を出す。

### 4. 相手オファーの一括評価

相手からオファーが届くと `counter_all()` が呼ばれます。

流れ:

1. 現在ステップ以外のオファーを除外する。
2. 仕入れ側と販売側を分ける。
3. 相手の全組み合わせを `powerset()` で列挙する。
4. 各組み合わせについて、数量・価格・ペナルティを計算する。
5. `QP_score` が最大の組み合わせを選ぶ。
6. 時間で緩和した閾値を超えたら受諾する。
7. 超えなければ反対提案を返す。

### 5. 受諾した場合

`counter_all()` 内で、選んだ組み合わせの相手に `ACCEPT_OFFER` を返します。

選ばれなかった相手には:

- `END_NEGOTIATION`
- または 0 数量の反対提案

を返します。

ただし、受諾しても数量不足が残る場合は、残り相手に不足分を再分配して追加提案します。

### 6. 受諾しない場合

`counter_all()` は、必要数量を再計算し、相手へ反対提案を返します。

通常時:

- 均等配分

後半:

- 契約実績の多い相手へ集中配分

買い手側:

- `over_buying` により必要数量より少し多めに提案

### 7. 契約成立後

`on_negotiation_success()` が呼ばれます。

ここで相手ごとの契約数量を `total_agreed_quantity` に加算します。

この記録は、後半の集中配分で使われます。

## このエージェントの特徴

### 数量優先

`quantity_price_balance = 0.85` なので、評価の大部分は数量一致で決まります。

価格よりも、「必要量に近い契約セットを作ること」を優先します。

### 複数交渉をまとめて判断

個別の相手ごとに受諾判断をするのではなく、`powerset()` で相手の組み合わせを総当たりします。

そのため、OneShot で重要な「複数契約の合計数量」を見た判断ができます。

### 後半は契約実績に寄せる

`total_agreed_quantity` により、契約が成立しやすかった相手を覚えます。

後半は、その相手に数量を多く振り、成立確率を上げようとします。

### 買い手は少し多めに買う

買い手側では `over_buying` により、必要数量より多めに提案します。

これは不足リスクを避けるための設計です。

### 価格戦略は単純

価格は主に `pmin` と `pmax` の二択です。

強みの中心は価格交渉ではなく、数量配分・組み合わせ選択・後半集中配分です。

## 注意点

### `powerset()` は相手数が多いと重い

相手が `n` 人いると、候補は `2^n` 個になります。

OneShot の相手数が少ない前提なら問題になりにくいですが、多数相手では計算負荷が上がります。

### 後半条件は短いシミュレーションでは発動しにくい

後半集中の条件に `max(50, n_steps * 0.5)` や `max(100, n_steps * 0.75)` が使われています。

そのため、`n_steps=8` のような短いローカル実験では、集中配分がほぼ発動しません。

### `normalized_quantity` は負になる可能性がある

`abs(quantity_diff) > needs` のとき、`1.0 - abs(quantity_diff) / needs` は負になります。

意図的に強く減点しているとも読めますが、0 から 1 に収めたいならクリップが必要です。

### `concentrated=True` の分配はかなり強く偏る

`concentrated_idx` に含まれない相手は 0 になりやすいです。

後半に実績相手へ寄せる意図には合っていますが、未実績相手への探索は減ります。

## 一言でまとめると

`AlmostEqualAgent` は、必要数量に近い相手の組み合わせを全探索で選ぶ数量優先エージェントです。通常時はほぼ均等配分し、後半は契約実績のある相手に数量を集中させます。価格戦略は単純ですが、複数交渉の合計数量をまとめて評価する点が中心的な強みです。
