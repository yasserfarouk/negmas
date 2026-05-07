# OneShotAgent の説明フローとインスタンス関数

対象クラス:

- [scml/oneshot/agent.py](/Users/fujimotokou/.pyenv/versions/3.10.0/lib/python3.10/site-packages/scml/oneshot/agent.py:54)

## 概要

`OneShotAgent` は `scml` パッケージの OneShot 用エージェント基底クラスです。

定義は次です。

```python
class OneShotAgent(SAOController, Entity, ABC):
```

つまり、

- `scml` の OneShot 競技用エージェント
- `negmas` の `SAOController` を土台にした制御クラス
- 各相手との交渉をまとめて持つコントローラ

です。

このクラス自体はかなり薄いラッパーで、主な役割は次の 3 つです。

- SCML の AWI を扱いやすくする
- 各相手との negotiator をまとめて管理する
- 子クラスが `propose()` / `respond()` などを実装できるようにする

## 継承関係

```python
OneShotAgent
  ├─ OneShotSyncAgent
  ├─ OneShotSingleAgreementAgent
  └─ OneShotIndNegotiatorsAgent
```

よく使うのは次です。

- `OneShotAgent`
  - 相手ごとに `propose()` / `respond()` を書く基本形

- `OneShotSyncAgent`
  - 全相手のオファーをまとめて `counter_all()` で処理する同期型

- `OneShotSingleAgreementAgent`
  - 最大 1 件だけ合意したい型

`AlmostEqualAgent` は `OneShotAgent` そのものではなく、`OneShotSyncAgent` を継承しています。

## 全体フロー

### 1. 生成

`__init__(owner=None, ufun=None, name=None)` が呼ばれます。

ここで `SAOController` の初期化が行われ、デフォルト negotiator として `ControlledSAONegotiator` が設定されます。

```python
super().__init__(
    default_negotiator_type=ControlledSAONegotiator,
    default_negotiator_params=None,
    auto_kill=False,
    name=name,
    preferences=ufun,
)
```

その後、

- `self._awi`
- `self._owner`

がセットされます。

### 2. アダプタ接続

実行時には `connect_to_oneshot_adapter()` または `connect_to_2021_adapter()` が呼ばれ、SCML 世界側の owner と AWI が接続されます。

ここで、

- `self._owner`
- `self._awi`
- `self.utility_function`

が実環境に紐づきます。

### 3. 初期化

`init()` が呼ばれます。

これはフックです。基底クラスでは何もしません。子クラス側で、

- 変数初期化
- ログ用状態初期化
- 相手情報の初期化

などを書く場所です。

### 4. 毎ステップ前

`before_step()` が呼ばれます。

基底クラスでは `pass` です。必要なら子クラスで、

- 日次の内部状態更新
- 統計のリセット
- 相手別メモリ更新

などを書きます。

### 5. 各交渉の提案・応答

各相手との SAO 交渉の中で、エージェントは提案や応答を行います。

基本形では、

- `propose(negotiator_id, state)`
- `respond(negotiator_id, state, source=None)`

が中心です。

`OneShotAgent` 自体では `propose()` は抽象メソッドです。子クラスが必ず実装します。

`respond()` はデフォルト実装があります。

```python
offer = state.current_offer
myoffer = self.propose(negotiator_id, state)
if myoffer == offer:
    return ResponseType.ACCEPT_OFFER
return ResponseType.REJECT_OFFER
```

つまりデフォルトでは、

- 自分が今その場で出すはずの提案と同じなら受諾
- そうでなければ拒否

です。

### 6. 契約成立・不成立

交渉が終わると、次のフックが呼ばれます。

- `on_negotiation_success()`
- `on_negotiation_failure()`

基底クラスでは中身は空です。子クラスで、

- 相手ごとの成功回数集計
- 数量の累積
- 相手モデル更新

などを行います。

### 7. 契約署名・実行

必要に応じて以下が使われます。

- `sign_all_contracts()`
- `on_contract_executed()`
- `on_contract_breached()`

`sign_all_contracts()` のデフォルトは、全契約に自分の ID を入れて全署名する挙動です。

### 8. 毎ステップ後

`step()` が呼ばれます。

基底クラスでは `pass` です。日次の集計や後処理を書く場所です。

## インスタンス関数の一覧と役割

### `__init__(owner=None, ufun=None, name=None)`

役割:

- `SAOController` の初期化
- デフォルト negotiator の設定
- owner / AWI / preferences の仮接続

ポイント:

- デフォルト negotiator は `ControlledSAONegotiator`
- `auto_kill=False` なので controller は継続的に使われる

### `awi` プロパティ

役割:

- SCML の `OneShotAWI` を返す

ロジック:

- `self._awi` がなければ `ValueError`
- あればそのまま返す

用途:

- `needed_supplies`
- `needed_sales`
- `current_step`
- `my_suppliers`
- `my_consumers`

などにアクセスする入口です。

### `running_negotiations` プロパティ

役割:

- 現在走っている交渉一覧を返す

実体:

- `self._owner.running_negotiations`

### `unsigned_contracts` プロパティ

役割:

- まだ署名されていない契約一覧を返す

実体:

- `self._owner.unsigned_contracts`

### `init()`

役割:

- AWI が使える状態になった後の初期化フック

基底実装:

- 何もしない

子クラスでやることの例:

- 相手別メモリ辞書の作成
- 契約実績の初期化
- 日次キャッシュ準備

### `make_ufun(add_exogenous=False)`

役割:

- エージェントの効用関数を取得する

実体:

- `self._owner.make_ufun(add_exogenous)`

意味:

- `self.ufun` を直接使うだけではなく、必要に応じて外生契約込みの効用を生成できる

### `before_step()`

役割:

- 各シミュレーションステップ開始時のフック

基底実装:

- `pass`

典型用途:

- その日だけ使う変数のリセット
- ログ状態の更新
- 価格帯や数量戦略の再計算

### `step()`

役割:

- 各シミュレーションステップ終了時のフック

基底実装:

- `pass`

典型用途:

- 日次集計
- 戦略パラメータの更新
- 学習結果の反映

### `connect_to_oneshot_adapter(owner)`

役割:

- SCML の oneshot adapter とこの agent を結びつける

処理:

```python
self._owner = owner
self._awi = owner._awi
self.utility_function = owner.ufun
```

意味:

- 実行環境の world 側オブジェクトとつながる
- AWI がここで使えるようになる

### `connect_to_2021_adapter(owner)`

役割:

- 2021 系 adapter との接続

中身:

- `connect_to_oneshot_adapter()` とほぼ同じ

### `propose(negotiator_id, state)`

役割:

- 指定相手に対する提案を返す抽象メソッド

返すもの:

- `Outcome | None`

子クラスで必須:

- この関数は `@abstractmethod`
- `OneShotAgent` を直接使うなら必ず実装する必要がある

### `respond(negotiator_id, state, source=None)`

役割:

- 指定相手から来たオファーへの応答を返す

デフォルトロジック:

1. 相手のオファー `state.current_offer` を取る
2. 自分なら今何を提案するか `self.propose()` で計算する
3. その 2 つが一致したら `ACCEPT_OFFER`
4. そうでなければ `REJECT_OFFER`

特徴:

- とてもシンプルなデフォルト
- 実用エージェントでは上書きされることが多い

### `internal_state` プロパティ

役割:

- デバッグやログ用の内部状態辞書を返す

基底実装:

- 空辞書 `{}` を返す

子クラス用途:

- その日の戦略値
- 相手別スコア
- 内部フラグ

などを出せる

### `on_negotiation_failure(partners, annotation, mechanism, state)`

役割:

- 交渉が合意なしで終わったときのフック

基底実装:

- 何もしない

典型用途:

- 失敗回数記録
- 相手の強さ推定更新
- 次回の閾値調整

### `on_negotiation_success(contract, mechanism)`

役割:

- 交渉が合意したときのフック

基底実装:

- 何もしない

典型用途:

- 相手ごとの成立数量記録
- 成功回数更新
- 契約価格や数量の履歴保存

### `sign_all_contracts(contracts)`

役割:

- 複数契約への署名可否を返す

基底実装:

```python
return [self.id] * len(contracts)
```

意味:

- 全契約に署名する

返り値:

- 各契約について、署名するなら自分の `id`
- 署名しないなら `None`

### `on_contract_executed(contract)`

役割:

- 契約が正常に実行された後のフック

基底実装:

- `pass`

### `on_contract_breached(contract, breaches, resolution)`

役割:

- 契約違反が起きた後のフック

基底実装:

- `pass`

### `get_negotiator(partner_id)`

役割:

- 相手 `partner_id` に対応する `SAONegotiator` を返す

実装:

```python
return self.negotiators[partner_id][0]
```

意味:

- `self.negotiators` から相手ごとの negotiator 本体を取り出す

### `get_ami(partner_id)`

役割:

- 相手との交渉インタフェースを返す旧名

注意:

- deprecated
- 内部では `nmi` を返している

### `get_nmi(partner_id)`

役割:

- 相手との `SAONMI` を返す

実装:

```python
return self.negotiators[partner_id][0].nmi
```

用途:

- issue 情報取得
- 相手との交渉状態確認
- `random_outcome()` などの利用

## OneShotAgent を使うときの考え方

`OneShotAgent` を直接継承する場合は、基本的に相手ごとに個別に提案・応答する設計です。

典型パターンは次です。

1. `init()` で内部状態を準備
2. `before_step()` で日次状態を更新
3. `propose()` で相手別の提案を返す
4. `respond()` で相手別の受諾/拒否を返す
5. `on_negotiation_success()` で成立結果を記録
6. `step()` で日次後処理

複数相手をまとめて見たい場合は、`OneShotAgent` ではなく `OneShotSyncAgent` を継承するのが自然です。

## OneShotSyncAgent との違い

`OneShotAgent`:

- 個別交渉ごとに `propose()` / `respond()` を考える
- シンプル
- 実装しやすい

`OneShotSyncAgent`:

- 全相手のオファーをまとめて `counter_all()` で処理する
- 数量の合計整合を取りやすい
- OneShot の本質により合いやすい

`AlmostEqualAgent` が `OneShotSyncAgent` を使っているのは、

- 必要数量と契約合計数量の整合を見たい
- どの相手の組み合わせを受諾するかを一括で決めたい

からです。
