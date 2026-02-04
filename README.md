# JacRED KG Extraction: Multi-Instance Majority Voting

日本語文書レベル関係抽出データセット **JacRED** を用いた、低コストLLM（Gemini Flash系）による知識グラフトリプル抽出の実験リポジトリ。本リポジトリでは、ベースライン（1回抽出）と **Multi-Instance Majority Voting**（3パス多様プロンプト + UNION集約 + 検証）の2条件を比較する。

---

## 目次

1. [概要](#1-概要)
2. [背景・動機](#2-背景動機)
3. [データセット](#3-データセット)
4. [実験設定](#4-実験設定)
5. [手法](#5-手法)
6. [結果](#6-結果)
7. [分析](#7-分析)
8. [再現方法](#8-再現方法)
9. [ファイル構成](#9-ファイル構成)
10. [参考文献](#10-参考文献)

---

## 1. 概要

本プロジェクトは、日本語文書群からエンティティ（固有表現）とエンティティ間の関係（relation）を抽出し、知識グラフ（Knowledge Graph）のトリプル `(head_entity, relation, tail_entity)` を自動構築する手法の実験である。

Google Gemini Flash系モデルの **Structured Outputs**（JSON Schema強制出力）を活用し、以下の2条件を比較する:

- **Baseline（One-shot抽出）**: 1回のLLM呼び出しでエンティティと関係を同時に抽出
- **Multi-Instance Majority Voting（3パス多様プロンプト + 検証）**: 3つの異なる視点のプロンプトで独立に抽出を行い、UNION集約 → Stage 2検証 → domain/range型制約の後処理パイプラインで最終結果を得る

評価はJacRED devセットから選択した10文書に対して、文書レベル関係抽出の標準指標（Precision / Recall / F1）で行う。

**主な結果**: Majority Voting手法はBaselineと比較してRecallを0.20から0.26に改善（+30%）し、F1を0.24から0.28に向上させた。一方、FP数が67から88に増加し、Precisionはほぼ同等（0.31 vs 0.31）であった。

---

## 2. 背景・動機

### 2.1 文書レベル関係抽出（Document-level Relation Extraction, DocRE）

文書レベル関係抽出（DocRE）は、1つの文書全体を入力として、文書中に出現するエンティティペア間の関係を全て抽出するタスクである。文単位の関係抽出（Sentence-level RE）とは異なり、複数文にまたがる推論や共参照解析が必要となる。

具体的には以下の処理を行う:
1. 文書中のエンティティ（人名、組織名、地名など）を認識する
2. 全てのエンティティペア `(head, tail)` について、既定の関係タイプ集合から該当する関係を判定する
3. 関係が存在しないペアには "NA"（関係なし）を割り当てる

### 2.2 JacREDデータセット

**JacRED**（Japanese Document-level Relation Extraction Dataset）は、英語DocREデータセット **DocRED**（Yao et al., ACL 2019）の構造を日本語Wikipedia記事に適用して構築されたデータセットである。Ma et al.（LREC-COLING 2024）が、cross-lingual transferを活用したアノテーション支援手法により作成した。

### 2.3 本実験の目的

1. **低コストLLMの有効性検証**: 高価なGPT-4系モデルではなく、Gemini Flash系（低コスト・高速）モデルでDocREがどの程度可能かを検証する
2. **Structured Outputsの活用**: 自由形式テキスト出力ではなくJSON Schema強制出力を用い、パースエラーや不正出力を排除する
3. **Multi-Instance Majority Votingの有効性**: 1回の抽出ではRecallに限界があるため、異なる視点のプロンプトで複数回抽出し、UNION集約でRecallを改善できるかを検証する
4. **多様プロンプトの効果**: 「網羅的抽出」「複数文間関係」「構造的関係」の3つの視点が、互いに補完的な関係を抽出できるかを確認する
5. **最終応用**: 日本語文書コレクションからの大規模知識グラフ自動構築への基盤技術確立

### 2.4 ベースリポジトリとの関係

本リポジトリは [kg-extraction-experiment](https://github.com/auto-res2/kg-extraction-experiment) をベースに、Multi-Instance Majority Voting手法を追加実装したものである。ベースリポジトリではBaseline（One-shot抽出）とTwo-Stage（候補生成 + 検証）の2条件を比較しているが、本リポジトリではBaselineとMajority Votingの2条件を比較する。

---

## 3. データセット

### 3.1 JacRED概要

| 項目 | 内容 |
|---|---|
| 名称 | JacRED (Japanese Document-level Relation Extraction Dataset) |
| ソース | https://github.com/YoumiMa/JacRED |
| 論文 | Ma et al., "Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer", LREC-COLING 2024 |
| 言語 | 日本語（Wikipedia記事由来） |
| ベース | DocRED（Yao et al., ACL 2019）の構造を日本語に適用 |

### 3.2 データ分割

| 分割 | 文書数 | 用途 |
|---|---|---|
| train | 1,400 | 訓練（本実験ではfew-shot例選択とdomain/range制約テーブル構築に使用） |
| dev | 300 | 開発・評価（本実験では10文書を選択して評価に使用） |
| test | 300 | テスト（本実験では未使用） |

各分割間に文書の重複はない。

### 3.3 データフォーマット

各文書は以下のフィールドを持つJSONオブジェクトである:

```json
{
  "title": "文書タイトル（Wikipedia記事名）",
  "sents": [
    ["トークン1", "トークン2", "..."],
    ["トークン1", "トークン2", "..."]
  ],
  "vertexSet": [
    [
      {"name": "エンティティ名", "type": "PER", "sent_id": 0, "pos": [3, 5]}
    ]
  ],
  "labels": [
    {"h": 0, "t": 1, "r": "P27", "evidence": [0, 2]}
  ]
}
```

**各フィールドの説明:**

- **`title`**: Wikipedia記事のタイトル文字列
- **`sents`**: トークン化済みの文のリスト。各文はトークン（文字列）のリスト。元のテキストは各文のトークンを結合（join）して再構成する
- **`vertexSet`**: エンティティのリスト。各エンティティは1つ以上の **mention**（言及）を持つ。各mentionは:
  - `name`: 言及テキスト（例: "東京都"）
  - `type`: エンティティタイプ（後述の9種類のいずれか）
  - `sent_id`: この言及が出現する文のインデックス（0始まり）
  - `pos`: 文中のトークン位置 `[start, end)`（半開区間）
- **`labels`**: 正解関係ラベルのリスト。各ラベルは:
  - `h`: headエンティティのvertexSetインデックス（0始まり）
  - `t`: tailエンティティのvertexSetインデックス（0始まり）
  - `r`: 関係タイプのPコード（例: "P27"）
  - `evidence`: 根拠となる文のインデックスリスト

### 3.4 エンティティタイプ（9種類）

| タイプコード | 日本語説明 | 例 |
|---|---|---|
| `PER` | 人物 | 織田信長、アインシュタイン |
| `ORG` | 組織 | トヨタ自動車、国連 |
| `LOC` | 場所・地名 | 東京都、ナイル川 |
| `ART` | 作品・人工物・賞 | あずきちゃん、ノーベル賞 |
| `DAT` | 日付 | 1964年5月12日、2011年9月 |
| `TIM` | 時間 | 午前10時 |
| `MON` | 金額 | 100万円 |
| `%` | パーセンテージ・数値 | 50%、3.14 |
| `NA` | 該当なし（未分類） | — |

注: 本実験のLLMプロンプトでは `NA` を除く8種類をエンティティタイプとして指定する。Structured Outputsのスキーマ（`schemas.py`の`EXTRACTION_SCHEMA`）では `enum: ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]` として8タイプに制限している。

### 3.5 関係タイプ（35種類）

以下はJacREDで定義される全35種類の関係タイプである。各行はWikidataのプロパティコード（Pコード）、英語名、日本語説明を示す。日本語説明は本実験のLLMプロンプト（`prompts.py`の`RELATION_JAPANESE`辞書）で使用されるものと同一である。

| Pコード | English Name | 日本語説明 |
|---|---|---|
| `P1376` | capital of | 首都（〜の首都である） |
| `P131` | located in the administrative territorial entity | 行政区画（〜に位置する行政区画） |
| `P276` | location | 所在地（〜に所在する） |
| `P937` | work location | 活動場所（〜で活動した） |
| `P27` | country of citizenship | 国籍（〜の国籍を持つ） |
| `P569` | date of birth | 生年月日 |
| `P570` | date of death | 没年月日 |
| `P19` | place of birth | 出生地 |
| `P20` | place of death | 死没地 |
| `P155` | follows | 前任・前作（〜の後に続く） |
| `P40` | child | 子（〜の子である） |
| `P3373` | sibling | 兄弟姉妹 |
| `P26` | spouse | 配偶者 |
| `P1344` | participant in | 参加イベント（〜に参加した） |
| `P463` | member of | 所属（〜に所属する） |
| `P361` | part of | 上位概念（〜の一部である） |
| `P6` | head of government | 首長（〜の首長である） |
| `P127` | owned by | 所有者（〜に所有される） |
| `P112` | founded by | 設立者（〜が設立した） |
| `P108` | employer | 雇用主（〜に雇用される） |
| `P137` | operator | 運営者（〜が運営する） |
| `P69` | educated at | 出身校（〜で教育を受けた） |
| `P166` | award received | 受賞（〜を受賞した） |
| `P170` | creator | 制作者（〜が制作した） |
| `P175` | performer | 出演者・パフォーマー |
| `P123` | publisher | 出版社（〜が出版した） |
| `P1441` | present in work | 登場作品（〜に登場する） |
| `P400` | platform | プラットフォーム |
| `P36` | capital | 首都（〜が首都である） |
| `P156` | followed by | 後任・次作（〜の前にある） |
| `P710` | participant | 参加者（〜が参加した） |
| `P527` | has part | 構成要素（〜を含む） |
| `P1830` | owner of | 所有物（〜を所有する） |
| `P121` | item operated | 運営対象（〜を運営する） |
| `P674` | characters | 登場人物（作品の登場人物） |

注: `P1376`（capital of）と`P36`（capital）、`P155`（follows）と`P156`（followed by）、`P361`（part of）と`P527`（has part）、`P127`（owned by）と`P1830`（owner of）、`P137`（operator）と`P121`（item operated）はそれぞれ逆方向の関係ペアである。

### 3.6 データセット統計（参考値）

| 指標 | 値（概算） |
|---|---|
| 平均エンティティ数/文書 | 約17 |
| 平均関係数/文書 | 約20 |
| 平均トークン数/文書 | 約253 |
| 関係密度（関係数 / 可能なペア数） | 約6.5% |

---

## 4. 実験設定

### 4.1 文書選択

#### 評価文書（10文書）

JacRED devセット（300文書）から、文書の文字数（`char_count`）でソートし、等間隔で10文書を選択する **層化サンプリング** を行った。具体的には:

```python
sorted_docs = sorted(dev_data, key=char_count)  # 文字数昇順ソート
total = len(sorted_docs)  # 300
indices = [int(total * (i + 0.5) / 10) for i in range(10)]
# indices = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285]
```

これにより、短い文書から長い文書まで均等にカバーする10文書が選ばれる。

選択された10文書:

| # | タイトル | Gold エンティティ数 | Gold 関係数 |
|---|---|---|---|
| 1 | ダニエル・ウールフォール | 9 | 12 |
| 2 | アンソニー世界を駆ける | 9 | 6 |
| 3 | 青ナイル州 | 11 | 20 |
| 4 | 小谷建仁 | 17 | 11 |
| 5 | 窪田僚 | 18 | 11 |
| 6 | イーオー | 17 | 10 |
| 7 | 堂山鉄橋 | 15 | 9 |
| 8 | 木村千歌 | 21 | 25 |
| 9 | バハン地区 | 18 | 28 |
| 10 | ジョー・ギブス | 31 | 16 |

合計: Gold関係数 = 148

#### Few-shot例（1文書）

**Few-shot prompting（少数例プロンプティング）** とは、LLMに対してタスクの入出力例を少数（1〜数個）プロンプト中に含めることで、タスクの期待フォーマットや振る舞いをモデルに示す手法である。例を0個与える場合を **zero-shot**、1個の場合を **one-shot**、複数個の場合を **few-shot** と呼ぶ。

本実験では **1例（one-shot）** を採用した。これは以下のトレードオフに基づく判断である:
- **コスト・コンテキスト長の制約**: 例を増やすと入力トークン数が増加し、APIコストが上昇する。また、モデルのコンテキストウィンドウ（入力可能なトークン数の上限）を圧迫し、対象文書の処理に使えるトークン数が減少する。
- **品質向上の限界**: 1例でタスク形式を十分に示せる場合、2例以上に増やしても品質改善は限定的であることが多い。
- **例の品質**: 使用する例は**訓練データのGold label（正解ラベル）** から構成される。つまり、人手でアノテーションされた正しい入出力ペアをモデルに見せることで、期待される出力形式と粒度を正確に伝えている。

訓練データから以下の条件を満たす文書を1つ選択する:

- 文字数: 150 - 250文字
- エンティティ数: 5 - 12
- 関係ラベル数: 3 - 15

条件を満たす候補のうち最も短いものを選択する。

選択されたfew-shot文書: **「スタッド (バンド)」**

### 4.2 モデル構成

本実験では以下の1つのモデル構成を使用した:

| 構成名 | モデルID | thinking_budget | 説明 |
|---|---|---|---|
| gemini-3-flash-preview (OFF) | `gemini-3-flash-preview` | 0 | `ThinkingConfig(thinking_budget=0)` を明示指定。thinking機能を無効化 |

注: コード上では `ThinkingConfig(thinking_budget=2048)` がハードコードされているが、実験実行時には `thinking_budget=0` の設定で実行された。

#### "thinking"（思考モード）とは何か

Gemini 2.5 Flash以降のモデルは **thinking**（内部推論 / 思考モード）機能を持つ。これは **chain-of-thought reasoning（連鎖的思考推論）** をモデル内部に組み込んだ機能であり、従来のプロンプトエンジニアリングで「ステップバイステップで考えてください」と指示する手法（chain-of-thought prompting）とは異なり、モデルのアーキテクチャレベルで推論プロセスが組み込まれている。

**動作原理:**

thinking が有効な場合、モデルはユーザのリクエストに対して以下の2段階で応答を生成する:

1. **内部推論フェーズ（thinking tokens）**: モデルはまず「推論トークン」（reasoning tokens / thinking tokens）を内部的に生成する。これはモデルが問題を分解し、ステップバイステップで考えるためのトークン列である。重要な点として、**これらの推論トークンはAPIレスポンスのテキストには含まれない**（呼び出し側からは見えない）。つまり、ユーザが受け取る最終出力には推論過程は表示されず、結論のみが返される。
2. **最終回答生成フェーズ**: 内部推論が完了した後、モデルはその推論結果を踏まえて最終的な回答を生成する。この回答のみがAPIレスポンスとして返される。

**`thinking_budget` パラメータ:**

`thinking_budget` は `ThinkingConfig` の設定項目で、**モデルが内部推論に使用できるトークンの最大数**を制御する。

- **`thinking_budget=0`**: thinking機能を**完全に無効化**する。モデルは内部推論フェーズをスキップし、通常の（non-reasoning）モデルと同等に動作する。即座に最終回答の生成を開始するため、レイテンシが低い。
- **`thinking_budget=2048`**: モデルが**最大2048トークンの内部推論**を行うことを許可する。ただし、タスクが単純な場合、モデルは2048トークン全てを使い切らず、より少ないトークン数で推論を完了する場合がある（上限であり、必ず使い切るわけではない）。
- **一般的な傾向**: thinking_budgetを大きくすると、(a) レイテンシが増加する（推論トークン生成の時間がかかる）、(b) APIコストが増加する（推論トークンも課金対象として計上される）、(c) 複雑なタスクでは回答品質が向上する可能性がある。

**コードでの指定方法（`llm_client.py`）:**
```python
from google.genai.types import GenerateContentConfig, ThinkingConfig

config = GenerateContentConfig(
    system_instruction=system_prompt,
    response_mime_type="application/json",
    response_schema=response_schema,
    temperature=0.2,
    thinking_config=ThinkingConfig(thinking_budget=0),  # 0でOFF
)
```

### 4.3 Structured Outputs（構造化出力）

全てのLLM呼び出しでGoogle GenAI SDKの **Structured Outputs** 機能を使用する。これにより、モデルの出力が指定したJSON Schemaに厳密に従うことが保証される。自由形式テキストの出力やJSONパースエラーは原理的に発生しない。

**Structured Outputsの仕組み:**

通常のLLM呼び出しでは、モデルは自由形式のテキストを生成する。プロンプトで「JSON形式で出力してください」と指示しても、モデルが不正なJSON（閉じ括弧の欠落、余分なテキストの混入など）を出力するリスクがある。Structured Outputsはこの問題を根本的に解決する。

APIリクエストで以下の2つのパラメータを指定する:
- **`response_mime_type="application/json"`**: モデルの出力をJSON形式に強制する。モデルのデコーディングプロセス（トークンを1つずつ選択する過程）において、JSON構文に違反するトークンは選択候補から除外される。これは単なるプロンプト指示ではなく、**デコーディング時のハード制約**（constrained decoding）である。
- **`response_schema=...`**: 出力JSONが準拠すべきJSON Schemaを指定する。スキーマで定義されたフィールド名、型、必須フィールドに違反するトークンはデコーディング時に除外される。

**`enum` 制約の効果:**

本実験では `entities[].type` フィールドに `enum: ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]` を指定している。これにより、モデルはエンティティタイプとしてこの8種類以外の文字列を**物理的に出力できない**。デコーディング時にenum外のトークン列は確率0に設定されるため、「モデルが勝手に新しいタイプを発明する」という問題は原理的に排除される。これはプロンプトで「以下のタイプのみ使用してください」と指示するよりも遥かに信頼性が高い。

#### 抽出用スキーマ（EXTRACTION_SCHEMA）

Baseline・Majority Votingの各パスで使用する。モデルの出力を以下の構造に強制する:

```json
{
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]
          }
        },
        "required": ["id", "name", "type"]
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "head": {"type": "string"},
          "relation": {"type": "string"},
          "tail": {"type": "string"},
          "evidence": {"type": "string"}
        },
        "required": ["head", "relation", "tail", "evidence"]
      }
    }
  },
  "required": ["entities", "relations"]
}
```

- `entities[].type` は `enum` 制約により8種類のエンティティタイプのいずれかに強制される
- `relations[].relation` は文字列型だが `enum` 制約はない（後処理で不正なPコードをフィルタする）
- `relations[].head` と `relations[].tail` は `entities[].id` を参照する文字列

#### 検証用スキーマ（VERIFICATION_SCHEMA）

Majority VotingのStage 2で使用する。モデルの出力を以下の構造に強制する:

```json
{
  "type": "object",
  "properties": {
    "decisions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "candidate_index": {"type": "integer"},
          "keep": {"type": "boolean"}
        },
        "required": ["candidate_index", "keep"]
      }
    }
  },
  "required": ["decisions"]
}
```

- `candidate_index`: 検証対象の候補番号（0始まりのバッチ内インデックス）
- `keep`: `true` なら候補を採用、`false` なら棄却

### 4.4 API呼び出し設定

**Temperature（温度パラメータ）について:**

LLMがトークン（単語の断片）を1つずつ生成する際、各ステップで次のトークンの確率分布が計算される。**temperature** はこの確率分布の「鋭さ」を制御するパラメータである:
- **temperature=0（またはそれに近い値）**: 確率分布が極端に鋭くなり、最も確率の高いトークンがほぼ確定的に選択される。出力の再現性が高いが、temperature=0では**退化的な繰り返し**（同じフレーズを無限に繰り返す現象）が発生するリスクがある。
- **temperature=1.0**: モデルの学習時の確率分布がそのまま使用される。出力に適度な多様性がある。
- **temperature > 1.0**: 確率分布が平坦化され、低確率のトークンも選択されやすくなる。出力が創造的になるが、不正確な内容が増える。

本実験では **temperature=0.2** を採用した。これは退化的な繰り返しを回避しつつ（temperature=0の問題を避ける）、出力の一貫性と再現性を高める設定である。情報抽出タスクでは創造性より正確性が重要なため、低いtemperatureが適切である。

**注意**: temperature=0.2は低い値だが、Majority Voting手法では各パスに異なるプロンプトを使用するため、各パスの出力には十分な多様性が生じる。プロンプトの違いがtemperatureの低さを補完している。

| パラメータ | 値 | 説明 |
|---|---|---|
| `temperature` | 0.2 | 低めの温度で出力の再現性を高める（上記の解説参照） |
| `max_retries` | 3 | API呼び出し失敗時の最大リトライ回数 |
| リトライ間隔 | 指数バックオフ（2秒、4秒、8秒） | `wait = 2 ** (attempt + 1)` |
| SDK | `google-genai` Python パッケージ | `from google import genai` |
| 認証 | APIキー方式 | 環境変数 `GEMINI_API_KEY` またはファイルから読み込み |

---

## 5. 手法

### 5.1 Baseline: One-shot抽出

1回のLLM呼び出しでエンティティと関係を同時に抽出する。

#### 処理フロー

**Step 1: システムプロンプト構築**

以下の情報を含むシステムプロンプトを構築する:
- タスク説明:「日本語文書から知識グラフ（エンティティと関係）を抽出する」
- エンティティタイプ一覧: 8種類（PER, ORG, LOC, ART, DAT, TIM, MON, %）とその日本語説明
- 関係タイプ一覧: 35種類のPコードと英語名と日本語説明
- ルール: 指定タイプのみ使用、Pコードのみ使用、evidence付与、headとtailにはentities IDを使用

システムプロンプトの実際のテンプレート:
```
あなたは日本語文書から知識グラフ（エンティティと関係）を抽出する専門家です。

## タスク
与えられた日本語文書から、エンティティ（固有表現）とエンティティ間の関係を抽出してください。

## エンティティタイプ（8種類）
  - PER: 人物
  - ORG: 組織
  - LOC: 場所・地名
  - ART: 作品・人工物・賞
  ...（以下略）

## 関係タイプ（35種類、Pコードで指定）
  - P1376 (capital of): 首都（〜の首都である）
  - P131 (located in the administrative territorial entity): 行政区画（〜に位置する行政区画）
  ...（以下略）

## ルール
- エンティティには上記のタイプのみ使用してください。
- 関係には上記のPコード（P131, P27等）のみ使用してください。自由記述は禁止です。
- 各関係には、根拠となる文書中のテキストをevidenceとして付与してください。
- headとtailにはentitiesのidを指定してください。
```

**Step 2: ユーザプロンプト構築**

以下を含むユーザプロンプトを構築する:
1. **Few-shot例**: 訓練データから選択した1文書のテキストと、その正解をJSON形式に変換した期待出力
2. **対象文書**: 抽出対象のテキスト

```
## 例
入力文書:
{few_shot文書のテキスト}

出力:
{few_shot文書の正解をJSON化したもの}

## 対象文書
{対象文書のテキスト}

上記の文書からエンティティと関係を抽出してください。
```

**Step 3: LLM呼び出し**

`EXTRACTION_SCHEMA` を `response_schema` として指定し、Gemini APIを1回呼び出す。レスポンスはJSON形式で `entities` と `relations` を含む。

**Step 4: 後処理（フィルタリング）**

1. **不正関係フィルタ**: `relations` 中の `relation` フィールドが35種類のPコードに含まれないものを除去
2. **不正エンティティタイプフィルタ**: `entities` 中の `type` が8種類のタイプに含まれないものを除去し、そのエンティティを参照する関係も除去

### 5.2 Multi-Instance Majority Voting（3パス多様プロンプト + UNION集約 + 検証）

**Majority Voting設計の根拠:**

本手法の設計は、LLMの抽出における以下の知見に基づいている:

1. **1回の抽出におけるRecallの限界**: LLMは1回の呼び出しで文書中の全ての関係を網羅的に抽出することが難しい。特に複数文にまたがる関係や暗黙的な関係は見落としやすい。
2. **プロンプトの多様性による補完**: 異なる視点や指示を持つプロンプトで複数回抽出を行うと、各パスが異なる関係を発見する傾向がある。1つのプロンプトが見落とした関係を別のプロンプトが捕捉できる。
3. **UNION集約によるRecall向上**: 複数パスの結果をUNION（和集合）で集約することで、各パスの網羅性を合算し、全体のRecallを改善できる。
4. **検証による品質維持**: UNION集約はFP（偽陽性）も増やすため、Stage 2の検証ステップで品質を維持する。

#### 3パスの多様プロンプト

各文書に対して、以下の3つの異なるモード（視点）でLLMによる抽出を独立に実行する:

**パス1: Recall重視（`mode="recall"`）**

標準の網羅的抽出。以下の指示をプロンプトに追加する:
```
重要: できるだけ多くの関係を漏れなく抽出してください。確信度が低い場合でも、
可能性がある関係は候補として含めてください。
後の検証ステップで精度を高めるため、この段階では再現率（recall）を優先してください。
```

**パス2: 複数文間関係重視（`mode="cross_sentence"`）**

複数文にまたがる関係に焦点を当てた抽出。以下の指示を追加する:
```
重要: 複数文にまたがる関係に注目してください。異なる文に登場するエンティティ間の
関係を特に注意して抽出してください。
できるだけ多くの関係を漏れなく抽出してください。確信度が低い場合でも、
可能性がある関係は候補として含めてください。
```

**パス3: 構造的関係重視（`mode="structural"`）**

階層関係や包含関係に焦点を当てた抽出。以下の指示を追加する:
```
重要: PartOf/HasPart, MemberOf等の構造的関係に注目してください。
組織の階層関係、地理的包含関係、作品の構成要素などを特に注意して抽出してください。
できるだけ多くの関係を漏れなく抽出してください。確信度が低い場合でも、
可能性がある関係は候補として含めてください。
```

#### UNION集約（重複排除）

3パスの結果を以下の方法でマージする:

1. **エンティティのマージ**: 3パスで抽出されたエンティティを名前（`name`）で重複排除し、統合エンティティリストを構築する。新しいIDを `e0`, `e1`, ... として再割り当てする。
2. **トリプルのマージ**: トリプルのキー `(head_name.strip(), relation, tail_name.strip())` で重複排除する。同一キーのトリプルが複数パスで抽出された場合、最初に抽出されたトリプルを保持し、`support_count`（何パスで抽出されたか）をインクリメントする。
3. **不正ラベルフィルタ**: マージ後のトリプルに対して、不正なPコードおよび不正なエンティティタイプのフィルタを適用する。

#### Stage 2: 検証（Precision重視）

UNION集約後の候補トリプルを **バッチサイズ10** で分割し、各バッチに対してLLMで検証を行う。

**バッチ検証プロンプトの構造:**

システムプロンプト:
```
あなたは関係抽出の検証者です。提示された関係候補が文書の内容に基づいて正しいかどうかを判定してください。
```

ユーザプロンプト:
```
以下の文書と、そこから抽出された関係候補を検証してください。

## 文書
{対象文書のテキスト}

## 関係候補
候補0: {head名} --[{Pコード}: {英語名}]--> {tail名}
  根拠: {evidence文字列}
  関係の定義: {日本語説明}
候補1: ...
...

各候補について、文書の内容がこの関係を支持しているかどうかを判定してください。
判定基準:
- 文書中に明確な根拠があるか
- エンティティの型が関係と整合しているか
- 関係の方向（head→tail）が正しいか
根拠が不十分な候補はkeep=falseとしてください。
```

レスポンスは `VERIFICATION_SCHEMA` に従い、各候補の `keep` (true/false) を返す。`keep=false` の候補は除去する。

**注意**: 検証結果にバッチ内の候補インデックスが含まれない場合（`candidate_index` が `decisions` に存在しない場合）、デフォルトで `keep=true`（採用）として扱う。

#### 後処理: Domain/Range型制約

Stage 2の後に、決定論的なフィルタとして **domain/range型制約** を適用する。追加のLLM呼び出しは不要（ゼロコスト）。

**Domain/Rangeとは何か:**

知識グラフやオントロジーの文脈において、**domain（定義域）** と **range（値域）** は関係（relation / property）に対する型制約を表す用語である:
- **Domain（定義域）**: その関係の **head（主語）** に許容されるエンティティタイプの集合。例えば、「生年月日」（P569）のdomainは `{PER}`（人物のみが生年月日を持つ）。
- **Range（値域）**: その関係の **tail（目的語）** に許容されるエンティティタイプの集合。例えば、「生年月日」（P569）のrangeは `{DAT}`（生年月日の値は日付である）。

したがって、`ORG --[P569]--> LOC`（組織の生年月日が場所である）というトリプルは、domainにもrangeにも違反するため明らかに不正であり、フィルタで除去すべきである。

本実験では、オントロジーで明示的に定義されたdomain/rangeではなく、**訓練データから経験的に観測されたtype pairの集合**を制約として使用する。これにより、厳密なオントロジー定義がなくても、データ駆動で型制約を適用できる。

**制約テーブルの構築方法:**

訓練データ全体（1,400文書）をスキャンし、各関係Pコードについて、実際に出現した `(head_entity_type, tail_entity_type)` ペアの集合を収集する。

```python
# data_loader.py の build_constraint_table() 関数
constraint_table = defaultdict(set)
for doc in train_data:
    for label in doc["labels"]:
        h_type = doc["vertexSet"][label["h"]][0]["type"]  # headの最初のmentionの型
        t_type = doc["vertexSet"][label["t"]][0]["type"]  # tailの最初のmentionの型
        constraint_table[label["r"]].add((h_type, t_type))
```

例: `P27`（国籍）の制約テーブルが `{("PER", "LOC")}` のみであれば、`(ORG, LOC)` のペアで `P27` を持つトリプルは訓練データで一度も観測されていないため除去する。

**フィルタの適用:**

```python
# extraction.py の apply_domain_range_constraints() 関数
for triple in candidates:
    allowed_pairs = constraint_table.get(triple.relation)
    if allowed_pairs is not None:
        if (triple.head_type, triple.tail_type) not in allowed_pairs:
            # 除去（訓練データで観測されていない型ペア）
            continue
    # 保持
```

### 5.3 評価方法

#### エンティティアライメント（予測エンティティ → Gold vertexSet）

予測されたエンティティをGoldデータのvertexSetインデックスに対応付ける。3パスマッチングを以下の優先順で行う:

**Pass 1: 完全一致**
- 予測エンティティの `name` が、GoldのvertexSet中のいずれかのmentionの `name` と完全一致する場合にマッチ

**Pass 2: 正規化一致**
- 予測エンティティの `name` をUnicode NFKC正規化 + 小文字化 + 前後空白除去したものが、Goldのいずれかのmention nameの同様の正規化結果と一致する場合にマッチ

**Pass 3: 部分文字列一致**
- 正規化後の予測名がGold名の部分文字列であるか、またはGold名が予測名の部分文字列である場合にマッチ
- 複数候補がある場合は、重複する文字数（`min(len(pred), len(gold))`）が最大のものを優先
- 最小重複文字数は2文字（1文字のみの一致は無視）

**制約**: 各予測エンティティは最大1つのGoldエンティティにマッチする（1:1マッピング、先着順）。一度マッチしたGoldエンティティは以降のマッチ候補から除外される。

#### 関係評価

- **True Positive (TP)**: 予測トリプル `(head_id, relation, tail_id)` のhead, tailがともにGoldエンティティにアライメント済みで、かつGoldラベルに `(aligned_h_idx, aligned_t_idx, relation)` が存在する
- **False Positive (FP)**: 予測トリプルが以下のいずれかに該当:
  - headまたはtailがGoldエンティティにアライメントされていない（`entity_not_aligned`）
  - アライメントは成功したが、対応するGoldラベルが存在しない（`wrong_relation`）
- **False Negative (FN)**: Goldラベルのうち、いずれの予測トリプルにもマッチしなかったもの

#### 集計

10文書全体で **マイクロ平均（micro-average）** を計算する。

**マイクロ平均 vs マクロ平均:**

評価指標の集計方法には主に2種類ある:
- **マイクロ平均（micro-average）**: 全文書のTP, FP, FNを**合算してから**P/R/F1を計算する。文書ごとの重みは関係数に比例する（関係数の多い文書がスコアに与える影響が大きい）。本実験ではこちらを採用。
- **マクロ平均（macro-average）**: 各文書のP/R/F1を**個別に計算してから平均**する。文書ごとの重みは均等（関係数によらず各文書のスコアが等しく寄与する）。関係数が少ない文書のスコア変動が大きいため、サンプル数が少ない場合は不安定になりやすい。

本実験でマイクロ平均を採用した理由は、(a) DocRE分野の標準的な評価方式であること、(b) 10文書というサンプル数では文書あたりのスコア変動が大きく、マクロ平均は不安定になりやすいことである。

```
Precision = TP_total / (TP_total + FP_total)
Recall    = TP_total / (TP_total + FN_total)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

---

## 6. 結果

### 6.1 Baseline vs Majority Voting 比較表

| 条件 | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Baseline | 0.31 | 0.20 | 0.24 | 30 | 67 | 118 |
| Majority Voting | 0.31 | 0.26 | 0.28 | 39 | 88 | 109 |

Majority VotingはBaselineと比較して:
- **TP +9**（30 → 39）: 9件多くの正しい関係を抽出
- **Recall +30%相対改善**（0.20 → 0.26）: Gold 148件中、20.3% → 26.4%を正しく抽出
- **F1 +16%相対改善**（0.24 → 0.28）
- **FP +21**（67 → 88）: Precision はほぼ同等（0.31 vs 0.31）を維持しつつ、FP数は増加

### 6.2 Baseline: 文書別結果

| # | 文書 | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 1 | ダニエル・ウールフォール | 0.88 | 0.58 | 0.70 | 7 | 1 | 5 |
| 2 | アンソニー世界を駆ける | 0.43 | 0.50 | 0.46 | 3 | 4 | 3 |
| 3 | 青ナイル州 | 0.43 | 0.15 | 0.22 | 3 | 4 | 17 |
| 4 | 小谷建仁 | 0.31 | 0.36 | 0.33 | 4 | 9 | 7 |
| 5 | 窪田僚 | 0.08 | 0.09 | 0.08 | 1 | 12 | 10 |
| 6 | イーオー | 0.20 | 0.10 | 0.13 | 1 | 4 | 9 |
| 7 | 堂山鉄橋 | 0.25 | 0.22 | 0.24 | 2 | 6 | 7 |
| 8 | 木村千歌 | 0.12 | 0.08 | 0.10 | 2 | 15 | 23 |
| 9 | バハン地区 | 0.25 | 0.07 | 0.11 | 2 | 6 | 26 |
| 10 | ジョー・ギブス | 0.45 | 0.31 | 0.37 | 5 | 6 | 11 |

### 6.3 Majority Voting: 文書別結果

| # | 文書 | P | R | F1 | TP | FP | FN | Pass1 | Pass2 | Pass3 | UNION | S2採用 | 制約後 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ダニエル・ウールフォール | 0.41 | 0.58 | 0.48 | 7 | 10 | 5 | 14 | 11 | 13 | 27 | 21 | 17 |
| 2 | アンソニー世界を駆ける | 0.18 | 0.33 | 0.24 | 2 | 9 | 4 | 9 | 10 | 12 | 16 | 12 | 11 |
| 3 | 青ナイル州 | 0.50 | 0.30 | 0.38 | 6 | 6 | 14 | 11 | 14 | 13 | 14 | 13 | 12 |
| 4 | 小谷建仁 | 0.45 | 0.45 | 0.45 | 5 | 6 | 6 | 16 | 16 | 16 | 21 | 18 | 11 |
| 5 | 窪田僚 | 0.08 | 0.09 | 0.09 | 1 | 11 | 10 | 15 | 15 | 18 | 24 | 18 | 12 |
| 6 | イーオー | 0.33 | 0.40 | 0.36 | 4 | 8 | 6 | 11 | 15 | 18 | 24 | 13 | 12 |
| 7 | 堂山鉄橋 | 0.18 | 0.22 | 0.20 | 2 | 9 | 7 | 12 | 14 | 8 | 23 | 13 | 11 |
| 8 | 木村千歌 | 0.13 | 0.08 | 0.10 | 2 | 14 | 23 | 21 | 19 | 19 | 28 | 21 | 16 |
| 9 | バハン地区 | 0.21 | 0.11 | 0.14 | 3 | 11 | 25 | 13 | 12 | 9 | 18 | 14 | 14 |
| 10 | ジョー・ギブス | 0.64 | 0.44 | 0.52 | 7 | 4 | 9 | 21 | 22 | 24 | 30 | 21 | 11 |

Pass1 = Recall重視パス候補数、Pass2 = 複数文間関係パス候補数、Pass3 = 構造的関係パス候補数、UNION = 重複排除後のUNION候補数、S2採用 = Stage 2検証通過数、制約後 = domain/range制約適用後の最終数

### 6.4 パイプライン段階別の候補数変化（Majority Voting、全10文書合計）

| 段階 | 候補数合計 |
|---|---|
| パス1: Recall重視 | 143 |
| パス2: 複数文間関係重視 | 148 |
| パス3: 構造的関係重視 | 150 |
| 3パス合計（重複込み） | 441 |
| UNION集約後（重複排除） | 225 |
| Stage 2: 検証後 | 164 |
| Domain/Range制約後（最終） | 127 |

3パスの合計441件からUNION集約で225件に重複排除（約49%が重複）され、Stage 2で164件に絞られ（27%棄却）、domain/range制約でさらに127件に削減（23%除去）される。

---

## 7. 分析

### 7.1 主な知見

1. **Majority VotingはRecallを有意に改善**: TP数が30 → 39に増加（+30%）。Baselineで見落としていた9件の正しい関係を追加的に抽出できた。これは3つの異なるプロンプト視点が互いに補完的であることを示す。

2. **Precisionはほぼ維持**: Majority Voting（P=0.31）はBaseline（P=0.31）とほぼ同等のPrecisionを達成。UNION集約でFPが増加するが、Stage 2検証とdomain/range制約が効果的にFPをフィルタしている。

3. **F1が0.24から0.28に改善**: Recallの改善がF1の向上に直結している。これはBaselineの主要ボトルネックがRecallであったことと整合する。

4. **3パスの候補数はほぼ均等**: Recall重視=143件、複数文間=148件、構造的=150件と、各パスが同程度の候補を生成している。プロンプトの違いが抽出量ではなく、抽出される関係の種類に影響していることを示唆する。

5. **重複率は約49%**: 3パス合計441件のうちUNION後は225件。約半数のトリプルは複数パスで共通に抽出されており、残り半数がいずれか1つのパスのみで抽出された「多様性」部分である。

6. **Stage 2検証の削減率は27%**: UNION後225件 → 検証後164件。検証ステップが約4分の1の候補を棄却し、FPの制御に寄与している。

7. **domain/range制約の追加削減率は23%**: 検証後164件 → 制約後127件。決定論的フィルタが追加のLLM呼び出しなしで不正トリプルを除去しており、コスト効率が高い。

### 7.2 文書別の改善パターン

**大きく改善した文書:**
- **イーオー**: F1が0.13 → 0.36（+177%）。TPが1 → 4に大幅増加。複数文間関係パスと構造的関係パスが、Baselineでは抽出できなかった親子関係（P40）や登場作品関係（P1441）を補完的に捕捉
- **ジョー・ギブス**: F1が0.37 → 0.52（+40%）。TPが5 → 7に増加しつつ、FPが6 → 4に減少。Recallが0.31 → 0.44に改善
- **小谷建仁**: F1が0.33 → 0.45（+36%）。TPが4 → 5に増加、FPが9 → 6に減少

**改善が限定的だった文書:**
- **窪田僚**: F1が0.08 → 0.09。TP=1のまま変化なし。主なFPの原因が`entity_not_aligned`（Goldエンティティとの名前不一致）であり、3パスでも同じエンティティ名の問題が解決されない
- **木村千歌**: F1が0.10 → 0.10。Gold関係数25件に対してTP=2のまま。多数のFN（23件）は制作者関係（P170）や階層関係（P361, P527）で、LLMが抽出自体に失敗している
- **バハン地区**: F1が0.11 → 0.14。Gold関係数28件中TPが2 → 3のみ。地理的包含関係が複雑で、LLMが正しい方向（headとtail）を判定できていない

### 7.3 FPの主要パターン

1. **エンティティアライメント失敗** (`entity_not_aligned`): 予測エンティティ名がGoldのどのmentionとも一致しない。
   - 例: 予測 "ダニエル_バーリー_ウールフォール" vs Gold "ダニエル・バーリー・ウールフォール"（区切り文字の違い: アンダースコア vs 中黒）
   - 例: 予測 "東京都足立区" vs Gold "足立区"（エンティティの粒度の違い）
   - Majority Votingでも3パス全てで同じ名前表記を生成するため、この問題は解決されない

2. **関係方向の誤り**: headとtailが逆転している。
   - 例: "バハン地区 --[P131]--> サナーグ州" は方向が逆（Goldでは "サナーグ州 --[P131]--> バハン地区"ではなくP361/P527の関係）

3. **類似関係の混同**: 意味的に近い関係タイプを誤って割り当てる。
   - 例: P108（雇用主）vs P463（所属）
   - 例: P170（制作者）vs P175（出演者）

### 7.4 FNの主要パターン

1. **暗黙的関係**: 文書中に直接記述されていないが推論で導出できる関係（例: 行政区画の包含関係）

2. **多ホップ推論**: 複数文にまたがる推論が必要な関係。複数文間関係パス（Pass 2）で一部改善されたが、完全には解決されていない

3. **逆方向関係ペア**: P361 (part of) と P527 (has part) の両方が正解に含まれるケースで、片方しか抽出できない。構造的関係パス（Pass 3）で一部改善されたが、双方向の関係を網羅するのは依然として困難

4. **制作者関係の大量のFN**: 木村千歌の文書では、複数の作品に対するP170（制作者）関係が大量にFNとなっている。Goldでは作品ごとに制作者関係が定義されているが、LLMは主要な作品のみに言及する傾向がある

### 7.5 コスト分析

Majority VotingはBaselineの約3〜4倍のAPI呼び出しを必要とする:

| 項目 | Baseline | Majority Voting |
|---|---|---|
| 抽出呼び出し / 文書 | 1 | 3（3パス） |
| 検証呼び出し / 文書 | 0 | 1〜3（バッチサイズ10） |
| 合計呼び出し（10文書） | 10 | 約50 |

F1の0.04ポイント改善（0.24 → 0.28）に対して3〜4倍のコストが必要であり、コストパフォーマンスの観点からは更なる改善手法の検討が必要である。

---

## 8. 再現方法

### 8.1 前提条件

- Python 3.10以上
- Google Gemini APIキー（[Google AI Studio](https://aistudio.google.com/)で取得）
- インターネット接続（API呼び出し用）

### 8.2 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/auto-res2/kg-extraction-majority-voting
cd kg-extraction-majority-voting

# 2. JacREDデータセットを取得
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# 3. 依存パッケージをインストール
pip install google-genai

# 4. APIキーを設定（2つの方法）
# 方法A: 環境変数（推奨）
export GEMINI_API_KEY="your-gemini-api-key"

# 方法B: ファイルから読み込み（デフォルトのパスを変更する場合はrun_experiment.pyのENV_PATHを編集）

# 5. 実験を実行
python3 run_experiment.py
```

**注意**: デフォルトでは `run_experiment.py` の `ENV_PATH` がDropbox上のファイルを参照するよう設定されている。環境変数 `GEMINI_API_KEY` を使用する場合は、`llm_client.py` の `load_api_key()` 呼び出し部分を `os.environ["GEMINI_API_KEY"]` に置き換えるか、`run_experiment.py` の `ENV_PATH` を適切なパスに変更する必要がある。

### 8.3 モデル・thinking設定の変更方法

`llm_client.py` を直接編集する:

```python
# llm_client.py の9行目
MODEL = "gemini-3-flash-preview"  # 変更先: "gemini-2.0-flash", "gemini-2.5-flash", etc.

# llm_client.py の41行目（call_gemini関数内）
thinking_config=ThinkingConfig(thinking_budget=2048),  # 0でOFF、2048でON
# gemini-2.0-flashを使う場合はこの行を削除する（thinking非対応のため）
```

### 8.4 JacREDデータのパス変更

デフォルトでは `/tmp/JacRED/` を参照する。変更する場合は `data_loader.py` の `load_jacred()` 関数の `base_path` 引数を変更する。

### 8.5 実行時間の目安

| 条件 | 概算時間（10文書） |
|---|---|
| Baseline | 約1〜2分 |
| Majority Voting | 約4〜6分 |

注: API呼び出しの待機時間に大きく依存するため、上記は概算である。Majority Votingは3パスの抽出 + 検証バッチのため、Baselineの約3〜4倍の時間がかかる。

### 8.6 出力形式

実行完了後、以下の出力が得られる:

1. **コンソール出力**: 文書ごとのP/R/F1とBaselineとMajority Votingの比較表
2. **`results.json`**: 全結果の詳細JSON（文書別のTP/FP/FN詳細、パイプライン段階別候補数を含む）

```
              Precision   Recall     F1    TP    FP    FN
      Baseline       0.31     0.20   0.24    30    67   118
  MajorityVote       0.31     0.26   0.28    39    88   109
```

---

## 9. ファイル構成

```
kg-extraction-majority-voting/
  run_experiment.py   # メインスクリプト
  data_loader.py      # データ読み込み・選択
  llm_client.py       # Gemini API呼び出し
  prompts.py          # プロンプトテンプレート
  extraction.py       # 抽出ロジック
  evaluation.py       # 評価ロジック
  schemas.py          # JSON Schema定義
  results.json        # 最新の実験結果
  README.md           # 本ファイル
```

### 9.1 `run_experiment.py` -- メインスクリプト

**目的**: 実験全体のオーケストレーション（データ読み込み → 条件実行 → 結果比較・保存）。

**主要関数:**
- `run_condition(name, docs, few_shot, client, schema_info, constraint_table=None, extraction_fn="baseline")`:
  - 1つの実験条件（BaselineまたはMajority Voting）を全文書に対して実行し、文書別・集計のP/R/F1を算出する
  - `extraction_fn="majority_voting"` かつ `constraint_table` が指定されている場合はMajority Votingとして動作する
  - 入力: 文書リスト、few-shot例、Geminiクライアント、スキーマ情報、（任意）制約テーブル、抽出関数種別
  - 出力: `{"per_doc": [...], "aggregate": {...}}` の辞書
- `main()`:
  - データ読み込み（`load_jacred()`）、文書選択（`select_dev_docs()`）、few-shot選択（`select_few_shot()`）、制約テーブル構築（`build_constraint_table()`）を実行
  - Baseline, Majority Voting の2条件を順に実行し、結果を比較表示
  - `results.json` に全結果を保存

### 9.2 `data_loader.py` -- データ読み込み・選択

**目的**: JacREDデータセットの読み込み、実験用文書の選択、domain/range制約テーブルの構築。

**主要関数:**
- `load_jacred(base_path="/tmp/JacRED/") -> dict`:
  - train/dev/test の3分割JSONと、メタデータ（rel2id, ent2id, rel_info）を読み込む
  - 出力: `{"train": [...], "dev": [...], "test": [...], "rel2id": {...}, "ent2id": {...}, "rel_info": {...}}`
- `doc_to_text(doc) -> str`:
  - トークン化された文（`doc["sents"]`）を平文テキストに変換する。各文のトークンを結合し、さらに全文を結合する
- `char_count(doc) -> int`:
  - 文書の総文字数を計算する（全トークンの文字数合計）
- `select_dev_docs(dev_data, n=10) -> list`:
  - devセットから文字数順にソートし、量子位置で `n` 文書を選択する層化サンプリング
- `select_few_shot(train_data) -> dict`:
  - 訓練データからfew-shot例に適した文書を選択する（150-250文字、5-12エンティティ、3-15ラベル）
- `format_few_shot_output(doc) -> dict`:
  - JacRED文書のvertexSetとlabelsから、EXTRACTION_SCHEMAに準拠したJSON形式の出力例を生成する
- `build_constraint_table(train_data) -> dict`:
  - 訓練データ全体から、各関係Pコードに対する観測済み `(head_type, tail_type)` ペアの集合を構築する

### 9.3 `llm_client.py` -- Gemini API呼び出し

**目的**: Google Gemini APIの呼び出し、Structured Outputs対応、リトライロジック。

**主要関数・定数:**
- `MODEL = "gemini-3-flash-preview"`: 使用するモデルID（変更時はここを編集）
- `load_api_key(env_path) -> str`: `.env` ファイルから `GEMINI_API_KEY` を読み込む
- `create_client(api_key) -> genai.Client`: Geminiクライアントを生成する
- `call_gemini(client, system_prompt, user_prompt, response_schema, temperature=0.2, max_retries=3) -> dict`:
  - Gemini APIを呼び出し、Structured OutputsでJSON応答を取得してパース済み辞書として返す
  - `GenerateContentConfig` に `response_mime_type="application/json"` と `response_schema` を設定
  - `ThinkingConfig(thinking_budget=2048)` がハードコードされている（変更時はここを編集）
  - 失敗時は指数バックオフ（2^(attempt+1) 秒）でリトライ

### 9.4 `prompts.py` -- プロンプトテンプレート

**目的**: 全LLM呼び出し用のプロンプト構築ロジック。

**主要定数:**
- `RELATION_JAPANESE`: 35種類の関係PコードからJapanese descriptionへのマッピング辞書
- `ENTITY_TYPES_JAPANESE`: 8種類のエンティティタイプからJapanese descriptionへのマッピング辞書

**主要関数:**
- `build_system_prompt(rel_info) -> str`: エンティティタイプ・関係タイプを含むシステムプロンプトを構築する。`rel_info` は `{Pコード: 英語名}` の辞書（JacREDメタデータ由来）
- `build_extraction_prompt(doc_text, few_shot_text, few_shot_output, mode="baseline") -> str`: 抽出用ユーザプロンプトを構築する。`mode` パラメータで以下の4つのモードを切り替える:
  - `"baseline"`: 追加指示なし（Baseline用）
  - `"recall"`: Recall重視の指示を追加（Majority Voting Pass 1用）
  - `"cross_sentence"`: 複数文間関係重視の指示を追加（Majority Voting Pass 2用）
  - `"structural"`: 構造的関係重視の指示を追加（Majority Voting Pass 3用）
- `build_verification_prompt(doc_text, candidates, entity_map, rel_info) -> str`: Stage 2検証用プロンプトを構築する。各候補トリプルのhead名・tail名・Pコード・英語名・日本語定義・evidence を含む

### 9.5 `extraction.py` -- 抽出ロジック

**目的**: Baseline・Majority Voting条件の抽出パイプライン全体を実装する。

**主要クラス:**
- `Triple`: データクラス。抽出されたトリプルを表現する
  - フィールド: `head`（エンティティID）, `head_name`, `head_type`, `relation`（Pコード）, `tail`, `tail_name`, `tail_type`, `evidence`

**主要関数:**
- `run_baseline(doc, few_shot, client, schema_info) -> (entities, triples)`:
  - Baseline条件を1文書に対して実行する。システムプロンプト構築 → ユーザプロンプト構築（mode="baseline"） → LLM呼び出し → パース → フィルタ
- `run_majority_voting(doc, few_shot, client, schema_info, constraint_table) -> (entities, triples, stats)`:
  - Majority Voting条件を1文書に対して実行する。3パス（recall, cross_sentence, structural） → エンティティ・トリプルのUNION集約 → 不正ラベル/タイプフィルタ → Stage 2バッチ検証 → domain/range制約適用
  - `stats` にはパイプライン各段階の候補数を記録:
    - `pass_recall`: パス1の候補数
    - `pass_cross_sentence`: パス2の候補数
    - `pass_structural`: パス3の候補数
    - `union_candidates`: UNION後の重複排除済み候補数
    - `stage2_kept`: Stage 2検証通過数
    - `after_constraints`: domain/range制約適用後の最終数
- `run_proposed(doc, few_shot, client, schema_info, constraint_table) -> (entities, triples, stats)`:
  - Two-Stage（候補生成 + 検証）条件。ベースリポジトリとの互換性のために残存
- `filter_invalid_labels(triples, valid_relations) -> list[Triple]`:
  - 不正なPコードを持つトリプルを除去する
- `filter_invalid_entity_types(triples, valid_types) -> list[Triple]`:
  - 不正なエンティティタイプを持つトリプルを除去する
- `apply_domain_range_constraints(triples, constraint_table) -> list[Triple]`:
  - 訓練データで未観測の `(head_type, tail_type)` ペアを持つトリプルを除去する
- `_verify_candidates(doc, candidates, entity_id_to_name, client, schema_info, batch_size=10) -> list[Triple]`:
  - Stage 2のバッチ検証を実行する。候補をbatch_size件ずつに分割し、各バッチに対して検証プロンプトを送信する
- `_parse_extraction_result(result) -> (entities, triples)`:
  - LLMの抽出出力（JSON辞書）をエンティティリストとTripleリストにパースする

### 9.6 `evaluation.py` -- 評価ロジック

**目的**: エンティティアライメントとP/R/F1の算出。

**主要関数:**
- `align_entities(predicted_entities, gold_vertex_set) -> dict[str, int]`:
  - 予測エンティティをGold vertexSetにアライメントする（3パスマッチング: 完全一致 → 正規化一致 → 部分文字列一致）
  - 出力: `{予測エンティティID: Gold vertexSetインデックス}`
- `evaluate_relations(predicted_triples, gold_labels, entity_alignment) -> dict`:
  - アライメント結果を用いて予測トリプルをGoldラベルと照合し、TP/FP/FN/P/R/F1 を算出する
  - FP詳細（理由: `entity_not_aligned` or `wrong_relation`）とFN詳細を含む
- `aggregate_results(per_doc) -> dict`:
  - 文書別結果リストからマイクロ平均のP/R/F1を算出する

### 9.7 `schemas.py` -- JSON Schema定義

**目的**: Gemini Structured Outputs用のJSON Schema定義。

**定数:**
- `EXTRACTION_SCHEMA`: 抽出用スキーマ。`entities`（id, name, typeの配列）と `relations`（head, relation, tail, evidenceの配列）を要求する
- `VERIFICATION_SCHEMA`: 検証用スキーマ。`decisions`（candidate_index, keepの配列）を要求する

### 9.8 `results.json` -- 最新の実験結果

**目的**: 最後に実行された実験の全結果をJSON形式で保存する。

**構造:**
```json
{
  "experiment": {
    "model": "gemini-3-flash-preview",
    "num_docs": 10,
    "few_shot_doc": "スタッド (バンド)",
    "timestamp": "2026-02-04T13:30:55.497014"
  },
  "conditions": {
    "baseline": {
      "per_doc": [{"title": "...", "precision": 0.88, ...}, ...],
      "aggregate": {"precision": 0.31, "recall": 0.20, "f1": 0.24, ...}
    },
    "majority_voting": {
      "per_doc": [{"title": "...", "precision": 0.41, ..., "stage_stats": {...}}, ...],
      "aggregate": {"precision": 0.31, "recall": 0.26, "f1": 0.28, ...}
    }
  }
}
```

---

## 10. 参考文献

1. Ma, Y., Tanaka, J., & Araki, M. **"Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer."** *Proceedings of LREC-COLING 2024.*
   - JacREDデータセットの構築論文。本実験のデータセット。

2. Yao, Y., Ye, D., Li, P., Han, X., Lin, Y., Liu, Z., Liu, Z., Huang, L., Zhou, J., & Sun, M. **"DocRED: A Large-Scale Document-Level Relation Extraction Dataset."** *Proceedings of ACL 2019.*
   - JacREDの基となった英語DocREデータセット。

3. Tan, C., Zhao, W., Wei, Z., & Huang, X. **"Document-level Relation Extraction: A Survey."** *arXiv preprint, 2023.*
   - 文書レベル関係抽出のサーベイ論文。

4. Li, D., Liu, Y., & Sun, M. **"A Survey on LLM-based Generative Information Extraction."** *arXiv preprint, 2024.*
   - LLMによる情報抽出のサーベイ論文。

5. Giorgi, J., Bader, G., & Wang, B. **"End-to-end Named Entity Recognition and Relation Extraction using Pre-trained Language Models."** *arXiv preprint, 2019.*
   - 事前学習言語モデルを用いたend-to-end NER+RE。

6. Dagdelen, J., Dunn, A., Lee, S., Walker, N., Rosen, A., Ceder, G., Persson, K., & Jain, A. **"Structured information extraction from scientific text with large language models."** *Nature Communications, 2024.*
   - LLMによる科学文献からの構造化情報抽出。

7. Willard, B., & Louf, R. **"Generating Structured Outputs from Language Models."** *arXiv preprint, 2025.*
   - 言語モデルからの構造化出力生成手法。

8. Harnoune, A., Rhanoui, M., Asri, B., Zellou, A., & Yousfi, S. **"Information extraction pipelines for knowledge graphs."** *Knowledge and Information Systems (Springer), 2022.*
   - 知識グラフ構築のための情報抽出パイプライン。

9. Mintz, M., Bills, S., Snow, R., & Jurafsky, D. **"Distant supervision for relation extraction without labeled data."** *Proceedings of ACL 2009.*
   - ラベルなしデータからの遠隔教師あり関係抽出。

10. Lu, Y., Liu, Z., & Huang, L. **"Cross-Lingual Structure Transfer for Relation and Event Extraction."** *Proceedings of ACL 2019.*
    - 言語間構造転移による関係・イベント抽出。
