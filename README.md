# KG Extraction from JacRED: Multi-Instance Majority Voting

## 1. Background

### JacRED Dataset: Japanese Document-level Relation Extraction Dataset

- **Source**: https://github.com/YoumiMa/JacRED (clone to `/tmp/JacRED`)
- **Splits**: train (1400 docs), dev (300 docs), test (300 docs)
- **Format**: Each document has:
  - `title`: document title
  - `sents`: tokenized sentences (list of list of tokens)
  - `vertexSet`: entities with mentions (list of entity groups, each containing mention dicts with `name`, `type`, `sent_id`, `pos`)
  - `labels`: relations as `{h, t, r, evidence}` where h/t are vertexSet indices and r is a Wikidata P-code
- **9 entity types**: PER, ORG, LOC, ART, DAT, TIM, MON, %, NA
- **35 relation types**: Wikidata P-codes (P131, P27, P569, P570, P19, P20, P40, P3373, P26, P1344, P463, P361, P6, P127, P112, P108, P137, P69, P166, P170, P175, P123, P1441, P400, P36, P1376, P276, P937, P155, P156, P710, P527, P1830, P121, P674)
- **Statistics**: Avg ~17 entities/doc, avg ~20 relations/doc, avg ~253 chars/doc

## 2. Base Implementation (already provided)

The following files implement the baseline extraction:

- **run_experiment.py**: Main orchestrator. Loads data, runs conditions (Baseline, Majority Voting), prints comparison table, saves results.json.
- **data_loader.py**: Data loading from JacRED JSON files, document selection (10 stratified from dev split), few-shot example selection, domain/range constraint table construction from training data.
- **llm_client.py**: Gemini API wrapper using `google-genai` library with Structured Outputs (`response_mime_type="application/json"` + `response_schema`), ThinkingConfig, and retry logic.
- **prompts.py**: All prompt templates including system prompt with 35 relation types defined in Japanese, extraction prompt (baseline and recall-oriented modes), and verification prompt for Stage 2.
- **extraction.py**: Conditions:
  - `run_baseline()`: Single LLM call extraction with post-filtering (invalid labels, invalid entity types).
  - `run_majority_voting()`: Multi-instance majority voting. Runs 3 extraction passes with diverse prompts, takes the union, verifies via Stage 2, and applies domain/range constraints.
- **evaluation.py**: Entity alignment (3-pass: exact match -> normalized match -> substring match) and micro-averaged P/R/F1 computation.
- **schemas.py**: JSON schemas for Gemini Structured Outputs (extraction schema with entities+relations, verification schema with decisions).

### Key code details

- Entity alignment maps predicted entity IDs to gold vertexSet indices using 3-pass matching.
- Domain/range constraints are built from training data: for each relation P-code, store the set of (head_type, tail_type) pairs observed.
- Verification (Stage 2) processes candidates in batches of 10, asking the LLM to judge each candidate.

## 3. Baseline Results (for comparison)

```
Model: gemini-3-flash-preview (thinking_budget=0)
              Precision   Recall     F1    TP    FP    FN
Baseline           0.26     0.16   0.20    24    70   124
```

**Key issue**: Recall is very low (0.16). Most of the 148 gold relations are missed. The LLM fails to extract many valid relations in a single pass.

## 4. Environment Setup

```bash
# Clone JacRED dataset
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# Install dependencies
pip install google-genai openai

# Set API key
export GEMINI_API_KEY="<your-key>"
```

## 5. API Configuration

- **Model**: `gemini-3-flash-preview` (recommended) or `gemini-2.0-flash`
- **Structured Outputs**: `response_mime_type="application/json"` + `response_schema` dict
- **Temperature**: 0.2
- **ThinkingConfig**: `thinking_budget=0` for speed, `2048` for quality
- Configuration is in `llm_client.py` (the `MODEL` constant and `call_gemini()` function)

## 6. Task: Implement Multi-Instance Majority Voting

### Goal

Run 3-5 independent extractions with diverse prompts, aggregate results via union to improve Recall.

### Design

1. **Run 3 extraction passes per document**, each with a different "search strategy" prompt:
   - **Pass 1**: Standard exhaustive extraction (same as current baseline prompt)
   - **Pass 2**: Focus on cross-sentence relations. Add instruction: `"複数文にまたがる関係に注目してください。異なる文に登場するエンティティ間の関係を特に注意して抽出してください。"`
   - **Pass 3**: Focus on structural/organizational relations. Add instruction: `"PartOf/HasPart, MemberOf等の構造的関係に注目してください。組織の階層関係、地理的包含関係、作品の構成要素などを特に注意して抽出してください。"`

2. **Aggregation strategy**: UNION -- keep a triple if it was extracted in at least 1 pass. Add `support_count` metadata (how many passes found it).

3. **After union**, apply Stage 2 verification (same as the Two-Stage method) to filter false positives.

4. **Also apply domain/range constraint filtering** after verification.

### Implementation Details

- **Add `run_majority_voting()` function in `extraction.py`**:
  - Takes the same arguments as `run_proposed()` plus prompt variants
  - Runs 3 extraction passes, each calling `call_gemini()` with a different user prompt
  - Merges results: union of all triples (deduplicate by (head_name, relation, tail_name))
  - Runs verification on the union set
  - Applies domain/range constraints
  - Returns entities (union), final triples, and stats dict

- **Add 2-3 new prompt variants in `prompts.py`**:
  - `build_extraction_prompt()` already supports a `mode` parameter
  - Add modes: `"cross_sentence"` and `"structural"` with the respective Japanese instructions
  - Or create a new function `build_diverse_extraction_prompts()` that returns a list of prompts

- **Update `run_experiment.py`** to add a second condition (in addition to Baseline):
  - `"Condition 2: Majority Voting (3-pass + verify)"`
  - Print comparison table with both conditions

- **Deduplication logic**: Two triples are considered duplicates if they have the same (head_name_normalized, relation, tail_name_normalized). When merging, keep the entity list from the pass that found the most entities.

### Expected Improvement

- **Recall should increase** because different prompts will surface different relations that a single prompt misses.
- **Precision should be maintained** via the verification step filtering out false positives from the union.
- **Cost**: ~3x baseline per document (3 extraction calls + verification calls for the union set).

### Evaluation

- Same P/R/F1 computation on the same 10 dev documents
- Report per-document results and aggregate metrics
- Compare: Baseline vs Majority Voting

### Output Format

The final comparison table should look like:
```
              Precision   Recall     F1    TP    FP    FN
Baseline           ...      ...    ...   ...   ...   ...
MajorityVote       ...      ...    ...   ...   ...   ...
```
