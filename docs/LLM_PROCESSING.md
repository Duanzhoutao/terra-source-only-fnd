# LLM Processing Details

TERRA uses the LLM as an offline event-slot extractor, not as an online
classifier. The LLM is not asked to predict whether a news item is fake or real,
and target-domain labels, target-domain identity, target-domain statistics, and
test-time predictions are not provided to the LLM processing pipeline.

## Fixed Processing Configuration

The FND5-LOCKED-V1 event-slot artifacts were generated before model training
and then frozen for all training, validation, and test-time evaluation steps.

```text
model = doubao-seed-2.0-pro
temperature = 0
max_input_chars = 4000
max_output_tokens = 512
prompt_version = fnd3_event_slots_v1_0
```

The API endpoint and authentication details are not required for reproducing
the released TERRA scoring package, because the released package starts from
fixed de-identified expert-score tables.

## Prompt Template

The processing pipeline uses a fixed prompt template. `{clean_text}` denotes
the normalized news text supplied to the extractor.

```text
Extract one neutral, check-worthy event structure from the news text below.
Rules:
1. Use only the given text; do not add outside knowledge.
2. Do not judge truthfulness. Do not use words such as fake, false, true,
hoax, rumor, debunked, verified, untrue, misleading, or fact-check in any
string field.
3. If the input contains those words, rewrite neutrally as related claim,
related information, related image, or related statement.
4. claim_text should be the core check-worthy claim, no more than 25 English
words when possible.
5. Unknown string fields should be empty strings; unknown category fields
should be "unknown".
6. emotion_intensity must be 0, 1, 2, or 3.
7. All flags must be 0 or 1.
8. Output one JSON object only, no Markdown and no explanation.

JSON schema:
{
  "claim_text": "",
  "subject": "",
  "action": "",
  "object": "",
  "time": "",
  "location": "",
  "event_type": "health|politics|entertainment|public_safety|science_technology|finance|society|international|other|unknown",
  "claim_specificity": "concrete|vague|opinion|unknown",
  "evidence_need": "low|medium|high|unknown",
  "emotion_intensity": 0,
  "sensational_style_flag": 0,
  "absolute_wording_flag": 0,
  "uncertainty_or_hearsay_flag": 0,
  "llm_refusal_flag": 0,
  "llm_parse_error_flag": 0
}

News text: {clean_text}
```

## Output Fields

The parsed event-slot table contains:

- `claim_text`, `subject`, `action`, `object`, `time`, and `location`.
- `event_type`, `claim_specificity`, and `evidence_need`.
- `emotion_intensity`, `sensational_style_flag`,
  `absolute_wording_flag`, and `uncertainty_or_hearsay_flag`.
- LLM processing flags, including refusal, parse, missing-slot, and
  direct-truthfulness-wording indicators.
- `event_clean_mask`, which marks whether the event slots can be used as clean
  structured event evidence.

The clean event fields are also linearized into an event text representation
for local event-semantic encoding. The local encoder is separate from the LLM
call.

## Parsing And Freezing

The parser strips optional JSON fences, requires a JSON object, normalizes
invalid category values to `unknown`, clips numeric flags to the allowed range,
and records processing flags for outputs that cannot be used as clean event
evidence. Samples with parsing failure, refusal-style responses, missing core
slots, or direct truthfulness wording are retained in the metadata but are not
used as clean event text.

The resulting event-slot features, event-clean mask, and downstream expert
scores are generated before model selection. They remain fixed during TERRA
training, source-validation threshold selection, and target-test evaluation.

## Release Boundary

This public repository releases the prompt template, output schema, parsing
boundary, model-side use of `event_clean_mask`, de-identified expert-score
inputs, and reproduction scripts. It does not redistribute raw news text, raw
LLM JSON outputs, API credentials, model checkpoints, or private training logs.
