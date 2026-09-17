# Framed APE generations (2026-09-17): meta-prompt templates, swept strings, and all candidates after near-duplicate filtering

Proposers: `openai/gpt-5.1` and `openai/gpt-5.6-terra`, reasoning high, temperature 1, output limit unset, one call per cell,
seeds 5000-5017 (RC) and 6000-6017 (MC), identical across proposers. Near-duplicate filter: difflib ratio ≥ 0.8 on
lower-cased alphanumeric text, within (formulation, proposer); the lowest-id member of a group is kept as representative.

## 1. Meta-prompt templates (placeholders in braces)

### RC

```text
I gave a friend an instruction and {k} science questions. The friend read the instruction and wrote the correct answer to each question. Here are the question-answer pairs, exactly as the friend saw them:

{demos}

{reader}Write the instruction I gave my friend. It will be placed once at the top of a prompt, before these examples and then a new question, so it should say how to answer this kind of question in general. {operator} {stance}Reply with the instruction only.
```

### MC

```text
I gave a friend an instruction and {k} multiple-choice science questions. The friend read the instruction and wrote the letter of the correct choice for each question. Here are the question-answer pairs, exactly as the friend saw them:

{demos}

{reader}Write the instruction I gave my friend. It will be placed once at the top of a prompt, before these examples and then a new question, so it should say how to answer this kind of question in general. {operator} {stance}Reply with the instruction only.
```

### `{reader}` slot when model-aware (empty string otherwise)

```text
The friend is a language model, not a person: {card}. It never writes anything; it is scored by the likelihood it assigns to each candidate answer after the prompt, so the instruction can only help by changing which answer it finds most likely.


```

## 2. Swept strings

### `{operator}` (rewrite operator)

- `plain`: "Write it as a plain statement of what to do."
- `rule`: "Write it as a rule the reader must follow."
- `short`: "Make it as short as possible."

### `{stance}` (a trailing space is appended when non-empty)

- `eliminate`: "It should tell the reader to rule out the wrong options first."
- `commonsense`: "It should tell the reader to prefer the most common-sense answer."
- `none`: (empty)

### `{card}` (model-aware cells; from `datadec.po.model_cards`)

```text
The friend is a language model, not a person: name: allenai/DataDecide-dclm-baseline-300M; revision: step45000-seed-default; family: OLMo-style decoder-only transformer from the DataDecide suite (allenai); nominal parameters: 300M; pretraining data recipe: DCLM-Baseline; training regime: trained to 100 tokens per parameter (5x Chinchilla), sequence length 2048; architecture: 16 layers, d_model 1024, 16 heads, MLP ratio 8, 319,980,544 exact parameters; checkpoint: training step 45000 (final seed-default run). It never writes anything; it is scored by the likelihood it assigns to each candidate answer after the prompt, so the instruction can only help by changing which answer it finds most likely.
```

### `{k}` and `{demos}`

k = 5; the five curated OLMES ARC-Easy demonstrations (MCAS_2007_8_5189, Mercury_SC_401169, MCAS_2004_8_27, NYSEDREGENTS_2006_8_10, Mercury_7013388), rendered exactly as the scored prompt renders them:

RC rendering:

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
Answer: food

Question: When a switch is used in an electrical circuit, the switch can
Answer: stop and start the flow of current.

Question: Which of the following is an example of an assistive device?
Answer: contact lens

Question: Rocks are classified as igneous, metamorphic, or sedimentary according to
Answer: how they formed

Question: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
Answer: neutralizes digestive acid.
```

MC rendering:

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
 A. carbon dioxide
 B. food
 C. protection
 D. water
Answer: B

Question: When a switch is used in an electrical circuit, the switch can
 A. cause the charge to build.
 B. increase and decrease the voltage.
 C. cause the current to change direction.
 D. stop and start the flow of current.
Answer: D

Question: Which of the following is an example of an assistive device?
 A. contact lens
 B. motorcycle
 C. raincoat
 D. coffee pot
Answer: A

Question: Rocks are classified as igneous, metamorphic, or sedimentary according to
 A. their color
 B. their shape
 C. how they formed
 D. the minerals they contain
Answer: C

Question: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
 A. has a pleasant flavor.
 B. is inexpensive to produce.
 C. neutralizes digestive acid.
 D. occurs naturally in the body.
Answer: C
```

## 3. Generations

Cells are (operator / stance / aware). Candidates absorbed by a near-duplicate group are listed under their representative in italics.

### MC · openai/gpt-5.1 (18 distinct of 18)

- **c000** (plain / eliminate / plain): Read each multiple-choice science question, first rule out all the clearly incorrect options, then choose the best remaining answer and respond with only the letter of that choice.
- **c001** (plain / eliminate / aware): First read the question carefully, then consider each of the four options and rule out the ones that are clearly incorrect; from the remaining choices, select the single best answer based on science knowledge, and respond with only the letter (A, B, C, or D) of that answer.
- **c002** (plain / commonsense / plain): For each multiple-choice question, select the letter of the option that is the most common-sense correct answer, and respond with just that single letter.
- **c003** (plain / commonsense / aware): For each question, choose the most common-sense correct answer and respond only with its letter (A, B, C, or D).
- **c004** (plain / none / plain): For each question, choose the correct answer from the options provided and respond with only the letter of that option.
- **c005** (plain / none / aware): For each multiple-choice science question, choose the correct option and respond with only its letter (A, B, C, or D), with no explanation or additional text.
- **c006** (rule / eliminate / plain): First rule out all of the incorrect options, then choose the best remaining option and answer by writing only the letter (A, B, C, or D) of the correct choice.
- **c007** (rule / eliminate / aware): For each multiple-choice science question, carefully consider the scientific facts, rule out the three options that are incorrect or less accurate, and then respond with only the letter of the single best remaining answer.
- **c008** (rule / commonsense / plain): For each question, write only the single letter (A, B, C, or D) that corresponds to the most scientifically correct and common-sense answer.
- **c009** (rule / commonsense / aware): You must answer each multiple-choice science question by selecting the single option (A, B, C, or D) that best fits the question, relying on basic scientific knowledge and everyday common sense, and respond with only the letter of the one most common-sense answer.
- **c010** (rule / none / plain): For each question, choose the single best answer and respond with only its letter (A, B, C, or D), with no other words or explanation.
- **c011** (rule / none / aware): Always answer each question by selecting the single best option and writing only its capital letter (A, B, C, or D), with no other text.
- **c012** (short / eliminate / plain): For each question, first eliminate the wrong options, then choose and write only the letter of the best answer.
- **c013** (short / eliminate / aware): For each question, first rule out the clearly wrong options, then pick the best remaining choice and answer with only its letter.
- **c014** (short / commonsense / plain): Answer each question by writing only the letter (A–D) of the most common-sense choice.
- **c015** (short / commonsense / aware): Choose the most common-sense correct option for each question and answer with its letter only.
- **c016** (short / none / plain): Write only the letter of the correct answer.
- **c017** (short / none / aware): Answer each question with only the letter (A–D) of the correct option.

### MC · openai/gpt-5.6-terra (14 distinct of 18)

- **c000** (plain / eliminate / plain): Eliminate the incorrect choices, then write only the letter of the correct answer.
- **c001** (plain / eliminate / aware): First rule out the incorrect options, then choose the letter of the best remaining answer.
- **c002** (plain / commonsense / plain): Choose the most common-sense correct answer and reply with only its letter.
- **c003** (plain / commonsense / aware): Answer each question with the letter of the most common-sense answer.
  - *c014 (short / commonsense / plain), near-duplicate: Reply with the letter of the most common-sense answer.*
- **c004** (plain / none / plain): Write the letter of the correct answer choice.
  - *c016 (short / none / plain), near-duplicate: Answer with the letter of the correct choice.*
- **c005** (plain / none / aware): For each multiple-choice question, write the letter of the correct answer.
  - *c008 (rule / commonsense / plain), near-duplicate: For each multiple-choice question, reply with the letter of the most common-sense correct answer.*
  - *c010 (rule / none / plain), near-duplicate: For each question, write only the letter of the correct answer choice.*
- **c006** (rule / eliminate / plain): Rule out the wrong options first, then answer with the letter of the remaining correct choice.
- **c007** (rule / eliminate / aware): First, rule out the wrong answer choices, then choose the best remaining answer.
- **c009** (rule / commonsense / aware): Always choose the answer that makes the most common sense.
- **c011** (rule / none / aware): For each question, choose the correct answer and respond with only its letter.
- **c012** (short / eliminate / plain): Rule out wrong options first, then give the correct letter.
- **c013** (short / eliminate / aware): Rule out wrong options first.
- **c015** (short / commonsense / aware): Choose the most common-sense answer.
- **c017** (short / none / aware): Answer with the correct letter.

### RC · openai/gpt-5.1 (17 distinct of 18)

- **c000** (plain / eliminate / plain): Read each science question and its answer choices, rule out the choices that are clearly incorrect, then select the best remaining choice and write it as your answer using only the words of that choice and no explanation.
- **c001** (plain / eliminate / aware): Answer each science question by carefully reading the question and all answer choices, ruling out the options that are incorrect, and then selecting the single best remaining option. Respond with only the text of the correct answer, not its letter and not an explanation.
- **c002** (plain / commonsense / plain): Answer each question with the single most common-sense correct answer, expressed as a short word or phrase, without any explanation.
- **c003** (plain / commonsense / aware): Answer each question by giving the single most common-sense correct answer as a short phrase or sentence, with no explanation or extra text.
- **c004** (plain / none / plain): Answer the question with the correct short answer only, without explanation.
  - *c016 (short / none / plain), near-duplicate: Answer each question with a brief, direct answer only, without explanation.*
- **c005** (plain / none / aware): Answer each science question by writing only the short, direct answer (a word or phrase) after “Answer:”, with no explanation or extra text.
- **c006** (rule / eliminate / plain): Rule: For each multiple-choice science question, first carefully evaluate and rule out all incorrect options, then choose the single best remaining option and respond with only the exact text of that correct choice, with no extra words or explanation.
- **c007** (rule / eliminate / aware): You must answer multiple-choice science questions by first reading the question and all of the answer options, then ruling out every option that is clearly wrong. From the remaining options, choose the single best answer and respond with only the exact text of that answer choice, with nothing added or removed and no explanation.
- **c008** (rule / commonsense / plain): You must answer each science question with a brief, direct word or phrase (not an explanation), completing the sentence if needed, and always choose the single most common-sense, commonly expected answer.
- **c009** (rule / commonsense / aware): You must answer each science question with only a short word or phrase that is the most common-sense correct answer, as on a basic school science test, and give no explanation or extra text.
- **c010** (rule / none / plain): For each question, write only the correct answer as a short word or phrase, without repeating the question or adding any explanation.
- **c011** (rule / none / aware): Always answer each question after `Answer:` with a short, direct phrase stating only the correct answer, and do not include any explanation or extra text.
- **c012** (short / eliminate / plain): For each question, first rule out the incorrect options, then write only the correct answer as briefly as possible.
- **c013** (short / eliminate / aware): For each question, first rule out the wrong options, then answer with the single best remaining option as a brief phrase.
- **c014** (short / commonsense / plain): For each question, give the most common-sense correct answer in a few words.
- **c015** (short / commonsense / aware): Answer each science question with a short, direct phrase giving the most common-sense correct answer.
- **c017** (short / none / aware): Write only the correct answer, with no explanation.

### RC · openai/gpt-5.6-terra (10 distinct of 18)

- **c000** (plain / eliminate / plain): For each question, rule out the incorrect answer choices first, then choose the best remaining answer.
- **c001** (plain / eliminate / aware): Eliminate the wrong answer choices first, then select the best answer.
- **c002** (plain / commonsense / plain): Answer each question with the most common-sense answer.
  - *c003 (plain / commonsense / aware), near-duplicate: Answer each question with the most common-sense answer.*
  - *c008 (rule / commonsense / plain), near-duplicate: Answer each question with the most common-sense answer.*
  - *c009 (rule / commonsense / aware), near-duplicate: Always answer each question with the most common-sense answer.*
  - *c014 (short / commonsense / plain), near-duplicate: Answer with the most common-sense answer.*
- **c004** (plain / none / plain): Answer each question with the correct answer only.
  - *c005 (plain / none / aware), near-duplicate: Answer each science question with the correct answer.*
  - *c010 (rule / none / plain), near-duplicate: Answer each question with only the correct answer.*
  - *c011 (rule / none / aware), near-duplicate: Answer each question with only the correct answer.*
- **c006** (rule / eliminate / plain): Rule: First rule out the incorrect answer choices, then write only the correct answer.
- **c007** (rule / eliminate / aware): Rule out the wrong answer choices first, then give the correct answer.
- **c012** (short / eliminate / plain): Rule out wrong options, then give the correct answer.
- **c013** (short / eliminate / aware): Eliminate wrong options first.
- **c015** (short / commonsense / aware): Use common sense.
- **c016** (short / none / plain): Answer the question.
  - *c017 (short / none / aware), near-duplicate: Answer the question.*

