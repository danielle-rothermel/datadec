# Instruction pool report (36 candidates, 2 formulations; near-duplicate threshold 0.8)

## MC (18 candidates; 0 near-duplicate groups absorbing 0 candidates; 18 distinct)

Pairwise similarity: median ratio 0.44, max 0.78; median Jaccard 0.32.

### Tag counts by factor

| tag | all | op=plain | op=rule | op=short | st=commonsense | st=eliminate | st=none | aware | not aware |
|---|---|---|---|---|---|---|---|---|---|
| answer_form_letter | 18 | 6 | 6 | 6 | 6 | 6 | 6 | 9 | 9 |
| answer_form_text | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| elimination | 6 | 2 | 2 | 2 | 0 | 6 | 0 | 3 | 3 |
| commonsense | 6 | 2 | 2 | 2 | 6 | 0 | 0 | 3 | 3 |
| no_explanation | 16 | 5 | 6 | 5 | 4 | 6 | 6 | 7 | 9 |
| mentions_science | 6 | 3 | 3 | 0 | 2 | 3 | 1 | 4 | 2 |
| descriptor_anchor | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| letters_listed | 10 | 3 | 5 | 2 | 4 | 2 | 4 | 6 | 4 |
| modal_or_rule | 2 | 0 | 2 | 0 | 1 | 0 | 1 | 2 | 0 |
| mentions_multiple_choice | 14 | 5 | 4 | 5 | 4 | 6 | 4 | 8 | 6 |

Words per instruction: op=plain: 28, op=rule: 30, op=short: 15; st=commonsense: 23, st=eliminate: 30, st=none: 19; aware: 27, not aware: 21.

### Length buckets

medium=12 long=3 short=3

### Near-duplicate groups


### Candidates

- **c000** [plain/eliminate/plain, 28 words]: Read each multiple-choice science question, first rule out all the clearly incorrect options, then choose the best remaining answer and respond with only the letter of that choice.
- **c001** [plain/eliminate/aware, 48 words]: First read the question carefully, then consider each of the four options and rule out the ones that are clearly incorrect; from the remaining choices, select the single best answer based on science knowledge, and respond with only the letter (A, B, C, or D) of that answer.
- **c002** [plain/commonsense/plain, 24 words]: For each multiple-choice question, select the letter of the option that is the most common-sense correct answer, and respond with just that single letter.
- **c003** [plain/commonsense/aware, 20 words]: For each question, choose the most common-sense correct answer and respond only with its letter (A, B, C, or D).
- **c004** [plain/none/plain, 20 words]: For each question, choose the correct answer from the options provided and respond with only the letter of that option.
- **c005** [plain/none/aware, 26 words]: For each multiple-choice science question, choose the correct option and respond with only its letter (A, B, C, or D), with no explanation or additional text.
- **c006** [rule/eliminate/plain, 30 words]: First rule out all of the incorrect options, then choose the best remaining option and answer by writing only the letter (A, B, C, or D) of the correct choice.
- **c007** [rule/eliminate/aware, 34 words]: For each multiple-choice science question, carefully consider the scientific facts, rule out the three options that are incorrect or less accurate, and then respond with only the letter of the single best remaining answer.
- **c008** [rule/commonsense/plain, 23 words]: For each question, write only the single letter (A, B, C, or D) that corresponds to the most scientifically correct and common-sense answer.
- **c009** [rule/commonsense/aware, 43 words]: You must answer each multiple-choice science question by selecting the single option (A, B, C, or D) that best fits the question, relying on basic scientific knowledge and everyday common sense, and respond with only the letter of the one most common-sense answer.
- **c010** [rule/none/plain, 25 words]: For each question, choose the single best answer and respond with only its letter (A, B, C, or D), with no other words or explanation.
- **c011** [rule/none/aware, 25 words]: Always answer each question by selecting the single best option and writing only its capital letter (A, B, C, or D), with no other text.
- **c012** [short/eliminate/plain, 19 words]: For each question, first eliminate the wrong options, then choose and write only the letter of the best answer.
- **c013** [short/eliminate/aware, 22 words]: For each question, first rule out the clearly wrong options, then pick the best remaining choice and answer with only its letter.
- **c014** [short/commonsense/plain, 14 words]: Answer each question by writing only the letter (A–D) of the most common-sense choice.
- **c015** [short/commonsense/aware, 15 words]: Choose the most common-sense correct option for each question and answer with its letter only.
- **c016** [short/none/plain, 8 words]: Write only the letter of the correct answer.
- **c017** [short/none/aware, 12 words]: Answer each question with only the letter (A–D) of the correct option.

## RC (18 candidates; 1 near-duplicate groups absorbing 1 candidates; 17 distinct)

Pairwise similarity: median ratio 0.42, max 0.84; median Jaccard 0.24.

### Tag counts by factor

| tag | all | op=plain | op=rule | op=short | st=commonsense | st=eliminate | st=none | aware | not aware |
|---|---|---|---|---|---|---|---|---|---|
| answer_form_letter | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| answer_form_text | 15 | 6 | 6 | 3 | 5 | 5 | 5 | 8 | 7 |
| elimination | 4 | 1 | 1 | 2 | 0 | 4 | 0 | 1 | 3 |
| commonsense | 6 | 2 | 2 | 2 | 6 | 0 | 0 | 3 | 3 |
| no_explanation | 13 | 5 | 5 | 3 | 2 | 5 | 6 | 7 | 6 |
| mentions_science | 8 | 3 | 4 | 1 | 3 | 4 | 1 | 5 | 3 |
| descriptor_anchor | 2 | 1 | 1 | 0 | 0 | 0 | 2 | 2 | 0 |
| letters_listed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| modal_or_rule | 5 | 0 | 5 | 0 | 2 | 2 | 1 | 3 | 2 |
| mentions_multiple_choice | 6 | 2 | 2 | 2 | 0 | 6 | 0 | 3 | 3 |

Words per instruction: op=plain: 26, op=rule: 34, op=short: 14; st=commonsense: 23, st=eliminate: 36, st=none: 17; aware: 27, not aware: 23.

### Length buckets

long=6 medium=8 short=4

### Near-duplicate groups

- rc-g4: c004, c016

### Candidates

- **c000** [plain/eliminate/plain, 38 words]: Read each science question and its answer choices, rule out the choices that are clearly incorrect, then select the best remaining choice and write it as your answer using only the words of that choice and no explanation.
- **c001** [plain/eliminate/aware, 44 words]: Answer each science question by carefully reading the question and all answer choices, ruling out the options that are incorrect, and then selecting the single best remaining option. Respond with only the text of the correct answer, not its letter and not an explanation.
- **c002** [plain/commonsense/plain, 20 words]: Answer each question with the single most common-sense correct answer, expressed as a short word or phrase, without any explanation.
- **c003** [plain/commonsense/aware, 23 words]: Answer each question by giving the single most common-sense correct answer as a short phrase or sentence, with no explanation or extra text.
- **c004** [plain/none/plain, 11 words]: Answer the question with the correct short answer only, without explanation.
- **c005** [plain/none/aware, 23 words]: Answer each science question by writing only the short, direct answer (a word or phrase) after “Answer:”, with no explanation or extra text.
- **c006** [rule/eliminate/plain, 39 words]: Rule: For each multiple-choice science question, first carefully evaluate and rule out all incorrect options, then choose the single best remaining option and respond with only the exact text of that correct choice, with no extra words or explanation.
- **c007** [rule/eliminate/aware, 54 words]: You must answer multiple-choice science questions by first reading the question and all of the answer options, then ruling out every option that is clearly wrong. From the remaining options, choose the single best answer and respond with only the exact text of that answer choice, with nothing added or removed and no explanation.
- **c008** [rule/commonsense/plain, 31 words]: You must answer each science question with a brief, direct word or phrase (not an explanation), completing the sentence if needed, and always choose the single most common-sense, commonly expected answer.
- **c009** [rule/commonsense/aware, 34 words]: You must answer each science question with only a short word or phrase that is the most common-sense correct answer, as on a basic school science test, and give no explanation or extra text.
- **c010** [rule/none/plain, 22 words]: For each question, write only the correct answer as a short word or phrase, without repeating the question or adding any explanation.
- **c011** [rule/none/aware, 25 words]: Always answer each question after `Answer:` with a short, direct phrase stating only the correct answer, and do not include any explanation or extra text.
- **c012** [short/eliminate/plain, 19 words]: For each question, first rule out the incorrect options, then write only the correct answer as briefly as possible.
- **c013** [short/eliminate/aware, 21 words]: For each question, first rule out the wrong options, then answer with the single best remaining option as a brief phrase.
- **c014** [short/commonsense/plain, 13 words]: For each question, give the most common-sense correct answer in a few words.
- **c015** [short/commonsense/aware, 15 words]: Answer each science question with a short, direct phrase giving the most common-sense correct answer.
- **c016** [short/none/plain, 11 words] (dup of rc-g4): Answer each question with a brief, direct answer only, without explanation.
- **c017** [short/none/aware, 8 words]: Write only the correct answer, with no explanation.

