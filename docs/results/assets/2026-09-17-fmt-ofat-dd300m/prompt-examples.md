# Exact model inputs, one item per format setting (DataDecide 300M train sweep, item Mercury_SC_400676 / first item)

Each block is the verbatim `context` string the model is scored on; the scored continuations follow it. Nothing is added or removed. The gold continuation is marked.

## RC · canonical

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

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
Answer:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · sep=colon_newline

```text
Question:
Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
Answer:
 food

Question:
When a switch is used in an electrical circuit, the switch can
Answer:
 stop and start the flow of current.

Question:
Which of the following is an example of an assistive device?
Answer:
 contact lens

Question:
Rocks are classified as igneous, metamorphic, or sedimentary according to
Answer:
 how they formed

Question:
A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
Answer:
 neutralizes digestive acid.

Question:
When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
Answer:

```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · sep=dash

```text
Question - Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
Answer - food

Question - When a switch is used in an electrical circuit, the switch can
Answer - stop and start the flow of current.

Question - Which of the following is an example of an assistive device?
Answer - contact lens

Question - Rocks are classified as igneous, metamorphic, or sedimentary according to
Answer - how they formed

Question - A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
Answer - neutralizes digestive acid.

Question - When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
Answer -
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · desc=q_a

```text
Q: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
A: food

Q: When a switch is used in an electrical circuit, the switch can
A: stop and start the flow of current.

Q: Which of the following is an example of an assistive device?
A: contact lens

Q: Rocks are classified as igneous, metamorphic, or sedimentary according to
A: how they formed

Q: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
A: neutralizes digestive acid.

Q: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
A:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · desc=input_output

```text
Input: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
Output: food

Input: When a switch is used in an electrical circuit, the switch can
Output: stop and start the flow of current.

Input: Which of the following is an example of an assistive device?
Output: contact lens

Input: Rocks are classified as igneous, metamorphic, or sedimentary according to
Output: how they formed

Input: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
Output: neutralizes digestive acid.

Input: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
Output:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · desc=random_vocab

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
ceil Ai aggress: food

Question: When a switch is used in an electrical circuit, the switch can
ceil Ai aggress: stop and start the flow of current.

Question: Which of the following is an example of an assistive device?
ceil Ai aggress: contact lens

Question: Rocks are classified as igneous, metamorphic, or sedimentary according to
ceil Ai aggress: how they formed

Question: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
ceil Ai aggress: neutralizes digestive acid.

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
ceil Ai aggress:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · desc=random_phrase

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
View Mobile Site: food

Question: When a switch is used in an electrical circuit, the switch can
View Mobile Site: stop and start the flow of current.

Question: Which of the following is an example of an assistive device?
View Mobile Site: contact lens

Question: Rocks are classified as igneous, metamorphic, or sedimentary according to
View Mobile Site: how they formed

Question: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
View Mobile Site: neutralizes digestive acid.

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
View Mobile Site:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · shots=1

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
Answer: food

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
Answer:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## RC · shots=0

```text
Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
Answer:
```

Scored continuations: ` osmosis`, ` diffusion` (gold), ` cell activity`, ` cell transport`

## MC · canonical

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

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
 A. osmosis
 B. diffusion
 C. cell activity
 D. cell transport
Answer:
```

Scored continuations: ` A`, ` B` (gold), ` C`, ` D`

## MC · choice_sep=tab

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
 A.	carbon dioxide
 B.	food
 C.	protection
 D.	water
Answer: B

Question: When a switch is used in an electrical circuit, the switch can
 A.	cause the charge to build.
 B.	increase and decrease the voltage.
 C.	cause the current to change direction.
 D.	stop and start the flow of current.
Answer: D

Question: Which of the following is an example of an assistive device?
 A.	contact lens
 B.	motorcycle
 C.	raincoat
 D.	coffee pot
Answer: A

Question: Rocks are classified as igneous, metamorphic, or sedimentary according to
 A.	their color
 B.	their shape
 C.	how they formed
 D.	the minerals they contain
Answer: C

Question: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
 A.	has a pleasant flavor.
 B.	is inexpensive to produce.
 C.	neutralizes digestive acid.
 D.	occurs naturally in the body.
Answer: C

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
 A.	osmosis
 B.	diffusion
 C.	cell activity
 D.	cell transport
Answer:
```

Scored continuations: ` A`, ` B` (gold), ` C`, ` D`

## MC · choice_sep=none

```text
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
 A.carbon dioxide
 B.food
 C.protection
 D.water
Answer: B

Question: When a switch is used in an electrical circuit, the switch can
 A.cause the charge to build.
 B.increase and decrease the voltage.
 C.cause the current to change direction.
 D.stop and start the flow of current.
Answer: D

Question: Which of the following is an example of an assistive device?
 A.contact lens
 B.motorcycle
 C.raincoat
 D.coffee pot
Answer: A

Question: Rocks are classified as igneous, metamorphic, or sedimentary according to
 A.their color
 B.their shape
 C.how they formed
 D.the minerals they contain
Answer: C

Question: A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
 A.has a pleasant flavor.
 B.is inexpensive to produce.
 C.neutralizes digestive acid.
 D.occurs naturally in the body.
Answer: C

Question: When a bottle of strong perfume is opened at the front of a classroom, the smell gradually spreads throughout the entire room. Which process explains this occurrence?
 A.osmosis
 B.diffusion
 C.cell activity
 D.cell transport
Answer:
```

Scored continuations: ` A`, ` B` (gold), ` C`, ` D`

