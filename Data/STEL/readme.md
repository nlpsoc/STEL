# Task instances

Can be found in ```Data/STEL``` in the tsv files `characteristics/quad_questions_char_contraction.tsv`, `characteristics/quad_questions_char_substitution.tsv` and  `dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv`. 

Each of the files have the following structure:

```
 id	| Anchor 1 | Anchor 2 | Alternative 1.1 | Alternative 1.2 | Correct Alternative | # Votes out of 5 for Correct Alternative	| ID	| style type  
0	| The line's name derives from its use in the Medieval French Roman d'Alexandre of 1170, although it'd already been used several decades earlier in Le Pèlerinage de Charlemagne.	| The line's name derives from its use in the Medieval French Roman d'Alexandre of 1170, although it had already been used several decades earlier in Le Pèlerinage de Charlemagne.	| It is also known as the Little Apocalypse because it includes the use of apocalyptic language, and it includes Jesus' warning to his followers that they will suffer tribulation and persecution before the ultimate triumph of the Kingdom of God.	It is also known as the Little Apocalypse because it includes the use of apocalyptic language, and it includes Jesus' warning to his followers that they'll suffer tribulation and persecution before the ultimate triumph of the Kingdom of God.	| 2	|	| QQ_ction-0-wiki-0_wiki-73-ction-73--1		| contraction
```

Where `Anchor 1`,`Anchor 2`, `Alternative 1.1`,`Alternative 1.2` correspond to Anchor 1, Anchor 2, Alternative Sentence 1 and Alternative Sentence 2 respectively (see Figure 1 in the paper). `Correct Alternative` is 1 if the correct order is S1-S2 and 2 if the correct order is S2-S1 (c.f. paper). `style type` takes the values `contraction`, `nbr_substitution`, `formality` and `simplicity` for the contraction, number substitution, formal/informal and simple/complex style dimension.  `# Votes out of 5 for Correct Alternative` only has non empty values for the tsv files including the annotations (e.g., `_QUAD-full_annotations.tsv`). `ID` includes the generated id for that task instance. They always start with  `QQ_`.  Then they include information of the underlying sentence pairs. For simple/complex and formal/informal it corresponds to the lines or ids in the original datasets. For contraction and number substitution they correspond  to the id in the underlying list.
