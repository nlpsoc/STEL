# Annotation Results 

```00_nsubs_sentences.txt``` includes 100 number substitution sentences (one per line) and ```00_nsubs_translated_sentences.txt``` the translations thereof.

```QUAD-full_annotations.tsv``` contains the human annotations for all tasks in quadruple setup. The column ```Correct Alternative``` has the value 1 or 2. 1 corresponds to the correct ordering ```Alternative 1.1``` and then ```Alternative 1.2```. ```# Votes out of 5 for Correct Alternative``` are the number out of 5 annotators that selected the correct ordering. ```style type``` contains the style dimension/characteristic in question (i.e., simplicity or formality). ```ID``` contains information about the generation from the original dataset.

```QUAD-subsample_annotations.tsv``` is a subsample of ```QUAD-full_annotations.tsv``` containing only those tasks that were annotated in the quadruple as well as the triple setup.

```QUAD-full_agreement_accuracy.log``` contains the logging of the agreement and accuracy results on the full quadruple setup. ```QUAD-subsample_agreement_accuracy.log``` contains the logging of the agreement and accuracy results of the quadruple setup on the subsample.

```TRIP-subsample_annotations.tsv``` contains the human annotations for the subsample of tasks in the triple setup.

```TRIP-subsample_agreement_accuracy.log``` contains the logging of the agreement and accuracy results on the triple setup subsample.

