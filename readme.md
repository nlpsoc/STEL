**NOTE:** changing the batch size of BERT/RoBERTa changes performance. To get the highest possible performance, use an eval batch size of 1. 


Thank you for your interest in STEL! This is the code going with the EMNLP 2021 main conference paper [Does It Capture STEL? A Modular, Similarity-based Linguistic Style Evaluation Framework](https://aclanthology.org/2021.emnlp-main.569/).

# Quickstart

You can find the raw data for STEL in Data/STEL. You will need to get permission to use the formality data from Yahoo ([L6 - Yahoo! Answers ComprehensiveQuestions and Answers version 1.0 (multi part)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l)) as this is also the prerequisite for receiving the [GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus). Please e-mail me (a.m.wegmann@uu.nl) with the permission to receive the full data necessary to run STEL. You will need to add the files to the repository as specified in ```to_add_const.py```.

To use it, in the project folder src, call

```python
import eval_style_models

eval_style_models.eval_sim()
```

This will call all implemented style similarity models on the current version of STEL (except for deepstyle and LIWC based models).  Can take a long time. Might run into RAM problems (depending on your machine).

To only call a specific method on style:

```python
import eval_style_models
import style_similarity

eval_style_models.eval_sim(style_objects=[style_similarity.WordLengthSimilarity()])
```

Do not forget the instantiation via '()'. See some further example calls in `example_eval_style_models.py`. You will need to set LOCAL_STEL_DIM_QUAD to `/Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv'`.

Expected output:

```
INFO : Running STEL framework 
INFO : Filtering out tasks with low agreement ... 
INFO :       on dimensions ['simplicity', 'formality'] using files ['/home/anna/Documents/UU/STEL/src/../Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']...
INFO :       on characteristics ['contraction', 'nbr_substitution'] using file ['/home/anna/Documents/UU/STEL/src/../Data/STEL/characteristics/quad_questions_char_contraction.tsv', '/home/anna/Documents/UU/STEL/src/../Data/STEL/characteristics/quad_questions_char_substitution.tsv']
INFO : Evaluating on 1630 style dim and 200 style char tasks ... 
INFO : Evaluation for method WordLengthSimilarity
INFO : random assignments: 156
INFO :   Accuracy at 0.5792349726775956, without random 0.5866188769414575 with 156 questions
INFO :   Accuracy simplicity at 0.5907975460122699 for 815 task instances, without random 0.5943877551020408 with 784 left questions
INFO :   Accuracy formality at 0.5300613496932515 for 815 task instances, without random 0.5313700384122919 with 781 left questions
INFO :   Accuracy contraction at 0.94 for 100 task instances, without random 0.94 with 100 left questions
INFO :   Accuracy nbr_substitution at 0.5 for 100 task instances, without random 0.7777777777777778 with 9 left questions
             Model Name  Accuracy  Accuracy simplicity  Accuracy formality  \
0  WordLengthSimilarity  0.579235             0.590798            0.530061   

   Accuracy contraction  Accuracy nbr_substitution  
0                  0.94                        0.5  
INFO : Saved results to output/STEL-quadruple_WordLengthSimilarity.tsv
INFO : Saved single predictions to output/STEL_single-pred-quadruple_WordLengthSimilarity.tsv
```

When running on the provided sample of STEL only, the output will look different (see below). Keep in mind that this does not include the full variability of STEL though. It is using a sample, i.e., `/Data/STEL/dimensions/quad_stel-dimension_simple-100_sample.tsv` and `Data/STEL/dimensions/quad_stel-dimension_formal-100_sample.tsv`.

```
INFO : Running STEL framework 
INFO : Filtering out tasks with low agreement ... 
INFO :       on dimensions ['simplicity', 'formality'] using files ['/home/anna/Documents/UU/STEL/src/../Data/STEL/dimensions/quad_stel-dimension_simple-100_sample.tsv', '/home/anna/Documents/UU/STEL/src/../Data/STEL/dimensions/quad_stel-dimension_formal-100_sample.tsv']...
INFO :       on characteristics ['contraction', 'nbr_substitution'] using file ['/home/anna/Documents/UU/STEL/src/../Data/STEL/characteristics/quad_questions_char_contraction.tsv', '/home/anna/Documents/UU/STEL/src/../Data/STEL/characteristics/quad_questions_char_substitution.tsv']
INFO : Evaluating on 200 style dim and 200 style char tasks ... 
INFO : Evaluation for method WordLengthSimilarity
INFO : random assignments: 100
INFO :   Accuracy at 0.6425, without random 0.69 with 100 questions
INFO :   Accuracy simplicity at 0.62 for 100 task instances, without random 0.625 with 96 left questions
INFO :   Accuracy formality at 0.485 for 100 task instances, without random 0.4842105263157895 with 95 left questions
INFO :   Accuracy contraction at 0.94 for 100 task instances, without random 0.94 with 100 left questions
INFO :   Accuracy nbr_substitution at 0.5 for 100 task instances, without random 0.7777777777777778 with 9 left questions
INFO : Saved results to output/STEL-quadruple_WordLengthSimilarity.tsv
INFO : Saved single predictions to output/STEL_single-pred-quadruple_WordLengthSimilarity.tsv
             Model Name  Accuracy  Accuracy simplicity  Accuracy formality  \
0  WordLengthSimilarity    0.6425                 0.62               0.485   

   Accuracy contraction  Accuracy nbr_substitution  
0                  0.94                        0.5  
```



Possible models to call upon (compare to the paper):

```
UncasedBertSimilarity(), LevenshteinSimilarity(), CharacterThreeGramSimilarity(), PunctuationSimilarity(), WordLengthSimilarity(), UppercaseSimilarity(), PosTagSimilarity(), CasedBertSimilarity(), UncasedSentenceBertSimilarity(), RobertaSimilarity(), USESimilarity(), BERTCasedNextSentenceSimilarity(), BERTUncasedNextSentenceSimilarity(), DeepstyleSimilarity(), LevenshteinSimilarity(), LIWCStyleSimilarity(), LIWCSimilarity() 
```

To add your own model, implement the abstract similarity class. Either implement the ```similarity(self, sentence_1: str, sentence_2: str) -> float``` or override the ```similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]``` function for a more efficient implementation. Only similarities will be called. 
```python
from style_similarity import Similarity 

class MySimilarity(Similarity):
    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        if len(sentence_1) == 0 and len(sentence_2) == 0:
            return self.SAME  # same style is at 1
        else:
            return self.DISTINCT # distinct style is at 0
```

To call this new method on STEL:

```python
eval_style_models.eval_sim(style_objects=[MySimilarity()])
```

# Structure

When you add all necessary (partly proprietary) data to use ALL functionalities, the folder should look something like this. Files starting with _ include proprietary data (see below). They are not included in the public release but will need to be acquired. The Datasets folder contains files that were used to generate STEL. They were not included because of size. Here, the GYAFC_Corpus also needs permission from Verizon. Everything else in the Datasets folder can be downloaded from different sources, see also ```to_add_const.py```. 

```
.
├── Data
│   ├── Datasets
│   │   ├── GYAFC_Corpus
│   │  	│	└── Entertainment_Music
│   │  	│	    ├── test
│   │  	│	    │   ├── formal
│   │  	│	    │   └── informal.ref0
│   │  	│	    └── tune
│   │  	│	        ├── formal
│   │  	│	        └── informal.ref0
│   │   └── turkcorpus
│   │  	│	└── truecased
│   │  	│	    ├── test.8turkers.organized.tsv
│   │  	│	    └── tune.8turkers.organized.tsv
│   │   ├── enwiki-20181220-abstract.xml
│   │   ├── RC_2007-05.bz2
│   │   ├── RC_2007-06.bz2
│   │   ├── RC_2007-07.bz2
│   │   ├── RC_2007-08.bz2
│   │   ├── RC_2007-09.bz2
│   │   ├── RC_2012-06.bz2
│   │   ├── RC_2016-06.bz2
│   │   └── RC_2017-06.bz2
│   ├── Experiment-Results
│   │   ├── annotations
│   │   │   ├── 00_nsubs_sentences.txt
│   │   │   ├── 00_nsubs_translated_sentences.txt
│   │   │   ├── QUAD-full_agreement_accuracy.log
│   │   │   ├── _QUAD-full_annotations.tsv
│   │   │   ├── QUAD-subsample_agreement_accuracy.log
│   │   │   ├── _QUAD-subsample_annotations.tsv
│   │   │   ├── readme.md
│   │   │   ├── TRIP-subsample_agreement_accuracy.log
│   │   │   └── _TRIP-subsample_annotations.tsv
│   │   └── models
│   │       ├── readme.md
│   │       ├── Significance testing.ipynb
│   │       ├── STEL_all-models.log
│   │       ├── STEL_deepstyle.log
│   │       ├── STEL_single-preds_all-models.tsv
│   │       ├── STEL_single-preds_deepstyle.tsv
│   │       ├── UNFILTERED_all-models.log
│   │       ├── UNFILTERED_deepstyle.log
│   │       ├── UNFILTERED_single-preds_all-models.tsv
│   │       └── UNFILTERED_single-preds_deepstyle.tsv
│   ├── Models
│   │   ├── coord-liwc-patterns.txt
│   │   └── _LIWC2015_Dictionary.dic
│   └── STEL
│       ├── characteristics
│       │   ├── quad_questions_char_contraction.tsv
│       │   └── quad_questions_char_substitution.tsv
│       ├── dimensions
│       │   ├── quad_stel-dimension_formal-100_sample.tsv
│       │   ├── _quad_stel-dimensions_formal-815_complex-815.tsv
│       │   └── quad_stel-dimension_simple-100_sample.tsv
│       └── readme.md
├── src
│   ├── set_for_global.py
│   ├── to_add_const.py
│   ├── generate_pot_quads.py
│   ├── sample_for_annotation.py
│   ├── eval_annotations.py
│   ├── eval_style_models.py
│   ├── output
│   └── utility
│       ├── base_generators.py
│       ├── file_utility.py
│       ├── nsubs_possibilities.py
│       ├── qualtrics_api.py
│       ├── qualtrics_constants.py
│       ├── const_generators.py
│       ├── quadruple_generators.py
│       ├── neural_model.py
│       ├── style_similarity.py
├── LICENSE
├── readme.md
└── .gitignore 
```



# Prerequisites

## Python Libraries

### STEL light

 Code tested on python 3.8.5. If you only want to use STEL with your own models, the relevant packages with the tested versions are:

```
pandas==1.1.3
numpy==1.18.5
scikit-learn==0.23.2
nltk==3.6.2
typing==3.10.0.0
```

Tested functionality for calling eval_style_models.py only.

### STEL

If you also want to call the neural similarity methods, you will also need the following packages: 

```
transformers==3.5.1
torch==1.7.0
sentence-transformers==2.0.0
scipy==1.7.1
```

### using deepstyle

see: https://github.com/hayj/DeepStyle

``` h5py==2.10.0 
h5py==2.10.0
transformers==2.4.1
tensorflow-gpu==2.0
```

Also, in ```model.py```, you will need to add the line ```import os```.

## Proprietary data

The file ```to_add_const.py``` contains the paths and information to all files that are not part of the GitHub release but are necessary to run (part) of the code. Some files were not added because of their size and others because the data is proprietary:

For **LIWC-based similarities** the repository expects the [LIWC 2015 dictionary](https://repositories.lib.utexas.edu/bitstream/handle/2152/31333/LIWC2015_LanguageManual.pdf) in ```Data/Models/_LIWC2015_Dictionary.dic```

For the **formal/informal STEL dimension**, you will need to get permission to use the formality data from Yahoo ([L6 - Yahoo! Answers ComprehensiveQuestions and Answers version 1.0 (multi part)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l)) as this is also the prerequisite for receiving the [GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus). Please e-mail me (a.m.wegmann@uu.nl) with the permission to receive the full data necessary to run STEL and add it to ```Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv```.  We received a limited waiver to release a  sample of 100 STEL formal/informal tasks with this GitHub release to test the code. It still falls under the Yahoo's Terms of Use and you will need to get permission from them to use Yahoo's data in your own publications.  



# Citation

When using STEL please cite our paper

```
@inproceedings{wegmann-nguyen-2021-capture,
    title = "Does It Capture {STEL}? A Modular, Similarity-based Linguistic Style Evaluation Framework",
    author = "Wegmann, Anna  and
      Nguyen, Dong",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.569",
    pages = "7109--7130",
    abstract = "Style is an integral part of natural language. However, evaluation methods for style measures are rare, often task-specific and usually do not control for content. We propose the modular, fine-grained and content-controlled similarity-based STyle EvaLuation framework (STEL) to test the performance of any model that can compare two sentences on style. We illustrate STEL with two general dimensions of style (formal/informal and simple/complex) as well as two specific characteristics of style (contrac{'}tion and numb3r substitution). We find that BERT-based methods outperform simple versions of commonly used style measures like 3-grams, punctuation frequency and LIWC-based approaches. We invite the addition of further tasks and task instances to STEL and hope to facilitate the improvement of style-sensitive measures.",
}
```

and the papers that the dataset was (partly) generated from:

```
@inproceedings{rao-tetreault-2018-dear,
    title = "Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer",
    author = "Rao, Sudha  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1012",
    doi = "10.18653/v1/N18-1012",
    pages = "129--140",
}

@article{xu-etal-2016-optimizing,
    title = "Optimizing Statistical Machine Translation for Text Simplification",
    author = "Xu, Wei  and
      Napoles, Courtney  and
      Pavlick, Ellie  and
      Chen, Quanze  and
      Callison-Burch, Chris",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "4",
    year = "2016",
    url = "https://aclanthology.org/Q16-1029",
    doi = "10.1162/tacl_a_00107",
    pages = "401--415",
}
```

We thank Yahoo for granting us the right to upload a sample of 100 task instances from the formal/informal dimension. Please make sure to adhere to their Terms of Use. Especially asking for their permission to reuse any examples via [L6 - Yahoo! Answers ComprehensiveQuestions and Answers version 1.0 (multi part)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l).

# Comments

Thank you for your comments and questions. You can use GitHub Issues or address me directly (Anna via a.m.wegmann @ uu.nl).
