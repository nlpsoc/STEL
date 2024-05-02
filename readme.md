**UPDATE:** Added another characteristic task: emotives vs. emojis. Thanks to the amazing work done a while ago by David Bikker (😊 vs . :)).

**NOTE:** With the current code, changing the batch size of RoBERTa changes its performance. To get the highest possible performance, use an eval batch size of 1. This is probably connected to the tokenization call and the used padding in the batch. To make sure you are not affected by this set eval batch size to 1 (performance of ~0.80 for RoBERTa instead of ~0.61), or even cleaner, 
use the sentence-bert implementation for encoding sentences, as described below 
(just change `"AnnaWegmann/Style-Embedding"` to `roberta-base`)


Thank you for your interest in STEL! This is the code going with the EMNLP 2021 main conference paper [Does It Capture STEL? A Modular, Similarity-based Linguistic Style Evaluation Framework](https://aclanthology.org/2021.emnlp-main.569/).

# Quickstart

You can find the raw data for STEL in Data/STEL. You will need to get permission to use the formality data from Yahoo ([L6 - Yahoo! Answers ComprehensiveQuestions and Answers version 1.0 (multi part)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l)) as this is also the prerequisite for receiving the [GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus). Please e-mail me (a.m.wegmann@uu.nl) with the permission to receive the full data necessary to run STEL. You will need to add the files to the repository as specified in ```to_add_const.py```. 
You will need to set LOCAL_STEL_DIM_QUAD to `/Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv'` 
after getting permission to use the formality data from Yahoo. 


To use it, on a specific method, call

```python
import STEL

STEL.STEL.eval_on_STEL(style_objects=[STEL.similarity.WordLengthSimilarity()])
```

To use your own method override the similarity class and call it in the same way. Example for a sentence BERT similarity:

```python
from STEL.similarity import Similarity, cosine_sim
from sentence_transformers import SentenceTransformer
import torch
from STEL import STEL

class SBERTSimilarity(Similarity):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SentenceTransformer("AnnaWegmann/Style-Embedding")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def similarities(self, sentences_1, sentences_2):
        with torch.no_grad():
            sentence_emb_1 = self.model.encode(sentences_1, show_progress_bar=False)
            sentence_emb_2 = self.model.encode(sentences_2, show_progress_bar=False)
        return [cosine_sim(sentence_emb_1[i], sentence_emb_2[i]) for i in range(len(sentences_1))]

STEL.eval_on_STEL(style_objects=[SBERTSimilarity("AnnaWegmann/Style-Embedding")])
```


Expected (printed) output:

```
Performance on original STEL tasks
        Model Name  Accuracy  Accuracy formality  Accuracy simplicity  \
0  SBERTSimilarity   0.73399            0.828221             0.581595   

   Accuracy nbr_substitution  Accuracy contraction  Accuracy emotives  
0                       0.56                  0.96              0.945  
Performance on STEL-Or-Content tasks
        Model Name  Accuracy  Accuracy formality  Accuracy simplicity  \
0  SBERTSimilarity  0.396059            0.696933             0.266258   

   Accuracy contraction  Accuracy nbr_substitution  Accuracy emotives  
0                  0.02                       0.03               0.07  
```

In case you receive "Running STEL on small demo data.", 
something went wrong with the GYAFC data. 
Make sure to gain permission from Yahoo, 
e-mail me to get acces to the file (a.m.wegmann@uu.nl) and 
put it in the correct folder: ```Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv```.

Expected printed output for the demo:
```
Running STEL on small demo data. Please gain permission to use the GYAFC corpus and add the data to location Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv
Performance on original STEL tasks
        Model Name  Accuracy  Accuracy formality  Accuracy simplicity  \
0  SBERTSimilarity    0.7375                0.84                 0.59   
   Accuracy nbr_substitution  Accuracy contraction  
0                       0.56                  0.96  
Performance on STEL-Or-Content tasks
        Model Name  Accuracy  Accuracy formality  Accuracy simplicity  \
0  SBERTSimilarity      0.25                0.69                 0.26   
   Accuracy contraction  Accuracy nbr_substitution  
0                  0.02                       0.03  
```

### Legacy (Maybe don't use)
Possible models to call upon (compare to the paper):

```
UncasedBertSimilarity(), LevenshteinSimilarity(), CharacterThreeGramSimilarity(), PunctuationSimilarity(), WordLengthSimilarity(), UppercaseSimilarity(), PosTagSimilarity(), CasedBertSimilarity(), UncasedSentenceBertSimilarity(), RobertaSimilarity(), USESimilarity(), BERTCasedNextSentenceSimilarity(), BERTUncasedNextSentenceSimilarity(), DeepstyleSimilarity(), LevenshteinSimilarity(), LIWCStyleSimilarity(), LIWCSimilarity() 
```


# Prerequisites

## Python Libraries

### STEL light

 Code tested on python 3.11. If you only want to use STEL with your own models, the relevant packages with the tested versions are in requirements.txt

Tested functionality for calling eval_style_models.py only.


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

When using STEL, consider citing our paper

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

When using the STEL-or-Content results, consider also citing:
```
@inproceedings{wegmann-etal-2022-author,
    title = "Same Author or Just Same Topic? Towards Content-Independent Style Representations",
    author = "Wegmann, Anna  and
      Schraagen, Marijn  and
      Nguyen, Dong",
    booktitle = "Proceedings of the 7th Workshop on Representation Learning for NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.repl4nlp-1.26",
    doi = "10.18653/v1/2022.repl4nlp-1.26",
    pages = "249--268",
    abstract = "Linguistic style is an integral component of language. Recent advances in the development of style representations have increasingly used training objectives from authorship verification (AV){''}:{''} Do two texts have the same author? The assumption underlying the AV training task (same author approximates same writing style) enables self-supervised and, thus, extensive training. However, a good performance on the AV task does not ensure good {``}general-purpose{''} style representations. For example, as the same author might typically write about certain topics, representations trained on AV might also encode content information instead of style alone. We introduce a variation of the AV training task that controls for content using conversation or domain labels. We evaluate whether known style dimensions are represented and preferred over content information through an original variation to the recently proposed STEL framework. We find that representations trained by controlling for conversation are better than representations trained with domain or no content control at representing style independent from content.",
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
