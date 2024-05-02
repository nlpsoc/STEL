import bz2
import json

from STEL.utility.const_generators import REDDIT_SKIP_COMMENTS
from STEL.to_add_const import ABSTRACT_WIKI


class PushshiftUtteranceGenerator:

    def __init__(self, filename: str, slow=True, bz2=True, load_to_memory=False):
        # super().__init__(filename, slow, bz2)
        self.filename = filename
        from convokit import text_processing
        self.text_cleaner = text_processing.textCleaner.TextCleaner()
        self.load_to_memory = load_to_memory

    def __iter__(self):
        # Create self.index

        file_object = bz2.open(self.filename, "rb")
        # if self.load_to_memory:
        #     file_object = file_object.readlines()
        for line in file_object:
            txt = json.loads(line)["body"]
            if txt not in REDDIT_SKIP_COMMENTS:
                for sentence in self.text_cleaner.transform_utterance(txt.replace("\n", " ")).text.split(". "):
                    yield sentence
        file_object.close()
        # while True:
        #     # repeatedly call buffering function
        #     for utterance_string in self.get_n_random_utt(3000):
        #         yield utterance_string.decode()


class WikiAbstractStream:
    """
    Generator of Wiki abstracts
    """

    def __init__(self, filename_wiki_abstracts=ABSTRACT_WIKI,
                 must_include=None):
        self.xml_filename = filename_wiki_abstracts
        # <feed>
        #   <doc>
        #     <abstract>
        import xml.etree.ElementTree as ET
        self.elem_iter = ET.iterparse(self.xml_filename)  # , events=("start", "end"))
        self.must_include = must_include

    def __iter__(self):
        for event, elem in self.elem_iter:
            # if event == 'end':
            # if elem.tag == 'doc':
            #    logging.info(elem.tag)
            if elem.tag == 'abstract':
                if elem.text and len(elem.text) > 10 and "." in elem.text and not "|" in elem.text \
                        and "{" not in elem.text and "}" not in elem.text \
                        and "(" not in elem.text and ")" not in elem.text:
                    if self.must_include and self.must_include not in elem.text:
                        continue
                    for sentence in elem.text.split(". "):
                        yield sentence
