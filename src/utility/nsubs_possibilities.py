import logging

from wordfreq import word_frequency

import base_generators
from to_add_const import PUSHSHIFT_MONTH

NUMBER_SUB_DICT = {  # from https://www.gamehouse.com/blog/leet-speak-cheat-sheet/
    # and https://simple.wikipedia.org/wiki/Leet and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3358692/pdf/fpsyg-03-00150.pdf
    # and https://h2g2.com/edited_entry/A787917
    '2morrow': '2morrow',
    'c00l': 'cool',
    'n!ce': 'nice',
    'l0ve': 'love',
    'sw33t': 'sweet',
    'l00k': 'look',
    'l33t': 'leet',
    '1337': 'leet',
    '0wn': 'own',
    '0wn3d': 'owned',
    '0wn4ge': 'ownage',
    't3h': 'teh',
    'n00b': 'noob',
    'd00dz': 'dude',
    'd00d': 'dude',
    'b00n': 'boon',
    'ph34r': 'fear',
    'c3n50red': 'censored',
    'w00t': 'woot',  # show celebration or happiness
    'p00nd': 'pwn',
    'l4m3r': 'lamer',
    't00l': 'tool',
    '0mg': 'omg',
    '0mfg': 'omfg',
    'sk1llz': 'skills',
    'sk1ll': 'skill',
    'sk1ll3d': 'skilled',
    'm4d': 'mad',
    'sp4wn': 'spawn',
    '13wt': 'loot',
    'j0': 'jo',
    'j00': 'joo',
    # sound number substitutions
    'm8': 'mate',
    'm2': 'me too',
    'gr8': 'great',
    'b4': 'before',
    '4ever': 'forever',
    '2day': 'today',
    '2nite': 'tonight',
    'h8': 'hate',
    'in2': 'into',
    'sum1': 'someone',
    'w8': 'wait',
    '4u': 'for you',
    'sk8r': 'skater'
}
NSUBS_UNITS = ["m", "cm", "mm", "km", "ha", "ft",
               "m2", "cm2", "mm2", "km2", "km2", "km2", "km2", "km2", "km2",
               "m3",
               "sqmi", "acre", "sqyd", "sqft", "sqin" "ft", "sq", "dunam", "tsubo"
                                                                           "kg", "lb", "nmi", "oz", "gr", "g", "ozt",
               "mg",
               "nm", "mi", "rd", "fathom", "yd", "in", "nmi", "pc", "ly", "au", "carat"
                                                                                "ft3", "yd3", "s", "kph",
               "fps", "hz", "mph", "lbs", "seconds", "am", "lb", "usd",
               "bit", "byte", "mb", "gb", "tb", "mo", "yo",
               "percent",
               "pt2", "pt",
               "er", "yr",
               "hour",
               "2nd"]
NSUBS_SCIENCE = ["w3c", "pp2", "4cycl", "cc2", "co2", "ln", "cat", "cat5", "f2l", "t1d", "fat32", "fat",
                 "h5n1", "vo2", "h2o", "cb2", "cyp3a4", "obd2",
                 "ipv4", "wpa2", "bzip2", "j2ee", "w3c", "utf8", "p2p", "ie7", "ie6", "ie8", "id3"]
NSUBS_BRANDS = ["bo4", "bo2", "bo3", "bo", "bo5", "7up", "bf2", "bf", "dota2", "dota", "nhl17", "tf2", "r2d2", "c2c",
                "ec2",
                "2br", "bl2", "kh3", "kh2", "kh", "hl2", "hl1", "hl", "fc4", "fc5", "fc", "kf2", "ao3", "4u", "h2a",
                "c25k",
                "ac4", "nhl", "ue4", "nba",
                "cod4", "fifa12", "fifa10", "ff7", "ck2", "eu3", "re4", "bc2", "wc2", "c4d"
                "2pac", "af2",
                "rio20"]
NSUBS_MODELS = ["f150", "g700", "t450", "t400", "t470", "t430", "t3i", "e550", "g500", "m750i", "cbr1000rr", "t500",
                "g430", "d750", "d810", "t440", "t3i", "d10s", "x100", "gt750", "g37", "t500", "c10h14", "a13",
                "t2i", "t3i", "db4", "2jz", "2cv", "db2", "rav4",
                "4runner",
                "x100", "p400", "e010", "t4s", "4wd", "tr4", "dr2", "db3", "bb7",
                "cbr2sor", "cbr250r",
                "ipad2", "iphone4"]
NSUBS_NON_VALID_WORDS = ["x", "4chan", "st", "rd", "hp", "vs", "th", "1v1",
                         "pm", "m4s", "e0",
                         "70s", "80s", "90s", "60s", "50s", "40s", "30s", "20s",
                         "ww2", "ww3", "ww1",
                         "mp4", "mp3", "p3D", "a1c", "e30", "m3s", "p3p", "cat5e",
                         "cal", "ish", "min", "ddr3", "1000rr", "a7r", "e3d", "ps4", "a7r", "m40",
                         "3ds", "cc1", "cc3", "bh3", "ps3", "pi3", "lt", "bo3", "ts4", "ts3", "lvl", "sl3", "sl4",
                         "fo4",
                         "cp3", "rs3", "win", "ddr", "wr1", "wr3", "ss3", "tier", "ak74", "rb14", "rb15", "ak47",
                         "rule",
                         "super35", "ar33", "ss", "bk", "mp5", "gen", "bh4", "ds3", "bf4", "level", "mil", "oz", "pc",
                         "spf",
                         "h2h", "top3", "top4", "top", "top12", "H2k", "p4p", "2ch",
                         "pro", "pac12", "r4r", "l4d", "y2k", "dr3", "dr2",
                         "cr7", "ap2", "np2", "h1b", "cop21",
                         'act2', 'act1', "unity3d", "tl2", "2wd", "b2k",
                         "c3po"]


class NumberSubsPossibilities:  # QuadrupleGenerator):  #
    """
        Generates Number Substitution possibilities
    """

    def __init__(self, total=100, pushshift_month=PUSHSHIFT_MONTH):
        logging.info('instantiating numbersub possibilities for month {}'.format(pushshift_month))
        # super(NumberSubsPossibilities, self).__init__()
        self.utt_stream_iter = iter(
            base_generators.PushshiftUtteranceGenerator(pushshift_month,
                                                        load_to_memory=True))
        self.leet_translation = NUMBER_SUB_DICT
        self.total = total
        self.to_leetspeak_dict = {
            "a": ["4"],
            "e": ["3"],
            "l": ["1"],
            "i": ["1", "!"],
            "o": ["0"],
            "t": ["7"],
            "s": ["5"],
        }
        self.word_leets = ["8", "2", "4", "1"]  # 2: "to" or similar, "8": "ight", "4": "for", "1": "one"
        self.leet_speak_symbols = ["4", "3", "1", "!", "0", "7", "5", "2"]
        self.leet_speak_translation = ["a", "e", "l", "i", "o", "t", "s"]
        self.non_valid_words = NSUBS_NON_VALID_WORDS + NSUBS_UNITS + NSUBS_BRANDS + NSUBS_MODELS + NSUBS_SCIENCE
        self._first_style_sentences = []
        self._second_style_sentences = []

    # @property
    # def first_style_sentences(self):
    #     # non leet speak, but it includes a word with leetspeak
    #     if self._first_style_sentences:
    #         pass
    #     else:
    #         self._init_style_sentences()
    #
    #     return self._first_style_sentences
    # @property
    # def second_style_sentences(self):
    #     if self._second_style_sentences:
    #         pass
    #     else:
    #         self._init_style_sentences()
    #     return self._second_style_sentences

    def _init_style_sentences(self):

        overall_poss_leet_count = {}

        f_leet_trans = open('leet_sentences_trans.txt', 'w', 10)
        f_leet_sent = open('leet_sentences.txt', 'w', 10)
        f_leets = open('leet_words.txt', 'w', 10)
        f_leet_by_sent = open('leet_by_sentence.txt', "w", 10)

        while len(overall_poss_leet_count) < self.total:
            cur_utt = next(self.utt_stream_iter)
            cur_utt = cur_utt.replace("\n", "")

            # REMOVE URL:
            #  see https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
            import re
            cur_utt = re.sub(r'http\S+', '', cur_utt)

            # FILTER: At least 2 leet speak symbols in the sentence
            if not len([char for char in cur_utt if char in self.leet_speak_symbols or char in self.word_leets]) >= 2:
                continue
            else:
                potential_sent_leet = []
                for word in cur_utt.split(" "):  # For every word test if it could be leet speak
                    word = word.lower()
                    if word in overall_poss_leet_count.keys() and overall_poss_leet_count[word] >= 10:
                        continue

                    possible_translations = []
                    # Is word in list of KNOWN leet words?
                    if word in self.leet_translation.keys():
                        potential_sent_leet.append(word)
                        possible_translations.append(self.leet_translation[word])
                    else:
                        # FILTER: only include words that have a certain 'frequency'
                        #   & adhere to some empirical structure
                        # if word_frequency(word, "en") < 2e-10:
                        #     continue
                        if self.filter_word(word):
                            continue
                        # test if this is a potential word
                        potential_sent_leet.append(word)
                        #   translate to a word
                        possible_translations = self.find_possible_translations(word)

                    if word not in overall_poss_leet_count.keys():
                        overall_poss_leet_count[word] = 1
                        f_leets.write(word + "\n")
                    else:
                        overall_poss_leet_count[word] += 1
                    # for w in possible_words:
                    #     if word_frequency(w)
                    logging.info("Found possible leet word {} with possible translations {}".
                                 format(word, possible_translations))
                    break

                if len(potential_sent_leet) == 0:
                    continue
                else:
                    cur_utt = cur_utt.replace("\n", " ")
                    cur_utt = cur_utt.replace("\r", " ")
                    logging.info("added {}".format(cur_utt.replace("\n", " ")))
                    logging.info("because of {}".format(potential_sent_leet))
                    self._first_style_sentences.append(cur_utt)
                    self._second_style_sentences.append(self.translate_to_non_leet(cur_utt))
                    f_leet_sent.write(cur_utt.replace("\n", " ") + "\n")
                    f_leet_by_sent.write(potential_sent_leet[0] + "\n")
                    f_leet_trans.write("".join(
                        [word if word not in potential_sent_leet else possible_translations[0] for word in
                         cur_utt.split(" ")]))

        logging.info(self._first_style_sentences)
        logging.info(overall_poss_leet_count)

        f_leets.close()
        f_leet_by_sent.close()
        f_leet_sent.close()

    def find_possible_translations(self, word):
        possible_translations = [""]
        for char in word:
            # Check if character is even a possible leet symbol
            if char not in [elem for leet_list in self.to_leetspeak_dict.values()
                            for elem in leet_list]:
                for i, w in enumerate(possible_translations):
                    possible_translations[i] += char
            else:
                pot_letters = [letter for letter, leet_list in self.to_leetspeak_dict.items()
                               if any(leet == char for leet in leet_list)]
                for j, letter in enumerate(pot_letters):
                    # append the translated symbol to the word
                    if j == 0:
                        for i, _ in enumerate(possible_translations):
                            possible_translations[i] += letter
                    else:
                        tmp_addition = []
                        for i, _ in enumerate(possible_translations):
                            tmp_addition.append(possible_translations[i][:-1] + letter)
                        possible_translations = possible_translations + tmp_addition
        return possible_translations

    def filter_word(self, word):
        # extract potential leet symbols and alpha characters from word
        symbol_string = "".join([char for char in word if char in self.leet_speak_symbols or
                                 char in self.word_leets])
        alpha_string = "".join([char for char in word if char.isalpha()])
        if len(symbol_string) == 0 or len(alpha_string) == 0 or len(symbol_string) + len(alpha_string) < len(word):
            return True
        #   only words with leet speak potential and alphanumerical symbols
        if all(char not in self.leet_speak_symbols and char not in self.word_leets for char in word) or \
                (all(not char.isalpha() for char in word)) or \
                any(char not in self.leet_speak_symbols + self.word_leets and not char.isalpha()
                    for char in word):
            return True
        if any(substr.lower() in word.lower() for substr in self.non_valid_words):
            return True
        if word[-1] == "s" and word[0].isalpha() and len([char for char in word if char.isalpha()]) == 2:
            return True
        if word[: len(symbol_string)] == symbol_string and symbol_string not in ["2", "4"]:
            return True
        if word[-len(symbol_string):] == symbol_string and len(symbol_string) > 2:
            return True
        if (len(word) == 6 and word[0] == "s" and word[3] == "e" and
                len([char for char in word if char.isalpha()]) == 2):
            return True
        if (set([char for char in symbol_string]).issubset({"!", "1"})
                and word[-1] in ["1", "!"]):
            return True
        if (set([char for char in symbol_string]).issubset({"!", "1"}) and
                len(symbol_string) > 1
                and symbol_string in word):
            return True
        if (set([char for char in symbol_string]).issubset({"!", "1"}) and
                symbol_string in word and word.split(symbol_string)[1].isupper()):
            return True
        if not len([char for char in word if char.isupper()]) <= 1:
            return True
        if not len(symbol_string) <= len(word) / 2:
            return True
        if (word[-len(symbol_string):] == symbol_string and len(symbol_string) >= 2 and
                len(symbol_string) >= len(alpha_string)):
            return True
        if not len([char for char in word if char.isalpha()]) >= 2:
            return True
        return False

    def translate_to_non_leet(self, utt_string):
        result = ""
        for i, word in enumerate(utt_string.split(" ")):
            if any(char in self.leet_speak_symbols for char in word) and any(char.isalpha() for char in word):
                # result += "".join( for char in word)
                result += " " + word
            elif i > 0:
                result += " " + word
            else:
                result += word
