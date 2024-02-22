import re
import pymorphy3
from ua_gec import AnnotatedText
from ua_gec.annotated_text import Annotation
from typing import List
from .log import create_logger
from .spell_checker import SpellChecker

morph = pymorphy3.MorphAnalyzer(lang='uk')
spell_checker = SpellChecker()
spell_checker.load()


class Token:

    def __init__(self, text: str, index: int):
        """
        :param text: annotated text that represents a token
        :param index: token index in the list
        """
        self.annotated = AnnotatedText(text)
        self.annotations = self.annotated.get_annotations()
        self.index = index
        self._tag = None

    def __repr__(self) -> str:
        return self.annotated.get_annotated_text()

    @property
    def first_annotation(self):
        """
        This is suboptimal, but we will process only the first annotation
        even if there are multiple annotations in a single token
        """
        if not self.annotations:
            return None
        ann: Annotation = self.annotations[0]
        return ann

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value


class Text:

    # inspired by https://stackoverflow.com/questions/9644784/splitting-on-spaces-except-between-certain-characters
    # split an annotated string by annotation and namespace but ommit spaces in annotations
    ANNOTATION_SPACE_SPLIT_REGEX = re.compile(
        r'\s+(?=[^{}]*(?:\{.*:::.*\}|$))')
    # split an annotated string only by annotation that contains source text
    # e.g. `мінімум {, =>:::error_type=Punctuation}{=>— :::error_type=Punctuation}знати`
    # -> ["мінімум ", "{, =>:::error_type=Punctuation}", "{=>— :::error_type=Punctuation}знати"]
    ANNOTATION_SPLIT_REGEX = re.compile(r'({[^{}]+?=>.*?:::.*?})')

    def __init__(self, text: str, metadata):
        self.text = AnnotatedText(text)
        self.tokens: List[Token] = self.tokenize()
        self.metadata = metadata

    def __repr__(self) -> str:
        return self.text.get_annotated_text()

    def tokenize(self) -> List[Token]:
        """
        Tokenize Sequence
        1. Split by whitespace excluding whitespaces in annotations
        2. Split by annotations that contain source text (source_text is not empty)
        """
        raw_tokens = []
        self.tokens = []
        for _token in self.ANNOTATION_SPACE_SPLIT_REGEX.split(
                self.text.get_annotated_text()):
            sub_tokens = [
                t for t
                in self.ANNOTATION_SPLIT_REGEX.split(_token) if t
            ]
            raw_tokens.extend(sub_tokens)

        for i, raw_token in enumerate(raw_tokens):
            self.tokens.append(Token(text=raw_token, index=i))

        return self.tokens

    def get_tag(self, token: Token):
        source: str = token.first_annotation.source_text
        return Transformations(source).get_tag(self.tokens, token.index)

    def tag(self):
        """
        Tag all tokens in the text
        """
        for token in self.tokens:
            if not token.first_annotation:
                continue
            tag = self.get_tag(token)
            token.tag = tag


class Transformation:

    def __init__(self, source):
        self.source: str = source

    def tag(self):
        pass

    def correct(self):
        pass


class Spelling(Transformation):

    def __init__(self, source):
        super().__init__(source)

    def _word_without_duplicate_chars(self, word):
        if not word:
            return word

        result = word[0]
        for char in word[1:]:
            if char != result[-1]:
                result += char

        return result

    def case_capital(self):
        return self.source.capitalize()

    def case_lower(self):
        return self.source.lower()

    def merge_hyphen(self):
        return self.source.replace(" ", "-")

    def merge_split(self):
        return self.source.replace("-", " ")

    def merge_space(self):
        return self.source.replace(" ", "")

    def duplicate_chars(self):
        return self._word_without_duplicate_chars(self.source)

    def excessive_soft_sign(self):
        return self.source.rstrip('ь')

    def missing_soft_sign(self):
        return f'{self.source}ь'

    def uk_e_soft(self):
        return self.source.replace("е", "є")

    def uk_e_hard(self):
        return self.source.replace("є", "е")

    def uk_v_1(self):
        if self.source[0].isupper():
            return self.source.replace("У", "В", 1)
        return self.source.replace("у", "в", 1)

    def uk_y_1(self):
        if self.source[0].isupper():
            return self.source.replace("В", "У", 1)
        return self.source.replace("в", "у", 1)

    def uk_v_upper_1(self):
        return self.source.replace("У", "В", 1)

    def uk_y_upper_1(self):
        return self.source.replace("В", "У", 1)

    def uk_g_special(self):
        return self.source.replace("г", "ґ", 1)

    def uk_g(self):
        return self.source.replace("ґ", "г", 1)

    def split_ne(self):
        return self.source.replace("не", "не ")

    def merge_ne(self):
        return self.source.replace("не ", "не")

    def split_na(self):
        return self.source.replace("на", "на ")

    def merge_na(self):
        return self.source.replace("на ", "на")

    def spell_check(self, tokens: List[str], token_pos: int) -> str:
        return spell_checker.suggest(self.source,
                                     tokens,
                                     token_pos)

    def _spelling_transformations_dispatch(self) -> dict:
        return {
            "CASE_CAPITAL": self.case_capital,
            "CASE_LOWER": self.case_lower,
            "MERGE_HYPHEN": self.merge_hyphen,
            "MERGE_SPLIT": self.merge_split,
            "MERGE_SPACE": self.merge_space,
            "DUPLICATE_CHARS": self.duplicate_chars,
            "EXCESSIVE_SOFT_SIGN": self.excessive_soft_sign,
            "MISSING_SOFT_SIGN": self.missing_soft_sign,
            "є": self.uk_e_soft,
            "е": self.uk_e_hard,
            "в_1": self.uk_v_1,
            "у_1": self.uk_y_1,
            "В_1": self.uk_v_upper_1,
            "У_1": self.uk_y_upper_1,
            "ґ_1": self.uk_g_special,
            "г_1": self.uk_g,
            "SPLIT_NE": self.split_ne,
            "MERGE_NE": self.merge_ne,
            "SPLIT_NA": self.split_na,
            "MERGE_NA": self.merge_na,
        }

    def tag(self, target: str, tokens: List[str] = [], token_pos: int = 0):
        """
        Tag mistakes related to Spelling.
        This includes mistakes like lower or upper case and potentially others
        """
        if not self.source:
            return target
        tag = ""
        if self.spell_check(tokens, token_pos) == target:
            return "SPELLCHECKER"
        dispatcher = self._spelling_transformations_dispatch()
        for key, value in dispatcher.items():
            if value() == target:
                tag = key
                break

        # default case
        if not tag:
            # tag = 'SPELLCHECKER'
            tag = target

        return tag

    def correct(self, gtransform: str):
        """
        Correct spelling mistakes
        """
        if gtransform == 'SPELLCHECKER':
            return self.spell_check(tokens=[self.source], token_pos=0)
        dispatcher = self._spelling_transformations_dispatch()
        if gtransform in dispatcher:
            return dispatcher[gtransform]()
        return gtransform


class Grammar(Transformation):

    def __init__(self, source):
        super().__init__(source)
        self.logger = create_logger(name='grammar')

    def _find_best_parsed_morph(self, target: str):
        """Pymorphy returns multiple results for a single word
        The first version is usually the best; however, there are
        many cases when it's not true for source and target corrections.
        For example, target correction may have a different POS
        in the first result or a different normal_form
        The goal is to find the best parsed version where the POS and normal form
        of source and target match
        This approach still has a downside, it doesn't take the context of the word
        into account. For example, the word `як` can be a noun, conjunction, etc.
        and the real POS will depend on the context that we don't check.
        """
        target_parsed = morph.parse(target)[0]
        source_parsed_all = morph.parse(self.source)
        source_parsed = None
        # the POS of the target should match source
        # otherwise, if they have different POS, we can't make a correct prediction
        for parse in source_parsed_all:
            if (parse.tag.POS == target_parsed.tag.POS
                    and parse.normal_form == target_parsed.normal_form):
                source_parsed = parse
                break

        if not source_parsed:
            self.logger.debug(
                "Didn't find a source Parse for target %s", target)
            return None

        # we need to get the normal form of the source word
        # because we cannot inflect some forms of words if they are not in dictionary
        # for example, кесара cannot be inflected but кесар саn be
        source_normal_form_parsed = [
            p for p
            in morph.parse(source_parsed.normal_form)
            if p.tag.POS == target_parsed.tag.POS
            and p.normal_form == source_parsed.normal_form
        ]

        # we need to iterate over each normal form since some of them may have
        # different attributes
        for normal_parsed in source_normal_form_parsed:
            correction = normal_parsed.inflect(target_parsed.tag.grammemes)
            if not correction:
                continue
            if correction.word == target.lower():
                return target_parsed

        return None

    def _is_person(self, parse):
        """Check if the parsed word is a name or surname"""
        named_grammemes = ['Name', 'Patr']
        return any(g for g in named_grammemes if g in parse.tag.grammemes)

    def tag(self, target: str):
        """
        Tag mistakes related to word number, case, gender, and tense.
        The reason for such broad generalization is that many mistakes
        require the same information to be corrected. We may not always
        get all the data but during correction, we can verify its validity,
        and if there are no correct tags, just return the default word.
        This approach is not ideal but simple for the beginning.
        Also, we separately tag words related to names and surnames
        because the only required detail is the target case
        Produced tag will depend on the part of speech of the word:
        VERB: VERB_GENDER_NUMBER_TENSE ->
        підняли: VERB_N/A_plur_past
        буде: VERB_3per_sing_futr
        NOUN/ADJF: NOUN_GENDER_NUMBER_CASE ->
        рукав: NOUN_musc_sing_nomn
        гарні: ADJF_N/A_plur_nomn
        :param target: the target word with the correct case
        """
        correct = ""
        singular = "sing"
        not_available = "N/A"  # replace None values with not available

        target_parsed = self._find_best_parsed_morph(target)
        if not target_parsed:
            self.logger.debug(
                "Couldn't find a correction for %s, source %s", target, self.source)
            return target

        target_tags = target_parsed.tag

        pos = target_tags.POS
        gender = target_tags.gender if target_tags.gender else not_available
        number = target_tags.number if target_tags.number else singular
        tense = target_tags.tense if target_tags.tense else not_available
        case = target_tags.case if target_tags.case else not_available

        if not pos:
            return target
        if self._is_person(target_parsed):
            self.logger.debug("We have a person %s", target)
            correct = f'PERSON_{case}'
        elif pos == "VERB":
            correct = f'{pos}_{gender}_{number}_{tense}'
        else:
            correct = f'{pos}_{gender}_{number}_{case}'

        correction = self.correct(correct)
        if target != correction:
            self.logger.debug("""correction %s doesn't match target %s\n"
                  Falling back to target as the correct version""", correction, target)
            correct = target

        return correct

    def correct(self, gtransform: str):
        """
        Tag example: $REPLACE__G/CASE__NPRO_N/A_plur_datv
        VERB: {pos}_{gender}_{number}_{tense}
        NOUN/etc.: {pos}_{gender}_{number}_{case}
        """
        attributes = gtransform.split('_')
        if len(attributes) != 4 and 'PERSON_' not in gtransform:
            # this means this tag doesn't have the needed structure
            # we will return default
            self.logger.debug(
                "%s is not a proper gtransformation tag", gtransform)
            return gtransform

        garbage_attr = ["N/A", "sing", "PERSON"]
        if 'PERSON_' in gtransform:
            correct_parsed_source = morph.parse(self.source)[0]
        else:
            part_of_speech = attributes[0]
            correct_parsed_source = next(
                (p for p in morph.parse(self.source)
                 if p.tag.POS == part_of_speech),
                None
            )

        if not correct_parsed_source:
            self.logger.debug("couldn't find a Parse for %s with tag %s",
                              self.source, gtransform)
            return self.source

        clean_attributes = [a for a in attributes if a not in garbage_attr]

        correct_token = correct_parsed_source.inflect(set(clean_attributes))

        if not correct_token:
            self.logger.debug("no correction found for %s", self.source)
            return self.source

        correct_word: str = correct_token.word

        # PyMorphy always returns lower case words
        # so we try to restore the word form
        if self.source[0].isupper():
            return correct_word.capitalize()

        self.logger.debug('source - %s, correction - %s',
                          self.source, correct_word)
        return correct_word


class VerbVoice(Transformation):

    def __init__(self, source):
        super().__init__(source)

    def uk_non_reflective(self):
        """Make a verb non-reflective by removing certain endings"""
        if self.source.endswith("ся"):
            return self.source.strip("ся")
        elif self.source.endswith("сь"):
            return self.source.strip("сь")
        return self.source

    def uk_reflective(self):
        """Make a verb reflective"""
        return f'{self.source}ся'

    def _verbvoice_transformations_dispatch(self) -> dict:
        return {
            "REFLECTIVE": self.uk_reflective,
            "NON_REFLECTIVE": self.uk_non_reflective
        }

    def tag(self, target: str):
        """
        Tag mistakes related to verb voice.
        Unfortunately, it's hard to produce a tag for such errors
        because the normal form of erroneous and correct words
        is different. Therefore, for now, we use a simple approach
        to remove or add some word endings.
        """
        tag = ""
        dispatcher = self._verbvoice_transformations_dispatch()
        for key, value in dispatcher.items():
            if value() == target:
                tag = key
                break

        # default case
        if not tag:
            tag = target

        return tag

    def correct(self, gtransform: str):
        """
        Correct spelling mistakes
        """
        dispatcher = self._verbvoice_transformations_dispatch()
        if gtransform in dispatcher:
            return dispatcher[gtransform]()
        return gtransform


class Punctuation(Transformation):

    LINE_START_TOKEN = 'START_'

    def __init__(self, source):
        super().__init__(source)
        self.logger = create_logger(name='punctuation')

    def tag(self, target: str, text_chunk: str):
        """Most of the punctuation errors are generalized on their own
        However, there are cases when we use dash in the beginning of the sentence
        and also in the middle of it. Automatic generalization of such mistakes
        creates new mistakes.
        I'm doing it in a very complex way, I hope to make it simpler
        """
        edge_character = "— "
        # we are performing a clear replacement so we don't need to do anything else
        if self.source:
            return target
        # '— ' is the problematic error, so if we are not handing, just return default
        if target != "— ":
            return target

        correct_text = AnnotatedText(text_chunk).get_corrected_text()

        edge_character_regex = re.compile("^— [А-Я].*")
        if edge_character_regex.match(correct_text):
            self.logger.debug("EDGE CASE: %s", correct_text)
            return f"{self.LINE_START_TOKEN}{edge_character}"

        # default case
        return target

    def correct(self, gtransform: str):
        if gtransform.startswith(self.LINE_START_TOKEN):
            return gtransform.removeprefix(self.LINE_START_TOKEN)
        return gtransform


class Transformations:

    # number, tense, case, gender
    NTCG = ["g/number", "g/tense", "g/case", "g/gender"]

    PREPEND_ERROR = re.compile(r'({[^{}]*?=>.*?:::.*?})\S+')
    APPEND_ERROR = re.compile(r'\S+({[^{}]*?=>.*?:::.*?})')
    SPACE_TOKEN = "_SPACE_"
    NEW_LINE_TOKEN = "_NEWLINE_"

    def __init__(self, source: str):
        self.source: str = source

    def _sanitize(self, raw_correction: str):
        tag = raw_correction.replace('\n', self.NEW_LINE_TOKEN
                                     ).replace(' ', self.SPACE_TOKEN)
        return tag

    def _decode_special_tags(self, raw_correction: str):
        tag = raw_correction.replace(self.NEW_LINE_TOKEN, '\n'
                                     ).replace(self.SPACE_TOKEN, ' ')
        return tag

    def _tag_router(self, error_type: str, target: str, **kwargs):
        """
        Choose the correct function for getting the tag
        """
        error_type = error_type.lower()
        tag = ""
        if error_type in self.NTCG:
            tag = Grammar(self.source).tag(target)
        elif error_type == "spelling":
            original_tokens: List[str] = kwargs.get("original_tokens", [])
            token_pos: int = kwargs.get("token_pos", 0)
            tag = Spelling(self.source).tag(target, original_tokens, token_pos)
        elif error_type == "g/verbvoice":
            tag = VerbVoice(self.source).tag(target)
        elif error_type == "punctuation":
            text_chunk = kwargs.get("text_chunk", "")
            tag = Punctuation(self.source).tag(target, text_chunk)
        else:
            tag = target

        return self._sanitize(tag)

    def get_error_position(self, text_chunk: str) -> str:
        """
        Given a string with an annotation attached to a neighboor word,
        find the position of the word next to the error
        It can be left, right, or empty. This is needed to understand
        if we append a new token or prepend
        e.g.
        right: {=>- :::error_type=Punctuation}паче
        left: паче{=>,:::error_type=Punctuation}
        empty: {  => :::error_type=Punctuation}
        """
        if self.APPEND_ERROR.match(text_chunk):
            return "right"
        if self.PREPEND_ERROR.match(text_chunk):
            return "left"

        return ""

    def get_tag(self, tokens: List[Token], token_pos: int):
        """
        :param error_type: type of the error like Spelling, G/CASE, etc.
        :param target: the correction
        :param text_chunk: the surrounding of the target
        error_position: can be left, right, or empty.
            Right means we append a token, left means we prepend a token,
            empty is discarded
        """
        token = tokens[token_pos]
        error_type: str = token.first_annotation.meta["error_type"]
        error_type = error_type.upper()
        target = token.first_annotation.top_suggestion
        text_chunk = str(token.annotated)
        # tokens without annotations
        original_tokens = [t.annotated.get_original_text() for t in tokens]

        operation = ''

        if target is None:
            return None
        elif target == '':
            return '$DELETE'

        tag = self._tag_router(error_type, target,
                               text_chunk=text_chunk,
                               original_tokens=original_tokens,
                               token_pos=token_pos)
        # we are looking at a non-existing token
        if not self.source:
            error_position = self.get_error_position(text_chunk)
            if error_position == "right":
                operation = '$APPEND'
            elif error_position == "left":
                operation = '$PREPEND'
            # This indicates some errors with parsing,
            # and they occur when there are 2+ errors
            # associated with the correction
            # let's default to APPEND as this is the most usual case
            else:
                operation = '$APPEND'
        else:
            operation = '$REPLACE'

        if error_type.lower() in self.NTCG:
            error_type = "NTCG"

        return f'{operation}__{error_type}__{tag}'

    def correct_by_tag(self, tag: str):
        """
        Parse tags from the prediction and suggest a correction
        """
        if tag == "$DELETE":
            return ""
        transform_type, error_type, gtransform = tag.split('__', maxsplit=2)
        error_type = error_type.lower()
        correction = ""
        if error_type in self.NTCG or error_type.upper() == "NTCG":
            correction = Grammar(self.source).correct(gtransform)
        elif error_type == "spelling":
            correction = Spelling(self.source).correct(gtransform)
        elif error_type == "g/verbvoice":
            correction = VerbVoice(self.source).correct(gtransform)
        elif error_type == "punctuation":
            correction = Punctuation(self.source).correct(gtransform)
        else:
            correction = gtransform

        correction = self._decode_special_tags(correction)

        if transform_type == "$APPEND":
            return f"{self.source}{correction}"
        elif transform_type == "$PREPEND":
            return f"{correction}{self.source}"
        elif transform_type == "$REPLACE":
            return correction

        return self.source
