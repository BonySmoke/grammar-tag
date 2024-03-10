import re
from ua_gec import AnnotatedText
from transformers import pipeline, Pipeline
from .transformations import Transformations
from .log import create_logger


class WordAligner:

    PUNCTUATION = [",", ".", "!", "?", ":", ";", "&"]
    MERGE_TAGS = ["$KEEP", "$DELETE"]

    def __init__(self) -> None:
        pass

    def separate_words(self, previous: dict, current: dict) -> bool:
        return previous["word"].endswith(' ') or current["word"].startswith(' ')

    def separate_punctuation(self, previous: dict, current: dict) -> bool:
        return (any(previous["word"].endswith(p) for p in self.PUNCTUATION)
                    or any(current["word"].startswith(p) for p in self.PUNCTUATION))

    def not_merge_tags(self, previous: dict, current: dict) -> bool:
        return (current['entity_group'] not in self.MERGE_TAGS
                    and previous["entity_group"] not in self.MERGE_TAGS)

    def replace_punctuation(self, previous: dict, current: dict) -> bool:
        return ("REPLACE__PUNCTUATION" in current['entity_group']
                    or "REPLACE__PUNCTUATION" in previous["entity_group"])

    def align_words(self, prediction: list):
        merge_tags = ["$KEEP", "$DELETE"]
        groups = list()
        for group in prediction:
            group = group.copy()
            if not groups:
                groups.append(group)
                continue

            previous = groups[-1]
            # the previous word ends with a space or the new word starts with a space
            # which means these are separate words
            if self.separate_words(previous, group):
                groups.append(group)
                continue
            # the previous word or the new word ends/starts with a punctuation sign
            # let's consider these 2 different groups
            if self.separate_punctuation(previous, group):
                groups.append(group)
                continue
            # we are dealing with different tags, we must not merge them
            # for example, ["APPEND", "APPEND"] will be merged but
            # ["KEEP", "KEEP"] won't be
            if self.not_merge_tags(previous, group):
                if group['entity_group'] != previous["entity_group"]:
                    groups.append(group)
                    continue
            if self.replace_punctuation(previous, group):
                groups.append(group)
                continue

            old_word = previous["word"].split()[-1]
            new_word = group["word"].split()[0]
            merged_group_word = old_word + new_word

            # this means the group we are merging with is the current group
            if group["entity_group"] not in merge_tags:
                target_group = group
                start, end = target_group["start"] - \
                    len(old_word), target_group["end"]
            # we are merging with is the previous group
            else:
                target_group = previous
                start, end = target_group["start"], target_group["end"] + \
                    len(new_word)

            previous.update({
                "entity_group": target_group["entity_group"],
                "score": target_group["score"],
                "word": merged_group_word,
                "start": start,
                "end": end
            }
            )
        return groups


class Correcter:

    DEFAULT_PIPELINE = "BonySmoke/gec_uk_seq2tag"

    def __init__(self,
                 text: str,
                 min_score: float = 0.2,
                 min_delete_score: float = 0.6,
                 classifier: Pipeline = None,
                 exclude_tags: list = []):
        """
        :param min_score: the minimum model confidence for a tag to perform a correction
        :param min_delete_score: the minimum model confidence for the $DELETE tag
          to perform a correction
        :param classifier: a Pipeline that will perform the tag prediction
        :param exclude_tags: a list of tags to exclude. This list must contain exact tags
            that are computed, e.g. $PREPEND__PUNCTUATION__START_—_SPACE_
        """
        self.classifier = classifier if classifier else self._load_classifier()
        self.annotated_text = AnnotatedText(re.sub(' +', ' ', text))
        self.min_score = min_score
        self.min_delete_score = min_delete_score
        self.exclude_tags = exclude_tags
        self.logger = create_logger(name='correcter')

    @property
    def original_text(self):
        return self.annotated_text.get_original_text()

    def align_words(self, prediction: list):
        return WordAligner().align_words(prediction)

    def predict(self):
        prediction = self.classifier(self.original_text)
        prediction = self.align_words(prediction)

        # prediction without $KEEP and with confidence over min_score
        clean_prediction = list()
        for span in prediction:
            entity = span["entity_group"]
            if not entity:
                continue
            if entity == "$KEEP":
                continue
            if entity in self.exclude_tags:
                self.logger.debug(
                    f"Skipping tag {entity} because it's present in the exclude list")
                continue
            # skip predictions with low confidence to avoid creating mistakes
            if span["score"] < self.min_score:
                continue
            if entity == "$DELETE" and span["score"] < self.min_delete_score:
                continue
            clean_prediction.append(span)

        return clean_prediction

    def _load_classifier(self):
        return pipeline(model=self.DEFAULT_PIPELINE,
                        aggregation_strategy="simple")

    def _correct_by_tag(self, tag: str, token: str):
        """
        Parse tags from the prediction and suggest a correction
        """
        return Transformations(token).correct_by_tag(tag)

    def annotate(self, prediction) -> AnnotatedText:
        """
        Create an annotated version of the text
        Each annotation contains an error and a top suggestion
        """
        for span in prediction:
            start, end = span["start"], span["end"]
            # if we need to remove an excessive space
            # pipeline returns the same start and end of the token
            # but the word is correct
            if start == end and str(span["word"]).isspace():
                start = start - len(span["word"])
            span_text = self.original_text[start:end]
            correction = self._correct_by_tag(span["entity_group"], span_text)
            self.annotated_text.annotate(
                start=start,
                end=end,
                correct_value=correction,
                meta=span["entity_group"]
            )

        return self.annotated_text

    def correct(self, stages=3) -> dict:
        """
        Get a corrected version of the text
        :param stages: the number of times to perform the correction
            Increasing this param may likely improve the end result,
            the cost is speed but the difference will be barely noticeable.
            Defaults to 3 as suggested in the GECToR paper
        """
        corrections = {
            "stages": list(),
            "final": str()
        }

        for _ in range(stages):
            stage = {"text": str(), "annotations": list(), "prediction": dict}
            prediction = self.predict()
            self.annotate(prediction)

            for ann in self.annotated_text.iter_annotations():
                stage["annotations"].append(ann)
                self.annotated_text.apply_correction(ann)

            stage["text"] = str(self.annotated_text)
            stage["prediction"] = prediction
            corrections["stages"].append(stage)

        corrections["final"] = str(self.annotated_text)
        return corrections
