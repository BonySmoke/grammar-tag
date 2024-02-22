import re
from ua_gec import Corpus, AnnotatedText
from ua_gec.annotated_text import Annotation
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
import numpy as np
import evaluate
from typing import List
from .transformations import Token, Text
from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForTokenClassification,
                          AutoModelForTokenClassification, EarlyStoppingCallback)

class Text:

    # inspired by https://stackoverflow.com/questions/9644784/splitting-on-spaces-except-between-certain-characters
    # split an annotated string by annotation and namespace but ommit spaces in annotations
    ANNOTATION_SPACE_SPLIT_REGEX = re.compile(
        r'\s+(?=[^{}]*(?:\{.*:::.*\}|$))')
    # split an annotated string only by annotation that contains source text
    # e.g. `мінімум {, =>:::error_type=Punctuation}{=>— :::error_type=Punctuation}знати`
    # -> ["мінімум ", "{, =>:::error_type=Punctuation}", "{=>— :::error_type=Punctuation}знати"]
    ANNOTATION_SPLIT_REGEX = re.compile(r'({[^{}]+?=>.*?:::.*?})')

    def __init__(self, text: str, metadata) -> None:
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

class Seq2TagManager:

    DEFAULT_TAGS = ["$KEEP", "$DELETE"]

    def __init__(self,
                 corpus: Corpus,
                 min_error_occurrence=3):
        """
        :param min_error_occurrence: the minimum number of times the error should be
            found in the corpus to create a tag. If this number is not reached,
            the error will be tagged as unhandled in the corpus.
            This is needed to avoid a lot of noisy errors that the model will not learn
            from a very small number of samples
        :attribute eligible_tags: tags that occur in the corpus >= min_error_occurrence
        """
        self._original_corpus = self._corpus = corpus
        self.processed_corpus = []
        self.tags = self._load_tags()
        self._tags_stats = {}  # tag:number of occurrences. Generated during training
        self.min_error_occurrence = min_error_occurrence
        self.eligible_tags = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seqeval = evaluate.load("seqeval")
        self.tokenizer = None
        self.data_collator = None
        self.model = None
        self.trainer = None

    def _load_tags(self):
        """
        Load tags from the file to have a base
        """
        return self.DEFAULT_TAGS.copy()

    @property
    def original_corpus(self):
        return self._original_corpus

    @property
    def corpus(self):
        return self._corpus

    @property
    def stats(self):
        return self._tags_stats

    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)
        self._add_stats(tag)

    def _add_stats(self, tag: str):
        """
        Account the tag in the statistics
        """
        self._tags_stats[tag] = self._tags_stats.get(tag, 0) + 1

    def get_index(self, tag):
        if tag not in self.tags:
            raise ValueError(
                f"Cannot find index for tag {tag} as it's not in the list")
        return self.tags.index(tag)

    def _set_eligible_tags(self):
        """Create a list of eligible tags"""
        self.eligible_tags = self.DEFAULT_TAGS + [
            tag for tag, value
            in self.stats.items()
            if value >= self.min_error_occurrence
            and tag not in self.DEFAULT_TAGS
        ]
        return self.eligible_tags

    def get_index_eligible(self, tag):
        """Get an eligible tag. If the requested tag is not present
        in the eligible_tags list, we return the index of the KEEP tag"""
        if tag not in self.eligible_tags:
            return self.eligible_tags.index("$KEEP")
        return self.eligible_tags.index(tag)

    def create_huggingface_dataset(self):
        df = self.create_pandas_dataset()
        train = df[df["partition"] == "train"]
        test = df[df["partition"] == "test"]
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train),
            "test": Dataset.from_pandas(test)
        })

        return dataset

    def id_to_label(self):
        """
        Map label IDs to values
        """
        return {i: self.eligible_tags[i] for i in range(len(self.eligible_tags))}

    def label_to_id(self):
        """
        Map label value to ID
        """
        return {self.eligible_tags[i]: i for i in range(len(self.eligible_tags))}

    def generate_error_type_stats(self):
        """
        Generate a repo with the number of unique tags per error_type in the corpus
        The error_type is the second index of a tag.
        The tag looks like this $TRANSFORMATION__ERROR_TYPE__TAG
        The statistics is generate based on the tags property
        """
        stats = {}
        for tag in self.tags:
            attributes = tag.split('__')
            # not a proper tag
            if len(attributes) < 3:
                continue
            error_type = attributes[1]
            stats[error_type] = stats.get(error_type, 0) + 1

        return stats

    def create_pandas_dataset(self):
        """
        Convert the processed corpus to the pandas dataframe
        If the corpus wasn't processed, process it
        """
        if not self.processed_corpus:
            self.process_corpus()

        tagged_sequences = list()
        for doc in self.processed_corpus:
            tagged_sequences.append(self.sequence_to_dataset_entry(doc))
        return pd.DataFrame(tagged_sequences)

    def process_corpus(self) -> List[List[Token]]:
        """
        Convert the corpus to the chunked version.
        Each element of the corpus will be a single line from the corpus
        split on whitespace as well as error annotation.
        """
        self.processed_corpus = []
        for doc in self._corpus:
            annotated: AnnotatedText = doc.annotated
            raw_annotated_rows = [a for a in
                                  annotated.get_annotated_text().split('\n') if a]

            for row in raw_annotated_rows:
                text = Text(text=row, metadata=doc.meta)
                text.tag()
                self.process_sequence(text)
                self.processed_corpus.append(text)

        self._set_eligible_tags()
        return self.processed_corpus

    def process_sequence(self, sequence: Text):
        for token in sequence.tokens:
            if not token.tag:
                continue
            self.add_tag(token.tag)

    def sequence_to_dataset_entry(self, text: Text):
        """
        Given a text with annotations, create a dictionary with mapping of tokens to tags.
        This format is suitable for creating Pandas or Huggingface datasets
        """
        default_tag = self.get_index("$KEEP")  # no errors in token
        data = dict(
            original_text=text.text.get_original_text(),
            tokens=[],
            ner_tags=[],
            partition=text.metadata.partition
        )

        for token in text.tokens:
            original = token.annotated.get_original_text()
            data["tokens"].append(original)

            if not token.tag:
                data["ner_tags"].append(default_tag)
                continue

            transformation_index = self.get_index_eligible(token.tag)
            data["ner_tags"].append(transformation_index)

        return data

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            # Map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.tags[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions,
                                       references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def load_model(self):
        """Load model, tokenizer, and data collator"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "youscan/ukr-roberta-base",
            add_prefix_space=True)
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer)

        self.model = AutoModelForTokenClassification.from_pretrained(
            "youscan/ukr-roberta-base",
            num_labels=len(self.eligible_tags),
            id2label=self.id_to_label(),
            label2id=self.label_to_id()
        ).to(self.device)

    def train(self):
        dataset = self.create_huggingface_dataset()
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        train_tokenized = train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names)
        test_tokenized = test_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=test_dataset.column_names)

        test_tokenized_selected = test_tokenized.select(range(500))

        training_args = TrainingArguments(
            output_dir="gec_uk_seq2tag",
            learning_rate=2e-5,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model = 'f1'
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized_selected,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        self.trainer.train()
        return self.trainer

    def push(self, commit_message: str):
        return self.trainer.push_to_hub(commit_message=commit_message)
