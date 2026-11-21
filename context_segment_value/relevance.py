"""A TF-IDF relevance scorer, stdlib only.

This is the baseline ranker the value packer has to beat: the way
most context builders actually choose what fits a budget -- score
each chunk by lexical relevance to the query and keep the top of the
list. It is a fair baseline and a dangerous one: distractor segments
are *built* to be lexically relevant, so relevance ranks the poison
near the top.
"""

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class TfidfScorer(object):
    """Cosine TF-IDF of each segment against the task query, with
    inverse document frequencies computed over the task's own
    segment corpus plus the query."""

    def __init__(self, task):
        docs = [tokenize(task.query)]
        for seg in task.segments:
            docs.append(tokenize(seg.text))
        self.n_docs = len(docs)
        doc_freq = {}
        for doc in docs:
            for token in set(doc):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self.idf = {}
        for token, freq in doc_freq.items():
            self.idf[token] = math.log(
                (1.0 + self.n_docs) / (1.0 + freq)) + 1.0
        self.query_vector = self._vector(tokenize(task.query))
        self.segment_vectors = {}
        for seg in task.segments:
            self.segment_vectors[seg.segment_id] = self._vector(
                tokenize(seg.text))

    def _vector(self, tokens):
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        vector = {}
        for token, count in counts.items():
            vector[token] = count * self.idf.get(token, 0.0)
        return vector

    @staticmethod
    def _cosine(vec_a, vec_b):
        if not vec_a or not vec_b:
            return 0.0
        dot = 0.0
        for token, weight in vec_a.items():
            if token in vec_b:
                dot += weight * vec_b[token]
        norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
        norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(self, segment_id):
        return self._cosine(self.query_vector,
                            self.segment_vectors[segment_id])

    def scores(self):
        return dict((sid, self.score(sid)) for sid in self.segment_vectors)
