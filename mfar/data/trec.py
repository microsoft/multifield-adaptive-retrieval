import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, TextIO, Tuple

csv.field_size_limit(sys.maxsize)

@dataclass
class QRels:
    """
    Represents a QRels (query relevance) item for the standard `trec_eval` tool.
    This represents one gold relevance judgement for a given query.
    """

    query_id: str
    doc_id: str
    relevance: float
    _iter: str = "0"  # useless but required by trec_eval

    def __str__(self):
        return f"{self.query_id}\t{self._iter}\t{self.doc_id}\t{self.relevance}"

    @classmethod
    def from_str(cls, s: str) -> "QRels":
        query_id, _iter, doc_id, relevance = s.split("\t")
        return cls(query_id, doc_id, float(relevance), _iter)

    @classmethod
    def from_text_io(cls, f: TextIO) -> List["QRels"]:
        return [cls.from_str(line.strip()) for line in f]


@dataclass
class QRes:
    """
    Represents a QRes (query result) item for the standard `trec_eval` tool.
    This represents one retrieved document (from a retriever) for a given query.
    """

    query_id: str
    doc_id: str
    sim: float
    run_id: str = "0"
    _iter: str = "0"  # useless but required by trec_eval
    _rank: int = 0  # useless but required by trec_eval

    def __str__(self):
        return f"{self.query_id}\t{self._iter}\t{self.doc_id}\t{self._rank}\t{self.sim}\t{self.run_id}"

    @classmethod
    def from_str(cls, s: str) -> "QRes":
        query_id, _iter, doc_id, _rank, sim, run_id = s.split()
        return cls(query_id, doc_id, float(sim), run_id, _iter, int(_rank))

    @classmethod
    def from_text_io(cls, f: TextIO) -> "List[QRes]":
        return [cls.from_str(line.strip()) for line in f]


def parse_trec_eval_output(output: str) -> Dict[str, float]:
    """
    Parses the command-line output from `trec_eval` into a dictionary of metrics.
    """
    non_metric_metadata_keys: Set[str] = {
        "runid",
        "num_q",
        "num_ret",
        "num_rel",
        "num_rel_ret",
    }
    metrics = {}
    for line in output.split("\n"):
        if line == "":
            continue
        metric, _, value = line.strip().split("\t")
        metric, value = metric.strip(), value.strip()
        if metric not in non_metric_metadata_keys:
            metrics[metric] = float(value)
    return metrics


def call_trec_eval_and_get_metrics(qrels: str, qres: str) -> Dict[str, float]:
    """
    Calls the `trec_eval` tool and returns the metrics.
    """
    proc = subprocess.run(
        ["trec_eval", "-m", "all_trec", qrels, qres], stdout=subprocess.PIPE, check=True
    )
    metrics_str = proc.stdout.decode("utf-8")
    metrics = parse_trec_eval_output(metrics_str)
    return metrics


def read_corpus(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                yield row[0], ""
            else:
                try:
                    yield row[0], json.loads(row[1])
                except:
                    yield row[0], "\t".join(row[1:])