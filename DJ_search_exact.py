import os
import sys
import nltk
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from dataclasses import dataclass, field
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from transformers import AutoTokenizer

md = MosesDetokenizer(lang='en')
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN,
                                          add_bos_token=False, add_eos_token=False)

# Initialized in main() via --index_dir
engine = None


@dataclass
class Document:
    doc_id: str
    tokens: List[str]  # [num_tokens]


@dataclass
class Span:
    start_index: int
    end_index: int
    span_text: str
    occurrence: int
    sources: list = field(default_factory=list)  # [{title, author, source_file}, ...]


class Hypothesis:
    def __init__(self, target_doc: Document, min_ngram: int) -> None:
        self.target_doc = target_doc
        self.min_ngram = min_ngram
        self.spans = []
        self.finished = False

    def add_span(self, new_span: Span) -> None:
        self.spans.append(new_span)
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def replace_span(self, new_span: Span) -> None:
        self.spans = self.spans[:-1] + [new_span]
        if new_span.end_index >= len(self.target_doc.tokens):
            self.finished = True

    def get_score(self) -> float:
        if not self.spans:
            return 0
        progress_len = self.spans[-1].end_index if not self.finished else len(self.target_doc.tokens)
        flags = [False for _ in range(progress_len)]
        for span in self.spans:
            span_length = span.end_index - span.start_index
            flags[span.start_index: span.end_index] = [True] * span_length
        coverage = len([fa for fa in flags if fa]) / len(flags)
        return coverage

    def format_span(self) -> str:
        return ' | '.join([s.span_text for s in self.spans])

    def __hash__(self) -> int:
        return hash(self.format_span())

    def __eq__(self, other) -> bool:
        if isinstance(other, Hypothesis):
            return self.format_span() == other.format_span()
        return NotImplemented

    def get_avg_span_len(self) -> float:
        if not self.spans:
            return 0
        span_len = [s.end_index - s.start_index for s in self.spans]
        return sum(span_len) / len(span_len)

    def export_json(self) -> dict:
        matched_spans = [{'start_index': s.start_index,
                          'end_index': s.end_index,
                          'span_text': s.span_text,
                          'occurrence': s.occurrence,
                          'sources': s.sources} for s in self.spans]
        return {
            'matched_spans': matched_spans,
            'coverage': self.get_score(),
            'avg_span_len': self.get_avg_span_len(),
        }


def find_exact_match(detokenize: Callable, doc: Document, min_ngram: int, exclude_source_file: str = None):
    hypothesis = Hypothesis(doc, min_ngram)

    first_pointer, second_pointer = 0, min_ngram
    while second_pointer <= len(doc.tokens):
        span_text = detokenize(doc.tokens[first_pointer: second_pointer])
        input_ids = tokenizer.encode(span_text)
        if len(input_ids) > 0 and input_ids[0] == 29871:
            input_ids = input_ids[1:]
        search_result = engine.count(input_ids=input_ids)
        occurrence = search_result.get('count', 0)

        if occurrence:
            # Get attribution: which reference book(s) contain this span
            sources = []
            try:
                doc_result = engine.search_docs(input_ids=input_ids, maxnum=10, max_disp_len=1)
                total_cnt = doc_result.get('cnt', occurrence)
                self_match_count = 0
                for ref_doc in doc_result.get('documents', []):
                    meta = json.loads(ref_doc.get('metadata', '{}')).get('metadata', {})
                    source = {
                        'title': meta.get('title', ''),
                        'author': meta.get('author', ''),
                        'source_file': meta.get('source_file', ''),
                    }
                    if exclude_source_file and source['source_file'] == exclude_source_file:
                        self_match_count += 1
                    else:
                        sources.append(source)
                # Deduplicate by title
                seen = set()
                sources = [s for s in sources if s['title'] not in seen and not seen.add(s['title'])]
                # If total count > self matches, other books definitely match
                # If we only got self-matches in the sample but total is higher, keep it
                if exclude_source_file:
                    if not sources and total_cnt <= self_match_count:
                        occurrence = 0
            except Exception:
                pass

        if occurrence:
            matched_span = Span(start_index=first_pointer,
                                end_index=second_pointer,
                                span_text=span_text,
                                occurrence=occurrence,
                                sources=sources)
            if not hypothesis.spans:
                hypothesis.add_span(matched_span)
            else:
                last_span = hypothesis.spans[-1]
                if matched_span.start_index <= last_span.start_index and last_span.end_index <= matched_span.end_index:
                    hypothesis.replace_span(matched_span)
                else:
                    hypothesis.add_span(matched_span)
            second_pointer += 1

            print("***************************************************************************************************")
            print(hypothesis.format_span())
            print(f'score: {hypothesis.get_score():.4f}  avg_span_length: {hypothesis.get_avg_span_len()}')
            print("***************************************************************************************************")

        else:
            if second_pointer - first_pointer > min_ngram:
                first_pointer += 1
            elif second_pointer - first_pointer == min_ngram:
                first_pointer += 1
                second_pointer += 1
            else:
                raise ValueError

    hypothesis.finished = True
    return hypothesis.export_json()


def dj_search(data_path, output_file, min_ngram, subset=100, lm_tokenizer=False):
    all_data = json.load(open(data_path))[:subset]
    if not lm_tokenizer:
        tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
        detokenize = lambda x: md.detokenize(x)
    else:
        tokenize_func = lambda x: tokenizer.tokenize(x)
        detokenize = lambda x: tokenizer.decode(tokenizer.convert_tokens_to_ids(x))

    # Get source_file from target data for self-match exclusion (before resume slicing)
    exclude_source_file = all_data[0].get('source_file', '') if all_data else ''
    if exclude_source_file:
        print(f'Excluding self-matches from: {exclude_source_file}')

    outputs = []
    data = all_data
    if os.path.isfile(output_file):
        outputs = json.load(open(output_file, 'r'))
        data = all_data[len(outputs):]
        print(f'resume from previous output file with {len(outputs)} items')

    for t_idx, t_doc in tqdm(enumerate(data), desc='target docs', total=len(data)):
        tokenized_text = tokenize_func(unidecode(t_doc['text']))
        tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)
        if len(tgt_doc.tokens) <= min_ngram:
            continue

        output = find_exact_match(detokenize, tgt_doc, min_ngram, exclude_source_file=exclude_source_file)
        t_doc.update(output)
        outputs.append(t_doc)

        avg_coverage = np.average([x['coverage'] for x in outputs])
        std = np.std([x['coverage'] for x in outputs])
        avg_len = np.average([x['avg_span_len'] for x in outputs])
        print(f'average {min_ngram}-ngram coverage: {avg_coverage:.3f}, std: {std:.3f}, average length: {avg_len}')

        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
            f.flush()


def main():
    global engine

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'infini-gram', 'pkg'))
    from infini_gram.engine import InfiniGramEngine

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='GPT3_book',
                        help="which type of corpus to analyze")
    parser.add_argument('--data', type=str,
                        default='data/book/GPT3_book.json')
    parser.add_argument('--output_dir', type=str,
                        default=f'outputs/exact/book')
    parser.add_argument("--min_ngram", type=int, default=5,
                        help="minimum n-gram size")
    parser.add_argument("--subset", type=int, default=100,
                        help="size of example subset to run search on")
    parser.add_argument('--lm_tokenizer', action='store_true',
                        help='whether to LM tokenizer instead of whitespace tokenizer')
    parser.add_argument('--index_dir', type=str, nargs='+', required=True,
                        help='path(s) to infini-gram index directory(ies). Multiple dirs for incremental indexing.')

    args = parser.parse_args()

    engine = InfiniGramEngine(
        s3_names=[],
        index_dir=args.index_dir,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f'Loaded infini-gram index from {args.index_dir} ({len(args.index_dir)} dir(s))')

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.task + '.json')
    dj_search(args.data, args.output_file, args.min_ngram, args.subset, args.lm_tokenizer)


if __name__ == '__main__':
    main()

