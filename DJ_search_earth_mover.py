import os
import json
try:
    import torch
except ImportError:
    torch = None  # Only needed for get_lookup_table(), not for prebuilt .npy tables
import pickle
import gc
import nltk
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from nltk.corpus import stopwords
from string import punctuation
from transformers import AutoModelForCausalLM, AutoTokenizer

from DJ_search_exact import Document, Hypothesis

HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
md = MosesDetokenizer(lang='en')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=HF_TOKEN)

punctuations = list(punctuation)
stop_words = stopwords.words('english') + ["'m", "'d", "'ll", "'o", "'re", "'ve", "'y"]
stop_tokens = set([t for w in stop_words for t in tokenizer.tokenize(w)])




@dataclass
class SoftSpan:
    start_index: int
    end_index: int
    span_text: str
    ref_span_text: str
    score: float
    ref_title: str = ''
    ref_author: str = ''
    ref_chunk_id: int = -1


@dataclass
class RefDocument:
    token_ids: List[int]
    content_token_ids: List[int]
    content_token_indices: List[int]
    title: str = ''
    author: str = ''
    chunk_id: int = -1


class SoftHypothesis(Hypothesis):
    def format_span(self) -> str:
        formatted_spans, start_indices = [], set()
        for s in self.spans[::-1]:
            if s.start_index in start_indices:
                continue
            if s.span_text == s.ref_span_text:
                formatted_spans.append(s.span_text)
            else:
                formatted_spans.append(f'{s.span_text} ({s.ref_span_text})')
            start_indices.add(s.start_index)
        return ' | '.join(formatted_spans[::-1])

    def export_json(self) -> dict:
        matched_spans = [{'start_index': s.start_index,
                          'end_index': s.end_index,
                          'span_text': s.span_text,
                          'ref_span_text': s.ref_span_text,
                          'score': s.score,
                          'ref_title': s.ref_title,
                          'ref_author': s.ref_author,
                          'ref_chunk_id': s.ref_chunk_id} for s in self.spans]
        return {
            'matched_spans': matched_spans,
            'coverage': self.get_score(),
            'avg_span_len': self.get_avg_span_len(),
        }


def tokenize(x):
    return nltk.tokenize.casual.casual_tokenize(unidecode(x))


def detokenize(x):
    return md.detokenize(x)


def convert_phrase_to_tokens(text, return_index=False):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    output_tokens, output_indices = [], []
    for i, t in enumerate(tokens):
        token = t.replace('Ġ', '').lower()
        is_stopword = token in stop_tokens
        is_punc = all([c in punctuations for c in token])
        if not is_stopword and not is_punc:
            output_tokens.append(t)
            output_indices.append(i)
    output_token_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    if not return_index:
        return output_token_ids
    return output_token_ids, output_indices, token_ids


def compute_similarity(source_token_ids, tgt_token_ids, sim_table):
    # Vectorized: same result as the original for-loop, but ~100x faster
    return float(sim_table[source_token_ids][:, tgt_token_ids].max(axis=1).mean())


def get_lookup_table(model_name="meta-llama/Meta-Llama-3-8B-Instruct", save_path='data/embed_distance/Llama-3-8B-Instruct.pkl'):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    embed_table = model.get_input_embeddings().weight.to('cuda')
    num_vocab = embed_table.shape[0]

    del model
    gc.collect()
    torch.cuda.empty_cache()

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    sim_table = torch.zeros((num_vocab, num_vocab)).to('cpu')
    with torch.no_grad():
        for i in tqdm(range(num_vocab)):
            # [1, H] -> [#vocab, H]
            word_embed = embed_table[i][None, :].expand(num_vocab, -1)
            # compute similarity -> [#vocab, #vocab]
            sim_score = cos_sim(word_embed, embed_table).to('cpu')
            sim_table[i] = sim_score

            del word_embed
            gc.collect()
            torch.cuda.empty_cache()

    sim_table = sim_table.detach().numpy()
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(sim_table, f)


def compute_earth_mover_distance(file_name, save_file_name, embed_table_path='data/embed_distance/Llama-3-8B-Instruct.pkl'):
    with open(embed_table_path, 'rb') as f:
        sim_table = pickle.load(f)
    print('embedding distance table loaded')
    with open(file_name, 'r') as f:
        items = json.load(f)

    outputs = []
    for item in tqdm(items):
        for span_item in item['matched_spans']:
            target_span, ref_span = span_item['target_span_text'], span_item['ref_span_text']
            target_token_ids, ref_token_ids = convert_phrase_to_tokens(target_span), convert_phrase_to_tokens(ref_span)
            if not target_token_ids or not ref_token_ids:
                continue
            similarity = min(compute_similarity(target_token_ids, ref_token_ids, sim_table),
                             compute_similarity(ref_token_ids, target_token_ids, sim_table))
            outputs.append({
                'target_span_text': target_span,
                'ref_span_text': ref_span,
                'spanBert_score': span_item['similarity_score'][0],
                'earth_mover_score': similarity,
            })
    with open(save_file_name, 'w') as f:
        json.dump(outputs, f, indent=4)


def find_matched_span(tgt_token_ids: List[int], threshold: float, min_content_word: int, sim_table: np.ndarray,
                      ref_doc: RefDocument):
    try:
        tgt_arr = np.array(tgt_token_ids)
        ref_arr = np.array(ref_doc.content_token_ids)

        if len(ref_arr) == 0 or len(tgt_arr) == 0:
            return {}

        # Vectorized ref_to_tgt: for each ref token, max similarity to any tgt token
        ref_to_tgt_scores = sim_table[ref_arr][:, tgt_arr].max(axis=1)

        # Cumulative sum for sub-array mean computation
        cumulative_sum = np.zeros(len(ref_to_tgt_scores) + 1)
        np.cumsum(ref_to_tgt_scores, out=cumulative_sum[1:])

        matched_spans = {}
        for start in range(len(ref_to_tgt_scores)):
            for end in range(start + min_content_word, len(ref_to_tgt_scores) + 1):
                subarray_sum = cumulative_sum[end] - cumulative_sum[start]
                subarray_length = end - start
                subarray_mean = subarray_sum / subarray_length

                if subarray_mean >= threshold:
                    cand_token_ids = ref_doc.content_token_ids[start: end]
                    tgt_to_ref_score = compute_similarity(tgt_arr, cand_token_ids, sim_table)
                    if tgt_to_ref_score >= threshold:
                        final_score = float(min(subarray_mean, tgt_to_ref_score))
                        start_idx = ref_doc.content_token_indices[start]
                        end_idx = ref_doc.content_token_indices[end] if end < len(ref_doc.content_token_ids) else len(ref_doc.token_ids)
                        ref_span_text = tokenizer.decode(ref_doc.token_ids[start_idx: end_idx])
                        matched_spans[ref_span_text] = (final_score, ref_doc.title, ref_doc.author, ref_doc.chunk_id)
    except Exception as e:
        print(f'    [warning] find_matched_span error: {e}', flush=True)
        return {}
    return matched_spans


def find_soft_match(doc: Document, reference_docs: List[RefDocument], min_ngram: int, min_content_word: int,
                    threshold: float, sim_table: np.ndarray):
    hypothesis = SoftHypothesis(doc, min_ngram)
    total_tokens = len(doc.tokens)
    num_ref_docs = len(reference_docs)
    step = 0

    first_pointer, second_pointer = 0, min_ngram
    while second_pointer <= total_tokens:
        step += 1
        if step % 50 == 0:
            progress = second_pointer / total_tokens * 100
            print(f'    [pointer progress] step={step}, pos={second_pointer}/{total_tokens} ({progress:.0f}%), '
                  f'coverage={hypothesis.get_score():.4f}, spans={len(hypothesis.spans)}', flush=True)

        span_text = detokenize(doc.tokens[first_pointer: second_pointer])
        span_token_ids = convert_phrase_to_tokens(span_text)

        if len(span_token_ids) < min_content_word:
            second_pointer += 1
            continue

        matched_spans = {}
        for ref_doc in reference_docs:
            matched_spans.update(find_matched_span(span_token_ids, threshold, min_content_word, sim_table, ref_doc))

        if matched_spans:
            for ref_span_text, (ref_score, ref_title, ref_author, ref_chunk_id) in matched_spans.items():
                matched_span = SoftSpan(start_index=first_pointer,
                                        end_index=second_pointer,
                                        span_text=span_text,
                                        ref_span_text=ref_span_text,
                                        score=ref_score,
                                        ref_title=ref_title,
                                        ref_author=ref_author,
                                        ref_chunk_id=ref_chunk_id)
                hypothesis.add_span(matched_span)
            second_pointer += 1

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


def _process_one_chunk(args):
    """Worker function for parallel chunk processing. Loads sim_table via mmap (shared memory)."""
    t_idx, t_doc, n_docs, min_ngram, min_content_word, threshold, embed_table_path = args
    import time as _time
    chunk_start = _time.time()

    # Load sim_table as read-only mmap — all workers share the same physical memory
    sim_table = np.load(embed_table_path, mmap_mode='r')

    tgt_doc = Document(f'tgt_{t_idx}', tokenize(t_doc['text']))
    if len(tgt_doc.tokens) <= min_ngram:
        return None

    retrieved_docs = []
    sorted_ratios = sorted([d['hit_ratio'] for k, v in t_doc['retrieved_docs'].items() for d in v], reverse=True)
    min_hit_ratio = sorted_ratios[min(n_docs, len(sorted_ratios) - 1)]
    for q_idx, q_docs in t_doc['retrieved_docs'].items():
        sorted_q_docs = sorted(q_docs, key=lambda x: x['hit_ratio'], reverse=True)
        selected_q_docs = [d for d in sorted_q_docs if d['hit_ratio'] >= min_hit_ratio]
        selected_q_docs = selected_q_docs if selected_q_docs else [sorted_q_docs[0]]
        retrieved_docs.extend(selected_q_docs)

    reference_docs = []
    for doc in retrieved_docs:
        content_token_ids, content_token_indices, token_ids = convert_phrase_to_tokens(unidecode(doc['doc_text']), return_index=True)
        reference_docs.append(RefDocument(token_ids, content_token_ids, content_token_indices,
                                          title=doc.get('title', ''), author=doc.get('author', ''),
                                          chunk_id=doc.get('chunk_id', -1)))

    print(f'  Chunk {t_idx}: {len(tgt_doc.tokens)} tokens, {len(reference_docs)} ref docs — starting', flush=True)

    output = find_soft_match(tgt_doc, reference_docs, min_ngram, min_content_word, threshold, sim_table)
    t_doc.update(output)
    del t_doc['retrieved_docs']

    chunk_time = _time.time() - chunk_start
    print(f'  Chunk {t_idx} done in {chunk_time:.1f}s | coverage={output["coverage"]:.4f}', flush=True)

    return t_doc


def dj_search_earth_mover(data_path, embed_table_path, output_file, min_ngram, min_content_word, threshold,
                          n_docs, num_workers=4, subset=100):
    import time as _time
    import multiprocessing as mp

    data = [json.loads(l) for l in open(data_path, 'r').readlines()][:subset]
    print(f'Loaded {len(data)} target chunks from {data_path}')

    # Ensure .npy mmap file exists
    npy_path = embed_table_path
    if embed_table_path.endswith('.pkl'):
        npy_path = embed_table_path.replace('.pkl', '.npy')
    if not os.path.exists(npy_path):
        print(f'Converting pickle to numpy mmap: {embed_table_path} -> {npy_path}')
        load_start = _time.time()
        with open(embed_table_path, 'rb') as f:
            sim_table = pickle.load(f)
        np.save(npy_path, sim_table)
        del sim_table
        print(f'Converted in {_time.time() - load_start:.1f}s')

    # Verify it loads
    print(f'Loading embedding table (mmap) from {npy_path}...')
    load_start = _time.time()
    sim_table = np.load(npy_path, mmap_mode='r')
    print(f'Embedding table loaded in {_time.time() - load_start:.1f}s (shape: {sim_table.shape}, mmap=True)')
    del sim_table  # Workers will each mmap it independently

    outputs = []
    if os.path.isfile(output_file):
        outputs = json.load(open(output_file, 'r'))
        data = data[len(outputs):]
        print(f'Resuming from {len(outputs)} previously processed chunks')

    print(f'Processing {len(data)} chunks with {num_workers} workers, n_docs={n_docs}, min_ngram={min_ngram}, threshold={threshold}')
    print(f'{"="*80}', flush=True)

    # Each worker gets the npy_path and loads via mmap independently (shared physical memory)
    worker_args = [(i, t_doc, n_docs, min_ngram, min_content_word, threshold, npy_path)
                   for i, t_doc in enumerate(data)]

    total_start = _time.time()
    with mp.get_context('fork').Pool(num_workers) as pool:
        for result in pool.imap_unordered(_process_one_chunk, worker_args):
            if result is not None:
                outputs.append(result)
                # Save after each completed chunk
                with open(output_file, 'w') as f:
                    json.dump(outputs, f, indent=4)
                    f.flush()

                avg_coverage = np.average([x['coverage'] for x in outputs])
                elapsed = _time.time() - total_start
                chunks_done = len(outputs)
                chunks_remaining = len(data) - chunks_done
                eta = (elapsed / chunks_done) * chunks_remaining if chunks_done > 0 else 0
                print(f'  [{chunks_done}/{len(data)}] running avg coverage={avg_coverage:.4f} | '
                      f'elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min', flush=True)

    total_time = _time.time() - total_start
    avg_coverage = np.average([x['coverage'] for x in outputs])
    print(f'\nDone! {len(outputs)} chunks in {total_time/60:.1f} min, avg coverage={avg_coverage:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="GPT3_book",
                        help="which type of corpus to analyze")
    parser.add_argument('--data_dir', type=str,
                        default=f"data/book/filtered")
    parser.add_argument('--embed_table_path', type=str,
                        default=f'data/embed_distance/Llama-3-8B-Instruct.pkl')
    parser.add_argument('--output_dir', type=str,
                        default=f"outputs/semantic/book")

    parser.add_argument("--min_ngram", type=int, default=5,
                        help="minimum n-gram size")
    parser.add_argument("--min_content_word", type=int, default=2,
                        help="minimum number of content words for earth mover")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="threshold of similarity to be considered as match")

    parser.add_argument('--n_docs', type=int, default=10,
                        help='number of retrieved documents to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of parallel workers for chunk processing')
    parser.add_argument("--subset", type=int, default=100,
                        help="size of example subset to run search on")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.task + '.json')
    args.data = os.path.join(args.data_dir, args.task + '_filtered.json')
    dj_search_earth_mover(args.data, args.embed_table_path, args.output_file, args.min_ngram, args.min_content_word,
                          args.threshold, args.n_docs, args.num_workers, args.subset)
