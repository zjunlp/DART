# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for evaluate results for all SuperGLUE tasks.
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr


def _exact_match(actuals: np.ndarray, predictions: np.ndarray, question_ids: np.ndarray):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, actuals))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)

    return em


def load_metrics(task_name):
    METRICS = {
        "cb": ["acc", "f1-macro"],
        "multirc": ["acc", "em", "f1"],
        "record": ["acc", "f1"],
        "cola": ["matt"],
        "mrpc": ["f1"],
        "qqp": ["f1"],
        "sts-b": ["pear"]
    }
    DEFAULT_METRICS = ["acc"]

    return METRICS.get(task_name, DEFAULT_METRICS)


def evaluate_results(results, metrics):
    predictions = np.argmax(results['logits'], axis=1)
    scores = {}
    for metric in metrics:
        if metric == 'acc':
            scores[metric] = accuracy_score(results['labels'], predictions)
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(
                results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = _exact_match(
                results['labels'], predictions, results['question_ids'])
        elif metric == 'matt':
            scores[metric] = matthews_corrcoef(results['labels'], predictions)
        elif metric == 'pear':
            scores[metric] = pearsonr(results['labels'], predictions)
        else:
            raise ValueError(f"Metric '{metric}' not implemented")
    results['scores'] = scores
    results['predictions'] = predictions

    return results
