"""
Copyright 2025 Justin Kreikemeyer, Miłosz Jankowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
 
  The above copyright notice and this permission notice shall be included in all copies or
  substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

from TransformersAPI import TransformersAPI
from tester import Tester

from itertools import product
from collections import defaultdict
import os
import re
import json

import pandas as pd
import numpy as np

def run_test(
  name: str,
  dataset: str,
  *,
  lora_path: str=None,
):
  if not re.match(r"[a-zA-Z-]+", name):
    raise ValueError("The sweep name must contain only letters and dashes and at least one character.")
  lora_str = "wo_lora" if lora_path is None else "w_lora"
  tester = Tester(
    f"lora_scan/test-{name}/test-V11.0_1000-{lora_str}-temperature_0",
    api=TransformersAPI(
      lora_path=lora_path,
      grammar_path=None,
      num_few_shot_examples=0,
      use_few_shot=False,
      use_sys_prompt=True
    ),
    dataset=dataset,
    temperature=0.0,
    extra_metadata={
      "lora": "" if lora_path is None else lora_path,
      "grammar": False,
      "comment": "Part of a lora scan."
    }
  )
  tester.do_tests_single(1234)
  tester.deinit()

if __name__ == "__main__":
  alphas = [8, 16]
  ranks = [4, 8, 16]
  dropouts = [0.3, 0.5, 0.7]

  # train LoRAs
  for replication in range(3):
    for idx, (alpha, rank, dropout) in enumerate(product(alphas, ranks, dropouts)):
      name = f"lora-v11.0-1000-alpha_{alpha}-rank_{rank}-do_{dropout}-rep_{replication}"
      if os.path.exists(os.path.join(TransformersAPI.LORA_DIR, name)):
        continue
      try:
        #if alpha != 8 or rank != 8 or dropout != 0.3 or replication != 1:
        #  continue
        print(f"Training LoRA with {alpha=} {rank=} {dropout=} {replication=}")
        api = TransformersAPI(use_few_shot=False, num_few_shot_examples=0)
        api.train_lora(
          name,
          "V11.0-1000",
          alpha=alpha,
          rank=rank,
          gradient_accumulation=80,
          batch_size=10,
          lora_dropout=dropout,
          seed=42+idx+replication,
          train_with_sys_prompt=True
        )
        api.destroy()
      except Exception as e:
        print("Could not train with", alpha, "and", rank, "because I got an exception:", e)

  # evaluate them
  for replication in range(3):
    for alpha, rank, dropout in product(alphas, ranks, dropouts):
      #if alpha != 8 or rank != 8 or dropout != 0.3 or replication != 1:
      #  continue
      print(f"Evaluating LoRA with {alpha=} {rank=} {dropout=} {replication=}")
      # take the third-last checkpoint as it was when we first observed overfitting/convergence during early stopping
      name = f"lora-v11.0-1000-alpha_{alpha}-rank_{rank}-do_{dropout}-rep_{replication}"
      # the third-last will automatically be in the root when using EarlyStopping
      lora_path = name
      # perform test for temp == 0.0
      run_test(
        f"lora-alpha_{alpha}-rank_{rank}-do_{dropout}-rep_{replication}",
        dataset="V11.0-1000",
        lora_path=lora_path
      )

  # load and evaluate data
  results = defaultdict(lambda: [])
  for alpha, rank, dropout in product(alphas, ranks, dropouts):
    accuracies = []
    for replication in range(3):
      name = os.path.join(
        Tester.RESULTS_DIR,
        "lora_scan",
        f"test-lora-alpha_{alpha}-rank_{rank}-do_{dropout}-rep_{replication}",
        "test-V11.0_1000-w_lora-temperature_0",
        "summary.json"
      )
      with open(name, "r") as file:
        accuracies.append(json.load(file)["mean"])
    results["alpha"].append(alpha)
    results["rank"].append(rank)
    results["scale"].append(alpha/rank)
    results["dropout"].append(dropout)
    results["mean"].append(np.mean(accuracies))
    results["stddev"].append(np.std(accuracies, ddof=1))
    pd.DataFrame(results).to_csv(os.path.join(Tester.RESULTS_DIR, "lora_scan", "summary.csv"), index=False)
    print(results)
