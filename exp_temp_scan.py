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

"""
Contains definitions of experiments to run.
Recommended command to run with:

(unbuffer time python tests.py 2>&1) | tee logs/<log_file_name>.txt
"""

from tester import Tester
from convergence_test import do_enough_replications
from TransformersAPI import TransformersAPI
from ChatGptAPI import ChatGptAPI

import re
import os
import json
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

def temp_sweep(
  name: str,
  dataset: str,
  temperatures: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  *,
  lora_path: str=None,
  grammar_path: str=None,
  num_few_shot_examples: int=0,
  embed_examples: bool = False,
  example_path: str = "data/V11.0-1000/train_wo_meta.json",
  model_name="mistralai/Mistral-7B-Instruct-v0.3"
):
  if not re.match(r"[a-zA-Z-]+", name):
    raise ValueError("The sweep name must contain only letters and dashes and at least one character.")
  grammar_str = "wo_grammar" if grammar_path is None else "w_grammar"
  lora_str = "wo_lora" if lora_path is None else "w_lora"
  fewshot_str = "few_shot" if num_few_shot_examples > 0 else "zero-shot"
  for temperature in temperatures:
    if "mistral" in model_name or "llama" in model_name:
      api=TransformersAPI(
        model_name=model_name,
        lora_path=lora_path,
        grammar_path=grammar_path,
        few_shot_dataset_path=example_path,
        num_few_shot_examples=num_few_shot_examples,
        use_few_shot=(num_few_shot_examples != 0),
        use_sys_prompt=True,
        embed_examples=embed_examples
      )
    elif "gpt" in model_name:
      api=ChatGptAPI(
        model_name=model_name,
        lora_path=lora_path,
        grammar_path=grammar_path,
        few_shot_dataset_path=example_path,
        num_few_shot_examples=num_few_shot_examples,
        use_few_shot=(num_few_shot_examples != 0),
        use_sys_prompt=True,
        embed_examples=embed_examples
      )
    else:
      raise ValueError("Unsupported Model.")
    tester = Tester(
      f"{name}/test-V11.0_1000-{lora_str}-{fewshot_str}-{grammar_str}-temperature_{temperature:.1f}",
      api=api,
      dataset=dataset,
      temperature=temperature,
      extra_metadata={
        "lora": "" if lora_path is None else lora_path,
        "fewshot_examples": num_few_shot_examples,
        "grammar": grammar_path is not None,
        "examples_embedded": embed_examples,
        "comment": "Part of a temperature parameter sweep."
      }
    )
    if temperature > 0:
      tester.do_tests(1234)
    else:
      tester.do_tests_single(1234)
    tester.deinit()


def temp_sweep_eval(
  name: str,
  dataset: str,
  temperatures: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  *,
  lora_path: str=None,
  grammar_path: str=None,
  num_few_shot_examples=0,
  model_name: str = "",
  embed_examples: bool = False,
  example_path: str = ""
):
  if not re.match(r"[a-zA-Z-]+", name):
    raise ValueError("The sweep name must contain only letters and dashes and at least one character.")
  grammar_str = "wo_grammar" if grammar_path is None else "w_grammar"
  lora_str = "wo_lora" if lora_path is None else "w_lora"
  fewshot_str = "few_shot" if num_few_shot_examples > 0 else "zero-shot"
  data = defaultdict(lambda: [])
  for temperature in temperatures:
    folder=f"{name}/test-V11.0_1000-{lora_str}-{fewshot_str}-{grammar_str}-temperature_{temperature:.1f}"
    summary = {}
    with open(os.path.join(Tester.RESULTS_DIR, folder, "summary.json"), "r") as file:
      summary = json.load(file)
    data["temperature"].append(temperature)
    data["n"].append(summary["n"])
    data["mean"].append(summary["mean"])
    data["stddev"].append(summary["stddev"])
  #plt.errorbar(data["temperature"], data["mean"], yerr=data["stddev"], label=f"{lora_str}/{grammar_str}")
  #plt.legend()
  #plt.savefig(os.path.join(Tester.RESULTS_DIR, f"{name}-summary-{grammar_str}.pdf"))
  print("Writing summary for", folder)
  pd.DataFrame(data).to_csv(os.path.join(Tester.RESULTS_DIR, f"{name}/summary-{grammar_str}.csv"))


if __name__ == "__main__":
  dataset="V11.0-1000"
  best_lora = "best_lora"
  best_shots = 30
  basedir = "temp_scan"

  if not os.path.exists(os.path.join(Tester.RESULTS_DIR, basedir)):
    os.mkdir(os.path.join(Tester.RESULTS_DIR, basedir))

  ## temperature sweep for one shot w/o grammar
  params = dict(
    name=f"{basedir}/one-shot",
    dataset=dataset,
    lora_path=None,
    grammar_path=None,
    num_few_shot_examples=1
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  ## temperature sweep for one shot w/ grammar
  params = dict(
    name=f"{basedir}/one-shot",
    dataset=dataset,
    lora_path=None,
    grammar_path="reactions.gbnf",
    num_few_shot_examples=1
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  ## temperature sweep for few shot w/o grammar
  params = dict(
    name=f"{basedir}/few-shot",
    dataset=dataset,
    lora_path=None,
    grammar_path=None,
    num_few_shot_examples=best_shots
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  ## temperature sweep for few shot w/ grammar
  params = dict(
    name=f"{basedir}/few-shot",
    dataset=dataset,
    lora_path=None,
    grammar_path="reactions.gbnf",
    num_few_shot_examples=best_shots
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  # temperature sweep for v11.0-1000 lora w/o grammar zero-shot
  params = dict(
    name=f"{basedir}/lora-v11.0-1000-zero-shot",
    dataset=dataset,
    lora_path=best_lora,
    grammar_path=None,
    num_few_shot_examples=0
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  # temperature sweep for v11.0-1000 lora w/ grammar zero-shot
  params = dict(
    name=f"{basedir}/lora-v11.0-1000-zero-shot",
    dataset=dataset,
    lora_path=best_lora,
    grammar_path="reactions.gbnf",
    num_few_shot_examples=0
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)


  # temperature sweep for v11.0-1000 lora w/o grammar few-shot
  params = dict(
    name=f"{basedir}/lora-v11.0-1000-few-shot",
    dataset=dataset,
    lora_path=best_lora,
    grammar_path=None,
    num_few_shot_examples=best_shots
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)
  
  # temperature sweep for v11.0-1000 lora w/ grammar few-shot
  params = dict(
    name=f"{basedir}/lora-v11.0-1000-few-shot",
    dataset=dataset,
    lora_path=best_lora,
    grammar_path="reactions.gbnf",
    num_few_shot_examples=best_shots
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)


  # llama
  best_shots = 20
  # temperature sweep for llama w/o grammar few-shot
  params = dict(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    name=f"{basedir}/llama",
    dataset=dataset,
    lora_path=None,
    grammar_path=None,
    num_few_shot_examples=best_shots,
    embed_examples=False
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  # gpt-4o
  best_shots = 30
  # temperature sweep for llama w/o grammar few-shot
  params = dict(
    model_name="gpt-4o-2024-08-06",
    name=f"{basedir}/gpt",
    dataset=dataset,
    lora_path=None,
    grammar_path=None,
    num_few_shot_examples=best_shots,
    embed_examples=False
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)


  # appendix
  best_shots = 40

  # temperature sweep for v11.0-1000 lora w/o grammar few-shot
  params = dict(
    name=f"{basedir}/lora-v11.0-1000-few-shot-embedded",
    dataset=dataset,
    lora_path=best_lora,
    grammar_path=None,
    num_few_shot_examples=best_shots,
    embed_examples=True,
    example_path="data/V11.0-1000/examples_wo_meta.json"
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)

  # temperature sweep for v11.0-1000 lora w/ grammar few-shot
  params = dict(
    name=f"{basedir}/lora-v11.0-1000-few-shot-embedded",
    dataset=dataset,
    lora_path=best_lora,
    grammar_path="reactions.gbnf",
    num_few_shot_examples=best_shots,
    embed_examples=True,
    example_path="data/V11.0-1000/examples_wo_meta.json"
  )
  temp_sweep(**params)
  temp_sweep_eval(**params)
