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


from tester import Tester
from TransformersAPI import TransformersAPI
from ChatGptAPI import ChatGptAPI

import re
import os
import json
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def test(
  name: str,
  dataset: str,
  temperature: float,
  few_shot_examples: int = 0,
  embed_examples: bool = False,
  example_path: str = "data/V11.0-1000/train_wo_meta.json",
  model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
):
  if not re.match(r"[a-zA-Z-]+", name):
    raise ValueError("The name must contain only letters and dashes and at least one character.")
  
  if "mistral" in model_name or "llama" in model_name:
    api=TransformersAPI(
      model_name=model_name,
      lora_path=None,
      grammar_path=None,
      num_few_shot_examples=few_shot_examples,
      use_few_shot=True,
      few_shot_dataset_path=example_path,
      embed_examples=embed_examples
    )
  elif "gpt" in model_name:
    api=ChatGptAPI(
      model_name=model_name,
      lora_path=None,
      grammar_path=None,
      num_few_shot_examples=few_shot_examples,
      use_few_shot=True,
      few_shot_dataset_path=example_path,
      embed_examples=embed_examples
    )
  else:
    raise ValueError("Unsupported Model.")
  tester = Tester(
    f"{name}/test-{dataset.replace('_', '-')}-{few_shot_examples}_shot-temperature_{temperature:.1f}",
    api=api,
    dataset=dataset,
    temperature=temperature,
    extra_metadata={
      "lora": "None",
      "grammar": False,
      "comment": f"Part of a {few_shot_examples} shot example parameter sweep.",
      "embed_examples_in_prompt": embed_examples
    }
  )
  if temperature == 0:
    tester.do_tests_single(1234)
  else:
    raise NotImplementedError()
  tester.deinit()


def test_eval(
  name: str,
  dataset: str,
  *,
  label: str,
  temperature: float,
  few_shotss: list[int],
):
  if not re.match(r"[a-zA-Z-]+", name):
    raise ValueError("The sweep name must contain only letters and dashes and at least one character.")
  data = defaultdict(lambda: [])
  for few_shot in few_shotss:
    folder = f"{name}-{few_shot}/test-{dataset.replace('_', '-')}-{few_shot}_shot-temperature_{temperature:.1f}"
    summary = {}
    with open(os.path.join(Tester.RESULTS_DIR, folder, "summary.json"), "r") as file:
      summary = json.load(file)
    data["few_shot_examples"].append(few_shot)
    data["n"].append(summary["n"])
    data["mean"].append(summary["mean"])
    data["stddev"].append(summary["stddev"])
  #plt.errorbar(data["few_shot_examples"], data["mean"], yerr=data["stddev"], label=f"{name}")
  #plt.plot(data["few_shot_examples"], data["eval_loss"], label=f"{name}")
  #plt.legend()
  #plt.savefig(os.path.join(Tester.RESULTS_DIR, f"test-{name}-{few_shotss[-1]}/summary.pdf"))
  print("Writing summary for", folder)
  pd.DataFrame(data).to_csv(os.path.join(Tester.RESULTS_DIR, f"{label}_few_shot_scan.csv"))


if __name__ == "__main__":
  dataset="V11.0-1000"
  shotss = [1, 5, 10, 20, 30, 40, 50, 60, 70, 0]
  basedir = "few_shot_scan"

  if not os.path.exists(os.path.join(Tester.RESULTS_DIR, basedir)):
    os.mkdir(os.path.join(Tester.RESULTS_DIR, basedir))

  for shots in shotss:
    test(
      f"{basedir}/test-few-shot-{shots}",
      dataset=dataset,
      temperature=0,
      few_shot_examples=shots,
      model_name="mistralai/Mistral-7B-Instruct-v0.3"
    )
  test_eval(
    f"{basedir}/test-few-shot",
    dataset=dataset,
    temperature=0,
    few_shotss=shotss,
    label="mistral"
  )
  
  for shots in shotss:
    test(
      f"{basedir}/test-llama-few-shot-{shots}",
      dataset=dataset,
      temperature=0,
      few_shot_examples=shots,
      model_name="meta-llama/Llama-3.1-8B-Instruct"
    )
  test_eval(
    f"{basedir}/test-llama-few-shot",
    dataset=dataset,
    temperature=0,
    few_shotss=shotss,
    label="llama"
  )

  for shots in shotss:
    test(
      f"{basedir}/test-gpt-few-shot-{shots}",
      dataset=dataset,
      temperature=0,
      few_shot_examples=shots,
      model_name="gpt-4o-2024-08-06"
    )
  test_eval(
    f"{basedir}/test-gpt-few-shot",
    dataset=dataset,
    temperature=0,
    few_shotss=shotss,
    label="gpt"
  )

  # Embedded examples
  basedir = "few_shot_scan_embedded"

  for shots in shotss:
    test(
      f"{basedir}/test-few-shot-{shots}",
      dataset=dataset,
      temperature=0.0,
      few_shot_examples=shots,
      embed_examples=True,
      example_path="data/V11.0-1000/examples_wo_meta.json"
    )
  test_eval(
    f"{basedir}/test-few-shot",
    dataset=dataset,
    temperature=0,
    few_shotss=shotss,
    label="mistral-embedded"
  )
   
  for shots in shotss:
    test(
      f"{basedir}/test-llama-few-shot-{shots}",
      dataset=dataset,
      temperature=0.0,
      few_shot_examples=shots,
      embed_examples=True,
      model_name="meta-llama/Llama-3.1-8B-Instruct",
      example_path="data/V11.0-1000/examples_wo_meta.json"
    )
  test_eval(
    f"{basedir}/test-llama-few-shot",
    dataset=dataset,
    temperature=0,
    few_shotss=shotss,
    label="llama-embedded"
  )