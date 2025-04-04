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
from KinModGptAPI import KinModGptAPI, OUR_SYS_PROMPT
from tester import Tester

import os
import json

if __name__ == "__main__":

  basedir = "kinmodgpt_scan"

  if not os.path.exists(os.path.join(Tester.RESULTS_DIR, basedir)):
    os.mkdir(os.path.join(Tester.RESULTS_DIR, basedir))

  api = TransformersAPI(
    grammar_path=None,
    lora_path="best_lora",
    few_shot_dataset_path= "data/V11.0-1000/train_wo_meta.json",
    num_few_shot_examples=0,
    use_sys_prompt=True,
    use_few_shot=False
  )

  tester = Tester(
    f"{basedir}/their_examples",
    api=api,
    dataset="KinModGPT",
    temperature=0,
    max_new_tokens=1000,
    extra_metadata={
      "lora": api.lora_path,
      "fewshot_examples": api.num_few_shot_examples,
      "grammar": api._grammar_path is not None,
      "examples_embedded": api.embed_examples,
      "comment": "Part of a kinmodgpt comparison using their examples."
    }
  )
  tester.do_tests_single(1234)
  tester.deinit()


  ## reproduce their results
  #api = KinModGptAPI()

  #with open("data/KinModGPT/test_wo_meta.json") as file:
  #  data = json.load(file)["samples"]

  #for sample in data[3:]:
  #  output = api.get_reaction_system_for_prompt(sample[0]["instruction"], 1234, 0.0, 2000)
  #  print(sample[0]["instruction"])
  #  print(output)

  # use with modified sys prompt to apply to our data
  api = KinModGptAPI(sys_prompt=OUR_SYS_PROMPT)
  tester = Tester(
    f"{basedir}/our_examples",
    api=api,
    dataset="V11.0-62-bio",
    temperature=0,
    extra_metadata={
      "comment": "Part of a kinmodgpt comparison using our examples."
    }
  )
  tester.do_tests_single(1234)
  tester.deinit()