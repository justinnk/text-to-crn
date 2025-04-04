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

from API import Api
#from TransformersAPI import TransformersAPI
from convergence_test import do_enough_replications
from logger import log

import json, os, re, time
from collections import defaultdict

import pandas as pd
import numpy as np


class Tester:
    
    class STATUS:
      OK = "OK"
      CRITICAL_FAIL = "CRITICAL FAIL"
      FAIL_SEMANTICS = "FAIL (semantics)"
      FAIL_NUM = "FAIL (num)"
      FAIL_ENTS = "FAIL (ents)"
      FAIL_BOTH = "FAIL (num & ents)"

    RESULTS_DIR: str = "results"

    def __init__(
      self,
      test_name: str,
      dataset: str,
      api: Api,
      temperature: float = 1.0,
      max_new_tokens: int = 300,
      extra_metadata: dict = {}
    ):
      self.test_name = test_name
      self.dataset = dataset
      self.data = []
      self._load_data()
      self.temperature = temperature
      self.max_new_tokens = max_new_tokens
      self.api = api
      self.extra_metadata = extra_metadata
      self.current_seed = 0
      self.current_iteration = -1
      # only for documentation of data frame for results
      _test_result_data = pd.DataFrame(columns={
        # Prompt
        "instruction": pd.Series(dtype="str"),
        # LLM output
        "output": pd.Series(dtype="str"),
        # whether the list of species exactly matches the label
        "correct_entities": pd.Series(dtype="bool"),
        # whether the list of species semantically matches the label
        "weak_correct_entities": pd.Series(dtype="bool"),
        # whether the number of reactions exactly matches the label
        "correct_num_reactions": pd.Series(dtype="bool"),
        # whether the rates exactly matches the label
        "correct_rates": pd.Series(dtype="bool"),
        # whether the list of entities and number of reactions match the label
        "correct_creative": pd.Series(dtype="bool"),
        # whether the whole system, including rates, exactly matches the label
        "correct_descriptive": pd.Series(dtype="bool"),
        # whether the whole system, including rates, semantically matches the label
        "weak_correct_descriptive": pd.Series(dtype="bool"),
      })
      # for efficiency, we append to lists first and only once in a while write the dataframe
      self.test_result_data_lists = {col: [] for col in _test_result_data.columns}
      self.test_result_metadata = {
        "seed": self.current_seed,
        "temperature": self.temperature,
        "max_new_tokens": self.max_new_tokens,
        "current_iteration": self.current_iteration,
        "result_code_counts": {},
        "extra_metadata": self.extra_metadata
      }
      self.result_code_counts = defaultdict(lambda: 0)
      if not os.path.exists(os.path.join(self.RESULTS_DIR, self.test_name)):
        os.makedirs(os.path.join(self.RESULTS_DIR, self.test_name))
      self.log_file = open(os.path.join(self.RESULTS_DIR, self.test_name, "log.txt"), "a")

    def deinit(self):
      self.api.destroy()
      self.log_file.close()

    def get_subfolder_name(self) -> str:
      return os.path.join(self.RESULTS_DIR, os.path.normpath(self.test_name), f"seed-{self.current_seed}-temperature-{self.temperature}") 

    def save_checkpoint(self):
      data = pd.DataFrame(self.test_result_data_lists)
      if not os.path.exists(self.get_subfolder_name()):
        os.makedirs(self.get_subfolder_name())
      data.to_csv(os.path.join(self.get_subfolder_name(), "results.csv"), index=False)
      self._update_metadata()
      with open(os.path.join(self.get_subfolder_name(), "metadata.json"), "w") as file:
        json.dump(self.test_result_metadata, file)

    def load_checkpoint(self):
      data = pd.read_csv(os.path.join(self.get_subfolder_name(), "results.csv"))
      self.test_result_data_lists = data.to_dict(orient="list")
      with open(os.path.join(self.get_subfolder_name(), "metadata.json"), "r") as file:
        self.test_result_metadata = json.load(file)
      self.current_iteration = self.test_result_metadata["current_iteration"]
      self.temperature = self.test_result_metadata["temperature"]
      self.max_new_tokens = self.test_result_metadata["max_new_tokens"]
      self.current_seed = self.test_result_metadata["seed"]
      self.result_code_counts = defaultdict(lambda: 0, **self.test_result_metadata["result_code_counts"])
      self.extra_metadata = self.test_result_metadata["extra_metadata"]

    def _update_metadata(self):
      self.test_result_metadata = {
        "seed": self.current_seed,
        "temperature": self.temperature,
        "max_new_tokens": self.max_new_tokens,
        "current_iteration": self.current_iteration,
        "result_code_counts": self.result_code_counts,
        "extra_metadata": self.extra_metadata
      }

    def _load_data(self):
      with open(os.path.join("data", self.dataset, f"test_wo_meta.json")) as file:
         self.data = json.load(file)

    def _log(self, *msgs):
      log("Tester", self.log_file, *msgs)

    def _parse_formal_model(self, response: str) -> list[dict]:
      model = dict(reactions=[], entities=[])
      reactions = response.split(";\n")[:-1]
      for reaction in reactions:
        components = re.split("->|@", reaction)
        if len(components) != 3:
          raise Exception(f"'{reaction}' does not conform to grammar. Parsing yielded {components}.")
        left, right, rate = components
        left_w_num = re.findall(r"[2-9]?[a-zA-Z_][a-zA-Z0-9_-]*", left)
        left_wo_num = re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]*", left)
        right_w_num = re.findall(r"[2-9]?[a-zA-Z_][a-zA-Z0-9_-]*", right)
        right_wo_num = re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]*", right)
        rate = re.findall(r"(?:k\d+)|(?:\d+(?:.\d+)?)", rate)
        if len(rate) != 1:
          raise Exception(f"Rate expression could not be translated, regex returned '{rate}'.")
        model["reactions"].append({
          "left": left_w_num,
          "right": right_w_num,
          "rate": rate[0]
        }) 
        model["entities"].extend(left_wo_num)
        model["entities"].extend(right_wo_num)
      model["entities"] = list(set(model["entities"]))
      return model
    
    def _check_entities_correct(self, predictions: list[str], targets: list[str]) -> bool:
      def transform(species_name):
        species_name = species_name.lower()
        if "_" not in species_name: return species_name
        digit_strs = [s for s in species_name if s.isdigit()]
        species_name = "".join(s for s in species_name if not s.isdigit())
        parts = species_name.split("_")
        parts.extend(digit_strs)
        return parts
      targets = [set(transform(name)) for name in targets]
      predictions = [set(transform(name)) for name in predictions]
      if len(targets) != len(predictions):
        return False
      for target_name in targets:
        for predicted_name in predictions:
          if predicted_name == target_name:
            break
        else:
          return False
      return True
    
    def _check_rates_correct(self, predictions: list[str], targets: list[str]) -> bool:
      for rate_target in targets:
        if rate_target not in predictions:
          return False
      return True

    def _check_reaction_equals(self, reaction1: dict, reaction2: dict) -> bool:
        return (reaction1["rate"] == reaction2["rate"] or\
                reaction1["rate"].startswith("k") and reaction2["rate"].startswith("k")) and\
                self._check_entities_correct(reaction1["left"], reaction2["left"]) and\
                self._check_entities_correct(reaction1["right"], reaction2["right"])

    def _check_semantics_correct(self, predicted_reactions: list[dict], reaction_targets: list[str]) -> bool:
        for reaction_target in reaction_targets:
          for predicted_reaction in predicted_reactions:
           if self._check_reaction_equals(reaction_target, predicted_reaction):
            break
          else:
           return False
        return True
    
    def _do_test(self, sample: dict, seed: int) -> str:
      """Test a single sample."""
      output = self.api.get_reaction_system_for_prompt(
        sample["instruction"],
        seed=seed,
        temperature=self.temperature,
        max_new_tokens=self.max_new_tokens
      )
      self._log("Instruction:\n", sample["instruction"])
      self._log("Model output:\n", output)
      self._log("Expected Target:\n", sample["output"])

      fail_state = {
          "instruction": sample["instruction"],
          "output": output,
          "correct_entities": False,
          "weak_correct_entities": False,
          "correct_num_reactions": False,
          "correct_rates": False,
          "correct_creative": False,
          "correct_descriptive": False,
          "weak_correct_descriptive": False,
        }

      # when there is no codeblock marker, something went very wrong
      # and we can't analyze the output
      if output.count("```") != 2:
        self._log("Reason for failure: no. code block markers != 2")
        for key, val in fail_state.items():
          self.test_result_data_lists[key].append(val)
        return self.STATUS.CRITICAL_FAIL
      
      # remove code block markers
      output = output.split("```")[1][1:]

      try:
        model = self._parse_formal_model(output)
      except Exception as ex:
        self._log("Reason for failure: Output does not conform to grammar.", "Exception was:", ex)
        for key, val in fail_state.items():
          self.test_result_data_lists[key].append(val)
        return self.STATUS.CRITICAL_FAIL

      num_reactions_correct = len(model["reactions"]) == sample["num_reactions"]
      ents_correct = self._check_entities_correct(model["entities"], sample["entities"])
      rates_correct = self._check_rates_correct([x["rate"] for x in model["reactions"]], [x["rate"] for x in sample["reactions"]])
      desc_correct = self._check_semantics_correct(model["reactions"], sample["reactions"])

      for key, val in {
        "instruction": sample["instruction"],
        "output": output,
        "correct_entities": ents_correct,
        # TODO: implement
        "weak_correct_entities": ents_correct,
        "correct_num_reactions": num_reactions_correct,
        "correct_rates": rates_correct,
        "correct_creative": ents_correct and num_reactions_correct,
        "correct_descriptive": ents_correct and num_reactions_correct and desc_correct,
        # TODO: implement
        "weak_correct_descriptive": ents_correct and num_reactions_correct and desc_correct,
      }.items():
        self.test_result_data_lists[key].append(val)
      
      if ents_correct and num_reactions_correct and desc_correct:
        return self.STATUS.OK
      if ents_correct and num_reactions_correct:
        return self.STATUS.FAIL_SEMANTICS
      if not ents_correct and not num_reactions_correct:
        return self.STATUS.FAIL_BOTH
      if not ents_correct:
        return self.STATUS.FAIL_ENTS
      if not num_reactions_correct:
        return self.STATUS.FAIL_NUM
    
    def _save_summary_csv(self):
      nsamples = len(self.data["samples"])
      results = self.test_result_data_lists
      samples = [sample[0] for sample in self.data["samples"]]
      summary = {
        "correct_entities": np.mean([results["correct_entities"][idx] for idx in range(nsamples)]),
        # TODO: actually implement. same as correct for now...
        "weak_correct_entities": np.mean([results["correct_entities"][idx] for idx in range(nsamples)]), 
        "correct_num_reactions": np.mean([results["correct_num_reactions"][idx] for idx in range(nsamples)]),
        "correct_rates": np.mean([results["correct_rates"][idx] for idx in range(nsamples)]),
        "correct_creative": np.mean([results["correct_creative"][idx] for idx in range(nsamples)]),
        "correct_descriptive": np.mean([results["correct_descriptive"][idx] for idx in range(nsamples)]),
        # TODO: as above
        "weak_correct_descriptive": np.mean([results["correct_descriptive"][idx] for idx in range(nsamples)]),
        "correct_bio": np.mean([results["correct_descriptive"][idx] for idx in range(nsamples) if samples[idx]["domain"] == "bio"]),
        "correct_eco": np.mean([results["correct_descriptive"][idx] for idx in range(nsamples) if samples[idx]["domain"] == "eco"]),
        "correct_epi": np.mean([results["correct_descriptive"][idx] for idx in range(nsamples) if samples[idx]["domain"] == "epi"]),
        "critical": self.result_code_counts[self.STATUS.CRITICAL_FAIL] if self.STATUS.CRITICAL_FAIL in self.result_code_counts else 0.0,
      }
      with open(os.path.join(self.get_subfolder_name(), "summary.json"), "w") as file:
        json.dump(summary, file)

    def do_tests_run(self, seed: int) -> float:
      """Run all tests and return resulting accuracy."""
      # reset
      self.__init__(self.test_name, self.dataset, self.api, self.temperature, self.max_new_tokens, self.extra_metadata)
      self.current_seed = seed
      # if there is a previous run with the same hyperparameters, load state and continue
      if os.path.exists(self.get_subfolder_name()):
        self._log(f"Found existing experiment with the name {self.test_name}")
        self._log(f"Reusing results for seed {self.current_seed} and temperature {self.temperature}")
        self.load_checkpoint()
      # start with the test...
      start_time = time.perf_counter()
      self._log(f"Starting at {time.ctime()}")
      self._log(f"Running tests for {self.test_name} with seed {self.current_seed} and temperature {self.temperature}")
      # start with the sample next to the last processed one from a possible previous interrupted run
      offset = self.current_iteration + 1
      self._log("Continuing existing run with offset", offset)
      for i, sample in enumerate(self.data["samples"][offset:]):
        self.current_iteration = offset+i
        self._log("Testing sample", self.current_iteration+1, "/", len(self.data["samples"]))
        result = self._do_test(sample[0], self.current_seed + i)
        self.result_code_counts[result] += 1
        self._log(f"=> Result: {result}")
        # save results every 10 samples to be safe
        if self.current_iteration % 2 == 0:
          self._log("Saving checkpoint after iteration", self.current_iteration + 1)
          self.save_checkpoint()
      # we are done :)
      self._log("Finished run. Took", time.perf_counter() - start_time, "[s]. Summary:\n", self.result_code_counts)
      self.save_checkpoint()
      self._save_summary_csv()
      # return percent of correct descriptive outputs
      return len(
        [x for x in self.test_result_data_lists["correct_descriptive"] if x]) /\
        len(self.test_result_data_lists["correct_descriptive"]
      )
    
    def do_tests(self, seed: int):
      """Perform the tests until their mean result converges."""
      n, mean, stddev, dns = do_enough_replications(seed, self.do_tests_run, self.log_file)
      # write summary of all reps
      with open(os.path.join(self.RESULTS_DIR, self.test_name, "summary.json"), "w") as file:
        json.dump({
          "n": n,
          "mean": mean,
          "stddev": stddev,
          "dns": dns
        }, file)

    def do_tests_single(self, seed: int):
      """Perform a single test (e.g., when the model is deterministic)."""
      n = 1
      mean = self.do_tests_run(seed)
      stddev = 0.0
      dns = [0.0]
      # write summary of all reps
      with open(os.path.join(self.RESULTS_DIR, self.test_name, "summary.json"), "w") as file:
        json.dump({
          "n": n,
          "mean": mean,
          "stddev": stddev,
          "dns": dns
        }, file)


if __name__ == "__main__":
  # USAGE: to write the log to a file and stdout, use the following pattern:
  # (unbuffer time python tester.py 2>&1) | tee log_<name>_<desc>_<date>.txt
  # unbuffer to immediately see the output on stdout, time for time
  # measurement of the command, tee to write to file and stdout.
  # To append to a previously interrupted run, use `tee -a ...`
  lora_path = "test-lora/train_output/checkpoint-17"
  tester = Tester(
    "test-V9_1000-w_grammar-w_lora",
    api=None,
    #api=TransformersAPI(
    #  lora_path=lora_path,
    #  grammar_path="reactions.gbnf",
    #),
    dataset="V9-1000",
    temperature=1.0,
    extra_metadata={
      "lora": lora_path,
      "grammar": True,
      "comment": "A first trial run of a trail lora."
    }
  )
  tester.do_tests(1234)
  tester.deinit()
