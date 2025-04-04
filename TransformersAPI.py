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

# change model storage location
import os
os.environ["HF_HOME"] = "./"
os.environ["HF_HUB_CACHE"] = "./models/"

from API import Api, SYSTEM_PROMPT

import copy
import time
import json
import gc

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, set_seed, TrainerCallback, EarlyStoppingCallback
# "transformer reinforcement learning" (TRL)
# "supervised fine-tuning" (SFT)
from trl import SFTTrainer, SFTConfig
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
# "parameter-efficient fine-tuning" (PEFT)
from peft import PeftModel, LoraConfig
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
import torch


class TransformersAPI(Api):
  """
  Interface with a local model using the huggingface transformers API. 
  """

  def __init__(
    self,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    device: str = "cuda",
    grammar_path: str = None,
    lora_path: str = None,
    few_shot_dataset_path: str = "data/V11.0-1000/train_wo_meta.json",
    use_few_shot: bool = True,
    use_sys_prompt: bool = True,
    num_few_shot_examples: int = 10,
    embed_examples: bool = False
  ):
    super().__init__(
      model_name,
      device,
      grammar_path,
      lora_path,
      few_shot_dataset_path,
      use_few_shot,
      use_sys_prompt,
      num_few_shot_examples,
      embed_examples
    )
    # model and tokenizer
    set_seed(4242)
    self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
    self._model.to(device)
    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    # to silence annoying warning in inference
    if self._tokenizer.pad_token is None:
      self._tokenizer.pad_token = self._tokenizer.eos_token
    # to silence warning when training LoRAs 
    self._tokenizer.padding_side = 'right'
    # lora
    self.lora_path = lora_path
    if self.lora_path is not None:
      self.apply_lora(self.lora_path)
    # few-shot
    # TODO: make max_cache_len a variable and prevent longer output with max_new_tokens later
    self._few_shot_cache = StaticCache(config=self._model.config, batch_size=1, max_cache_len=16384, device=device, dtype=torch.float16)
    if self.use_few_shot or self.use_sys_prompt:
      self._cache_few_shot_prompt()
    # grammar
    self._grammar_path = grammar_path
    self._grammar_str = self._load_grammar()

  def destroy(self):
    del self._model
    del self._tokenizer
    gc.collect()
    torch.cuda.empty_cache()

  def _cache_few_shot_prompt(self):
    """Cache the few shot conversation so that the model does not need to process for every new query."""
    start_time = time.time()
    input_tokens = self._tokenizer.apply_chat_template(self._few_shot_prompt, return_tensors="pt").to(self._device)
    with torch.no_grad():
      self._few_shot_cache = self._model(input_tokens, use_cache=True, past_key_values=self._few_shot_cache).past_key_values
    print(f"Caching the few shot examples took {time.time() - start_time}[s]")

  def _load_grammar(self):
    """Load grammar from .gbnf file."""
    if self._grammar_path is None:
      return None
    if not os.path.exists(self._grammar_path):
      raise FileNotFoundError(f"Cannot find grammar at {self._grammar_path}.")
    with open(self._grammar_path, "r") as file:
      grammar_str = file.read()
    return grammar_str

  def _prepare_dataset(self, name: str, split: str, train_with_sys_promt: bool):
    """Apply the correct chat template and tokenize training data."""
    data = []
    with open(name, "r") as file:
      data = json.load(file)
    if split == "test":
      data = data["samples"]
    samples = []
    print("Preparing data...")
    for sample in data:
      s = []
      if train_with_sys_promt:
        s.append({"role": "system", "content": SYSTEM_PROMPT})
      for subsample in sample:
        s.extend([
          {"role": "user", "content": subsample["instruction"]},
          {"role": "assistant", "content": subsample["output"]},
        ])
      samples.append(s)
    dataset = Dataset.from_dict({"messages": samples}, split=split)
    return dataset
  
  def chat(self, messages, prompt, seed: int, temperature: float = 1.0, max_new_tokens: int = 300):
    set_seed(seed)
    messages.append({"role": "user", "content": prompt})
    #import pprint
    #pprint.pprint(self._few_shot_prompt)
    #exit()
    input_tokens = self._tokenizer.apply_chat_template(self._few_shot_prompt + messages, return_tensors="pt").to(self._device)
    #print("Tokens:", len(input_tokens[0]))
    #exit()
    # use cached system messages
    past_key_values = copy.deepcopy(self._few_shot_cache)
    if self._grammar_str is not None:
      grammar_processor = [GrammarConstrainedLogitsProcessor(IncrementalGrammarConstraint(self._grammar_str, "root", self._tokenizer))]
    else:
      grammar_processor = []
    generated_ids = self._model.generate(
      input_tokens,
      # cache
      past_key_values=past_key_values,
      use_cache=True,
      # grammar
      logits_processor=grammar_processor,
      do_sample=True if temperature > 0 else False,
      # to silence warning...
      pad_token_id=self._tokenizer.pad_token_id,
      # generation parameters
      max_new_tokens=max_new_tokens,
      temperature=temperature,
    )
    decoded = self._tokenizer.batch_decode(generated_ids)
    if "mistral" in self.model_name:
      delim = "[/INST]"
      reaction_model = decoded[0][decoded[0].rfind(delim) + len(delim) - 1:]
      reaction_model = reaction_model[:reaction_model.rfind("</s>")]
    elif "llama" in self.model_name:
      delim = "<|start_header_id|>assistant<|end_header_id|>"
      reaction_model = decoded[0][decoded[0].rfind(delim) + len(delim) - 1:]
      reaction_model = reaction_model[:reaction_model.rfind("<|eot_id|>")]
    else:
      raise ValueError("Unsupported model")
    messages.append({"role": "assistant", "content": reaction_model[1:]})
    return reaction_model[1:], messages

  def apply_lora(self, lora_path: str):
    self._model = PeftModel.from_pretrained(self._model, os.path.join(self.LORA_DIR, lora_path))

  def train_lora(
    self,
    name: str,
    dataset: str,
    *,
    alpha: int = 384,
    rank: int = 384,
    gradient_accumulation=80,
    batch_size=10,
    lora_dropout: float = 0.1,
    lr: float = 1.0e-4,
    seed: int = 42,
    train_with_sys_prompt: bool = False,
    resume_from_checkpoint: bool = False
  ):
    print("Preparing training data...")
    training_data = self._prepare_dataset(f"data/{dataset}/train_wo_meta.json", "train", train_with_sys_prompt)
    #test_data = self._prepare_dataset(f"data/{dataset}/test_wo_meta.json", "test", train_with_sys_prompt)
    eval_data = self._prepare_dataset(f"data/{dataset}/eval_wo_meta.json", "eval", train_with_sys_prompt)
    peft_config = LoraConfig(
      r=rank,
      lora_alpha=alpha,
      lora_dropout=lora_dropout,
      bias="none",
      #init_lora_weights="pissa",  # <-- this is supposed to be very beneficial but takes veeery long...
      task_type="CAUSAL_LM",
      #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    )
    print("Adding adapter to model...")
    self._model.add_adapter(peft_config)
    # based on https://huggingface.co/docs/trl/sft_trainer#supervised-fine-tuning-trainer
    # and https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
    sft_config = SFTConfig(
      output_dir=os.path.join(self.LORA_DIR, name, "train_output"),
      overwrite_output_dir=True,
      # pack together multiple examples to reduce padding
      #packing=False,
      #eval_packing=False,
      ## max number of tokens in a "pack"
      #max_seq_length=624,
      # use special half-precision floating point values
      bf16=True,
      # run evaluation every epoch
      do_eval=True,
      eval_strategy="epoch",
      eval_steps=1,
      num_train_epochs=100, # will likely be much smaller due to early stopping
      # over how many steps to accumulate the gradient (to avoid storing all of them)
      # gradient_accum * batch_size = effective_batch_size
      gradient_accumulation_steps=gradient_accumulation, 
      per_device_eval_batch_size=1,
      per_device_train_batch_size=batch_size,
      # trade some compute for more memory when determining gradient
      gradient_checkpointing=True,
      #gradient_checkpointing_kwargs={"use_reentrant": False},
      learning_rate=lr,
      log_level="info",
      logging_steps=1,
      logging_strategy="epoch",
      lr_scheduler_type="cosine",
      max_steps=-1,
      report_to="tensorboard",
      # a checkpoint will be saved after every epoch (note: the checkpoint number will be the *step*!)
      save_strategy="epoch",
      save_steps=1,
      save_total_limit=None,
      save_safetensors=True,
      restore_callback_states_from_checkpoint=True,
      # the best of the checkpoints will be returned at the end
      load_best_model_at_end=True,
      metric_for_best_model="eval_loss",
      seed=seed
    )
    trainer = SFTTrainer(
      model=self._model,
      tokenizer=self._tokenizer,
      args=sft_config,
      peft_config=peft_config,
      train_dataset=training_data,
      eval_dataset=eval_data,
      callbacks=[EarlyStoppingCallback(2)],
    )
    print("Starting with training now...")
    # if the LoRA has the same name, then resume training
    results = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Saving model and results...")
    self._model.save_pretrained(os.path.join(self.LORA_DIR, name))
    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()
    print("Done.")

  def get_eval_performance(self, dataset, train_with_sys_prompt=False):   
    eval_data = self._prepare_dataset(f"data/{dataset}/eval_wo_meta.json", "eval", train_with_sys_prompt)
    args = SFTConfig(output_dir="./results", evaluation_strategy="epoch")
    trainer = SFTTrainer(
        model=self._model,
        tokenizer=self._tokenizer,
        args=args,
        train_dataset=eval_data,
        eval_dataset=eval_data
    )
    results = trainer.evaluate()
    return results


if __name__ == "__main__":
  prompt = "The predator species leaves the system at rate 10.0."

  api = TransformersAPI(
    #grammar_path="reactions.gbnf",
    num_few_shot_examples=5
  )
  out = api.get_reaction_system_for_prompt(prompt, seed=42)
  print(out)
