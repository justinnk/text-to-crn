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
from ChatGptAPI import ChatGptAPI
import datetime
import os
import json

if __name__ == "__main__":
  import readline

  prompt = "The following describes a reaction system. Please translate to a formal description. "

  api = TransformersAPI(
    grammar_path=None,
    lora_path="best_lora",
    few_shot_dataset_path= "data/V11.0-1000/train_wo_meta.json",
    num_few_shot_examples=0,
    use_sys_prompt=True
  )

  name = input("Participant Name:\n")
  agreement_participation = input("Your name, all your answers in the questionaire and the full chat history will be recorded. Do you consent to the (potential) use and publication of all recorded material from this study?:\n")
  time = datetime.datetime.now().replace(microsecond=0).isoformat()
  if not os.path.exists("user_study_chats"):
    os.mkdir("user_study_chats")
  first_prompt = True

  messages = []
  while True:
    if first_prompt:
      user = input(f"Please describe your simulation model:\n")
      user = prompt + user
      first_prompt = False
    else:
      user = input(f"{name} says:\n")
    assistant, messages = api.chat(messages, user, 1234, temperature=0.0)
    print("Modeling Assistant:\n", assistant)
    #print(messages)
    with open(f"user_study_chats/chat-{name.replace(' ', '-')}-{time}.json", "w") as file:
      json.dump(dict(
        metadata=dict(
          participant=name,
          agreement_participation=agreement_participation,
          lora=api.lora_path,
          few_shot_dataset=api.few_shot_dataset_path,
          few_shot_examples=api.num_few_shot_examples,
          use_few_shot=api.use_few_shot),
        messages=messages
      ), file, indent=2)
