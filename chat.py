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
import pprint
import os

if __name__ == "__main__":
  import readline
  api = TransformersAPI(
    #grammar_path="reactions.gbnf",
    #lora_path="",
    num_few_shot_examples=10,
    use_few_shot=True,
    use_sys_prompt=True,
    embed_examples=False
  )
  #api = ChatGptAPI(
  #  num_few_shot_examples=10
  #)
  messages = []
  while True:
    user = input("User:\n")
    if user == "!save":
      if not os.path.exists("chats"):
        os.mkdir("chats")
      with open(f"chats/chat-{datetime.datetime.now().replace(microsecond=0).isoformat()}", "w") as file:
        pprint.pprint(messages, file)
      exit()
    assistant, messages = api.chat(messages, user, 1234, temperature=0.0)
    print("Assistant:\n", assistant)
    print(messages)
