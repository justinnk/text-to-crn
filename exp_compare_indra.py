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


import json
import os

from IndraAPI import IndraAPI
from tester import Tester


if __name__ == "__main__":

  basedir = "indra"

  if not os.path.exists(os.path.join(Tester.RESULTS_DIR, basedir)):
    os.mkdir(os.path.join(Tester.RESULTS_DIR, basedir))


  api = IndraAPI()
  tester = Tester(
    f"{basedir}/bio-examples-test",
    api=api,
    dataset="V11.0-62-bio",
    temperature=0,
    extra_metadata={
      "comment": "Part of INDRA comparison."
    }
  )
  tester.do_tests_single(1234)
  tester.deinit()

