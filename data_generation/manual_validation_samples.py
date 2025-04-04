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


manual_validation_set = [
       [
         dict(
             instruction="The interaction of susceptible with infected individuals results "
                         "in two infected individuals at a rate of 0.02. Infected individuals "
                         "recover at a rate of 5.",
             output=     "```\n"
                         "susceptible + infected -> 2infected @ 0.02;\n"
                         "infected -> recovered @ 5;\n"
                         "```"
         ),
         dict(
             instruction="Change the rate of the first reaction in the above system to 0.5.",
             output=     "```\n"
                         "susceptible + infected -> 2infected @ 0.5;\n"
                         "infected -> recovered @ 5;\n"
                         "```"
         )
       ],
       [
         dict(
             instruction="Two healthy bear can reproduce at a rate of 7, resulting in two "
                         "healthy bear and one bear pup.",
             output=     "```\n"
                         "bear_healthy_female + bear_healthy_male -> bear_healthy_female + bear_pup + bear_healthy_male @ 7;\n"
                         "```"
         ),
         dict(
             instruction="Additionally, bear pups progress to healthy bear with a rate of 3.",
             output=     "```\n"
                         "bear_healthy_female + bear_healthy_male -> bear_healthy_female + bear_pup + bear_healthy_male @ 7;\n"
                         "bear_pup -> bear_healthy @ 3;"
                         "```"
         )
       ],
       [
         dict(
             instruction="The Enzyme binds to the Substrate, forming an EnzymeSubstrate complex at a rate of 2.1."
                         "At a rate of 3.4, the EnzymeSubstrate complex may move to an EnzymeProduct complex."
                         "Finally, the EnzymeProduct complex may dissociate into the Enzyme and Product with a rate of 1.1.",
             output=     "```\n"
                         "Enzyme + Substrate -> EnzymeSubstrate @ 2.1;\n"
                         "EnzymeSubstrate -> EnzymeProduct @ 3.4;\n"
                         "EnzymeProduct -> Enzyme + Product @ 1.1;\n"
                         "```"
         ),
         dict(
             instruction="I was mistaken. The rate of EnzymeProduct formation is 4.2.",
             output=     "```\n"
                         "Enzyme + Substrate -> EnzymeSubstrate @ 2.1;\n"
                         "EnzymeSubstrate -> EnzymeProduct @ 4.2;\n"
                         "EnzymeProduct -> Enzyme + Product @ 1.1;\n"
                         "```"
         )
       ],
    ]