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
Based on the paper: 
Kathryn Hoad, Stewart Robinson, and Ruth Davies. 2007. Automating des output
analysis: How many replications to run. In 2007 Winter Simulation Conference.
505–512. https://doi.org/10.1109/WSC.2007.4419641
"""

from logger import log

import numpy as np
from scipy.stats import t


def _log(log_file, *msgs):
  log("ConvergenceTest", log_file, *msgs)


def do_enough_replications(seed: int, func: callable, log_file=None) -> int:
  """
  Statistical convergence test from "Automating DES output analysis: how many replications to run"

  seed: starting value for the seed
  func: seed -> percent of successful tests (function delivering one sample per call)
  """
  alpha = 0.01 # confidence interval percentage 0.01 -> 1% CI
  d_req = 0.02 # requested precision (half-width of the CI), expressed as percentage of
               # the cumulative mean
  kLimit = 2   # additional repliactions to perform to test whether we really converged
               # after first reaching d_req
  n = 3        # current number of replications (at the beginning: num samples for initial estimate)

  def convergence_criteria_met(dns: list[float], fklimit: int = 1) -> bool:
      xx = [x for x in dns[-fklimit:] if np.fabs(x) <= d_req]
      return len(xx) == fklimit
  
  def f(kLimit: int, n: int) -> int:
      return kLimit if n <= 100 else np.floor( n * kLimit / 100 )
  
  results = []
  dns = [1.]

  _log(log_file, f"Making {n} initial replications.")

  for i in range(n - 1):
    _log(log_file, f"Starting with initial replication {i+1}/{n} (seed {seed}).")
    results.append(func(seed))
    seed += 1

  while not convergence_criteria_met(dns, f(kLimit, n)):
    if i < n:
      _log(log_file, f"Starting with initial replication {i+1}/{n} (seed {seed}).")
    else:
      _log(log_file, f"Starting with replication {i+1} (seed {seed}).")
    i += 1
    results.append(func(seed))
    seed += 1
    mean = np.mean(results)
    stddev = np.std(results, ddof=1)
    _log(log_file, f"New mean is {mean} and new stddev is {stddev}.")
    stddev = stddev if stddev != 0 else 1e-20
    n = len(results)
    t_val = t.ppf(1-alpha/2, n-1, loc=mean, scale=stddev/np.sqrt(n))
    if mean == 0:
      #raise ValueError("mean cannot be zero")
      dn = 0.0
    else:
      dn = ( t_val * ( stddev / np.sqrt(n) ) ) / mean
    dns.append(dn) # type: ignore

    _log(log_file, f"---\n{n=}\n{mean=}\n{stddev=}\n{dns[-1]=}\n---")

    if np.fabs(dns[-1]) <= d_req:
      Nsol = n

  _log(log_file, "Finished after " + str(Nsol) + " replications.")
  return Nsol, mean, stddev, dns


if __name__ == "__main__":
  def foo(seed):
    np.random.seed(seed)
    return np.random.normal(1, 0.1)
  do_enough_replications(42, foo)