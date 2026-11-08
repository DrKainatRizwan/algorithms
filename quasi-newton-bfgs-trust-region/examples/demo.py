"""Compare quasi-Newton, CG and trust-region optimisers on standard problems."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quasi_newton import (compare, extended_rosenbrock, logistic_regression, powell_singular,
                          rosenbrock, beta_function)


def show(title, make):
    print(f"\n== {title} ==")
    print(f"{'method':<13}{'ok':<4}{'iters':>6}{'f-ev':>7}{'g-ev':>7}{'H-ev':>6}{'f*':>12}{'|g|inf':>11}{'ms':>8}")
    for r in compare(make):
        print(f"{r['method']:<13}{'Y' if r['converged'] else 'N':<4}{r['iters']:>6}{r['f_evals']:>7}"
              f"{r['g_evals']:>7}{r['h_evals']:>6}{r['f']:>12.3e}{r['grad']:>11.2e}{r['time_ms']:>8.1f}")


if __name__ == "__main__":
    show("Rosenbrock (2-D)", rosenbrock)
    show("Extended Rosenbrock (n=40)", lambda: extended_rosenbrock(40))
    show("Exponential chain (n=30)", lambda: beta_function(30))
    show("Logistic regression (200x20)", logistic_regression)
