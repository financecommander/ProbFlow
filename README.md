cd "C:\Users\Crypt\Claude Local\PROBFLOW"

cat > README.md << 'EOP'
# ProbFlow

**A declarative Python DSL for probabilistic flows in finance & decision-making**

Write finance-native code that *describes* uncertain cash flows, portfolios, risk factors, causal relationships, and decisions — then simulate, visualize (Graphviz), and optimize them declaratively.

## ✨ v0.1.0 Features (just shipped)
- Full modular Python package (`probflow.core`, `.types`, `.flows`, `.decision`, `.visualization`, etc.)
- Fluent DSL for probabilistic modeling
- **Graphviz-powered visualization** of flows, DAGs, belief networks, and decision trees
- Decision & Integration layer
- Sphinx documentation + interactive tutorials
- Post-merge cleanup complete (all `__init__.py`, imports, Graphviz deps)

## Quick Example

```python
from probflow import *

portfolio = (
    Asset("SPY", weight=0.60)
        .returns(Normal(mu=0.10, sigma=0.18))
    + Asset("TLT", weight=0.40)
        .returns(Normal(mu=0.05, sigma=0.08))
).with_correlation("SPY", "TLT", -0.35)

result = portfolio.simulate(n=10_000)
result.plot_distribution()
result.visualize_flow()   # ← renders interactive Graphviz DAG
