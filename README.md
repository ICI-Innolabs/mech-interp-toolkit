# Mech-Interp Toolkit

A library of useful **mechanistic interpretability methods** for analyzing transformer-based language models, built on top of [TransformerLens](https://transformerlensorg.github.io/TransformerLens/) and [AutoCircuit](https://ufo-101.github.io/auto-circuit/). This is a work of organizing existing methods grounded in the mech-interp literature that are paper-specific and sometimes verbosely implemented. 

All methods are demonstrated on the classic **Indirect Object Identification [(IOI) task](https://arxiv.org/pdf/2211.00593)** and most of them can be easily adapted to custom tasks.

## Requirements

- Python 3.10+
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [AutoCircuit](https://github.com/ufo-101/auto-circuit)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Supported Methods

### 🔍 Observational Methods
- **Logit Lens (1)**: Projecting hidden states of Transformer components via the unembedding to interpret intermediate representations.
- **Direct Logit Attribution (2)**: Tracing the effect on the output logits of each residual stream component at different granularities of the Transformer (3).

### 🔧 Interventional Methods
- **Activation Patching (4, 5)**: Measuring the effect of replacing internal activations with those from a corrupted run (particular variation of the original prompt).
- **Path Patching (6, 7)**: Testing the causal effect of direct paths from specific senders nodes to receivers in isolation from other indirect effects.

### 🤖 Automatic Circuit Discovery
- **Edge Attribution Patching (8, 9)**: Finding a minimal circuit by computing the attribution (importance) for each edge in the model and masking unimportant edges.
- **ACDC Algorithm (10)**: Greedy circuit discovery by applying a form of iterative Path Patching.


## Examples

| Notebook                 | Description |
|--------------------------|-------------|
| `ioi_observational.ipynb`| Applying observational methods for investigating internal computation |
| `ioi_patching.ipynb`     | Applying manual patching methods for finding the IOI circuit |
| `ioi_auto_circuit.ipynb` | Finding minimal circuits using automatic circuit discovery algorithms |

## 📖 References

1. [interpreting GPT: the logit lens](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), by nostalgebraist (2020)
2. [A Comprehensive Mechanistic Interpretability Explainer & Glossary. Mechanistic Interpretability Techniques](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=disz2gTx-jooAcR0a5r8e7LZ£), by Neel Nanda
3. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), Elhage et al. (2021)
4. [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/pdf/2309.16042), Zhang & Nanda (2023)
5. [How to use and interpret activation patching](https://arxiv.org/pdf/2404.15255), Heimersheim & Nanda (2024)
6. [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/pdf/2211.00593), Wang et al. (2022)
7. Two replications of Path Patching, [first](https://colab.research.google.com/drive/15CJ1WAf8AWm6emI3t2nVfnO85-hxwyJU#scrollTo=vWnh6D5GDmL2), [second](https://colab.research.google.com/drive/1AA0wj2sHoZwtmy82WXORcZzk9urL1lVA#scrollTo=teSb1k5Ul6mS) by Callum McDougall
8. [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/pdf/2310.10348), Syed et al. (2023)
9. [How to Do Patching Fast](https://www.lesswrong.com/posts/caZ3yR5GnzbZe2yJ3/how-to-do-patching-fast#Fast_Edge_Patching), by Joseph Miller (2024)
10. [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/pdf/2304.14997), Conmy et al. (2023)


## TODO: 

- [ ] Attribution Patching 🟡 MEDIUM: `todo`
- [ ] Run different automatic circuit discovery algorithms on the IOI circuit using <span style="color: yellow">run_prune_algos()</span> from <span style="color: green">auto_circuit.prune_algos</span> 🟡 MEDIUM: `in progress`
- [ ] JSON prompt dataset generator on custom tasks for auto circuit stuff (with `seq_labels` and `word_idxs`) 🟢 LOW: `todo`


## Note

This library puts altogether work I’ve done over the past year in mechanistic interpretability, across undergraduate and personal research projects, and is intended for facilitating easy access to fundamental mech-interp methods for future projects. While it does not yet incorporate newer approaches such as dictionary learning with Sparse Autoencoders or Transcoders for circuit discovery and model steering, it implements and explains many of the foundational methods that underpin current interpretability research, which I am contented with in this first version of the library. 

Hopefully in the near future I will add support for the latter methods in compatibility with TransformerLens. 