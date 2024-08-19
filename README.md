# Expressive Power of Temporal Message Passing

This contains code for the experimental part of _Expressive Power of Temporal Message Passing_, in which the theoretical expressive power of various temporal message-passing formalisms are analysed.

The code relies on [PyTorch](https://pytorch.org/) and [TGB](https://tgb.complexdatalab.com/).

## Usage

First ensure that hyper-parameters in `hyper.py` are set correctly. Then, a link-prediction experiment can be run by invoking `linkpred.py` with two arguments: (i) the model type, either 'T1' or 'T2', corresponding to global and local models respectively and (ii) the [TGB link-prediction benchmark](https://tgb.complexdatalab.com/docs/linkprop/) you wish to use, e.g. `tgbl-wiki`.

Datasets are downloaded and processed automatically by the TGB infrastructure, with a prompt on first use. Coarse-grained status is provided on standard output, and fine-grained loss curves to TensorBoard `runs/`.
