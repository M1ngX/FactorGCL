# FactorGCL
A simple unofficial implementation of FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual  Contrastive Learning for Stock Returns Prediction

Paper: https://arxiv.org/pdf/2502.05218

There are a couple of things that may not match perfectly with the paper:
1. The paper seems to create a fully-connected layer from hidden_size to 1 for the output of each module, which is then linearly weighted. I concatenate the outputs of the three modules and feed them into the one fully-connected layer instead, which should theoretically be equivalent.
2. The paper doesn't use the “Individual Alpha Module” when calculating e_alpha_future, and I don't understand why this is inconsistent with calculating e_alpha, but my implementation here is still consistent with the paper anyway.
3. Personally I am skeptical about the marginal effectiveness of the "Temporal Residual Contrastive Learning", so I also implemented a model that does not use this module, which is much simpler to train.
