# DNA Accessibility Predictor 

This project showcases a 1-dimensional convolutional neural network (CNN) trained to predict protein binding and chromatin accessibility directly from raw DNA sequence. It is built in PyTorch and trained on ENCODE ChIP-seq and ATAC-seq data from the human genome. The model takes a short window of DNA sequence as input and predicts whether that region is bound by a protein or accessible to transcription machinery. 

Two tasks are covered:
1. CTCF Binding Prediction: CTCF is a transcriptional repressor that plays a major role in 3D genome organization. The model learns to recognize the CTCF binding motif from ChIP-seq peak data in the A549 lung cancer cell line.
2. Chromatin Accessibility Prediction: Using ATAC-seq data, the model predicts which regions of the genome are "open" and accessible, which is an important indicator of regulatory activity

## how it works: 
DNA sequences are encoded as one-hot matrices of shape 4 x sequence_length; one row per nucleotide (A, C, G, T), with a 1 in the position where that base appears. Missing bases (Ns) are encoded as all zeros. Positive examples come directly from ChIP-seq or ATAC-seq peaks. Negative examples are sampled from the genomic regions between peaks on the same chromosome, keeping the GC content distribution realistic.

The dataset is split by chromosome rather than randomly. This prevents data leakage since nearby genomic regions share similar sequence composition.
```
Training:     chromosomes 4–22
Validation:   chromosomes 2–3
Test:         chromosome 1
```

## model architecture 
1. DNA Sequence (one-hot encoded)
2. Conv1D → BatchNorm → Dropout → MaxPool → ELU
3. Conv1D → BatchNorm → Dropout → MaxPool → ELU
4. Flatten
5. Linear → Dropout → ELU
6. Linear → Sigmoid → P(binding)

## some decision choices:
- BatchNorm after each convolutional layer for faster convergence
- Dropout as regularization to prevent memorizing training sequences
- ELU activation instead of ReLU to avoid dead neurons
- Early stopping based on validation loss with a patience of 10 epochs
- Adam with AMSGrad as the optimizer

## projct structure 
```
dna-accessibility-predictor/
    chromatin-accessibility-predictor.ipynb                      # Full notebook with training, evaluation, and interpretation
```

## how to understand this model: 
Two interpretation methods are implemented to understand what sequence features the model is responding to:
- In Silico Mutagenesis: for each position in a sequence, every possible single base substitution is tested and the change in predicted probability is recorded. It is computationally expensive (4 × L forward passes per sequence) but intuitive. The resulting sequence logos closely match the known CTCF consensus motif from JASPAR.
- Saliency Maps: gradients of the output with respect to the input sequence, multiplied elementwise by the input itself. Only requires one forward and one backward pass, making it much faster. Highlights similar regions to mutagenesis but with more noise outside the binding site.

For positive examples both methods consistently highlight a clear binding motif. For negative examples the signal is scattered and weak, which is as expected since there is no specific motif present.

## how I experimented with the model: 
- Underfitting: reducing the number of filters and removing pooling layers produces a model that cannot capture the complexity of binding patterns, resulting in low train and validation accuracy around 60%.
- Overfitting: increasing model capacity significantly (128 filters, 3 dense layers) without regularization causes training accuracy to climb while validation accuracy plateaus, a classic sign of memorization.
- Depth vs Accuracy: varying the number of convolutional layers from 1 to 4 shows that 2 layers gives the best tradeoff. Beyond that, validation accuracy stops improving while training accuracy keeps climbing.
- Dropout Tuning: testing dropout rates of 0.2, 0.4, 0.6, and 0.8 on a large capacity model shows that higher dropout (0.8) gives the best validation accuracy by forcing the network to learn more generalizable features rather than memorizing the training set.

## performance
- Simple single-filter CNN performed with ~75% accuracy
- Full LeNet-style CNN (CTCF) performed with ~93% accuracy
- CNN (ATAC-seq task) performed with ~75% accuracy

## required tools: 
torch, numpy, pandas, matplotlib, logomaker, Cython

# to run this project: 
Please reach out to sr3986@columbia regarding details about how to run this project.
