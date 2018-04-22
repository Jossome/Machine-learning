# Hidden-Markov-Models
Implement EM to train an HMM on the two-dimensional data set points.dat.

## Files
- hmm.py: the main part, including training and evaluating and printing outputs.
- plot.py: run to plot the experiment results.
- result.png: Visualized comparison among different K.
- points.dat: the data.

## Algorithm
```
Initialize mu, sigma, pi, A(matrix for transition probabilities)
Repeat:
    # E-step
    Using forward and backward to get alpha and beta.
    gamma = alpha * beta
    Calculate ksi using alpha, beta, transition probabilities and emission probabilities.
    
    # M-step
    Update pi, A, mu and sigma
```
- For likelihood, I divide it by the number of samples to get same scale of the results of train set and dev set.

## Instructions
- To generate textual output, here is the sample CLI:
```
./hmm.py -K 4 --epochs 5
```
- Or you can discard the arguments by doing:
```
./hmm.py
```
where K=4, epochs=10, tied=False are default settings.
- To generate plot for the experiment:
```
./plot.py
```

## Results
- Number of states (K)
    - For K = 1..8, the algorithm converges before epoch 20.
    - For K = 9, the learning curve is ugly.
- Train vs dev
    - The curves in train increase and converge, while those in dev oscillate and converge.
    - As K goes larger, likelihood on dev goes smaller and unstable.

## Your interpretation
- "Larger better" rule does not apply on number of states.
- When compared to gaussian mixture models (GMM):
    - HMM gets better likelihood.
    - GMM is more flexible since we can set different variance of the model, e.g. tied or separate covariance matrices.
    - HMM can model the data better than the original non-sequence model if number of states is chosen carefully.
- The best number of state is four, according to the result on train set.


## References
https://docs.scipy.org/doc/
https://blog.csdn.net/tostq/article/details/70849271
