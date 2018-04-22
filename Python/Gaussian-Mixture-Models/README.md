# Gaussian Mixture Models
Implement EM fitting of a mixture of gaussians on the two-dimensional data set points.dat.

## Files=
- em.py: the main part, including training and evaluating and printing outputs.
- plot.py: run to plot the experiment results.
- CompareK.png: Visualized comparison among different number of mixtures.
- CompareTied.png: Visualized comparison between tied and separated covariance matrices.
- points.dat: the data.

## Algorithm
- Initialize the three parameters lambda, mu, sigma
```
Repeat:
    # E-step
    For n:
        For k:
            Calculate P(z_n = k|x_n)
    # M-step
    For k:
        Update the three parameters
```

- For likelihood, I divide it by the number of samples to get same scale of the results of train set and dev set.

## Instructions
- To generate textual output, here is the sample CLI:
```
./em.py -K 4 --epochs 5 --tied False
```
- Or you can discard the arguments by doing:
```
./em.py
```
where K=4, epochs=10, tied=False are default settings.
- To generate plot for the experiment:
```
./plot.py
```
which generates the two images.

## Results
- In CompareK.png
    - Number of mixtures
        - K = 1 is a flat line.
        - For "tied", results of K >= 2 does not differ too much.
        - For "separate", results of K = 2 is not as good as others.
        - When K goes larger, the rise in likelihood is not as apparent.
    - Tied vs separate
        - We can see that for separate covariance matrices, the algorithm always converges.
        - But for tied covariance matrix, the algorithm may not converge in some cases.
        - Nevertheless, the curves for tied covariance matrix are more well-shaped.
    - Train vs dev
        - The curves in train always increase, while those in dev shows some peaks.
        - Generally, the results are of the same magnitude, so the models are not over-fitted.
- In CompareTied.png
    - We can see that "tied" converges earlier than "separate" does.
    - "Separate" has better performance than "tied" when K >= 3.

## Your interpretation
- Comparison among different number of mixtures
    - In this dataset, K >= 3 is a good choice.
    - We cannot tell which specific K is the best, because likelihood curve for training data is different from that of dev data.
- Comparison between tied and separate covariance matrices
    - Separate covariance matrices can provide higher likelihood.
    - Tied covariance matrix can provide speed (because it converges faster).
    
## References
https://docs.scipy.org/doc/
