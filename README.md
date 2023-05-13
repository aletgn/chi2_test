# A Simple Python Module for $\chi^2$ Goodness-of-fit Test

## Disclamier
This module should preferably be used to test unimodal distributions. The module currently implements the following:

- Normal Distribution
- Lognormal Distribution

## Theoretical Primer
Let $X$ be a scalar random variable. Upon an adequate number of tests $n$ we wish to ascertain wether X follows a certain underlying distribution we hypothesise, $\mathcal{D}$. Upon an adquated number of samples, $N$, we wish to ascertain whether $X$ follows a distribution we hypothesise _a priori_. Generally, this is accomplished by setting out the so called $\chi^2$, or _goodness-of-fit_ test. Let us formalise the structure of such test:

$$\begin{cases} H_0:\ X \sim \mathcal{D} \\
H_1:\ X\nsim \mathcal{D}\end{cases}$$

Let us partition the samples $N$ into $M$ bins (or classes):

$$ \mathcal{K}_i\ \forall i=1,2,\dots, M$$

so that each bins is characterised by an _observed numerosity_ $\mathcal{O}_{i}$. Alongside, we can compute the _theoretical numerosity_ for each bin:

$$ \mathcal{E}_i = N\cdot \mathbb{P}[X\in\mathcal{X}_i]$$

As the best practice suggests, should the i-th class have $\mathcal{O}_i < 5$, such a class should be merged with its left or right neighbours. If this adjustment is not done, the denominator of $U$ might become far too high for the i-th class, thus eventually leading to huge, yet unjustified values of $U$. As widely employed for this kind of test, we adopt the following _test statistic_:

$$ U = \sum_{i  = 1}^{k}\frac{(\mathcal{O}_i - \mathcal{E}_i)^2}{\mathcal{E}_i} $$

It is widely known that, under $H_0$, $U\sim \chi^2_l$ where $l$ is the number of degrees of freedom, computed as

$$ l = N - 1 - P$$

where $P$ is the number of parameters _estimated from the sample_. If we do not estimate any parameter, then $P=0$. For instance, if $\mathcal{D}$ is hypothesised as a Normal distribution whose mean and variance are estimated from the given sample, then $P = 2$.

In order to accomplish the test we need to set a significatity level thereof: $\alpha$. Next, we computed the $1-\alpha$ quartile of a $\chi^2$ distribution having $l$ degrees of freedom, namely $\chi^2_{1-\alpha, l}$. Finally, the last check:

$$\begin{cases} U <  \chi^2_{1-\alpha, l}\ \text{do not reject}\ H_0\\
U > \chi^2_{1-\alpha, l}\ \text{reject}\ H_0\end{cases} $$


## Examples
`test` is meant to collect working examples, both `.py` and `ipynb`. Thus, `jupyter notebook` or `jupyter lab` is recommended.
