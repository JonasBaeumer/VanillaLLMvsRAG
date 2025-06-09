sample_paper_one = """
Input: Title: Identity Matters in Deep Learning Abstract: An emerging design principle in deep learning is that each layer of a deep
artificial neural network should be able to easily express the identity
transformation. This idea not only motivated various normalization techniques,
such as batch normalization, but was also key to the immense success of
residual networks.

In this work, we put the principle of identity parameterization on a more 
solid theoretical footing alongside further empirical progress. We first
give a strikingly simple proof that arbitrarily deep linear residual networks
have no spurious local optima. The same result for feed-forward networks in
their standard parameterization is substantially more delicate.  Second, we
show that residual networks with ReLu activations have universal finite-sample
expressivity in the sense that the network can represent any function of its
sample provided that the model has more parameters than the sample size.

Directly inspired by our theory, we experiment with a radically simple
residual architecture consisting of only residual convolutional layers and
ReLu activations, but no batch normalization, dropout, or max pool. Our model
improves significantly on previous all-convolutional networks on the CIFAR10,
CIFAR100, and ImageNet classification benchmarks.
 Main Text: ## 1 Introduction

Traditional convolutional neural networks for image classification, such as AlexNet (Krizhevsky et al. (2012)), are parameterized in such a way that when all trainable weights are 0, a convolutional layer represents the 0-mapping. Moreover, the weights are initialized symmetrically around 0. This standard parameterization makes it non-trivial for a convolutional layer trained with stochastic gradient methods to preserve features that were already good. Put differently, such convolutional layers cannot easily converge to the identity transformation at training time. This shortcoming was observed and partially addressed by Ioffe & Szegedy (2015) through batch normalization, i.e., layer-wise whitening of the input with a learned mean and covariance. But the idea remained somewhat implicit until *residual networks* (He et al. (2015); He et al. (2016)) explicitly introduced a reparameterization of the convolutional layers such that when all trainable weights are 0, the layer represents the identity function. Formally, for an input x, each residual layer has the form x + h(x), rather than h(x). This simple reparameterization allows for much deeper architectures largely avoiding the problem of vanishing (or exploding) gradients. Residual networks, and subsequent architectures that use the same parameterization, have since then consistently achieved state-of-the-art results on various computer vision benchmarks such as CIFAR10 and ImageNet.

## 1.1 Our Contributions

In this work, we consider identity parameterizations from a theoretical perspective, while translating some of our theoretical insight back into experiments. Loosely speaking, our first result underlines how identity parameterizations make optimization easier, while our second result shows the same is true for representation.

Linear residual networks.

Since general non-linear neural networks, are beyond the reach of current theoretical methods in optimization, we consider the case of deep *linear* networks as a simplified model. A linear network represents an arbitrary linear map as a sequence of matrices Aℓ · · · A2A1.

The objective function is E∥y − Aℓ *· · ·* A1x∥2, where y = Rx for some unknown linear transformation R and x is drawn from a distribution. Such linear networks have been studied actively in recent years as a stepping stone toward the general non-linear case (see Section 1.2). Even though Aℓ *· · ·* A1 is just a linear map, the optimization problem over the factored variables (Aℓ*, . . . , A*1) is non-convex. In analogy with residual networks, we will instead parameterize the objective function asmin A1,...,Aℓ E∥y − (I + Aℓ) · · · (I + A1)x∥2 . (1.1)
 To give some intuition, when the depth ℓ is large enough, we can hope that the target function R has a factored representation in which each matrix Ai has small norm. Any symmetric positive semidefinite matrix O can, for example, be written as a product O = Oℓ *· · ·* O1, where each Oi = O1/ℓ is very close to the identity for large ℓ so that Ai = Oi − I has small spectral norm. We first prove that an analogous claim is true for all linear transformations R. Specifically, we prove that for every linear transformation R, there exists a global optimizer (A1*, . . . , A*ℓ) of (1.1) such that for large enough depth ℓ, max 1≤i≤ℓ ∥Ai∥ ≤ O(1/ℓ).

(1.2)
Here, ∥A∥ denotes the spectral norm of A. The constant factor depends on the conditioning of R.

We give the formal statement in Theorem 2.1. The theorem has the interesting consequence that as the depth increases, smaller norm solutions exist and hence regularization may offset the increase in parameters. Having established the existence of small norm solutions, our main result on linear residual networks shows that the objective function (1.1) is, in fact, easy to optimize when all matrices have sufficiently small norm. More formally, letting A = (A1*, . . . , A*ℓ) and f(A) denote the objective function in (1.1), we can show that the gradients of vanish only when f(A) = 0 provided that maxi ∥Ai∥ ≤
 O(1/ℓ). See Theorem 2.2. This result implies that linear residual networks have no critical points other than the global optimum. In contrast, for standard linear neural networks we only know, by work of Kawaguchi (2016) that these networks don't have local optima except the global optimum, but it doesn't rule out other critical points. In fact, setting Ai = 0 will always lead to a bad critical point in the standard parameterization.

Universal finite sample expressivity.

Going back to non-linear residual networks with ReLU activations, we can ask: How expressive are deep neural networks that are solely based on residual layers with ReLU activations? To answer this question, we give a very simple construction showing that such residual networks have perfect finite sample expressivity. In other words, a residual network with ReLU activations can easily express any functions of a sample of size n, provided that it has sufficiently more than n parameters. Note that this requirement is easily met in practice. On CIFAR 10 (n = 50000), for example, successful residual networks often have more than 106 parameters. More formally, for a data set of size n with r classes, our construction requires O(n log n+r2)
 parameters. Theorem 3.2 gives the formal statement. Each residual layer in our construction is of the form x + V ReLU(Ux), where U and V are linear transformations. These layers are significantly simpler than standard residual layers, which typically have two ReLU activations as well as two instances of batch normalization.

The power of all-convolutional residual networks.

Directly inspired by the simplicity of our expressivity result, we experiment with a very similar architecture on the CIFAR10, CIFAR100, and ImageNet data sets. Our architecture is merely a chain of convolutional residual layers each with a single ReLU activation, but without batch normalization, dropout, or max pooling as are common in standard architectures. The last layer is a fixed random projection that is not trained. In line with our theory, the convolutional weights are initialized near 0, using Gaussian noise mainly as a symmetry breaker. The only regularizer is standard weight decay (ℓ2-regularization) and there is no need for dropout. Despite its simplicity, our architecture reaches 6.38% top-1 classification error on the CIFAR10 benchmark (with standard data augmentation). This is competitive with the best residual network reported in He et al. (2015), which achieved 6.43%. Moreover, it improves upon the performance of the previous best *all-convolutional* network, 7.25%, achieved by Springenberg et al. (2014). Unlike ours, this previous all-convolutional architecture additionally required dropout and a non-standard preprocessing (ZCA) of the entire data set. Our architecture also improves significantly upon Springenberg et al. (2014) on both Cifar100 and ImageNet.

## 1.2 Related Work

Since the advent of residual networks (He et al. (2015); He et al. (2016)), most state-of-the-art networks for image classification have adopted a residual parameterization of the convolutional layers. Further impressive improvements were reported by Huang et al. (2016) with a variant of residual networks, called *dense nets*. Rather than adding the original input to the output of a convolutional layer, these networks preserve the original features directly by concatenation. In doing so, dense nets are also able to easily encode an identity embedding in a higher-dimensional space. It would be interesting to see if our theoretical results also apply to this variant of residual networks.

There has been recent progress on understanding the optimization landscape of neural networks, though a comprehensive answer remains elusive.

Experiments in Goodfellow et al. (2014)
 and Dauphin et al. (2014) suggest that the training objectives have a limited number of bad local minima with large function values. Work by Choromanska et al. (2015) draws an analogy between the optimization landscape of neural nets and that of the spin glass model in physics (Auffinger et al. (2013)). Soudry & Carmon (2016) showed that 2-layer neural networks have no bad differentiable local minima, but they didn't prove that a good differentiable local minimum does exist. Baldi & Hornik (1989) and Kawaguchi (2016) show that linear neural networks have no bad local minima. In contrast, we show that the optimization landscape of deep linear residual networks has no bad critical point, which is a stronger and more desirable property. Our proof is also notably simpler illustrating the power of re-parametrization for optimization. Our results also indicate that deeper networks may have more desirable optimization landscapes compared with shallower ones.

## 2 Optimization Landscape Of Linear Residual Networks

Consider the problem of learning a linear transformation R: Rd → Rd from noisy measurements y = Rx + ξ, where ξ *∈ N*(0, Idd) is a d-dimensional spherical Gaussian vector. Denoting by D the distribution of the input data x, let Σ = Ex∼D[xx⊤] be its covariance matrix.

There are, of course, many ways to solve this classical problem, but our goal is to gain insights into the optimization landscape of neural nets, and in particular, residual networks. We therefore parameterize our learned model by a sequence of weight matrices $A_{1},\ldots,A_{\ell}\in\mathbb{R}^{d\times d}$,

$$h_{0}=x\,,\qquad h_{j}=h_{j-1}+A_{j}h_{j-1}\,,\qquad\hat{y}=h_{\ell}\,.\tag{2.1}$$

Here $h_{1},\ldots,h_{\ell-1}$ are the $\ell-1$ hidden layers and $\hat{y}=h_{\ell}$ are the predictions of the learned model on input $x$. More succinctly, we have

$$\hat{y}=(\mathrm{Id}_{d}+A_{\ell})\ldots(\mathrm{Id}+A_{1})x\,.$$
It is easy to see that this model can express any linear transformation R. We will use A as a shorthand for all of the weight matrices, that is, the ℓ × d × d-dimensional tensor the contains A1*, . . . , A*ℓ as slices. Our objective function is the maximum likelihood estimator,

$$f(A,(x,y))=\|\hat{y}-y\|^{2}=\|({\rm Id}+A_{\ell})\dots({\rm Id}+A_{1})x-Rx-\xi\|^{2}\;.\tag{2.2}$$

We will analyze the landscape of the _population risk_, defined as,

$$f(A):=\mathbb{E}\left[f(A,(x,y))\right]\;.$$
Recall that ∥Ai∥ is the spectral norm of Ai. We define the norm *|||·|||* for the tensor A as the maximum of the spectral norms of its slices,

$$\|A\|:=\operatorname*{max}_{1\leq i\leq\ell}\|A_{i}\|\,.$$
The first theorem of this section states that the objective function f has an optimal solution with small *|||·|||*-norm, which is *inversely* proportional to the number of layers ℓ.

Thus, when

the architecture is deep, we can shoot for fairly small norm solutions. We define $\gamma:=\max\{|\log\sigma_{\max}(R)|,|\log\sigma_{\min}(R)|\}$. Here $\sigma_{\min}(\cdot),\sigma_{\max}(\cdot)$ denote the least and largest singular values of $R$ respectively.

**Theorem 2.1**.: _Suppose $\ell\geq3\gamma$. Then, there exists a global optimum solution $A^{*}$ of the population risk $f(\cdot)$ with norm_

$$\|A^{*}\|\leq2(\sqrt{\pi}+\sqrt{3\gamma})^{2}/\ell\,.$$

Here $\gamma$ should be thought of as a constant since if $R$ is too large (or too small), we can scale the Here γ should be thought of as a constant since if R is too large (or too small), we can scale the data properly so that σmin(R) ≤ 1 ≤ σmax(R). Concretely, if σmax(R)/σmin(R) = κ, then we can scaling for the outputs properly so that σmin(R) = 1/√κ and σmax(R) = √κ. In this case, we have γ = log √κ, which will remain a small constant for fairly large condition number κ. We also point out that we made no attempt to optimize the constant factors here in the analysis. The proof of Theorem 2.1 is rather involved and is deferred to Section A.

Given the observation of Theorem 2.1, we restrict our attention to analyzing the landscape of $f(\cdot)$ in the set of $A$ with $\|\cdot\|$-norm less than $\tau$,

$$\mathcal{B}_{\tau}=\left\{A\in\mathbb{R}^{\ell\times d\times d}:\|A\|\leq\tau\right\}.$$
Here using Theorem 2.1, the radius τ should be thought of as on the order of 1/ℓ. Our main theorem in this section claims that there is no bad critical point in the domain Bτ for any *τ <* 1. Recall that a critical point has vanishing gradient.

Theorem 2.2. For any τ < 1, we have that any critical point A of the objective function f(·) inside the domain Bτ must also be a global minimum.

Theorem 2.2 suggests that it is sufficient for the optimizer to converge to critical points of the population risk, since all the critical points are also global minima.

Moreover, in addition to Theorem 2.2, we also have that any A inside the domain Bτ satisfies that ∥∇f(A)∥2
 F ≥ 4ℓ(1 − τ)ℓ−1σmin(Σ)2(f(A) − Copt) .

(2.3)
Here Copt is the global minimal value of f(·) and ∥∇f(A)∥F denotes the euclidean norm1 of the ℓ × d × d-dimensional tensor ∇f(A). Note that σmin(Σ) denote the minimum singular value of Σ.

Equation (2.3) says that the gradient has fairly large norm compared to the error, which guarantees convergence of the gradient descent to a global minimum (Karimi et al. (2016)) if the iterates stay inside the domain Bτ, which is not guaranteed by Theorem 2.2 by itself.

Towards proving Theorem 2.2, we start off with a simple claim that simplifies the population risk.

We also use *∥·∥*F to denote the Frobenius norm of a matrix.

Claim 2.3. In the setting of this section, we have,

F + C . (2.4) f(A) = ���((Id + Aℓ) *. . .* (Id + A1) − R)Σ1/2��� 2
Here C is a constant that doesn't depend on A, and Σ1/2 denote the square root of Σ, that is, the unique symmetric matrix B *that satisfies* B2 = Σ.

Proof of Claim 2.3. Let tr(A) denotes the trace of the matrix A. Let E = (Id+Aℓ) *. . .* (Id+A1)−R.

Recalling the definition of f(A) and using equation (2.2), we have

f(A) = E � ∥Ex − ξ∥2� (by equation (2.2)) = E � ∥Ex∥2 + ∥ξ∥2 − 2⟨Ex, ξ⟩ � = E � tr(Exx⊤E⊤) � + E � ∥ξ∥2� (since E [⟨*Ex, ξ*⟩] = E [⟨Ex, E [ξ|x]⟩] = 0) = tr � E E � xx⊤� E⊤� + C (where C = E[xx⊤]) $=\operatorname{tr}(E\Sigma E^{\top})+C=\|E\Sigma^{1/2}\|_{F}^{2}+C\,.$ (since $\mathbb{E}\left[xx^{\top}\right]=\Sigma$)
ijk T 2
ijk.

1That is, ∥T∥F :=
��

Next we compute the gradients of the objective function $f(\cdot)$ from straightforward matrix calculus. We defer the full proof to Section A.

**Lemma 2.4**.: _The gradients of $f(\cdot)$ can be written as,_

$$\frac{\partial f}{\partial A_{\rm i}}=2({\rm Id}+A_{i}^{\top})\ldots({\rm Id}+A_{i+1}^{\top})E\Sigma({\rm Id}+A_{i-1}^{\top})\ldots({\rm Id}+A_{1}^{\top})\;,\tag{2.5}$$
where E = (Id + Aℓ) . . . (Id + A1) − R .

Now we are ready to prove Theorem 2.2. The key observation is that each matric Aj has small norm and cannot cancel the identity matrix. Therefore, the gradients in equation (2.5) is a product of non-zero matrices, except for the error matrix E. Therefore, if the gradient vanishes, then the only possibility is that the matrix E vanishes, which in turns implies A is an optimal solution.

Proof of Theorem 2.2. Using Lemma 2.4, we have, ���� ∂f ∂Ai ���� F = 2 ��(Id + A⊤ ℓ ) . . . (Id + A⊤ i+1)EΣ(Id + A⊤ i−1) . . . (Id + A⊤ 1 ) �� F (by Lemma 2.4) j̸=i σmin(Id + A⊤ i ) · σmin(Σ)∥E∥F (by Claim C.2) ≥ 2 � ≥ 2(1 − τ)ℓ−1σmin(Σ)∥E∥ . (since σmin(Id + A) ≥ 1 *− ∥*A∥)
It follows that

2 ∥∇f(A)∥2 F = F ≥ 4ℓ(1 − τ)ℓ−1σmin(Σ)2∥E∥2 i=1 ℓ � ���� ∂f ∂Ai ���� ≥ 4ℓ(1 − τ)ℓ−1σmin(Σ)2(f(A) − C) (by the definition of E and Claim 2.3) ≥ 4ℓ(1 − τ)ℓ−1σmin(Σ)2(f(A) − Copt) . (since Copt = minA f(A) ≥ C by Claim 2.3) Therefore we complete the proof of equation (2.3). Finally, if $A$ is a critical point, namely, $\nabla f(A)=0$, then by equation (2.3) we have that $f(A)=C_{\text{opt}}$. That is, $A$ is a global minimum.

## 3 Representational Power Of The Residual Networks

In this section we characterize the finite-sample emissivity of residual networks. We consider a residual layers with a single ReLU activation and no batch normalization. The basic residual building block is a function $\mathcal{T}_{U,V,s}(\cdot):\mathbb{R}^{k}\rightarrow\mathbb{R}^{k}$ that is parameterized by two weight matrices $U\in\mathbb{R}^{\times k},V\in\mathbb{R}^{k\times k}$ and a bias vector $s\in\mathbb{R}^{k}$,

$$\mathcal{T}_{U,V,s}(h)=V\text{ReLU}(Uh+s)\,.\tag{3.1}$$

A residual network is composed of a sequence of such residual blocks. In comparison with the full pre-activation architecture in He et al. (2016), we remove two batch normalization layers and one ReLU layer in each building block.

We assume the data has r labels, encoded as r standard basis vectors in Rr, denoted by e1*, . . . , e*r.

We have n training examples (x(1), y(1)), . . . , (x(n), y(n)), where x(i) ∈ Rd denotes the i-th data and y(i) ∈ {e1*, . . . , e*r} denotes the i-th label. Without loss of generality we assume the data are normalized so that x(i) = 1. We also make the mild assumption that no two data points are very close to each other.

Assumption 3.1. We assume that for every 1 ≤ i < j ≤ n, we have ∥x(i) − x(j)∥2 ≥ ρ for some absolute constant ρ > 0.

Images, for example, can always be imperceptibly perturbed in pixel space so as to satisfy this assumption for a small but constant ρ.

Under this mild assumption, we prove that residual networks have the power to express any possible labeling of the data as long as the number of parameters is a logarithmic factor larger than n.

Theorem 3.2. Suppose the training examples satisfy Assumption 3.1. Then, there exists a residual network N *(specified below) with* O(n log n + r2) parameters that perfectly expresses the training data, i.e., for all i ∈ {1, . . . , n}, the network N maps x(i) to y(i).

It is common in practice that *n > r*2, as is for example the case for the Imagenet data set where n > 106 and r = 1000.

We construct the following residual net using the building blocks of the form T*U,V,s* as defined in equation (3.1). The network consists of ℓ + 1 hidden layers h0*, . . . , h*ℓ, and the output is denoted by
 ˆy ∈ Rr. The first layer of weights matrices A0 maps the d-dimensional input to a k-dimensional hidden variable h0. Then we apply ℓ layers of building block T with weight matrices Aj, Bj ∈ Rk×k.

Finally, we apply another layer to map the hidden variable hℓ to the label ˆy in Rk. Mathematically, we have h0 = A0x , hj = hj−1 + TAj,Bj,bj(hj−1),
∀j ∈ {1, . . . , ℓ}

ˆy = hℓ + TAℓ+1,Bℓ+1,sℓ+1(hℓ) . We note that here $A_{\ell+1}\in\mathbb{R}^{k\times r}$ and $B_{\ell+1}\in\mathbb{R}^{r\times r}$ so that the dimension is compatible. We assume the number of labels $r$ and the input dimension $d$ are both smaller than $n$, which is safety true in practical applications.[2] The hyperparameter $k$ will be chosen to be $O(\log n)$ and the number of layers is chosen to be $\ell=\lceil n/k\rceil$. Thus, the first layer has $dk$ parameters, and each of the middle $\ell$ building blocks contains $2k^{2}$ parameters and the final building block has $kr+r^{2}$ parameters. Hence, the total number of parameters is $O(kd+\ell k^{2}+rk+r^{2})=O(n\log n+r^{2})$.

Towards constructing the network N of the form above that fits the data, we first take a random matrix A0 ∈ Rk×d that maps all the data points x(i) to vectors h(i)
0
:= A0x(i). Here we will use h(i)
j to denote the j-th layer of hidden variable of the i-th example. By Johnson-Lindenstrauss Theorem
 (Johnson & Lindenstrauss (1984), or see Wikipedia (2016)), with good probability, the resulting vectors h(i)
0 's remain to satisfy Assumption 3.1 (with slightly different scaling and larger constant ρ), that is, any two vectors h(i)
0
 and h(j)
0
 are not very correlated.

Then we construct ℓ middle layers that maps h(i)
0
to h(i)
ℓ
for every i ∈ {1*, . . . , n*}. These vectors h(i)
ℓ
 will clustered into r groups according to the labels, though they are in the Rk instead of in Rr as desired. Concretely, we design this cluster centers by picking r random unit vectors q1, . . . , qr in Rk. We view them as the surrogate label vectors in dimension k (note that k is potentially much smaller than r). In high dimensions (technically, if *k >* 4 log r) random unit vectors q1*, . . . , q*r are pair-wise uncorrelated with inner product less than < 0.5. We associate the i-th example with the target surrogate label vector v(i) defined as follows,

if $y^{(i)}=e_{j}$, then $v^{(i)}=q_{j}$.

Then we will construct the matrices (A1, B1), . . . , (Aℓ, Bℓ) such that the first ℓ layers of the network maps vector h(i)
0
to the surrogate label vector v(i). Mathematically, we will construct (A1, B1), . . . , (Aℓ, Bℓ) such that

$\forall i\in\{1,\ldots,n\},h^{(i)}_{\ell}=v^{(i)}\,.$

Finally we will construct the last layer TAℓ+1,Bℓ+1,bℓ+1 so that it maps the vectors q1, . . . , qr ∈ Rk
 to e1, . . . , er ∈ Rr,
    ∀j ∈ {1, . . . , r}, qj + TAℓ+1,Bℓ+1,bℓ+1(qj) = ej .
                                                                              (3.4)
Putting these together, we have that by the definition (3.2) and equation (3.3), for every i, if the label is y(i) is ej, then h(i)
              ℓ
        will be qj.
                                        Then by equation (3.4), we have that ˆy(i) = qj + TAℓ+1,Bℓ+1,bℓ+1(qj) = ej. Hence we obtain that ˆy(i) = y(i).

The key part of this plan is the construction of the middle ℓ layers of weight matrices so that h(i)
                                                                                                ℓ
                                                                                                    =
 v(i). We encapsulate this into the following informal lemma. The formal statement and the full proof is deferred to Section B.

Lemma 3.3 (Informal version of Lemma B.2). In the setting above, for (almost) arbitrary vectors h(1)
0 *, . . . , h*(n)
0
 and v(1)*, . . . , v*(n)
∈ {q1, . . . , qr}, there exists weights matrices (A1, B1), . . . , (Aℓ, Bℓ), such that,

$$\forall i\in\{1,\ldots,n\},\quad h_{\ell}^{(i)}=v^{(i)}\;.$$
We briefly sketch the proof of the Lemma to provide intuitions, and defer the full proof to Section B. The operation that each residual block applies to the hidden variable can be abstractly written as,

ˆh → h + TU,V,s(h) . (3.5)
 where h corresponds to the hidden variable before the block and ˆh corresponds to that after. We claim that for an (almost) arbitrary sequence of vectors h(1)*, . . . , h*(n), there exist T*U,V,s*(·) such that operation (3.5) transforms k vectors of h(i)'s to an arbitrary set of other k vectors that we can freely choose, and maintain the value of the rest of n − k vectors. Concretely, for any subset S of size k, and any desired vector v(i)(i ∈ S), there exist *U, V, s* such that

v(i) = h(i) + TU,V,s(h(i)) ∀i ∈ S h(i) = h(i) + TU,V,s(h(i)) ∀i ̸∈ S (3.6)
This claim is formalized in Lemma B.1. We can use it repeatedly to construct ℓ layers of building blocks, each of which transforms a subset of k vectors in {h(1)
0 *, . . . , h*(n)
0 } to the corresponding vectors in {v(1)*, . . . , v*(n)}, and maintains the values of the others. Recall that we have ℓ = ⌈n/k⌉
 layers and therefore after ℓ layers, all the vectors h(i)
0 's are transformed to v(i)'s, which complete the proof sketch.

## 4 Power Of All-Convolutional Residual Networks

Inspired by our theory, we experimented with all-convolutional residual networks on standard image classification benchmarks.

## 4.1 Cifar10 And Cifar100

Our architectures for CIFAR10 and CIFAR100 are identical except for the final dimension corresponding to the number of classes 10 and 100, respectively. In Table 1, we outline our architecture.

Each *residual block* has the form x + C2(ReLU(C1x)), where C1, C2 are convolutions of the specified dimension (kernel width, kernel height, number of input channels, number of output channels). The second convolution in each block always has stride 1, while the first may have stride 2 where indicated. In cases where transformation is not dimensionality-preserving, the original input x is adjusted using averaging pooling and padding as is standard in residual layers. We trained our models with the Tensorflow framework, using a momentum optimizer with momentum 0.9, and batch size is 128. All convolutional weights are trained with weight decay 0.0001.

The initial learning rate is 0.05, which drops by a factor 10 and 30000 and 50000 steps. The model reaches peak performance at around 50k steps, which takes about 24h on a single NVIDIA Tesla K40 GPU. Our code can be easily derived from an open source implementation3 by removing batch normalization, adjusting the residual components and model architecture. An important departure from the code is that we initialize a residual convolutional layer of kernel size k × k and c output channels using a random normal initializer of standard deviation σ = 1/k2c, rather than 1/k√c used for standard convolutional layers. This substantially smaller weight initialization helped training, while not affecting representation. A notable difference from standard models is that the last layer is not trained, but simply a fixed random projection. On the one hand, this slightly improved test error (perhaps due to a regularizing effect). On the other hand, it means that the only trainable weights in our model are those of the convolutions, making our architecture "all-convolutional".

3https://github.com/tensorflow/models/tree/master/resnet

| variable dimensions   | initial stride                  |
|-----------------------|---------------------------------|
| 3                     |                                 |
| ×                     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 16                    | 1                               |
| 1 standard conv       |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 16                    |                                 |
| ×                     |                                 |
| 64                    | 1                               |
| 9 residual blocks     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 64                    |                                 |
| ×                     |                                 |
| 128                   | 2                               |
| 9 residual blocks     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 3                     |                                 |
| ×                     |                                 |
| 128                   |                                 |
| ×                     |                                 |
| 256                   | 2                               |
| 9 residual blocks     |                                 |
| -                     | -                               |
| 8                     |                                 |
| ×                     |                                 |
| 8                     |                                 |
| global average pool   |                                 |
| 256                   |                                 |
| ×                     |                                 |
| num classes           |                                 |
| -                     | random projection (not trained) |

An interesting aspect of our model is that despite its massive size of 13.59 million trainable parameters, the model does not seem to overfit too quickly even though the data set size is 50000. In contrast, we found it difficult to train a model with batch normalization of this size without significant overfitting on CIFAR10. Table 2 summarizes the top-1 classification error of our models compared with a non-exhaustive list of previous works, restricted to the best previous all-convolutional result by Springenberg et al. (2014), the first residual results He et al. (2015), and state-of-the-art results on CIFAR by Huang et al. (2016). All results are with standard data augmentation.

| Method                                            | CIFAR10   |   CIFAR100 |   ImageNet | remarks   |
|---------------------------------------------------|-----------|------------|------------|-----------|
| All-CNN                                           |           |            |            |           |
| 7                                                 | .         |         25 |         32 | .         |
| all-convolutional, dropout, extra data processing |           |            |            |           |
| Ours                                              |           |            |            |           |
| 6                                                 | .         |         38 |         24 | .         |
| all-convolutional                                 |           |            |            |           |
| ResNet                                            |           |            |            |           |
| 6                                                 | .         |         43 |         25 | .         |
| DenseNet                                          |           |            |            |           |
| 3                                                 | .         |         74 |         19 | .         |
| N/A                                               |           |            |            |           |

## 4.2 Imagenet

The ImageNet ILSVRC 2012 data set has 1, 281, 167 data points with 1000 classes. Each image is resized to 224 × 224 pixels with 3 channels. We experimented with an all-convolutional variant of the 34-layer network in He et al. (2015). The original model achieved 25.03% classification error. Our derived model has 35.7M trainable parameters. We trained the model with a momentum optimizer (with momentum 0.9) and a learning rate schedule that decays by a factor of 0.94 every two epochs, starting from the initial learning rate 0.1. Training was distributed across 6 machines updating asynchronously. Each machine was equipped with 8 GPUs (NVIDIA Tesla K40) and used batch size 256 split across the 8 GPUs so that each GPU updated with batches of size 32.

In contrast to the situation with CIFAR10 and CIFAR100, on ImageNet our all-convolutional model performed significantly worse than its original counterpart. Specifically, we experienced a significant amount of *underfitting* suggesting that a larger model would likely perform better.

Despite this issue, our model still reached 35.29% top-1 classification error on the test set (50000 data points), and 14.17% top-5 test error after 700, 000 steps (about one week of training). While no longer state-of-the-art, this performance is significantly better than the 40.7% reported by Krizhevsky et al. (2012), as well as the best all-convolutional architecture by Springenberg et al. (2014). We believe it is quite likely that a better learning rate schedule and hyperparameter settings of our model could substantially improve on the preliminary performance reported here.

## 5 Conclusion

Our theory underlines the importance of identity parameterizations when training deep artificial neural networks. An outstanding open problem is to extend our optimization result to the non-linear case where each residual has a single ReLU activiation as in our expressivity result. We conjecture that a result analogous to Theorem 2.2 is true for the general non-linear case. Unlike with the standard parameterization, we see no fundamental obstacle for such a result. We hope our theory and experiments together help simplify the state of deep learning by aiming to explain its success with a few fundamental principles, rather than a multitude of tricks that need to be delicately combined. We believe that much of the advances in image recognition can be achieved with residual convolutional layers and ReLU activations alone. This could lead to extremely simple (albeit deep) architectures that match the state-of-the-art on all image classification benchmarks.

## References

Antonio Auffinger, G´erard Ben Arous, and Jiˇr´ı ˇCern`y. Random matrices and complexity of spin glasses. *Communications on Pure and Applied Mathematics*, 66(2):165–201, 2013.
P. Baldi and K. Hornik. Neural networks and principal component analysis: Learning from examples without local minima. *Neural Netw.*, 2(1):53–58, January 1989. ISSN 0893-6080. doi: 10. 1016/0893-6080(89)90014-2.
URL http://dx.doi.org/10.1016/0893-6080(89) 90014-2.
Anna Choromanska, Mikael Henaff, Michael Mathieu, G´erard Ben Arous, and Yann LeCun. The loss surfaces of multilayer networks. In *AISTATS*, 2015.
 Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In *Advances in neural information processing systems*, pp. 2933–2941, 2014.
I. J. Goodfellow, O. Vinyals, and A. M. Saxe. Qualitatively characterizing neural network optimization problems. *ArXiv e-prints*, December 2014.
 Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *arXiv prepring arXiv:1506.01497*, 2015.
 Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part IV, pp. 630–645, 2016. doi: 10.1007/ 978-3-319-46493-0 38. URL http://dx.doi.org/10.1007/978-3-319-46493-0_ 38. Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks. CoRR, abs/1608.06993, 2016. URL http://arxiv.org/abs/1608.06993.
 Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, pp. 448–456, 2015. URL http://jmlr. org/proceedings/papers/v37/ioffe15.html.
 William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space. Contemporary mathematics, 26(189-206):1, 1984.
 H. Karimi, J. Nutini, and M. Schmidt. Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-\L{}ojasiewicz Condition. *ArXiv e-prints*, August 2016.
 K. Kawaguchi. Deep Learning without Poor Local Minima. *ArXiv e-prints*, May 2016. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems*, pp. 1097–1105, 2012.
 D. Soudry and Y. Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. *ArXiv e-prints*, May 2016.
 J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for Simplicity: The All Convolutional Net. *ArXiv e-prints*, December 2014.
 Eric W. Weisstein. Normal matrix, from mathworld–a wolfram web resource., 2016. URL http: //mathworld.wolfram.com/NormalMatrix.html.

Wikipedia.
               Johnsonlindenstrauss
                                    lemma
                                            —
                                                wikipedia,
                                                          the
                                                                free
                                                                     encyclopedia,
                                                                                   2016.
  URL
            https://en.wikipedia.org/w/index.php?title=Johnson%E2%80%
 93Lindenstrauss_lemma&oldid=743553642.

## A Missing Proofs In Section 2

In this section, we give the complete proofs for Theorem 2.1 and Lemma 2.4, which are omitted in Section 2.

## A.1 Proof Of Theorem 2.1

It turns out the proof will be significantly easier if R is assumed to be a symmetric positive semidefinite (PSD) matrix, or if we allow the variables to be complex matrices. Here we first give a proof sketch for the first special case. The readers can skip it and jumps to the full proof below. We will also prove stronger results, namely, |||A⋆||| ≤ 3γ/ℓ, for the special case.

When R is PSD, it can be diagonalized by orthonormal matrix U in the sense that R = *UZU* ⊤, where Z = diag(z1*, . . . , z*d) is a diagonal matrix with non-negative diagonal entries z1*, . . . , z*d. Let A⋆
1 = · · · = A⋆
ℓ = U diag(z1/ℓ
i
)U ⊤ − Id, then we have

(Id + A⋆
        ℓ) · · · (Id + A⋆
                      1) = (U diag(z1/ℓ
                                      i
                                         )U ⊤)ℓ = U diag(z1/ℓ
                                                             i
                                                                )ℓU
                                                                            (since U ⊤U = Id)

= UZU ⊤ = R .

We see that the network defined by A⋆ reconstruct the transformation R, and therefore it's a global minimum of the population risk (formally see Claim 2.3 below). Next, we verify that each of the A⋆
                                                                                              j
has small spectral norm:

$$\|A_{j}^{*}\|=\|{\rm Id}-U\,{\rm diag}(z_{i}^{1/\ell})U^{\top})\|=\|U({\rm Id}-{\rm diag}(z_{i})^{1/\ell})U^{\top}\|=\|{\rm Id}-{\rm diag}(z_{i})^{1/\ell}\|$$ (since $$U$$ is orthonormal) $$=\max_{i}|z_{i}^{1/\ell}-1|\,.$$ (A.1)
Since σmin(R) ≤ zi ≤ σmax(R), we have ℓ ≥ 3γ *≥ |* log zi|. It follows that

|z1/ℓ
  i
       − 1| = |e(log zi)/ℓ − 1| ≤ 3|(log zi)/ℓ| ≤ 3γ/ℓ .
                                                                                   (since |ex − 1| ≤ 3|x| for all |x| ≤ 1)

Then using equation (A.1) and the equation above, we have that |||A*||| ≤* maxj∥A⋆
j∥ ≤ 3γ/ℓ, which completes the proof for the special case. Next we give the formal full proof of Theorem 2.1.

Proof of Theorem 2.1. We assume the dimension d is an even number. The odd case has very similar proof and is left to the readers. Let R = *UKV* ⊤ be its singular value decomposition, where U,V are two orthonormal matrices and K is a diagonal matrix. Since U is a normal matrix (that is, U satisfies that UU ⊤ = U ⊤U), by Claim C.1, we have that U can be block-diagnolaized by orthonormal matrix S into U = *SDS*−1, where D = diag(D1*, . . . , D*d/2) is a real block diagonal matrix with each block Di being of size 2 × 2.

Since U is orthonormal, U has all its eigenvalues lying on the unit circle (in complex plane). Since D
 and U are unitarily similar to each other, D also has eigenvalues lying on the unit circle, and so does each of the block Di. This means that each Di is a 2 × 2 dimensional rotation matrix. Each rotation matrix can be written as T(θ) =
�
cos θ
− sin θ
sin θ
cos θ
�
. Suppose Di = T(θi) where θi ∈ [−*π, π*]. Then we have that Di = T(θi/q)q for any integer q (that is chosen later). Let W = diag(T(θi/q)).

Therefore, it follows that D = diag(Di) = W q. Moreover, we have U = *SDS*−1 = (SWS−1)q.

Therefore, let B1 = B2 = · · · = Bq = Id − *SWS*−1, then we have U = (Id + Bq) *. . .* (Id + B1).

We verify the spectral norm of these matrices are indeed small,

$$\|B_{j}\|=\left\|\mathrm{Id}-SWS^{-1}\right\|=\left\|S(\mathrm{Id}-W)S^{-1}\right\|$$ $$=\|\mathrm{Id}-W\|$$ (since $$S$$ is unitary) $$=\max_{i\in[d/2]}\|T(0)-T(\theta_{i}/q)\|$$ (since $$W=\mathrm{diag}(T(\theta_{i}/q))$$ is block diagonal) $$=\max|\sin(\theta_{i}/q)|\leq\pi/q\,.$$

Similarly, we can choose B′
                          1, . . . , B′
                                  q with ∥Cj∥ ≤ π/q so that V ⊤ = (Id + B′
                                                                           q) . . . (Id + B′
                                                                                        1).

Last, we deal with the diagonal matrix K.
                                            Let K
                                                    =
                                                        diag(ki).
                                                                   We have min ki
                                                                                    =
 σmin(R), max ki = σmax(R). Then, we can write K = (K′)p where K′ = diag(k1/p
                                                                            i
                                                                               ) and p
 is an integer to be chosen later. We have that ∥K′ − Id∥ ≤ max |k1/p
                                                         i
                                                             − 1| ≤ max |elog ki·1/p − 1|.
When p ≥ γ = max{log max ki, − log min ki} = max{log σmax(R), − log σmin(R)}, we have
that

$\|K^{\prime}-\operatorname{Id}\|\leq\max|e^{\log k_{i}\cdot1/p}-1|\leq3\max|\log k_{i}\cdot1/p|=3\gamma/p$.

(since $|e^{x}-1|\leq3|x|$ for $|x|\leq1$)

Let B′′
     1 = · · · = B′′
                  p = K′ − Id and then we have K = (Id + B′′
 p) · · · (Id + B′′
 1 ). Finally, we
choose p = ℓ√3γ

 2(√π+√3γ) and q =
 ℓ√π
 √π+√3γ , 4 and let A2p+q = Bq, · · · = Ap+q+1 = B1, Ap+q = B′′
 p, . . . , Aq+1 = B′′
 1 , Aq = B′
 q, . . . , A1 = B′
 1. We have that 2q + ℓ = 1 and

$$R=U K V^{\top}=(\mathrm{Id}+A_{\ell})\dots(\mathrm{Id}+A_{1})\,.$$

Moreover, we have |||A||| ≤ max{∥Bj∥, ∥B′
  j∥.∥B′′
  j ∥} ≤ π/q + 3γ/p ≤ 2(√π + √3γ)2/ℓ, as desired.

## A.2 Proof Of Lemma 2.4

We compute the partial gradients by definition. Let ∆j ∈ Rd×d be an infinitesimal change to Aj.

Using Claim 2.3, consider the Taylor expansion of f(A1*, . . . , A*ℓ + ∆j*, . . . , A*ℓ)
f(A1*, . . . , A*ℓ + ∆j*, . . . , A*ℓ)

F = ���((Id + Aℓ) *· · ·* (Id + Aj + ∆j) *. . .* (Id + A1) − R)Σ1/2��� 2 F = ���((Id + Aℓ) *· · ·* (Id + A1) − R)Σ1/2 + (Id + Aℓ) · · · ∆j *. . .* (Id + A1)Σ1/2��� 2 F + 2⟨((Id + Aℓ) · · · (Id + A1) − R)Σ1/2, (Id + Aℓ) · · · ∆j *. . .* (Id + A1)Σ1/2⟩ + O(∥∆j∥2 F ) = ���(Id + Aℓ) *· · ·* (Id + A1) − R)Σ1/2��� 2 = f(A) + 2⟨(Id + A⊤ ℓ ) . . . (Id + A⊤ j+1)EΣ(Id + A⊤ j−1) . . . (Id + A⊤ 1 ), ∆j⟩ + O(∥∆j∥2 F ) .

By definition, this means that the
                                   ∂f
                                   ∂Aj = 2(Id + A⊤
                                                    ℓ ) . . . (Id + A⊤
                                                                   j+1)EΣ(Id + A⊤
                                                                                   j−1) . . . (Id +
A⊤
 1 ).

## B Missing Proofs In Section 3

In this section, we provide the full proof of Theorem 3.2. We start with the following Lemma that constructs a building block T that transform k vectors of an arbitrary sequence of n vectors to any arbitrary set of vectors, and main the value of the others. For better abstraction we use α(i),β(i) to denote the sequence of vectors.

Lemma B.1. Let S ⊂ [n] be of size k*. Suppose* α(1), . . . , α(n) is a sequences of n vectors satisfying a) for every 1 ≤ i ≤ n, we have 1−ρ′ ≤ ∥αi∥2 ≤ 1+ρ′, and b) if i ̸= j and S contains at least one of i, j, then ∥α(i) −β(j)∥ ≥ 3ρ′*. Let* β(1), . . . , β(n) be an arbitrary sequence of vectors. Then, there exists U, V ∈ Rk×k, s such that for every i ∈ S, we have T*U,V,s*(α(i)) = β(i) − α(i), and moreover, for every i ∈ [n]\S we have T*U,V,s*(α(i)) = 0.

We can see that the conclusion implies

$$\beta^{(i)}=\alpha^{(i)}+\mathcal{T}_{U,V,s}(\alpha^{(i)})\ \ \forall i\in S$$ $$\alpha^{(i)}=\alpha^{(i)}+\mathcal{T}_{U,V,s}(\alpha^{(i)})\ \ \forall i\not\in S$$

which is a different way of writing equation (3.6).

Proof of Lemma B.1. Without loss of generality, suppose S = {1*, . . . , k*}. We construct *U, V, s* as follows. Let the i-th row of U be α(i) for i ∈ [k], and let s = −(1 − 2ρ′) · 1 where 1 denotes the all
1's vector. Let the i-column of V be
1
∥α(i)∥2−(1−2ρ′)(β(i) − α(i)) for i ∈ [k].

Next we verify that the correctness of the construction. We first consider 1 ≤ i ≤ k. We have that Uα(i) is a a vector with i-th coordinate equal to ∥α(i)∥2 ≥ 1 − ρ′. The j-th coordinate of Uα(i) is equal to ⟨α(j), α(i)⟩, which can be upperbounded using the assumption of the Lemma by

2 ⟨α(j), α(i)⟩ = 1 � ∥α(i)∥2 + ∥α(j)∥2� − ∥α(i) − α(j)∥2 ≤ 1 + ρ′ − 3ρ′ ≤ 1 − 2ρ′ . (B.1)

Therefore, this means Uα(i) − (1 − 2ρ′) · 1 contains a single positive entry (with value at least ∥α(i)∥2 − (1 − 2ρ′) ≥ ρ′), and all other entries being non-positive.
 This means that ReLu(Uα(i) +b) = �
 ∥α(i)∥2 − (1 − 2ρ′)
 �
 ei where ei is the i-th natural basis vector. It follows that V ReLu(Uα(i) + b) = (∥α(i)∥2 − (1 − 2ρ′)) V ei = β(i) − α(i).

Finally, consider n ≥ i > k. Then similarly to the computation in equation (B.1), Uα(i) is a vector with all coordinates less than 1 − 2ρ′. Therefore Uα(i) + b is a vector with negative entries. Hence we have ReLu(Uα(i) + b) = 0, which implies V ReLu(Uα(i) + b) = 0.

Now we are ready to state the formal version of Lemma 3.3.

**Lemma B.2**.: _Suppose a sequence of $n$ vectors $z^{(1)},\ldots,z^{(n)}$ satisfies a relaxed version of Assumption 3.1: a for every $i,1=-\rho^{\prime}\leq\|z^{(i)}\|^{2}\leq1+\rho^{\prime}\,b)$ for every $i\neq j$, we have $\|z^{(i)}-z^{(j)}\|^{2}\geq\rho^{\prime}$; Let $v^{(1)},\ldots,v^{(n)}$ be defined above. Then there exists weigh matrices $(A_{1},B_{1}),\ldots,(A_{\ell},B_{\ell})$, such that given $\forall i,h_{0}^{(i)}=z^{(i)}$, we have_

$$\forall i\in\{1,\ldots,n\},\ \ h_{\ell}^{(i)}=v^{(i)}\,.$$
We will use Lemma B.1 repeatedly to construct building blocks TAj,Bk,sj(·), and thus prove Lemma B.2. Each building block TAj,Bk,sj(·) takes a subset of k vectors among {z(1), . . . , z(n)}
 and convert them to v(i)'s, while maintaining all other vectors as fixed. Since they are totally n/k layers, we finally maps all the z(i)'s to the target vectors v(i)'s.

Proof of Lemma B.2. We use Lemma B.1 repeatedly. Let S1 = [1*, . . . , k*]. Then using Lemma B.1
 with α(i) = z(i) and β(i) = v(i) for i ∈ [n], we obtain that there exists A1, B1, b1 such that for i ≤ k, it holds that h(i)
1
= z(i) + TA1,B1,b1(z(i)) = v(i), and for i ≥ k, it holds that h(i)
1
=
 z(i) + TA1,B1,b1(z(i)) = z(i).

Now we construct the other layers inductively. We will construct the layers such that the hidden variable at layer j satisfies h(i)
j
= v(i) for every 1 ≤ i ≤ jk, and h(i)
j
= z(i) for every n ≥ i >
 jk. Assume that we have constructed the first j layer and next we use Lemma B.1 to construct the j + 1 layer. Then we argue that the choice of α(1) = v(1)*, . . . , α*(jk) = v(jk), α(jk+1) =
 z(jk+1)*, . . . , α*(n) = z(n), and S = {jk + 1*, . . . ,* (j + 1)k} satisfies the assumption of Lemma B.1.

Indeed, because qi's are chosen uniformly randomly, we have w.h.p for every s and i, ⟨qs, z(i)⟩ ≤
 1 − ρ′. Thus, since v(i) ∈ {q1*, . . . , q*r}, we have that v(i) also doesn't correlate with any of the z(i).

Then we apply Lemma B.1 and conclude that there exists Aj+1 = U, Bj+1 = *V, b*j+1 = s such that TAj+1,bj+1,bj+1(v(i)) = 0 for i ≤ jk, TAj+1,bj+1,bj+1(z(i)) = v(i) − z(i) for *jk < i* ≤ (j + 1)k, and TAj+1,bj+1,bj+1(z(i)) = 0 for n ≥ *i >* (j + 1)k. These imply that

$$h^{(i)}_{j+1}=h^{(i)}_{j}+\mathcal{T}_{A_{j+1},b_{j+1},b_{j+1}}(v^{(i)})=v^{(i)}\quad\forall1\leq i\leq jk$$ $$h^{(i)}_{j+1}=h^{(i)}_{j}+\mathcal{T}_{A_{j+1},b_{j+1},b_{j+1}}(z^{(i)})=v^{(i)}\quad\forall jk+1\leq i\leq(j+1)k$$ $$h^{(i)}_{j+1}=h^{(i)}_{j}+\mathcal{T}_{A_{j+1},b_{j+1},b_{j+1}}(z^{(i)})=z^{(i)}\quad\forall(j+1)k<i\leq n$$

Therefore we constructed the $j+1$ layers that meets the inductive hypothesis for layer $j+1$. Therefore, by induction we get all the layers, and the last layer satisfies that $h^{(i)}_{t}=v^{(i)}$ for every example $i$.

Now we ready to prove Theorem 3.2, following the general plan sketched in Section 3. Proof of Theorem 3.2. We use formalize the intuition discussed below Theorem 3.2. First, take k = c(log n)/ρ2 for sufficiently large absolute constant c (for example, c = 10 works), by Johnson- Lindenstrauss Theorem (Johnson & Lindenstrauss (1984), or see Wikipedia (2016)) we have that when A0 is a random matrix with standard normal entires, with high probability, all the pairwise distance between the the set of vectors {0, x(1)*, . . . , x*(n)} are preserved up to 1 ± ρ/3 factor. That is, we have that for every i, 1−ρ/3 ≤ ∥A0x(i)∥ ≤ 1+ρ/3, and for every i ̸= j, ∥A0x(i)−A0x(j)∥ ≥ ρ(1 − ρ/3) ≥ 2ρ/3. Let z(i) = A0x(i) and ρ′ = ρ/3. Then we have z(i)'s satisfy the condition of Lemam B.2. We pick r random vectors q1*, . . . , q*r in Rk. Let v(1)*, . . . , v*(n) be defined as in equation (3.2). Then by Lemma B.2, we can construct matrices (A1, B1), . . . , (Aℓ, Bℓ) such that

$h_{\ell}^{(i)}=v^{(i)}$.

Note that v(i) ∈ {q1*, . . . , q*r}, and qi's are random unit vector. Therefore, the choice of α(1) = q1*, . . . , α*(r) = qr, β(1) = e1*, . . . , β*(r) = er, and satisfies the condition of Lemma B.1, and using Lemma B.1 we conclude that there exists Aℓ+1, Bℓ+1, sℓ+1 such that

$e_{j}=v_{j}+\mathcal{T}_{A_{\ell+1},B_{\ell+1},b_{\ell+1}}(v_{j}),\ \text{for every}\ j\in\{1,\ldots,r\}\,.$ (B.3)

By the definition of v(i) in equation (3.2) and equation (B.2), we conclude that ˆy(i) = h(i)
                                                                                           ℓ
                                                                                              +
TAℓ+1,Bℓ+1,bℓ+1(h(i)
                 ℓ ) = y(i)., which complete the proof.

## C Toolbox

In this section, we state two folklore linear algebra statements. The following Claim should be known, but we can't find it in the literature. We provide the proof here for completeness.

Claim C.1. Let U ∈ Rd×d be a real normal matrix (that is, it satisfies UU ⊤ = U ⊤U). Then, there exists an orthonormal matrix S ∈ Rd×d such that U = SDS⊤ , where D is a real block diagonal matrix that consists of blocks with size at most 2 × 2. Moreover, if d is even, then D consists of blocks with size exactly 2 × 2.

Proof. Since U is a normal matrix, it is unitarily diagonalizable (see
                                                 Weisstein (2016) for backgrounds).
 Therefore, there exists unitary matrix V in Cd×d and diagonal matrix in Cd×d such that U has eigen-decomposition U=
 V ΛV ∗.
 Since U itself is a real ma- trix, we have that the eigenvalues (the diagonal entries of Λ) come as conjugate pairs, and so do the eigenvectors (which are the columns of V ).
 That is, we can group the columns of V into pairs (v1, ¯v1), . . . , (vs, ¯vs), vs+1, . . . , vt, and let the corresponding eigenval- ues be λ1, ¯λ1, . . . , λλs, ¯λs, λs+1, . . . , λt.
 Here λs+1, . . . , λt ∈ R.
Then we get that U
                                                                          =
�s
  i=1 2ℜ(viλiv∗
           i ) + �t
         i=s+1 viλiv⊤
  i . Let Qi = ℜ(viλiv∗
           i ), then we have that Qi is a real matrix of rank-2. Let Si ∈ Rd×2 be a orthonormal basis of the column span of Qi and then we have that Qi can be written as Qi = SiDiS⊤
 i where Di is a 2 × 2 matrix. Finally, let S = [S1, . . . , Ss, vs+1, . . . , vt], and D = diag(D1, . . . , Ds, λs+1, . . . , λt) we complete the proof.

The following Claim is used in the proof of Theorem 2.2. We provide a proof here for completeness.

Claim C.2 (folklore). For any two matrices A, B ∈ Rd×d, we have that

∥AB∥F ≥ σmin(A)∥B∥F .

Proof. Since σmin(A)2 is the smallest eigenvalue of A⊤A, we have that

$$B^{\top}A^{\top}A B\succeq B^{\top}\cdot\sigma_{\operatorname*{min}}(A)^{2}\mathbf{Id}\cdot B\,.$$
Therefore, it follows that

$$\|AB\|_{F}^{2}=\text{tr}(B^{\top}A^{\top}AB)\geq\text{tr}(B^{\top}\cdot\sigma_{\min}(A)^{2}\text{Id}\cdot B)$$ $$=\sigma_{\min}(A)^{2}\text{tr}(B^{\top}B)=\sigma_{\min}(A)^{2}\|B\|_{F}^{2}\,.$$

Taking square root of both sides completes the proof.
"""