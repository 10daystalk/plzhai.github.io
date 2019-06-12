---
layout: post
title: "How do we understand Information Bottleneck?"
subtitle: 'A good measure to evaluate your network'
author: "Zpl"
header-style: text
tags:
  - Representation Learning
  - Deep Learning
---
>Despite the great success, there is still no comprehensive theoretical understanding of learning with Deep Neural Networks (DNNs) and their inner organization. *Tishby* has been trying to explain the effective of deep learning in the perspective of information bottleneck proposed in 1999.He think the goal of deep learning is trying to optimize the information bottleneck(IB) trade-oﬀ between the compression of the input and prediction
of the label,successively,for each layer,which they verified in recent work.

### Introduction to Information Bottleneck

It is believed that the effectiveness of the deep learning lies on the remarkable feature representation of the data ,which depends on the special architecture of neural networks. A fundamental problem in formalizing our intuitive ideas about information is to provide a quantitative notion of "meaningful" or "relevant" information.


The problem of extracting a relevant summary of data, a compressed description that captures only the relevant or meaningful information, is not well posed without a suitable definition of relevance. A typical example is that of speech compression. One can consider loss compression, but in any compression beyond the entropy of speech some components of the signal cannot be reconstructed. On the other hand, a transcript of the spoken words has much lower entropy (by orders of magnitude) than the acoustic waveform, which means that it is possible to compress (much) further without losing any information about the words and their meaning.

The standard analysis of lossy source compression is "rate distortion theory," which characterizes the trade-off between the rate, or signal representation size, and the average distortion of the reconstructed signal. Rate distortion theory determines the level of inevitable expected distortion, D,given the desired information rate, R, in terms of the rate distortion function R(D).

The main problem with rate distortion theory is in the need to specify the distortion function first, which in turn determines the relevant features of the signal. A possible solution comes from the fact that in many interesting cases we have access to an additional variable that determines what is relevant:extract the information from one variable that is relevant for the prediction of another one. The choice of additional variable determines the relevant components or features of the signal in each case.

Once we get the optimal representation,we can compare it with the representation of the hidden layer in the information plane () and evaluate the effect of the trained model,also the dynamics of the training.

#### Information Bottleneck method

##### Rate distortion theory


Let X denote the signal (message) space with a fixed probability measure p(x), and let $\tilede{X}$ denote its quantized codebook or compressed representation. For ease of exposition we assume here that both of these sets are finite, that is, a continuous space should first be quantized.

What determines the quality of a quantization? The first factor is of
course the rate, or the average number of bits per message needed to specify an element in the codebook without confusion. This number per element of X is bounded from below by the mutual information,

<img src="https://latex.codecogs.com/gif.latex?I(X;\tilde{X})&space;=&space;\sum_{x\in&space;X}&space;\sum_{\tilde{x}&space;\in&space;\tilde{X}&space;}p(x,\tilde{x})\log&space;[p(\tilde{x}|x)/p(\tilde{x})]" title="I(X;\tilde{X}) = \sum_{x\in X} \sum_{\tilde{x} \in \tilde{X} }p(x,\tilde{x})\log [p(\tilde{x}|x)/p(\tilde{x})]" />


However, information rate alone is not enough to characterize good quantization since the rate can always be reduced by throwing away details of the original signal x. We need therefore some additional constraints.

In rate distortion theory such a constraint is provided through a distortion function, d :X \times X \rightarrow $R^{+}$,which is presumed to be small for good representations. Thus the distortion function specifies implicitly what are the most relevant aspects of values in X.There is a monotonic tradeoff between the rate of the quantization and the expected distortion: the larger the rate, the smaller is the achievable distortion.

The celebrated rate distortion theorem of Shannon and Kolmogorov characterizes this tradeoff through the rate distortion function, R(D), defined as the minimal achievable rate under a given constraint on the expected distortion:

<img src="https://latex.codecogs.com/gif.latex?R(D)&space;\equiv&space;\min_{p(\tilde{x}|x):<d(x,\tilde{x})>&space;\leq&space;D}I(X;\tilde{X})" title="R(D) \equiv \min_{p(\tilde{x}|x):<d(x,\tilde{x})> \leq D}I(X;\tilde{X})" />

Finding the rate distortion function is a variational problem that can be solved by introducing a Lagrange multiplier, β, for the constrained expected distortion. One then needs to minimize the functional
\begin{equation*}

F[p(\tilde{x}|x] = I(X;\tilde{X}) + \beta <d(x,\tilde{x})>_{p(x,\tilde{x})}

\end{equation*}
over all normalized distributions$p(\tilde{x}|x)$.

\subsection{Information Bottleneck and Blahut Arimoto algorithm}
In the context of the representation learning of supervised tasks, we need to compress the source X like rate distortion problem and preserve the relevant information about the label variable. The relevance variable, denoted here by Y , must not be independent from the original signal X, namely they have positive mutual information I(X; Y ). It is assumed here that we have access to the joint distribution p(x; y), which is part of the setup of the problem, similarly to p(x) in the rate distortion case.\\
As before, we would like our relevant quantization $\tilde{X}$ to compress X as much as possible. In contrast to the rate distortion problem, however, we now want this quantization to capture as much of the information about Y as possible. The amount of information about Y in $\tilde{X}$ is given by
\begin{equation*}

I(\tilde{X};Y) = \sum_{y}\sum_{\tilde{x}}p(y,\tilde{x})\log p(y,\tilde{x})/p(y)p(\tilde{x}) \leq I(X;Y)

\end{equation*}
Obviously lossy compression cannot convey more information than the original data. As with rate and distortion, there is a tradeoff between compressing the representation and preserving meaningful information, and there is
no single right solution for the tradeoff. The assignment we are looking for is
the one that keeps a fixed amount of meaningful information about the relevant signal Y while minimizing the number of bits from the original signal
X (maximizing the compression). In effect we pass the information that X
provides about Y through a bottleneck" formed by the compact summaries
in $\tilde{X}$.\\
We can find the optimal assignment by minimizing the functional
\begin{equation*}

L[p(\tilde{x}|x)] = I(\tilde{X};X) - \beta I(\tilde{X};Y)

\end{equation*}
where $\beta$ is the Lagrange multiplier attached to the constrained meaningful information, while maintaining the normalization of the mapping $p(\tilde{x}|x)$ for every x. \\
In fact we can optimize on the information bottleneck function with $p(\tilde{x}|x)$ using lagrange multiplier method.We can take the derivative of the lagrange function on $p(\tilde{x}|x)$ and let the derivative equal to 0,which is
\begin{equation*}

p(\tilde{x}|x) = p(\tilde{x})/Z(x,\beta) \exp [-\beta \sum_{y} p(y|x) \log p(y|x)/p(y|\tilde{x})],

\end{equation*}
where Z(x,\beta) is a normalization constantand the distribution $p(y|\tilde{x})$ in the exponent is given via Bayes’ rule,as,
\begin{equation}

p(y|\tilde{x}) = \sum_{x\in X}p(y|x)p(x|\tilde{x}),
p(\tilde{x}) = \sum_{x}p(\tilde{x}|x)p(x).

\end{equation}
As for the BA algorithm, the self consistent equations suggest a natural method for finding the unknown distributions, at every value of $\beta$. Indeed, these equations can be turned into converging, alternating iterations among the three convex distribution sets, $p(\tilde{x}|x),p(\tilde{x}),p(y|\tilde{x})$.The algorithm can be stated as the following:

\begin{algorithm}[h]
		\caption{BA algorithm on Information Bottleneck}
		\hspace*{0.02 in} {\bf Input:} p(x,y)
		\hspace*{0.02 in} {\bf Output:} $p(\tilde{x}|x),p(\tilde{x}),p(y|\tilde{x})$
		\hspace*{0.02 in} {\bf Initialization:} $p(\tilde{x}),p(y|\tilde{x}),t = 0$
		\begin{algorithmic}[1]
			
			\While {IB $\leq \epsilon$}
			\State $p_{t}(\tilde{x}|x) = p_{t}(\tilde{x})/Z_{t}(x,\beta) \exp(-\beta d(x,\tilde{x}))$;
			\State $p_{t+1}(\tilde{x}) = \sum_{x}p(x)p_{t}(\tilde{x}|x)$;
			\State $p_{t+1}(y|\tilde{x}) = \sum_{y}p(y|x)p_{t}(x|\tilde{x})$;
			\State $t = t + 1$
			\EndWhile
		\end{algorithmic}
	\end{algorithm}
The formal solution of the self consistent equations, described above, still requires a specification of the structure and cardinality of $\tilde{X}$, as in rate distortion theory. For every value of the Lagrange multiplier $\beta$ there are corresponding values of the mutual information $I_{X} \equiv I(X;\tilde{X}), and I_Y \equiv I(\tilde{X};Y) $ for every choice of the cardinality of $\tilde{X}$.

\section{3.Information Bottleneck principle of Deep Learning}
We have studied that the level of the compression and relevance that the optimal representation can get to.Now we consider the structure of Deep Neural Networks and propose a new principle to evaluate the extracted layer representation.\\DNNs are comprised of multiple layers of artificial neurons, or simply units, and are known for their remarkable performance in learning useful hierarchical representations
of the data for various machine learning tasks. Typically, the input, denoted by X, is a high dimensional variable, being a low level representation of the data such as pixels of an image, whereas the desired output, Y , has a significantly lower dimensionality of the predicted categories. This generally means that most of the entropy of X is not very informative about Y , and that the relevant features in X are highly distributed and difficult to extract. The remarkable success of DNNs in learning to extract such features is mainly attributed to the sequential processing of the data, namely that each hidden layer operates as the input to the
next one, which allows the construction of higher level distributed representations.\\
\begin{figure} [!htbp]
	\centering
	% Requires \usepackage{graphicx}
	\includegraphics[width=11.00cm]{./DNN.jpg}
\end{figure}\\[-40pt]
As depicted in figure 1, each layer in a DNN processes inputs only from the previous layer, which means that the network layers form a Markov chain. An immediate consequence of the Data Processing Inequality is that information about Y that is lost in one layer cannot be recovered in higher layers. Namely, for any i $\geq$ j it holds that
\begin{equation}

I(Y;X) \geq I(Y,h_j) \geq T(Y;h_i) \geq I(Y;\hat{Y}).

\end{equation}
Achieving the equality above is possible if and only if each layer is a sufficient statistic of its input. By requiring not only the most relevant representation at each layer, but also the
most concise representation of the input, each layer should attempt to maximize $I(Y;h_i)$ while minimizing $I(h_{i−1};h_i)$ as much as possible.\\
Here we consider $I(Y;\hat{Y})$ as the natural quantifier of the quality of the DNN, as it measures precisely how much of the predictive features in X for Y is captured by the model. Reducing $I(h_{i−1};h_i)$ also has a clear learning theoretic interpretation as the minimal description length of the layer.\\
The information distortion of the IB principle provides a new measure of optimality which can be applied not only for the output layer, as done when evaluating the performance of DNNs with other distortion or error measures, but also for evaluating the optimality of each hidden layer or unit of the network. Namely, each layer can be compared to the optimal IB limit for some $\beta$,
\begin{equation}

I(h_{i-1;i}) + \beta I(Y;h_{i-1}|h_i)

\end{equation}
 This optimality
criterion also give a nice interpretation of the construction of higher level representations along the network. Since each point on the information curve is uniquely defined by $\beta$,shifting from low to higher level representations is analogous to successively decreasing β. Notice that other cost functions,
such as the squared error, are not applicable for evaluating the optimality of the hidden layers, nor can they account for multiple levels of description.\\
The theoretical IB limit and the limitations that are imposed by the DPI on the flow of information between the
layers, gives a general picture as to to where each layer of a trained network can be on the information plane. The input level clearly has the least IB distortion, and requires the longest description (even after dimensionality reduction, X is the lowest representation level in the network). Each consecutive layer can only increase the IB distortion level, but it also compresses its inputs, hopefully eliminating only irrelevant information. \\
SO we can compute the IB functions layer by layer and evaluate their performance.Ulteriorly we can visualize the dynamics of the training process and information flow along the layers,which can give a profile to the optimization and goal of DNN.The experiments will be in section 4.

\section{4.Experiment}
As we said in the last section,our two order parameters, I(T;X) and I(T;Y), allow us to visualize and compare different network architectures in terms of their efficiency in preserving the relevant information in P(X; Y ).
By visualizing the paths of different networks in the information plane we explore the following fundamental issues:\\
1. The SGD layer dynamics in the Information plane.\\
2. The effect of the training sample size on the layers.\\
3. What is the benefit of the hidden layers?\\
4. What is the final location of the hidden layers?\\
5. Do the hidden layers form optimal IB representations?\\
\subsection{The two optimization phases in the Information Plane}
As can be seen, at the beginning of the optimization the deeper layers of the randomly-initialize
network fail to preserve the relevant information, and there is a sharp decrease in $I_Y$ along the
path. During the SGD optimization the layers first increase $I_Y$ , and later significantly decrease $I_X$,
thus compressing the representation. Another striking observation is that the layers of the different
randomized networks seem to follow very similar paths during the optimization and eventually converge to nearby points in the information plane.\\
\begin{figure} [!htbp]
	\centering
	% Requires \usepackage{graphicx}
	\includegraphics[width=11.00cm]{./figure2.jpg}
\end{figure}\\[-40pt]
The two optimization phases are clearly visible in all cases. During
the fast - ERM - phase, which takes a few hundred epochs, the layers increase the information on
the labels (increase $I_Y$ ) while preserving the DPI order (lower layers have higher information). In
the second and much longer training phase the layers’ information on the input, $I_X$, decreases and
the layers lose irrelevant information until convergence (the yellow points). We call this phase the
representation compression phase.
\begin{figure} [!htbp]
	\centering
	% Requires \usepackage{graphicx}
	\includegraphics[width=11.00cm]{./figure3.jpg}
\end{figure}\\[-40pt]
\subsection{The computational benefit of the hidden layers}
We now turn to one of the fundamental questions about Deep Learning - what is the benefit of the hidden layers?\\
o address this, we trained 6 different architectures with 1 to 6 hidden layers (with layers as in
Figure 4), trained on 80$%$ the patterns, randomly sampled. As before, we repeated each training 50 times with randomized initial weights and training samples. Figure 5 shows the information plane paths for these 6 architectures during the training epochs, each averaged over the randomized networks.\\
There are several important outcomes of this experiment:\\
1. Adding hidden layers dramatically reduces the number of training epochs for good generalization.\\
2. The compression phase of each layer is shorter when it starts from a previous compressed layer.\\
3. The compression is faster for the deeper (narrower and closer to the output) layers.\\
It is clear from panels 4, 5 and 6 that the first hidden layer (of width 12) remains in the upper
right corner, without lose of information on either X or Y . Our simulations suggest that such 1-1
transformations do not help learning, since they do not generate compressed representations. In our simulations all the hidden layers eventually compress the inputs, given enough SGD epochs.\\
\begin{figure} [!htbp]
	\centering
	% Requires \usepackage{graphicx}
	\includegraphics[width=11.00cm]{./figure5.jpg}
\end{figure}\\[-40pt]
\subsection{Convergence to the layers to the Information Bottleneck bound}
In order to quantify the IB optimality of the layers we tested whether the converged layers satisfied
the encoder-decoder relations, for some value of the Lagrange multiplier β. For each converged layer we used the encoder and decoder distributions based on the layer neurons’ quantized
values, $p_i(t|x)$ and $p_i(y|t$) with which we calculated the information values $(I_{X} ^{i}, I_{Y} ^{i})$.\\

In Figure 6 we plot the information plane with the layers’ information values $(I_{X} ^{i}, I_{Y} ^{i})$ and the
IB information curve (blue line). The 5 empirical layers (trained with SGD) lie remarkably close to
the theoretical IB limit, where the slope of the curve, $\beta ^{-1}$, matches their estimated optimal \bata_{i} ^{*}.\\
Hence, the DNN layers’ encoder-decoder distributions satisfy the IB self-consistent equations
within our numerical precision, with decreasing β as we move to deeper layers. The error bars are
calculated over the 50 randomized networks. As predicted by the IB equations, near the information
curve.
\begin{figure} [!htbp]
	\centering
	% Requires \usepa$$ckage{graphicx}
	\includegraphics[width=11.00cm]{./figure6.jpg}
\end{figure}\\[-40pt]
\subsection{Discussion}
Our numerical experiments were motivated by the Information Bottleneck framework. We demonstrated that the visualization of the layers in the information plane reveals many - so far unknown - details about the inner working of Deep Learning and Deep Neural Networks. They revealed the distinct phases of the SGD optimization, drift and diffusion, which explain the ERM and the representation compression trajectories of the layers. The stochasticity of SGD methods is usually motivated as a way of escaping local minima of the training error. In this paper we give it a new, perhaps much more important role: it generates highly efficient internal representations through compression by diffusion. This is consistent with other recent suggestions on the role of noise in Deep Learning.\\
\section{Conclusion}
We develop a new principle to measure the extent that how good is the representation and we use this principle to compute the best representation should be given data X and label Y.Then we use this measure to evaluate the layer representation and prove that different layers can almost get best representation under different tradeoff parameter $\beta$.\\
However, the IB consistent algorithm seems to be an elegant algorithm ,but in the continuous and high-dimensional case it is out of work.
On the other hand,the computation of mutual information here is limited to low-dimensional data and discrete case.So we want to improve these situations in the future work.\\
\section{Acknowledgement}
Thanks to professor Zhang to give an interesting class,in which I get lots of new perspectives into deep learning.
In the process of the selection of the topic,professor Zhang give me many beneficial advice, which help me finish my final report.



\begin{figure} [!htbp]
	\centering
	% Requires \usepackage{graphicx}
	\includegraphics[width=11.00cm]{./video seperation.png}
	\caption{\small{from the video seperating the moving objects}}
\end{figure}\\[-40pt]

\begin{flushleft}
REFERENCES
\end{flushleft}



Liam Paninski. Estimation of entropy and mutual information. Neural Comput., 15(6):1191–1253, June 2003. ISSN 0899-7667. doi: 10.1162/089976603321780272.\\
Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In Information Theory Workshop (ITW), 2015 IEEE, pages 1–5. IEEE, 2015.\\
Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In Proceedings of the 37-th Annual Allerton Conference on Communication, Control and Computing, 1999.\\
Ravid Schwartz-Ziv,Naftali Tishby.Opening the black box of Deep Neural Networks via Information.arxiv preprint 1703.00810,2017.\\
Andrew M. Saxe, Yamini Bansal, Joel Dapello, Madhu Advani,Artemy Kolchinsky, Brendan D. Tracey.ON THE INFORMATION BOTTLENECK THEORY OF DEEP LEARNING.Published as a conference paper at ICLR 2018.\\
