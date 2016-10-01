# DistributionalModels
Distributional Models and tools for refining notions of similarity in such models.

We present the Distributional Sentiment Hypothesis for detecting synonymy and antonymy in distributional models. The Distributional Sentiment Hypothesis states that synonyms and antonyms tend to occur in similar narrow distributional contexts, but are distinguished by their broader tonal contexts. We show that sentimental polarity features computed using standard sentiment analysis tool-kits outperform pattern-based and narrow-context approaches in the literature and thus affirm the validity of this hypothesis. We also introduce an unsupervised method based on this hypothesis. 

We develop and evaluate a methodology based upon a distributional space constructed according to the distributional sentiment hypothesis. We use features pulled from this distributional space to train a classifier and test our hypothesis by comparing the performance of classification using these features to that using features proposed by other methods in the literature.

############################################################
# Contents                                                 #
############################################################

Corpora
	gutenbergspider.sh - Script for acquiring Gutenberg
			     corpus.