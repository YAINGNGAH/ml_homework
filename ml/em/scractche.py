class WordAlignerZeroToken(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words+1
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words+1, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []
        for sentence_pair in parallel_corpus:
            source = np.append(np.array([0]), sentence_pair.source_tokens+1)
            target = sentence_pair.target_tokens
            rows, cols = np.ix_(source, target)
            q = self.translation_probs[rows, cols]
            q = q / np.sum(q, axis=0)
            posteriors.append(q)

        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbo = 0
        thr = 1e-12
        for pair, prob in zip(parallel_corpus, posteriors):
            source = np.append(np.array([0]), pair.source_tokens+1)
            target = pair.target_tokens
            rows, cols = np.ix_(source, target)
            elbo += np.sum(np.log(self.translation_probs[rows, cols] + thr) * prob)
            elbo -= target.shape[0] * np.log(source.shape[0]+1)
            elbo -= np.sum(np.log(prob + thr) * prob)
        return elbo

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs *= 0
        for pair, prob in zip(parallel_corpus, posteriors):
            source = np.append(np.array([0]), pair.source_tokens+1)
            target = pair.target_tokens
            rows, cols = np.ix_(source, target)
            np.add.at(self.translation_probs, (rows, cols), prob)

        self.translation_probs /= np.sum(self.translation_probs, axis=1)[:, None]
        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences, bias=0):
        result = []
        posteriors = self._e_step(sentences)
        for (sentence, posterior) in zip(sentences, posteriors):
            alignment = []
            for (i, target_token) in enumerate(sentence.target_tokens, 1):
                if np.max(posterior[:, i - 1]) > bias:
                    j = np.argmax(posterior[:, i - 1])
                    if j != 0:
                        alignment.append((j, i))
            result.append(alignment)
        return result
