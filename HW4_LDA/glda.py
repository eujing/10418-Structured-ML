import numpy as np
from scipy.stats import dirichlet, multivariate_normal, multinomial
from utils import *
import json
from tqdm import tqdm
import pdb

class GaussianLDA(object):
    """Object that encapsulates Gaussian LDA algorithm"""
    def __init__(self, n_topics):
        #Set up variables used throughout process.
        self.n_topics = n_topics

        embeddings, word2index = load_embeddings()
        self.embeddings = embeddings
        self.emb_size = embeddings.shape[1]
        self.word2index = word2index
        self.index2word = {i: word for word, i in word2index.items()}

        self.corpus = load_corpus().astype(np.int64)
        self.n_docs = self.corpus.shape[0]
        self.n_words = self.corpus.shape[1]

        # Randomly assign words to topics
        self.doc_word_assignment = {}
        self.doc_words = []
        idx = np.arange(self.n_words)
        for d in range(self.n_docs):
            counts = self.corpus[d]
            self.doc_words.append(np.repeat(idx, counts))
            for t in range(len(self.doc_words[d])):
                self.doc_word_assignment[d, t] = np.random.choice(self.n_topics)

        # Count document / topic ocurrences
        self.doc_topic_counts = np.zeros((self.n_docs, self.n_topics))
        for (d, _), topic in self.doc_word_assignment.items():
            self.doc_topic_counts[d, topic] += 1

        #Set up prior parameters
        self.alpha = np.ones(self.n_topics) * 10
        self.nu = self.emb_size
        self.mu = self.embeddings.mean(axis=0)
        self.kappa = 0.01

        #Set up topic parameters
        self.topic_emb_sums = np.zeros((self.n_docs, self.n_topics, self.emb_size))
        self.topic_mus = np.zeros((self.n_topics, self.emb_size))
        self.topic_kappas = np.zeros((self.n_topics, ))

        #Initialize Topic Parameters
        for (d, t), topic in self.doc_word_assignment.items():
            word = self.doc_words[d][t]
            self.topic_emb_sums[d, topic, :] += self.embeddings[word, :]

        self.update_topic_parameters()

        assert self.topic_kappas.shape == (self.n_topics, )
        assert self.topic_mus.shape == (self.n_topics, self.emb_size)

        # Go through every word and calculate Full Conditionals
        log_likes = []
        log_like = self.log_likelihood()
        log_likes.append(log_like)
        print(f"Initial log-likelihood = {log_like}")
        for i in range(50):
            changes = 0
            for (d, t), topic in tqdm(self.doc_word_assignment.items()):
                word = self.doc_words[d][t]
                embed = self.embeddings[word, :]

                # Remove current assignment from statistisc
                self.doc_topic_counts[d, topic] -= 1
                self.topic_emb_sums[d, topic, :] -= embed

                # Calculate Posterior Parameters
                topic_counts = self.doc_topic_counts.sum(axis=0)
                topic_kappas = self.kappa + topic_counts
                topic_mus = (self.kappa * self.mu + self.topic_emb_sums.sum(axis=0)) / topic_kappas.reshape(-1, 1)
                dofs = self.nu + topic_counts - self.emb_size + 1

                assert topic_counts.shape == (self.n_topics, )
                assert topic_kappas.shape == (self.n_topics, )
                assert topic_mus.shape == (self.n_topics, self.emb_size)
                assert dofs.shape == (self.n_topics, )

                # Calculate Full Conditionals
                topic_probs = np.zeros(self.n_topics)
                for k in range(self.n_topics):
                    log_prob = np.log(self.doc_topic_counts[d, k] + self.alpha[k]) + \
                            multivariate_t_distribution(embed, topic_mus[k], dofs[k], self.emb_size)
                    topic_probs[k] = log_prob

                # Normalize topic probabilities w/ numerical stability
                topic_probs -= np.max(topic_probs)
                topic_probs = np.exp(topic_probs)
                # Make sure probs do not sum to > 1 after normalization
                # topic_probs = topic_probs / (np.sum(topic_probs, axis=0) + 1e-7)
                topic_probs = topic_probs / np.sum(topic_probs, axis=0)

                # Sample and update statistisc
                new_topic = np.argmax(multinomial.rvs(1, topic_probs))
                self.doc_topic_counts[d, new_topic] += 1
                self.topic_emb_sums[d, new_topic, :] += embed
                self.doc_word_assignment[d, t] = new_topic
                if new_topic != topic:
                    changes += 1

            # TODO: Update posterior parameters
            self.update_topic_parameters()

            if (i + 1) % 10 == 0:
                log_like = self.log_likelihood()
                log_likes.append(log_like)
                print(f"Iteration {i}: Log-likelihood = {log_like}, Changed = {changes}/{self.corpus.sum()}")
                self.print_top()

        with open("results.json", "w") as f:
            json.dump({"log_likes": log_likes}, f)


    def log_likelihood(self):
        #Return the log-likelihood of the joint assignment of topics and words under G-LDA

        # Add pseudo count to all so no document has a 0 count for some topic
        doc_topic_counts = self.doc_topic_counts + 0.1
        doc_topic_prob = doc_topic_counts / doc_topic_counts.sum(axis=1).reshape(-1, 1)
        result = 0
        for d in range(self.n_docs):
            logp_theta = dirichlet.logpdf(doc_topic_prob[d, :], self.alpha)
            result += logp_theta

        for topic in range(self.n_topics):
            logp_mu = multivariate_normal.logpdf(
                    self.topic_mus[topic], self.mu, np.eye(self.emb_size) / self.kappa)
            result += logp_mu

        for (d, t), topic in self.doc_word_assignment.items():
            word = self.doc_words[d][t]
            embed = self.embeddings[word, :]

            logp_embedding = multivariate_normal.logpdf(embed, self.topic_mus[topic])
            logp_topic = np.log(doc_topic_prob[d, topic])

            result += logp_embedding + logp_topic

        return result / len(self.doc_word_assignment)


    def update_topic_parameters(self):
        #Update topic parameters for topic k
        topic_counts = self.doc_topic_counts.sum(axis=0)
        self.topic_kappas = self.kappa + topic_counts
        self.topic_mus = (self.kappa * self.mu + self.topic_emb_sums.sum(axis=0)) / self.topic_kappas.reshape(-1, 1)


    def print_top(self, n=10):
        #Print top n words for each topic
        for topic in range(self.n_topics):
            embed_probs = multivariate_normal.pdf(self.embeddings, self.topic_mus[topic])
            top_word_idx = np.argsort(embed_probs)
            top_word_idx = top_word_idx[-n:]
            top_words = [self.index2word[idx] for idx in top_word_idx]
            print(f"Topic {topic}: {top_words}")

        return None

    def sample(self):
        #Do calculation of parameters and sample from posterior
        return None

if __name__ == "__main__":
    glda = GaussianLDA(5)
