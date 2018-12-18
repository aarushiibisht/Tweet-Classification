import sys
import os


class TweetClassifier:
    total_tweets = 0
    label_probs = {}
    word_prob = {}
    unique_words_per_label = {}

    def _total_tweets(self, training_data):
        for k in training_data.keys():
            self.total_tweets += len(training_data[k])

    def _populate_label_probs(self, training_data):
        for k in training_data.keys():
            self.label_probs[k] = len(training_data[k]) / self.total_tweets

    def _update_unique_words_for_label(self, label):
        if label in self.unique_words_per_label:
            self.unique_words_per_label[label] += 1
        else:
            self.unique_words_per_label[label] = 1

    def _populate_word_prob(self, training_data):
        for k in training_data.keys():
            no_of_t_per_loc = len(training_data[k])
            for i in range(0, no_of_t_per_loc):
                tweet = training_data[k][i].split()
                for word in tweet:
                    if word + "-" + k in self.word_prob:
                        self.word_prob[word + "-" + k] += 1 / no_of_t_per_loc
                    else:
                        self.word_prob[word + "-" + k] = 1 / no_of_t_per_loc
                        self._update_unique_words_for_label(k)

    def train(self, training_data):
        self._total_tweets(training_data)
        self._populate_label_probs(training_data)
        self._populate_word_prob(training_data)

    def _get_laplace_smoothed_prob(self, word, label):
        word_count = 0
        alpha = 0.01
        if word + "-" + label in self.word_prob:
            word_count = self.word_prob[word + "-" + label] * self.label_probs[label] * self.total_tweets
        return (word_count + alpha) / (
                (self.label_probs[label] * self.total_tweets) + alpha * self.unique_words_per_label[label])

    def _predict_label_for_tweet(self, tweet):
        predicted_label = None
        max_prob = 0
        bag_of_words = tweet.split()
        for k in self.label_probs.keys():
            numerator = 1
            for word in bag_of_words:
                numerator = numerator * self._get_laplace_smoothed_prob(word, k)
            numerator = numerator * self.label_probs[k]
            if numerator > max_prob:
                predicted_label = k
                max_prob = numerator
        return predicted_label

    def _predict_label_for_tweets(self, test_data):
        predicted_labels_for_tweets = []
        for k in test_data.keys():
            for i in range(0, len(test_data[k])):
                tweet = test_data[k][i]
                predicted_labels_for_tweets.append(self._predict_label_for_tweet(tweet))
        return predicted_labels_for_tweets

    def test(self, test_data):
        labels = self._predict_label_for_tweets(test_data)
        return labels

    def _calculate_accuracy(self, predicted_labels, test_data):
        correct_prediction = 0
        total_tweets = 0
        pred_count = 0
        for k in test_data.keys():
            total_tweets += len(test_data[k])
            for i in range(0, len(test_data[k])):
                if k == predicted_labels[pred_count]:
                    correct_prediction += 1
                pred_count += 1
        return correct_prediction / total_tweets

    def accuracy(self, test_data):
        predicted_labels = self.test(test_data)
        return self._calculate_accuracy(predicted_labels, test_data)


def get_tweets_and_labels(filename):
    labels_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            split_ = l.split(maxsplit=1)
            if len(split_) != 2:
                continue
            c, t = split_[0], split_[1].strip()
            if c in labels_dict:
                labels_dict[c].append(t)
            else:
                labels_dict[c] = [t]
    return labels_dict


if __name__ == "__main__":

    # Check if the file is being called with correct parameters
    error_msg = "Please give arguments as follows: %s <training_file> <testing_file> <output_file>" % sys.argv[0]
    if len(sys.argv) != 4:
        print("Illegal arguments!")
        print(error_msg)
        exit(1)
    if not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Test or training file not found!")
        print(error_msg)
        exit(1)

    # read the training data
    training_data = get_tweets_and_labels(sys.argv[1])
    test_data = get_tweets_and_labels(sys.argv[2])

    classifier = TweetClassifier()
    classifier.train(training_data)
    print("Accuracy on test data:", classifier.accuracy(test_data))
