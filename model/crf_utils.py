import matplotlib.pyplot as plt
from sklearn_crfsuite import CRF

plt.style.use('ggplot')

class Custom_CRF(CRF):
    def predict_proba(self, X):
        return self.predict_marginals(X)


def word2features(sent,
                  i,
                  pos_features,
                  ent_type_features,
                  lemma_features,
                  is_features,
                  position_features,
                  bias,
                  begin,
                  end):
    features = {}
    if bias:
        features['bias'] = 1.0
    if position_features:
        features['AbsolutePosition'] = i
        features['RelativePosition'] = i/len(sent)
        features['QuatilePosition'] = int(4*(i/len(sent)))+1
    if sent[i].is_space:
        features['Whitespace'] = True
    else:
        for n in range(begin, end+1):
            if i + n <= 0:
                features['{} BOS'.format(n)] = True
            elif i + n >= len(sent):
                features['{} EOS'.format(n)] = True
            else:
                if sent[i+n].is_space:
                    features['{}_Whitespace'.format(n)] = True
                else:
                    word = sent[i+n]
                    if pos_features:
                        features['{}:word.pos_'.format(n)] = word.pos_
                        features['{}:word.tag_'.format(n)] = word.tag_
                        features['{}:word.dep_'.format(n)] = word.dep_
                    if ent_type_features:
                        if not word.ent_type_ == "":
                            features['{}:word.ent_type'.format(n)] = word.ent_type_
                        features['{}:word.ent_iob_'.format(n)] = word.ent_iob_
                    if lemma_features:
                        features['{}:word.lemma'.format(n)] = word.lemma_
                    if is_features:
                        features.update({
                            '{}:word.is_alpha()'.format(n): word.is_alpha,
                            '{}:word.is_ascii()'.format(n): word.is_ascii,
                            '{}:word.is_digit()'.format(n): word.is_digit,
                            '{}:word.is_lower()'.format(n): word.is_lower,
                            '{}:word.is_upper()'.format(n): word.is_upper,
                            '{}:word.is_title()'.format(n): word.is_title,
                            '{}:word.is_punct'.format(n):word.is_punct,
                            '{}:word.is_left_punct'.format(n):word.is_left_punct,
                            '{}:word.is_right_punct'.format(n):word.is_right_punct,
                            '{}:word.is_space'.format(n):word.is_space,
                            '{}:word.is_bracket'.format(n):word.is_bracket,
                            '{}:word.is_quote'.format(n):word.is_quote,
                            '{}:word.is_currency'.format(n):word.is_currency,
                            '{}:word.like_url'.format(n):word.like_url,
                            '{}:word.like_num'.format(n):word.like_num,
                            '{}:word.like_email'.format(n):word.like_email,
                            '{}:word.is_oov'.format(n):word.is_oov,
                            '{}:word.is_stop'.format(n):word.is_stop,
                        })
    return features

def text2features(sent,
                  pos_features=True,
                  ent_type_features=True,
                  lemma_features=True,
                  is_features=True,
                  position_features=True,
                  bias=True,
                  begin=-1,
                  end=1):
    return [word2features(sent,
                          i,
                          pos_features,
                          ent_type_features,
                          lemma_features,
                          is_features,
                          position_features,
                          bias,
                          begin,
                          end
                          ) for i in range(len(sent))]

def visualize_rs_result(randomized_search_cv):
    _x = [s['c1'] for s in randomized_search_cv.cv_results_["params"]]
    _y = [s['c2'] for s in randomized_search_cv.cv_results_["params"]]
    _c = [s for s in randomized_search_cv.cv_results_["mean_test_score"]]

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    axis = plt.gca()
    axis.set_yscale('log')
    axis.set_xscale('log')
    axis.set_xlabel('C1')
    axis.set_ylabel('C2')
    axis.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    axis.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])

    print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    print('best params:', randomized_search_cv.best_params_)
    print('best CV score:', randomized_search_cv.best_score_)
    print('model size: {:0.2f}M'.format(randomized_search_cv.best_estimator_.size_ / 1000000))

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
