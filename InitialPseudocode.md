Pseudocode for challenge approach:


train = readcsv(train.csv)

Data exploration:
- Scatterplot of phrase length and sentiment
- Pie graph of # of reviews with each sentiment 


Preprocessing:
    split features and target:
        y_train = train[sentiment]
        X_train = train.drop(sentiment)

    remove stopwords:
    
    text vectorization:
        vectizer = CountVectorizer(ngram_range = (some tuple))
        vectizer.fit(X_train[Phrase])
        vectized_X = vectizer.transform(X_train[Phrase])) # vectized_X will be the matrix for the n-gram range specified
        
    tf-itf, term frequency inverse term frequency:
        tfitf = TfidfTransformer().fit(vectized_X)
        tf_X = tfitf.transform(vectized_X)
        
Modeling: put in def, so we can perform it on the unigram, bigram, tf-itf and not have to repeat code
          Also perform grid search hyperparameter tuning in here and return best estimator

def model:
    parameters: x_train, target

    split into validation and training data with test_train_split
    instantiate clf = SGDClassifier()
    clf.train(X_train, target)
    perform grid search
    print scores on validation data
    return best estimator 
    
call model(unigram)
call model(bigram)
call model(unigram tfitf)
call model(bigram tfif)



Final evaluation:

y_pred = best_model(X_test)
print score(y_test, y_pred)
