import pickle
import pandas as pd

if __name__ == '__main__':
    predictor = pickle.load(open('final_model', 'rb'))
    testing = pd.read_csv('./test.tsv', sep='\t', header=0)
    test_data = testing.Phrase
    final_pred = pd.read_csv(r'sampleSubmission.csv', sep=',')
    final_pred.Sentiment = predictor.predict(test_data)
    final_pred.to_csv(r'results.csv', sep=',', index=False)
