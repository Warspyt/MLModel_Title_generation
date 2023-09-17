import stanfordnlp
stanfordnlp.download('en')

nlp = stanfordnlp.Pipeline(lang='en')