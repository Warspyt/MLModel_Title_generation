import stanfordnlp

""" Instalacion de los modulos de la libreria """
# stanfordnlp.download('en')

""" Configuracion de los procesadores """
nlp = stanfordnlp.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')