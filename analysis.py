import os, io, string, logging
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

IGNORE_FILENAMES = set([".Ulysses-Group.plist"])
ACCEPT_FILETYPES = set([".txt", ".md"])

STOP_WORDS = set(["a", "about", "above", "above", "across", "after", "afterwards", 
    "again", "against", "all", "almost", "alone", "along", "already", "also","although",
    "always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", 
    "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  
    "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", 
    "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", 
    "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", 
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", 
    "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", 
    "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", 
    "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", 
    "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", 
    "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", 
    "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", 
    "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", 
    "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", 
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", 
    "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", 
    "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", 
    "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", 
    "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", 
    "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", 
    "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", 
    "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", 
    "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", 
    "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", 
    "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", 
    "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", 
    "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", 
    "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", 
    "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", 
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", 
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", 
    "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "an", "and", 
    "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", 
    "on", "that", "the", "to", "was", "were", "will", "with", "it's", "i", "i'm", "a", "you", 
    "not", "this", "what", "if", "have", "or", "your", "do", "can", "we", "more", "1", "so", 
    "one", "they", "2", "am", "don't", ""])

class NoteHelper(object):
    def __init__(self, target_path, filename, num_topics=5, initialize=True):
        self.target_path = target_path
        self.dic_path = "tmp/" + filename + ".dict"
        self.corpus_path = "tmp/" + filename + ".mm"
        self.index_path = "tmp/" + filename + ".index"
        self.num_topics = num_topics
        self.initialize = initialize
        self.texts = None
        self.pathToIndex = None
        self.indexToPath = None        
        self.dictionary = None
        self.corpus = None
        self.lsi = None
        self.index = None

        self.initializeTexts()

    # return list of lists of words in each document
    def initializeTexts(self):
        self.texts = []
        self.pathToIndex = {}
        self.indexToPath = []
        for (dirpath, dirnames, filenames) in os.walk(self.target_path):
            for f in filenames:
                for af in ACCEPT_FILETYPES:
                    if str(f).endswith(af) and f not in IGNORE_FILENAMES:
                        path = os.path.join(dirpath, f)
                        with io.open(path, mode="r", encoding="utf-8") as f:
                            lines = f.readlines()
                            doc = [word.lower().strip(string.punctuation).encode('ascii','ignore') for line in lines for word in line.split()]
                            strip_doc = [word for word in doc if word not in STOP_WORDS]
                            self.texts.append(strip_doc)
                            index = len(self.texts) - 1
                            self.pathToIndex[path] = index
                            self.indexToPath.append(path)

    # return cleaned and formatted similarities
    def formatSimilarities(self, similarities, filename, limit_query):
        path_similarities = [(self.indexToPath[i], sim) for i, sim in enumerate(similarities)]
        sorted_similarities = sorted(path_similarities, key=lambda sim: sim[1], reverse=True)
        top_similarities = [sim for sim in sorted_similarities if sim[1] > 0 and sim[0] != filename]
        if len(top_similarities) > limit_query:
            top_similarities = top_similarities[:limit_query]
        else:
            top_similarities = top_similarities
        pprint(top_similarities)

    def initializeDictionary(self):
        self.dictionary = corpora.Dictionary(self.texts)
        self.dictionary.save(self.dic_path)

    def loadDictionary(self):
        self.dictionary = corpora.Dictionary.load(self.dic_path)

    def initializeCorpus(self):
        if self.dictionary is None:
            self.initializeDictionary() if self.initialize else self.loadDictionary() 
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        corpora.MmCorpus.serialize(self.corpus_path, self.corpus)

    def loadCorpus(self):
        if self.dictionary is None:
            self.initializeDictionary() if self.initialize else self.loadDictionary()         
        self.corpus = corpora.MmCorpus(self.corpus_path)

    def initializeLsiModel(self):
        if self.corpus is None:
            self.initializeCorpus() if self.initialize else self.loadCorpus() 
        self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics, power_iters=10)
        #lsi.print_topics(num_topics=5, num_words=20)

    def initializeIndex(self):
        if self.lsi is None:
            self.initializeLsiModel()
        self.index = similarities.MatrixSimilarity(self.lsi[self.corpus])
        self.index.save(self.index_path)

    def loadIndex(self):
        if self.lsi is None:
            self.initializeLsiModel()        
        self.index = similarities.MatrixSimilarity.load(self.index_path)

    def query(self, filename, limit_query=10):
        print "querying for " + str(limit_query) + " most similar documents to " + filename + "..."
        if self.index is None:
            self.initializeIndex() if self.initialize else self.loadIndex()
        doc_index = self.pathToIndex[filename]
        query = self.lsi[self.corpus[doc_index]]
        similarities = self.index[query]
        self.formatSimilarities(similarities, filename, limit_query)


current_dir = '/Users/Merry/Dropbox/Documents/Notes/unotes/' # os.path.dirname(os.path.realpath(__file__))
notehelper = NoteHelper(current_dir, 'notes', num_topics=50, initialize=False)
notehelper.query('/Users/Merry/Dropbox/Documents/Notes/unotes/Inbox/to do.md', limit_query=10)

# why does initialize=True/False produce different results?
# best interface (input, output) ?


"""
if __name__ == "__main__":

    if len(sys.argv)>1:
        # do something
    else:
        # do something else
"""