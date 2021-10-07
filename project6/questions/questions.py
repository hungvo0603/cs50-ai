import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    content = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            content[filename] = f.read()
    return content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized_document = nltk.word_tokenize(document.lower())
    content = [
        word.lower() for word in tokenized_document 
        if word not in nltk.corpus.stopwords.words("english") 
        and word not in string.punctuation 
    ]
    return content


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    words = set()
    total_documents = len(documents)
    for document in documents:
        words.update(set(documents[document]))

    for word in words:
        freq = sum(word in documents[doc] for doc in documents)
        idf = math.log(total_documents / freq)
        idfs[word] = idf
    
    return idfs



def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = list()
    for file in files:
        tfidf = 0
        for word in query:
            tfidf += files[file].count(word) * idfs[word]
        tfidfs.append((file, tfidf))
    tfidfs.sort(key=lambda t: t[1], reverse=True)
    return [x[0] for x in tfidfs[:n]]



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_idfs = list()
    for sentence in sentences:
        idf = 0
        counter = 0
        for word in query:
            if word in sentences[sentence]:
                counter += 1
                idf += idfs[word]
        term_density = float(counter) / len(sentences[sentence])
        sentence_idfs.append((sentence, idf, term_density))
    sentence_idfs.sort(key=lambda t: (t[1], t[2]), reverse=True)
    
    return [x[0] for x in sentence_idfs[:n]]


if __name__ == "__main__":
    main()
