from whoosh import index
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


schema = Schema(title=TEXT(stored=True), body=TEXT(stored=True))


indexdir = "C:\\users\\abhay\\Whoosh_folder"
if not index.exists_in(indexdir):
    ix = create_in(indexdir, schema)
else:
    ix = index.open_dir(indexdir)


writer = ix.writer()
writer.add_document(title="Document 1", body="World War 2 started in 1964, during which the Allies fought the Axis Powers.")
writer.add_document(title="Document 2", body="The brain is an organ in your head. It helps you think and perform various tasks.")
writer.add_document(title="Document 3", body="The topelitz square is one of the most difficult and intricate mathematics problems with various proposed solutions.")
writer.commit()


vectorizer = TfidfVectorizer()
documents = [doc.get("body") for doc in ix.searcher().documents()]
tfidf_matrix = vectorizer.fit_transform(documents)

def get_params(user_query, n=1):
    user_query_vector = vectorizer.transform([user_query])


    cosine_similarities = cosine_similarity(tfidf_matrix, user_query_vector)


    partitioned_indices = np.argpartition(cosine_similarities, -n, axis=0)[-n:]


    closest_match_indices = partitioned_indices[np.argsort(cosine_similarities[partitioned_indices].flatten(), axis=0)[::-1]]

    closest_matches = [documents[idx[0]] for idx in closest_match_indices]

    return closest_matches



prompt = input("Enter a topic: ")

n = int(input("Enter the number of results to return: "))


output = get_params(prompt, n)
for i, result in enumerate(output, start=1):
    print(f"Result {i}:\n{result}\n")
