import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path="/Users/vishalpatel/Desktop/Data Science/Generative_AI/coldemail/app/resource/github_repositories.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore2')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio2")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Repositories"],
                                    metadatas={"links": row["Link"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=3).get('metadatas', [])
