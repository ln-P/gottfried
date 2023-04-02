import pinecone

class PineconeConnector():
    """
    Class to connect to Pinecone and perform operations on index
    """
    def __init__(self, index_name, pinecone_api_key, pinecone_env):
        self.index_name = index_name
        # initialize connection to pinecone
        pinecone.init(
            api_key=pinecone_api_key, 
            environment=pinecone_env
        )
        self.index = self.init_index()
    
    def init_index(self):
        # check if index already exists
        if self.index_name not in pinecone.list_indexes():
            # if does not exist, create index
            pinecone.create_index(
                self.index_name,
                dimension=1536,
                metric='dotproduct'
            )
        index = pinecone.Index(self.index_name)    
        return index
    
    def add_documents(self, documents):
        self.index.upsert(documents)
    
    def search(self, query, top_k=10):
        return self.index.query(query, top_k=top_k)