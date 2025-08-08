from utils import *
import os
from glob import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
def doc2vec():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=10
    )
    # 读取并分割文件
    dir_path = os.path.join(os.path.dirname(__file__), './data/input/')
    documents = []
    for file_path in glob(dir_path + '*.*'):
        loader = None
        if '.csv' in file_path:
            print("csv")
            loader = CSVLoader(file_path)
        # if '.pdf' in file_path:
        #     loader = PyMuPDFLoader(file_path)
        if '.txt' in file_path:
            print("txt")
            loader = TextLoader(file_path, encoding='utf-8')
        if loader:
            documents += loader.load_and_split(text_splitter)
    if documents:
        vdb = Chroma.from_documents(
            documents = documents, embedding = get_embeddings_model(),
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db/')
        )
            # vdb.persist()
if __name__ == '__main__':
    doc2vec()