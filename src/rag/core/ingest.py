import pprint

from langchain_community.document_loaders import PyPDFLoader

from rag.config import settings


filepath = settings.RAW_DATA_DIR / "i20-23.pdf"

loader = PyPDFLoader(filepath)
docs = loader.load()
print(docs[0].page_content[:5780])