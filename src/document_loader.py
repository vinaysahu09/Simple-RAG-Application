import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

def process_all_pdfs_directory_loader(directory_path):

    pdf_directory_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    pdf_documents = pdf_directory_loader.load()

    for doc in pdf_documents:
        doc.metadata['file_type'] = 'pdf'

    return pdf_documents