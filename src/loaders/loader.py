from src.logging.logging import logger
from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Loader:
    def __init__(self):
        pass
    def directory_loader(self):
        try:
            logger.info('directory_loader has been Intialized')
            dirload  = DirectoryLoader(
                path='data/pdf_file/',
                glob='**/*.pdf',
                loader_cls=PyMuPDFLoader
            )
            dir_docs = dirload.load()
            logger.info('files has been loaded successfully')
            return dir_docs
        except Exception as e:
            print(f"Unable to load to directory {e}")
        
    def chunk_maker(self):
        try:
            logger.info('chunk_maker has been Intialized')
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=200,  
                add_start_index=True,  
            )
            
            all_splits = text_splitter.split_documents(self.directory_loader())
            with open('data/text_file/text_split.txt','w',encoding='utf-8') as f:
                for i, doc in enumerate(all_splits, start=1):
                    f.write(f"--- Chunk {i} ---\n")
                    f.write(doc.page_content + "\n\n")
                logger.info(f'{len(all_splits)} chuck files has been saved successfully')
        except Exception as e:
            print(f"Unable to split the text: {e}")

l = Loader()
l.directory_loader()
l.chunk_maker()