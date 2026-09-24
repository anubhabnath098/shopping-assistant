pip install -r requirements.txt   
set .env   
put pdfs in data/raw_pdfs/  
python main.py ingest   
python main.py chat  