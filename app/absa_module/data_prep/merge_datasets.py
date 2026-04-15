import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def merge_all_datasets(
    input_pattern: str = "*.csv",
    output_file: str = "reviews.csv",
    encoding: str = "utf-8",
    sep: str = ",",
    add_source_file: bool = True,
    add_processing_date: bool = True
) -> None:

    data_dir = Path("./absa_reviews_2014")
    input_path = data_dir / input_pattern
    
    files = sorted(glob.glob(str(input_path)))
    
    if not files:
        logger.error("Erro")
        return
    
    logger.info(f"Encontrados ficheiros.")
    
    dfs = []
    for file_path in files:
        try:            
            df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
            
            dfs.append(df)
            
        except Exception as e:
            logger.error("Erro")
            continue
    
    if not dfs:
        logger.error("Nenhum dataset foi carregado com sucesso.")
        return
    
    merged_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    merged_df.drop_duplicates(inplace=True)
    
    output_path = data_dir / output_file
    merged_df.to_csv(output_path, index=False, encoding=encoding)
    
    logger.info("=" * 60)
    logger.info(f"   União concluída com sucesso!")
    logger.info(f"   Total de reviews consolidadas: {len(merged_df):,}")
    logger.info(f"   Ficheiro gerado: {output_path.resolve()}")
    logger.info(f"   Colunas: {list(merged_df.columns)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    merge_all_datasets(
        input_pattern="*.csv",       
        output_file="reviews.csv",
        encoding="utf-8",
        sep=",",                         
        add_source_file=True,
        add_processing_date=True
    )