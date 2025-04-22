import pandas as pd
from sqlalchemy import create_engine

def fetch_table_from_db(db_user, db_password, db_host, db_port, db_name, table_name):
    """
    Verilen PostgreSQL bağlantı bilgileri ve tablo adı ile tabloyu bir pandas DataFrame olarak döndürür.
    """
    try:
        # SQLAlchemy bağlantı motorunu oluştur
        engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
        
        # Tabloyu pandas DataFrame olarak çek
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        
        # Çekilen veriyi görüntüle
        print(df.head())
        
        # İsteğe bağlı olarak CSV'ye kaydet
        # df.to_csv(f"{table_name}.csv", index=False)
        
        return df
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None

if __name__ == "__main__":
    # Kullanıcıdan gerekli bilgileri al
    db_user = input("Veritabanı kullanıcı adı: ")
    db_password = input("Veritabanı şifresi: ")
    db_host = input("Veritabanı host adresi (varsayılan: localhost): ") or "localhost"
    db_port = input("Veritabanı portu (varsayılan: 5432): ") or "5432"
    db_name = input("Veritabanı adı: ")
    table_name = input("Tablo adı: ")

    # Tabloyu çek
    fetch_table_from_db(db_user, db_password, db_host, db_port, db_name, table_name)

