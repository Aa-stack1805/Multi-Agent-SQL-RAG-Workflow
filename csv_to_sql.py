import pandas as pd
from sqlalchemy import create_engine
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperstoreDatabaseLoader:
    def __init__(self, db_type='sqlite', **db_config):
        self.db_type = db_type
        self.db_config = db_config
        self.engine = None
        
    def create_connection(self):
        """Create database connection"""
        try:
            if self.db_type == 'sqlite':
                db_path = self.db_config.get('database', 'superstore.db')
                self.engine = create_engine(f'sqlite:///{db_path}')
                logger.info(f"Connected to SQLite database: {db_path}")
                
            elif self.db_type == 'mysql':
                connection_string = (
                    f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}"
                    f"@{self.db_config['host']}:{self.db_config.get('port', 3306)}"
                    f"/{self.db_config['database']}"
                )
                self.engine = create_engine(connection_string)
                logger.info("Connected to MySQL database")
                
            elif self.db_type == 'postgresql':
                connection_string = (
                    f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                    f"@{self.db_config['host']}:{self.db_config.get('port', 5432)}"
                    f"/{self.db_config['database']}"
                )
                self.engine = create_engine(connection_string)
                logger.info("Connected to PostgreSQL database")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def preview_csv(self, csv_file_path, rows=5):
        """Preview CSV file structure and detect encoding"""
        logger.info(f"Previewing CSV file: {csv_file_path}")
        
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding, nrows=rows)
                logger.info(f"✅ Successfully read with {encoding} encoding")
                logger.info(f"File shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                logger.info("Sample data:")
                print(df.head())
                return encoding, df
            except Exception as e:
                logger.warning(f"❌ Failed with {encoding}: {str(e)[:100]}")
                continue
        
        raise Exception("Could not read CSV file with any supported encoding")
    
    def clean_column_names(self, df):
        """Clean and standardize column names"""
        column_mapping = {
            'Row ID': 'row_id',
            'Order ID': 'order_id', 
            'Order Date': 'order_date',
            'Ship Date': 'ship_date',
            'Ship Mode': 'ship_mode',
            'Customer ID': 'customer_id',
            'Customer Name': 'customer_name',
            'Segment': 'segment',
            'Country': 'country',
            'City': 'city',
            'State': 'state',
            'Postal Code': 'postal_code',
            'Region': 'region',
            'Product ID': 'product_id',
            'Category': 'category',
            'Sub-Category': 'sub_category',
            'Product Name': 'product_name',
            'Sales': 'sales',
            'Quantity': 'quantity',
            'Discount': 'discount',
            'Profit': 'profit'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Clean column names: lowercase, replace spaces/special chars with underscores
        df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', col.lower().strip()) for col in df.columns]
        
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        logger.info("Cleaning data...")
        
        date_columns = ['order_date', 'ship_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        numeric_columns = ['sales', 'quantity', 'discount', 'profit']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        string_columns = ['customer_name', 'product_name', 'city', 'state', 'country']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        if 'customer_id' not in df.columns and 'customer_name' in df.columns:
            df['customer_id'] = 'CUST_' + df.index.astype(str).str.zfill(6)
        
        if 'product_id' not in df.columns and 'product_name' in df.columns:
            df['product_id'] = 'PROD_' + df.groupby('product_name').ngroup().astype(str).str.zfill(4)
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        critical_columns = ['order_date', 'sales']
        df = df.dropna(subset=critical_columns)
        
        logger.info(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def create_tables(self):
        """Create database tables using the SQL schema"""
        from sqlalchemy import text
        
        if self.db_type == 'sqlite':
            tables_sql = {
                'sales_data': """
                CREATE TABLE IF NOT EXISTS sales_data (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    order_date DATE,
                    ship_date DATE,
                    ship_mode TEXT,
                    customer_id TEXT,
                    customer_name TEXT,
                    segment TEXT,
                    country TEXT,
                    city TEXT,
                    state TEXT,
                    postal_code TEXT,
                    region TEXT,
                    product_id TEXT,
                    category TEXT,
                    sub_category TEXT,
                    product_name TEXT,
                    sales REAL,
                    quantity INTEGER,
                    discount REAL,
                    profit REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                'base_forecasts': """
                CREATE TABLE IF NOT EXISTS base_forecasts (
                    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    region TEXT,
                    segment TEXT,
                    category TEXT,
                    sub_category TEXT,
                    forecast_date DATE,
                    forecasted_sales REAL,
                    forecasted_quantity INTEGER,
                    forecast_method TEXT,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            }
        else:
            return
            
        with self.engine.connect() as conn:
            for table_name, sql in tables_sql.items():
                conn.execute(text(sql))
                conn.commit()
                logger.info(f"Created table: {table_name}")
    
    def load_csv_to_database(self, csv_file_path, table_name='sales_data'):
        """Load CSV data into the database"""
        try:
            logger.info(f"Loading CSV file: {csv_file_path}")
            
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Trying encoding: {encoding}")
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    logger.info(f"Successfully read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read with {encoding} encoding")
                    continue
                except Exception as e:
                    logger.warning(f"Error with {encoding} encoding: {e}")
                    continue
            
            if df is None:
                raise Exception("Could not read CSV file with any supported encoding")
            
            logger.info(f"Read {len(df)} rows from CSV")
            
            df = self.clean_column_names(df)
            df = self.clean_data(df)
            
            if not self.engine:
                self.create_connection()
            
            self.create_tables()
            
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {len(df)} rows to {table_name} table")
            
            self.generate_sample_forecasts(df)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV to database: {e}")
            return False
    
    def generate_sample_forecasts(self, sales_df):
        """Generate sample forecast data from historical sales"""
        try:
            logger.info("Generating sample forecasts...")
            
            forecast_groups = sales_df.groupby(['product_id', 'region', 'category', 'sub_category']).agg({
                'sales': ['mean', 'std'],
                'quantity': 'mean'
            }).reset_index()
            
            forecast_groups.columns = ['product_id', 'region', 'category', 'sub_category', 
                                     'avg_sales', 'std_sales', 'avg_quantity']
            
            import numpy as np
            from datetime import timedelta
            
            forecast_data = []
            base_date = datetime.now().date()
            
            for _, row in forecast_groups.iterrows():
                for days_ahead in range(1, 31):  # Next 30 days
                    forecast_date = base_date + timedelta(days=days_ahead)
                    
                    base_forecast = row['avg_sales']
                    forecast_sales = max(0, base_forecast * (0.8 + np.random.random() * 0.4))
                    
                    forecast_data.append({
                        'product_id': row['product_id'],
                        'region': row['region'],
                        'category': row['category'],
                        'sub_category': row['sub_category'],
                        'forecast_date': forecast_date,
                        'forecasted_sales': round(forecast_sales, 2),
                        'forecasted_quantity': int(row['avg_quantity']),
                        'forecast_method': 'Simple Moving Average',
                        'confidence_interval_lower': round(forecast_sales * 0.8, 2),
                        'confidence_interval_upper': round(forecast_sales * 1.2, 2)
                    })
            
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df.to_sql('base_forecasts', self.engine, if_exists='replace', index=False)
            logger.info(f"Generated {len(forecast_df)} forecast records")
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
    
    def verify_data(self):
        """Verify the loaded data"""
        from sqlalchemy import text
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as count FROM sales_data")).fetchone()
                logger.info(f"sales_data table has {result[0]} rows")
                
                result = conn.execute(text("SELECT COUNT(*) as count FROM base_forecasts")).fetchone()
                logger.info(f"base_forecasts table has {result[0]} rows")
                
                sample_sales = conn.execute(text("SELECT * FROM sales_data LIMIT 3")).fetchall()
                logger.info("Sample sales data:")
                for row in sample_sales:
                    logger.info(f"  {row}")
                    
        except Exception as e:
            logger.error(f"Error verifying data: {e}")

def main():
    """Main function to demonstrate usage"""
    
    loader = SuperstoreDatabaseLoader(db_type='sqlite', database='superstore.db')
    
    csv_file = '/Users/aaditsingal/Development/Summer Projects/SQLMultiAgent/Sample - Superstore.csv' 
    
    try:
        print("🔍 Previewing CSV file...")
        encoding, sample_df = loader.preview_csv(csv_file)
        print(f"\n📊 File uses {encoding} encoding")
        print(f"📈 Found {len(sample_df.columns)} columns")
        
        response = input("\nDo you want to proceed with loading? (y/n): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            return
            
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Please update the csv_file variable with the correct path to your Superstore CSV file")
        return
    except Exception as e:
        print(f"❌ Error previewing file: {e}")
        return
    
    success = loader.load_csv_to_database(csv_file)
    
    if success:
        loader.verify_data()
        print("\n✅ Database setup complete!")
        print("Your database is ready for the multi-agent workflow")
        print("\nNext steps:")
        print("1. Use the SQL Agent to query sales_data and base_forecasts tables")
        print("2. Set up your Knowledge Agent RAG with schema information")
        print("3. Test queries like:")
        print("   - SELECT * FROM sales_data WHERE region = 'West' LIMIT 10;")
        print("   - SELECT * FROM base_forecasts WHERE forecast_date >= date('now');")
    else:
        print("❌ Database setup failed. Check the logs above.")

if __name__ == "__main__":
    main()