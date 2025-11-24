# app/database.py
def init_database():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_id INTEGER,
        agency_name TEXT,
        unique_id TEXT,
        user_code TEXT,
        post_url TEXT,
        post_topic TEXT,
        comment_text TEXT,
        preprocessed_text TEXT,
        comment_timestamp DATETIME,
        sentiment_label TEXT,
        year INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (upload_id) REFERENCES upload_history (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS upload_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL,
        record_count INTEGER,
        agencies_detected TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS action_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agency_name TEXT,
        action TEXT,
        ip_address TEXT,
        details TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized with processed_comments, upload_history, and action_logs tables")

# --- processed_comments functions ---
def save_processed_data(df, upload_id):
    """Save processed data to the processed_comments table with an upload_id."""
    conn = sqlite3.connect(DATABASE_FILE)
    
    df_to_save = df.copy() # Work on a copy
    df_to_save['upload_id'] = upload_id

    if 'preprocessed_text' not in df_to_save.columns:
        logger.warning("preprocessed_text column missing from dataframe, adding empty column")
        df_to_save['preprocessed_text'] = ''

    columns_to_save = [
        'upload_id',
        'agency_name', 'unique_id', 'user_code', 'post_url',
        'post_topic', 'comment_text', 'preprocessed_text',
        'comment_timestamp', 'sentiment_label', 'year'
    ]

    existing_columns = [col for col in columns_to_save if col in df_to_save.columns]
    df_to_save = df_to_save[existing_columns]

    if 'comment_timestamp' in df_to_save.columns:
        df_to_save['comment_timestamp'] = pd.to_datetime(df_to_save['comment_timestamp'], errors='coerce').astype(str)
        df_to_save['comment_timestamp'] = df_to_save['comment_timestamp'].replace('NaT', None)

    for col in df_to_save.columns:
        if pd.api.types.is_integer_dtype(df_to_save[col]) and not isinstance(df_to_save[col].dtype, type(int)):
             try:
                 df_to_save[col] = df_to_save[col].fillna(0).astype(int)
             except Exception as e:
                 logger.warning(f"Could not convert column {col} to standard int: {e}")
        elif pd.api.types.is_bool_dtype(df_to_save[col]):
            df_to_save[col] = df_to_save[col].astype(int)


    df_to_save.to_sql('processed_comments', conn, if_exists='append', index=False)
    conn.close()
    logger.info(f"Saved {len(df_to_save)} records to 'processed_comments' for upload_id {upload_id}")


def get_all_processed_data():
    """Get all processed data from the processed_comments table."""
    if not Path(DATABASE_FILE).exists():
        logger.warning(f"Database file {DATABASE_FILE} not found.")
        return pd.DataFrame()

    conn = sqlite3.connect(DATABASE_FILE)
    try:
        df = pd.read_sql_query('SELECT * FROM processed_comments ORDER BY comment_timestamp DESC', conn)
    except Exception as e:
        logger.error(f"Error reading from processed_comments table: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        logger.info("No data found in processed_comments table.")
        return df

    if 'comment_timestamp' in df.columns:
        df['comment_timestamp'] = pd.to_datetime(df['comment_timestamp'], errors='coerce')

    if 'comment_timestamp' in df.columns:
        df['year'] = df['comment_timestamp'].dt.year.fillna(0).astype(int)
    else:
        df['year'] = 0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna('')

    logger.info(f"Retrieved {len(df)} records from database table 'processed_comments'")
    return df
# --- END processed_comments functions ---


# --- upload_history functions ---
def add_upload_record(file_name: str, status: str, record_count: int, agencies_detected: str) -> int:
    """Adds a record to the upload_history table and returns the new upload_id."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    upload_id = -1
    try:
        cursor.execute('''
            INSERT INTO upload_history (file_name, upload_timestamp, status, record_count, agencies_detected)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_name, datetime.now(), status, record_count, agencies_detected))
        conn.commit()
        upload_id = cursor.lastrowid
        logger.info(f"Added upload record for {file_name} (ID: {upload_id}), status: {status}, count: {record_count}, agencies: {agencies_detected}")
    except Exception as e:
        logger.error(f"Failed to add upload record for {file_name}: {e}")
        conn.rollback()
    finally:
        conn.close()
    return upload_id

def update_upload_record_status(upload_id: int, status: str, record_count: int = -1, agencies_detected: str = None):
    """Updates an existing upload record."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        fields_to_update = ["status = ?"]
        params = [status]
        
        if record_count >= 0:
            fields_to_update.append("record_count = ?")
            params.append(record_count)
            
        if agencies_detected is not None:
            fields_to_update.append("agencies_detected = ?")
            params.append(agencies_detected)
            
        params.append(upload_id)
        
        query = f"UPDATE upload_history SET {', '.join(fields_to_update)} WHERE id = ?"
        
        cursor.execute(query, tuple(params))
        conn.commit()
        logger.info(f"Updated upload record ID {upload_id} to status: {status}")
    except Exception as e:
        logger.error(f"Failed to update upload record ID {upload_id}: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_upload_history(agency_name: str | None = None) -> pd.DataFrame:
    """Retrieves upload history, optionally filtered by agency relevance."""
    if not Path(DATABASE_FILE).exists():
        logger.warning(f"Database file {DATABASE_FILE} not found for history retrieval.")
        return pd.DataFrame()

    conn = sqlite3.connect(DATABASE_FILE)
    query = 'SELECT id as upload_id, file_name, upload_timestamp, status, record_count, agencies_detected FROM upload_history ORDER BY upload_timestamp DESC'
    params = []

    if agency_name:
        query = '''
            SELECT id as upload_id, file_name, upload_timestamp, status, record_count, agencies_detected
            FROM upload_history
            WHERE agencies_detected = ?
                OR agencies_detected = 'Multiple'
                OR agencies_detected = 'None'
                OR agencies_detected LIKE ?
                OR agencies_detected LIKE ?
                OR agencies_detected LIKE ?
            ORDER BY upload_timestamp DESC
        '''
        params = [
            agency_name,
            f'{agency_name},%',
            f'%,{agency_name},%',
            f'%,{agency_name}'
            ]

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        logger.error(f"Error reading from upload_history table: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    # *** START REMOVED BLOCK ***
    # The problematic formatting code that was here has been removed.
    # We will let the backend (main.py) and frontend (dashboard.html)
    # handle timestamp formatting.
    # *** END REMOVED BLOCK ***

    logger.info(f"Retrieved {len(df)} upload history records for agency context: {agency_name or 'All'}")
    return df

def get_upload_history(agency_name: str | None = None) -> pd.DataFrame:
    """Retrieves upload history, optionally filtered by agency relevance."""
    if not Path(DATABASE_FILE).exists():
        logger.warning(f"Database file {DATABASE_FILE} not found for history retrieval.")
        return pd.DataFrame()

    conn = sqlite3.connect(DATABASE_FILE)
    query = 'SELECT id as upload_id, file_name, upload_timestamp, status, record_count, agencies_detected FROM upload_history ORDER BY upload_timestamp DESC'
    params = []

    if agency_name:
        # This new query correctly finds files specific to the agency OR files with multiple agencies
        query = '''
            SELECT id as upload_id, file_name, upload_timestamp, status, record_count, agencies_detected
            FROM upload_history
            WHERE 
                agencies_detected = ? 
                OR agencies_detected = 'Multiple'
                OR agencies_detected LIKE ?
                OR agencies_detected LIKE ?
                OR agencies_detected LIKE ?
            ORDER BY upload_timestamp DESC
        '''
        params = [
            agency_name,
            f'{agency_name},%',  # Starts with the agency
            f'%,{agency_name},%', # Contains the agency in the middle
            f'%,{agency_name}'   # Ends with the agency
        ]

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        logger.error(f"Error reading from upload_history table: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    logger.info(f"Retrieved {len(df)} upload history records for agency context: {agency_name or 'All'}")
    return df


# --- action_logs functions ---
def add_action_log(agency_name: str, action: str, ip_address: str, details: str = ""):
    """Adds a log entry for user actions."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO action_logs (agency_name, action, ip_address, details)
            VALUES (?, ?, ?, ?)
        ''', (agency_name, action, ip_address, details))
        conn.commit()
        logger.info(f"Logged action '{action}' for {agency_name} from {ip_address}")
    except Exception as e:
        logger.error(f"Failed to add action log: {e}")
    finally:
        conn.close()

def get_action_logs(agency_name: str | None = None) -> pd.DataFrame:
    """Retrieves action logs. If agency_name is provided (and not SUPER_ADMIN), filters by agency."""
    if not Path(DATABASE_FILE).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DATABASE_FILE)
    query = 'SELECT * FROM action_logs ORDER BY timestamp DESC'
    params = []

    if agency_name and agency_name != "SUPER_ADMIN":
        query = 'SELECT * FROM action_logs WHERE agency_name = ? ORDER BY timestamp DESC'
        params = [agency_name]

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        logger.error(f"Error reading from action_logs table: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    
    return df
