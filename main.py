# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pdf_generator import router as pdf_router
import pandas as pd
import numpy as np
import math
import io
import re
import os
import asyncio
import sqlite3
from typing import Any, List
from pathlib import Path
from typing import Optional
from datetime import datetime
from huggingface_hub import snapshot_download

# Import database functions
from database import (
    init_database,
    save_processed_data,
    get_all_processed_data,
    add_upload_record,
    update_upload_record_status,
    get_upload_history,
    DATABASE_FILE
)
from logging_config import setup_logging
from model_loader import load_model, process_text, analyze_sentiment

logger = setup_logging()
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf_router)
init_database()

# --- Model Loading ---
# MODEL_PATH = os.getenv("MODEL_PATH", "../model")
model = None
tokenizer = None
MODEL_PATH = None

def initialize_model():
    global model, tokenizer, MODEL_PATH # Add MODEL_PATH here
    MODEL_PATH = None
    
    # 1. Define your Hugging Face repo ID
    # (This is "YourUsername/YourRepoName")
    # !!! REPLACE "YourUsername/citisense-sentiment-model" with YOUR repo ID !!!
    REPO_ID = "yojiyo/citisense-model" 
    
    # 2. Define where to save the model locally on the server's disk
    # This path is used by Render's free disk
    LOCAL_MODEL_DIR = Path("./downloaded_sentiment_model") 

    try:
        # 3. Download all files from the repo
        # This will download all files from your HF repo to the LOCAL_MODEL_DIR
        # It's smart and will only download if the files are missing.
        logger.info(f"Checking for model files... downloading from {REPO_ID} to {LOCAL_MODEL_DIR}")
        snapshot_path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False # Important for Render/Docker
        )
        logger.info(f"Model files are ready at: {snapshot_path}")

        # 4. Set the global MODEL_PATH to this new download directory
        MODEL_PATH = str(snapshot_path) # This is now our new model path

        # 5. Load the model from the new download path
        # We are still using your original load_model function
        model, tokenizer = load_model(MODEL_PATH) 
        
        if model is not None and tokenizer is not None:
            logger.info("Sentiment analysis model loaded successfully from downloaded files.")
            return True
        else:
            logger.error("Model or tokenizer failed to load (returned None) from downloaded files.")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading/loading model from Hugging Face repo {REPO_ID}: {str(e)}", exc_info=True)
        return False

model_loaded = initialize_model()

PROCESSED_DATA_FILE = "processed_comments.csv"
VALID_EXTENSIONS = [".csv", ".txt"]


# ---------------------------
# Preprocessing Helper
# ---------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe for analysis."""
    df.columns = [col.lower().strip() for col in df.columns]
    logger.debug(f"Normalized columns: {df.columns.tolist()}")

    timestamp_col_names = ["comment_timestamp", "timestamp", "date", "datetime", "comment_date"]
    actual_timestamp_col = None
    for name in timestamp_col_names:
        if name in df.columns:
            actual_timestamp_col = name
            logger.info(f"Using column '{actual_timestamp_col}' for timestamps.")
            break

    if actual_timestamp_col:
        formats_to_try = [
            "%A, %B %d, %Y, %I:%M %p",  # "Monday, July 18, 2025, 1:59 PM" (with space)
            "%B %d, %Y, %I:%M %p",       # "January 17, 2022, 10:23 AM" (with space)
            "%A, %B %d, %Y, %I:%M%p",   # "Monday, July 18, 2025, 1:59PM" (no space)
            "%B %d, %Y, %I:%M%p",        # "January 17, 2022, 10:23AM" (no space)
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %I:%M %p",         # "01/26/2022 3:02 PM" (with space)
            "%m/%d/%Y %I:%M%p",         # "01/26/2022 3:02PM" (no space)
            "%m/%d/%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ]
        
        invalid_timestamp_pattern = r'^0000-00-00(?:[ T]00:00:00(?:\.0)?)?$'
        
        df[actual_timestamp_col] = df[actual_timestamp_col].astype(str).str.strip()
        replaced_count = df[actual_timestamp_col].str.contains(invalid_timestamp_pattern, regex=True, na=False).sum()
        df[actual_timestamp_col] = df[actual_timestamp_col].replace(
            to_replace=invalid_timestamp_pattern, value=None, regex=True
        )
        logger.info(f"Replaced {replaced_count} invalid '0000-00-00...' timestamps with None.")

        # Try each format sequentially
        parsed_timestamps = pd.Series([pd.NaT] * len(df), index=df.index)
        
        for fmt in formats_to_try:
            failed_indices = parsed_timestamps.isna()
            if not failed_indices.any():
                break  # All timestamps parsed successfully
            
            try:
                temp_parsed = pd.to_datetime(
                    df.loc[failed_indices, actual_timestamp_col],
                    format=fmt,
                    errors='coerce'
                )
                parsed_timestamps.loc[failed_indices] = temp_parsed
                successful = temp_parsed.notna().sum()
                if successful > 0:
                    logger.info(f"Successfully parsed {successful} timestamps with format: {fmt}")
            except Exception as e:
                logger.debug(f"Format {fmt} failed: {e}")
                continue
        
        # Final attempt with 'mixed' for any remaining unparsed timestamps
        failed_indices = parsed_timestamps.isna()
        if failed_indices.any():
            logger.info(f"Retrying {failed_indices.sum()} failed timestamps with 'mixed' format.")
            try:
                mixed_parsed = pd.to_datetime(
                    df.loc[failed_indices, actual_timestamp_col],
                    errors='coerce',
                    format='mixed'
                )
                parsed_timestamps.loc[failed_indices] = mixed_parsed
                successful = mixed_parsed.notna().sum()
                if successful > 0:
                    logger.info(f"Successfully parsed {successful} additional timestamps with mixed format.")
            except Exception as e:
                logger.warning(f"Mixed format parsing failed: {e}")
        
        df[actual_timestamp_col] = parsed_timestamps
        df["year"] = df[actual_timestamp_col].dt.year.fillna(0).astype(int)
        
        # Log parsing statistics
        total_timestamps = len(df)
        parsed_count = parsed_timestamps.notna().sum()
        failed_count = total_timestamps - parsed_count
        logger.info(f"Timestamp parsing complete: {parsed_count}/{total_timestamps} successful, {failed_count} failed (will show as N/A)")

        if actual_timestamp_col != "comment_timestamp":
             df.rename(columns={actual_timestamp_col: "comment_timestamp"}, inplace=True)
             logger.info(f"Renamed column '{actual_timestamp_col}' to 'comment_timestamp'.")
    else:
        logger.warning(f"Could not find timestamp column. Adding empty 'comment_timestamp' and 'year'.")
        df["comment_timestamp"] = pd.NaT
        df["year"] = 0

    required_input_cols = [
        "agency_name", "comment_text", "comment_timestamp",
        "unique_id", "user_code", "post_url", "post_topic", "year"
    ]
    for col in required_input_cols:
        if col not in df.columns:
             if col == "comment_timestamp": df[col] = pd.NaT
             elif col == "year": df[col] = 0
             else: df[col] = ""
             logger.warning(f"Missing required input column '{col}' - filled default.")

    if "comment_timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["comment_timestamp"]):
        df.loc[:, "comment_timestamp"] = df["comment_timestamp"].apply(lambda x: x if pd.notna(x) else None)

    for col in df.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_float_dtype(df[col]):
            if np.isinf(df[col]).any():
                 logger.warning(f"Found infinite values in numeric column '{col}'. Replacing with None.")
                 df.loc[:, col] = df[col].replace([np.inf, -np.inf], None)
    
    for col in df.columns:
         if df[col].dtype == 'object':
             df.loc[:, col] = df[col].fillna('')
         elif pd.api.types.is_numeric_dtype(df[col]):
             df.loc[:, col] = df[col].fillna(0)
             try:
                 if (df[col].dropna() % 1 == 0).all():
                     df.loc[:, col] = df[col].astype(int)
             except (TypeError, ValueError): pass

    if 'year' in df.columns:
         df['year'] = df['year'].replace('', 0).astype(int)

    logger.debug("Preprocessing complete.")
    return df


# ---------------------------
# Routes
# ---------------------------
@app.post("/upload_and_analyze")
async def upload_and_analyze(request: Request, file: UploadFile = File(...)):
    logger.info(f"File upload started: {file.filename}")
    upload_id = -1
    record_count = 0
    agencies_detected_str = "None"
    
    try:
        # Create the initial history record. We need the ID.
        upload_id = add_upload_record(file.filename, "Processing", 0, "N/A")
        if upload_id == -1:
            raise HTTPException(status_code=500, detail="Failed to create initial upload history record.")
    except Exception as e:
        logger.error(f"Failed to create initial history record: {e}")
        raise HTTPException(status_code=500, detail=f"Database error on history creation: {e}")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in VALID_EXTENSIONS:
        logger.warning(f"Invalid file type uploaded: {file_extension}")
        update_upload_record_status(upload_id, "Failed - Invalid Type")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(VALID_EXTENSIONS)} files are accepted. Received: {file_extension}",
        )

    try:
        global model_loaded, model, tokenizer
        if not model_loaded or model is None or tokenizer is None:
            # ... (model loading logic is fine) ...
            logger.warning("Model not loaded. Attempting to reload...")
            model_loaded = initialize_model()
            if not model_loaded:
                logger.error("Failed to reload model.")
                raise HTTPException(status_code=500, detail="Sentiment analysis model could not be loaded.")

        # --- CANCELLABLE UPLOAD BLOCK ---
        logger.info(f"Starting file read for {file.filename} (upload_id {upload_id})")
        file_contents_io = io.BytesIO()
        chunk_count = 0
        
        try:
            while True:
                if chunk_count % 5 == 0 and await request.is_disconnected(): 
                    logger.warning(f"[Cancel] Client disconnected during file upload for upload_id {upload_id}.")
                    
                    # --- DELETE ON CANCEL ---
                    try:
                        conn = sqlite3.connect(DATABASE_FILE)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM upload_history WHERE id = ?", (upload_id,))
                        conn.commit()
                        conn.close()
                        logger.info(f"[Cancel] Deleted upload_history record {upload_id}.")
                    except Exception as db_e:
                        logger.error(f"[Cancel] Failed to delete upload_history record {upload_id}: {db_e}")
                    # --- END DELETE ON CANCEL ---
                        
                    raise HTTPException(status_code=499, detail="Client disconnected during upload")
                
                chunk = await file.read(1024 * 1024) # 1MB chunks
                
                if not chunk:
                    break 
                
                file_contents_io.write(chunk)
                chunk_count += 1

            logger.info(f"File read complete for upload_id {upload_id}. Total chunks: {chunk_count}")
            
            file_contents_io.seek(0)
            
            try:
                # --- MODIFIED: Run pd.read_csv in a thread ---
                logger.info(f"Pandas read (utf-8) starting for upload_id {upload_id}...")
                df = await asyncio.to_thread(pd.read_csv, file_contents_io, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 failed, trying 'latin-1'.")
                file_contents_io.seek(0)
                # --- MODIFIED: Run pd.read_csv in a thread ---
                df = await asyncio.to_thread(pd.read_csv, file_contents_io, encoding='latin-1')
        
        finally:
            file_contents_io.close()
        # --- END OF CANCELLABLE UPLOAD BLOCK ---
            
        record_count = len(df)
        logger.info(f"CSV file loaded with {record_count} rows for upload_id {upload_id}")

        # --- MODIFIED: ADDED DISCONNECT CHECK ---
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected after file read")

        # --- MODIFIED: Run preprocess_dataframe in a thread ---
        logger.info(f"Running preprocess_dataframe for upload_id {upload_id}...")
        df = await asyncio.to_thread(preprocess_dataframe, df)
        logger.info(f"Finished preprocess_dataframe for upload_id {upload_id}")

        # --- MODIFIED: ADDED DISCONNECT CHECK ---
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected after dataframe preprocessing")
        # --- END OF MODIFICATIONS ---

        if 'agency_name' in df.columns:
            # ... (this logic is fast, no change needed) ...
            unique_agencies = df['agency_name'].astype(str).str.strip().replace('', 'Unknown').unique()
            valid_agencies = [agency for agency in unique_agencies if agency != 'Unknown' and agency]
            if len(valid_agencies) == 1: agencies_detected_str = valid_agencies[0]
            elif len(valid_agencies) > 1: agencies_detected_str = "Multiple"
            elif 'Unknown' in unique_agencies and len(valid_agencies) == 0: agencies_detected_str = "Unknown"
            logger.info(f"Agencies detected for upload_id {upload_id}: {agencies_detected_str}")

        # --- CANCELLABLE PREPROCESSING LOOP (This block is correct from last time) ---
        logger.info(f"Applying text preprocessing for upload_id {upload_id}...")
        if 'comment_text' in df.columns:
            preprocessed_texts = []
            texts_to_process = df["comment_text"].astype(str)

            for text in texts_to_process:
                if await request.is_disconnected():
                    logger.warning(f"[Cancel] Client disconnected during preprocessing for upload_id {upload_id}.")
                    # --- DELETE ON CANCEL ---
                    try:
                        conn = sqlite3.connect(DATABASE_FILE)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM upload_history WHERE id = ?", (upload_id,))
                        conn.commit()
                        conn.close()
                        logger.info(f"[Cancel] Deleted upload_history record {upload_id} due to preprocess cancel.")
                    except Exception as db_e:
                        logger.error(f"[Cancel] Failed to delete upload_history record {upload_id}: {db_e}")
                    raise HTTPException(status_code=499, detail="Client disconnected during preprocessing")

                processed_text = await asyncio.to_thread(process_text, text)
                preprocessed_texts.append(processed_text)
            
            df["preprocessed_text"] = preprocessed_texts
        else:
            df["preprocessed_text"] = ""
        # --- END OF CANCELLABLE PREPROCESSING LOOP ---

        # --- CANCELLABLE ANALYSIS LOOP (This block is correct from last time) ---
        logger.info(f"Running sentiment analysis for upload_id {upload_id}...")
        if "preprocessed_text" in df.columns:
            sentiments = []
            text_to_process = df["preprocessed_text"].astype(str)
            for text in text_to_process:
                if await request.is_disconnected():
                    logger.warning(f"[Cancel] Client disconnected during analysis for upload_id {upload_id}.")
                    # --- DELETE ON CANCEL ---
                    try:
                        conn = sqlite3.connect(DATABASE_FILE)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM upload_history WHERE id = ?", (upload_id,))
                        cursor.execute("DELETE FROM processed_comments WHERE upload_id = ?", (upload_id,))
                        conn.commit()
                        conn.close()
                        logger.info(f"[Cancel] Deleted upload_history and processed_comments for {upload_id}.")
                    except Exception as db_e:
                        logger.error(f"[Cancel] Failed to delete records for {upload_id}: {db_e}")
                    raise HTTPException(status_code=499, detail="Client disconnected during analysis")
                
                if isinstance(text, str) and text.strip():
                    label = await asyncio.to_thread(analyze_sentiment, text, model, tokenizer)
                else:
                    label = "Neutral"
                sentiments.append(label)
            df["sentiment_label"] = sentiments
        else:
            df["sentiment_label"] = "Neutral"
        # --- END OF CANCELLABLE ANALYSIS LOOP ---

        # --- MODIFIED: Run save_processed_data in a thread ---
        logger.info(f"Saving data for upload_id {upload_id}...")
        await asyncio.to_thread(save_processed_data, df, upload_id)
        
        # update_upload_record_status is fast, no thread needed
        update_upload_record_status(upload_id, "Complete", record_count, agencies_detected_str)
        logger.info(f"Successfully processed upload_id {upload_id}")

        # --- MODIFIED: Run get_all_processed_data in a thread ---
        all_data_df = await asyncio.to_thread(get_all_processed_data)
        logger.info(f"Returning {len(all_data_df)} total records after successful upload.")
        
        # This part (dict conversion) is fast
        all_data_dict = all_data_df.to_dict(orient='records')
        cleaned_all_data = []
        # ... (rest of dict conversion) ...
        for record in all_data_dict:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value): cleaned_record[key] = None
                elif isinstance(value, pd.Timestamp): cleaned_record[key] = value.isoformat()
                elif isinstance(value, (np.int64, np.int32)): cleaned_record[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)): cleaned_record[key] = float(value) if not (math.isnan(value) or math.isinf(value)) else None
                else: cleaned_record[key] = value
            cleaned_all_data.append(cleaned_record)

        return {"data": cleaned_all_data}

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception for upload_id {upload_id}: {http_exc.detail}")
        
        # This logic is correct: if 499, we already deleted, so do nothing.
        if http_exc.status_code != 499:
            update_upload_record_status(upload_id, f"Failed - {http_exc.status_code}", record_count, agencies_detected_str)
        
        raise http_exc
    
    except Exception as e:
        logger.error(f"Unexpected error for upload_id {upload_id}: {str(e)}", exc_info=True)
        # This is also correct: a real error should be marked as Failed.
        update_upload_record_status(upload_id, f"Failed - Internal Error", record_count, agencies_detected_str)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/get_all_data")
async def get_all_data():
    """
    Retrieve all processed comments with sentiment analysis results
    """
    try:
        df = get_all_processed_data()
        
        if df.empty:
            return {"data": []}
        
        # Convert DataFrame to list of dictionaries
        data_dict = df.to_dict(orient='records')
        
        # Clean the data for JSON serialization
        cleaned_data = []
        for record in data_dict:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, pd.Timestamp):
                    cleaned_record[key] = value.isoformat()
                elif isinstance(value, (np.int64, np.int32)):
                    cleaned_record[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    if math.isnan(value) or math.isinf(value):
                        cleaned_record[key] = None
                    else:
                        cleaned_record[key] = float(value)
                else:
                    cleaned_record[key] = value
            cleaned_data.append(cleaned_record)
        
        logger.info(f"Returning {len(cleaned_data)} records")
        return {"data": cleaned_data}
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


@app.get("/get_upload_history")
async def get_upload_history_endpoint(agency_name: Optional[str] = None):
    """
    Get upload history, optionally filtered by agency
    """
    try:
        df = get_upload_history(agency_name)
        
        if df.empty:
            return {"history": []}
        
        # Convert DataFrame to list of dictionaries
        history_dict = df.to_dict(orient='records')
        
        # Clean the data for JSON serialization
        cleaned_history = []
        for record in history_dict:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, pd.Timestamp):
                    cleaned_record[key] = value.isoformat()
                elif isinstance(value, (np.int64, np.int32)):
                    cleaned_record[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    if math.isnan(value) or math.isinf(value):
                        cleaned_record[key] = None
                    else:
                        cleaned_record[key] = float(value)
                else:
                    cleaned_record[key] = value
            cleaned_history.append(cleaned_record)
        
        logger.info(f"Returning {len(cleaned_history)} history records")
        return {"history": cleaned_history}
        
    except Exception as e:
        logger.error(f"Error fetching upload history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@app.get("/test_simple")
async def test_simple():
    logger.info("Test simple endpoint called")
    return {"message": "API is working!"}

@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    model_status = "loaded" if model_loaded and model is not None and tokenizer is not None else "not loaded"
    db_exists = Path(DATABASE_FILE).exists()
    model_path_exists = Path(MODEL_PATH).exists() and Path(MODEL_PATH).is_dir()
    status_code = 200 if model_status == "loaded" and db_exists and model_path_exists else 503
    return {
        "status": "healthy" if status_code == 200 else "unhealthy",
        "model_status": model_status,
        "database_status": "connected" if db_exists else "file_not_found",
        "model_path_status": "found" if model_path_exists else "not_found",
        "model_path_configured": MODEL_PATH,
    }

@app.get("/reload_model")
async def reload_model():
    global model_loaded
    try:
        logger.info("Manual model reload requested via API endpoint")
        model_loaded = initialize_model()
        if model_loaded:
            logger.info("Model reloaded successfully via API.")
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            logger.error("Failed to reload model via API.")
            raise HTTPException(status_code=500, detail="Failed to reload model. Check server logs.")
    except Exception as e:
        logger.error(f"Error during manual model reload via API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


@app.on_event("startup")
async def startup_event():
    logger.info("="*30)
    logger.info("CitiSense API starting up...")
    
    # --- MODIFIED: Check if MODEL_PATH was set ---
    if MODEL_PATH:
        logger.info(f"Model path configured: {Path(MODEL_PATH).resolve()}")
    else:
        logger.warning("MODEL_PATH is not set. Model download may have failed.")
    # --- END MODIFICATION ---

    logger.info(f"Model loaded successfully on startup: {model_loaded}")
    logger.info(f"Database file: {Path(DATABASE_FILE).resolve()}")
    init_database()
    logger.info("CitiSense API ready to serve requests")
    logger.info("="*30)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("CitiSense API shutting down...")
    logger.info("Shutdown complete.")

import sqlite3

@app.delete("/delete_upload/{upload_id}")
async def delete_upload(upload_id: int):
    """
    Delete an upload and all its associated data from the database
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # First, get the upload info before deleting (using 'id' column, not 'upload_id')
        cursor.execute("""
            SELECT file_name, record_count 
            FROM upload_history 
            WHERE id = ?
        """, (upload_id,))
        
        upload_info = cursor.fetchone()
        
        if not upload_info:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Upload with ID {upload_id} not found")
        
        file_name, record_count = upload_info
        
        # Delete all comments associated with this upload
        cursor.execute("DELETE FROM processed_comments WHERE upload_id = ?", (upload_id,))
        deleted_comments = cursor.rowcount
        
        # Delete the upload record
        cursor.execute("DELETE FROM upload_history WHERE id = ?", (upload_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted upload_id {upload_id} ('{file_name}'): {deleted_comments} comments removed")
        
        return {
            "success": True,
            "message": f"Successfully deleted '{file_name}' and {deleted_comments} associated comments",
            "upload_id": upload_id,
            "deleted_comments": deleted_comments
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting upload {upload_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting upload: {str(e)}")