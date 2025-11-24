# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # <--- Added pydantic import
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
import requests

# Import database functions
from database import (
    init_database,
    save_processed_data,
    get_all_processed_data,
    add_upload_record,
    update_upload_record_status,
    get_upload_history,
    add_action_log,     # <--- Import the new function
    DATABASE_FILE
)
from logging_config import setup_logging
# from model_loader import load_model, process_text, analyze_sentiment # <--- REMOVED

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

# --- REMOVED ENTIRE MODEL LOADING BLOCK ---
# All functions like initialize_model, model, tokenizer, etc., are gone.

PROCESSED_DATA_FILE = "processed_comments.csv"
VALID_EXTENSIONS = [".csv", ".txt"]


# ---------------------------
# Preprocessing Helper
# (This function is unchanged, as it's still needed)
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

class LogActionRequest(BaseModel):
    agency_name: str
    action: str
    details: Optional[str] = ""

@app.post("/log_action")
async def log_action(request: Request, log_data: LogActionRequest):
    """
    Log a user action. Captures IP from the request.
    """
    try:
        ip = request.client.host
        # If behind a proxy (like Render), try to get the real IP header
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip = forwarded.split(",")[0]
            
        await asyncio.to_thread(add_action_log, log_data.agency_name, log_data.action, ip, log_data.details)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error logging action: {e}")
        # Return error status but don't crash the caller
        return {"status": "error", "detail": str(e)}

@app.get("/get_logs")
async def get_logs_endpoint(agency_name: Optional[str] = None):
    """
    Retrieve audit logs.
    """
    try:
        df = await asyncio.to_thread(get_action_logs, agency_name)
        if df.empty:
            return {"logs": []}
        
        # Convert to dict and handle datetimes
        logs = df.to_dict(orient='records')
        cleaned_logs = []
        for record in logs:
            cleaned = {}
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    cleaned[k] = v.isoformat()
                elif pd.isna(v):
                    cleaned[k] = None
                else:
                    cleaned[k] = v
            cleaned_logs.append(cleaned)
            
        return {"logs": cleaned_logs}
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_and_analyze")
async def upload_and_analyze(request: Request, file: UploadFile = File(...)):
    logger.info(f"File upload started: {file.filename}")
    upload_id = -1
    record_count = 0
    agencies_detected_str = "N/A"
    
    # 1. Get the Sentiment API URL from the environment (CRITICAL CHECK)
    SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL")
    if not SENTIMENT_API_URL:
        logger.error("SENTIMENT_API_URL environment variable not set.")
        # Return HTTP 503 Service Unavailable if the model service is not defined
        raise HTTPException(status_code=503, detail="Sentiment analysis service endpoint is not configured (SENTIMENT_API_URL missing).")

    try:
        # Create initial history record
        upload_id = add_upload_record(file.filename, "Processing", 0, "N/A")
        if upload_id == -1:
            raise Exception("Failed to create initial upload history record.")
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
        # 2. Read and Preprocess CSV Data
        file_contents_io = io.BytesIO()
        await file.seek(0)
        file_contents_io.write(await file.read())
        file_contents_io.seek(0)
        
        try:
            df = await asyncio.to_thread(pd.read_csv, file_contents_io, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed, trying 'latin-1'.")
            file_contents_io.seek(0)
            df = await asyncio.to_thread(pd.read_csv, file_contents_io, encoding='latin-1')
        finally:
            file_contents_io.close()
            
        record_count = len(df)
        logger.info(f"CSV file loaded with {record_count} rows for upload_id {upload_id}")
        
        df = await asyncio.to_thread(preprocess_dataframe, df)

        if 'agency_name' in df.columns:
            unique_agencies = df['agency_name'].astype(str).str.strip().replace('', 'Unknown').unique()
            valid_agencies = [agency for agency in unique_agencies if agency != 'Unknown' and agency]
            if len(valid_agencies) == 1: agencies_detected_str = valid_agencies[0]
            elif len(valid_agencies) > 1: agencies_detected_str = "Multiple"
            elif 'Unknown' in unique_agencies and len(valid_agencies) == 0: agencies_detected_str = "Unknown"
            logger.info(f"Agencies detected for upload_id {upload_id}: {agencies_detected_str}")

        # --- SENTIMENT ANALYSIS VIA EXTERNAL API ---
        
        # 3. Prepare texts for external API call
        texts_to_analyze = df["comment_text"].astype(str).tolist()
        
        logger.info(f"Sending {len(texts_to_analyze)} comments to Model API at {SENTIMENT_API_URL}...")
        
        # NOTE: requests.post is synchronous, so it MUST be run in a thread
        api_response = await asyncio.to_thread(
            requests.post, 
            SENTIMENT_API_URL, 
            json={"texts": texts_to_analyze}, 
            timeout=3600 
        )
        
        # 4. Check API response status
        if not api_response.ok:
            logger.error(f"Sentiment API failed with status {api_response.status_code}: {api_response.text}")
            update_upload_record_status(upload_id, f"Failed - API {api_response.status_code}", record_count, agencies_detected_str)
            raise HTTPException(status_code=502, detail=f"Sentiment API failed (Status: {api_response.status_code}). Server message: {api_response.text[:200]}...")

        # 5. Extract results
        api_result = api_response.json()
        sentiments = api_result.get("sentiments")

        # CRITICAL FIX: Ensure sentiments variable is a list and matches row count
        if not isinstance(sentiments, list) or len(sentiments) != record_count:
            logger.error(f"Sentiment API returned invalid data structure or length. Expected list of length {record_count}, got {type(sentiments)} of length {len(sentiments) if isinstance(sentiments, list) else 'N/A'}")
            update_upload_record_status(upload_id, "Failed - API Mismatch", record_count, agencies_detected_str)
            raise HTTPException(status_code=502, detail="Sentiment API returned invalid data (length/structure mismatch).")

        # 6. Map results back to DataFrame and clean up
        df["sentiment_label"] = sentiments
        # We assume preprocessed_text column is just the input text for now, as local preprocessor was removed
        df["preprocessed_text"] = texts_to_analyze 
        logger.info("Batch analysis complete. Saving to database...")
        
        # 7. Save and Finalize
        await asyncio.to_thread(save_processed_data, df, upload_id)
        
        update_upload_record_status(upload_id, "Complete", record_count, agencies_detected_str)
        logger.info(f"Successfully processed upload_id {upload_id}")

        all_data_df = await asyncio.to_thread(get_all_processed_data)
        
        all_data_dict = all_data_df.to_dict(orient='records')
        cleaned_all_data = []
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
        # Note: Status update is already done within the try block when raising HTTPExcs related to API failure
        if upload_id != -1:
            update_upload_record_status(upload_id, f"Failed - Client/API Error", record_count, agencies_detected_str)
        raise http_exc
    
    except Exception as e:
        logger.error(f"Unexpected fatal error for upload_id {upload_id}: {str(e)}", exc_info=True)
        if upload_id != -1:
            update_upload_record_status(upload_id, "Failed - Internal Error", record_count, agencies_detected_str)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

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

# <--- THIS IS THE MODIFIED, SIMPLER FUNCTION --->
@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    db_exists = Path(DATABASE_FILE).exists()
    status_code = 200 if db_exists else 503
    return {
        "status": "healthy" if status_code == 200 else "unhealthy",
        "database_status": "connected" if db_exists else "file_not_found",
    }

# <--- THIS FUNCTION IS REMOVED --->
# @app.get("/reload_model") ...


# <--- THIS IS THE MODIFIED, SIMPLER FUNCTION --->
@app.on_event("startup")
async def startup_event():
    logger.info("="*30)
    logger.info("CitiSense API starting up...")
    
    # --- All model loading logs are removed ---

    logger.info(f"Database file: {Path(DATABASE_FILE).resolve()}")
    init_database()
    logger.info("CitiSense API ready to serve requests")
    logger.info("="*30)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("CitiSense API shutting down...")
    logger.info("Shutdown complete.")


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
