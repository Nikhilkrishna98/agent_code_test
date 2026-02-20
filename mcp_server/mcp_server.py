from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import boto3
import botocore
from typing import Optional
from datetime import datetime
from fastmcp import FastMCP
import httpx
import requests
from typing import Dict, Any, Optional, Tuple
import threading

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import pandas as pd
import duckdb
from openpyxl import load_workbook
from typing import Optional, List, Dict, Any
from io import BytesIO  


os.environ['AZURE_OPENAI_API_KEY']= os.getenv('AZURE_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT']= os.getenv('AZURE_API_BASE')

mcp_host = os.getenv('MCP_HOST')
mcp_port = os.getenv('MCP_PORT')

mcp_host = "0.0.0.0"
mcp_port = "8001"

mcp = FastMCP(name="MCPServer")

@mcp.tool
def get_current_datetime() -> str:
    '''Get the current date and time as a formatted string.'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool
def count_words(text: str) -> int:
    """
    Counts the number of words in a given text string.

    Args:
        text (str): The text whose words are to be counted.

    Returns:
        int: The number of words in the input text.
    """
    return len(text.split())


@mcp.tool
def retrieve_control_violations(user_query:str):
    """
    Given an audit issue statement, retriece relevant control violation details from the vector database.
    Use this tool when you need historical or similar control violations to support your analysis or recommendation.
    Input:
        A single string that contains the issue statement based on which the relevant control violations need to be fetched. 
    Output:
        A list of dictionaries, where each dictionary has the retrieved chunk and the metadata associated with it.
    """
    embedding_model_name = 'text-embedding-3-large'
    azure_embeddings= AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        azure_deployment=embedding_model_name,
        openai_api_version='2024-02-01',
    )
    vector_store = FAISS.load_local(
            "vector_stores/iap_vector_db", azure_embeddings, allow_dangerous_deserialization=True
        )
    docs = vector_store.similarity_search_with_relevance_scores(user_query, k=2)
    new_dict_list = []
    for doc, score in docs:
        new_dict = {
            'page_content': doc.page_content,
            'page': doc.metadata['page'],
            'source': doc.metadata['source'],
            # 'gpt_violation_attempt': con,
            'relevance_score': str(score)  # Add relevance score
        }
        new_dict_list.append(new_dict)

    return new_dict_list


@mcp.tool
def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """
    Fetches the exchange rate from one currency to another for a specific date. 

    Args:
        currency_from: The currency to convert from (e.g., "USD").
        currency_to: The currency to convert to (e.g., "EUR").
        currency_date: The date for the exchange rate or "latest". Defaults to
            "latest".

    Returns:
        A dictionary containing the exchange rate data, or an error message if
        the request fails.
    """
    try:
        response = httpx.get(
            f'https://api.frankfurter.app/{currency_date}',
            params={'from': currency_from, 'to': currency_to},
        )
        response.raise_for_status()

        data = response.json()
        # data = {'amount': 1.0, 'base': 'USD', 'date': '2025-06-18', 'rates': {'INR': 86.43}}
        if 'rates' not in data:
            return {'error': 'Invalid API response format.'}
        return data
    except httpx.HTTPError as e:
        return {'error': f'API request failed: {e}'}
    except ValueError:
        return {'error': 'Invalid JSON response from API.'}


def generate_access_token(api_domain_url: str, environment: str, version: str, username: str, password: str) -> str:
    url = f"https://{api_domain_url}/ingestion/{environment}/api/{version}/users/generateAccessToken"
    payload = {"username": username, "password": password}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["data"]

@mcp.tool
def upload_document(api_domain_url: str, environment: str, version: str, username: str, password: str, file_path: str) -> str:
    """
    Uploads a document (from local file path or URL) to the ingestion API.

    Args:
        api_domain_url (str): Base domain of the API.
        environment (str): Target environment for ingestion (e.g., dev, test, prod).
        version (str): API version.
        username (str): Username for authentication.
        password (str): Password for authentication.
        file_path (str): Local file path or URL of the document to upload.

    Returns:
        str: The unique document ID (fileLocation) assigned by the API.
    """
    print("upload_document tool used.")
    access_token = generate_access_token(api_domain_url, environment, version, username, password)
    url = f"https://{api_domain_url}/ingestion/{environment}/api/{version}/documents/uploadDocuments"
    headers = {"Authorization": access_token}
    if file_path.startswith("http://") or file_path.startswith("https://"):
        file_content = requests.get(file_path).content
        file_name = file_path.split("/")[-1]
    else:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_content = f.read()
    files = {"docs": (file_name, file_content)}
    response = requests.post(url, files=files, headers=headers)
    response.raise_for_status()
    data = response.json()
    document_id = data["data"][0]["fileLocation"]
    return document_id

@mcp.tool
def start_processing_job(document_id: str, api_domain_url: str, environment: str, version: str, username: str, password: str) -> str:
    """
    Starts a processing job for an uploaded document.

    Args:
        api_domain_url (str): Base domain of the API.
        environment (str): Target environment for ingestion.
        version (str): API version.
        username (str): Username for authentication.
        password (str): Password for authentication.
        document_id (str): The document ID returned from upload_document.

    Returns:
        str: The job ID created for processing the document.
    """
    print("start_processing_job tool used.")
    access_token = generate_access_token(api_domain_url, environment, version, username, password)
    url = f"https://{api_domain_url}/ingestion/{environment}/api/{version}/jobs/startJobs"
    headers = {"Authorization": access_token, "Content-Type": "application/json"}
    filename = os.path.basename(document_id)
    filetype = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
    second_half = document_id.split("xaas-storage")[-1]
    file_key="xaas-storage/"+second_half
    payload = {
        "docs": [
            {
                "originalFileName": filename,
                "fileType": filetype,
                "fileSize": None,
                "fileLocation": document_id,
                "fileKey": file_key
            }
        ],
        "jobType": 13,
        "tags": None
    }
    try:
        import urllib.parse
        from urllib.parse import urlparse
        import tempfile
        import shutil

        parsed_url = urlparse(document_id)
        if parsed_url.scheme.startswith('http'):
            with requests.get(document_id, stream=True) as r:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    shutil.copyfileobj(r.raw, tmp_file)
                    tmp_file.flush()
                    file_size = os.path.getsize(tmp_file.name)
                    payload["docs"][0]["fileSize"] = file_size
                    os.unlink(tmp_file.name)
        else:
            payload["docs"][0]["fileSize"] = None
    except Exception:
        payload["docs"][0]["fileSize"] = None

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    job_id = response.json()["data"][0]["jobId"]
    return job_id

@mcp.tool
def fetch_job_details(api_domain_url: str, environment: str, version: str, username: str, password: str, job_id: str) -> (str, dict):
    """
    Fetches the current status and metadata for a given job.

    Args:
        api_domain_url (str): Base domain of the API.
        environment (str): Target environment.
        version (str): API version.
        username (str): Username for authentication.
        password (str): Password for authentication.
        job_id (str): The job ID returned from start_processing_job.

    Returns:
        tuple: A tuple containing:
            - str: The job execution status.
            - dict: Detailed metadata about the job.
    """
    access_token = generate_access_token(api_domain_url, environment, version, username, password)
    print("...trying fetch_job_details tool! ")
    print(">>> pausing for 30 seconds <<<")
    time.sleep(30)
    print(">>> time elapsed! <<<")
    url = f"https://{api_domain_url}/ingestion/{environment}/api/{version}/job/getJobDetails/{job_id}"
    headers = {"Authorization": access_token}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    status = data["data"].get("executionStatusText", "")
    print(f">>> status of job: {status} <<<")
    metadata = data["data"]
    return status, metadata

@mcp.tool
def get_job_output(api_domain_url: str, environment: str, version: str, username: str, password: str, job_id: str) -> str:
    """
    Fetches the current status and metadata for a given job.

    Args:
        api_domain_url (str): Base domain of the API.
        environment (str): Target environment.
        version (str): API version.
        username (str): Username for authentication.
        password (str): Password for authentication.
        job_id (str): The job ID returned from start_processing_job.

    Returns:
        tuple: A tuple containing:
            - str: The job execution status.
            - dict: Detailed metadata about the job.
    """
    print("get_job_output tool used.")
    access_token = generate_access_token(api_domain_url, environment, version, username, password)
    url = f"https://{api_domain_url}/ingestion/{environment}/api/{version}/job/getJobOutput/{job_id}"
    headers = {"Authorization": access_token}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    # time.sleep(15)
    return str(data["data"])

@mcp.tool
def read_email_from_s3(
    aws_account_details: dict,
    s3_path: str
) -> str:
    """
    Reads the body text of an email file stored in AWS S3. Provide AWS credential dict and S3 path (s3://bucket/key).
    Args:
        aws_account_details (dict): Dictionary with AWS credentials.
        s3_path (str): S3 path to email file (e.g. 's3://mybucket/email.eml').
    Returns:
        str: Plain text content of the email body.
    Raises:
        ValueError: If path format or credentials missing/invalid.
    """
    import boto3
    from email import policy
    from email.parser import BytesParser
    session = boto3.Session(
        aws_access_key_id=aws_account_details.get("aws_access_key_id"),
        aws_secret_access_key=aws_account_details.get("aws_secret_access_key"),
        aws_session_token=aws_account_details.get("aws_session_token"),
        region_name=aws_account_details.get("region_name")
    )
    s3 = session.client("s3")
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Expected to start with 's3://'")
    path_parts = s3_path[5:].split("/", 1)
    if len(path_parts) != 2:
        raise ValueError("Invalid S3 path format. Expected 's3://bucket/key'")
    bucket, key = path_parts
    obj = s3.get_object(Bucket=bucket, Key=key)
    email_bytes = obj['Body'].read()
    msg = BytesParser(policy=policy.default).parsebytes(email_bytes)
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in disp:
                body = part.get_payload(decode=True)
                if body is not None:
                    return body.decode(part.get_content_charset() or "utf-8", errors="replace")
        return ""
    else:
        body = msg.get_payload(decode=True)
        if body is not None:
            return body.decode(msg.get_content_charset() or "utf-8", errors="replace")
        return ""

# Integration of new tool script as an MCP tool 
def read_email_from_s3_tool(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    bucket_name: str,
    email_file_path: str
) -> str:
    """
    Reads the body text of an email file stored in AWS S3 bucket and key. Use explicit credential and resource parameters. Returns the plain text content of the email body.
    Args:
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        region_name (str): AWS region name
        bucket_name (str): S3 bucket name
        email_file_path (str): S3 key (path) to email file
    Returns:
        str: Plain text content of the email body
    """
    import boto3
    import email
    from email import policy
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    response = s3_client.get_object(Bucket=bucket_name, Key=email_file_path)
    raw_email = response['Body'].read()
    msg = email.message_from_bytes(raw_email, policy=policy.default)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' and not part.get_filename():
                return part.get_content()
        return ''
    else:
        return msg.get_content()

@mcp.tool
def read_email_from_s3_explicit(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    bucket_name: str,
    email_file_path: str
) -> str:
    """
    Reads the body text of an email file stored in AWS S3 bucket with explicit credential parameters.
    Args:
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        region_name (str): AWS region name
        bucket_name (str): S3 bucket name
        email_file_path (str): S3 key (path) to email file
    Returns:
        str: Plain text content of the email body
    """
    return read_email_from_s3_tool(
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
        bucket_name,
        email_file_path
    )


# BEGIN: Sanctions-related tools

def check_individual_in_sanctions_list(individual_name: str) -> bool:
    """
    Checks if a given individual's name appears on the sanctioned individuals list.
    Args:
        individual_name (str): The full name of the individual to check.
    Returns:
        bool: True if the individual is on the sanctions list, False otherwise.
    Notes:
        - The check is case-insensitive and ignores leading/trailing whitespace.
    """
    print("tool used: check_individual_in_sanctions_list")
    sanctioned_individuals = {
        "John Doe",
        "Jane Smith",
        "Ali Hassan",
        "Maria Garcia"
    }
    return individual_name.strip().lower() in {name.lower() for name in sanctioned_individuals}


def draft_email_to_broker(broker_id: str, individual_name: str) -> str:
    """
    Drafts an email notification to a broker informing them that a specified individual
    has failed the sanctions check.
    Args:
        broker_id (str): The unique identifier of the broker to whom the email will be addressed.
        individual_name (str): The name of the individual who failed the sanctions check.
    Returns:
        str: A formatted email message string containing the notification.
    """
    print("tool used: draft_email_to_broker")
    email_content = (
        f"Subject: Sanctions Check Failure Notification\n\n"
        f"Dear Broker ({broker_id}),\n\n"
        f"We regret to inform you that the individual '{individual_name}' did not pass the sanctions check and appears on the sanctioned individuals list.\n"
        f"Please take appropriate actions in accordance with compliance requirements.\n\n"
        f"Best regards,\n"
        f"Sanctions Compliance Team"
    )
    return email_content

# ---- Begin Integrated Tools ----

@mcp.tool
def read_email_from_s3_simple(aws_account_details: str, s3_path: str) -> str:
    """
    Reads the body text from an email file stored in S3, given AWS credentials as a string and an S3 path. Returns the raw file text.
    Args:
        aws_account_details (str): String representation of a dictionary of AWS credentials.
        s3_path (str): The path to the S3 object. Format should be 's3://bucket/key'.
    Returns:
        str: Contents of the S3 file, decoded from bytes if necessary.
    """
    import boto3
    aws_info = eval(aws_account_details)
    session = boto3.Session(
        aws_access_key_id=aws_info.get('aws_access_key_id'),
        aws_secret_access_key=aws_info.get('aws_secret_access_key'),
        aws_session_token=aws_info.get('aws_session_token'),
        region_name=aws_info.get('region_name')
    )
    s3 = session.client('s3')
    bucket_name = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    body = obj['Body'].read()
    if isinstance(body, bytes):
        return body.decode('utf-8')
    else:
        return body

@mcp.tool
def upload_document_v2(api_base_url: str, api_key: str, file_path: str) -> str:
    """
    Uploads a document to an external API, returning a document ID.
    Args:
        api_base_url (str): Base URL of the external API.
        api_key (str): API key for authorization.
        file_path (str): Local path to the file to upload.
    Returns:
        str: Document ID for the uploaded file.
    """
    with open(file_path, 'rb') as f:
        files = {'file': f}
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.post(f'{api_base_url}/documents/upload', files=files, headers=headers)
        response.raise_for_status()
        document_id = response.json()['document_id']
        return document_id

@mcp.tool
def start_processing_job_v2(api_base_url: str, api_key: str, document_id: str) -> str:
    """
    Starts a document processing job via an external API, returning a job ID.
    Args:
        api_base_url (str): Base URL of the external API.
        api_key (str): API key for authorization.
        document_id (str): ID of the previously uploaded document.
    Returns:
        str: Job ID for the started job.
    """
    data = {'document_id': document_id}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    response = requests.post(f'{api_base_url}/jobs/start', json=data, headers=headers)
    response.raise_for_status()
    job_id = response.json()['job_id']
    return job_id

@mcp.tool
def fetch_job_details_v2(api_base_url: str, api_key: str, job_id: str) -> dict:
    """
    Fetches the job details from an external API, given job ID.
    Args:
        api_base_url (str): Base URL of the external API.
        api_key (str): API key for authorization.
        job_id (str): Job ID to retrieve details for.
    Returns:
        dict: Job details as returned by the API.
    """
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(f'{api_base_url}/jobs/{job_id}/details', headers=headers)
    response.raise_for_status()
    job_details = response.json()
    return job_details

@mcp.tool
def get_job_output_v2(api_base_url: str, api_key: str, job_id: str) -> dict:
    """
    Gets the job output from an external API, given job ID.
    Args:
        api_base_url (str): Base URL of the external API.
        api_key (str): API key for authorization.
        job_id (str): Job ID whose results to fetch.
    Returns:
        dict: Output data as returned by the API.
    """
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(f'{api_base_url}/jobs/{job_id}/output', headers=headers)
    response.raise_for_status()
    output_data = response.json()
    return output_data

# ---- Helper Function ----  
def load_excel_from_s3(s3_path: str) -> pd.ExcelFile:  
    """  
    Loads an Excel file from S3 into a Pandas ExcelFile object using environment's AWS authentication.  
  
    Args:  
        s3_path (str): S3 path to Excel file (e.g. 's3://mybucket/path/to/file.xlsx')  
  
    Returns:  
        pd.ExcelFile: ExcelFile object for reading sheets.  
    """  
    # Validate S3 path  
    if not s3_path.startswith("s3://"):  
        raise ValueError("Invalid S3 path format. Expected to start with 's3://'")  
  
    path_parts = s3_path[5:].split("/", 1)  
    if len(path_parts) != 2:  
        raise ValueError("Invalid S3 path format. Expected 's3://bucket/key'")  
      
    bucket, key = path_parts  
  
    # Use default AWS credentials from environment / IAM role  
    s3 = boto3.client("s3")  
  
    try:  
        obj = s3.get_object(Bucket=bucket, Key=key)  
    except s3.exceptions.NoSuchKey:  
        raise FileNotFoundError(f"{key} not found in bucket {bucket}")  
  
    file_data = BytesIO(obj["Body"].read())  
    return pd.ExcelFile(file_data, engine="openpyxl")  
  
  
# ---- MCP Tools ----  
@mcp.tool()  
def list_sheets(s3_path: str) -> List[str]:  
    """Lists all sheet names in an Excel file from S3."""  
    xl = load_excel_from_s3(s3_path)  
    return xl.sheet_names  
  
  
@mcp.tool()  
def create_data_dictionary(s3_path: str, sheet_name: str = None) -> Dict[str, Any]:  
    """Creates a data dictionary from an Excel file in S3."""  
    xl = load_excel_from_s3(s3_path)  
    sheets = xl.sheet_names  
      
    if sheet_name is None:  
        sheet_name = sheets[0]  
    elif sheet_name not in sheets:  
        raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {sheets}")  
      
    df = xl.parse(sheet_name)  
      
    type_mapping = {  
        "int64": "INTEGER",  
        "float64": "NUMERIC",  
        "object": "TEXT",  
        "datetime64[ns]": "DATE",  
        "bool": "BOOLEAN"  
    }  
      
    columns = {col: type_mapping.get(str(df[col].dtype), "TEXT") for col in df.columns}  
    return {"sheet_name": sheet_name, "columns": columns}  
  
  
@mcp.tool()  
def read_sample(s3_path: str, sheet_name: Optional[str] = None, max_rows: int = 10) -> str:  
    """Reads a sample of rows from an Excel sheet in S3."""  
    xl = load_excel_from_s3(s3_path)  
    df = xl.parse(sheet_name or xl.sheet_names[0])  
    if isinstance(df, dict):  
        df = next(iter(df.values()))  
    return df.head(max_rows).to_json(orient="records", date_format="iso")  
  
  
@mcp.tool()  
def read_sheet(s3_path: str, sheet_name: Optional[str] = None, as_csv: bool = False, max_rows: int = 5000) -> str:  
    """Reads data from an Excel sheet in S3."""  
    xl = load_excel_from_s3(s3_path)  
    df = xl.parse(sheet_name or xl.sheet_names[0])  
    if isinstance(df, dict):  
        df = next(iter(df.values()))  
    if max_rows and len(df) > max_rows:  
        df = df.head(max_rows)  
    return df.to_csv(index=False) if as_csv else df.to_json(orient="records", date_format="iso")  
  
  
@mcp.tool()  
def query_sheet(s3_path: str, sheet_name: str, sql_query: str, max_rows: int = 10000) -> str:  
    """Executes an SQL query on an Excel sheet's data from S3."""  
    xl = load_excel_from_s3(s3_path)  
    df = xl.parse(sheet_name)  
    if isinstance(df, dict):  
        df = next(iter(df.values()))  
      
    con = duckdb.connect()  
    print("SQL:", sql_query)  
    try:  
        con.register(sheet_name, df)  
        res_df = con.execute(sql_query).fetchdf()  
    finally:  
        con.close()  
      
    if max_rows and len(res_df) > max_rows:  
        res_df = res_df.head(max_rows)  
    print("Result")  
    print(res_df)  
    return res_df.to_json(orient="records", date_format="iso")  


if __name__ == "__main__":
    mcp.run(transport="sse", host=mcp_host, port=8001)  
