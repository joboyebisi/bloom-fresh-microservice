from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import httpx
import trimesh
import os
import uuid
import shutil
import logging
from typing import Optional
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin
cred = credentials.Certificate("path/to/your/firebase-credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-bucket-name.appspot.com'
})

print("Microservice app.py starting...")
# Imports like FastAPI and CORSMiddleware should only be done once at the top.
# No need to re-import them here.
print("Initial imports successful.")

# --- Initialize FastAPI app ONCE ---
app = FastAPI(title="Bloom Fresh GLB to STL Conversion Microservice")
print("FastAPI app initialized.")

# --- CORS Configuration ---
origins = [
    "https://bloom-v0.vercel.app",    # Your deployed Next.js frontend
    "http://localhost:3000",        # Your local Next.js development server
    # Add any other frontend origins if necessary
]
print(f"CORS origins defined: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware added successfully.")

# --- Pydantic Models for Request/Response ---
class ConversionRequest(BaseModel):
    glb_url: HttpUrl
    output_format: str = "stl"  # Default to STL
    optimize_mesh: Optional[bool] = True  # Optional mesh optimization
    store_in_firebase: Optional[bool] = True  # Whether to store files in Firebase

class ConversionResponse(BaseModel):
    stl_download_url: str
    glb_storage_url: Optional[str] = None
    stl_storage_url: Optional[str] = None
    message: str

# --- Helper Functions ---
TEMP_DIR = "temp_conversion_files"
# ... and so on, the rest of your file was fine from here.
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_files(paths: list):
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            logger.info(f"Cleaned up: {path}")
        except Exception as e:
            logger.error(f"Error cleaning up {path}: {e}")


async def download_file(url: str, destination: str):
    async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout
        try:
            response = await client.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(destination, 'wb') as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded file to {destination}")
        except httpx.RequestError as e:
            logger.error(f"Error downloading file from {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download GLB file from URL: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during download: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching the GLB: {e}")

def upload_to_firebase(file_path: str, file_type: str, unique_id: str) -> str:
    """Upload a file to Firebase Storage and return its download URL"""
    try:
        bucket = storage.bucket()
        blob_name = f"{file_type}/{unique_id}/{os.path.basename(file_path)}"
        blob = bucket.blob(blob_name)
        
        # Upload the file
        blob.upload_from_filename(file_path)
        
        # Generate a signed URL that expires in 1 hour
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(hours=1),
            method="GET"
        )
        
        logger.info(f"Successfully uploaded {file_type} to Firebase: {blob_name}")
        return url
    except Exception as e:
        logger.error(f"Error uploading to Firebase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload {file_type} to Firebase: {str(e)}")

def process_mesh(mesh, optimize: bool = True):
    """Process the mesh for optimal STL conversion"""
    try:
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError("GLB scene contains no geometry to convert.")
            
            # Combine all geometries into a single mesh
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        
        if optimize:
            # Remove duplicate vertices
            mesh.process(validate=True)
            # Remove zero-area faces
            mesh.remove_degenerate_faces()
            # Remove duplicate faces
            mesh.remove_duplicate_faces()
            # Remove infinite values
            mesh.remove_infinite_values()
            
        return mesh
    except Exception as e:
        logger.error(f"Error processing mesh: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process mesh: {str(e)}")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Bloom Fresh GLB to STL Conversion Microservice is running!"}

@app.post("/convert", response_model=ConversionResponse)
async def convert_model_endpoint(request: ConversionRequest, background_tasks: BackgroundTasks):
    if request.output_format.lower() != "stl":
        raise HTTPException(status_code=400, detail="This service only supports STL conversion.")

    unique_id = str(uuid.uuid4())
    temp_subdir = os.path.join(TEMP_DIR, unique_id)
    os.makedirs(temp_subdir, exist_ok=True)

    input_glb_path = os.path.join(temp_subdir, f"input_{unique_id}.glb")
    output_file_path = os.path.join(temp_subdir, f"output_{unique_id}.stl")
    
    # Add files to be cleaned up by the background task
    background_tasks.add_task(cleanup_files, [temp_subdir])

    try:
        # 1. Download the GLB file from the provided URL
        logger.info(f"Downloading GLB from: {request.glb_url}")
        await download_file(str(request.glb_url), input_glb_path)
        logger.info(f"GLB file downloaded to: {input_glb_path}")

        if not os.path.exists(input_glb_path) or os.path.getsize(input_glb_path) == 0:
            logger.error(f"Failed to download or GLB file is empty: {input_glb_path}")
            raise HTTPException(status_code=500, detail="Failed to download GLB file or file is empty.")

        # 2. Load and process the GLB model
        logger.info(f"Loading model with trimesh: {input_glb_path}")
        try:
            mesh = trimesh.load_mesh(input_glb_path)
            mesh = process_mesh(mesh, optimize=request.optimize_mesh)
            logger.info("Model loaded and processed successfully")
        except Exception as e:
            logger.error(f"Error loading/processing GLB: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process GLB model: {str(e)}")

        # 3. Export to STL
        logger.info(f"Exporting model to STL at {output_file_path}")
        try:
            mesh.export(file_type='stl', file_obj=output_file_path)
            if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
                raise ValueError("STL export failed or produced an empty file.")
            logger.info("Model exported to STL successfully")
        except Exception as e:
            logger.error(f"Error exporting to STL: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export model to STL: {str(e)}")
            
        # 4. Upload to Firebase if requested
        stl_download_url = None
        glb_storage_url = None
        stl_storage_url = None
        
        if request.store_in_firebase:
            # Upload both GLB and STL to Firebase
            glb_storage_url = upload_to_firebase(input_glb_path, "glb", unique_id)
            stl_storage_url = upload_to_firebase(output_file_path, "stl", unique_id)
            stl_download_url = stl_storage_url
        else:
            # Return the STL file directly
            return FileResponse(
                path=output_file_path,
                filename=f"converted_model_{unique_id}.stl",
                media_type="model/stl"
            )

        return ConversionResponse(
            stl_download_url=stl_download_url,
            glb_storage_url=glb_storage_url,
            stl_storage_url=stl_storage_url,
            message="Conversion completed successfully"
        )

    except HTTPException as http_exc:
        # Re-raise HTTPException to ensure FastAPI handles it correctly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"An unexpected error occurred in /convert endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Main execution for local development (optional) ---
if __name__ == "__main__":
    import uvicorn
    # Make sure TEMP_DIR exists when running locally
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8001)
