from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import httpx # For making async HTTP requests to download the GLB file
import trimesh # Assuming you'll use trimesh for conversion
import os
import uuid
import shutil # For cleaning up temporary files

# Initialize FastAPI app
app = FastAPI(title="3D Model Conversion Microservice")

# --- CORS Configuration ---
# Define the list of allowed origins (your frontend applications)
origins = [
    "https://bloom-v0.vercel.app",    # Your deployed Next.js frontend
    "http://localhost:3000",        # Your local Next.js development server
    # Add any other frontend origins if necessary
]

# Add the CORS middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Allow specific origins
    allow_credentials=True,       # Allow cookies to be included in requests
    allow_methods=["*"],            # Allow all standard HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all headers
)

# --- Pydantic Models for Request/Response ---
class ConversionRequest(BaseModel):
    glb_url: HttpUrl
    output_format: str # Should be "stl" or "obj"

# --- Helper Functions ---
TEMP_DIR = "temp_conversion_files"
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_files(paths: list):
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            print(f"Cleaned up: {path}")
        except Exception as e:
            print(f"Error cleaning up {path}: {e}")


async def download_file(url: str, destination: str):
    async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout
        try:
            response = await client.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(destination, 'wb') as f:
                f.write(response.content)
        except httpx.RequestError as e:
            print(f"Error downloading file from {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download GLB file from URL: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching the GLB: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Bloom Fresh 3D Model Conversion Microservice is running!"}

@app.post("/convert")
async def convert_model_endpoint(request: ConversionRequest, background_tasks: BackgroundTasks):
    if request.output_format.lower() not in ["stl", "obj"]:
        raise HTTPException(status_code=400, detail="Invalid output_format. Must be 'stl' or 'obj'.")

    unique_id = uuid.uuid4()
    temp_subdir = os.path.join(TEMP_DIR, str(unique_id))
    os.makedirs(temp_subdir, exist_ok=True)

    input_glb_path = os.path.join(temp_subdir, f"input_{unique_id}.glb")
    output_file_path = os.path.join(temp_subdir, f"output_{unique_id}.{request.output_format.lower()}")
    
    # Add files to be cleaned up by the background task
    background_tasks.add_task(cleanup_files, [temp_subdir])

    try:
        # 1. Download the GLB file from the provided URL
        print(f"Attempting to download GLB from: {request.glb_url}")
        await download_file(str(request.glb_url), input_glb_path)
        print(f"GLB file downloaded to: {input_glb_path}")

        if not os.path.exists(input_glb_path) or os.path.getsize(input_glb_path) == 0:
            print(f"Failed to download or GLB file is empty: {input_glb_path}")
            raise HTTPException(status_code=500, detail="Failed to download GLB file or file is empty.")

        # 2. Load the GLB model using trimesh
        print(f"Loading model with trimesh: {input_glb_path}")
        try:
            mesh = trimesh.load_mesh(input_glb_path)
            if isinstance(mesh, trimesh.Scene): # If it's a scene, try to get a single geometry
                 if len(mesh.geometry) > 0:
                    # Concatenate all geometries into a single mesh
                    # This is a common approach, but might need adjustment based on your models
                    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
                 else: # No geometry in scene
                    raise ValueError("GLB scene contains no geometry to convert.")
            elif not hasattr(mesh, 'export'): # Check if it's a valid mesh object
                 raise ValueError("Loaded GLB is not a valid mesh object.")
            print("Model loaded successfully with trimesh.")
        except Exception as e:
            print(f"Error loading GLB with trimesh: {e}")
            # Before raising, check if the temp_subdir still exists for cleanup context
            if not os.path.exists(temp_subdir):
                print(f"Temp subdir {temp_subdir} was unexpectedly removed before trimesh loading error.")
            raise HTTPException(status_code=500, detail=f"Failed to load GLB model for conversion: {str(e)}")


        # 3. Export to the desired format (STL or OBJ)
        print(f"Exporting model to {request.output_format.lower()} at {output_file_path}")
        try:
            export_result = mesh.export(file_type=request.output_format.lower(), file_obj=output_file_path)
            if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
                 raise ValueError(f"Trimesh export to {request.output_format.lower()} failed or produced an empty file.")
            print(f"Model exported to {request.output_format.lower()} successfully.")
        except Exception as e:
            print(f"Error exporting model with trimesh: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export model to {request.output_format.lower()}: {str(e)}")
            
        # 4. Return the converted file
        # The actual file sending will be handled by FileResponse.
        # The background task will clean up the file after response is sent.
        return FileResponse(
            path=output_file_path,
            filename=f"converted_model_{unique_id}.{request.output_format.lower()}",
            media_type=f"model/{request.output_format.lower()}" if request.output_format.lower() == "stl" else "application/octet-stream",
            # background=background_tasks # FileResponse can take background tasks directly too
        )

    except HTTPException as http_exc:
        # Re-raise HTTPException to ensure FastAPI handles it correctly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred in /convert endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Main execution for local development (optional) ---
if __name__ == "__main__":
    import uvicorn
    # Make sure TEMP_DIR exists when running locally
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8001)
