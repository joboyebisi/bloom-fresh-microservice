# File: bloom-fresh-microservice/app.py
# Required libraries: fastapi, uvicorn, trimesh, requests (or httpx for async)

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse 
from fastapi.middleware.cors import CORSMiddleware
import trimesh
import requests # Using requests for simplicity; httpx is better for async FastAPI
import io
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
# IMPORTANT: Replace "https://your-nextjs-app-name.vercel.app" with your actual deployed Next.js frontend URL
origins = [
    "http://localhost:3000",  # For local Next.js development
    "https://bloom-fresh-microservice-ap1bf01at-job-oyebisis-projects.vercel.app", # Placeholder for your deployed Next.js app
    # You might need to add your Vercel preview deployment URLs for the frontend too,
    # e.g., "https://bloom-fresh-git-your-branch-your-org.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"], # Specify methods or use ["*"] for all
    allow_headers=["*"], # Specify headers or use ["*"] for all
)

# Configuration - In a real app, use environment variables
PYTHON_MICROSERVICE_PORT = 8001 

@app.post("/convert")
async def convert_model_endpoint( 
    glb_url: str = Body(..., embed=True), 
    output_format: str = Body(..., embed=True) # "stl" or "obj"
):
    logger.info(f"Received conversion request for URL: {glb_url} to format: {output_format}")

    if output_format.lower() not in ["stl", "obj"]:
        logger.error(f"Invalid output_format: {output_format}")
        raise HTTPException(status_code=400, detail="Invalid output_format. Must be 'stl' or 'obj'.")

    try:
        # 1. Fetch GLB from URL
        logger.info(f"Fetching GLB model from: {glb_url}")
        # Using a timeout for the request
        response = requests.get(glb_url, stream=True, timeout=30) # 30 seconds timeout
        response.raise_for_status() 
        
        glb_data = response.content
        logger.info(f"Successfully fetched GLB data, size: {len(glb_data)} bytes.")

        # 2. Load GLB with Trimesh
        with io.BytesIO(glb_data) as glb_file_like:
            scene_or_mesh = trimesh.load(glb_file_like, file_type='glb')
        
        logger.info(f"Trimesh loaded model. Type: {type(scene_or_mesh)}")

        if isinstance(scene_or_mesh, trimesh.Scene):
            if not scene_or_mesh.geometry:
                logger.error("GLB scene is empty or contains no processable geometry.")
                raise HTTPException(status_code=400, detail="GLB scene is empty or unsupported.")
            mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
            logger.info(f"Converted Trimesh scene to a single mesh with {len(mesh.vertices)} vertices.")
        elif isinstance(scene_or_mesh, trimesh.Trimesh):
            mesh = scene_or_mesh
            logger.info(f"Loaded Trimesh mesh directly with {len(mesh.vertices)} vertices.")
        else:
            logger.error(f"Unsupported GLB content type after Trimesh load: {type(scene_or_mesh)}")
            raise HTTPException(status_code=500, detail=f"Unsupported GLB content type: {type(scene_or_mesh)}")

        if not mesh.vertices.size > 0:
             logger.error("Mesh has no vertices after processing.")
             raise HTTPException(status_code=400, detail="Processed mesh has no vertices.")

        # 3. Export to target format
        export_data = None
        content_type = "application/octet-stream"
        filename = f"converted_model_{trimesh.util.now()}.{output_format.lower()}"

        logger.info(f"Exporting mesh to {output_format.lower()}")
        if output_format.lower() == "stl":
            export_data = mesh.export(file_type="stl")
            content_type = "model/stl"
        elif output_format.lower() == "obj":
            export_data = mesh.export(file_type="obj")
        
        if not export_data:
            logger.error("Model conversion failed to produce data during export.")
            raise HTTPException(status_code=500, detail="Model conversion failed to produce data.")
        logger.info(f"Successfully exported to {output_format.lower()}. Data type: {type(export_data)}")

        # 4. Return as a streaming response
        response_headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}

        if isinstance(export_data, bytes): 
            return StreamingResponse(
                io.BytesIO(export_data),
                media_type=content_type,
                headers=response_headers
            )
        elif isinstance(export_data, str): 
            return StreamingResponse(
                io.BytesIO(export_data.encode('utf-8')),
                media_type=content_type, 
                headers=response_headers
            )
        else:
            logger.error(f"Unknown export data type: {type(export_data)}")
            raise HTTPException(status_code=500, detail="Unknown export data type after conversion.")

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching GLB model from URL: {glb_url}")
        raise HTTPException(status_code=504, detail="Timeout fetching GLB model from URL.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch GLB model: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch GLB model: {str(e)}")
    except trimesh.exceptions.TrimeshException as e: 
        logger.error(f"Trimesh processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid or corrupt GLB file, or Trimesh error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error during conversion") 
        raise HTTPException(status_code=500, detail=f"Internal server error during conversion: {str(e)}")

# To run: uvicorn app:app --reload --port 8001