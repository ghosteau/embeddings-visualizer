# =====================================================================================
# MAIN APPLICATION RUNNER
# =====================================================================================
import uvicorn

from Backend.backend import EXPORT_DIR

if __name__ == "__main__":
    print("Starting Vector Embedding Visualizer Backend v1.0.0")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Interactive API testing at: http://localhost:8000/redoc")

    # Ensure directories exist
    EXPORT_DIR.mkdir(exist_ok=True)

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
