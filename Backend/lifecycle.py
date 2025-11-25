# =====================================================================================
# APPLICATION LIFECYCLE MANAGEMENT
# =====================================================================================
from Backend.backend import viz, app


def cleanup_on_shutdown():
    """Cleanup function to run on application shutdown"""
    try:
        # Clear model from memory
        viz.clear_model()
        print("Application shutdown: Model cleared from memory")
    except Exception as e:
        print(f"Error during shutdown cleanup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    cleanup_on_shutdown()
