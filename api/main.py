from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as chat_router

app = FastAPI(
    title="Qanary AI Clinical Agent",
    description="Enterprise REST API for Qanary Quantum ML Agent",
    version="1.0.0"
)

# CORS Middleware to allow frontend (React/Vue/Flutter) to communicate with the API
# In production, replace ["*"] with your specific frontend domain (e.g., ["https://app.qanary.ai"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the routes
app.include_router(chat_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    """Simple health check endpoint for Cloud Run load balancers."""
    return {"status": "healthy", "version": "1.0.0"}
