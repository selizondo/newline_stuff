"""FastAPI application for Synthetic Data Resume Coach."""

from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logfire.configure()
    logfire.info("Starting Synthetic Data Resume Coach API")
    yield
    logfire.info("Shutting down Synthetic Data Resume Coach API")


app = FastAPI(
    title="Synthetic Data Resume Coach",
    description="""
    Synthetic Data Resume Coach API for analyzing resumes against job descriptions.

    Features:
    - Resume-job fit analysis
    - Skills overlap calculation (Jaccard similarity)
    - Failure mode detection (6 metrics)
    - LLM-based quality assessment
    - Structured recommendations
    """,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Synthetic Data Resume Coach API",
        "version": "0.1.0",
        "status": "healthy",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
