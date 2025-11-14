"""
FastAPI Application Initialization and Router Registration
for Awetales Diarization System

This module sets up the main FastAPI application with comprehensive
configuration, middleware, and router registration for the real-time
Target Speaker Diarization + ASR System.
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
import uvicorn

from src.monitoring.health_check import HealthChecker
from src.monitoring.performance_monitor import PerformanceMonitor
from src.orchestration.resource_manager import ResourceManager
from src.api.endpoints import api_router
from src.api.websocket_handler import websocket_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log', mode='a')
    ]
)
logger = logging.getLogger("awetales_api")


class AwetalesAPI:
    """Main FastAPI application class for Awetales Diarization System"""
    
    def __init__(self):
        self.app = None
        self.health_checker = None
        self.performance_monitor = None
        self.resource_manager = None
        
    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan management"""
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        # Create FastAPI app with custom lifespan
        self.app = FastAPI(
            title="Awetales Diarization System API",
            description="Real-time Target Speaker Diarization + ASR System for Awetales Campus Drive",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            lifespan=lifespan
        )
        
        # Configure application
        self._setup_middleware()
        self._setup_exception_handlers()
        self._setup_routes()
        self._customize_openapi()
        
        return self.app
    
    async def _startup(self):
        """Initialize system components on startup"""
        logger.info("Starting Awetales Diarization System API...")
        
        try:
            # Initialize monitoring components
            self.health_checker = HealthChecker()
            self.performance_monitor = PerformanceMonitor()
            
            # Initialize resource manager
            self.resource_manager = ResourceManager()
            await self.resource_manager.initialize()
            
            # Start health monitoring
            await self.health_checker.start_monitoring()
            
            logger.info("Awetales Diarization System API started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start API: {str(e)}")
            raise
    
    async def _shutdown(self):
        """Cleanup system components on shutdown"""
        logger.info("Shutting down Awetales Diarization System API...")
        
        try:
            if self.health_checker:
                await self.health_checker.stop_monitoring()
            
            if self.resource_manager:
                await self.resource_manager.cleanup()
                
            logger.info("Awetales Diarization System API shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def _setup_middleware(self):
        """Configure application middleware"""
        
        # CORS Middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://awetales.com",
                "https://*.awetales.com"
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["*"],
            max_age=600,
        )
        
        # GZip Compression Middleware
        self.app.add_middleware(
            GZipMiddleware,
            minimum_size=1000,
        )
        
        # Request/Response Logging Middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Middleware for logging requests and responses"""
            start_time = time.time()
            
            # Generate request ID
            request_id = request.headers.get('X-Request-ID', str(int(start_time * 1000)))
            
            # Log request
            logger.info(
                f"Request {request_id}: {request.method} {request.url} - "
                f"Client: {request.client.host if request.client else 'unknown'}"
            )
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                # Log response
                logger.info(
                    f"Response {request_id}: {response.status_code} - "
                    f"Process Time: {process_time:.3f}s"
                )
                
                # Add performance headers
                response.headers["X-Process-Time"] = str(process_time)
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(
                    f"Error {request_id}: {str(e)} - "
                    f"Process Time: {process_time:.3f}s"
                )
                raise
        
        # Rate Limiting Middleware (placeholder for actual implementation)
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            """Placeholder for rate limiting middleware"""
            # TODO: Implement actual rate limiting based on client IP and endpoint
            # For now, just pass through
            return await call_next(request)
    
    def _setup_exception_handlers(self):
        """Configure global exception handlers"""
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler for unhandled exceptions"""
            logger.error(
                f"Unhandled exception in {request.method} {request.url}: {str(exc)}",
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": request.headers.get('X-Request-ID', 'unknown')
                }
            )
        
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """Handler for request validation errors"""
            logger.warning(
                f"Validation error in {request.method} {request.url}: {str(exc)}"
            )
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "Validation Error",
                    "message": "Invalid request parameters",
                    "details": exc.errors(),
                    "request_id": request.headers.get('X-Request-ID', 'unknown')
                }
            )
        
        @self.app.exception_handler(404)
        async def not_found_exception_handler(request: Request, exc: Exception):
            """Handler for 404 errors"""
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "Not Found",
                    "message": f"Endpoint {request.url} not found",
                    "request_id": request.headers.get('X-Request-ID', 'unknown')
                }
            )
    
    def _setup_routes(self):
        """Register all API routes and routers"""
        
        # Health check endpoint
        @self.app.get("/", include_in_schema=False)
        async def root():
            """Root endpoint with basic system info"""
            return {
                "message": "Awetales Diarization System API",
                "version": "1.0.0",
                "status": "operational"
            }
        
        @self.app.get("/health", tags=["monitoring"])
        async def health_check():
            """Comprehensive health check endpoint"""
            if not self.health_checker:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"status": "unhealthy", "message": "Health checker not initialized"}
                )
            
            health_status = await self.health_checker.get_system_status()
            
            if health_status["overall_status"] == "healthy":
                return health_status
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=health_status
                )
        
        @self.app.get("/health/live", tags=["monitoring"])
        async def liveness_probe():
            """Kubernetes liveness probe endpoint"""
            return {"status": "alive"}
        
        @self.app.get("/health/ready", tags=["monitoring"])
        async def readiness_probe():
            """Kubernetes readiness probe endpoint"""
            if (self.health_checker and 
                await self.health_checker.is_system_ready()):
                return {"status": "ready"}
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"status": "not_ready"}
                )
        
        # Metrics endpoint
        @self.app.get("/metrics", tags=["monitoring"])
        async def get_metrics():
            """System metrics endpoint"""
            if not self.performance_monitor:
                return {"error": "Performance monitor not initialized"}
            
            metrics = self.performance_monitor.get_current_metrics()
            return metrics
        
        # Register API routers
        self.app.include_router(api_router, prefix="/api/v1")
        self.app.include_router(websocket_router, prefix="/api/v1")
    
    def _customize_openapi(self):
        """Customize OpenAPI schema"""
        
        def custom_openapi():
            if self.app.openapi_schema:
                return self.app.openapi_schema
            
            openapi_schema = get_openapi(
                title=self.app.title,
                version=self.app.version,
                description=self.app.description,
                routes=self.app.routes,
            )
            
            # Customize OpenAPI schema
            openapi_schema["info"]["contact"] = {
                "name": "Awetales Engineering",
                "email": "engineering@awetales.com",
                "url": "https://awetales.com"
            }
            
            openapi_schema["info"]["license"] = {
                "name": "Proprietary",
                "url": "https://awetales.com/license"
            }
            
            # Add servers
            openapi_schema["servers"] = [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.awetales.com",
                    "description": "Production server"
                }
            ]
            
            # Add security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
            
            # Add global security
            openapi_schema["security"] = [{"BearerAuth": []}]
            
            self.app.openapi_schema = openapi_schema
            return self.app.openapi_schema
        
        self.app.openapi = custom_openapi


def create_application() -> FastAPI:
    """Factory function to create the FastAPI application"""
    api = AwetalesAPI()
    return api.create_app()


# Global application instance
app = create_application()


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server using uvicorn"""
    
    uvicorn_config = uvicorn.Config(
        app="src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        timeout_notify=30,
        timeout_graceful_shutdown=30,
    )
    
    server = uvicorn.Server(uvicorn_config)
    
    logger.info(f"Starting Awetales Diarization API server on {host}:{port}")
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise


if __name__ == "__main__":
    # For development
    start_server(host="127.0.0.1", port=8000, reload=True)