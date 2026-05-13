"""
FastAPI Main Application
Entry point for the Chatbot API Server
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import auth, users, sessions, chat
from chatbot.config import config as app_config
from chatbot.core.db import init_db
from chatbot.core.watcher import app_watcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    print("🚀 Starting Chatbot API Server...")
    init_db()
    app_watcher.start()
    print("✅ API Server ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    app_watcher.stop()
    print("👋 Goodbye!")


app = FastAPI(
    title="Law Chatbot API",
    description="""
## Law Chatbot - Multi-Agent RAG System API

API Backend cho hệ thống Chatbot thông minh sử dụng công nghệ Multi-Agent và RAG.

### Tính năng chính:
- 🔐 **Xác thực JWT**: Đăng ký, đăng nhập bảo mật
- 👤 **Quản lý User**: Cập nhật profile, đổi mật khẩu, xóa tài khoản
- 💬 **Chat AI**: Hội thoại với AI agent thông minh
- 📁 **Upload File**: Tải lên PDF để AI phân tích
- 📜 **Lịch sử Chat**: Quản lý các phiên hội thoại

### Xác thực:
Sử dụng JWT Bearer Token. Sau khi đăng nhập, thêm header:
```
Authorization: Bearer <your_token>
```
    """,
    version="1.0.0",
    contact={
        "name": "CUSC Chatbot Team",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)


# CORS Middleware
#
# With cookie-based auth we MUST whitelist explicit origins — the combination
# of `allow_origins=["*"]` and `allow_credentials=True` is invalid per the
# CORS spec and most browsers silently drop the credentials. Configure
# `CORS_ALLOW_ORIGINS` (comma-separated) in the environment to add staging
# or preview-deploy origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(chat.router)


@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Welcome endpoint with API information"
)
async def root():
    """
    Root endpoint - returns API information
    """
    return {
        "message": "Welcome to Law ChatbotAPI",
        "version": "1.0.0",
        "docs": "/docs",  # http://127.0.0.1:8000/docs#/
        "redoc": "/redoc"
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Check if the API server is running"
)
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "service": "chatbot-api"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    """
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if app.debug else "An unexpected error occurred"
        }
    )


# Run with: uvicorn backend.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
