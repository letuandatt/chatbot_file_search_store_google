"""
Chat Router
Handles chat interactions with the AI agent
"""
import uuid
import os
import base64
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from langchain_core.messages import HumanMessage

from backend.models.chat import (
    ChatRequest, ChatResponse, FileUploadResponse,
    TextChatRequest, PdfChatRequest, ImageChatRequest
)
from backend.dependencies import get_current_user, get_app_container
from chatbot.core.history import save_session_message
from chatbot.core.file_store import save_pdf_to_mongo
from chatbot.core.queue import enqueue_pdf_processing, is_worker_enabled


router = APIRouter(prefix="/chat", tags=["Chat"])

# Mai làm cái xác thực email người dùng. Mở bên antigravity trước. Hỏi kĩ lại is_active là user đang hoạt động hay ý nghĩa gì
@router.post(
    "/text",
    response_model=ChatResponse,
    summary="Send a text-only chat message",
    description="Send a text message to the AI chatbot (no image support)"
)
async def chat_text(
    request: TextChatRequest,
    current_user: dict = Depends(get_current_user),
    app=Depends(get_app_container)
):
    """
    Chat với AI bằng tin nhắn văn bản.
    
    - **message**: Nội dung tin nhắn
    - **session_id**: ID phiên chat (tạo mới nếu không cung cấp)
    """
    if not app.agent_executor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI agent is not ready. Please try again later."
        )
    
    user_id = str(current_user["_id"])
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        user_profile = app.memory_service.get_profile(user_id) if app.memory_service else None
        
        inputs = {
            "messages": [HumanMessage(content=request.message)],
            "user_info": user_profile or "Chưa có thông tin.",
            "image_path": None  # No image for text chat
        }
        
        result = app.agent_executor.invoke(
            inputs,
            config={"configurable": {"session_id": session_id, "user_id": user_id}}
        )
        
        # Extract thinking steps from all messages
        from backend.models.chat import ThinkingStep
        thinking_steps = []
        
        for msg in result.get("messages", []):
            msg_name = getattr(msg, 'name', None)
            msg_content = getattr(msg, 'content', '')
            msg_type = type(msg).__name__
            
            if msg_type == "HumanMessage":
                continue  # Skip user message
            
            if msg_name == "Supervisor" or msg_type == "AIMessage" and not msg_name:
                # Supervisor deciding which agent to use
                thinking_steps.append(ThinkingStep(
                    agent="Supervisor",
                    action="Phân tích câu hỏi và chuyển đến chuyên gia phù hợp",
                    detail=None
                ))
            elif msg_name:
                # Worker agent response
                agent_actions = {
                    "LawResearcher": "Tra cứu văn bản quy phạm pháp luật",
                    "PersonalAnalyst": "Tìm kiếm trong tài liệu cá nhân",
                    "GeneralResponder": "Xử lý câu hỏi xã giao",
                    "VisionAnalyst": "Phân tích hình ảnh"
                }
                thinking_steps.append(ThinkingStep(
                    agent=msg_name,
                    action=agent_actions.get(msg_name, "Xử lý yêu cầu"),
                    detail=msg_content[:100] + "..." if len(msg_content) > 100 else msg_content
                ))
        
        last_message = result["messages"][-1]
        response_text = last_message.content
        agent_name = getattr(last_message, 'name', None)
        
        save_session_message(
            session_id=session_id,
            user_id=user_id,
            question=request.message,
            answer=response_text,
            thinking_steps=[{"agent": s.agent, "action": s.action, "detail": s.detail} for s in thinking_steps] if thinking_steps else None
        )
        
        if app.memory_service:
            app.memory_service.update_profile_background(user_id, request.message)
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            agent_name=agent_name,
            thinking_steps=thinking_steps if thinking_steps else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.post(
    "/image",
    response_model=ChatResponse,
    summary="Send a chat message with an image",
    description="Upload an image and ask AI to analyze it"
)
async def chat_image(
    message: str = Form(..., min_length=1, max_length=10000, description="Câu hỏi về ảnh"),
    image: UploadFile = File(..., description="File ảnh (jpg, png, gif, webp)"),
    session_id: Optional[str] = Form(None, description="ID phiên chat"),
    current_user: dict = Depends(get_current_user),
    app = Depends(get_app_container)
):
    """
    Chat với AI kèm hình ảnh để phân tích.
    
    - **message**: Câu hỏi về ảnh (ví dụ: "Mô tả ảnh này")
    - **image**: File ảnh upload trực tiếp (jpg, png, gif, webp)
    - **session_id**: ID phiên chat (tạo mới nếu không cung cấp)
    """
    if not app.agent_executor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI agent is not ready. Please try again later."
        )
    
    # Validate image type
    allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chỉ hỗ trợ các định dạng ảnh: jpg, png, gif, webp. Đã nhận: {image.content_type}"
        )
    
    user_id = str(current_user["_id"])
    session_id = session_id or str(uuid.uuid4())
    
    # Save uploaded image temporarily
    image_path = None
    try:
        # Determine file extension
        ext = image.filename.split('.')[-1] if image.filename else 'jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            content = await image.read()
            tmp.write(content)
            image_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi đọc file ảnh: {e}"
        )
    
    try:
        user_profile = app.memory_service.get_profile(user_id) if app.memory_service else None
        
        inputs = {
            "messages": [HumanMessage(content=message)],
            "user_info": user_profile or "Chưa có thông tin.",
            "image_path": image_path
        }
        
        result = app.agent_executor.invoke(
            inputs,
            config={"configurable": {"session_id": session_id, "user_id": user_id}}
        )
        
        last_message = result["messages"][-1]
        response_text = last_message.content
        agent_name = getattr(last_message, 'name', None)
        
        save_session_message(
            session_id=session_id,
            user_id=user_id,
            question=message,
            answer=response_text,
            image_gridfs_id=image_path
        )
        
        if app.memory_service:
            app.memory_service.update_profile_background(user_id, message)
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            agent_name=agent_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass


@router.post(
    "/pdf",
    response_model=ChatResponse,
    summary="Send a chat message about uploaded PDF",
    description="Send a message to query content from uploaded PDF files"
)
async def chat_pdf(
    request: PdfChatRequest,
    current_user: dict = Depends(get_current_user),
    app = Depends(get_app_container)
):
    """
    Chat với AI về nội dung PDF đã tải lên.
    
    - **message**: Câu hỏi về nội dung PDF
    - **session_id**: ID phiên chat (tạo mới nếu không cung cấp)
    - **file_id**: ID của file PDF đã upload (tùy chọn, nếu không có sẽ tìm trong tất cả file của user)
    """
    if not app.agent_executor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI agent is not ready. Please try again later."
        )
    
    user_id = str(current_user["_id"])
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        user_profile = app.memory_service.get_profile(user_id) if app.memory_service else None
        
        # Add hint for PDF context if file_id provided
        message_content = request.message
        if request.file_id:
            message_content = f"[Tìm trong file PDF: {request.file_id}] {request.message}"
        
        inputs = {
            "messages": [HumanMessage(content=message_content)],
            "user_info": user_profile or "Chưa có thông tin.",
            "image_path": None
        }
        
        result = app.agent_executor.invoke(
            inputs,
            config={"configurable": {"session_id": session_id, "user_id": user_id}}
        )
        
        last_message = result["messages"][-1]
        response_text = last_message.content
        agent_name = getattr(last_message, 'name', None)
        
        save_session_message(
            session_id=session_id,
            user_id=user_id,
            question=request.message,
            answer=response_text
        )
        
        if app.memory_service:
            app.memory_service.update_profile_background(user_id, request.message)
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            agent_name=agent_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="Upload a PDF file",
    description="Upload a PDF file for the chatbot to use in responses"
)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a PDF file for RAG (Retrieval-Augmented Generation).
    
    The file will be processed in the background and become available
    for queries once processing is complete.
    
    - **file**: PDF file to upload
    - **session_id**: Optional session ID to associate the file with
    """
    user_id = str(current_user["_id"])
    session_id = session_id or str(uuid.uuid4())
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Save file temporarily
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Save to MongoDB/GridFS with original filename
        file_id = save_pdf_to_mongo(temp_path, session_id, user_id, original_filename=file.filename)

        if not file_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save file"
            )

        # When the optional arq-based PDF worker is enabled, push the
        # row directly onto the queue. Otherwise the in-process
        # watcher will discover it on its next poll (≤5 s).
        if is_worker_enabled():
            await enqueue_pdf_processing(str(file_id))

        return FileUploadResponse(
            file_id=str(file_id),
            filename=file.filename,
            status="processing",
            message="File uploaded successfully. Processing in background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@router.get(
    "/file/{file_id}/status",
    summary="Check file processing status",
    description="Check the processing status of an uploaded file"
)
async def check_file_status(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Check the status of a file that was uploaded for processing.
    
    Returns:
    - status: "uploaded" | "processing" | "processed" | "error_processing"
    - filename: Original filename
    - file_store_name: (only if processed) The file store name
    """
    from chatbot.core.db import DB_DOCUMENTS_COLLECTION
    from bson import ObjectId
    
    user_id = str(current_user["_id"])
    
    if DB_DOCUMENTS_COLLECTION is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        doc = DB_DOCUMENTS_COLLECTION.find_one({
            "_id": ObjectId(file_id),
            "user_id": user_id
        })
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return {
            "file_id": file_id,
            "filename": doc.get("filename"),
            "status": doc.get("status", "uploaded"),
            "file_store_name": doc.get("file_store_name"),
            "error": doc.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking file status: {str(e)}"
        )


@router.get(
    "/files",
    summary="List all uploaded files",
    description="Get a list of all files uploaded by the current user"
)
async def list_user_files(
    current_user: dict = Depends(get_current_user)
):
    """
    List all files uploaded by the current user.
    Returns file info including session name.
    """
    from chatbot.core.db import DB_DOCUMENTS_COLLECTION, get_mongo_collection
    from pymongo import DESCENDING
    
    user_id = str(current_user["_id"])
    
    if DB_DOCUMENTS_COLLECTION is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        # Get sessions for mapping session_id -> title
        sessions_coll = get_mongo_collection("sessions")
        session_titles = {}
        if sessions_coll is not None:
            for s in sessions_coll.find({"user_id": user_id}, {"session_id": 1, "title": 1}):
                sid = s.get("session_id", "")
                if sid:
                    session_titles[sid] = s.get("title") or f"Phiên {sid[:8]}..."
        
        # Get files
        cursor = DB_DOCUMENTS_COLLECTION.find(
            {"user_id": user_id},
            {"filename": 1, "status": 1, "session_id": 1, "created_at": 1, "file_gridfs_id": 1}
        ).sort("created_at", -1)
        
        files = []
        for doc in cursor:
            session_id = doc.get("session_id") or ""
            created_at = doc.get("created_at")
            # Handle datetime serialization
            if created_at and hasattr(created_at, 'isoformat'):
                created_at_str = created_at.isoformat()
            elif created_at:
                created_at_str = str(created_at)
            else:
                created_at_str = None
            
            files.append({
                "file_id": str(doc["_id"]),
                "filename": doc.get("filename", "unknown"),
                "status": doc.get("status", "unknown"),
                "session_id": session_id,
                "session_name": session_titles.get(session_id, f"Phiên {session_id[:8]}..." if len(session_id) >= 8 else "Không rõ"),
                "created_at": created_at_str,
                "has_file": bool(doc.get("file_gridfs_id"))
            })
        
        return {"files": files, "total": len(files)}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing files: {str(e)}"
        )


from fastapi.responses import StreamingResponse
import io

@router.get(
    "/files/{file_id}/download",
    summary="Download uploaded file",
    description="Download the original uploaded file"
)
async def download_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Download the original file from GridFS.
    """
    from chatbot.core.db import DB_DOCUMENTS_COLLECTION, FS
    from bson import ObjectId
    
    user_id = str(current_user["_id"])
    
    if DB_DOCUMENTS_COLLECTION is None or FS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        # Find the document
        doc = DB_DOCUMENTS_COLLECTION.find_one({
            "_id": ObjectId(file_id),
            "user_id": user_id
        })
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        gridfs_id = doc.get("file_gridfs_id")
        if not gridfs_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File content not available"
            )
        
        # Get file from GridFS
        try:
            grid_file = FS.get(ObjectId(gridfs_id))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File content not found in storage"
            )
        
        # Read file content
        file_content = grid_file.read()
        filename = doc.get("filename", "download.pdf")
        
        # Determine content type
        if filename.lower().endswith('.pdf'):
            content_type = "application/pdf"
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            content_type = "image/jpeg"
        elif filename.lower().endswith('.png'):
            content_type = "image/png"
        elif filename.lower().endswith('.gif'):
            content_type = "image/gif"
        elif filename.lower().endswith('.webp'):
            content_type = "image/webp"
        else:
            content_type = "application/octet-stream"
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading file: {str(e)}"
        )


@router.delete(
    "/files/{file_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete uploaded file",
    description="Delete an uploaded file from the database"
)
async def delete_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a file from documents collection and optionally from GridFS.
    """
    from chatbot.core.db import DB_DOCUMENTS_COLLECTION, FS
    from bson import ObjectId
    
    user_id = str(current_user["_id"])
    
    if DB_DOCUMENTS_COLLECTION is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    try:
        # Find the document
        doc = DB_DOCUMENTS_COLLECTION.find_one({
            "_id": ObjectId(file_id),
            "user_id": user_id
        })
        
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Delete from GridFS if exists and no other documents reference it
        gridfs_id = doc.get("file_gridfs_id")
        if gridfs_id and FS is not None:
            # Check if any other document uses this gridfs file
            other_refs = DB_DOCUMENTS_COLLECTION.count_documents({
                "file_gridfs_id": gridfs_id,
                "_id": {"$ne": ObjectId(file_id)}
            })
            if other_refs == 0:
                try:
                    FS.delete(ObjectId(gridfs_id))
                except Exception:
                    pass  # GridFS file may already be deleted
        
        # Delete the document
        DB_DOCUMENTS_COLLECTION.delete_one({"_id": ObjectId(file_id)})
        
        return {"message": "File deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}"
        )
