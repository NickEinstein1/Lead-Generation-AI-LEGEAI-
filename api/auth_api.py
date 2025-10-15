from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from security.authentication import auth_manager, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = UserRole.AGENT.value

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register(payload: RegisterRequest):
    try:
        try:
            role = UserRole(payload.role)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid role")

        ok, result = auth_manager.create_user(payload.username, payload.email, payload.password, role)
        if not ok:
            raise HTTPException(status_code=400, detail=result)
        # auto-login after registration for convenience
        success, session_id, _ = auth_manager.authenticate_user(payload.username, payload.password, "127.0.0.1", "web")
        token = auth_manager.generate_jwt_token(result, auth_manager.role_permissions.get(role, [])) if success else None
        return {"status": "success", "user_id": result, "session_id": session_id, "token": token}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

@router.post("/login")
async def login(payload: LoginRequest, req: Request):
    ip = req.client.host if req and req.client else ""
    ua = req.headers.get("user-agent", "")
    success, session_id, message = auth_manager.authenticate_user(payload.username, payload.password, ip, ua)
    if not success:
        raise HTTPException(status_code=401, detail=message)
    # find user id
    user_id = None
    for u in auth_manager.users.values():
        if u.username == payload.username or u.email == payload.username:
            user_id = u.user_id
            role = u.role.value
            break
    token = auth_manager.generate_jwt_token(user_id, auth_manager.role_permissions.get(u.role, [])) if user_id else None
    return {"status": "success", "session_id": session_id, "token": token, "user_id": user_id, "role": role}

@router.post("/refresh")
async def refresh(req: Request):
    session_id = req.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")
    # resolve session to user
    session = auth_manager.sessions.get(session_id)
    if not session or not session.is_active:
        raise HTTPException(status_code=401, detail="Invalid session")
    user = auth_manager.users.get(session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    token = auth_manager.generate_jwt_token(user.user_id, user.permissions)
    return {"status": "success", "token": token}

@router.post("/logout")
async def logout(req: Request):
    session_id = req.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")
    ok = auth_manager.logout_user(session_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Logout failed")
    return {"status": "success"}

@router.get("/me")
async def me(req: Request):
    session_id = req.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")
    valid, user = auth_manager.validate_session(session_id)
    if not valid or not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "permissions": [p.value for p in user.permissions],
    }

