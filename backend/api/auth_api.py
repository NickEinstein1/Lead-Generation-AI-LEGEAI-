from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import secrets

from backend.security.authentication import auth_manager, UserRole
from backend.database.connection import SessionLocal, session_dep
from backend.models.user import User as UserModel
from backend.models.session import UserSession as SessionModel

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str
    role: Optional[str] = None

class LoginRequest(BaseModel):
    username: Optional[str] = None  # username or email
    email: Optional[str] = None
    password: str

@router.post("/register")
async def register(payload: RegisterRequest, req: Request):
    try:
        username = (payload.username or (payload.email or "").split("@")[0]).strip() if (payload.username or payload.email) else None
        email = (payload.email or "").strip() or None
        if not username or not email or not payload.password:
            raise HTTPException(status_code=400, detail="username/email and password are required")

        role_str = (payload.role or UserRole.AGENT.value)
        try:
            role = UserRole(role_str.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid role")

        use_db = os.getenv("USE_DB", "false").lower() == "true"
        ip = req.client.host if req and req.client else ""
        ua = req.headers.get("user-agent", "")

        if use_db:
            async with SessionLocal() as session:
                # uniqueness check
                exists = (await session.execute(
                    select(UserModel).where((UserModel.username == username) | (UserModel.email == email))
                )).scalar_one_or_none()
                if exists:
                    raise HTTPException(status_code=400, detail="User already exists")

                pwd_hash = auth_manager.hash_password(payload.password)
                db_user = UserModel(
                    username=username,
                    email=email,
                    password_hash=pwd_hash,
                    role=role.value,
                    permissions=[p.value for p in auth_manager.role_permissions.get(role, [])],
                )
                session.add(db_user)
                await session.flush()

                session_id = secrets.token_urlsafe(32)
                expires_at = datetime.utcnow() + timedelta(hours=8)
                db_sess = SessionModel(
                    session_id=session_id,
                    user_id=db_user.id,
                    expires_at=expires_at,
                    ip_address=ip,
                    user_agent=ua,
                    is_active=True,
                )
                session.add(db_sess)
                await session.commit()

                token = auth_manager.generate_jwt_token(str(db_user.id), auth_manager.role_permissions.get(role, []))
                return {"status": "success", "user_id": str(db_user.id), "session_id": session_id, "token": token, "role": role.value}

        ok, result = auth_manager.create_user(username, email, payload.password, role)
        if not ok:
            raise HTTPException(status_code=400, detail=result)
        success, session_id, _ = auth_manager.authenticate_user(username, payload.password, ip, ua)
        token = auth_manager.generate_jwt_token(result, auth_manager.role_permissions.get(role, [])) if success else None
        return {"status": "success", "user_id": result, "session_id": session_id, "token": token, "role": role.value}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

@router.post("/login")
async def login(payload: LoginRequest, req: Request, session: AsyncSession = Depends(session_dep)):
    ip = req.client.host if req and req.client else ""
    ua = req.headers.get("user-agent", "")
    identifier = (payload.username or payload.email or "").strip()
    if not identifier or not payload.password:
        raise HTTPException(status_code=400, detail="username/email and password are required")

    use_db = os.getenv("USE_DB", "false").lower() == "true"
    if use_db:
        row = (await session.execute(
            select(UserModel).where((UserModel.username == identifier) | (UserModel.email == identifier))
        )).scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not auth_manager.verify_password(payload.password, row.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        row.last_login = datetime.utcnow()
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=8)
        db_sess = SessionModel(
            session_id=session_id,
            user_id=row.id,
            expires_at=expires_at,
            ip_address=ip,
            user_agent=ua,
            is_active=True,
        )
        session.add(db_sess)
        await session.commit()
        role_enum = UserRole(row.role)
        token = auth_manager.generate_jwt_token(str(row.id), auth_manager.role_permissions.get(role_enum, []))
        return {"status": "success", "session_id": session_id, "token": token, "user_id": str(row.id), "role": row.role}

    success, session_id, message = auth_manager.authenticate_user(identifier, payload.password, ip, ua)
    if not success:
        raise HTTPException(status_code=401, detail=message)
    user_id = None
    role = None
    u = None
    for u in auth_manager.users.values():
        if u.username == identifier or u.email == identifier:
            user_id = u.user_id
            role = u.role.value
            break
    token = auth_manager.generate_jwt_token(user_id, auth_manager.role_permissions.get(u.role, [])) if user_id else None
    return {"status": "success", "session_id": session_id, "token": token, "user_id": user_id, "role": role}

@router.post("/refresh")
async def refresh(req: Request, session: AsyncSession = Depends(session_dep)):
    session_id = req.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")

    use_db = os.getenv("USE_DB", "false").lower() == "true"
    if use_db:
        db_sess = (await session.execute(
            select(SessionModel).where(SessionModel.session_id == session_id)
        )).scalar_one_or_none()
        if not db_sess or not db_sess.is_active or db_sess.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Invalid session")
        user = (await session.execute(select(UserModel).where(UserModel.id == db_sess.user_id))).scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid session")
        role_enum = UserRole(user.role)
        token = auth_manager.generate_jwt_token(str(user.id), auth_manager.role_permissions.get(role_enum, []))
        return {"status": "success", "token": token}

    # fallback in-memory
    session_obj = auth_manager.sessions.get(session_id)
    if not session_obj or not session_obj.is_active:
        raise HTTPException(status_code=401, detail="Invalid session")
    user = auth_manager.users.get(session_obj.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    token = auth_manager.generate_jwt_token(user.user_id, user.permissions)
    return {"status": "success", "token": token}

@router.post("/logout")
async def logout(req: Request, session: AsyncSession = Depends(session_dep)):
    session_id = req.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")

    use_db = os.getenv("USE_DB", "false").lower() == "true"
    if use_db:
        db_sess = (await session.execute(select(SessionModel).where(SessionModel.session_id == session_id))).scalar_one_or_none()
        if db_sess:
            db_sess.is_active = False
            await session.commit()
        return {"status": "success"}

    ok = auth_manager.logout_user(session_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Logout failed")
    return {"status": "success"}

@router.get("/me")
async def me(req: Request, session: AsyncSession = Depends(session_dep)):
    session_id = req.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")

    use_db = os.getenv("USE_DB", "false").lower() == "true"
    if use_db:
        db_sess = (await session.execute(select(SessionModel).where(SessionModel.session_id == session_id))).scalar_one_or_none()
        if not db_sess or not db_sess.is_active or db_sess.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Invalid session")
        user = (await session.execute(select(UserModel).where(UserModel.id == db_sess.user_id))).scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid session")
        return {
            "user_id": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "permissions": user.permissions or [],
        }

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

