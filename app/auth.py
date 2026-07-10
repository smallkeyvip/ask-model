"""邀请制本地认证、授权与聊天会话所有权存储。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64


class AuthError(ValueError):
    """可安全返回给调用方的认证业务错误。"""


class LoginRateLimitError(AuthError):
    """登录失败次数超过当前限速窗口。"""


class LoginAttemptLimiter:
    """单进程登录失败窗口限速器。"""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        with self._lock:
            recent = [
                timestamp
                for timestamp in self._attempts.get(key, [])
                if now - timestamp < self.window_seconds
            ]
            self._attempts[key] = recent
            if len(recent) >= self.max_attempts:
                raise LoginRateLimitError("登录尝试过于频繁，请稍后再试。")

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._attempts.setdefault(key, []).append(time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    created_at: float

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


def _normalize_email(email: str) -> str:
    normalized = email.strip().casefold()
    if not normalized or "@" not in normalized or len(normalized) > 254:
        raise AuthError("请输入有效的邮箱地址。")
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < 10:
        raise AuthError("密码至少需要 10 个字符。")
    if len(password) > 128:
        raise AuthError("密码不能超过 128 个字符。")


def _hash_password(password: str) -> str:
    _validate_password(password)
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N,
        _SCRYPT_R,
        _SCRYPT_P,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_b64, digest_b64 = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(
        self,
        db_path: str,
        session_ttl_seconds: int,
        invitation_ttl_seconds: int,
        max_active_chat_sessions: int,
    ):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.session_ttl_seconds = session_ttl_seconds
        self.invitation_ttl_seconds = invitation_ttl_seconds
        self.max_active_chat_sessions = max_active_chat_sessions
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                invited_by TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS invitations (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
                token_hash TEXT NOT NULL UNIQUE,
                expires_at REAL NOT NULL,
                used_at REAL,
                created_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(created_by) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_auth_sessions_token
                ON auth_sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_invitations_email
                ON invitations(email, used_at, expires_at);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner
                ON chat_sessions(user_id, updated_at DESC);
            """
        )
        self._conn.commit()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> AuthUser:
        return AuthUser(
            id=row["id"],
            email=row["email"],
            display_name=row["display_name"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=float(row["created_at"]),
        )

    def bootstrap_admin(self, email: str, password: str) -> bool:
        if not email or not password:
            return False
        normalized_email = _normalize_email(email)
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count:
                return False
            now = time.time()
            self._conn.execute(
                """INSERT INTO users
                   (id, email, display_name, password_hash, role,
                    is_active, invited_by, created_at)
                   VALUES (?, ?, ?, ?, 'admin', 1, NULL, ?)""",
                (
                    str(uuid.uuid4()),
                    normalized_email,
                    normalized_email.split("@", 1)[0],
                    _hash_password(password),
                    now,
                ),
            )
            self._conn.commit()
            return True

    def user_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def authenticate(self, email: str, password: str) -> tuple[AuthUser, str]:
        normalized_email = _normalize_email(email)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()
            if row is None or not row["is_active"]:
                raise AuthError("邮箱或密码不正确。")
            if not _verify_password(password, row["password_hash"]):
                raise AuthError("邮箱或密码不正确。")
            user = self._row_to_user(row)
            token = self._create_auth_session_locked(user.id)
            return user, token

    def _create_auth_session_locked(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._conn.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?",
            (now,),
        )
        self._conn.execute(
            """INSERT INTO auth_sessions
               (id, user_id, token_hash, expires_at, created_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                user_id,
                _token_digest(token),
                now + self.session_ttl_seconds,
                now,
                now,
            ),
        )
        self._conn.commit()
        return token

    def resolve_session(self, token: str) -> Optional[AuthUser]:
        if not token:
            return None
        now = time.time()
        digest = _token_digest(token)
        with self._lock:
            row = self._conn.execute(
                """SELECT users.*, auth_sessions.id AS auth_session_id,
                          auth_sessions.expires_at AS session_expires_at
                   FROM auth_sessions
                   JOIN users ON users.id = auth_sessions.user_id
                   WHERE auth_sessions.token_hash = ?""",
                (digest,),
            ).fetchone()
            if row is None:
                return None
            if row["session_expires_at"] <= now or not row["is_active"]:
                self._conn.execute(
                    "DELETE FROM auth_sessions WHERE id = ?",
                    (row["auth_session_id"],),
                )
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?",
                (now, row["auth_session_id"]),
            )
            self._conn.commit()
            return self._row_to_user(row)

    def logout(self, token: str) -> None:
        if not token:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (_token_digest(token),),
            )
            self._conn.commit()

    def create_invitation(
        self,
        creator: AuthUser,
        email: str,
        role: str = "user",
    ) -> tuple[dict, str]:
        if not creator.is_admin:
            raise AuthError("需要管理员权限。")
        if role not in {"user", "admin"}:
            raise AuthError("无效的用户角色。")
        normalized_email = _normalize_email(email)
        token = secrets.token_urlsafe(32)
        now = time.time()
        invitation_id = str(uuid.uuid4())
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()
            if exists:
                raise AuthError("该邮箱已经注册。")
            self._conn.execute(
                """UPDATE invitations SET used_at = ?
                   WHERE email = ? AND used_at IS NULL""",
                (now, normalized_email),
            )
            self._conn.execute(
                """INSERT INTO invitations
                   (id, email, role, token_hash, expires_at,
                    used_at, created_by, created_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    invitation_id,
                    normalized_email,
                    role,
                    _token_digest(token),
                    now + self.invitation_ttl_seconds,
                    creator.id,
                    now,
                ),
            )
            self._conn.commit()
        return {
            "id": invitation_id,
            "email": normalized_email,
            "role": role,
            "expires_at": now + self.invitation_ttl_seconds,
        }, token

    def accept_invitation(
        self,
        token: str,
        display_name: str,
        password: str,
    ) -> tuple[AuthUser, str]:
        if not token:
            raise AuthError("邀请链接无效。")
        name = display_name.strip()
        if not name or len(name) > 80:
            raise AuthError("请输入 1 到 80 个字符的姓名。")
        _validate_password(password)
        now = time.time()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                invitation = self._conn.execute(
                    """SELECT * FROM invitations
                       WHERE token_hash = ? AND used_at IS NULL""",
                    (_token_digest(token),),
                ).fetchone()
                if invitation is None or invitation["expires_at"] <= now:
                    raise AuthError("邀请链接无效或已经过期。")
                if self._conn.execute(
                    "SELECT 1 FROM users WHERE email = ?",
                    (invitation["email"],),
                ).fetchone():
                    raise AuthError("该邮箱已经注册。")
                password_hash = _hash_password(password)
                user_id = str(uuid.uuid4())
                self._conn.execute(
                    """INSERT INTO users
                       (id, email, display_name, password_hash, role,
                        is_active, invited_by, created_at)
                       VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                    (
                        user_id,
                        invitation["email"],
                        name,
                        password_hash,
                        invitation["role"],
                        invitation["created_by"],
                        now,
                    ),
                )
                self._conn.execute(
                    "UPDATE invitations SET used_at = ? WHERE id = ?",
                    (now, invitation["id"]),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            row = self._conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            user = self._row_to_user(row)
            session_token = self._create_auth_session_locked(user.id)
            return user, session_token

    def list_invitations(self) -> list[dict]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, email, role, expires_at, used_at, created_at
                   FROM invitations
                   ORDER BY created_at DESC LIMIT 100""",
            ).fetchall()
        return [
            {
                **dict(row),
                "status": (
                    "used" if row["used_at"] else
                    "expired" if row["expires_at"] <= now else "pending"
                ),
            }
            for row in rows
        ]

    def list_users(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT 200",
            ).fetchall()
        return [self._row_to_user(row).as_dict() for row in rows]

    def set_user_active(self, user_id: str, active: bool) -> None:
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE users SET is_active = ?
                   WHERE id = ?""",
                (1 if active else 0, user_id),
            )
            if cursor.rowcount != 1:
                raise AuthError("用户不存在。")
            if not active:
                self._conn.execute(
                    "DELETE FROM auth_sessions WHERE user_id = ?",
                    (user_id,),
                )
            self._conn.commit()

    def create_chat_session(self, user: AuthUser, title: str = "新对话") -> dict:
        now = time.time()
        safe_title = title.strip()[:80] or "新对话"
        with self._lock:
            count = self._conn.execute(
                """SELECT COUNT(*) FROM chat_sessions
                   WHERE user_id = ?""",
                (user.id,),
            ).fetchone()[0]
            if count >= self.max_active_chat_sessions:
                raise AuthError("活跃对话数量已达上限，请先删除旧对话。")
            session_id = str(uuid.uuid4())
            self._conn.execute(
                """INSERT INTO chat_sessions
                   (id, user_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, user.id, safe_title, now, now),
            )
            self._conn.commit()
        return {
            "id": session_id,
            "title": safe_title,
            "created_at": now,
            "updated_at": now,
        }

    def list_chat_sessions(self, user: AuthUser) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, title, created_at, updated_at
                   FROM chat_sessions
                   WHERE user_id = ?
                   ORDER BY updated_at DESC""",
                (user.id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def require_chat_session(self, user: AuthUser, session_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                """SELECT id, title, created_at, updated_at
                   FROM chat_sessions
                   WHERE id = ? AND user_id = ?""",
                (session_id, user.id),
            ).fetchone()
        if row is None:
            raise AuthError("对话不存在或无权访问。")
        return dict(row)

    def touch_chat_session(
        self,
        user: AuthUser,
        session_id: str,
        title: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            if title:
                cursor = self._conn.execute(
                    """UPDATE chat_sessions SET title = ?, updated_at = ?
                       WHERE id = ? AND user_id = ?""",
                    (title.strip()[:80], now, session_id, user.id),
                )
            else:
                cursor = self._conn.execute(
                    """UPDATE chat_sessions SET updated_at = ?
                       WHERE id = ? AND user_id = ?""",
                    (now, session_id, user.id),
                )
            if cursor.rowcount != 1:
                raise AuthError("对话不存在或无权访问。")
            self._conn.commit()

    def delete_chat_session(self, user: AuthUser, session_id: str) -> None:
        with self._lock:
            cursor = self._conn.execute(
                """DELETE FROM chat_sessions
                   WHERE id = ? AND user_id = ?""",
                (session_id, user.id),
            )
            if cursor.rowcount != 1:
                raise AuthError("对话不存在或无权访问。")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
