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
        """创建按任意字符串键分组的线程安全失败计数器。

        Args:
            max_attempts: 一个窗口内允许记录的最大失败次数；达到该数量即拒绝检查。
            window_seconds: 每次检查向前保留失败时间戳的窗口秒数。

        Notes:
            状态仅存在于当前服务进程，不会跨进程、重启或多实例共享。
        """
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """清理指定键的过期失败记录并判断是否已经限速。

        Args:
            key: 调用方构造的限速维度，例如“客户端 IP:规范化邮箱”。

        Raises:
            LoginRateLimitError: 当前窗口内失败次数大于等于配置上限。

        Notes:
            本方法只检查和清理，不会把当前登录尝试计为失败。
        """
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
        """在锁内为指定键追加当前失败时间戳。

        Args:
            key: 与 :meth:`check` 使用相同的限速维度。
        """
        with self._lock:
            self._attempts.setdefault(key, []).append(time.time())

    def reset(self, key: str) -> None:
        """移除指定键的全部失败记录；键不存在时不执行操作。

        Args:
            key: 成功登录后需要解除限速状态的维度。
        """
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
        """返回用户角色是否严格等于 ``admin``。"""
        return self.role == "admin"

    def as_dict(self) -> dict:
        """将可公开的用户字段转换为 API 响应字典。

        Returns:
            包含 ID、邮箱、显示名、角色、启用状态和创建时间的新字典；
            不包含密码哈希、邀请人或任何会话令牌。
        """
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


def _normalize_email(email: str) -> str:
    """清理并验证用于唯一存储和登录匹配的邮箱地址。

    Args:
        email: 用户输入的邮箱字符串。

    Returns:
        去除首尾空白并执行 Unicode 大小写折叠后的邮箱。

    Raises:
        AuthError: 结果为空、不含 ``@`` 或长度超过 254 个字符。

    Notes:
        这是满足本应用需求的轻量校验，不尝试完整实现 RFC 邮箱语法。
    """
    normalized = email.strip().casefold()
    if not normalized or "@" not in normalized or len(normalized) > 254:
        raise AuthError("请输入有效的邮箱地址。")
    return normalized


def _validate_password(password: str) -> None:
    """验证密码是否满足认证存储允许的长度边界。

    Args:
        password: 待注册或设置的明文密码。

    Raises:
        AuthError: 密码少于 10 个或超过 128 个字符。
    """
    if len(password) < 10:
        raise AuthError("密码至少需要 10 个字符。")
    if len(password) > 128:
        raise AuthError("密码不能超过 128 个字符。")


def _hash_password(password: str) -> str:
    """使用随机盐和固定 scrypt 参数生成自描述密码哈希。

    Args:
        password: 已由用户提供的明文密码；本方法会再次执行长度校验。

    Returns:
        ``scrypt$N$r$p$salt$digest`` 格式字符串，盐和派生值使用 URL-safe Base64。

    Raises:
        AuthError: 密码长度不符合要求。
        ValueError: 当前运行环境无法执行指定 scrypt 参数。
    """
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
    """按照哈希字符串内记录的 scrypt 参数验证候选密码。

    Args:
        password: 登录请求中的候选明文密码。
        encoded: 数据库存储的自描述 scrypt 哈希字符串。

    Returns:
        使用恒定时间摘要比较得到的验证结果。算法名不支持、字段数量错误、
        Base64/整数/参数格式无效时统一返回 ``False``，不向登录接口泄露细节。
    """
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
    """计算认证或邀请令牌的 SHA-256 存储摘要。

    Args:
        token: 仅在客户端与当前调用栈中出现的原始随机令牌。

    Returns:
        小写十六进制 SHA-256 摘要；数据库从不需要保存原令牌。
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(
        self,
        db_path: str,
        session_ttl_seconds: int,
        invitation_ttl_seconds: int,
        max_active_chat_sessions: int,
    ):
        """打开认证 SQLite 数据库并初始化线程安全的持久化结构。

        Args:
            db_path: SQLite 文件路径；不存在的父目录会被递归创建。
            session_ttl_seconds: 新登录会话从创建起的固定有效期秒数。
            invitation_ttl_seconds: 新邀请从创建起的有效期秒数。
            max_active_chat_sessions: 每位用户允许保留的聊天会话上限。

        Notes:
            连接允许跨线程使用，但所有访问都由可重入锁保护；构造启用 WAL、
            外键约束并幂等创建表和索引。

        Raises:
            sqlite3.Error: 目录可创建但数据库无法打开或初始化。
        """
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
        """幂等创建用户、邀请、认证会话、聊天会话及查询索引。

        本方法在构造器持有存储锁时调用，并在脚本成功后提交。数据库错误由 SQLite
        原样抛出，使服务启动失败而不是在不完整结构上继续运行。
        """
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
        """把包含用户列的 SQLite 行转换为不可变领域对象。

        Args:
            row: 至少包含 ``id``、``email``、``display_name``、``role``、
                ``is_active`` 和 ``created_at`` 的查询行。

        Returns:
            ``is_active`` 转为布尔值、``created_at`` 转为浮点数的 ``AuthUser``。

        Raises:
            IndexError: 查询行缺少必需列。
        """
        return AuthUser(
            id=row["id"],
            email=row["email"],
            display_name=row["display_name"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=float(row["created_at"]),
        )

    def bootstrap_admin(self, email: str, password: str) -> bool:
        """仅在用户表为空时创建初始管理员账号。

        Args:
            email: 启动配置中的管理员邮箱；为空时跳过引导。
            password: 启动配置中的管理员明文密码；为空时跳过引导。

        Returns:
            本次确实创建账号时返回 ``True``；配置不完整或已有任意用户时返回
            ``False``，且不会覆盖现有管理员。

        Raises:
            AuthError: 非空邮箱或密码不符合验证要求。
            sqlite3.Error: 查询、插入或提交失败。
        """
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
        """在线程锁内查询当前用户总数。

        Returns:
            ``users`` 表的当前行数，转换为普通整数。该计数包含启用和停用账号。

        Raises:
            sqlite3.Error: 数据库连接已关闭或计数查询失败。
        """
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def authenticate(self, email: str, password: str) -> tuple[AuthUser, str]:
        """验证启用用户的邮箱和密码，并创建新的登录会话。

        Args:
            email: 登录表单中的邮箱。
            password: 登录表单中的候选明文密码。

        Returns:
            已验证的 ``AuthUser`` 与只返回本次调用方的原始会话令牌。

        Raises:
            AuthError: 邮箱无效、用户不存在/已停用或密码不匹配。不存在与密码错误
                使用同一提示，避免账号枚举。
            sqlite3.Error: 数据库查询或会话创建失败。
        """
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
        """创建固定过期时间的登录会话，并只持久化随机令牌摘要。

        Args:
            user_id: 会话所属用户的数据库 ID。

        Returns:
            256 位 URL-safe 随机令牌原文，调用方应仅通过安全 Cookie 返回一次。

        Notes:
            调用约定要求外层已持有 ``self._lock``。写入前会清理所有已过期会话，
            然后提交删除和新增；后续访问只更新 ``last_seen_at``，不会滑动延长过期时间。

        Raises:
            sqlite3.Error: 清理、插入或提交失败。
        """
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
        """解析 Cookie 令牌，并验证会话时效及账号启用状态。

        Args:
            token: 客户端提供的原始会话令牌；空值直接视为未登录。

        Returns:
            有效会话对应的 ``AuthUser``；令牌未知、已过期或用户已停用时返回
            ``None``。有效访问会更新 ``last_seen_at`` 但不延长 ``expires_at``。

        Notes:
            发现过期或停用会话时会立即从数据库删除并提交。
        """
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
        """删除与给定原始令牌匹配的登录会话。

        Args:
            token: 当前 Cookie 中的原始会话令牌。

        Notes:
            空令牌或数据库中不存在的令牌均视为幂等成功。
        """
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
        """由管理员为尚未注册的邮箱创建一次性、限时邀请。

        Args:
            creator: 发起邀请的当前用户，必须具有管理员角色。
            email: 被邀请邮箱，会先规范化并检查是否已注册。
            role: 激活后授予的 ``user`` 或 ``admin`` 角色。

        Returns:
            可持久展示的邀请元数据与仅本次返回的原始随机令牌。数据库只保存摘要。

        Raises:
            AuthError: 创建者非管理员、角色非法、邮箱无效或邮箱已经注册。
            sqlite3.Error: 更新旧邀请、插入或提交失败。

        Notes:
            创建新邀请会把同邮箱所有未使用旧邀请标记为已使用，使旧链接立即失效。
        """
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
        """原子验证一次性邀请并创建用户，随后建立登录会话。

        Args:
            token: 邀请链接携带的原始随机令牌。
            display_name: 新用户显示名，清理后必须为 1 到 80 个字符。
            password: 新账号明文密码，必须通过统一长度校验。

        Returns:
            新建 ``AuthUser`` 和可设置到 Cookie 的原始登录会话令牌。

        Raises:
            AuthError: 令牌为空、邀请无效/过期/已使用、邮箱已注册或用户资料非法。
            sqlite3.Error: 事务、用户读取或登录会话创建失败。

        Notes:
            ``BEGIN IMMEDIATE`` 保证检查邮箱、插入用户和消费邀请在同一事务中；
            该部分任何异常都会回滚。登录会话在用户事务提交后单独创建。
        """
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
        """列出最近创建的邀请并根据当前时间计算展示状态。

        Returns:
            最多 100 条按创建时间倒序的元数据字典。``status`` 为 ``used``、
            ``expired`` 或 ``pending``；不包含令牌摘要和原始令牌。
        """
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
        """列出最近创建的用户公开资料。

        Returns:
            最多 200 个按创建时间倒序的公开用户字典，不包含密码哈希。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT 200",
            ).fetchall()
        return [self._row_to_user(row).as_dict() for row in rows]

    def set_user_active(self, user_id: str, active: bool) -> None:
        """更新账号启用状态，并在停用时立即撤销全部登录会话。

        Args:
            user_id: 目标用户数据库 ID。
            active: 需要写入的启用状态。

        Raises:
            AuthError: 目标用户不存在。
            sqlite3.Error: 更新、会话删除或提交失败。
        """
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
        """在用户配额内创建一个新的聊天会话元数据记录。

        Args:
            user: 新会话所有者。
            title: 候选标题；去除首尾空白并截断为 80 字符，空值回退“新对话”。

        Returns:
            含新 UUID、规范化标题、创建和更新时间的字典。

        Raises:
            AuthError: 当前用户的会话数量已达到配置上限。
            sqlite3.Error: 计数、插入或提交失败。
        """
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
        """按最近活动时间列出指定用户自己的聊天会话。

        Args:
            user: 当前已认证用户，查询只使用其 ID 作为所有权条件。

        Returns:
            按 ``updated_at`` 倒序排列的会话元数据新字典列表。
        """
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
        """读取一个会话，并强制要求它属于当前用户。

        Args:
            user: 当前已认证用户。
            session_id: 请求访问的聊天会话 ID。

        Returns:
            匹配会话的 ID、标题和时间戳字典。

        Raises:
            AuthError: 会话不存在或属于其他用户；两种情况使用同一提示避免泄露。
        """
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
        """在所有权约束下更新会话活动时间，并可同时修改标题。

        Args:
            user: 当前已认证用户。
            session_id: 需要更新的聊天会话 ID。
            title: 可选新标题；真值时清理并截断为 80 字符，空值只更新时间。

        Raises:
            AuthError: 会话不存在或不属于当前用户。
            sqlite3.Error: 更新或提交失败。
        """
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
        """在所有权约束下永久删除一条聊天会话元数据。

        Args:
            user: 当前已认证用户。
            session_id: 需要删除的聊天会话 ID。

        Raises:
            AuthError: 会话不存在或不属于当前用户。
            sqlite3.Error: 删除或提交失败。

        Notes:
            本方法不清理 Pipeline 内存或 Chroma 长期记忆，调用方必须先协调这些资源。
        """
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
        """在线程锁内关闭底层 SQLite 连接。

        关闭后不应再次调用其他存储方法；SQLite 对重复关闭的行为由驱动决定。
        """
        with self._lock:
            self._conn.close()
