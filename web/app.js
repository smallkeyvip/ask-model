"use strict";

/**
 * 返回文档中第一个匹配选择器的元素。
 *
 * @param {string} selector - 传给 `document.querySelector` 的 CSS 选择器。
 * @returns {Element|null} 匹配到的首个元素；不存在时返回 `null`。
 */
const $ = (selector) => document.querySelector(selector);

/**
 * 返回文档中匹配选择器的全部元素快照。
 *
 * @param {string} selector - 传给 `document.querySelectorAll` 的 CSS 选择器。
 * @returns {Element[]} 从静态 `NodeList` 转换得到的普通数组。
 */
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  user: null,
  sessions: [],
  currentSessionId: null,
  busy: false,
  currentView: "chat",
};

let toastTimer = null;

/**
 * 向同源后端发送 JSON 请求并统一解析成功或错误响应。
 *
 * @param {string} path - 相对当前站点的 API 路径。
 * @param {RequestInit} [options={}] - `fetch` 配置；该对象最后展开，可覆盖默认凭据或请求头。
 * @returns {Promise<any>} 后端 JSON 响应；空响应或非法 JSON 被解析为空对象。
 * @throws {Error} HTTP 状态非成功时抛出带用户消息和数字 `status` 属性的错误；
 *   已登录状态收到 401 时会先清空本地认证状态。
 */
async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) {
    if (response.status === 401 && state.user) resetAuthentication();
    const error = new Error(payload.detail || "请求未完成，请稍后重试。");
    error.status = response.status;
    throw error;
  }
  return payload;
}

/**
 * 显示一条会自动消失的全局提示。
 *
 * @param {string} message - 使用 `textContent` 展示的提示文本。
 * @returns {void}
 * 副作用：重置上一次提示计时器，并在 3.2 秒后隐藏提示元素。
 */
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3200);
}

/**
 * 批量切换表单内所有可交互控件的禁用状态。
 *
 * @param {HTMLFormElement} form - 需要锁定或解锁的表单。
 * @param {boolean} busy - `true` 表示禁止输入、选择和提交。
 * @returns {void}
 */
function setFormBusy(form, busy) {
  form.querySelectorAll("input, textarea, select, button").forEach((control) => {
    control.disabled = busy;
  });
}

/**
 * 显示认证视图，并在登录与邀请激活表单之间切换。
 *
 * @param {"login"|"invite"} [mode="login"] - 需要显示的认证流程。
 * @returns {void}
 */
function showAuth(mode = "login") {
  $("#auth-view").classList.remove("hidden");
  $("#app-view").classList.add("hidden");
  $("#login-form").classList.toggle("hidden", mode !== "login");
  $("#invite-form").classList.toggle("hidden", mode !== "invite");
}

/**
 * 建立前端登录状态、更新用户资料区并进入成员工作区。
 *
 * @param {{id:string, email:string, display_name:string, role:string}} user - 后端返回的当前用户。
 * @returns {Promise<void>} 会话列表加载并切换到问答视图后完成。
 * @throws {Error} 初始会话列表请求失败时向调用方传播。
 */
async function enterApp(user) {
  state.user = user;
  $("#auth-view").classList.add("hidden");
  $("#app-view").classList.remove("hidden");
  $("#profile-name").textContent = user.display_name;
  $("#profile-role").textContent = user.role === "admin" ? "管理员" : "企业成员";
  $("#profile-avatar").textContent = (user.display_name || user.email).slice(0, 1).toUpperCase();
  $("#nav-admin").classList.toggle("hidden", user.role !== "admin");
  await loadSessions();
  navigate("chat");
}

/**
 * 绑定页面事件并根据邀请 URL 或现有 Cookie 选择首屏。
 *
 * @returns {Promise<void>} 初始化流程完成后兑现；恢复登录失败会显示登录表单而不抛错。
 * 副作用：只应在页面启动时调用一次，重复调用会重复注册事件监听器。
 */
async function initialize() {
  bindEvents();
  const params = new URLSearchParams(window.location.search);
  if (window.location.pathname === "/invite" && params.get("token")) {
    showAuth("invite");
    return;
  }
  try {
    const data = await api("/api/auth/me");
    await enterApp(data.user);
  } catch (_) {
    showAuth("login");
  }
}

/**
 * 为认证、导航、会话、问答、管理和上传控件注册事件处理器。
 *
 * Enter（未按 Shift）会提交问答；示例问题会写入输入框；管理标签会触发对应
 * 数据加载。内联回调仅捕获当前按钮或事件，实际业务由具名处理函数完成。
 *
 * @returns {void}
 */
function bindEvents() {
  $("#login-form").addEventListener("submit", handleLogin);
  $("#invite-form").addEventListener("submit", handleInviteAcceptance);
  $("#logout-button").addEventListener("click", handleLogout);
  $("#new-session").addEventListener("click", createSession);
  $("#chat-form").addEventListener("submit", handleChatSubmit);
  $("#chat-input").addEventListener("input", resizeComposer);
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#nav-chat").addEventListener("click", () => navigate("chat"));
  $("#nav-admin").addEventListener("click", () => navigate("admin"));
  $("#sidebar-open").addEventListener("click", openSidebar);
  $("#sidebar-close").addEventListener("click", closeSidebar);
  $("#sidebar-scrim").addEventListener("click", closeSidebar);
  $$("[data-question]").forEach((button) => button.addEventListener("click", () => {
    $("#chat-input").value = button.dataset.question;
    resizeComposer();
    $("#chat-input").focus();
  }));
  $$(".admin-tab").forEach((button) => button.addEventListener("click", () => selectAdminTab(button.dataset.adminTab)));
  $("#create-invite-form").addEventListener("submit", handleCreateInvitation);
  $("#copy-invite").addEventListener("click", copyInvitation);
  $("#upload-files").addEventListener("change", updateFileSummary);
  $("#upload-form").addEventListener("submit", handleUpload);
}

/**
 * 处理登录表单提交，并把认证错误显示在表单内。
 *
 * @param {SubmitEvent} event - 登录表单提交事件。
 * @returns {Promise<void>} 请求、视图切换及表单解锁完成后兑现。
 * 副作用：请求期间禁用表单；成功时设置全局用户状态，失败时更新错误文本。
 */
async function handleLogin(event) {
  event.preventDefault();
  const form = event.currentTarget;
  $("#login-error").textContent = "";
  setFormBusy(form, true);
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({
        email: $("#login-email").value,
        password: $("#login-password").value,
      }),
    });
    await enterApp(data.user);
  } catch (error) {
    $("#login-error").textContent = error.message;
  } finally {
    setFormBusy(form, false);
  }
}

/**
 * 校验两次密码并使用 URL 中的一次性令牌激活邀请账号。
 *
 * @param {SubmitEvent} event - 邀请激活表单提交事件。
 * @returns {Promise<void>} 激活请求和后续工作区进入流程完成后兑现。
 * 副作用：成功后从地址栏移除邀请令牌；密码不一致时不发送请求。
 */
async function handleInviteAcceptance(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const password = $("#invite-password").value;
  $("#invite-error").textContent = "";
  if (password !== $("#invite-password-confirm").value) {
    $("#invite-error").textContent = "两次输入的密码不一致。";
    return;
  }
  setFormBusy(form, true);
  try {
    const data = await api("/api/auth/invitations/accept", {
      method: "POST",
      body: JSON.stringify({
        token: new URLSearchParams(window.location.search).get("token"),
        display_name: $("#invite-name").value,
        password,
      }),
    });
    window.history.replaceState({}, "", "/");
    await enterApp(data.user);
  } catch (error) {
    $("#invite-error").textContent = error.message;
  } finally {
    setFormBusy(form, false);
  }
}

/**
 * 尝试撤销服务端会话，并无条件完成本地退出。
 *
 * @returns {Promise<void>} 服务端请求成功或失败、且本地状态已清理后兑现。
 * 副作用：即使网络请求失败，也会清空用户、会话、消息和密码输入框。
 */
async function handleLogout() {
  try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } catch (_) { /* local logout still proceeds */ }
  resetAuthentication();
}

/**
 * 清空所有本地认证与对话状态并返回登录视图。
 *
 * @returns {void}
 * 副作用：不发送服务端请求；同时清除消息 DOM、关闭侧边栏和清空密码字段。
 */
function resetAuthentication() {
  state.user = null;
  state.sessions = [];
  state.currentSessionId = null;
  clearMessages();
  $("#login-password").value = "";
  closeSidebar();
  showAuth("login");
}

/**
 * 在问答与管理视图之间切换，并同步导航选中状态。
 *
 * @param {"chat"|"admin"} view - 目标工作区。
 * @returns {void}
 * 副作用：非管理员请求管理视图时直接忽略；进入管理视图会默认选择邀请标签并加载数据。
 */
function navigate(view) {
  if (view === "admin" && state.user?.role !== "admin") return;
  state.currentView = view;
  $("#chat-view").classList.toggle("hidden", view !== "chat");
  $("#admin-view").classList.toggle("hidden", view !== "admin");
  $("#nav-chat").classList.toggle("active", view === "chat");
  $("#nav-admin").classList.toggle("active", view === "admin");
  $("#view-eyebrow").textContent = view === "chat" ? "KNOWLEDGE DESK" : "ADMIN CONSOLE";
  $("#view-title").textContent = view === "chat" ? "企业问答" : "系统管理";
  if (view === "admin") selectAdminTab("invite");
  closeSidebar();
}

/**
 * 打开移动端侧边栏及其遮罩层。
 *
 * @returns {void}
 */
function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#sidebar-scrim").classList.remove("hidden");
}

/**
 * 关闭侧边栏并隐藏遮罩层；重复调用保持幂等。
 *
 * @returns {void}
 */
function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#sidebar-scrim").classList.add("hidden");
}

/**
 * 从后端刷新当前用户会话列表并重新渲染侧边栏。
 *
 * @returns {Promise<void>} 全局 `state.sessions` 更新及 DOM 渲染完成后兑现。
 * @throws {Error} API 请求失败时向调用方传播。
 */
async function loadSessions() {
  const data = await api("/api/sessions");
  state.sessions = data.sessions;
  renderSessions();
}

/**
 * 根据全局会话状态重建可选择、可删除的会话列表。
 *
 * @returns {void}
 * 副作用：清空原列表 DOM；无会话时渲染占位文本，当前会话获得 `active` 类。
 */
function renderSessions() {
  const list = $("#session-list");
  list.replaceChildren();
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = "暂无对话";
    list.append(empty);
    return;
  }
  state.sessions.forEach((session) => {
    const row = document.createElement("div");
    row.className = `session-item${session.id === state.currentSessionId ? " active" : ""}`;
    const select = document.createElement("button");
    select.type = "button";
    select.className = "session-select";
    select.textContent = session.title;
    select.title = session.title;
    select.addEventListener("click", () => selectSession(session.id));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "session-delete";
    remove.textContent = "×";
    remove.title = "删除对话";
    remove.setAttribute("aria-label", `删除 ${session.title}`);
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(session.id);
    });
    row.append(select, remove);
    list.append(row);
  });
}

/**
 * 创建一个“新对话”，设为当前会话并重置问答视图。
 *
 * @returns {Promise<void>} 创建成功或错误提示已经显示后兑现。
 * 副作用：全局问答忙碌时直接返回；成功时把新会话插入列表首位并聚焦输入框。
 */
async function createSession() {
  if (state.busy) return;
  try {
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title: "新对话" }),
    });
    state.sessions.unshift(session);
    state.currentSessionId = session.id;
    clearMessages();
    renderSessions();
    navigate("chat");
    $("#chat-input").focus();
  } catch (error) {
    showToast(error.message);
  }
}

/**
 * 切换当前会话并加载其仍可用的短期消息。
 *
 * @param {string} sessionId - 目标聊天会话 ID。
 * @returns {Promise<void>} 消息加载或错误提示完成后兑现。
 * 副作用：忙碌中或已选中目标时不执行；请求期间先清空当前消息，最后关闭侧边栏。
 */
async function selectSession(sessionId) {
  if (state.busy || state.currentSessionId === sessionId) return;
  state.currentSessionId = sessionId;
  renderSessions();
  clearMessages();
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);
    data.messages.forEach((message) => appendMessage(message.role, message.content));
  } catch (error) {
    showToast(error.message);
  }
  closeSidebar();
}

/**
 * 经用户确认后删除会话及服务端个人记忆，并同步本地状态。
 *
 * @param {string} sessionId - 需要删除的聊天会话 ID。
 * @returns {Promise<void>} 删除、取消或错误提示处理完成后兑现。
 * 副作用：忙碌状态或用户取消确认时不请求；删除当前会话会同时清空消息区域。
 */
async function deleteSession(sessionId) {
  if (state.busy || !window.confirm("删除这段对话及其个人记忆？")) return;
  try {
    await api(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    state.sessions = state.sessions.filter((session) => session.id !== sessionId);
    if (state.currentSessionId === sessionId) {
      state.currentSessionId = null;
      clearMessages();
    }
    renderSessions();
    showToast("对话已删除");
  } catch (error) {
    showToast(error.message);
  }
}

/**
 * 移除所有已渲染消息并恢复空会话占位状态。
 *
 * @returns {void}
 * 副作用：同时清空问答阶段状态文本，但不修改服务端或会话列表。
 */
function clearMessages() {
  const list = $("#message-list");
  list.querySelectorAll(".message").forEach((message) => message.remove());
  $("#empty-chat").classList.remove("hidden");
  $("#chat-status").textContent = "";
}

/**
 * 安全创建并追加一条用户或助手消息。
 *
 * @param {"user"|"assistant"} role - 决定样式与角色徽标的消息角色。
 * @param {string} [content=""] - 通过 `textContent` 写入的初始正文。
 * @returns {{message: HTMLElement, body: HTMLElement}} 外层消息和可继续追加正文的内容元素。
 * 副作用：隐藏空状态、追加到消息列表并平滑滚动到新消息。
 */
function appendMessage(role, content = "") {
  $("#empty-chat").classList.add("hidden");
  const message = document.createElement("article");
  message.className = `message message--${role}`;
  const badge = document.createElement("div");
  badge.className = "message-role";
  badge.textContent = role === "assistant" ? "A" : "我";
  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  message.append(badge, body);
  $("#message-list").append(message);
  message.scrollIntoView({ behavior: "smooth", block: "end" });
  return { message, body };
}

/**
 * 在指定消息正文末尾渲染后端返回的引用来源链接或标签。
 *
 * @param {HTMLElement} message - `appendMessage` 创建的消息容器。
 * @param {Array<{index:number, source:string, url?:string}>} [sources=[]] - 后端已校验的来源元数据。
 * @returns {void}
 * 副作用：先删除该消息已有来源列表；带 URL 的项在新标签页打开并设置安全 `rel`。
 */
function renderSources(message, sources = []) {
  message.querySelector(".source-list")?.remove();
  if (!sources.length) return;
  const list = document.createElement("div");
  list.className = "source-list";
  sources.forEach((source) => {
    const item = document.createElement(source.url ? "a" : "span");
    item.className = "source-chip";
    item.textContent = `[${source.index}] ${source.source || "知识库"}`;
    if (source.url) {
      item.href = source.url;
      item.target = "_blank";
      item.rel = "noopener noreferrer";
    }
    list.append(item);
  });
  message.querySelector(".message-content").append(list);
}

/**
 * 提交问题、消费 SSE 回答并维护整个问答交互的忙碌状态。
 *
 * @param {SubmitEvent} event - 问答表单提交事件。
 * @returns {Promise<void>} 会话创建、流消费、列表刷新或错误展示完成后兑现。
 * 副作用：必要时先创建会话；立即追加用户和空助手消息。完成事件会用服务端最终
 *   `answer` 覆盖增量文本并渲染来源，所有退出路径都会恢复发送按钮和状态栏。
 */
async function handleChatSubmit(event) {
  event.preventDefault();
  const query = $("#chat-input").value.trim();
  if (!query || state.busy) return;
  if (!state.currentSessionId) await createSession();
  if (!state.currentSessionId) return;

  state.busy = true;
  $("#send-button").disabled = true;
  $("#chat-input").value = "";
  resizeComposer();
  appendMessage("user", query);
  const assistant = appendMessage("assistant", "");
  $("#chat-status").textContent = "正在连接知识库...";

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: state.currentSessionId }),
    });
    if (response.status === 401) resetAuthentication();
    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "问答服务暂时不可用。");
    }
    await consumeEventStream(response.body, (event, data) => {
      if (event === "status") $("#chat-status").textContent = data;
      if (event === "chunk") assistant.body.textContent += data;
      if (event === "blocked") assistant.body.textContent = data;
      if (event === "error") throw new Error(data);
      if (event === "done") {
        if (data.answer !== undefined) assistant.body.textContent = data.answer;
        renderSources(assistant.message, data.sources || []);
      }
    });
    await loadSessions();
  } catch (error) {
    assistant.body.textContent = error.message || "问答服务暂时不可用。";
  } finally {
    state.busy = false;
    $("#send-button").disabled = false;
    $("#chat-status").textContent = "";
    assistant.message.scrollIntoView({ behavior: "smooth", block: "end" });
  }
}

/**
 * 增量解析一个 UTF-8 Server-Sent Events 字节流。
 *
 * @param {ReadableStream<Uint8Array>} stream - `fetch` 响应正文流。
 * @param {(event:string, data:any) => void} onEvent - 每个完整事件块的同步回调。
 * @returns {Promise<void>} 流读取结束且所有完整双换行分隔事件已分发后兑现。
 * @throws {Error} 底层读取失败或 `onEvent` 主动抛出的错误。
 * @description 支持 CRLF，合并多行 `data:`；JSON 解析失败时把原始文本交给回调。
 *   流结束时未形成完整事件块的尾部缓冲不会分发。
 */
async function consumeEventStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const dataLines = [];
      block.split("\n").forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      });
      if (!dataLines.length) continue;
      const raw = dataLines.join("\n");
      let data;
      try { data = JSON.parse(raw); } catch (_) { data = raw; }
      onEvent(event, data);
    }
    if (done) break;
  }
}

/**
 * 根据内容高度自动调整问答输入框，最大高度限制为 150 像素。
 *
 * @returns {void}
 * 副作用：先重置为自动高度以允许收缩，再写入计算后的像素高度。
 */
function resizeComposer() {
  const input = $("#chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

/**
 * 切换管理子页面，并按需刷新邀请或用户数据。
 *
 * @param {"invite"|"documents"|"users"} tab - 目标管理标签。
 * @returns {void}
 * 副作用：邀请和用户标签会异步发起数据加载，但本函数不等待其完成。
 */
function selectAdminTab(tab) {
  $$(".admin-tab").forEach((button) => button.classList.toggle("active", button.dataset.adminTab === tab));
  $("#admin-invite").classList.toggle("hidden", tab !== "invite");
  $("#admin-documents").classList.toggle("hidden", tab !== "documents");
  $("#admin-users").classList.toggle("hidden", tab !== "users");
  if (tab === "invite") loadInvitations();
  if (tab === "users") loadUsers();
}

/**
 * 提交管理员邀请表单并显示本次唯一可见的完整邀请 URL。
 *
 * @param {SubmitEvent} event - 创建邀请表单提交事件。
 * @returns {Promise<void>} 请求、邀请列表刷新及表单解锁完成后兑现。
 * 副作用：请求期间禁用表单；成功后清空邮箱并显示可复制链接。
 */
async function handleCreateInvitation(event) {
  event.preventDefault();
  const form = event.currentTarget;
  setFormBusy(form, true);
  try {
    const invitation = await api("/api/admin/invitations", {
      method: "POST",
      body: JSON.stringify({
        email: $("#admin-invite-email").value,
        role: $("#admin-invite-role").value,
      }),
    });
    $("#invite-url").value = new URL(invitation.invite_url, window.location.origin).href;
    $("#invite-result").classList.remove("hidden");
    $("#admin-invite-email").value = "";
    await loadInvitations();
  } catch (error) {
    showToast(error.message);
  } finally {
    setFormBusy(form, false);
  }
}

/**
 * 将当前邀请 URL 写入剪贴板，并为旧浏览器提供 `execCommand` 降级。
 *
 * @returns {Promise<void>} 复制成功或失败提示显示后兑现；输入为空时立即完成。
 * 副作用：降级或失败时选中邀请 URL 输入框，便于用户手动复制。
 */
async function copyInvitation() {
  const value = $("#invite-url").value;
  if (!value) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      $("#invite-url").select();
      if (!document.execCommand("copy")) throw new Error("copy failed");
    }
    showToast("邀请链接已复制");
  } catch (_) {
    $("#invite-url").select();
    showToast("复制失败，请手动复制邀请链接");
  }
}

/**
 * 获取管理员可见的邀请列表并渲染，失败时显示全局提示。
 *
 * @returns {Promise<void>} 请求与渲染或错误提示完成后兑现，不向事件调用方抛错。
 */
async function loadInvitations() {
  try {
    const data = await api("/api/admin/invitations");
    renderInvitations(data.invitations);
  } catch (error) { showToast(error.message); }
}

/**
 * 重建邀请数据表并把后端状态映射为中文标签和样式。
 *
 * @param {Array<{email:string, role:string, status:string, expires_at:number}>} invitations - 邀请记录列表。
 * @returns {void}
 * 副作用：总是先创建固定表头；时间戳按 `formatDate` 使用本地时区展示。
 */
function renderInvitations(invitations) {
  const table = $("#invitation-table");
  table.replaceChildren(createDataRow(["邮箱", "角色", "状态", "有效期"], true));
  invitations.forEach((invite) => {
    const status = document.createElement("span");
    status.className = `status-label${invite.status === "pending" ? "" : invite.status === "expired" ? " status-label--warning" : " status-label--muted"}`;
    status.textContent = { pending: "待激活", used: "已使用", expired: "已过期" }[invite.status] || invite.status;
    table.append(createDataRow([
      invite.email,
      invite.role === "admin" ? "管理员" : "成员",
      status,
      formatDate(invite.expires_at),
    ]));
  });
}

/**
 * 获取管理员可见的用户列表并渲染，失败时显示全局提示。
 *
 * @returns {Promise<void>} 请求与渲染或错误提示完成后兑现，不向事件调用方抛错。
 */
async function loadUsers() {
  try {
    const data = await api("/api/admin/users");
    renderUsers(data.users);
  } catch (error) { showToast(error.message); }
}

/**
 * 重建成员数据表，并为每个成员创建启用或停用操作。
 *
 * @param {Array<{id:string, display_name:string, email:string, role:string, is_active:boolean, created_at:number}>} users - 用户公开资料。
 * @returns {void}
 * 副作用：当前登录用户自己的操作按钮会被禁用，防止前端发起自停用请求。
 */
function renderUsers(users) {
  const table = $("#user-table");
  table.replaceChildren(createDataRow(["成员", "角色", "加入时间", "状态"], true));
  users.forEach((user) => {
    const member = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = user.display_name;
    const email = document.createElement("div");
    email.textContent = user.email;
    email.style.color = "var(--muted)";
    email.style.fontSize = "10px";
    member.append(name, email);
    const action = document.createElement("button");
    action.type = "button";
    action.className = "row-action";
    action.textContent = user.is_active ? "停用" : "启用";
    action.disabled = user.id === state.user.id;
    action.addEventListener("click", () => updateUserStatus(user.id, !user.is_active));
    table.append(createDataRow([
      member,
      user.role === "admin" ? "管理员" : "成员",
      formatDate(user.created_at),
      action,
    ]));
  });
}

/**
 * 更新指定成员的启用状态，并在成功后刷新整个用户列表。
 *
 * @param {string} userId - 目标用户 ID。
 * @param {boolean} active - 需要设置的新状态。
 * @returns {Promise<void>} 更新、刷新和提示处理完成后兑现；错误会转为全局提示。
 */
async function updateUserStatus(userId, active) {
  try {
    await api(`/api/admin/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      body: JSON.stringify({ active }),
    });
    await loadUsers();
    showToast("成员状态已更新");
  } catch (error) { showToast(error.message); }
}

/**
 * 创建统一的数据表行，可安全容纳纯文本或预构造 DOM 节点。
 *
 * @param {Array<string|Node>} cells - 按列顺序排列的内容。
 * @param {boolean} [header=false] - 是否附加表头样式类。
 * @returns {HTMLDivElement} 尚未插入文档的数据行元素。
 */
function createDataRow(cells, header = false) {
  const row = document.createElement("div");
  row.className = `data-row${header ? " data-row--header" : ""}`;
  cells.forEach((cell) => {
    const wrapper = document.createElement("div");
    if (cell instanceof Node) wrapper.append(cell);
    else wrapper.textContent = cell;
    row.append(wrapper);
  });
  return row;
}

/**
 * 根据文件选择控件更新上传摘要文本。
 *
 * @returns {void}
 * 副作用：有文件时显示数量，否则恢复支持格式提示；不会读取文件内容。
 */
function updateFileSummary() {
  const files = Array.from($("#upload-files").files || []);
  $("#file-summary").textContent = files.length ? `${files.length} 个文件` : "TXT / MD / CSV / JSON";
}

/**
 * 读取所选文本文件与粘贴内容，并作为一次管理员文档入库请求提交。
 *
 * @param {SubmitEvent} event - 文档上传表单提交事件。
 * @returns {Promise<void>} 文件读取、请求和表单解锁完成后兑现。
 * 副作用：请求期间禁用表单；无内容时产生本地错误。成功后更新文档总数、
 *   清空文件和粘贴区并显示服务端消息，失败时在表单内显示错误。
 */
async function handleUpload(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const files = Array.from($("#upload-files").files || []);
  const documents = [];
  $("#upload-error").textContent = "";
  setFormBusy(form, true);
  try {
    for (const file of files) documents.push(await file.text());
    const pasted = $("#upload-text").value.trim();
    if (pasted) documents.push(pasted);
    if (!documents.length) throw new Error("请选择文件或粘贴文档内容。");
    const data = await api("/documents/upload", {
      method: "POST",
      body: JSON.stringify({ documents, source: $("#upload-source").value }),
    });
    $("#document-total").textContent = `${data.total} 条`;
    $("#upload-files").value = "";
    $("#upload-text").value = "";
    updateFileSummary();
    showToast(data.message);
  } catch (error) {
    $("#upload-error").textContent = error.message;
  } finally {
    setFormBusy(form, false);
  }
}

/**
 * 把 Unix 秒时间戳格式化为中文本地月日和时分。
 *
 * @param {number} timestamp - 自 Unix 纪元起的秒数。
 * @returns {string} 由浏览器 `zh-CN` 国际化实现生成的本地时区文本。
 */
function formatDate(timestamp) {
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(timestamp * 1000));
}

initialize();
