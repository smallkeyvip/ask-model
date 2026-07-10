"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  user: null,
  sessions: [],
  currentSessionId: null,
  busy: false,
  currentView: "chat",
};

let toastTimer = null;

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

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3200);
}

function setFormBusy(form, busy) {
  form.querySelectorAll("input, textarea, select, button").forEach((control) => {
    control.disabled = busy;
  });
}

function showAuth(mode = "login") {
  $("#auth-view").classList.remove("hidden");
  $("#app-view").classList.add("hidden");
  $("#login-form").classList.toggle("hidden", mode !== "login");
  $("#invite-form").classList.toggle("hidden", mode !== "invite");
}

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

async function handleLogout() {
  try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } catch (_) { /* local logout still proceeds */ }
  resetAuthentication();
}

function resetAuthentication() {
  state.user = null;
  state.sessions = [];
  state.currentSessionId = null;
  clearMessages();
  $("#login-password").value = "";
  closeSidebar();
  showAuth("login");
}

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

function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#sidebar-scrim").classList.remove("hidden");
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#sidebar-scrim").classList.add("hidden");
}

async function loadSessions() {
  const data = await api("/api/sessions");
  state.sessions = data.sessions;
  renderSessions();
}

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

function clearMessages() {
  const list = $("#message-list");
  list.querySelectorAll(".message").forEach((message) => message.remove());
  $("#empty-chat").classList.remove("hidden");
  $("#chat-status").textContent = "";
}

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

function resizeComposer() {
  const input = $("#chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

function selectAdminTab(tab) {
  $$(".admin-tab").forEach((button) => button.classList.toggle("active", button.dataset.adminTab === tab));
  $("#admin-invite").classList.toggle("hidden", tab !== "invite");
  $("#admin-documents").classList.toggle("hidden", tab !== "documents");
  $("#admin-users").classList.toggle("hidden", tab !== "users");
  if (tab === "invite") loadInvitations();
  if (tab === "users") loadUsers();
}

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

async function loadInvitations() {
  try {
    const data = await api("/api/admin/invitations");
    renderInvitations(data.invitations);
  } catch (error) { showToast(error.message); }
}

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

async function loadUsers() {
  try {
    const data = await api("/api/admin/users");
    renderUsers(data.users);
  } catch (error) { showToast(error.message); }
}

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

function updateFileSummary() {
  const files = Array.from($("#upload-files").files || []);
  $("#file-summary").textContent = files.length ? `${files.length} 个文件` : "TXT / MD / CSV / JSON";
}

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

function formatDate(timestamp) {
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(timestamp * 1000));
}

initialize();
