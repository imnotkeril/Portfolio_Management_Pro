/** Browser API client — requests go through same-origin BFF proxy. */

const API_BASE = "/api/proxy";

function redirectToLogin(): void {
  if (typeof window === "undefined") return;
  const from = window.location.pathname + window.location.search;
  const params = new URLSearchParams();
  if (from && from !== "/login" && from !== "/register") {
    params.set("from", from);
  }
  const qs = params.toString();
  window.location.href = qs ? `/login?${qs}` : "/login";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (response.status === 401) {
    redirectToLogin();
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    const text = await response.text();
    let detail = text || `HTTP ${response.status}`;
    try {
      const body = JSON.parse(text) as { detail?: unknown };
      if (body?.detail != null) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* use raw text */
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

/** Same-origin auth endpoints (not proxied to FastAPI path-for-path). */
export const authApi = {
  me: () =>
    fetch("/api/auth/me", { credentials: "include", cache: "no-store" }).then(
      async (res) => {
        if (res.status === 401) return null;
        if (!res.ok) throw new Error("Failed to load session");
        return res.json() as Promise<import("./types").User>;
      },
    ),
  login: (email: string, password: string) =>
    fetch("/api/auth/login", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),
  register: (email: string, password: string) =>
    fetch("/api/auth/register", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),
  logout: () =>
    fetch("/api/auth/logout", { method: "POST", credentials: "include" }),
};
