"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { authApi } from "@/lib/api";
import type { User } from "@/lib/types";

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  deleteAccount: (password: string) => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const me = await authApi.me();
      setUser(me);
    } catch {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    authApi
      .me()
      .then((me) => {
        if (!cancelled) setUser(me);
      })
      .catch(() => {
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await authApi.login(email, password);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          typeof body?.detail === "string"
            ? body.detail
            : "Login failed",
        );
      }
      await refresh();
    },
    [refresh],
  );

  const register = useCallback(
    async (email: string, password: string) => {
      const res = await authApi.register(email, password);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          typeof body?.detail === "string"
            ? body.detail
            : "Registration failed",
        );
      }
      await refresh();
    },
    [refresh],
  );

  const logout = useCallback(async () => {
    await authApi.logout();
    setUser(null);
    router.push("/login");
    router.refresh();
  }, [router]);

  const deleteAccount = useCallback(
    async (password: string) => {
      const res = await authApi.deleteAccount(password);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : "Could not delete the account",
        );
      }
      setUser(null);
      router.push("/");
      router.refresh();
    },
    [router],
  );

  const value = useMemo(
    () => ({ user, loading, login, register, logout, deleteAccount, refresh }),
    [user, loading, login, register, logout, deleteAccount, refresh],
  );

  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
