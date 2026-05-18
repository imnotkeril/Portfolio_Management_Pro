import { NextRequest, NextResponse } from "next/server";

import {
  AUTH_COOKIE,
  getBackendUrl,
  isPublicProxyPath,
} from "@/lib/auth-cookie";

type RouteContext = { params: Promise<{ path: string[] }> };

async function proxyRequest(
  request: NextRequest,
  context: RouteContext,
): Promise<NextResponse> {
  const { path } = await context.params;
  const pathStr = path.join("/");
  const token = request.cookies.get(AUTH_COOKIE)?.value;

  if (!token && !isPublicProxyPath(pathStr)) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  const target = new URL(`${getBackendUrl()}/${pathStr}`);
  request.nextUrl.searchParams.forEach((value, key) => {
    target.searchParams.set(key, value);
  });

  const headers: HeadersInit = {};
  const contentType = request.headers.get("content-type");
  if (contentType) headers["Content-Type"] = contentType;
  if (token) headers.Authorization = `Bearer ${token}`;

  const init: RequestInit = {
    method: request.method,
    headers,
    cache: "no-store",
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
  }

  const upstream = await fetch(target.toString(), init);
  const body = await upstream.text();

  return new NextResponse(body, {
    status: upstream.status,
    headers: {
      "Content-Type":
        upstream.headers.get("content-type") ?? "application/json",
    },
  });
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PATCH = proxyRequest;
export const PUT = proxyRequest;
export const DELETE = proxyRequest;
