import { NextRequest, NextResponse } from "next/server";

export function middleware(request: NextRequest) {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    return NextResponse.next();
  }

  const headers = new Headers(request.headers);
  headers.set("x-api-key", apiKey);
  return NextResponse.next({ request: { headers } });
}

export const config = {
  matcher: "/api/ingest/:path*",
};
