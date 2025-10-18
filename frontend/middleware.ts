import { NextRequest, NextResponse } from "next/server";

export function middleware(req: NextRequest) {
  const path = req.nextUrl.pathname;
  const protectedPrefixes = ["/dashboard", "/leads"];
  const isProtected = protectedPrefixes.some((p) => path.startsWith(p));

  if (isProtected) {
    const sessionCookie = req.cookies.get("session_id")?.value;
    if (!sessionCookie) {
      const url = req.nextUrl.clone();
      url.pathname = "/login";
      url.searchParams.set("redirect", path);
      return NextResponse.redirect(url);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/leads/:path*"],
};

