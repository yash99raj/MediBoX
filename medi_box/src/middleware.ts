import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

export default withAuth(
  function middleware(req) {
    const token = req.nextauth.token;
    const path = req.nextUrl.pathname;

    // Direct based on role
    if (path.startsWith("/doctor") && token?.role !== "doctor") {
      return NextResponse.redirect(new URL("/", req.url));
    }
    if (path.startsWith("/patient") && token?.role !== "patient") {
      return NextResponse.redirect(new URL("/", req.url));
    }
    return NextResponse.next();
  },
  {
    callbacks: {
      authorized: ({ token }) => !!token,
    },
  }
);

export const config = {
  matcher: ["/doctor/:path*", "/patient/:path*"],
};
