import { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

// Extend NextAuth typings to support custom roles
declare module "next-auth" {
  interface User {
    id: string;
    role?: string;
  }
  interface Session {
    user?: {
      name?: string | null;
      email?: string | null;
      image?: string | null;
      role?: string;
    };
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    role?: string;
  }
}

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
        role: { label: "Role", type: "text" }
      },
      async authorize(credentials) {
        if (!credentials) return null;
        const { username, password, role } = credentials;

        // Hardcoded credentials for development
        if (role === "doctor" && username === "doctor1" && password === "password123") {
          return { id: "d1", name: "Dr. Smith", email: "smith@medibox.com", role: "doctor" };
        }
        if (role === "patient" && username === "patient1" && password === "password123") {
          return { id: "p1", name: "John Doe", email: "john@medibox.com", role: "patient" };
        }
        
        return null;
      }
    })
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = user.role;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.role = token.role;
      }
      return session;
    }
  },
  pages: {
    signIn: "/",
  },
  secret: process.env.NEXTAUTH_SECRET || "supersecretdevelopmentkey",
};
