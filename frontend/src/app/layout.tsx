import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { MainNav } from "@/components/main-nav";
import { Providers } from "@/components/providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "WILD MARKET Portfolio Terminal",
  description: "Portfolio Management, Analytics, Risk and Forecasting",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Providers>
          <MainNav />
          <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
            {children}
          </main>
          <footer className="border-t border-white/10 px-6 py-4 text-sm text-white/60">
            <nav
              aria-label="Legal"
              className="mx-auto flex max-w-7xl flex-wrap gap-x-6"
            >
              <a className="py-3" href="/privacy">
                Privacy
              </a>
              <a className="py-3" href="/terms">
                Terms
              </a>
              <a className="py-3" href="/refund">
                Refunds
              </a>
            </nav>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
