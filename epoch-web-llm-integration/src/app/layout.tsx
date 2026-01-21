import { AgentSidebar } from "@/components/agent-sidebar";
import { BrowserView } from "@/components/browser-view";
import { Footer } from "@/components/footer";
import { Navbar } from "@/components/nav";
import { NetworkStatusListener } from "@/components/network-status";
import { Sidebar } from "@/components/sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import { cn } from "@/lib/utils";
import Providers from "@/providers";
import { ClerkProvider } from "@clerk/nextjs";
import { Analytics } from "@vercel/analytics/react";
import { GeistSans } from "geist/font/sans";
import type { Metadata } from "next";
import { JetBrains_Mono as Mono } from "next/font/google";
import "./globals.css";

const mono = Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono",
});

const title = "Think AI";
const description = "Open-source AI powered answer engine.";

export const metadata: Metadata = {
  metadataBase: new URL("https://farfalle.dev/"),
  title,
  description,
  openGraph: {
    title,
    description,
  },
  twitter: {
    title,
    description,
    card: "summary_large_image",
    creator: "@rashadphz",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider>
      <html lang="en" suppressHydrationWarning>
        <body
          className={cn("antialiased", GeistSans.className, mono.className)}
        >
          <Providers>
            {/* ... rest of the layout ... */}
            <ThemeProvider
              attribute="class"
              defaultTheme="dark"
              enableSystem
              disableTransitionOnChange
            >
              <Navbar />
              <div className="flex h-screen pt-[60px] overflow-hidden">
                <Sidebar />
                <main className="flex-1 w-full h-full overflow-y-auto relative bg-background">
                  <BrowserView />
                  {children}
                  <Footer />
                </main>
              </div>
              <Toaster />
              <NetworkStatusListener />
              <AgentSidebar />
              <Analytics />
              <script
                dangerouslySetInnerHTML={{
                  __html: `
                    // Service Worker management is now handled in Providers
                  `,
                }}
              />
            </ThemeProvider>
          </Providers>
        </body>
      </html>
    </ClerkProvider>
  );
}
