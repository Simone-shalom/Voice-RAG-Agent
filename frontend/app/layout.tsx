import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Knowledge Agent",
  description: "Ask questions about your audio library using voice",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">{children}</body>
    </html>
  );
}
