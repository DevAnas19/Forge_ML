import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "ForgeML",
  description: "Intelligent ML Experimentation & Deployment Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900">
        <nav className="bg-white border-b px-8 py-4 flex gap-6 items-center">
          <span className="font-bold text-lg mr-4">ForgeML</span>
          <Link href="/" className="text-sm text-gray-600 hover:text-black">Dashboard</Link>
          <Link href="/datasets" className="text-sm text-gray-600 hover:text-black">Datasets</Link>
          <Link href="/experiments" className="text-sm text-gray-600 hover:text-black">Experiments</Link>
          <Link href="/models" className="text-sm text-gray-600 hover:text-black">Models</Link>
          <Link href="/playground" className="text-sm text-gray-600 hover:text-black">Playground</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}