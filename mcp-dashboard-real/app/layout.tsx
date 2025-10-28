import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Protein Binder Design Dashboard',
  description: 'MCP Dashboard for NVIDIA BioNeMo Protein Binder Design',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  )
}