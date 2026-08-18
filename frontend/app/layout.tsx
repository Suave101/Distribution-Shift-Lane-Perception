import "./globals.css";

export const metadata = {
  title: "Distribution Shift Detection",
  description: "UI for Shift Detection Pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="dark">
      <body>{children}</body>
    </html>
  );
}