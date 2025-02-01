import { Montserrat } from "next/font/google"

const montserrat = Montserrat({ 
  subsets: ["latin"],
  weight: ["300", "400", "500", "700", "900"]
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className={`${montserrat.className} min-h-screen bg-background text-foreground antialiased`}>
      {children}
    </div>
  );
}