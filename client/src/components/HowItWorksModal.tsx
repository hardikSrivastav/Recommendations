import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { Brain, Users, Music2, Sparkles, History, Database } from "lucide-react"

interface HowItWorksModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function HowItWorksModal({ open, onOpenChange }: HowItWorksModalProps) {
  const features = [
    {
      icon: <Brain className="h-6 w-6 text-primary" />,
      title: "Intelligent Predictions",
      description: "Our AI model learns from your music preferences and combines multiple prediction strategies to suggest songs you'll love."
    },
    {
      icon: <Users className="h-6 w-6 text-primary" />,
      title: "Demographic Analysis",
      description: "We analyze listening patterns of users with similar demographics to enhance recommendation accuracy."
    },
    {
      icon: <History className="h-6 w-6 text-primary" />,
      title: "Learning History",
      description: "Your listening history helps us understand your taste and improve predictions over time."
    },
    {
      icon: <Database className="h-6 w-6 text-primary" />,
      title: "Rich Music Database",
      description: "Access to a vast collection of songs with detailed metadata including genres, artists, and more."
    },
    {
      icon: <Sparkles className="h-6 w-6 text-primary" />,
      title: "Smart Caching",
      description: "Intelligent caching system remembers your preferences and provides faster recommendations."
    },
    {
      icon: <Music2 className="h-6 w-6 text-primary" />,
      title: "Music Discovery",
      description: "Discover new songs that match your taste while maintaining a balance between familiarity and exploration."
    }
  ]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl p-8">
        <DialogHeader className="space-y-3">
          <DialogTitle className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
            How It Works
          </DialogTitle>
          <DialogDescription className="text-lg text-muted-foreground/90 leading-relaxed">
            Discover how our AI-powered music recommendation system brings you personalized song suggestions
          </DialogDescription>
        </DialogHeader>
        
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group flex gap-4 p-5 rounded-xl bg-gradient-to-br from-muted/50 to-muted/30 border border-border hover:border-primary/20 hover:shadow-md transition-all duration-300"
            >
              <div className="mt-1 p-3 rounded-lg bg-background shadow-sm border border-border/50 group-hover:border-primary/20 group-hover:shadow-lg transition-all duration-300">
                {feature.icon}
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-lg text-foreground/90">
                  {feature.title}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 p-6 rounded-xl bg-gradient-to-br from-primary/5 via-primary/3 to-transparent border border-primary/10">
          <h3 className="font-semibold text-lg text-primary mb-3">Technical Overview</h3>
          <p className="text-sm text-muted-foreground/90 leading-relaxed">
            Our system uses a sophisticated weighted ensemble approach combining collaborative filtering, 
            demographic analysis, and popularity metrics. The model adapts to both cold-start scenarios 
            and evolving user preferences, utilizing Redis for caching and PostgreSQL for persistent storage. 
            Real-time predictions are enhanced through a confidence calculation system that considers multiple factors.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
} 