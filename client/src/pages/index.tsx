import { DemographicsForm } from "@/components/DemographicsForm"
import { SongSearch } from "@/components/SongSearch"
import { ListeningHistory } from "@/components/ListeningHistory"
import { PredictionDisplay } from "@/components/PredictionDisplay"
import { HowItWorksModal } from "@/components/HowItWorksModal"
import { useState, useEffect, useRef } from "react"
import { userAPI, feedbackAPI } from "@/services/api"
import { useRouter } from 'next/router'
import { Button } from "@/components/ui/button"
import { RefreshCw, Lock, HelpCircle } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { toast } from "sonner"

interface Song {
  track_title: string;
  artist_name: string;
  album_title: string;
  id: number;
}

export default function HomePage() {
  const router = useRouter();
  const [listeningHistory, setListeningHistory] = useState<Song[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [demographics, setDemographics] = useState<any>(null);
  const [isEditingDemographics, setIsEditingDemographics] = useState(false);
  const regeneratePredictionsRef = useRef<(() => void) | undefined>();
  const [showHowItWorks, setShowHowItWorks] = useState(false);

  const isDemographicsSet = demographics !== null;

  useEffect(() => {
    // Load listening history and demographics on component mount
    const loadData = async () => {
      try {
        setLoading(true);
        const [historyResponse, demographicsResponse] = await Promise.all([
          feedbackAPI.getHistory(),
          userAPI.getDemographics()
        ]);
        
        setListeningHistory(historyResponse.data.feedback_history || []);
        if (demographicsResponse.data) {
          setDemographics(demographicsResponse.data);
        }
      } catch (error) {
        console.error('Failed to load data:', error);
        setError('Failed to load data');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  const handleDemographicsSubmit = async (data: any) => {
    try {
      const response = await userAPI.updateDemographics(data);
      setDemographics(response.data.data);
      setIsEditingDemographics(false);
    } catch (error) {
      console.error('Failed to update demographics:', error);
    }
  };

  const handleForgetMe = async () => {
    try {
      // Call the API to anonymize user data and get new session user ID
      const response = await userAPI.anonymizeUserData();
      const newUserId = response.data.new_user_id;
      
      // Clear local state
      setDemographics(null);
      setListeningHistory([]);
      
      // Clear any stored predictions cache
      localStorage.removeItem('predictions_cached');
      
      // Update session with new user ID
      if (newUserId) {
        localStorage.setItem('user_id', newUserId);
        localStorage.setItem('session_id', newUserId);
        
        // Clear any other user-specific data from localStorage
        const keysToKeep = ['theme'];
        Object.keys(localStorage).forEach(key => {
          if (!keysToKeep.includes(key)) {
            localStorage.removeItem(key);
          }
        });
      }
      
      // Show success message
      toast.success('Your data has been anonymized and a new session started');
      
      // Refresh the page to reset all components with new user ID
      window.location.reload();
      
    } catch (error) {
      console.error('Failed to anonymize user data:', error);
      toast.error('Failed to reset your data. Please try again.');
    }
  };

  const handleAddToHistory = async (song: Song) => {
    try {
      await feedbackAPI.addToHistory(song.id);
      setListeningHistory((prev) => {
        // Don't add if already in history
        if (prev.some((s) => s.id === song.id)) {
          return prev;
        }
        return [song, ...prev];
      });
    } catch (error) {
      console.error('Failed to add song to history:', error);
    }
  };

  const handleRemoveFromHistory = async (songId: number) => {
    try {
      await feedbackAPI.removeFromHistory(songId);
      setListeningHistory((prev) => prev.filter((song) => song.id !== songId));
    } catch (error) {
      console.error('Failed to remove song from history:', error);
    }
  };

  const handleEditToggle = () => {
    setIsEditingDemographics(!isEditingDemographics);
  };

  const handleRegeneratePredictions = () => {
    regeneratePredictionsRef.current?.();
  };

  const renderLockedOverlay = () => {
    if (isDemographicsSet) return null;
    
    return (
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 rounded-xl">
        <div className="flex flex-col items-center gap-3 p-6 text-center">
          <Lock className="h-8 w-8 text-gray-400" />
          <p className="text-gray-200">
            Please fill out your demographics first<br />
            <span className="text-sm text-gray-400">This helps us provide better recommendations</span>
          </p>
        </div>
      </div>
    );
  };

  return (
    <main className="min-h-screen gradient-background">
      <div className="container mx-auto px-6 py-8 flex flex-col min-h-screen">
        <div className="text-center mb-6 flex-none flex items-center justify-center gap-4">
          <h1 className="text-6xl font-black brand-text mb-2">
            prdct
          </h1>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowHowItWorks(true)}
              className="gap-2 text-muted-foreground hover:text-foreground"
            >
              <HelpCircle className="h-4 w-4" />
              How it Works
            </Button>
            <ThemeToggle />
          </div>
        </div>
        
        <HowItWorksModal 
          open={showHowItWorks} 
          onOpenChange={setShowHowItWorks} 
        />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
          <div className="space-y-6 h-full flex flex-col">
            <div className="p-6 rounded-xl gradient-card glow-border flex-none">
              <h2 className="text-2xl font-semibold mb-6 text-foreground/90">Your Profile</h2>
              <div>
                <DemographicsForm 
                  onSubmit={handleDemographicsSubmit}
                  initialData={demographics}
                  onForgetMe={handleForgetMe}
                  onEditToggle={handleEditToggle}
                />
              </div>
            </div>

            {!isEditingDemographics && (
              <div className="p-6 rounded-xl gradient-card glow-border flex-1 min-h-[500px] flex flex-col relative">
                {renderLockedOverlay()}
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-semibold text-foreground/90">Predictions</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRegeneratePredictions}
                    className="hover:bg-accent"
                    disabled={!isDemographicsSet}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Regenerate
                  </Button>
                </div>
                <div className="flex-1 min-h-0">
                  <PredictionDisplay 
                    listeningHistory={listeningHistory}
                    onAddToHistory={handleAddToHistory}
                    regenerateRef={regeneratePredictionsRef}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6 h-full flex flex-col">
            <div className="p-6 rounded-xl gradient-card glow-border flex-none relative">
              {renderLockedOverlay()}
              <h2 className="text-2xl font-semibold mb-4 text-foreground/90">Track Your Music</h2>
              <div className="h-[300px]">
                <SongSearch 
                  onAddToHistory={handleAddToHistory}
                />
              </div>
            </div>

            <div className="p-6 rounded-xl gradient-card glow-border flex-1 min-h-[500px] overflow-auto relative">
              {renderLockedOverlay()}
              <ListeningHistory
                songs={listeningHistory}
                onRemove={handleRemoveFromHistory}
                onPlay={() => {}}
                error={!isDemographicsSet ? "Please fill out your demographics first" : error}
                loading={loading}
              />
            </div>

            {isEditingDemographics && (
              <div className="p-6 rounded-xl gradient-card glow-border flex-none">
                {renderLockedOverlay()}
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-semibold text-foreground/90">Predictions</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRegeneratePredictions}
                    className="hover:bg-accent"
                    disabled={!isDemographicsSet}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Regenerate
                  </Button>
                </div>
                <div className="flex-1 min-h-0">
                  <PredictionDisplay 
                    listeningHistory={listeningHistory}
                    onAddToHistory={handleAddToHistory}
                    regenerateRef={regeneratePredictionsRef}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}