import { DemographicsForm } from "@/components/DemographicsForm"
import { SongSearch } from "@/components/SongSearch"
import { ListeningHistory } from "@/components/ListeningHistory"
import { PredictionDisplay } from "@/components/PredictionDisplay"
import { useState, useEffect, useRef } from "react"
import { userAPI, feedbackAPI } from "@/services/api"
import { useRouter } from 'next/router'
import { Button } from "@/components/ui/button"
import { RefreshCw } from "lucide-react"

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
    } catch (error) {
      console.error('Failed to update demographics:', error);
    }
  };

  const handleForgetMe = async () => {
    try {
      // Delete user data
      await userAPI.deleteUserData();
      
      // Clear local state
      setDemographics(null);
      setListeningHistory([]);
      
      // Clear any stored session/tokens
      localStorage.removeItem('token');
      sessionStorage.clear();
      
      // Redirect to login page or home
      router.push('/');
      
    } catch (error) {
      console.error('Failed to delete user data:', error);
      setError('Failed to delete user data. Please try again.');
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

  return (
    <main className="min-h-screen gradient-dark">
      <div className="container mx-auto px-6 py-8 flex flex-col">
        <div className="text-center mb-6 flex-none">
          <h1 className="text-4xl font-black brand-text mb-2">
            prdct
          </h1>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto">
            Add songs to your listening history and get recommendations based on your music tastes
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
          <div className="space-y-6 h-full flex flex-col">
            <div className="p-6 rounded-xl gradient-card glow-border flex-none">
              <h2 className="text-2xl font-semibold mb-6 text-white">Your Profile</h2>
              <div>
                <DemographicsForm 
                  onSubmit={handleDemographicsSubmit}
                  initialData={demographics}
                  onForgetMe={handleForgetMe}
                  onEditToggle={handleEditToggle}
                />
              </div>
            </div>

            {/* Show Predictions here by default */}
            {!isEditingDemographics && (
              <div className="p-6 rounded-xl gradient-card glow-border flex-none">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-semibold text-white">Predictions</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRegeneratePredictions}
                    className="hover:bg-white/10"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Regenerate
                  </Button>
                </div>
                <PredictionDisplay 
                  listeningHistory={listeningHistory}
                  onAddToHistory={handleAddToHistory}
                  regenerateRef={regeneratePredictionsRef}
                />
              </div>
            )}
          </div>

          <div className="space-y-6 h-full flex flex-col">
            <div className="p-6 rounded-xl gradient-card glow-border flex-none">
              <h2 className="text-2xl font-semibold mb-4 text-white">Track Your Music</h2>
              <div className="h-[300px]">
                <SongSearch 
                  onAddToHistory={handleAddToHistory}
                />
              </div>
            </div>

            <div className="p-6 rounded-xl gradient-card glow-border flex-1 min-h-[500px] overflow-auto">
              <ListeningHistory
                songs={listeningHistory}
                onRemove={handleRemoveFromHistory}
                onPlay={() => {}}
                error={error}
                loading={loading}
              />
            </div>

            {/* Show Predictions here when in edit mode */}
            {isEditingDemographics && (
              <div className="p-6 rounded-xl gradient-card glow-border flex-none">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-semibold text-white">Predictions</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRegeneratePredictions}
                    className="hover:bg-white/10"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Regenerate
                  </Button>
                </div>
                <PredictionDisplay 
                  listeningHistory={listeningHistory}
                  onAddToHistory={handleAddToHistory}
                  regenerateRef={regeneratePredictionsRef}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}