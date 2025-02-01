import { DemographicsForm } from "@/components/DemographicsForm"
import { SongSearch } from "@/components/SongSearch"
import { ListeningHistory } from "@/components/ListeningHistory"
import { useState, useEffect } from "react"
import { userAPI, feedbackAPI } from "@/services/api"

interface Song {
  track_title: string;
  artist_name: string;
  album_title: string;
  id: number;
}

export default function HomePage() {
  const [listeningHistory, setListeningHistory] = useState<Song[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [demographics, setDemographics] = useState<any>(null);

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

  return (
    <main className="min-h-screen gradient-dark">
      <div className="container mx-auto px-6 py-8 h-screen flex flex-col">
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
                />
              </div>
            </div>
          </div>

          <div className="space-y-6 h-full flex flex-col">
            <div className="p-6 rounded-xl gradient-card glow-border flex-1 min-h-0 flex flex-col">
              <h2 className="text-2xl font-semibold mb-6 text-white flex-none">Track Your Music</h2>
              <div className="flex-1 min-h-0">
                <SongSearch onAddToHistory={handleAddToHistory} />
              </div>
            </div>

            <div className="p-6 rounded-xl gradient-card glow-border flex-none">
              {loading ? (
                <div className="text-center py-4">
                  <p className="text-lg text-gray-300">Loading history...</p>
                </div>
              ) : error ? (
                <div className="text-center py-4">
                  <p className="text-lg text-red-400">{error}</p>
                </div>
              ) : (
                <ListeningHistory
                  songs={listeningHistory}
                  onRemove={handleRemoveFromHistory}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}