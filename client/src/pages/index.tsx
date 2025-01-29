import { DemographicsForm } from "@/components/DemographicsForm"
import { Recommendations } from "@/components/Recommendations"
import { SongSearch } from "@/components/SongSearch"
import { ListeningHistory } from "@/components/ListeningHistory"
import { useState } from "react"

interface Song {
  track_title: string;
  artist_name: string;
  album_title: string;
  id: number;
}

export default function HomePage() {
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [listeningHistory, setListeningHistory] = useState<Song[]>([]);

  const handleDemographicsSubmit = (data: any) => {
    setShowRecommendations(true);
  };

  const handleAddToHistory = (song: Song) => {
    setListeningHistory((prev) => {
      // Don't add if already in history
      if (prev.some((s) => s.id === song.id)) {
        return prev;
      }
      return [song, ...prev];
    });
  };

  const handleRemoveFromHistory = (songId: number) => {
    setListeningHistory((prev) => prev.filter((song) => song.id !== songId));
  };

  return (
    <main className="min-h-screen gradient-dark">
      <div className="container mx-auto px-6 py-20 space-y-16">
        <div className="text-center space-y-6">
          <h1 className="text-5xl font-black brand-text">
            prdct
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Add songs to your listening history and get recommendations based on your music tastes
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 max-w-[1400px] mx-auto">
          <div className="space-y-10">
            <div className="p-8 rounded-xl gradient-card glow-border">
              <h2 className="text-3xl font-semibold mb-8 text-white">Your Profile</h2>
              {!showRecommendations ? (
                <DemographicsForm onSubmit={handleDemographicsSubmit} />
              ) : (
                <Recommendations />
              )}
            </div>
          </div>

          <div className="space-y-10">
            <div className="p-8 rounded-xl gradient-card glow-border">
              <h2 className="text-3xl font-semibold mb-8 text-white">Track Your Music</h2>
              <SongSearch onAddToHistory={handleAddToHistory} />
            </div>

            <div className="p-8 rounded-xl gradient-card glow-border">
              <ListeningHistory
                songs={listeningHistory}
                onRemove={handleRemoveFromHistory}
              />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}