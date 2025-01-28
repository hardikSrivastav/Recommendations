import { MusicPlayer } from "./MusicPlayer"
import { useEffect, useState } from "react"

interface Song {
  id: string;
  title: string;
  artist: string;
  url: string;
}

export function Recommendations() {
  const [recommendations, setRecommendations] = useState<Song[]>([]);

  useEffect(() => {
    // GET from /recommendations (to be implemented)
    // Fetch initial recommendations
  }, []);

  const handlePlayComplete = (songId: string) => {
    // GET from /recommendations (to be implemented)
    // Fetch new recommendations based on listening history
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-center">
        Recommended for You
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {recommendations.map((song) => (
          <MusicPlayer
            key={song.id}
            song={song}
            onPlayComplete={handlePlayComplete}
          />
        ))}
      </div>
    </div>
  );
}