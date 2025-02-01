import { MusicPlayer } from "@/components/MusicPlayer"
import { useState, useEffect } from "react"
import { recommendationsAPI } from "@/services/api"

interface Song {
  track_title: string;
  artist_name: string;
  album_title: string;
  id: number;
}

export default function PlayerPage() {
  const [recommendations, setRecommendations] = useState<Song[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadRecommendations = async () => {
      try {
        const response = await recommendationsAPI.getPersonalized();
        setRecommendations(response.data);
      } catch (error) {
        setError('Failed to load recommendations');
        console.error('Failed to load recommendations:', error);
      } finally {
        setLoading(false);
      }
    };
    loadRecommendations();
  }, []);

  if (loading) {
    return (
      <main className="flex min-h-screen flex-col items-center p-24">
        <p>Loading...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="flex min-h-screen flex-col items-center p-24">
        <p className="text-red-500">{error}</p>
      </main>
    );
  }
}