import { MusicPlayer } from "@/components/MusicPlayer"
import { useState, useEffect } from "react"
import { recommendationsAPI, type RecommendationResponse } from "@/services/api"

export default function PlayerPage() {
  const [recommendations, setRecommendations] = useState<RecommendationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadRecommendations = async () => {
      try {
        const response = await recommendationsAPI.getTop5Suggestions();
        setRecommendations(response.data);
        // Log the raw recommendations from the model
        console.log('Top 5 song recommendations:', response.data);
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

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <div className="space-y-4">
        <h2 className="text-2xl font-bold">Top Song Recommendations</h2>
        <pre className="bg-gray-800 p-4 rounded-lg overflow-auto">
          {JSON.stringify(recommendations, null, 2)}
        </pre>
      </div>
    </main>
  );
}