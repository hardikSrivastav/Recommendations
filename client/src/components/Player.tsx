import { Button } from "@/components/ui/button"
import { useState, useEffect } from "react"
import { recommendationsAPI } from "@/services/api"

interface PredictionDisplayProps {
  listeningHistory: Array<{
    track_title: string;
    artist_name: string;
    album_title: string;
    id: number;
  }>;
  onClose: () => void;
}

export function PredictionDisplay({ listeningHistory, onClose }: PredictionDisplayProps) {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Only fetch predictions if we have listening history
    if (listeningHistory.length > 0) {
      setLoading(true);
      recommendationsAPI.getTop5Suggestions()
        .then(response => {
          setPredictions(response.data);
        })
        .catch(error => {
          console.error('Failed to get predictions:', error);
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [listeningHistory]); // Update whenever listening history changes

  // Don't show anything if no listening history
  if (listeningHistory.length === 0) return null;

  // Get the most recent song
  const latestSong = listeningHistory[0];

  return (
    <div className="bg-background/95 backdrop-blur-sm border-t border-accent/20 p-4">
      <div className="container mx-auto">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-accent/20 rounded-md flex items-center justify-center">
              <span className="text-2xl">🎯</span>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white">Next Song Predictions</h3>
              <p className="text-sm text-gray-400">Based on {latestSong.track_title}</p>
            </div>
          </div>

          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="text-white hover:text-primary"
          >
            <span className="text-xl">✕</span>
          </Button>
        </div>

        <div className="space-y-4">
          {loading ? (
            <p className="text-sm text-gray-400">Generating predictions...</p>
          ) : predictions.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {predictions.map((prediction, index) => (
                <div 
                  key={index}
                  className="p-3 rounded-lg bg-accent/10 border border-accent/20"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-sm font-medium text-white">Song ID: {prediction.song_id}</p>
                      <p className="text-xs text-gray-400">Score: {prediction.score.toFixed(3)}</p>
                    </div>
                    <div className="text-xs text-gray-500">#{index + 1}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-400">No predictions available</p>
          )}
        </div>
      </div>
    </div>
  );
} 