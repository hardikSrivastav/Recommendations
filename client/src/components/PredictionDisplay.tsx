import { useState, useEffect, useCallback, useMemo } from "react"
import { recommendationsAPI, musicAPI } from "@/services/api"
import { SongCard, type Song } from "./SongCard"
import { Button } from "@/components/ui/button"
import { Code, List, Loader2, Star, StarOff, Music2 } from "lucide-react"
import { toast } from "sonner"
import {
  Select,
  Content as SelectContent,
  Option as SelectItem,
  Trigger as SelectTrigger,
} from "@/components/ui/select"

interface PredictionResponse {
  user_id: string;
  timestamp: string;
  context: {
    total_songs: number;
    recent_songs: string[];
    demographics: Record<string, any>;
    is_cold_start: boolean;
  };
  predictions: Array<{
    song_id: string;
    confidence: number;
    predictor_weights: Record<string, number>;
    was_shown: boolean;
    was_selected: boolean;
  }>;
}

interface PredictionDisplayProps {
  listeningHistory: Array<{
    track_title?: string;
    artist_name?: string;
    album_title?: string;
    id: number;
  }>;
  onAddToHistory?: (song: Song) => void;
  regenerateRef?: React.MutableRefObject<(() => void) | undefined>;
}

type ViewMode = 'json' | 'cards';

export function PredictionDisplay({ 
  listeningHistory, 
  onAddToHistory, 
  regenerateRef 
}: PredictionDisplayProps) {
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('cards');
  const [songDetails, setSongDetails] = useState<Record<string, Song>>({});
  const [recommendationLimit, setRecommendationLimit] = useState<number>(5);
  const [isCaching, setIsCaching] = useState(false);

  // Load caching state from localStorage on mount
  useEffect(() => {
    const cached = localStorage.getItem('predictions_cached');
    if (cached === 'true') {
      console.log('Restoring cached state from localStorage');
      setIsCaching(true);
    }
  }, []);

  const fetchSongDetails = useCallback(async (songIds: string[]) => {
    try {
      setDetailsLoading(true);
      const details: Record<string, Song> = {};
      const errors: string[] = [];
      
      await Promise.all(
        songIds.map(async (id) => {
          try {
            console.log('Fetching details for song:', id);
            const response = await musicAPI.getDetails(parseInt(id));
            console.log('Response for song', id, ':', response);
            if (response.data) {
              details[id] = {
                id: parseInt(id),
                track_title: response.data.track_title || 'Unknown Title',
                artist_name: response.data.artist_name || 'Unknown Artist',
                album_title: response.data.album_title || 'Unknown Album',
                track_genres: Array.isArray(response.data.track_genres) 
                  ? response.data.track_genres 
                  : [],
                track_duration: response.data.track_duration || 0,
                track_date_created: response.data.track_date_created,
                tags: Array.isArray(response.data.track_tags) 
                  ? response.data.track_tags 
                  : []
              };
            }
          } catch (error: any) {
            console.error(`Failed to fetch details for song ${id}:`, error);
            if (error.response?.status === 404) {
              errors.push(`Song ${id} not found`);
            } else {
              errors.push(`Error loading song ${id}: ${error.message}`);
            }
          }
        })
      );

      if (errors.length > 0) {
        console.log(`${errors.length} songs couldn't be loaded:`, errors);
        // Only show error toast if we don't have enough valid songs
        if (Object.keys(details).length < recommendationLimit) {
          toast.error(`Some songs couldn't be loaded and we couldn't find enough alternatives`);
        }
      }

      console.log('Final song details:', details);
      return details;
    } catch (error) {
      console.error('Failed to fetch song details:', error);
      setError('Failed to load song details');
      toast.error('Failed to load song details');
      return {};
    } finally {
      setDetailsLoading(false);
    }
  }, [recommendationLimit]);

  const fetchPredictions = useCallback(async (force: boolean = false) => {
    if (listeningHistory.length > 0) {
      setLoading(true);
      setError(null);
      setSongDetails({}); // Clear existing song details
      try {
        console.log(`Fetching predictions (force=${force}, isCaching=${isCaching})`);
        const response = await recommendationsAPI.getTop5Suggestions(recommendationLimit, isCaching && !force);
        if (response.data && response.data.predictions) {
          console.log('Received predictions:', response.data);
          
          // Get song details for all predictions
          const songIds = response.data.predictions.map(p => p.song_id);
          const details = await fetchSongDetails(songIds);
          
          // Filter predictions to only include songs we found details for
          const validPredictions = response.data.predictions.filter(p => details[p.song_id]);
          
          // Trim to requested limit if we have enough valid predictions
          const finalPredictions = validPredictions.slice(0, recommendationLimit);
          
          if (finalPredictions.length < recommendationLimit) {
            console.warn(`Only found ${finalPredictions.length} valid songs out of ${recommendationLimit} requested`);
          }

          // Update the predictions with only the valid ones
          setPredictions({
            ...response.data,
            predictions: finalPredictions
          });
          
          // Only keep details for the final predictions
          const finalDetails: Record<string, Song> = {};
          finalPredictions.forEach(p => {
            finalDetails[p.song_id] = details[p.song_id];
          });
          setSongDetails(finalDetails);

          // Cache predictions if caching is enabled
          if (isCaching && !force) {
            console.log('Caching predictions in Redis');
            try {
              await recommendationsAPI.cachePredictions({
                ...response.data,
                predictions: finalPredictions
              });
            } catch (error) {
              console.error('Failed to cache predictions:', error);
              toast.error('Failed to cache predictions');
            }
          }
        } else {
          setError('Invalid prediction data received');
          toast.error('Invalid prediction data received');
          setPredictions(null);
        }
      } catch (error: any) {
        console.error('Failed to get predictions:', error);
        setError('Failed to get predictions');
        toast.error('Failed to get predictions: ' + (error.message || 'Unknown error'));
        setPredictions(null);
      } finally {
        setLoading(false);
      }
    } else {
      setPredictions(null);
      setError(null);
      setSongDetails({});
    }
  }, [listeningHistory, recommendationLimit, isCaching, fetchSongDetails]);

  // Only update predictions when listening history changes if we're not caching
  useEffect(() => {
    console.log('Listening history changed, fetching predictions');
    fetchPredictions();
  }, [listeningHistory, recommendationLimit]);

  // Set up regenerate ref
  useEffect(() => {
    if (regenerateRef) {
      regenerateRef.current = () => {
        console.log('Regenerating predictions');
        fetchPredictions(true); // Force fetch new predictions
      };
    }
  }, [regenerateRef, fetchPredictions]);

  const toggleCaching = async () => {
    try {
      if (isCaching) {
        console.log('Disabling caching and clearing Redis cache');
        await recommendationsAPI.clearCache();
        localStorage.removeItem('predictions_cached');
      } else {
        console.log('Enabling caching and storing current predictions');
        if (predictions) {
          await recommendationsAPI.cachePredictions(predictions);
          localStorage.setItem('predictions_cached', 'true');
        }
      }
      setIsCaching(!isCaching);
      toast.success(isCaching ? 'Predictions will no longer be remembered' : 'These predictions will be remembered');
    } catch (error) {
      console.error('Failed to toggle caching:', error);
      toast.error('Failed to toggle prediction caching');
    }
  };

  const handleCopyToClipboard = () => {
    if (predictions) {
      navigator.clipboard.writeText(JSON.stringify(predictions, null, 2))
        .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        })
        .catch(err => {
          console.error('Failed to copy:', err);
        });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-6 w-6 animate-spin text-white" />
        <span className="ml-2 text-white">Loading predictions...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-400 p-4 rounded-lg bg-red-900/20 border border-red-900">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1 bg-gray-800/50 rounded-lg p-1">
          <Button
            variant={viewMode === 'json' ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => setViewMode('json')}
            className="gap-2"
          >
            <Code className="h-4 w-4" />
            JSON
          </Button>
          <Button
            variant={viewMode === 'cards' ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => setViewMode('cards')}
            className="gap-2"
          >
            <List className="h-4 w-4" />
            Cards
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleCaching}
            className="gap-2"
            title={isCaching ? "Predictions are being remembered" : "Click to remember these predictions"}
          >
            {isCaching ? (
              <Star className="h-4 w-4 text-yellow-400" />
            ) : (
              <StarOff className="h-4 w-4" />
            )}
          </Button>
          <span className="text-sm text-gray-400">Show</span>
          <Select
            value={recommendationLimit.toString()}
            onValueChange={(value) => setRecommendationLimit(parseInt(value))}
          >
            <SelectTrigger className="w-[100px]">
              <span>{recommendationLimit} songs</span>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="3">3 songs</SelectItem>
              <SelectItem value="5">5 songs</SelectItem>
              <SelectItem value="10">10 songs</SelectItem>
              <SelectItem value="15">15 songs</SelectItem>
              <SelectItem value="20">20 songs</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="h-[400px] bg-gray-800/50 rounded-lg">
        {viewMode === 'json' ? (
          <div className="relative h-full">
            <div className="sticky top-0 right-0 p-2 bg-gray-800/50 z-10 border-b border-white/10">
              <button
                onClick={handleCopyToClipboard}
                className="text-xs bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 px-2 py-1 rounded flex items-center gap-1"
                disabled={!predictions}
              >
                {copied ? (
                  <>
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>

            <div className="p-4 overflow-auto h-[calc(400px-3rem)]">
              <pre className="text-white font-mono text-sm">
                {predictions ? JSON.stringify(predictions, null, 2) : 'No predictions available'}
              </pre>
            </div>
          </div>
        ) : (
          <div className="h-full overflow-auto p-4">
            <div className="space-y-3">
              {detailsLoading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-6 w-6 animate-spin text-white" />
                  <span className="ml-2 text-white">Loading song details...</span>
                </div>
              ) : predictions?.predictions && predictions.predictions.length > 0 ? (
                predictions.predictions.map((prediction) => {
                  const songDetail = songDetails[prediction.song_id];
                  if (!songDetail) return null;
                  
                  return (
                    <SongCard
                      key={prediction.song_id}
                      song={songDetail}
                      className="bg-gray-800/50 hover:bg-gray-700/50"
                      actionLabel={`Add to History (${(prediction.confidence * 100).toFixed(1)}% match)`}
                      showAction={true}
                      isCompact={true}
                      onAction={() => onAddToHistory?.(songDetail)}
                    />
                  );
                })
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400 text-center">
                  No predictions available. Add some songs to your history to get started.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 