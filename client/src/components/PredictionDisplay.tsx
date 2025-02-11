import { useState, useEffect, useCallback, useMemo } from "react"
import { recommendationsAPI, musicAPI } from "@/services/api"
import { SongCard, type Song } from "./SongCard"
import { Button } from "@/components/ui/button"
import { Code, List, Loader2, Star, StarOff, Music2, AlertCircle, Check, Copy } from "lucide-react"
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
  songDetails?: Record<string, Song>;
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
          
          let details: Record<string, Song> = {};
          
          // If we're using cache and the response includes cached song details, use those
          if (!force && isCaching && response.data.songDetails) {
            console.log('Using cached song details');
            details = response.data.songDetails;
          } else {
            // Otherwise fetch fresh song details
            console.log('Fetching fresh song details');
            const songIds = response.data.predictions.map(p => p.song_id);
            details = await fetchSongDetails(songIds);
          }
          
          // Filter predictions to only include songs we found details for
          const validPredictions = response.data.predictions.filter(p => details[p.song_id]);
          
          // Trim to requested limit if we have enough valid predictions
          const finalPredictions = validPredictions.slice(0, recommendationLimit);
          
          if (finalPredictions.length < recommendationLimit) {
            console.warn(`Only found ${finalPredictions.length} valid songs out of ${recommendationLimit} requested`);
          }

          // Update the predictions with only the valid ones
          const finalResponse = {
            ...response.data,
            predictions: finalPredictions,
            songDetails: details  // Include song details in the cached data
          };
          
          setPredictions(finalResponse);
          
          // Only keep details for the final predictions
          const finalDetails: Record<string, Song> = {};
          finalPredictions.forEach(p => {
            finalDetails[p.song_id] = details[p.song_id];
          });
          setSongDetails(finalDetails);

          // Cache predictions if caching is enabled
          if (isCaching && !force) {
            console.log('Caching predictions and song details in Redis');
            try {
              await recommendationsAPI.cachePredictions(finalResponse);
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
          toast.error('Failed to copy to clipboard');
        });
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleCaching}
            className="gap-2"
            title={isCaching ? "Predictions are being remembered" : "Click to remember these predictions"}
          >
            {isCaching ? (
              <Star className="h-4 w-4 text-primary" />
            ) : (
              <StarOff className="h-4 w-4 text-muted-foreground/60" />
            )}
          </Button>
          <span className="text-sm text-muted-foreground">Show</span>
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

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode(viewMode === 'json' ? 'cards' : 'json')}
            className="gap-2"
          >
            {viewMode === 'json' ? (
              <List className="h-4 w-4 text-muted-foreground/60" />
            ) : (
              <Code className="h-4 w-4 text-muted-foreground/60" />
            )}
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <Loader2 className="h-6 w-6 animate-spin text-primary/40" />
          <p className="text-sm text-muted-foreground">Generating predictions...</p>
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <AlertCircle className="h-6 w-6 text-destructive/60" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      ) : viewMode === 'json' ? (
        <div className="h-full flex-1 relative bg-muted/30 rounded-lg border border-border">
          <div className="absolute right-3 top-3 z-10">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleCopyToClipboard}
              className="h-8 px-3 gap-2 text-xs font-medium"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  Copy
                </>
              )}
            </Button>
          </div>
          <div className="absolute inset-0 overflow-auto rounded-lg pt-14 px-4 pb-4">
            <pre className="h-full text-sm font-mono leading-relaxed">
              <code className="block h-full text-foreground/90 whitespace-pre-wrap break-words">
                {JSON.stringify(predictions, null, 2)}
              </code>
            </pre>
          </div>
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-auto">
          <div className="space-y-3">
            {detailsLoading ? (
              <div className="flex items-center justify-center h-full">
                <Loader2 className="h-6 w-6 animate-spin text-primary/40" />
                <span className="ml-2 text-muted-foreground">Loading song details...</span>
              </div>
            ) : predictions?.predictions && predictions.predictions.length > 0 ? (
              predictions.predictions.map((prediction) => {
                const songDetail = songDetails[prediction.song_id];
                if (!songDetail) return null;
                
                return (
                  <SongCard
                    key={prediction.song_id}
                    song={songDetail}
                    onAction={() => onAddToHistory?.(songDetail)}
                    actionLabel={`Add to History (${(prediction.confidence * 100).toFixed(1)}% match)`}
                    showAction={true}
                    isCompact={true}
                  />
                );
              })
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center">
                <Music2 className="h-12 w-12 text-muted-foreground/30" />
                <div className="flex flex-col items-center">
                  <p className="text-muted-foreground">
                    Your song predictions will appear here<br />
                    <span className="text-sm text-muted-foreground/60">once you add songs to your history</span>
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
} 