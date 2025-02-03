import { useState, useEffect, useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Loader2, Music2, AlertCircle } from "lucide-react";
import { musicAPI } from "@/services/api";
import { SongCard, type Song } from "./SongCard";
import { cn } from "@/lib/utils";
import debounce from "lodash/debounce";
import { toast } from "sonner";

interface SearchResponse {
  results: Song[];
  total_results: number;
  limit: number;
  query: string;
}

interface SongSearchProps {
  onAddToHistory: (song: Song) => Promise<void>;
}

export function SongSearch({ onAddToHistory }: SongSearchProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [songs, setSongs] = useState<Song[]>([]);
  const [totalResults, setTotalResults] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [addingToHistory, setAddingToHistory] = useState<number | null>(null);

  // Debounced search function
  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (!query.trim()) {
        setSongs([]);
        setTotalResults(0);
        setHasSearched(false);
        return;
      }

      setIsLoading(true);
      setError(null);
      setHasSearched(true);
      
      try {
        const response = await musicAPI.search(query);
        const data = response.data as SearchResponse;
        setSongs(data.results || []);
        setTotalResults(data.total_results || 0);
      } catch (error) {
        console.error("Failed to search songs:", error);
        setError("Failed to search songs");
      } finally {
        setIsLoading(false);
      }
    }, 200),
    []
  );

  useEffect(() => {
    debouncedSearch(searchQuery);
    return () => {
      debouncedSearch.cancel();
    };
  }, [searchQuery, debouncedSearch]);

  const handleAddToHistory = async (song: Song) => {
    try {
      setAddingToHistory(song.id);
      await onAddToHistory(song);
      toast.success(`Added "${song.track_title}" to history`);
    } catch (error: any) {
      console.error("Failed to add song to history:", error);
      const errorMessage = error.response?.data?.error || "Failed to add song to history";
      toast.error(errorMessage);
    } finally {
      setAddingToHistory(null);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-none pb-4">
        <div className="relative">
          <Input
            type="text"
            placeholder="Search for songs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={cn(
              "pl-12 py-4 text-base",
              "bg-background/80 backdrop-blur-sm",
              "border-white/10 focus:border-primary/50",
              "transition-all duration-300",
              "placeholder:text-gray-500"
            )}
          />
          <Search className="absolute left-4 top-2 h-5 w-5 text-gray-400" />
        </div>
      </div>

      <div className="flex-1 overflow-hidden relative">
        {isLoading ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-primary/60" />
            <p className="text-sm text-gray-400">Searching for "{searchQuery}"...</p>
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <AlertCircle className="h-6 w-6 text-red-400" />
            <p className="text-sm text-red-400">{error}</p>
          </div>
        ) : !hasSearched ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <Music2 className="h-8 w-8 text-gray-500/50" />
            <p className="text-center text-sm">
              Start typing to search for songs<br />
              <span className="text-xs text-gray-500">Results will appear as you type</span>
            </p>
          </div>
        ) : songs.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <Music2 className="h-6 w-6 text-gray-500/50" />
            <p className="text-center text-sm">
              No songs found for "{searchQuery}"<br />
              <span className="text-xs text-gray-500">Try a different search term</span>
            </p>
          </div>
        ) : (
          <div className="absolute inset-0 overflow-y-auto">
            <div className="space-y-2">
              <p className="text-xs text-gray-400 sticky top-0 backdrop-blur-sm py-2 z-10">
                Showing top {songs.length} of {totalResults} song{totalResults !== 1 ? 's' : ''} for "{searchQuery}"
              </p>
              <div className="space-y-2">
                {songs.map((song) => (
                  <SongCard
                    key={song.id}
                    song={song}
                    onAction={handleAddToHistory}
                    actionLabel={addingToHistory === song.id ? "Adding..." : "Add to History"}
                    showAction={addingToHistory !== song.id}
                    isCompact
                  />
                ))}
              </div>
              {totalResults > songs.length && (
                <p className="text-xs text-gray-500 text-center py-2">
                  {totalResults - songs.length} more result{totalResults - songs.length !== 1 ? 's' : ''} available
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 