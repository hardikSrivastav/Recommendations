import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Loader2 } from "lucide-react";

interface Song {
  track_title: string;
  artist_name: string;
  album_title: string;
  id: number;
}

interface SongSearchProps {
  onAddToHistory: (song: Song) => void;
}

export function SongSearch({ onAddToHistory }: SongSearchProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [songs, setSongs] = useState<Song[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch(`/api/songs/search?q=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setSongs(data);
    } catch (error) {
      console.error("Failed to search songs:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      <div className="flex gap-3">
        <div className="relative flex-1">
          <Input
            type="text"
            placeholder="Search for songs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="pl-12 py-6 text-lg bg-background/80"
          />
          <Search className="absolute left-4 top-4 h-5 w-5 text-gray-400" />
        </div>
        <Button 
          onClick={handleSearch} 
          disabled={isLoading}
          className="px-8 py-6 text-lg"
          size="lg"
        >
          {isLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            "Search"
          )}
        </Button>
      </div>

      <div className="space-y-3">
        {songs.map((song) => (
          <div
            key={song.id}
            className="flex items-center justify-between p-5 rounded-lg bg-background/80 hover:bg-background/60 transition-colors"
          >
            <div>
              <h3 className="text-lg font-medium text-white">{song.track_title}</h3>
              <p className="text-base text-gray-300">
                {song.artist_name} • {song.album_title}
              </p>
            </div>
            <Button
              variant="secondary"
              size="lg"
              onClick={() => onAddToHistory(song)}
              className="ml-4 text-base"
            >
              Add to History
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
} 