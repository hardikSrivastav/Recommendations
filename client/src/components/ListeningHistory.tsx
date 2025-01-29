import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";

interface Song {
  track_title: string;
  artist_name: string;
  album_title: string;
  id: number;
}

interface ListeningHistoryProps {
  songs: Song[];
  onRemove: (songId: number) => void;
}

export function ListeningHistory({ songs, onRemove }: ListeningHistoryProps) {
  if (songs.length === 0) {
    return (
      <div className="w-full p-8 text-center rounded-lg bg-background/80">
        <p className="text-lg text-gray-300">
          No songs in your listening history yet.
          <br />
          Search and add songs to start tracking your music journey!
        </p>
      </div>
    );
  }

  return (
    <div className="w-full space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-white">Your Listening History</h2>
        <span className="text-base text-gray-300">{songs.length} songs</span>
      </div>
      
      <div className="space-y-3">
        {songs.map((song) => (
          <div
            key={song.id}
            className="flex items-center justify-between p-5 rounded-lg bg-background/80 hover:bg-background/60 transition-colors group"
          >
            <div>
              <h3 className="text-lg font-medium text-white">{song.track_title}</h3>
              <p className="text-base text-gray-300">
                {song.artist_name} • {song.album_title}
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={() => onRemove(song.id)}
            >
              <Trash2 className="h-5 w-5" />
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
} 