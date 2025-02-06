import { SongCard, type Song } from "./SongCard";
import { Music2, Loader2 } from "lucide-react";

interface ListeningHistoryProps {
  songs: Song[];
  onRemove: (songId: number) => void;
  onPlay: (song: Song) => void;
  error?: string | null;
  loading?: boolean;
}

export function ListeningHistory({ 
  songs, 
  onRemove, 
  onPlay, 
  error, 
  loading 
}: ListeningHistoryProps) {
  const renderContent = () => {
    if (loading) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <Loader2 className="h-6 w-6 animate-spin text-primary/60" />
          <p className="text-sm text-gray-400">Loading your listening history...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <Music2 className="h-8 w-8 text-gray-500/50" />
          <p className="text-center text-sm">
            Add your demographics, then start adding songs to your history<br />
            <span className="text-xs text-gray-500">Your listening history will appear here</span>
          </p>
        </div>
      );
    }

    if (!songs.length) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <Music2 className="h-8 w-8 text-gray-500/50" />
          <p className="text-center text-sm">
            Start adding songs to your history<br />
            <span className="text-xs text-gray-500">Your listening history will appear here</span>
          </p>
        </div>
      );
    }

    return (
      <div className="flex-1 min-h-0 overflow-auto">
        <div className="space-y-2">
          {songs.map((song) => (
            <SongCard
              key={song.id}
              song={song}
              onAction={() => onRemove(song.id)}
              actionLabel="Remove"
              isCompact
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <h2 className="text-2xl font-semibold text-white mb-4 flex-none">Listening History</h2>
      {renderContent()}
    </div>
  );
} 