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
          <Loader2 className="h-6 w-6 animate-spin text-primary/40" />
          <p className="text-sm text-muted-foreground">Loading your listening history...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <Music2 className="h-8 w-8 text-muted-foreground/30" />
          <p className="text-center text-sm text-muted-foreground">
            Add your demographics, then start adding songs to your history<br />
            <span className="text-xs text-muted-foreground/60">Your listening history will appear here</span>
          </p>
        </div>
      );
    }

    if (!songs.length) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <Music2 className="h-8 w-8 text-muted-foreground/30" />
          <p className="text-center text-sm text-muted-foreground">
            Start adding songs to your history<br />
            <span className="text-xs text-muted-foreground/60">Your listening history will appear here</span>
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
      <div className="flex justify-between items-center mb-4 flex-none">
        <h2 className="text-2xl font-semibold text-foreground">Listening History</h2>
        <span className="text-sm text-muted-foreground">{songs.length}/50 songs</span>
      </div>
      {renderContent()}
    </div>
  );
} 