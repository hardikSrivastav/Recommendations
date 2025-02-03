import { SongCard, type Song } from "./SongCard";
import { Music2 } from "lucide-react";

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
        <div className="flex-1 flex items-center justify-center">
          <p className="text-gray-400 text-center">
            Loading your listening history...
          </p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-4">
          <Music2 className="h-12 w-12 text-gray-500/50" />
          <p className="text-gray-400 text-center">
            Add your demographics, then start adding songs to your history
          </p>
        </div>
      );
    }

    if (!songs.length) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-4">
          <Music2 className="h-12 w-12 text-gray-500/50" />
          <p className="text-gray-400 text-center">
            Your listening history will appear here<br />
            once you start adding songs
          </p>
        </div>
      );
    }

    return (
      <div className="flex-1 min-h-0 overflow-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
        <div className="space-y-3 pr-2">
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