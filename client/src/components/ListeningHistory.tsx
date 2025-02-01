import { SongCard, type Song } from "./SongCard";

interface ListeningHistoryProps {
  songs: Song[];
  onRemove: (songId: number) => void;
}

export function ListeningHistory({ songs, onRemove }: ListeningHistoryProps) {
  if (!songs.length) {
    return (
      <div className="text-center py-8">
        <h2 className="text-2xl font-semibold text-white mb-2">Listening History</h2>
        <p className="text-gray-400">
          Your listening history will appear here once you start adding songs.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold text-white">Listening History</h2>
      <div className="space-y-3">
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
} 