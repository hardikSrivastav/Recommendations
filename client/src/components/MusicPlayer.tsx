import { Button } from "@/components/ui/button"
import { useState } from "react"

interface Song {
  id: string;
  title: string;
  artist: string;
  url: string;
}

interface MusicPlayerProps {
  song: Song;
  onPlayComplete: (songId: string) => void;
}

export function MusicPlayer({ song, onPlayComplete }: MusicPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlay = () => {
    setIsPlaying(true);
    // POST to /history (to be implemented)
    // Log listening event
  };

  const handleComplete = () => {
    setIsPlaying(false);
    onPlayComplete(song.id);
  };

  return (
    <div className="flex flex-col items-center space-y-4 p-6 border rounded-lg">
      <div className="text-center">
        <h3 className="text-lg font-semibold">{song.title}</h3>
        <p className="text-sm text-gray-500">{song.artist}</p>
      </div>

      <div className="flex space-x-4">
        <Button
          variant={isPlaying ? "destructive" : "default"}
          onClick={() => isPlaying ? handleComplete() : handlePlay()}
        >
          {isPlaying ? "Stop" : "Play"}
        </Button>
      </div>
    </div>
  );
}