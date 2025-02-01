import { Button } from "@/components/ui/button"
import { useState } from "react"

interface Song {
  id: number;
  track_title: string;
  artist_name: string;
  album_title: string;
}

interface MusicPlayerProps {
  song: Song;
}

export function MusicPlayer({ song }: MusicPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlay = () => {
    setIsPlaying(true);
  };

  const handleComplete = () => {
    setIsPlaying(false);
  };

  return (
    <div className="flex flex-col items-center space-y-4 p-6 border rounded-lg">
      <div className="text-center">
        <h3 className="text-lg font-semibold">{song.track_title}</h3>
        <p className="text-sm text-gray-500">{song.artist_name}</p>
        <p className="text-xs text-gray-400">{song.album_title}</p>
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