import { Button } from "@/components/ui/button";
import { Music, Clock, Tag, Disc } from "lucide-react";
import { cn } from "@/lib/utils";

export interface Song {
  id: number;
  track_title: string;
  artist_name: string;
  album_title: string;
  track_genres?: string[];
  track_duration?: number;
  track_date_created?: string;
  tags?: string[];
}

interface SongCardProps {
  song: Song;
  onAction?: (song: Song) => void;
  actionLabel?: string;
  className?: string;
  showAction?: boolean;
  isCompact?: boolean;
}

export function SongCard({ 
  song, 
  onAction, 
  actionLabel = "Add to History",
  className,
  showAction = true,
  isCompact = false
}: SongCardProps) {
  // Format duration if available
  const formatDuration = (seconds?: number) => {
    if (!seconds) return null;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-xl bg-gradient-to-br from-background/90 to-background/60 backdrop-blur-sm",
        "border border-white/10 hover:border-white/20 transition-all duration-300",
        "shadow-lg hover:shadow-xl",
        className
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-secondary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      
      <div className={cn(
        "flex items-start gap-4 p-4",
        isCompact ? "p-3" : "p-4"
      )}>
        <div className={cn(
          "flex-shrink-0 rounded-lg bg-primary/10 flex items-center justify-center",
          isCompact ? "w-10 h-10" : "w-12 h-12"
        )}>
          <Music className={cn(
            "text-primary/60",
            isCompact ? "h-5 w-5" : "h-6 w-6"
          )} />
        </div>

        <div className="flex-grow min-w-0">
          <h3 className={cn(
            "font-semibold text-white truncate",
            isCompact ? "text-base" : "text-lg"
          )}>
            {song.track_title}
          </h3>
          
          <div className="flex items-center gap-2 mt-1">
            <span className="text-gray-400 text-sm truncate">
              {song.artist_name}
            </span>
            <span className="text-gray-600">•</span>
            <span className="text-gray-400 text-sm truncate">
              {song.album_title}
            </span>
          </div>

          {!isCompact && (
            <div className="mt-3 flex flex-wrap gap-2">
              {song.track_genres?.map((genre, index) => (
                <span
                  key={index}
                  className="px-2 py-1 rounded-full bg-primary/10 text-primary-foreground/60 text-xs"
                >
                  {genre}
                </span>
              ))}
              {song.track_duration && (
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-secondary/10 text-secondary-foreground/60 text-xs">
                  <Clock className="h-3 w-3" />
                  {formatDuration(song.track_duration)}
                </span>
              )}
              {song.tags?.map((tag, index) => (
                <span
                  key={index}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-accent/10 text-accent-foreground/60 text-xs"
                >
                  <Tag className="h-3 w-3" />
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>

        {showAction && onAction && (
          <Button
            variant="secondary"
            size={isCompact ? "sm" : "default"}
            onClick={() => onAction(song)}
            className={cn(
              "flex-shrink-0 transition-transform duration-300",
              "opacity-0 group-hover:opacity-100 scale-95 group-hover:scale-100",
              isCompact ? "ml-2" : "ml-4"
            )}
          >
            {actionLabel}
          </Button>
        )}
      </div>
    </div>
  );
} 