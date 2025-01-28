import { MusicPlayer } from "@/components/MusicPlayer"
import { Recommendations } from "@/components/Recommendations"

export default function PlayerPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">Your Music</h1>
      <div className="w-full max-w-6xl">
        <Recommendations />
      </div>
    </main>
  );
}