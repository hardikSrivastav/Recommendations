import { DemographicsForm } from "@/components/DemographicsForm"
import { Recommendations } from "@/components/Recommendations"
import { useState } from "react"

export default function HomePage() {
  const [showRecommendations, setShowRecommendations] = useState(false);

  const handleDemographicsSubmit = (data: any) => {
    setShowRecommendations(true);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">Music Recommendation System</h1>
      
      {!showRecommendations ? (
        <DemographicsForm onSubmit={handleDemographicsSubmit} />
      ) : (
        <Recommendations />
      )}
    </main>
  );
}