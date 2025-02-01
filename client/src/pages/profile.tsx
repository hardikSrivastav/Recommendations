import { DemographicsForm } from "@/components/DemographicsForm"
import { useState, useEffect } from "react"
import { userAPI } from "@/services/api"

export default function ProfilePage() {
  const [demographics, setDemographics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadDemographics = async () => {
      try {
        const response = await userAPI.getDemographics();
        setDemographics(response.data);
      } catch (error) {
        setError('Failed to load demographics');
        console.error('Failed to load demographics:', error);
      } finally {
        setLoading(false);
      }
    };
    loadDemographics();
  }, []);

  const handleDemographicsUpdate = async (data: any) => {
    try {
      await userAPI.updateDemographics(data);
      setDemographics(data);
    } catch (error) {
      setError('Failed to update demographics');
      console.error('Failed to update demographics:', error);
    }
  };

  if (loading) {
    return (
      <main className="flex min-h-screen flex-col items-center p-24">
        <p>Loading...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="flex min-h-screen flex-col items-center p-24">
        <p className="text-red-500">{error}</p>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">Your Profile</h1>
      <div className="w-full max-w-md">
        <DemographicsForm onSubmit={handleDemographicsUpdate} initialData={demographics} />
      </div>
    </main>
  );
}