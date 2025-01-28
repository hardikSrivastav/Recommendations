import { DemographicsForm } from "@/components/DemographicsForm"

export default function ProfilePage() {
  const handleDemographicsUpdate = (data: any) => {
    // Update user demographics (to be implemented)
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">Your Profile</h1>
      <div className="w-full max-w-md">
        <DemographicsForm onSubmit={handleDemographicsUpdate} />
      </div>
    </main>
  );
}