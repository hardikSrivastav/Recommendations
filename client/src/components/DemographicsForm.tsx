import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import * as Select from "@/components/ui/select"
import { useState } from "react"

interface DemographicsFormProps {
  onSubmit: (data: UserDemographics) => void;
}

interface UserDemographics {
  age: number;
  gender: string;
  location: string;
  occupation: string;
}

export function DemographicsForm({ onSubmit }: DemographicsFormProps) {
  const [formData, setFormData] = useState<UserDemographics>({
    age: 0,
    gender: '',
    location: '',
    occupation: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
    // POST to /demographics (to be implemented)
  };

  const genderOptions = ['M', 'F', 'NB', 'O'];
  const locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR', 'IN'];
  const occupations = [
    'Student', 'Professional', 'Artist', 'Engineer', 
    'Teacher', 'Healthcare', 'Business', 'Service', 'Retired'
  ];

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-5">
        <div className="space-y-2.5">
          <label htmlFor="age" className="block text-lg font-medium text-white">
            Age
          </label>
          <Input
            id="age"
            type="number"
            value={formData.age}
            onChange={(e) => setFormData({...formData, age: parseInt(e.target.value)})}
            min={13}
            max={100}
            required
            className="py-6 text-lg bg-background/80 border-accent/20"
          />
        </div>

        <div className="space-y-2.5">
          <label htmlFor="gender" className="block text-lg font-medium text-white">
            Gender
          </label>
          <Select.Select value={formData.gender} onValueChange={(value) => setFormData({...formData, gender: value})}>
            <Select.Trigger className="w-full py-6 text-lg bg-background/80 border-accent/20">
              <span className="text-lg">{formData.gender || 'Select gender'}</span>
            </Select.Trigger>
            <Select.Content className="bg-background/90 border border-accent/20">
              {genderOptions.map((option) => (
                <Select.Option 
                  key={option} 
                  value={option}
                  className="text-lg py-3 hover:bg-accent/20"
                >
                  {option}
                </Select.Option>
              ))}
            </Select.Content>
          </Select.Select>
        </div>

        <div className="space-y-2.5">
          <label htmlFor="location" className="block text-lg font-medium text-white">
            Location
          </label>
          <Select.Select value={formData.location} onValueChange={(value) => setFormData({...formData, location: value})}>
            <Select.Trigger className="w-full py-6 text-lg bg-background/80 border-accent/20">
              <span className="text-lg">{formData.location || 'Select location'}</span>
            </Select.Trigger>
            <Select.Content className="bg-background/90 border border-accent/20">
              {locations.map((location) => (
                <Select.Option 
                  key={location} 
                  value={location}
                  className="text-lg py-3 hover:bg-accent/20"
                >
                  {location}
                </Select.Option>
              ))}
            </Select.Content>
          </Select.Select>
        </div>

        <div className="space-y-2.5">
          <label htmlFor="occupation" className="block text-lg font-medium text-white">
            Occupation
          </label>
          <Select.Select value={formData.occupation} onValueChange={(value) => setFormData({...formData, occupation: value})}>
            <Select.Trigger className="w-full py-6 text-lg bg-background/80 border-accent/20">
              <span className="text-lg">{formData.occupation || 'Select occupation'}</span>
            </Select.Trigger>
            <Select.Content className="bg-background/90 border border-accent/20">
              {occupations.map((occupation) => (
                <Select.Option 
                  key={occupation} 
                  value={occupation}
                  className="text-lg py-3 hover:bg-accent/20"
                >
                  {occupation}
                </Select.Option>
              ))}
            </Select.Content>
          </Select.Select>
        </div>
      </div>

      <Button 
        type="submit" 
        className="w-full py-6 text-lg font-medium mt-8 bg-primary/90 hover:bg-primary/100 text-primary-foreground transition-colors"
        size="lg"
      >
        Submit Demographics
      </Button>
    </form>
  );
}