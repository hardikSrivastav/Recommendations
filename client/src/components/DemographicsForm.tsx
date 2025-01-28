import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"
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
    <form onSubmit={handleSubmit} className="space-y-6 w-full max-w-md">
      <div className="space-y-4">
        <div>
          <label htmlFor="age" className="block text-sm font-medium">
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
          />
        </div>

        <div>
          <label htmlFor="gender" className="block text-sm font-medium">
            Gender
          </label>
          <Select
            value={formData.gender}
            onValueChange={(value) => setFormData({...formData, gender: value})}
          >
            {genderOptions.map((option) => (
              <Select.Option key={option} value={option}>
                {option}
              </Select.Option>
            ))}
          </Select>
        </div>

        <div>
          <label htmlFor="location" className="block text-sm font-medium">
            Location
          </label>
          <Select
            value={formData.location}
            onValueChange={(value) => setFormData({...formData, location: value})}
          >
            {locations.map((location) => (
              <Select.Option key={location} value={location}>
                {location}
              </Select.Option>
            ))}
          </Select>
        </div>

        <div>
          <label htmlFor="occupation" className="block text-sm font-medium">
            Occupation
          </label>
          <Select
            value={formData.occupation}
            onValueChange={(value) => setFormData({...formData, occupation: value})}
          >
            {occupations.map((occupation) => (
              <Select.Option key={occupation} value={occupation}>
                {occupation}
              </Select.Option>
            ))}
          </Select>
        </div>
      </div>

      <Button type="submit" className="w-full">
        Submit Demographics
      </Button>
    </form>
  );
}