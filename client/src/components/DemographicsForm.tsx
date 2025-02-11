import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import * as Select from "@/components/ui/select"
import { useState, useEffect } from "react"

interface DemographicsFormProps {
  onSubmit: (data: UserDemographics) => void;
  initialData?: UserDemographics | null;
  onForgetMe?: () => void;
  onEditToggle: () => void;
}

interface UserDemographics {
  age: number;
  gender: string;
  location: string;
  occupation: string;
}

export function DemographicsForm({ onSubmit, initialData, onForgetMe, onEditToggle }: DemographicsFormProps) {
  const [formData, setFormData] = useState<UserDemographics>({
    age: 0,
    gender: '',
    location: '',
    occupation: ''
  });
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    if (initialData) {
      setFormData(initialData);
    }
  }, [initialData]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
    setIsEditing(false);
    onEditToggle();
  };

  const handleForgetMe = () => {
    if (window.confirm('Are you sure you want to delete all your data? This action cannot be undone.')) {
      onForgetMe?.();
    }
  };

  const handleEditClick = () => {
    setIsEditing(!isEditing);
    onEditToggle();
  };

  if (!isEditing && initialData) {
    return (
      <div className="space-y-4">
        <h3 className="text-xl font-semibold text-foreground mb-4">Current Demographics</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-base text-muted-foreground font-medium">Age</label>
            <p className="text-lg text-foreground font-semibold">{initialData.age}</p>
          </div>
          <div className="space-y-1.5">
            <label className="text-base text-muted-foreground font-medium">Gender</label>
            <p className="text-lg text-foreground font-semibold">{initialData.gender}</p>
          </div>
          <div className="space-y-1.5">
            <label className="text-base text-muted-foreground font-medium">Location</label>
            <p className="text-lg text-foreground font-semibold">{initialData.location}</p>
          </div>
          <div className="space-y-1.5">
            <label className="text-base text-muted-foreground font-medium">Occupation</label>
            <p className="text-lg text-foreground font-semibold">{initialData.occupation}</p>
          </div>
        </div>
        <div className="flex gap-2 pt-4">
          <Button
            variant="secondary"
            onClick={handleEditClick}
            size="lg"
            className="text-lg"
          >
            Edit Demographics
          </Button>
          <Button
            variant="destructive"
            onClick={handleForgetMe}
            size="lg"
            className="text-lg"
          >
            Forget Me
          </Button>
        </div>
      </div>
    );
  }

  const genderOptions = ['M', 'F', 'NB', 'O'];
  const locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR', 'IN'];
  const occupations = [
    'Student', 'Professional', 'Artist', 'Engineer', 
    'Teacher', 'Healthcare', 'Business', 'Service', 'Retired'
  ];

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <h3 className="text-xl font-semibold text-foreground mb-4">
        {initialData ? 'Edit Demographics' : 'Enter Your Demographics'}
      </h3>
      <div className="space-y-5">
        <div className="space-y-2.5">
          <label htmlFor="age" className="block text-base font-medium text-muted-foreground">
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
            className="py-6 text-lg bg-background border-border focus:border-primary"
          />
        </div>

        <div className="space-y-2.5">
          <label htmlFor="gender" className="block text-base font-medium text-muted-foreground">
            Gender
          </label>
          <Select.Select value={formData.gender} onValueChange={(value) => setFormData({...formData, gender: value})}>
            <Select.Trigger className="w-full py-6 text-lg bg-background border-border">
              <span className="text-lg">{formData.gender || 'Select gender'}</span>
            </Select.Trigger>
            <Select.Content className="bg-background border border-border">
              {genderOptions.map((option) => (
                <Select.Option 
                  key={option} 
                  value={option}
                  className="text-lg py-3 hover:bg-accent"
                >
                  {option}
                </Select.Option>
              ))}
            </Select.Content>
          </Select.Select>
        </div>

        <div className="space-y-2.5">
          <label htmlFor="location" className="block text-base font-medium text-muted-foreground">
            Location
          </label>
          <Select.Select value={formData.location} onValueChange={(value) => setFormData({...formData, location: value})}>
            <Select.Trigger className="w-full py-6 text-lg bg-background border-border">
              <span className="text-lg">{formData.location || 'Select location'}</span>
            </Select.Trigger>
            <Select.Content className="bg-background border border-border">
              {locations.map((location) => (
                <Select.Option 
                  key={location} 
                  value={location}
                  className="text-lg py-3 hover:bg-accent"
                >
                  {location}
                </Select.Option>
              ))}
            </Select.Content>
          </Select.Select>
        </div>

        <div className="space-y-2.5">
          <label htmlFor="occupation" className="block text-base font-medium text-muted-foreground">
            Occupation
          </label>
          <Select.Select value={formData.occupation} onValueChange={(value) => setFormData({...formData, occupation: value})}>
            <Select.Trigger className="w-full py-6 text-lg bg-background border-border">
              <span className="text-lg">{formData.occupation || 'Select occupation'}</span>
            </Select.Trigger>
            <Select.Content className="bg-background border border-border">
              {occupations.map((occupation) => (
                <Select.Option 
                  key={occupation} 
                  value={occupation}
                  className="text-lg py-3 hover:bg-accent"
                >
                  {occupation}
                </Select.Option>
              ))}
            </Select.Content>
          </Select.Select>
        </div>
      </div>

      <div className="flex gap-4 mt-8">
        <Button 
          type="submit" 
          className="flex-1 py-6 text-lg font-medium bg-primary hover:bg-primary/90 text-primary-foreground transition-colors"
          size="lg"
        >
          {initialData ? 'Update Demographics' : 'Submit Demographics'}
        </Button>
        {initialData && (
          <Button 
            type="button"
            onClick={() => {
              setIsEditing(false);
              onEditToggle();
            }}
            className="flex-1 py-6 text-lg font-medium bg-secondary hover:bg-secondary/90 text-secondary-foreground transition-colors"
            size="lg"
            variant="secondary"
          >
            Cancel
          </Button>
        )}
      </div>
    </form>
  );
}