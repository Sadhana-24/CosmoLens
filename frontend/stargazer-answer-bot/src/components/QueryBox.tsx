
import { useState, useRef } from 'react';
import { Search, Telescope, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';

interface QueryBoxProps {
  onSubmit: (query: string, imageFile?: File) => void;
  isLoading: boolean;
}

export function QueryBox({ onSubmit, isLoading }: QueryBoxProps) {
  const [query, setQuery] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && !selectedImage) {
      toast({
        title: "Input Required",
        description: "Please enter a question or upload an image.",
        variant: "destructive",
      });
      return;
    }
    onSubmit(query, selectedImage || undefined);
    setQuery('');
    setSelectedImage(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = '';
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type and size
      const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
      const maxSize = 5 * 1024 * 1024; // 5MB

      if (!allowedTypes.includes(file.type)) {
        toast({
          title: "Invalid File Type",
          description: "Please upload a JPEG, PNG, or GIF image.",
          variant: "destructive",
        });
        return;
      }

      if (file.size > maxSize) {
        toast({
          title: "File Too Large",
          description: "Image must be less than 5MB.",
          variant: "destructive",
        });
        return;
      }

      setSelectedImage(file);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="relative flex items-center gap-2">
        <div className="relative flex-1">
          <Input
            type="text"
            placeholder="Ask about astronomy or the Hubble telescope..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-transparent border-space-purple/30 text-white placeholder:text-white/50 focus-visible:ring-0 focus-visible:border-space-purple/70"
          />
          <Search className="absolute right-3 top-1/2 -translate-y-1/2 text-white/50" size={20} />
        </div>
        <div className="flex items-center gap-2">
          <input
            type="file"
            ref={imageInputRef}
            accept="image/jpeg,image/png,image/gif"
            onChange={handleImageUpload}
            className="hidden"
          />
          <Button 
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => imageInputRef.current?.click()}
            className="text-white/70 hover:text-white hover:bg-transparent"
            title="Upload Image"
          >
            <Upload size={20} className={selectedImage ? "text-space-purple" : ""} />
          </Button>
          <Button 
            type="submit" 
            disabled={isLoading}
            className="bg-space-purple hover:bg-space-purple/90 text-white"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin">
                  <Telescope className="h-4 w-4" />
                </div>
                <span>Searching...</span>
              </div>
            ) : (
              <span>Search</span>
            )}
          </Button>
        </div>
      </div>
      {selectedImage && (
        <div className="mt-2 text-sm text-white/70 flex items-center gap-2">
          <span>Selected: {selectedImage.name}</span>
          <Button 
            type="button" 
            variant="ghost" 
            size="sm" 
            onClick={() => {
              setSelectedImage(null);
              if (imageInputRef.current) {
                imageInputRef.current.value = '';
              }
            }}
            className="text-red-500 hover:text-red-600 h-6 px-2"
          >
            Remove
          </Button>
        </div>
      )}
    </form>
  );
}
