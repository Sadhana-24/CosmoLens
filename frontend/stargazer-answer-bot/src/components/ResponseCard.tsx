import { Card } from "./ui/card";
import { Star } from "lucide-react";
import { useEffect, useState } from "react";

interface ResponseCardProps {
  answer: string;
  contexts: string[];
  image?: string;
}

export function ResponseCard({ answer, contexts, image }: ResponseCardProps) {
  const [contextImages, setContextImages] = useState<string[]>([]);

  useEffect(() => {
    // Extract image URLs from contexts (they start with "Source: ")
    const imageUrls = contexts
      .filter(context => context.startsWith("Source: "))
      .map(context => context.replace("Source: ", ""))
      .slice(0, 5); // Take only first 5 images
    
    setContextImages(imageUrls);
  }, [contexts]);

  return (
    <Card className="p-4 bg-space-blue/20 backdrop-blur-md border-space-blue/50 max-w-[80%] transform hover:scale-[1.02] transition-all duration-200">
      {image && (
        <div className="mb-4">
          <img 
            src={image} 
            alt="Response related" 
            className="w-full h-auto rounded-md object-cover" 
          />
        </div>
      )}
      <div className="flex items-start gap-2 mb-4">
        <Star className="text-space-light w-5 h-5 mt-1 animate-pulse-slow" />
        <p className="text-white/90 flex-1">{answer}</p>
      </div>
      
      {contextImages.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-4">
          {contextImages.map((url, index) => (
            <img
              key={index}
              src={url}
              alt={`Context image ${index + 1}`}
              className="w-full h-48 object-cover rounded-md hover:scale-105 transition-transform"
              onError={(e) => {
                // Remove failed images from the array
                setContextImages(prev => prev.filter(item => item !== url));
              }}
            />
          ))}
        </div>
      )}
      
      {/* Keep URLs as hidden text for reference */}
      <div className="hidden">
        {contexts.map((context, index) => (
          <p key={index} className="text-xs text-white/70 mb-1 italic">
            {context}
          </p>
        ))}
      </div>
    </Card>
  );
}
