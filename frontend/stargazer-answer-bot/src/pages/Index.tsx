
import { useState } from 'react';
import { QueryBox } from '@/components/QueryBox';
import { Messages } from '@/components/Messages';
import { Message } from '@/types/message';
import { SpaceBackground } from '@/components/SpaceBackground';
import { Telescope, Star } from 'lucide-react';

const Index = () => {
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);

  const handleQuery = async (query: string, imageFile?: File) => {
    const newUserMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      image: imageFile ? URL.createObjectURL(imageFile) : undefined,
    };

    setMessages(prev => [...prev, newUserMessage]);
    setLoading(true);

    try {
      // Prepare form data for multipart upload
      const formData = new FormData();
      if (query) {
        formData.append('text', query);
      }
      if (imageFile) {
        formData.append('image', imageFile);
      }

      const res = await fetch('http://localhost:8000/query_multimodal/', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) throw new Error('Failed to fetch response');
      
      const data = await res.json();
      
      // Process content URLs to create context items
      const contexts = data.content_urls ? data.content_urls.map((url: string) => `Source: ${url}`) : [];
      
      const newAssistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.text_response,
        contexts: contexts,
        image: data.image_url || undefined,
      };

      setMessages(prev => [...prev, newAssistantMessage]);
    } catch (error) {
      console.error('Query error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "Sorry, there was an error processing your query. Please try again.",
        contexts: [],
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-transparent text-white overflow-y-auto relative">
      <SpaceBackground />
      
      <div className="container px-4 py-12 mx-auto relative">
        <div className="text-center mb-12 relative">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-space-purple/20 rounded-full blur-3xl" />
          <div className="flex justify-center items-center gap-4 mb-6">
            <Telescope className="h-16 w-16 text-space-purple animate-floating" />
            <Star className="h-8 w-8 text-space-light animate-pulse-slow" />
          </div>
          <h1 className="text-5xl font-bold mb-6 relative">
            <span className="bg-gradient-to-r from-space-purple via-space-blue to-space-light bg-clip-text text-transparent">
            CosmoLens: A Multimodal Retrieval-Augmented Generation System for Astronomical Data Exploration
            </span>
          </h1>
          <p className="text-lg text-white/70 max-w-2xl mx-auto backdrop-blur-sm p-4 rounded-lg bg-black/20 border border-space-purple/20">
            Explore the mysteries of the universe, ask questions and discover amazing insights about our cosmos.
          </p>
        </div>

        <div className="space-y-8 relative">
          <Messages messages={messages} />
          <div className="fixed bottom-0 left-0 right-0 z-50 p-4 bg-transparent">
            <div className="max-w-3xl mx-auto">
              <QueryBox onSubmit={handleQuery} isLoading={loading} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
