
import { Message } from "@/types/message";
import { ResponseCard } from "./ResponseCard";
import { Card } from "./ui/card";

interface MessagesProps {
  messages: Message[];
}

export function Messages({ messages }: MessagesProps) {
  return (
    <div className="w-full max-w-3xl mx-auto space-y-6 relative">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`animate-fade-in ${
            message.type === 'user' ? 'flex justify-end' : ''
          }`}
        >
          {message.type === 'user' ? (
            <Card className="p-4 bg-space-purple/20 backdrop-blur-md border-space-purple/50 max-w-[80%] transform hover:scale-[1.02] transition-all duration-200">
              <p className="text-white/90 mb-2">{message.content}</p>
              {message.image && (
                <img 
                  src={message.image} 
                  alt="User uploaded" 
                  className="max-w-full h-auto rounded-md mt-2 object-cover" 
                />
              )}
            </Card>
          ) : (
            <ResponseCard 
              answer={message.content} 
              contexts={message.contexts || []}
              image={message.image}
            />
          )}
        </div>
      ))}
    </div>
  );
}
