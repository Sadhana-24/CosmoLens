
export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  contexts?: string[];
  image?: string; // Base64 or URL of the image
}
