
import { useEffect, useState } from 'react';

export function SpaceBackground() {
  const [stars, setStars] = useState<Array<{ x: number; y: number; size: number; opacity: number }>>([]);

  useEffect(() => {
    const generateStars = () => {
      const newStars = Array.from({ length: 100 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 3,
        opacity: Math.random() * 0.8 + 0.2,
      }));
      setStars(newStars);
    };

    generateStars();
  }, []);

  return (
    <div className="fixed inset-0 -z-10">
      {stars.map((star, i) => (
        <div
          key={i}
          className="absolute rounded-full animate-pulse-slow"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            width: `${star.size}px`,
            height: `${star.size}px`,
            opacity: star.opacity,
            backgroundColor: 'white',
          }}
        />
      ))}
      <div className="absolute inset-0 bg-gradient-to-b from-[#1a0b2e] via-[#1a1b3b] to-[#162447] opacity-95" />
    </div>
  );
}
