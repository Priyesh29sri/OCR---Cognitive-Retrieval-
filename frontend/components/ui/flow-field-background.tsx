'use client';

import React, { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
}

interface FlowFieldBackgroundProps {
  className?: string;
  particleCount?: number;
  particleColor?: string;
  lineColor?: string;
  backgroundColor?: string;
  speed?: number;
  connectionDistance?: number;
}

export function FlowFieldBackground({
  className,
  particleCount = 100,
  particleColor = 'rgba(99, 102, 241, 0.6)',
  lineColor = 'rgba(99, 102, 241, 0.2)',
  backgroundColor = 'transparent',
  speed = 0.5,
  connectionDistance = 150,
}: FlowFieldBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const mouseRef = useRef({ x: 0, y: 0 });
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize particles
    const initParticles = () => {
      particlesRef.current = [];
      for (let i = 0; i < particleCount; i++) {
        particlesRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * speed,
          vy: (Math.random() - 0.5) * speed,
          life: Math.random() * 100,
          maxLife: 100,
        });
      }
    };
    initParticles();

    // Mouse move handler
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Perlin-like noise function
    const noise = (x: number, y: number, time: number) => {
      return Math.sin(x * 0.01 + time) * Math.cos(y * 0.01 + time);
    };

    // Animation loop
    let time = 0;
    const animate = () => {
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      time += 0.005;

      // Update and draw particles
      particlesRef.current.forEach((particle, i) => {
        // Flow field influence
        const angle = noise(particle.x, particle.y, time) * Math.PI * 2;
        const force = 0.1;
        particle.vx += Math.cos(angle) * force;
        particle.vy += Math.sin(angle) * force;

        // Mouse influence
        const dx = mouseRef.current.x - particle.x;
        const dy = mouseRef.current.y - particle.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 200) {
          const force = (200 - dist) / 200;
          particle.vx += (dx / dist) * force * 0.2;
          particle.vy += (dy / dist) * force * 0.2;
        }

        // Damping
        particle.vx *= 0.95;
        particle.vy *= 0.95;

        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;

        // Update life
        particle.life += 0.5;
        if (particle.life > particle.maxLife) {
          particle.life = 0;
        }

        // Draw particle
        const opacity = 1 - Math.abs(particle.life - particle.maxLife / 2) / (particle.maxLife / 2);
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = particleColor.replace(/[\d.]+\)$/g, `${opacity * 0.6})`);
        ctx.fill();

        // Draw connections
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const other = particlesRef.current[j];
          const dx = particle.x - other.x;
          const dy = particle.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < connectionDistance) {
            const opacity = (1 - distance / connectionDistance) * 0.5;
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(other.x, other.y);
            ctx.strokeStyle = lineColor.replace(/[\d.]+\)$/g, `${opacity})`);
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [particleCount, particleColor, lineColor, backgroundColor, speed, connectionDistance]);

  return (
    <canvas
      ref={canvasRef}
      className={cn('fixed inset-0 -z-10', className)}
      style={{ pointerEvents: 'none' }}
    />
  );
}
