import React, { useRef } from 'react';
import { Button } from './button';

interface FileTriggerButtonProps {
  onSelect?: (files: FileList | null) => void;
  acceptedFileTypes?: string[];
  allowsMultiple?: boolean;
  children?: React.ReactNode;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  disabled?: boolean;
}

export function FileTriggerButton({
  onSelect,
  acceptedFileTypes,
  allowsMultiple = false,
  children = 'Choose File',
  variant = 'default',
  size = 'default',
  className,
  disabled = false,
}: FileTriggerButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (onSelect) onSelect(e.target.files);
    // Reset so same file can be re-selected
    e.target.value = '';
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={acceptedFileTypes?.join(',')}
        multiple={allowsMultiple}
        onChange={handleChange}
        className="hidden"
      />
      <Button
        variant={variant}
        size={size}
        className={className}
        disabled={disabled}
        onClick={handleClick}
        type="button"
      >
        {children}
      </Button>
    </>
  );
}
